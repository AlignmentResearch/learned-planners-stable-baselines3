import dataclasses
from functools import partial
from typing import Any, Callable, Generator, Optional, Tuple, Union

import optree as ot
import torch as th
from gymnasium import spaces
from optree import PyTree

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.pytree_dataclass import OT_NAMESPACE as NS
from stable_baselines3.common.pytree_dataclass import dataclass_frozen_pytree
from stable_baselines3.common.recurrent.type_aliases import (
    HiddenState,
    PyTreeGeneric,
    RecurrentRolloutBufferData,
    RecurrentRolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize


@dataclass_frozen_pytree
class LSTMStates:
    pi: Tuple[th.Tensor, th.Tensor]
    vf: Tuple[th.Tensor, th.Tensor]


def pad(
    seq_start_indices: th.Tensor,
    seq_end_indices: th.Tensor,
    device: th.device,
    tensor: th.Tensor,
    padding_value: float = 0.0,
) -> th.Tensor:
    """
    Chunk sequences and pad them to have constant dimensions.

    :param seq_start_indices: Indices of the transitions that start a sequence
    :param seq_end_indices: Indices of the transitions that end a sequence
    :param device: PyTorch device
    :param tensor: Tensor of shape (batch_size, *tensor_shape)
    :param padding_value: Value used to pad sequence to the same length
        (zero padding by default)
    :return: (n_seq, max_length, *tensor_shape)
    """
    # Create sequences given start and end
    seq = [tensor[start : end + 1].to(device) for start, end in zip(seq_start_indices, seq_end_indices)]
    return th.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=padding_value)


def pad_and_flatten(
    seq_start_indices: th.Tensor,
    seq_end_indices: th.Tensor,
    device: th.device,
    tensor: th.Tensor,
    padding_value: float = 0.0,
) -> th.Tensor:
    """
    Pad and flatten the sequences of scalar values,
    while keeping the sequence order.
    From (batch_size, 1) to (n_seq, max_length, 1) -> (n_seq * max_length,)

    :param seq_start_indices: Indices of the transitions that start a sequence
    :param seq_end_indices: Indices of the transitions that end a sequence
    :param device: PyTorch device (cpu, gpu, ...)
    :param tensor: Tensor of shape (max_length, n_seq, 1)
    :param padding_value: Value used to pad sequence to the same length
        (zero padding by default)
    :return: (n_seq * max_length,) aka (padded_batch_size,)
    """
    return pad(seq_start_indices, seq_end_indices, device, tensor, padding_value).flatten()


def create_sequencers(
    episode_starts: th.Tensor,
    env_change: th.Tensor,
    device: th.device,
) -> Tuple[th.Tensor, Callable, Callable]:
    """
    Create the utility function to chunk data into
    sequences and pad them to create fixed size tensors.

    :param episode_starts: Indices where an episode starts
    :param env_change: Indices where the data collected
        come from a different env (when using multiple env for data collection)
    :param device: PyTorch device
    :return: Indices of the transitions that start a sequence,
        pad and pad_and_flatten utilities tailored for this batch
        (sequence starts and ends indices are fixed)
    """
    # Create sequence if env changes too
    seq_start = (episode_starts | env_change).flatten()
    # First index is always the beginning of a sequence
    seq_start[0] = True
    # Retrieve indices of sequence starts
    seq_start_indices = th.argwhere(seq_start).squeeze(1)
    # End of sequence are just before sequence starts
    # Last index is also always end of a sequence
    seq_end_indices = th.cat([(seq_start_indices - 1)[1:], th.tensor([len(episode_starts)]).to(seq_start_indices)])

    # Create padding method for this minibatch
    # to avoid repeating arguments (seq_start_indices, seq_end_indices)
    local_pad = partial(pad, seq_start_indices, seq_end_indices, device)
    local_pad_and_flatten = partial(pad_and_flatten, seq_start_indices, seq_end_indices, device)
    return seq_start_indices, local_pad, local_pad_and_flatten


def index_into_pytree(
    idx: Any,
    tree: PyTreeGeneric,
    is_leaf: Optional[Union[bool, Callable[[PyTreeGeneric], bool]]] = None,
    none_is_leaf: bool = False,
    namespace: str = NS,
) -> PyTreeGeneric:
    return ot.tree_map(lambda x: x[idx], tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)  # type: ignore


def space_to_example(
    batch_shape: Tuple[int, ...],
    space: spaces.Space,
    *,
    device: Optional[th.device] = None,
    ensure_non_batch_dim: bool = False,
) -> PyTree[th.Tensor]:
    if isinstance(space, spaces.Dict):
        return {
            k: space_to_example(batch_shape, v, device=device, ensure_non_batch_dim=ensure_non_batch_dim)
            for k, v in space.items()
        }
    if isinstance(space, spaces.Tuple):
        return tuple(space_to_example(batch_shape, v, device=device, ensure_non_batch_dim=ensure_non_batch_dim) for v in space)

    if isinstance(space, spaces.Box):
        space_shape = space.shape
        space_dtype = th.float32
    elif isinstance(space, spaces.Discrete):
        space_shape = ()
        space_dtype = th.long
    elif isinstance(space, spaces.MultiDiscrete):
        space_shape = (len(space.nvec),)
        space_dtype = th.long
    elif isinstance(space, spaces.MultiBinary):
        space_shape = (space.n,)
        space_dtype = th.float32
    else:
        raise TypeError(f"Unknown space type {type(space)} for {space}")

    if ensure_non_batch_dim and not space_shape:
        space_shape = (1,)
    return th.zeros((*batch_shape, *space_shape), dtype=space_dtype, device=device)


class RecurrentRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that also stores the LSTM cell and hidden states.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_state_shape: Shape of the buffer that will collect lstm states
        (n_steps, lstm.num_layers, n_envs, lstm.hidden_size)
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    advantages: th.Tensor
    returns: th.Tensor
    data: RecurrentRolloutBufferData

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_example: HiddenState,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device=device, n_envs=n_envs)
        self.hidden_state_example = hidden_state_example
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        batch_shape = (self.buffer_size, self.n_envs)
        device = self.device

        self.hidden_state_example = ot.tree_map(
            lambda x: th.zeros((), dtype=x.dtype, device=device).expand_as(x), hidden_state_example, namespace=NS
        )
        self.advantages = th.zeros(batch_shape, dtype=th.float32, device=device)
        self.returns = th.zeros(batch_shape, dtype=th.float32, device=device)
        self.data = RecurrentRolloutBufferData(
            observations=space_to_example(batch_shape, self.observation_space, device=device, ensure_non_batch_dim=True),
            actions=th.zeros(
                (*batch_shape, self.action_dim),
                dtype=th.long if isinstance(self.action_space, (spaces.Discrete, spaces.MultiDiscrete)) else th.float32,
                device=device,
            ),
            rewards=th.zeros(batch_shape, dtype=th.float32, device=device),
            episode_starts=th.zeros(batch_shape, dtype=th.bool, device=device),
            values=th.zeros(batch_shape, dtype=th.float32, device=device),
            log_probs=th.zeros(batch_shape, dtype=th.float32, device=device),
            hidden_states=ot.tree_map(
                lambda x: th.zeros(self._reshape_hidden_state_shape(batch_shape, x.shape), dtype=x.dtype, device=device),
                hidden_state_example,
                namespace=NS,
            ),
        )

    @staticmethod
    def _reshape_hidden_state_shape(batch_shape: Tuple[int, ...], state_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(state_shape) < 2:
            raise NotImplementedError("State shape must be 2+ dimensions currently")
        return (*batch_shape[:-1], state_shape[0], batch_shape[-1], *state_shape[1:])

    # Expose attributes of the RecurrentRolloutBufferData in the top-level to conform to the RolloutBuffer interface
    @property
    def episode_starts(self) -> th.Tensor:
        return self.data.episode_starts

    @property
    def values(self) -> th.Tensor:
        return self.data.values

    @property
    def rewards(self) -> th.Tensor:
        assert self.data.rewards is not None, "RecurrentRolloutBufferData should store rewards"
        return self.data.rewards

    def reset(self):
        self.returns.zero_()
        self.advantages.zero_()
        ot.tree_map(lambda x: x.zero_(), self.data, namespace=NS)
        super(RolloutBuffer, self).reset()

    def extend(self, *args) -> None:
        """
        Add a new batch of transitions to the buffer
        """

        # Do a for loop along the batch axis.
        # Treat lists as leaves to avoid flattening the infos.
        def _is_list(t):
            return isinstance(t, list)

        tensors, _ = ot.tree_flatten(args, is_leaf=_is_list, namespace=NS)
        len_tensors = len(tensors[0])
        assert all(len(t) == len_tensors for t in tensors), "All tensors must have the same batch size"
        for i in range(len_tensors):
            self.add(*index_into_pytree(i, args, is_leaf=_is_list, namespace=NS))

    def add(self, data: RecurrentRolloutBufferData, **kwargs) -> None:
        """
        :param hidden_states: LSTM cell and hidden state
        """
        if data.rewards is None:
            raise ValueError("Recorded samples must contain a reward")
        new_data = dataclasses.replace(data, actions=data.actions.reshape((self.n_envs, self.action_dim)))

        ot.tree_map(
            lambda buf, x: buf[self.pos].copy_(x if x.ndim + 1 == buf.ndim else x.unsqueeze(-1), non_blocking=True),
            self.data,
            new_data,
            namespace=NS,
        )
        # Increment pos
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RecurrentRolloutBufferSamples, None, None]:  # type: ignore[signature-mismatch] #FIXME
        assert self.full, "Rollout buffer must be full before sampling from it"

        env_change = th.zeros((self.buffer_size, self.n_envs), dtype=th.bool, device=self.device)
        # Flag first timestep as change of environment
        env_change[0, :] = True
        env_change = self.swap_and_flatten(env_change)

        # Return everything, don't create minibatches
        if batch_size is None or batch_size == self.buffer_size * self.n_envs:
            yield self._get_samples(slice(None), env_change)
            return

        for start_idx in range(0, self.buffer_size, batch_size):
            out = self._get_samples(slice(start_idx, start_idx + batch_size, None), env_change)
            assert len(out.observations) != 0
            yield out

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: Union[slice, th.Tensor],
        env_change: th.Tensor,
        env: Optional[VecNormalize] = None,
    ) -> RecurrentRolloutBufferSamples:
        self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device
        )

        # Number of sequences
        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.data.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        # We retrieve the lstm hidden states that will allow
        # to properly initialize the LSTM at the beginning of each sequence
        if isinstance(self.data.hidden_states, LSTMStates):
            lstm_states_pi = self.data.hidden_states.pi
            lstm_states_vf = self.data.hidden_states.vf
        else:
            lstm_states_pi = lstm_states_vf = self.data.hidden_states

        (hidden_states_pi, cell_states_pi) = lstm_states_pi
        (hidden_states_vf, cell_states_vf) = lstm_states_vf

        lstm_states_pi = (
            # 1. (n_envs * n_steps, n_layers, dim) -> (batch_size, n_layers, dim)
            # 2. (batch_size, n_layers, dim)  -> (n_seq, n_layers, dim)
            # 3. (n_seq, n_layers, dim) -> (n_layers, n_seq, dim)
            hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            cell_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_pi = (self.to_device(lstm_states_pi[0]).contiguous(), self.to_device(lstm_states_pi[1]).contiguous())
        lstm_states_vf = (self.to_device(lstm_states_vf[0]).contiguous(), self.to_device(lstm_states_vf[1]).contiguous())

        # samples = RecurrentRolloutBufferSamples(
        #     observations=self.data.observations,
        #     actions=self.data.actions,
        #     episode_starts=self.data.episode_starts,
        #     old_values=self.data.values,
        #     old_log_prob=self.data.log_probs,
        #     advantages=self.advantages,
        #     returns=self.returns,
        #     hidden_states=self.data.hidden_states,
        # )

        samples = RecurrentRolloutBufferSamples(
            # (batch_size, obs_dim) -> (n_seq, max_length, obs_dim) -> (n_seq * max_length, obs_dim)
            observations=self.pad(self.data.observations[batch_inds]).reshape((padded_batch_size, *self.obs_shape)),
            actions=self.pad(self.data.actions[batch_inds]).reshape((padded_batch_size,) + self.data.actions.shape[1:]),
            old_values=self.pad_and_flatten(self.data.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.data.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            hidden_states=LSTMStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad_and_flatten(self.data.episode_starts[batch_inds]),
        )
        return ot.tree_map(lambda tens: self.to_device(tens), samples, namespace=NS)
