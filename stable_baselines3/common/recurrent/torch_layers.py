import abc
from typing import Generic, Optional, Tuple, TypeVar

import gymnasium as gym
import torch as th

from stable_baselines3.common.pytree_dataclass import TensorTree, tree_flatten, tree_map
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import TorchGymObs

RecurrentState = TypeVar("RecurrentState", bound=TensorTree)

RecurrentSubState = TypeVar("RecurrentSubState", bound=TensorTree)


class RecurrentFeaturesExtractor(BaseFeaturesExtractor, abc.ABC, Generic[RecurrentState]):
    @abc.abstractmethod
    def recurrent_initial_state(
        self, n_envs: Optional[int] = None, *, device: Optional[th.device | str] = None
    ) -> RecurrentState:
        ...

    @abc.abstractmethod
    def forward(
        self, observations: TorchGymObs, state: RecurrentState, episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, RecurrentState]:
        ...

    @staticmethod
    def _process_sequence(
        rnn: th.nn.RNNBase, inputs: th.Tensor, init_state: RecurrentSubState, episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, RecurrentSubState]:
        (state_example, *_), _ = tree_flatten(init_state, is_leaf=None)
        n_layers, batch_sz, *_ = state_example.shape

        # Batch to sequence
        # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
        seq_len = inputs.shape[0] // batch_sz
        seq_inputs = inputs.view((batch_sz, seq_len, *inputs.shape[1:])).swapaxes(0, 1)
        episode_starts = episode_starts.view((batch_sz, seq_len)).swapaxes(0, 1)

        if th.any(episode_starts[1:]):
            raise NotImplementedError("Resetting state in the middle of a sequence is not supported")

        first_state_is_not_reset = (~episode_starts[0]).contiguous()
        # Shape here is (n_layers, batch_sz)
        init_state = tree_map(lambda x: x * first_state_is_not_reset.view((1, batch_sz, *(1,) * (x.ndim - 2))), init_state)
        rnn_output, end_state = rnn(seq_inputs, init_state)

        # (seq_len, batch_size, ...) -> (batch_size, seq_len, ...) -> (batch_size * seq_len, ...)
        rnn_output = rnn_output.transpose(0, 1).reshape((batch_sz * seq_len, *rnn_output.shape[2:]))
        return rnn_output, end_state


GRURecurrentState = th.Tensor


class GRUWrappedFeaturesExtractor(RecurrentFeaturesExtractor[GRURecurrentState]):
    def __init__(
        self,
        base_extractor: BaseFeaturesExtractor,
        hidden_size: Optional[int] = None,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(base_extractor._observation_space, base_extractor.features_dim)
        self.base_extractor = base_extractor

        if hidden_size is None:
            # Ensure hidden_size is at least 64 by default so it optimizes fine
            hidden_size = max(base_extractor.features_dim, 64)

        self.rnn = th.nn.GRU(
            input_size=base_extractor.features_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=False,
            dropout=dropout,
            bidirectional=False,
        )

    def recurrent_initial_state(
        self, n_envs: Optional[int] = None, *, device: Optional[th.device | str] = None
    ) -> GRURecurrentState:
        shape: Tuple[int, ...]
        if n_envs is None:
            shape = (self.rnn.num_layers, self.rnn.hidden_size)
        else:
            shape = (self.rnn.num_layers, n_envs, self.rnn.hidden_size)
        return th.zeros(shape, device=device)

    def forward(
        self, observations: TorchGymObs, state: GRURecurrentState, episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, GRURecurrentState]:
        features: th.Tensor = self.base_extractor(observations)
        return self._process_sequence(self.rnn, features, state, episode_starts)

    @property
    def features_dim(self) -> int:
        return self.rnn.hidden_size


class GRUFlattenExtractor(GRUWrappedFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        hidden_size: Optional[int] = None,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        base_extractor = FlattenExtractor(observation_space)
        super().__init__(base_extractor, hidden_size=hidden_size, num_layers=num_layers, bias=bias, dropout=dropout)


class GRUNatureCNNExtractor(GRUWrappedFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        cnn_output_dim: int = 512,
        normalized_image: bool = False,
        hidden_size: Optional[int] = None,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        base_extractor = NatureCNN(observation_space, features_dim=cnn_output_dim, normalized_image=normalized_image)
        super().__init__(base_extractor, hidden_size=hidden_size, num_layers=num_layers, bias=bias, dropout=dropout)


class GRUCombinedExtractor(GRUWrappedFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
        hidden_size: Optional[int] = None,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        base_extractor = CombinedExtractor(observation_space, cnn_output_dim=cnn_output_dim, normalized_image=normalized_image)
        super().__init__(base_extractor, hidden_size=hidden_size, num_layers=num_layers, bias=bias, dropout=dropout)
