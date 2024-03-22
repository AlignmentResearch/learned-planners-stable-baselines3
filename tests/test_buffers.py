import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import (
    DictReplayBuffer,
    DictRolloutBuffer,
    ReplayBuffer,
    RolloutBuffer,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.pytree_dataclass import tree_flatten
from stable_baselines3.common.recurrent.buffers import (
    RecurrentRolloutBuffer,
    SamplingType,
    TimeContiguousBatchesDataset,
)
from stable_baselines3.common.recurrent.type_aliases import RecurrentRolloutBufferData
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    ReplayBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

EP_LENGTH: int = 100


class DummyEnv(gym.Env):
    """
    Custom gym environment for testing purposes
    """

    def __init__(self):
        self.action_space = spaces.Box(1, 5, (1,))
        self.observation_space = spaces.Box(1, 5, (1,))
        self._observations = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
        self._rewards = [1, 2, 3, 4, 5]
        self._t = 0
        self._ep_length = EP_LENGTH

    def reset(self, *, seed=None, options=None):
        self._t = 0
        obs = self._observations[0]
        return obs, {}

    def step(self, action):
        self._t += 1
        index = self._t % len(self._observations)
        obs = self._observations[index]
        terminated = False
        truncated = self._t >= self._ep_length
        reward = self._rewards[index]
        return obs, reward, terminated, truncated, {}


class DummyDictEnv(gym.Env):
    """
    Custom gym environment for testing purposes
    """

    def __init__(self):
        # Test for multi-dim action space
        self.action_space = spaces.Box(1, 5, shape=(10, 7))
        space = spaces.Box(1, 5, (1,))
        self.observation_space = spaces.Dict({"observation": space, "achieved_goal": space, "desired_goal": space})
        self._observations = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
        self._rewards = [1, 2, 3, 4, 5]
        self._t = 0
        self._ep_length = EP_LENGTH

    def reset(self, seed=None, options=None):
        self._t = 0
        obs = {key: self._observations[0] for key in self.observation_space.spaces.keys()}
        return obs, {}

    def step(self, action):
        self._t += 1
        index = self._t % len(self._observations)
        obs = {key: self._observations[index] for key in self.observation_space.spaces.keys()}
        terminated = False
        truncated = self._t >= self._ep_length
        reward = self._rewards[index]
        return obs, reward, terminated, truncated, {}


@pytest.mark.parametrize("env_cls", [DummyEnv, DummyDictEnv])
def test_env(env_cls):
    # Check the env used for testing
    # Do not warn for assymetric space
    check_env(env_cls(), warn=False, skip_render_check=True)


@pytest.mark.parametrize("replay_buffer_cls", [ReplayBuffer, DictReplayBuffer])
def test_replay_buffer_normalization(replay_buffer_cls):
    env = {ReplayBuffer: DummyEnv, DictReplayBuffer: DummyDictEnv}[replay_buffer_cls]
    env = make_vec_env(env)
    env = VecNormalize(env)

    buffer = replay_buffer_cls(EP_LENGTH, env.observation_space, env.action_space, device="cpu")

    # Interract and store transitions
    env.reset()
    obs = env.get_original_obs()
    for _ in range(EP_LENGTH):
        action = th.as_tensor(env.action_space.sample())
        _, _, done, info = env.step(action)
        next_obs = env.get_original_obs()
        reward = env.get_original_reward()
        buffer.add(obs, next_obs, action, reward, done, info)
        obs = next_obs

    sample = buffer.sample(50, env)
    # Test observation normalization
    for observations in [sample.observations, sample.next_observations]:
        if isinstance(sample, DictReplayBufferSamples):
            for key in observations.keys():
                assert th.allclose(observations[key].mean(0), th.zeros(1), atol=1)
        elif isinstance(sample, ReplayBufferSamples):
            assert th.allclose(observations.mean(0), th.zeros(1), atol=1)
    # Test reward normalization
    assert np.allclose(sample.rewards.mean(0), np.zeros(1), atol=1)


HIDDEN_STATES_EXAMPLE = {"a": {"b": th.zeros(2, 4)}}


@pytest.mark.parametrize(
    "replay_buffer_cls, sampling_type",
    [
        (DictReplayBuffer, None),
        (DictRolloutBuffer, None),
        (ReplayBuffer, None),
        (RolloutBuffer, None),
        (RecurrentRolloutBuffer, SamplingType.CLASSIC),
        (RecurrentRolloutBuffer, SamplingType.SKEW_ZEROS),
        (RecurrentRolloutBuffer, SamplingType.SKEW_RANDOM),
    ],
)
@pytest.mark.parametrize("n_envs", [1, 4])
@pytest.mark.parametrize("device", ["cpu", "cuda", "auto"])
def test_device_buffer(replay_buffer_cls, sampling_type, n_envs, device):
    if device == "cuda" and not th.cuda.is_available():
        pytest.skip("CUDA not available")

    env = {
        RolloutBuffer: DummyEnv,
        DictRolloutBuffer: DummyDictEnv,
        ReplayBuffer: DummyEnv,
        DictReplayBuffer: DummyDictEnv,
        RecurrentRolloutBuffer: DummyDictEnv,
    }[replay_buffer_cls]
    env = make_vec_env(env, n_envs=n_envs)
    hidden_states_shape = HIDDEN_STATES_EXAMPLE["a"]["b"].shape
    N_ENVS_HIDDEN_STATES = {"a": {"b": th.zeros((hidden_states_shape[0], env.num_envs, *hidden_states_shape[1:]))}}

    if replay_buffer_cls == RecurrentRolloutBuffer:
        buffer = RecurrentRolloutBuffer(
            EP_LENGTH,
            env.observation_space,
            env.action_space,
            hidden_state_example=N_ENVS_HIDDEN_STATES,
            device=device,
            n_envs=n_envs,
            sampling_type=sampling_type,
        )
    else:
        buffer = replay_buffer_cls(EP_LENGTH, env.observation_space, env.action_space, device=device, n_envs=n_envs)

    # Interract and store transitions
    obs = env.reset()
    episode_start, values, log_prob = th.zeros(n_envs), th.zeros(n_envs), th.ones(n_envs)

    for _ in range(EP_LENGTH):
        action = th.as_tensor([env.action_space.sample() for _ in range(n_envs)])

        next_obs, reward, done, info = env.step(action)
        if replay_buffer_cls in [RolloutBuffer, DictRolloutBuffer]:
            buffer.add(obs, action, reward, episode_start, values, log_prob)
        elif replay_buffer_cls == RecurrentRolloutBuffer:
            buffer.add(RecurrentRolloutBufferData(obs, action, reward, episode_start, values, log_prob, N_ENVS_HIDDEN_STATES))
        else:
            buffer.add(obs, next_obs, action, reward, done, info)
        obs = next_obs

    # Get data from the buffer
    batch_envs = max(1, env.num_envs // 2)
    batch_time = EP_LENGTH // 2
    if replay_buffer_cls in [RolloutBuffer, DictRolloutBuffer]:
        data = buffer.get(batch_time)
    elif replay_buffer_cls in [ReplayBuffer, DictReplayBuffer]:
        data = [buffer.sample(batch_time)]
    elif replay_buffer_cls == RecurrentRolloutBuffer:
        data = buffer.get(batch_envs=batch_envs, batch_time=batch_time)

    # Check that all data are on the desired device
    desired_device = get_device(device).type
    for minibatch in list(data):
        flattened_tensors, _ = tree_flatten(minibatch)
        assert len(flattened_tensors) > 3
        for value in flattened_tensors:
            assert isinstance(value, th.Tensor)
            assert value.device.type == desired_device

        # Check that data are of the desired shape
        if isinstance(minibatch, (ReplayBufferSamples, DictReplayBufferSamples)):
            assert minibatch.rewards.shape == (batch_time, 1)
        else:
            if replay_buffer_cls == RecurrentRolloutBuffer:
                assert minibatch.old_log_prob.shape == (batch_time, batch_envs)
            else:
                assert minibatch.old_log_prob.shape == (batch_time,)


@pytest.mark.parametrize("num_envs", [1, 3, 4, 5])
@pytest.mark.parametrize("num_time, batch_time", [(1, 1), (3, 1), (6, 2), (10, 5)])
@pytest.mark.parametrize("skew_zero", [True, False])
def test_time_contiguous_batches_dataset(num_envs, num_time, batch_time, skew_zero):
    num_time_batches = num_time // batch_time
    if skew_zero:
        skew = th.zeros(num_envs, dtype=th.long)
    else:
        skew = th.arange(1, num_envs + 1, dtype=th.long) % num_time

    d = TimeContiguousBatchesDataset(
        num_envs=num_envs, num_time=num_time, batch_time=batch_time, skew=skew, device=th.device("cpu")
    )

    assert len(d) == num_time_batches * num_envs

    set_all: set[int] = set()
    for i in range(len(d)):
        which_time_batch = (i // num_envs) * batch_time
        which_env = i % num_envs
        this_skew = skew[which_env].item()
        this_batch: list[int] = [
            which_env + num_envs * ((which_time_batch + j + this_skew) % num_time) for j in range(batch_time)
        ]

        set_all.update(this_batch)

        assert th.equal(d[i], th.as_tensor(this_batch))

    assert set_all == set(range(num_envs * num_time))

    for batch in th.split(th.arange(len(d)), min(len(d), 4)):
        if len(batch) > 0:
            individual = th.vstack([d[int(b.item())] for b in batch]).T
            (first_time, first_envs), collective = d.get_batch_and_init_times(batch)
            assert th.equal(individual, collective)

            assert th.equal(first_time * num_envs + first_envs, collective[0])
