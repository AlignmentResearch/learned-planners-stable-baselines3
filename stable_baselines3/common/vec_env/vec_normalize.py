import inspect
import pickle
from typing import Any, Dict, List, Optional, Union

import numpy as np
from gymnasium import spaces
import torch as th

from stable_baselines3.common import utils
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.util import clone_obs


class VecNormalize(VecEnvWrapper):
    """
    A moving average, normalizing wrapper for vectorized environment.
    has support for saving/loading moving average,

    :param venv: the vectorized environment to wrap
    :param training: Whether to update or not the moving average
    :param norm_obs: Whether to normalize observation or not (default: True)
    :param norm_reward: Whether to normalize rewards or not (default: True)
    :param clip_obs: Max absolute value for observation
    :param clip_reward: Max value absolute for discounted reward
    :param gamma: discount factor
    :param epsilon: To avoid division by zero
    :param norm_obs_keys: Which keys from observation dict to normalize.
        If not specified, all keys will be normalized.
    """

    def __init__(
        self,
        venv: VecEnv,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        norm_obs_keys: Optional[List[str]] = None,
    ):
        VecEnvWrapper.__init__(self, venv)

        self.norm_obs = norm_obs
        self.norm_obs_keys = norm_obs_keys
        # Check observation spaces
        if self.norm_obs:
            self._sanity_checks()

            if isinstance(self.observation_space, spaces.Dict):
                self.obs_spaces = self.observation_space.spaces
                self.obs_rms = {key: RunningMeanStd(shape=self.obs_spaces[key].shape) for key in self.norm_obs_keys}
                # Update observation space when using image
                # See explanation below and GH #1214
                for key in self.obs_rms.keys():
                    if is_image_space(self.obs_spaces[key]):
                        self.observation_space.spaces[key] = spaces.Box(
                            low=-clip_obs,
                            high=clip_obs,
                            shape=self.obs_spaces[key].shape,
                            dtype=np.float32,
                        )

            else:
                self.obs_spaces = None
                self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
                # Update observation space when using image
                # See GH #1214
                # This is to raise proper error when
                # VecNormalize is used with an image-like input and
                # normalize_images=True.
                # For correctness, we should also update the bounds
                # in other cases but this will cause backward-incompatible change
                # and break already saved policies.
                if is_image_space(self.observation_space):
                    self.observation_space = spaces.Box(
                        low=-clip_obs,
                        high=clip_obs,
                        shape=self.observation_space.shape,
                        dtype=np.float32,
                    )

        self.ret_rms = RunningMeanStd(shape=())
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        # Returns: discounted rewards
        self.returns = th.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.old_obs = th.tensor([], dtype=th.float32)
        self.old_reward = th.tensor([], dtype=th.float32)

    def _sanity_checks(self) -> None:
        """
        Check the observations that are going to be normalized are of the correct type (spaces.Box).
        """
        if isinstance(self.observation_space, spaces.Dict):
            # By default, we normalize all keys
            if self.norm_obs_keys is None:
                self.norm_obs_keys = list(self.observation_space.spaces.keys())
            # Check that all keys are of type Box
            for obs_key in self.norm_obs_keys:
                if not isinstance(self.observation_space.spaces[obs_key], spaces.Box):
                    raise ValueError(
                        f"VecNormalize only supports `gym.spaces.Box` observation spaces but {obs_key} "
                        f"is of type {self.observation_space.spaces[obs_key]}. "
                        "You should probably explicitely pass the observation keys "
                        " that should be normalized via the `norm_obs_keys` parameter."
                    )

        elif isinstance(self.observation_space, spaces.Box):
            if self.norm_obs_keys is not None:
                raise ValueError("`norm_obs_keys` param is applicable only with `gym.spaces.Dict` observation spaces")

        else:
            raise ValueError(
                "VecNormalize only supports `gym.spaces.Box` and `gym.spaces.Dict` observation spaces, "
                f"not {self.observation_space}"
            )

    def __getstate__(self) -> Dict[str, Any]:
        """
        Gets state for pickling.

        Excludes self.venv, as in general VecEnv's may not be pickleable."""
        state = self.__dict__.copy()
        # these attributes are not pickleable
        del state["venv"]
        del state["class_attributes"]
        # these attributes depend on the above and so we would prefer not to pickle
        del state["returns"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restores pickled state.

        User must call set_venv() after unpickling before using.

        :param state:"""
        # Backward compatibility
        if "norm_obs_keys" not in state and isinstance(state["observation_space"], spaces.Dict):
            state["norm_obs_keys"] = list(state["observation_space"].spaces.keys())
        self.__dict__.update(state)
        assert "venv" not in state
        self.venv = None

    def set_venv(self, venv: VecEnv) -> None:
        """
        Sets the vector environment to wrap to venv.

        Also sets attributes derived from this such as `num_env`.

        :param venv:
        """
        if self.venv is not None:
            raise ValueError("Trying to set venv of already initialized VecNormalize wrapper.")
        self.venv = venv
        self.num_envs = venv.num_envs
        self.class_attributes = dict(inspect.getmembers(self.__class__))

        # Check that the observation_space shape match
        utils.check_shape_equal(self.observation_space, venv.observation_space)
        self.returns = th.zeros(self.num_envs)

    def step_wait(self) -> VecEnvStepReturn:
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, dones)

        where ``dones`` is a boolean vector indicating whether each element is new.
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        self.old_obs = obs
        self.old_reward = rewards

        if self.training and self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.obs_rms.keys():
                    self.obs_rms[key].update(obs[key])
            else:
                self.obs_rms.update(obs)

        obs = self.normalize_obs(obs)

        if self.training:
            self._update_reward(rewards)
        rewards = self.normalize_reward(rewards)

        # Normalize the terminal observations
        for idx, done in enumerate(dones):
            if not done:
                continue
            if "terminal_observation" in infos[idx]:
                infos[idx]["terminal_observation"] = self.normalize_obs(infos[idx]["terminal_observation"])

        self.returns[dones] = 0
        return obs, rewards, dones, infos

    def _update_reward(self, reward: th.Tensor) -> None:
        """Update reward normalization statistics."""
        self.returns = self.returns * self.gamma + reward
        self.ret_rms.update(self.returns)

    def _normalize_obs(self, obs: th.Tensor, obs_rms: RunningMeanStd) -> th.Tensor:
        """
        Helper to normalize observation.
        :param obs:
        :param obs_rms: associated statistics
        :return: normalized observation
        """
        return th.clip((obs - obs_rms.mean) / th.sqrt(obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def _unnormalize_obs(self, obs: th.Tensor, obs_rms: RunningMeanStd) -> th.Tensor:
        """
        Helper to unnormalize observation.
        :param obs:
        :param obs_rms: associated statistics
        :return: unnormalized observation
        """
        assert obs.is_floating_point(), ".to(obs) in the next line assumes obs is floating"
        return (obs * th.sqrt(obs_rms.var + self.epsilon).to(obs)) + obs_rms.mean.to(obs)

    def normalize_obs(self, obs: Union[th.Tensor, Dict[str, th.Tensor]]) -> Union[th.Tensor, Dict[str, th.Tensor]]:
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        # Avoid modifying by reference the original object
        obs_ = clone_obs(obs)
        if self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                # Only normalize the specified keys
                for key in self.norm_obs_keys:
                    dtype = obs[key].dtype if obs[key].is_floating_point() else th.get_default_dtype()
                    obs_[key] = self._normalize_obs(obs[key], self.obs_rms[key]).to(dtype)
            else:
                obs_ = self._normalize_obs(obs, self.obs_rms).to(th.float32)
        return obs_

    def normalize_reward(self, reward: th.Tensor) -> th.Tensor:
        """
        Normalize rewards using this VecNormalize's rewards statistics.
        Calling this method does not update statistics.
        """
        if self.norm_reward:
            dtype = reward.dtype if reward.is_floating_point() else th.get_default_dtype()
            reward = th.clip(reward / th.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward).to(dtype)
        return reward

    def unnormalize_obs(self, obs: Union[th.Tensor, Dict[str, th.Tensor]]) -> Union[th.Tensor, Dict[str, th.Tensor]]:
        # Avoid modifying by reference the original object
        obs_ = clone_obs(obs)
        if self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.norm_obs_keys:
                    obs_[key] = self._unnormalize_obs(obs[key], self.obs_rms[key])
            else:
                obs_ = self._unnormalize_obs(obs, self.obs_rms)
        return obs_

    def unnormalize_reward(self, reward: th.Tensor) -> th.Tensor:
        if self.norm_reward:
            assert reward.is_floating_point(), ".to(reward) in the next line assumes reward is floating"
            return reward * th.sqrt(self.ret_rms.var + self.epsilon).to(reward)
        return reward

    def get_original_obs(self) -> Union[th.Tensor, Dict[str, th.Tensor]]:
        """
        Returns an unnormalized version of the observations from the most recent
        step or reset.
        """
        return clone_obs(self.old_obs)

    def get_original_reward(self) -> th.Tensor:
        """
        Returns an unnormalized version of the rewards from the most recent step.
        """
        return self.old_reward.clone()

    def reset(self) -> Union[th.Tensor, Dict[str, th.Tensor]]:
        """
        Reset all environments
        :return: first observation of the episode
        """
        obs = self.venv.reset()
        self.old_obs = obs
        self.returns = th.zeros(self.num_envs)
        if self.training and self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.obs_rms.keys():
                    self.obs_rms[key].update(obs[key])
            else:
                self.obs_rms.update(obs)
        return self.normalize_obs(obs)

    @staticmethod
    def load(load_path: str, venv: VecEnv) -> "VecNormalize":
        """
        Loads a saved VecNormalize object.

        :param load_path: the path to load from.
        :param venv: the VecEnv to wrap.
        :return:
        """
        with open(load_path, "rb") as file_handler:
            vec_normalize = pickle.load(file_handler)
        vec_normalize.set_venv(venv)
        return vec_normalize

    def save(self, save_path: str) -> None:
        """
        Save current VecNormalize object with
        all running statistics and settings (e.g. clip_obs)

        :param save_path: The path to save to
        """
        with open(save_path, "wb") as file_handler:
            pickle.dump(self, file_handler)
