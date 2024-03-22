"""Common aliases for type hints"""

from abc import abstractproperty
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Protocol,
    SupportsFloat,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_origin,
)

import gymnasium as gym
import numpy as np
import torch as th

from stable_baselines3.common import callbacks, vec_env

SB3_TREE_NAMESPACE: str = "stable-baselines3"


GymEnv = Union[gym.Env, vec_env.VecEnv]
GymObs = Union[Tuple["GymObs", ...], Dict[str, "GymObs"], np.ndarray, int]
TensorDict = Dict[str, th.Tensor]
TorchGymObsBasic = Union[Tuple[th.Tensor, ...], Dict[str, th.Tensor], th.Tensor]
TorchGymObs = Union[Tuple["TorchGymObs", ...], Dict[str, "TorchGymObs"], th.Tensor, TensorDict, int]
GymResetReturn = Tuple[GymObs, Dict]
AtariResetReturn = Tuple[np.ndarray, Dict[str, Any]]
GymStepReturn = Tuple[GymObs, float, bool, bool, Dict]
AtariStepReturn = Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]
TensorIndex = Union[int, slice, th.Tensor]
OptimizerStateDict = Dict[str, Any]
MaybeCallback = Union[None, Callable, List[callbacks.BaseCallback], callbacks.BaseCallback]

# A schedule takes the remaining progress as input
# and outputs a scalar (e.g. learning rate, clip range, ...)
Schedule = Callable[[float], float]


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class DictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class DictReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    next_observations: TensorDict
    dones: th.Tensor
    rewards: th.Tensor


class RolloutReturn(NamedTuple):
    episode_timesteps: int
    n_episodes: int
    continue_training: bool


class TrainFrequencyUnit(Enum):
    STEP = "step"
    EPISODE = "episode"


class TrainFreq(NamedTuple):
    frequency: int
    unit: TrainFrequencyUnit  # either "step" or "episode"


class PolicyPredictor(Protocol):
    def predict(
        self,
        observation: Union[th.Tensor, Dict[str, th.Tensor]],
        state: Optional[Tuple[th.Tensor, ...]] = None,
        episode_start: Optional[th.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, Optional[Tuple[th.Tensor, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """

    @abstractproperty
    def device(self) -> th.device:
        """
        :return: the device on which this predictor lives
        """
        ...


T = TypeVar("T")


def non_null(v: Optional[T]) -> T:
    """
    Checks that `v` is not None, and returns it.
    """
    if v is None:
        raise ValueError("Expected a value, got None")
    return v


def check_cast(cls: Type[T], v: Any) -> T:
    """
    Checks that `v` is of type `cls`, and returns it.

    NOTE: this function does not check the template arguments, only the type itself.
    """
    if not isinstance(v, get_origin(cls) or cls):
        raise TypeError(f"{v} should be of type {cls}")
    return v
