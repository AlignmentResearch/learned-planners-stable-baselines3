"""
Helpers for dealing with vectorized environments.
"""
from collections import OrderedDict
from typing import Any, Dict, Hashable, List, Mapping, Optional, Tuple, TypeVar, Union, overload

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import check_for_nested_spaces
from stable_baselines3.common.vec_env.base_vec_env import EnvObs, VecEnvObs

TensorObsType = TypeVar("TensorObsType", bound=Union[th.Tensor, Dict[str, th.Tensor]])


def as_torch_dtype(dtype: Union[th.dtype, np.typing.DTypeLike]) -> th.dtype:
    """
    Convert a numpy dtype to a PyTorch dtype, if it is not already one.

    :param dtype: Numpy or Pytorch dtype
    :return: PyTorch dtype
    """
    if isinstance(dtype, th.dtype):
        return dtype
    return getattr(th, np.dtype(dtype).name)


def as_numpy_dtype(dtype: Union[th.dtype, np.dtype]) -> np.dtype:
    """
    Convert a PyTorch dtype to a numpy dtype, if it is not already one.

    :param dtype: Pytorch or Numpy dtype
    :return: Numpy dtype
    """
    if isinstance(dtype, np.dtype):
        return dtype
    return np.dtype(str(dtype).removeprefix("torch."))


KT = TypeVar("KT", bound=Hashable)


@overload
def obs_as_tensor(obs: Union[np.ndarray, th.Tensor], device: Optional[th.device] = None) -> th.Tensor:
    ...


@overload
def obs_as_tensor(
    obs: Union[Tuple[np.ndarray, ...], Tuple[th.Tensor, ...]], device: Optional[th.device] = None
) -> Tuple[th.Tensor, ...]:
    ...


@overload
def obs_as_tensor(
    obs: Union[Mapping[KT, np.ndarray], Mapping[KT, th.Tensor]], device: Optional[th.device] = None
) -> Dict[KT, th.Tensor]:
    ...


def obs_as_tensor(obs, device=None):
    if isinstance(obs, dict):
        return {k: th.as_tensor(v, device=device) for k, v in obs.items()}
    elif isinstance(obs, tuple):
        return tuple(th.as_tensor(v, device=device) for v in obs)
    else:
        return th.as_tensor(obs, device=device)


def obs_as_np(obs: Union[EnvObs, VecEnvObs], space: Optional[spaces.Space] = None) -> EnvObs:
    def _as_np(x: Any, space: Optional[spaces.Space]) -> Any:
        if isinstance(x, th.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(space, spaces.Discrete) and x.shape == ():
            return np.int64(x)
        if space is not None:
            x = x.astype(space.dtype)
        return x

    if isinstance(obs, dict):
        if space is None:
            return {k: _as_np(v, None) for k, v in obs.items()}
        else:
            assert isinstance(space, spaces.Dict), f"Expected Dict, got {type(space)}"
            return {k: _as_np(v, space[k]) for k, v in obs.items()}

    elif isinstance(obs, tuple):
        if space is None:
            return tuple(_as_np(o, None) for o in obs)
        else:
            assert isinstance(space, spaces.Tuple), f"Expected Tuple, got {type(space)}"
            return tuple(_as_np(obs, space[i]) for i, obs in enumerate(obs))

    else:
        return _as_np(obs, space)


def clone_obs(obs: VecEnvObs) -> VecEnvObs:
    """
    Deep-copy a VecEnvObs
    """
    if isinstance(obs, dict):
        return OrderedDict([(k, v.clone()) for k, v in obs.items()])
    elif isinstance(obs, tuple):
        return tuple(v.clone() for v in obs)
    else:
        return obs.clone()


def copy_obs_dict(obs: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
    """
    Deep-copy a dict of torch Tensors.

    :param obs: a dict of torch Tensors.
    :return: a dict of copied torch Tensors.
    """
    assert isinstance(obs, OrderedDict), f"unexpected type for observations '{type(obs)}'"
    return OrderedDict([(k, v.clone()) for k, v in obs.items()])


def dict_to_obs(obs_space: spaces.Space, obs_dict: Dict[Any, th.Tensor]) -> VecEnvObs:
    """
    Convert an internal representation raw_obs into the appropriate type
    specified by space.

    :param obs_space: an observation space.
    :param obs_dict: a dict of torch Tensors.
    :return: returns an observation of the same type as space.
        If space is Dict, function is identity; if space is Tuple, converts dict to Tuple;
        otherwise, space is unstructured and returns the value raw_obs[None].
    """
    if isinstance(obs_space, spaces.Dict):
        return obs_dict
    elif isinstance(obs_space, spaces.Tuple):
        assert len(obs_dict) == len(obs_space.spaces), "size of observation does not match size of observation space"
        return tuple(obs_dict[i] for i in range(len(obs_space.spaces)))
    else:
        assert set(obs_dict.keys()) == {None}, "multiple observation keys for unstructured observation space"
        return obs_dict[None]


def obs_space_info(obs_space: spaces.Space) -> Tuple[List[str], Dict[Any, Tuple[int, ...]], Dict[Any, th.dtype]]:
    """
    Get dict-structured information about a gym.Space.

    Dict spaces are represented directly by their dict of subspaces.
    Tuple spaces are converted into a dict with keys indexing into the tuple.
    Unstructured spaces are represented by {None: obs_space}.

    :param obs_space: an observation space
    :return: A tuple (keys, shapes, dtypes):
        keys: a list of dict keys.
        shapes: a dict mapping keys to shapes.
        dtypes: a dict mapping keys to dtypes.
    """
    check_for_nested_spaces(obs_space)
    if isinstance(obs_space, spaces.Dict):
        assert isinstance(obs_space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        subspaces = obs_space.spaces
    elif isinstance(obs_space, spaces.Tuple):
        subspaces = {i: space for i, space in enumerate(obs_space.spaces)}  # type: ignore[assignment]
    else:
        assert not hasattr(obs_space, "spaces"), f"Unsupported structured space '{type(obs_space)}'"
        subspaces = {None: obs_space}  # type: ignore[assignment]
    keys = []
    shapes = {}
    dtypes = {}
    for key, box in subspaces.items():
        keys.append(key)
        shapes[key] = box.shape
        dtypes[key] = as_torch_dtype(box.dtype)
    return keys, shapes, dtypes
