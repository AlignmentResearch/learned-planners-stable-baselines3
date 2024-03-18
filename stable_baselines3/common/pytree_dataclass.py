import dataclasses
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import optree as ot
import torch as th
from optree import CustomTreeNode, PyTree
from typing_extensions import dataclass_transform

from stable_baselines3.common.type_aliases import SB3_TREE_NAMESPACE, TensorIndex
from stable_baselines3.common.utils import zip_strict

__all__ = [
    "FrozenPyTreeDataclass",
    "MutablePyTreeDataclass",
    "TensorTree",
    "tree_empty",
    "tree_flatten",
    "tree_index",
    "tree_map",
]

T = TypeVar("T")
U = TypeVar("U")

_RESERVED_NAMES = ["_PyTreeDataclassBase", "FrozenPyTreeDataclass", "MutablePyTreeDataclass"]


# We need to inherit from `type(CustomTreeNode)` to prevent conflicts due to different-inheritance in metaclasses.
# - For some reason just inheriting from `typing._ProtocolMeta` does not get rid of that error.
# - Inheriting from `typing._GenericAlias` is impossible, as it's a `typing._Final` class.
#
# But in mypy, inheriting from a dynamic base class from `type` is not supported, so we disable type checking for this
# line.
class _PyTreeDataclassMeta(type(CustomTreeNode)):  # type: ignore[misc]
    """Metaclass to register dataclasses as PyTrees.

    Usage:
      class MyDataclass(metaclass=_DataclassPyTreeMeta):
        ...
    """

    # We need to have this `currently_registering` variable because, in the course of making a DataClass with __slots__,
    # another class is created. So this will be called *twice* for every dataclass we annotate with this metaclass.
    currently_registering: ClassVar[Optional[type]] = None

    def __new__(mcs, name, bases, namespace, slots=True, **kwargs):
        # First: create the class in the normal way.
        cls = super().__new__(mcs, name, bases, namespace)

        if dataclasses.is_dataclass(cls):
            # If the class we're registering is already a Dataclass, it means it is a descendant of FrozenPyTreeDataclass or
            # MutablePyTreeDataclass.
            # This includes the children which are created when we create a dataclass with __slots__.

            if mcs.currently_registering is not None:
                # We've already created and annotated a class without __slots__, now we create the one with __slots__
                # that will actually get returned after from the __new__ method.
                assert mcs.currently_registering.__module__ == cls.__module__
                assert mcs.currently_registering.__name__ == cls.__name__
                mcs.currently_registering = None
                return cls

            else:
                assert name not in _RESERVED_NAMES, (
                    f"Class with name {name}: classes {_RESERVED_NAMES} don't inherit from a dataclass, so they should "
                    "not be in this branch."
                )

                # Otherwise we just mark the current class as what we're registering.
                if not issubclass(cls, (FrozenPyTreeDataclass, MutablePyTreeDataclass)):
                    raise TypeError(f"Dataclass {cls} should inherit from FrozenPyTreeDataclass or MutablePyTreeDataclass")

        mcs.currently_registering = cls

        if name in _RESERVED_NAMES:
            if not (
                namespace["__module__"] == "stable_baselines3.common.pytree_dataclass" and namespace["__qualname__"] == name
            ):
                raise TypeError(f"You cannot have another class named {name} with metaclass=_PyTreeDataclassMeta")

            if name == "_PyTreeDataclassBase":
                return cls
            frozen = kwargs.pop("frozen")
        else:
            if "frozen" in kwargs:
                raise TypeError(
                    "You should not specify frozen= for descendants of FrozenPyTreeDataclass or MutablePyTreeDataclass"
                )

            frozen = issubclass(cls, FrozenPyTreeDataclass)
            if frozen:
                if issubclass(cls, MutablePyTreeDataclass) or not issubclass(cls, FrozenPyTreeDataclass):
                    raise TypeError(f"Frozen dataclass {cls} should inherit from FrozenPyTreeDataclass")
            else:
                if not issubclass(cls, MutablePyTreeDataclass) or issubclass(cls, FrozenPyTreeDataclass):
                    raise TypeError(f"Mutable dataclass {cls} should inherit from MutablePyTreeDataclass")

            # Calling `dataclasses.dataclass` here, with slots, is what triggers the EARLY RETURN path above.
            cls = dataclasses.dataclass(frozen=frozen, slots=slots, **kwargs)(cls)

            assert issubclass(cls, CustomTreeNode)
            ot.register_pytree_node_class(cls, namespace=SB3_TREE_NAMESPACE)
        return cls


if TYPE_CHECKING:
    # For some reason, having an exotic metaclass confuses Pyright, which then doesn't report accessing non-existing
    # attributes of the dataclasses below as an error. Thus we remove the metaclass for type-checking purposes.

    class _PyTreeDataclassBase(CustomTreeNode[T], Generic[T]):
        """Dummy class without _PyTreeDataclassMeta as a metaclass."""

        def __init_subclass__(cls, frozen: bool = False) -> None:  # Allow passing `frozen=` in subclasses
            return super().__init_subclass__()

        def tree_flatten(self) -> tuple[Sequence[T], None, tuple[str, ...]]:  # pytype: disable=invalid-annotation
            return (), None, ()

        @classmethod
        def tree_unflatten(
            cls, metadata: None, children: Sequence[T]
        ) -> CustomTreeNode[T]:  # pytype: disable=invalid-annotation
            return cls()

else:

    class _PyTreeDataclassBase(CustomTreeNode[T], Generic[T], metaclass=_PyTreeDataclassMeta):
        """
        Provides utility methods common to both MutablePyTreeDataclass and FrozenPyTreeDataclass.

        However _PyTreeDataclassBase is *not* a dataclass. as it hasn't been passed through the `dataclasses.dataclass(...)`
        creation function.
        """

        _names_cache: ClassVar[Optional[Tuple[str, ...]]] = None

        # Mark this class as a dataclass, for type checking purposes.
        # Instead, it provides utility methods used by both Frozen and Mutable dataclasses.
        __dataclass_fields__: ClassVar[Dict[str, dataclasses.Field[Any]]]

        @classmethod
        def _names(cls) -> Tuple[str, ...]:
            if cls._names_cache is None:
                cls._names_cache = tuple(f.name for f in dataclasses.fields(cls))
            return cls._names_cache

        def __iter__(self):
            seq, _, _ = self.tree_flatten()
            return iter(seq)

        # The annotations here are invalid for Pytype because T does not appear in the rest of the function. But it does
        # appear as a parameter of the containing class, so it's actually not an error.
        def tree_flatten(self) -> tuple[Sequence[T], None, tuple[str, ...]]:  # pytype: disable=invalid-annotation
            names = self._names()
            return tuple(getattr(self, n) for n in names), None, names

        @classmethod
        def tree_unflatten(
            cls, metadata: None, children: Sequence[T]
        ) -> CustomTreeNode[T]:  # pytype: disable=invalid-annotation
            return cls(**dict(zip_strict(cls._names(), children)))


@dataclass_transform(frozen_default=True)  # pytype: disable=not-supported-yet
class FrozenPyTreeDataclass(_PyTreeDataclassBase[T], Generic[T], frozen=True):
    "Abstract class for immutable dataclass PyTrees"

    ...


@dataclass_transform(frozen_default=False)  # pytype: disable=not-supported-yet
class MutablePyTreeDataclass(_PyTreeDataclassBase[T], Generic[T], frozen=False):
    "Abstract class for mutable dataclass PyTrees"

    ...


# Manually expand the concrete type PyTree[th.Tensor] to make mypy happy.
# See links in https://github.com/metaopt/optree/issues/6, generic recursive types are not currently supported in mypy
TensorTree = Union[
    th.Tensor,
    Sequence["TensorTree"],
    Sequence[th.Tensor],
    Dict[Any, "TensorTree"],
    Dict[Any, th.Tensor],
    CustomTreeNode[th.Tensor],
    PyTree[th.Tensor],
    FrozenPyTreeDataclass[th.Tensor],
    MutablePyTreeDataclass[th.Tensor],
]

ConcreteTensorTree = TypeVar("ConcreteTensorTree", bound=TensorTree)


@overload
def tree_flatten(
    tree: TensorTree,
    is_leaf: Callable[[TensorTree], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = SB3_TREE_NAMESPACE,
) -> tuple[list[th.Tensor], ot.PyTreeSpec]:
    ...


@overload
def tree_flatten(
    tree: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = SB3_TREE_NAMESPACE,
) -> tuple[list[T], ot.PyTreeSpec]:
    ...


def tree_flatten(tree, is_leaf=None, *, none_is_leaf=False, namespace=SB3_TREE_NAMESPACE):
    """
    Flattens the PyTree (see `optree.tree_flatten`), expanding nodes using the SB3_TREE_NAMESPACE by default.
    """
    return ot.tree_flatten(tree, is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)


@overload
def tree_map(
    func: Callable[..., th.Tensor],
    tree: ConcreteTensorTree,
    *rests: TensorTree,
    is_leaf: Callable[[TensorTree], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = SB3_TREE_NAMESPACE,
) -> ConcreteTensorTree:
    ...


@overload
def tree_map(
    # This annotation is supposedly invalid for Pytype because U only appears once.
    func: Callable[..., U],  # pytype: disable=invalid-annotation
    tree: PyTree[T],
    *rests: Any,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = "",
) -> PyTree[U]:
    ...


def tree_map(func, tree, *rests, is_leaf=None, none_is_leaf=False, namespace=SB3_TREE_NAMESPACE):  # type: ignore
    """
    Maps a function over a PyTree (see `optree.tree_map`), over the trees in `tree` and `*rests`, expanding nodes using
    the SB3_TREE_NAMESPACE by default.
    """
    return ot.tree_map(func, tree, *rests, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)


def tree_empty(
    tree: ot.PyTree,
    *,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = SB3_TREE_NAMESPACE,
) -> bool:
    """Is the tree `tree` empty, i.e. without leaves?

    :param tree: the tree to check
    :param namespace: when expanding nodes, use this namespace
    :return: True iff the tree is empty
    """
    flattened_state, _ = ot.tree_flatten(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    return not bool(flattened_state)


def tree_index(
    tree: ConcreteTensorTree,
    idx: TensorIndex | tuple[TensorIndex, ...],
    *,
    is_leaf: None | Callable[[TensorTree], bool] = None,
    none_is_leaf: bool = False,
    namespace: str = SB3_TREE_NAMESPACE,
) -> ConcreteTensorTree:
    """
    Index each leaf of a PyTree of Tensors using the index `idx`.

    :param tree: the tree of tensors to index
    :param idx: the index to use
    :param is_leaf: whether to stop tree traversal at any particular node. `is_leaf(x: PyTree[Tensor])` should return
        True if the traversal should stop at `x`.
    :param none_is_leaf: Whether to consider `None` as a leaf that should be indexed.
    :param namespace:
    :returns: tree of indexed Tensors
    """
    return tree_map(lambda x: x[idx], tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
