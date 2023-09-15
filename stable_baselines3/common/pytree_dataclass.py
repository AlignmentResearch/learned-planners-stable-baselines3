import dataclasses
from typing import Optional, Sequence, Type

import optree as ot
from typing_extensions import dataclass_transform

__all__ = ["register_dataclass_as_pytree", "dataclass_frozen_pytree", "tree_empty", "OT_NAMESPACE"]

OT_NAMESPACE = "stable-baselines3"


def register_dataclass_as_pytree(Cls, whitelist: Optional[Sequence[str]] = None):
    """Register a dataclass as a pytree, using the given whitelist of field names.

    :param Cls: The dataclass to register.
    :param whitelist: The names of the fields to include in the pytree. If None, all fields are included.
    :return: The dataclass, with the pytree registration applied. This is useful to be able to register a decorator.
    """

    assert dataclasses.is_dataclass(Cls)

    names = tuple(f.name for f in dataclasses.fields(Cls) if whitelist is None or f.name in whitelist)

    def flatten_fn(inst):
        return (getattr(inst, n) for n in names), None, names

    def unflatten_fn(context, values):
        return Cls(**dict(zip(names, values)))

    ot.register_pytree_node(Cls, flatten_fn, unflatten_fn, namespace=OT_NAMESPACE)

    Cls.__iter__ = lambda self: iter(getattr(self, n) for n in names)
    return Cls


@dataclass_transform()
def dataclass_frozen_pytree(Cls: Type, **kwargs) -> Type[ot.PyTree]:
    """Decorator to make a frozen dataclass and register it as a PyTree."""
    dataCls = dataclasses.dataclass(frozen=True, slots=True, **kwargs)(Cls)
    register_dataclass_as_pytree(dataCls)
    return dataCls


def tree_empty(tree: ot.PyTree) -> bool:
    flattened_state, _ = ot.tree_flatten(tree, namespace=OT_NAMESPACE)
    return not bool(len(flattened_state))
