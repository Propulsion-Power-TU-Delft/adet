from dataclasses import dataclass
from typing import Any
from pint.facets.plain import PlainQuantity


@dataclass
class Inlet:
    boundary_conditions: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Shaft:
    omega: float | PlainQuantity
    is_constrained: bool

    # ------------------------------
    # WARN: * Could be confusing *
    # These method ensure that when reusing a BladeRow
    # the shaft object is preserved in the new instance,
    # so shaft links occur correctly
    # -------------------------------
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self
