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
