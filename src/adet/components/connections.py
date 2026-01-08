from dataclasses import dataclass
from typing import Any
from pint import Quantity
from pint.facets.plain import PlainQuantity


@dataclass
class Inlet:
    boundary_conditions: dict[str, dict[str, Any]]


@dataclass
class Shaft:
    omega: float | PlainQuantity
    is_constrained: bool

    def __post_init__(self):
        if isinstance(self.omega, float):
            self.omega = Quantity(self.omega, 'rad/s')
