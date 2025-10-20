from dataclasses import dataclass
from pint import Quantity
from pint.facets.plain import PlainQuantity

from adet.components import BoundaryConditions


@dataclass
class Inlet:
    boundary_conditions: BoundaryConditions


@dataclass
class Shaft:
    omega: float | PlainQuantity

    def __post_init__(self):
        if isinstance(self.omega, float):
            self.omega = Quantity(self.omega, 'rad/s')
