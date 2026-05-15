from adet.varspec import VarSpec
from dataclasses import dataclass
from typing import Any
from pint.facets.plain import PlainQuantity


@dataclass
class Inlet:
    boundary_conditions: dict[VarSpec, Any]


class Shaft:
    def __init__(self, omega: float | PlainQuantity, is_constrained: bool):
        self.omega = omega
        self.is_constrained = is_constrained

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self
