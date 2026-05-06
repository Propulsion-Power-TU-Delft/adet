from pint import Quantity
from typing import Annotated
from casadi import MX
from dataclasses import dataclass, replace
from enum import Enum


DUMMY_NODE_IDX = -1


class NodeStates(Enum):
    STATIC = 'stc_'
    TOTAL = 'tot_'
    RELTOT = 'rlt_'


# *** Specifications for a single variable
@dataclass(frozen=True)
class VarSpec:
    symbol: str
    unit: str
    guess: float | None = None
    bounds: tuple[float, float] | None = None
    node: int = DUMMY_NODE_IDX
    scalar: bool = False
    state: NodeStates | None = None

    def _with_state(self, state: NodeStates | None):
        return replace(self, state=state)

    def _at_node(self, node: int):
        return replace(self, node=node)

    def _with_bounds(self, bounds: tuple[float, float] | None):
        return replace(self, bounds=bounds)

    def _with_guess(self, guess: float | None):
        return replace(self, guess=guess)

    @property
    def Hint(self):
        """Type hint to be used in function signature"""
        return Annotated[MX | Quantity, self]
