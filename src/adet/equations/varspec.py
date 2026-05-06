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
@dataclass(frozen=True, eq=False)
class VarSpec:
    symbol: str
    unit: str
    guess: float | None = None
    bounds: tuple[float, float] | None = None
    node: int = DUMMY_NODE_IDX
    scalar: bool = False
    state: NodeStates | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VarSpec):
            return NotImplemented
        return (
            self.symbol == other.symbol
            and self.state == other.state
            and self.node == other.node
        )

    def __hash__(self) -> int:
        return hash((self.symbol, self.state, self.node))

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
