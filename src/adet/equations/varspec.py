from pint import Quantity
from typing import Annotated
from casadi import MX
from dataclasses import dataclass
from enum import Enum


class NodeStates(Enum):
    STATIC = 'stc_'
    TOTAL = 'tot_'
    RELTOT = 'rlt_'


# *** Specifications for a single variable
@dataclass(frozen=True)
class VarSpec:
    symbol: str
    description: str
    unit: str
    node: int = 0
    state: None | NodeStates = None
    scalar: bool = False

    def _with_state(self, state: NodeStates):
        return VarSpec(
            self.symbol, self.description, self.unit, self.node, state, self.scalar
        )

    def _at_node(self, node: int):
        return VarSpec(
            self.symbol, self.description, self.unit, node, self.state, self.scalar
        )

    @property
    def Hint(self):
        """Type hint to be used in function signature"""
        return Annotated[MX | Quantity, self]
