from pint import Quantity
from casadi import MX
from typing import Annotated
from dataclasses import dataclass
from enum import Enum

class NodeStates(Enum):
    STATIC = 'stc'
    TOTAL = 'tot'
    RELTOT = 'rlt'

@dataclass(frozen=True)
class VarSpec:
    symbol: str
    description: str
    unit: str
    node: int = ...
    state: None | NodeStates = ...
    scalar: bool = ...

    def _with_state(self, state: NodeStates | None) -> 'VarSpec': ...
    def _at_node(self, node: int) -> 'VarSpec': ...
    Hint = Annotated[MX | Quantity, 'VarSpec']
