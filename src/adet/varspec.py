from dataclasses import dataclass, replace
from enum import Enum
from typing import Annotated

from casadi import MX
from pint import Quantity

DEF_NODE = -1
DEF_STATE = None


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
    node: int = DEF_NODE
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

    def at_node(self, node: int):
        return replace(self, node=node)

    def _with_bounds(self, bounds: tuple[float, float] | None):
        return replace(self, bounds=bounds)

    def _with_guess(self, guess: float | None):
        return replace(self, guess=guess)

    def _with_symbol(self, symbol: str | None):
        return replace(self, symbol=symbol)

    def full_symbol(self, index: bool = False) -> str:
        prefix = self.state.value if self.state else ''
        postfix = str(self.node) if index else ''
        return prefix + self.symbol + postfix

    def __str__(self) -> str:
        repr = f'{self.__class__.__name__}: {self.symbol}, node={self.node}'

        if self.state is not None:
            repr += f', state={self.state.value.removesuffix("_")}'

        return repr

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def Glob(self):
        return replace(self, state=DEF_STATE, node=DEF_NODE)

    @property
    def Hint(self):
        """Type hint to be used in function signature"""
        return Annotated[MX | Quantity, self]
