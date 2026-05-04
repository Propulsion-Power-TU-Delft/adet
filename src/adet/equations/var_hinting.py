# New implementation
from pint.registry import Quantity
import casadi as cs
from typing import Annotated, TypeVar, Generic, Type
from dataclasses import dataclass
from enum import Enum


class NodeStates(Enum):
    STATIC = 'stc'
    TOTAL = 'tot'
    RELTOT = 'rlt'


# *** Specifications for a single variable
@dataclass(frozen=True)
class VarSpec:
    symbol: str
    description: str
    unit: str
    node: int = 0
    state: NodeStates | None = None


# *** Enums for storing the actual variable specs
class ThermoVariables(Enum):
    Entropy = VarSpec('smass', 'Specific entropy', 'J / kg / K')
    Density = VarSpec('rhomass', 'Density', 'kg / m**3')
    Pressure = VarSpec('p', 'Pressure', 'Pa')
    Enthalpy = VarSpec('hmass', 'Specific enthalpy', 'J / kg')
    Temperature = VarSpec('T', 'Temperature', 'K')
    InternalEnergy = VarSpec('umass', 'Internal Energy', 'J / kg')
    Cp = VarSpec('cpmass', 'Spefic heat (pressure)', 'J / kg')
    Cv = VarSpec('cvmass', 'Spefic heat (volume)', 'J / kg')


class GenericVariables(Enum):
    V_mag = VarSpec('V', 'Absolute velocity', 'm / s')
    V_tan = VarSpec('Vt', 'Absolute velocity (tangential)', 'm / s')
    V_mer = VarSpec('Vm', 'Absolute velocity (meridional)', 'm / s')
    W_mag = VarSpec('W', 'Relative Velocity', 'm / s')
    W_tan = VarSpec('Wt', 'Relative Velocity (tangential)', 'm / s')
    W_mer = VarSpec('Wm', 'Relative Velocity (meridional)', 'm / s')
    RelAngle = VarSpec('beta', 'Relative flow angle', 'rad')
    AbsAngle = VarSpec('alpha', 'Absolute flow angle', 'rad')


# Hint typevar
H = TypeVar('H', bound=Enum)


# Template class for type hint storage
class VariableHints(Generic[H]):
    def __init__(self, node: int, state: NodeStates | None, var_enum: Type[H]):
        """
        Parameters
        ----------
        prefix: str
            Prefix to the variable symbol
        var_enum: Type[H]
            Enum class from which to draw the variable specs
        """
        self._state = state
        self._node = node
        self._var_enum = var_enum

    def __getattr__(self, name: str):
        var_spec = self._var_enum[name].value

        return Annotated[
            cs.MX | Quantity,
            VarSpec(
                var_spec.symbol,
                var_spec.description,
                var_spec.unit,
                self._node,
                self._state,
            ),
        ]


class ThermoHints(VariableHints[ThermoVariables]):
    def __init__(self, state: NodeStates, node: int):
        super().__init__(node, state, ThermoVariables)


class OtherHints(VariableHints[GenericVariables]):
    def __init__(self, node: int):
        super().__init__(node, None, GenericVariables)


class CustomVar:
    def __init__(self, symbol: str, node: int, unit: str):
        self._var_spec = VarSpec(symbol, 'Custom symbol', unit, node)

    @property
    def Type(self) -> type[Annotated[cs.MX | Quantity, VarSpec]]:
        return Annotated[cs.MX | Quantity, self._var_spec]


class NodeHints(OtherHints):
    def __init__(self, index: int):
        super().__init__(index)

        self._stc = ThermoHints(NodeStates.STATIC, index)
        self._tot = ThermoHints(NodeStates.TOTAL, index)
        self._rlt = ThermoHints(NodeStates.RELTOT, index)

    @property
    def tot(self) -> ThermoHints:
        return self._tot

    @property
    def stc(self) -> ThermoHints:
        return self._stc

    @property
    def rlt(self) -> ThermoHints:
        return self._rlt

    @property
    def cust(self) -> ThermoHints:
        return self._rlt
