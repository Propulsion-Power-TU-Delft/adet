# New implementation
from pint.registry import Quantity
import casadi as cs
from typing import Annotated, TypeVar, Generic, Type
from dataclasses import dataclass
from enum import Enum


# *** Specifications for a single variable
@dataclass(frozen=True)
class VarSpec:
    symbol: str
    description: str
    unit: str


# *** Enums for storing the actual variable specs
class ThermoVariables(Enum):
    Entropy = VarSpec('smass', 'Specific entropy', 'J / kg / K')
    Pressure = VarSpec('p', 'Pressure', 'Pa')
    Enthalpy = VarSpec('hmass', 'Specific enthalpy', 'J / kg')
    Temperature = VarSpec('T', 'Temperature', 'K')


class OtherVariables(Enum):
    V_mag = VarSpec('V', 'Absolute velocity', 'm / s')
    V_tan = VarSpec('Vt', 'Absolute velocity (tangential)', 'm / s')
    V_mer = VarSpec('Vm', 'Absolute velocity (meridional)', 'm / s')
    W_mag = VarSpec('W', 'Relative Velocity', 'm / s')
    W_tan = VarSpec('Wt', 'Relative Velocity (tangential)', 'm / s')
    W_mer = VarSpec('Wm', 'Relative Velocity (meridional)', 'm / s')


# Hint typevar
H = TypeVar('H', bound=Enum)


# Template class for type hint storage
class VariableHints(Generic[H]):
    def __init__(self, prefix: str, postfix: str, var_enum: Type[H]):
        """
        Parameters
        ----------
        prefix: str
            Prefix to the variable symbol
        var_enum: Type[H]
            Enum class from which to draw the variable specs
        """
        self._prefix = prefix
        self._postfix = postfix
        self._var_enum = var_enum

    def __getattr__(self, name: str):
        var_spec = self._var_enum[name].value

        return Annotated[
            cs.MX | Quantity,
            VarSpec(
                self._prefix + var_spec.symbol,
                var_spec.description,
                var_spec.unit,
            ),
        ]


class ThermoHints(VariableHints[ThermoVariables]):
    def __init__(self, prefix: str, postfix: str):
        super().__init__(prefix, postfix, ThermoVariables)


class OtherHints(VariableHints[OtherVariables]):
    def __init__(self, postfix: str):
        # NO PREFIX
        super().__init__('', postfix, OtherVariables)


TOT_PREFIX = 'tot_'
RLT_PREFIX = 'rlt_'
STC_PREFIX = 'stc_'


class NodeHints(OtherHints):
    def __init__(self, index: int):
        index_str = str(index)
        super().__init__(index_str)

        self._stc = ThermoHints(STC_PREFIX, index_str)
        self._tot = ThermoHints(TOT_PREFIX, index_str)
        self._rlt = ThermoHints(RLT_PREFIX, index_str)

    @property
    def tot(self) -> ThermoHints:
        return self._tot

    @property
    def stc(self) -> ThermoHints:
        return self._stc

    @property
    def rlt(self) -> ThermoHints:
        return self._rlt
