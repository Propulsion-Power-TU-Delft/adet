import casadi as cs
from dataclasses import dataclass
from enum import Enum
from pint.registry import Quantity
from typing import Annotated, Generic, TypeVar

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
    state: NodeStates | None = ...

class ThermoVariables(Enum):
    Entropy = Annotated[cs.MX | Quantity, VarSpec]
    Density = Annotated[cs.MX | Quantity, VarSpec]
    Pressure = Annotated[cs.MX | Quantity, VarSpec]
    Enthalpy = Annotated[cs.MX | Quantity, VarSpec]
    Temperature = Annotated[cs.MX | Quantity, VarSpec]
    InternalEnergy = Annotated[cs.MX | Quantity, VarSpec]
    Cp = Annotated[cs.MX | Quantity, VarSpec]
    Cv = Annotated[cs.MX | Quantity, VarSpec]

class GenericVariables(Enum):
    V_mag = Annotated[cs.MX | Quantity, VarSpec]
    V_tan = Annotated[cs.MX | Quantity, VarSpec]
    V_mer = Annotated[cs.MX | Quantity, VarSpec]
    W_mag = Annotated[cs.MX | Quantity, VarSpec]
    W_tan = Annotated[cs.MX | Quantity, VarSpec]
    W_mer = Annotated[cs.MX | Quantity, VarSpec]
    RelAngle = Annotated[cs.MX | Quantity, VarSpec]
    AbsAngle = Annotated[cs.MX | Quantity, VarSpec]

H = TypeVar('H', bound=Enum)

class VariableHints(Generic[H]):
    def __init__(
        self, node: int, state: NodeStates | None, var_enum: type[H]
    ) -> None: ...
    def __getattr__(self, name: str): ...

class ThermoHints(VariableHints[ThermoVariables]):
    def __init__(self, state: NodeStates, node: int) -> None: ...

class OtherHints(VariableHints[GenericVariables]):
    def __init__(self, node: int) -> None: ...

class CustomVar:
    def __init__(self, symbol: str, node: int, unit: str) -> None: ...
    Type = Annotated[cs.MX | Quantity, VarSpec]

class NodeHints(OtherHints):
    def __init__(self, index: int) -> None: ...
    @property
    def tot(self) -> ThermoHints: ...
    @property
    def stc(self) -> ThermoHints: ...
    @property
    def rlt(self) -> ThermoHints: ...
    @property
    def cust(self) -> ThermoHints: ...
