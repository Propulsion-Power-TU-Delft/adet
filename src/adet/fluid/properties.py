"""
Simple mixin class to extract some composed properties from the flow node
This can be used as an alternative to defining trivial equations to extract
secondary or composed properties
"""

from typing import Callable, Protocol, runtime_checkable, cast
from numpy.typing import NDArray

from adet.variables import KinematicContainer, VariableContainer
from adet.variables import ThermostateContainer


@runtime_checkable
class HasThermodynamicProperties(Protocol):
    """
    Protocol defining required properties for gas property calculations
    within the :class:`DerivedGasProperties`
    """

    # Require thermo states
    @property
    def stc(self) -> ThermostateContainer: ...
    @property
    def tot(self) -> ThermostateContainer: ...
    @property
    def rlt(self) -> ThermostateContainer: ...
    @property
    def kin(self) -> KinematicContainer: ...
    @property
    def oth(self) -> VariableContainer: ...


def thermo_property(func: Callable) -> property:
    """Decorator that casts self to HasThermodynamicProperties
    before calling the property getter"""

    def wrapper(self):
        self_with_props = cast(HasThermodynamicProperties, self)
        return func(self_with_props)

    return property(wrapper)


class GasPropertiesMixin:
    """Utility class for calculating derived gas properties"""

    @thermo_property
    def Mach(self: HasThermodynamicProperties) -> NDArray:
        return self.kin.get('V') / self.stc.get('speed_sound')

    @thermo_property
    def MachRel(self: HasThermodynamicProperties) -> NDArray:
        return self.kin.get('W') / self.stc.get('speed_sound')
