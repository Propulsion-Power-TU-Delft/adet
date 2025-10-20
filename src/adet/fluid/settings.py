from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Generic, Any, TypeVar

from pint import Quantity

from adet.equations.base_equation import EquationBase
from adet.equations.ideal_gas import IdealRltEos, IdealStcEos, IdealTotEos


logger = logging.getLogger(__name__)


# - - - - - - - - - - - - - - - FLUID MODELS
class FluidModel(ABC):
    """Abstract base for any fluid model backend."""

    @abstractmethod
    def get_constraints(self) -> dict[str, Any]:
        """Return derived quantities for this fluid model."""
        raise NotImplementedError


class AnalyticalFluidModel(FluidModel):
    """
    Models which do not require passing through
    external thermodynamic libraries
    """

    @abstractmethod
    def get_equations(self) -> tuple[EquationBase, ...]:
        raise NotImplementedError


class EmptyFluidModel(AnalyticalFluidModel):
    def get_equations(self):
        return ()

    def get_constraints(self) -> dict[str, Any]:
        return {}


@dataclass
class IdealGasModel(AnalyticalFluidModel):
    R: float
    gamma: float
    T_ref: float = 1.0
    p_ref: float = 1.0

    def get_equations(self):
        return IdealStcEos(), IdealTotEos(), IdealRltEos()

    def get_constraints(self) -> dict[str, Any]:
        cpmass_mag = self.R * self.gamma / (self.gamma - 1)
        cvmass_mag = cpmass_mag / self.gamma
        return {
            'oth': {
                'cpmassid': Quantity(cpmass_mag, 'J / kg / K'),
                'cvmassid': Quantity(cvmass_mag, 'J / kg / K'),
                'T_ref': Quantity(self.T_ref, 'K'),
                'p_ref': Quantity(self.p_ref, 'Pa'),
            }
        }


@dataclass
class RealGasModel(AnalyticalFluidModel):
    R: float
    gamma: float
    Z: float
    T_ref: float = 1.0
    p_ref: float = 1.0

    def get_expression(self):
        pass

    def get_constraints(self) -> dict[str, Any]:
        cpmass_mag = self.R * self.gamma / (self.gamma - 1)
        cvmass_mag = cpmass_mag / self.gamma
        return {
            'oth': {
                'cpmassid': Quantity(cpmass_mag, 'J / kg / K'),
                'cvmassid': Quantity(cvmass_mag, 'J / kg / K'),
                'T_ref': Quantity(self.T_ref, 'K'),
                'p_ref': Quantity(self.p_ref, 'Pa'),
                'Z': Quantity(self.Z, 'Dimensionless'),
            }
        }


T = TypeVar('T')


@dataclass
class AbstractStateModel(FluidModel, Generic[T]):
    eos_object: T

    def get_constraints(self) -> dict[str, Any]:
        return {}

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj

        # Just copy the same object
        # Abstract state has problems being deepcopied
        new_obj.eos_object = self.eos_object

        return new_obj


# - - - - - - - - - - - - - - - FLUID SETTINGS
@dataclass
class FluidSettings:
    model: FluidModel
    update_variables: tuple[str, ...] = ()
    update_length: int = 2

    def get_virtual_constraints(self) -> dict[str, Any]:
        return self.model.get_constraints()


if __name__ == '__main__':
    import CoolProp as cp
    from copy import deepcopy

    eos = cp.AbstractState('HEOS', 'R134a')

    sett = FluidSettings(
        AbstractStateModel(eos),
        ('p', 'T'),
        2,
    )

    new_sett = deepcopy(sett)
