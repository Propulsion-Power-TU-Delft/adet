from abc import abstractmethod
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Generic, TypeVar

from adet.fluid.symbolic_eos import IdealGasEos


if TYPE_CHECKING:
    from adet.equations.base_equation import EquationBase


T = TypeVar('T')

logger = logging.getLogger(__name__)


class FluidModel:
    """Parent class for identifying fluid models"""

    pass


class AnalyticalFluidModel(FluidModel):
    """
    Models which do not require passing through
    external thermodynamic libraries
    """

    @abstractmethod
    def geo_eos_object(self):
        raise NotImplementedError


class EmptyFluidModel(AnalyticalFluidModel):
    def geo_eos_object(self):
        return ()


@dataclass
class IdealGasModel(AnalyticalFluidModel):
    def geo_eos_object(self):
        return IdealGasEos


@dataclass
class ExternalFluidModel(FluidModel, Generic[T]):
    eos_object: T

    def __copy__(self):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        new_obj.eos_object = self.eos_object
        return new_obj

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj

        # Just copy the same object because
        # Abstract state has problems being deepcopied
        new_obj.eos_object = self.eos_object

        return new_obj


@dataclass
class FluidSettings:
    model: FluidModel
    update_variables: tuple[str, ...] = ()
    update_length: int = 2


if __name__ == '__main__':
    import CoolProp as cp
    from copy import deepcopy

    eos = cp.AbstractState('HEOS', 'R134a')

    sett = FluidSettings(
        ExternalFluidModel(eos),
        ('p', 'T'),
        2,
    )

    new_sett = deepcopy(sett)
