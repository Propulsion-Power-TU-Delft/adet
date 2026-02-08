from dataclasses import dataclass
import logging
from typing import Generic, TypeVar

from adet.fluid.symbolic_eos import SymbolicAbstractState


logger = logging.getLogger(__name__)

E = TypeVar('E')  # External fluid object typevar


class FluidModel:
    """Parent class for identifying fluid models"""

    def __init__(self, eos_object):
        self.eos_object = eos_object

    def get_eos_object(self):
        return self.eos_object


class EmptyFluidModel(FluidModel):
    def __init__(self, eos_object=None):
        pass

    def get_eos_object(self):
        raise AttributeError('Empty fluid model cannot be called')


@dataclass
class AnalyticalFluidModel(FluidModel):
    """
    Models which do not require passing through
    external thermodynamic libraries
    """

    eos_object: SymbolicAbstractState

    def get_eos_object(self) -> SymbolicAbstractState:
        return super().get_eos_object()


@dataclass
class ExternalFluidModel(FluidModel, Generic[E]):
    eos_object: E

    def get_eos_object(self) -> E:
        return super().get_eos_object()

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
