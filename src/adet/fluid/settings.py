from adet.tools.coolprop_utils import pair_id_from_tuple
from adet.constants import COOLPROP_NAMES_MAP, CoolProperties
from adet.equations.variables import ThermoVariables
from adet.equations.varspec import VarSpec
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


# TODO: Make this recognize which model you are feeding it and
# adjust its type accordingly


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
    update_variables: tuple[VarSpec, ...] = ()
    update_length: int = 2

    def __post_init__(self):
        # Sort variables in the correct order
        sorted_upd = tuple(
            sorted(
                self.update_variables,
                key=lambda x: COOLPROP_NAMES_MAP[x.symbol],
            )
        )
        self.update_variables = tuple(sp.Plain for sp in sorted_upd)

    @property
    def input_pair(self) -> int:
        upd_pties = tuple(x.symbol for x in self.update_variables)
        return pair_id_from_tuple(upd_pties)


if __name__ == '__main__':
    import CoolProp as cp
    from copy import deepcopy

    eos = cp.AbstractState('HEOS', 'R134a')

    sett = FluidSettings(
        ExternalFluidModel(eos),
        (
            ThermoVariables.Pressure,
            ThermoVariables.Temperature,
        ),
        2,
    )

    new_sett = deepcopy(sett)
