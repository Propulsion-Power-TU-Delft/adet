import logging
from dataclasses import dataclass
from typing import Generic, TypeVar

from adet.constants import COOLPROP_NAMES_MAP
from adet.tools.coolprop_utils import pair_id_from_tuple
from adet.varspec import VarSpec

logger = logging.getLogger(__name__)

E = TypeVar('E')  # External fluid object typevar


@dataclass
class FluidModel(Generic[E]):
    eos_object: E

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
        self.update_variables = tuple(sp.Glob for sp in sorted_upd)

    @property
    def input_pair(self) -> int:
        upd_pties = tuple(x.symbol for x in self.update_variables)
        return pair_id_from_tuple(upd_pties)


if __name__ == '__main__':
    from copy import deepcopy

    import CoolProp as cp

    from adet.variables import ThermoVariables

    eos = cp.AbstractState('HEOS', 'R134a')

    sett = FluidSettings(
        FluidModel(eos),
        (
            ThermoVariables.Pressure,
            ThermoVariables.Temperature,
        ),
        2,
    )

    new_sett = deepcopy(sett)
