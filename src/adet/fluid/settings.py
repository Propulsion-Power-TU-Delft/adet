import copy
import logging
from dataclasses import dataclass
from typing import Any

from adet.constants import COOLPROP_NAMES_MAP
from adet.fluid.ideal_eos import AnalyticalFluidState
from adet.tools.coolprop_utils import pair_id_from_tuple
from adet.varspec import VarSpec

logger = logging.getLogger(__name__)


@dataclass
class FluidSettings:
    fluid_state: Any | AnalyticalFluidState
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

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj

        # Just copy the same object because
        # Abstract state has problems being deepcopied
        new_obj.fluid_state = self.fluid_state
        new_obj.update_length = self.update_length
        new_obj.update_variables = self.update_variables

        return new_obj


if __name__ == '__main__':
    from copy import deepcopy

    from CoolProp import AbstractState

    from adet.variables import ThermoVariables

    eos = AbstractState('HEOS', 'R134a')

    sett = FluidSettings(
        eos,
        (
            ThermoVariables.Pressure,
            ThermoVariables.Temperature,
        ),
        2,
    )

    new_sett_dc = deepcopy(sett)
    new_sett_cp = copy.copy(sett)
