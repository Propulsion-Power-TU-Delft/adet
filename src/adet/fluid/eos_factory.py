import logging
from typing import Any, ClassVar, Sequence

import casadi as cs

from adet.fluid.casadi_eos import CasadiEos
from adet.fluid.ideal_eos import AnalyticalFluidState
from adet.tools.coolprop_utils import inames_from_id
from adet.varspec import VarSpec

logger = logging.getLogger(__name__)


class EosFactory:
    instance_counter: ClassVar[int] = 0

    def __init__(self, fluid_state: Any | AnalyticalFluidState) -> None:
        self.fluid_state = fluid_state

    def make_eos(
        self,
        input_pair: int,
        out_properties: Sequence[VarSpec],
        length: int,
        name: str = '',
    ) -> cs.Function | CasadiEos:
        eos_obj = self.fluid_state

        # Convert VarSpecs to strings
        out_props_names = [s.symbol for s in out_properties]

        if not name:
            name += f'generic_eos_pair{input_pair}'

        if isinstance(
            self.fluid_state,
            AnalyticalFluidState,
        ):
            return self._make_symbolic_eos(
                eos_obj,
                input_pair,
                out_props_names,
                length,
                name,
            )
        else:
            return self._make_external_eos(
                eos_obj,
                input_pair,
                out_props_names,
                length,
                name,
            )

    def _make_symbolic_eos(
        self,
        fl_state: AnalyticalFluidState,
        input_pair: int,
        output_quantities: tuple[str, ...] | list[str],
        length: int,
        name: str,
    ) -> cs.Function:
        pair_vars = inames_from_id(input_pair)

        input_syms = [cs.MX.sym(var, length) for var in pair_vars]
        fl_state.update(input_pair, *input_syms)  # Update with symbols
        # Extract symbols
        output_syms = [getattr(fl_state, qty)() for qty in output_quantities]
        # Create updater func
        updater_func = cs.Function(
            name, input_syms, output_syms, pair_vars, output_quantities
        )

        return updater_func

    def _make_external_eos(
        self,
        external_state: Any,
        input_pair: int,
        output_quantities: tuple[str, ...] | list[str],
        length: int,
        name: str,
    ) -> CasadiEos:
        eos_callback = CasadiEos(
            name,
            external_state,
            input_pair,
            output_quantities,
            length,
        )

        return eos_callback
