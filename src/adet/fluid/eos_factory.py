import logging
from typing import ClassVar, Sequence

import casadi as cs

from adet.fluid.casadi_eos import CasadiEos
from adet.fluid.settings import FluidModel
from adet.fluid.symbolic_eos import SymbolicAbstractState
from adet.tools.coolprop_utils import inames_from_id
from adet.varspec import VarSpec

logger = logging.getLogger(__name__)


class EosFactory:
    instance_counter: ClassVar[int] = 0

    def __init__(self, fluid_model: FluidModel) -> None:
        self.fluid_model = fluid_model

    def make_eos(
        self,
        input_pair: int,
        out_properties: Sequence[VarSpec],
        length: int,
        name: str = '',
    ) -> cs.Function | CasadiEos:
        eos_obj = self.fluid_model.eos_object

        # Convert VarSpecs to strings
        out_props_names = [s.symbol for s in out_properties]

        if not name:
            name += f'generic_eos_pair{input_pair}'

        if isinstance(
            self.fluid_model.eos_object,
            SymbolicAbstractState,
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
        eos_object: SymbolicAbstractState,
        input_pair: int,
        output_quantities: tuple[str, ...] | list[str],
        length: int,
        name: str,
    ) -> cs.Function:
        pair_vars = inames_from_id(input_pair)

        input_syms = [cs.MX.sym(var, length) for var in pair_vars]
        eos_object.update(input_pair, *input_syms)  # Update with symbols
        # Extract symbols
        output_syms = [getattr(eos_object, qty)() for qty in output_quantities]
        # Create updater func
        updater_func = cs.Function(
            name, input_syms, output_syms, pair_vars, output_quantities
        )

        return updater_func

    def _make_external_eos(
        self,
        eos_object: CasadiEos,
        input_pair: int,
        output_quantities: tuple[str, ...] | list[str],
        length: int,
        name: str,
    ) -> CasadiEos:
        eos_callback = CasadiEos(
            name,
            eos_object,
            input_pair,
            output_quantities,
            length,
        )

        return eos_callback
