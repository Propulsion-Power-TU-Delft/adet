import logging
from typing import ClassVar, Generic, TypeVar

from adet.fluid.casadi_eos import CasadiEos
from adet.fluid.settings import AnalyticalFluidModel, ExternalFluidModel, FluidModel
from adet.fluid.symbolic_eos import SymbolicAbstractState
from adet.tools.coolprop_utils import get_input_names


logger = logging.getLogger(__name__)

M = TypeVar('M', bound=FluidModel)


class EosFactory(Generic[M]):
    instance_counter: ClassVar[int] = 0

    def __init__(self, fluid_model: M) -> None:
        self.fluid_model = fluid_model

    def make_eos(
        self,
        input_pair: int,
        output_quantities: tuple[str, ...] | list[str],
        length: int,
        name: str = '',
    ):
        eos_obj = self.fluid_model.get_eos_object()

        if isinstance(self.fluid_model, ExternalFluidModel):
            return self.make_external_eos(
                eos_obj, input_pair, output_quantities, length, name
            )
        elif isinstance(self.fluid_model, AnalyticalFluidModel):
            return self.make_analytical_eos(eos_obj, input_pair, output_quantities)
        else:
            raise TypeError('Unknown fluid model type')

    def make_analytical_eos(
        self,
        eos_object: SymbolicAbstractState,
        input_pair: int,
        output_quantities: tuple[str, ...] | list[str],
    ):
        def updater_func(x, y):
            eos_object.update(input_pair, x, y)
            return [eos_object.get_property(qty) for qty in output_quantities]

        return updater_func

    def make_external_eos(
        self,
        eos_object: CasadiEos,
        input_pair: int,
        output_quantities: tuple[str, ...] | list[str],
        length: int,
        name: str = '',
    ):
        pair_name = ''.join(get_input_names(input_pair))

        if not name:
            name = f'eos_{pair_name}_l{length}'

        eos_callback = CasadiEos(
            name,
            eos_object,
            input_pair,
            output_quantities,
            length,
        )

        return eos_callback
