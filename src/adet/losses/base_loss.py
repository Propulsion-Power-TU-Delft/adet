"""
Library of basics loss models
"""

# from inspect import getfullargspec
# from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, TypeAlias

from adet.equations import EquationBase
from adet.variables import Losses, VarSpec

if TYPE_CHECKING:
    from adet.assemblers import SystemAssembler

LossType: TypeAlias = Literal['delta_smass', 'delta_tot_p', 'delta_tot_hmass']

# The canonical per-mechanism entropy-rise specs a `LossModel` can declare
# as its output -- `Ds_total`/`Ds_main` are aggregates, not mechanisms, and
# excluded. Matched against each equation's args by `.Glob` (symbol only),
# since a `LossModel` may also declare unrelated ad-hoc ``ds_*``-named
# intermediates of its own (e.g. `EndwallLoss`'s per-station entropy march)
# that are not members of `Losses` and must not be mistaken for its output.
_MECHANISM_DS_SPECS = {
    spec.Glob
    for name, spec in vars(Losses).items()
    if name.startswith('Ds_') and name not in ('Ds_total', 'Ds_main')
}


class LossModel(EquationBase):
    pass


def _ds_mechanism_spec(equation: LossModel) -> VarSpec:
    """The single ``Losses.Ds_*`` argument a `LossModel` equation declares
    as its output -- its "footprint". Every `LossModel` subclass declares
    exactly one such argument (checked across `losses/rit.py`,
    `losses/mixing.py`, etc.)."""
    ds_specs = [spec for spec in equation.arg_specs if spec.Glob in _MECHANISM_DS_SPECS]

    if len(ds_specs) != 1:
        raise ValueError(
            f'{equation.__class__.__name__} does not declare exactly one '
            f'Ds_* argument (found {ds_specs}); cannot use it as a loss '
            f'mechanism footprint'
        )

    return ds_specs[0]


def collect_loss_mechanisms(system: 'SystemAssembler', node: int) -> list[VarSpec]:
    """Every per-mechanism ``Losses.Ds_*`` spec produced by a `LossModel`
    equation already added to `system` and landing on `node`, found by
    their `LossModel` "footprint" rather than listed by hand. Used to feed
    `LossCoefficient`/`EfficiencyDeficit`
    (`equations/nondimensional.py`)."""
    specs = []
    for equation, position in system.data.equations.items():
        if not isinstance(equation, LossModel):
            continue

        local_to_abs = dict(zip(equation.arg_nodes, position, strict=True))
        ds_spec = _ds_mechanism_spec(equation)
        if local_to_abs[ds_spec.node] == node:
            specs.append(ds_spec.Glob)

    return specs


# class LossModel(EquationBase, ABC):
#     definition: LossType
#     """Which definition of losses this model returns in its `value` function"""
#
#     identifier: str
#     """Unique identifier for that model"""
#
#     def __init__(
#         self,
#         scaling_factor: list[float] | None = None,
#         **parameters,
#     ):
#         self._generate_residual_method()
#         self.parameters = parameters
#         super().__init__(scaling_factor)
#
#     def __init_subclass__(cls) -> None:
#         cls.VALUE_VARIABLE = f'oth_{cls.definition}_{cls.identifier}1'
#
#         if not hasattr(cls, 'definition'):
#             raise AttributeError(f'{cls.__name__} does not define its type')
#
#         if not hasattr(cls, 'identifier'):
#             raise AttributeError(f'{cls.__name__} does not define its identifier')
#
#         cls._generate_residual_method()
#
#     def read_and_validate_arguments(self, all_arguments: list[str]):
#         all_arguments = [self.VALUE_VARIABLE] + getfullargspec(self.value).args[1:]
#         return super()._read_and_validate_arguments(all_arguments)
#
#     @classmethod
#     def _generate_residual_method(cls):
#         """Generate a residual function formulation of the loss model"""
#
#         def generated_res_function(self, *args):
#             return args[0] - self.value(*args[1:])
#
#         setattr(cls, 'residual', generated_res_function)
#
#     @abstractmethod
#     def value(self, *args, **kwargs):
#         raise NotImplementedError
#
#     def residual(self, *args, **kwargs):
#         """This is generated dynamically"""
#         pass
