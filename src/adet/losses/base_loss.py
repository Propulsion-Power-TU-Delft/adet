"""
Library of basics loss models
"""

# from inspect import getfullargspec
# from abc import ABC, abstractmethod
from typing import Literal, TypeAlias
from adet.equations import EquationBase


LossType: TypeAlias = Literal['delta_smass', 'delta_tot_p', 'delta_tot_hmass']


class LossModel(EquationBase):
    pass


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
