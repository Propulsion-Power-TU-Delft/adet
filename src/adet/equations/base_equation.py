from adet.equations.var_hinting import NodeHints
import logging
from abc import ABC, abstractmethod
from inspect import getfullargspec
from typing import Any, Callable, ClassVar, Self, cast

import casadi as cs
import sympy as sp

from adet.fluid.casadi_eos import CasadiEos
from adet.tools.context import override_operators
from adet.tools.strings import get_index, validate_arg_format

logger = logging.getLogger(__name__)


# TODO: A lot of argument validation could be done at the class level
# instead of instance
class EquationBase(ABC):
    """
    Base Class for defining equations, including argument validation and organization,
    node variable creation and simple storage of the last arguments.

    Supports argument aliasing: allowing the residual function signature to use
    different names than the system-level variable names.
    """

    manual_units: ClassVar[tuple[str, ...]] = ()
    scaling_factor: tuple[float | None, ...] | None = None
    # EoS accessories
    input_pair: ClassVar[int] = 0
    output_quantities: ClassVar[tuple[str, ...]] = ()
    _eos: None | CasadiEos | cs.Function = None

    def __init__(self, custom_scaling_factor: list[float] | None = None):
        """
        Parameters
        ----------
        scaling_factor : list[float] | None
            Custom scaling factors for equations
        """

        # Read arguments from residual signature
        residual_args = getfullargspec(self.residual).args[1:]

        # Apply aliasing: use aliased names if provided, otherwise use original names
        self._arguments: tuple[str, ...] = self._read_and_validate_arguments(
            residual_args,
        )

        if custom_scaling_factor:
            self._scaling_factor = custom_scaling_factor
        else:
            self._scaling_factor = self.__class__.scaling_factor

    def __call__(self, *args):
        return self.residual(*args)

    @abstractmethod
    def residual(self, *args) -> Any | tuple[Any, ...]:
        """
        Expected format for argument is <node_state>_<var_type><index>
        where the index corresponds to the FlowNode in the order
        specified during the class definition. The indices are
        expected to be only one digit.
        """
        raise NotImplementedError

    @property
    def arguments(self):
        """Arguments, in the format of <node_state>_<var_type><index>"""
        return self._arguments

    @property
    def num_equations(self):
        raise NotImplementedError(
            'This method has been deprecated, the number of equations '
            'is free to change depending on the structure of the arguments'
        )

    @property
    def num_args(self):
        return len(self._arguments)

    def _read_and_validate_arguments(self, all_arguments: list[str]):
        # The 1 is removed because it is the self instance
        # Careful if residual is changed to a static method
        validated_arguments = []
        seen_indices = []

        for residual_arg in all_arguments:
            # Validate the SYSTEM variable name (the aliased one)
            validated_system_var, arg_index = self._validate_argument(residual_arg)
            seen_indices.append(arg_index)
            validated_arguments.append(validated_system_var)

        if min(seen_indices) > 0:
            raise ValueError(
                f'Minimum relative argument in `{self.__class__.__name__}` '
                f'is greater than 0, bad equation formatting'
            )

        expected_sequence = range(max(seen_indices) + 1)
        if set(seen_indices) != set(expected_sequence):
            raise ValueError(
                f'Non sequential nodes found {set(seen_indices)} '
                f'in {self.__class__.__name__}'
            )

        return tuple(validated_arguments)

    def _validate_argument(self, full_argument: str):
        try:
            arg_index = get_index(full_argument)
        except AttributeError:
            logger.info(f'No index found, assigning relative node 0 to {full_argument}')
            arg_index = 0
            full_argument += '0'

        if not validate_arg_format(full_argument, include_digits=True):
            logger.warning(
                f'Argument {full_argument[:-1]} in equation `{self.__class__.__name__}`'
                f' does not declare a state or has an unrecognized format, '
                f'assigning to `oth` state'
            )
            full_argument = 'oth_' + full_argument

        return full_argument, int(arg_index)

    def to_symbolic(self) -> sp.Expr | str:
        """
        Return a symbolic rendering of the equation

        - Temporarily convert numpy functions to sympy
        - Return class attributes as symbols
        """

        # Add a shape to the symbol class for
        # symbolic representation
        class ShapedSymbol(sp.Symbol):
            shape = (1,)

        class SymbolMaker:
            """
            This overwrites the self class to return the symbols
            self.ratio -> sp.Symbol('ratio')
            """

            def __getattr__(self, name):
                return ShapedSymbol(name)

        # This is so that any attribute that appears
        # in the equation is translated to a symbol
        # in theory this is not used anymore (for jax compatibility)
        dummy_self = SymbolMaker()
        # Recast it as an instance of Self
        dummy_self = cast(Self, dummy_self)

        res_func = self.residual

        # Build the residual function arguments as symbols
        symbolic_args = []
        for arg in self.arguments:
            symbolic_args.append(ShapedSymbol(arg))

        # Substitute numpy with sympy
        symbolic_res = override_operators(res_func, 'numpy', sp)

        try:
            return symbolic_res(*symbolic_args)
        except Exception:
            raise

    def __init_subclass__(cls) -> None:
        if bool(cls.output_quantities) != bool(cls.input_pair):
            raise ValueError(
                f'Please specify both input_pair and output_quantities in {cls}'
            )

        if cls.input_pair and not cls.manual_units:
            raise ValueError('Multi state equations requires manual unit inputs')

        return super().__init_subclass__()

    @property
    def eos(self):
        cls = self.__class__
        if cls._eos is None:
            raise AttributeError(f'Missing equation of state for {cls}')

        return cast(Callable[[Any, Any], tuple[Any, ...]], cls._eos)

    # TODO: Fix typing here for analytical/symbolic EoS
    @eos.setter
    def eos(self, eos: CasadiEos | cs.Function | Any):
        cls = self.__class__
        if cls._eos is not None:
            logger.debug(f'Overwriting EoS for {cls}')
        cls._eos = eos


class UniqueEquation(EquationBase):
    """
    Inherit this for all equations families which can be defined only
    once per component, either on one or two of the component nodes
    """

    def __init__(self, scaling_factor: list[float] | None = None):
        if self.__class__.__base__ == UniqueEquation:
            raise TypeError(f'Do not inherit directly from {self}')
        super().__init__(scaling_factor)


# fmt: off
class LossApplier(UniqueEquation): ...
class DeviationModel(UniqueEquation): ...
class IncidenceModel(UniqueEquation): ...
class CamberLineGeom(UniqueEquation): ...
class MeridionalGeom(UniqueEquation): ...
class MeridAreaBlockage(UniqueEquation): ...
# fmt: on


# class Equation:
#     def residual(self) -> float:
#         raise NotImplementedError
#
#     @property
#     def arguments(self):
#         args_type_hints = get_type_hints(self.residual, include_extras=True)
#
#         var_types = []
#         for hint in args_type_hints.values():
#             for valid_types in hint.__metadata__:
#                 if isinstance(valid_types, (ThermoVariables, OtherVariables)):
#                     var_types.append(valid_types)
#                 else:
#                     raise KeyError
#
#         return var_types
#
#
#
# class DummyEq(Equation):
#     def residual(
#         self,
#         s: ThermoHints().Enthalpy,
#         h: ThermoHints().Enthalpy,
#     ):
#         return s + h
#

if __name__ == '__main__':
    pass
    a = NodeHints(0).tot.Enthalpy
    print(a)
    # eq = DummyEq()
    #
    # print([arg.value.symbol for arg in eq.arguments])
