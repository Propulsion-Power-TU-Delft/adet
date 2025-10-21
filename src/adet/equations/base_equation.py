from inspect import getfullargspec
from abc import ABC, abstractmethod
import logging
import re
from typing import ClassVar, get_args, cast, Self, TypeAlias

import sympy as sp
import numpy as np

from adet.tools.strings import verify_string_pattern, get_arg_state
from adet.tools.context import override_operators
from adet.constants import NodeStatesNames

logger = logging.getLogger(__name__)

ScalingFactor: TypeAlias = float | tuple[float, ...] | None


class EquationBase(ABC):
    """
    Base Class for defining equations, including argument validation and organization,
    node variable creation and simple storage of the last arguments.
    """

    variable_units = ClassVar[dict[str, str]]

    def __init__(self, scaling_factor: ScalingFactor = None):
        self._arguments: tuple[str, ...] = self.read_and_validate_arguments(
            getfullargspec(self.residual).args[1:],
        )
        self.scaling_factor = scaling_factor

    def __call__(self, *args):
        return self.residual(*args)

    @abstractmethod
    def residual(self, *args):
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
        return self._count_equations()

    @property
    def num_args(self):
        return len(self._arguments)

    def _count_equations(self):
        """
        Count how many residual equations are contained
        in this residual formulation
        """
        num_args = len(self.arguments)
        dummy_args = np.full((num_args, 1), np.nan)
        dummy_res = self.residual(*dummy_args)

        if hasattr(dummy_res, '__len__'):
            num_equations = len(dummy_res)
        else:
            num_equations = 1

        return num_equations

    def read_and_validate_arguments(self, all_arguments: list[str]):
        """
        Retrieve all the arguments of the residual function
        """
        # The 1 is removed because it is the self instance
        # Careful if residual is changed to a static method
        self._kwarg_map = {}
        validated_arguments = []
        for arg in all_arguments:
            validated_arg = self._validate_argument(arg)
            validated_arguments.append(validated_arg)
            self._kwarg_map[validated_arg] = arg

        return tuple(validated_arguments)

    def _validate_argument(self, full_argument: str):
        TEMPLATE_PATTERN = r'^[a-z]{3}_[a-zA-Z_]*\d{1}$'

        # === indices check
        arg_digits = re.findall(r'\d', full_argument)  # All single digits
        arg_index = re.findall(r'\d$', full_argument)  # Final digit only

        if len(arg_digits) > 1:
            raise ValueError(
                f'Invalid argument {full_argument}: digits are reserved '
                f'for nodal positions, one digit at the end of the argument'
            )

        if not arg_index:
            logger.info(f'No index found, assigning to state 0 to {full_argument}')
            full_argument += '0'

        if verify_string_pattern(full_argument, TEMPLATE_PATTERN) is False:
            logger.warning(
                f'Argument {full_argument[:-1]} in equation `{self.__class__.__name__}`'
                f' does not declare a state or has an unrecognized format, '
                f'assigning to `oth` state'
            )
            full_argument = 'oth_' + full_argument

        # Validate the node state
        var_state = get_arg_state(full_argument)
        valid_states = get_args(NodeStatesNames)
        if var_state not in valid_states:
            raise ValueError(
                f'Unknown state for `{full_argument}`, valid states are:\n'
                f'{valid_states}'
            )

        return full_argument

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

    def __str__(self):
        return str(self.to_symbolic()) + ' = 0'
