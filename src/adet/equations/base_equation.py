from inspect import getfullargspec
from abc import ABC, abstractmethod
import logging
import re
from typing import ClassVar, get_args, cast, Self
import ast
import inspect
import textwrap

import sympy as sp
import numpy as np

from adet.fluid.casadi_eos import CasadiEos
from adet.tools.strings import verify_string_pattern, get_arg_state
from adet.tools.context import override_operators, suppress_output
from adet.constants import NodeStatesNames


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
    input_pair: ClassVar[int] = 0
    output_quantities: ClassVar[tuple[str, ...]] = ()
    _eos: None | CasadiEos = None

    def __init__(self, scaling_factor: list[float] | None = None):
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

        # TODO: Move scaling factor to class attribute
        self.scaling_factor = scaling_factor

        self._num_equations: int | None = None

        # If the unit are not checked, make sure the user added units correclty
        if self.manual_units:
            eq_name = self.__class__.__name__
            if self.num_equations != len(self.manual_units):
                raise ValueError(
                    f'Mismatch in equation `{eq_name}` between manual '
                    f'units length ({len(self.manual_units)}) {self.manual_units} '
                    f'and number of equations ({self.num_equations})'
                )

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
        # Maybe add a setter for manually imposing num
        # equations?
        if not self._num_equations:
            try:
                # Avoid printing if fails
                # in particular CoolProp stuff
                with suppress_output():
                    self._num_equations = self._count_equations_arg_inj()
            except Exception:
                self._num_equations = self._count_equations_ast()

        return self._num_equations

    @property
    def num_args(self):
        return len(self._arguments)

    def _count_equations_arg_inj(self):
        """
        Count how many residual equations are contained
        in this residual formulation by argument
        injection
        """
        num_args = len(self.arguments)
        dummy_args = np.full((num_args, 1), np.nan)
        dummy_res = self.residual(*dummy_args)

        if hasattr(dummy_res, '__len__'):
            num_equations = len(dummy_res)
        else:
            num_equations = 1

        return num_equations

    def _count_equations_ast(self):
        """
        Count the number of residual equation using
        abstract syntax trees
        """
        method = self.residual.__func__

        try:
            src = inspect.getsource(method)
            # Remove class indentation
            src = textwrap.dedent(src)
        except (OSError, TypeError):
            raise RuntimeError(
                f'Could not get source code for residual function in'
                f' {self.__class__.__name__}'
            )

        tree = ast.parse(src)
        num_ret = 0  # number of returns

        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                v = node.value
                if isinstance(v, ast.Tuple):
                    num_ret += len(v.elts)
                else:
                    num_ret += 1

        return num_ret

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

        return tuple(validated_arguments)

    def _validate_argument(self, full_argument: str):
        # Updated pattern to allow digits in variable names (for intermediate states)
        # and multiple trailing digits for node indices
        # Example matches: stc_p0, stc_p_ss0_0, stc_rhomass_ps2_1
        VALID_STATES = get_args(NodeStatesNames)
        states_id_re = '|'.join(VALID_STATES)
        TEMPLATE_PATTERN = rf'^({states_id_re})_[a-zA-Z0-9_]*\d+$'

        # Check for trailing digits (node index)
        # We allow digits anywhere in the name (e.g., 'ss0' in 'stc_p_ss0_0')
        # Only the trailing digits are interpreted as the node index
        arg_index = re.findall(r'\d+$', full_argument)[0]

        if not arg_index:
            logger.info(f'No index found, assigning relative node 0 to {full_argument}')
            arg_index = '0'
            full_argument += arg_index

        if verify_string_pattern(full_argument, TEMPLATE_PATTERN) is False:
            logger.warning(
                f'Argument {full_argument[:-1]} in equation `{self.__class__.__name__}`'
                f' does not declare a state or has an unrecognized format, '
                f'assigning to `oth` state'
            )
            full_argument = 'oth_' + full_argument

        # Validate the node state
        var_state = get_arg_state(full_argument)
        if var_state not in VALID_STATES:
            raise ValueError(
                f'Unknown state for `{full_argument}`, valid states are:\n'
                f'{VALID_STATES}'
            )

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

        return cls._eos

    @eos.setter
    def eos(self, eos: CasadiEos):
        cls = self.__class__
        if cls._eos is not None:
            logger.warning(f'Overwriting EoS for {cls}')
        cls._eos = eos

    def __str__(self):
        return str(self.to_symbolic()) + ' = 0'


class UniqueEquation(EquationBase):
    """
    Inherit this for all equations families which can be defined only
    once per component, either on one or two of the component nodes
    """

    def __init__(self, scaling_factor: list[float] | None = None):
        if self.__class__.__base__ == UniqueEquation:
            raise TypeError(f'Do not inherit directly from {self}')
        super().__init__(scaling_factor)


class DeviationModel(UniqueEquation):
    pass


class IncidenceModel(UniqueEquation):
    pass


class CamberLineGeom(UniqueEquation):
    pass


class MeridAreaBlockage(UniqueEquation):
    pass


class MeridionalGeom(UniqueEquation):
    pass
