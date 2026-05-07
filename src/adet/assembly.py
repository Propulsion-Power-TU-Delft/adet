"""
Module that contains the tools to define equations, build systems made
up of assembly of such residual equations and routines to generate reasonable
initial guesses based on the available thermodynamic, geometric and kinematic
data.
Sometimes the CasADi api is slightly cryptic, sorry.
"""

from adet.tools.loggers import setup_logger

import logging
import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import accumulate
from typing import Any, Callable, Iterable, Literal, Self, Sequence, Type, cast

import casadi as cs
import jax as jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from pint import Quantity, Unit
from pint.facets.plain import PlainQuantity

from adet.constants import AdetArray
from adet.equations.base_equation import EquationBase, EquationConfig
from adet.equations.variables import NodeVariables, ThermoVariables
from adet.equations.varspec import NodeStates, VarSpec
from adet.errors import ExistingEquationError
from adet.fluid.casadi_eos import CasadiEos
from adet.fluid.eos_factory import EosFactory
from adet.fluid.settings import EmptyFluidModel, ExternalFluidModel, FluidSettings
from adet.registries import (
    GuessRegistry,
    ScalarsRegistry,
    ScalingRegistry,
    VariableBoundsRegistry,
)
from adet.tools.context import override_operators
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.interpolation import resample_linear
from adet.tools.iter import ensure_tuple
from adet.tools.strings import get_index, rm_index

logger = logging.getLogger(__name__)

THERMO_PREFIXES = ('stc', 'tot', 'rlt')
THERMO_CONST_SUFFIX = '__THERMOCONSTR'  # I don't like this


_scale_reg = ScalingRegistry()
_scalars_reg = ScalarsRegistry()
_bounds_reg = VariableBoundsRegistry()
_guess_reg = GuessRegistry()


def get_units_string(var):
    return str(var.to_base_units().units)


def get_absolute_arg(int_map: dict[int, int], rel_arg: str):
    abs_idx = get_index(rel_arg)
    rel_idx = int_map[abs_idx]
    return rm_index(rel_arg) + str(rel_idx)


def get_relative_arg(int_map: dict[int, int], abs_arg: str):
    abs_idx = get_index(abs_arg)
    arg_map_inv = {v: k for k, v in int_map.items()}
    rel_idx = arg_map_inv[abs_idx]
    return rm_index(abs_arg) + str(rel_idx)


class SystemSharedData:
    """
    Shared data container accessible to all managers.
    This holds all the state that needs to be shared between different
    components of the system assembler.
    """

    def __init__(self):
        # Core structures
        self.equations: dict[EquationBase, tuple[int, ...]] = {}
        self._arg_maps: dict[EquationBase, dict[int, int]] = {}

        # Arguments
        self.decl_args: tuple[VarSpec, ...] = ()
        self.free_args: tuple[VarSpec, ...] = ()
        self.boun_cond: dict[VarSpec, AdetArray | PlainQuantity] = {}

        # Boundary conditions and constraints
        self.equalities: list[set[VarSpec]] = []
        self.spanwise_constants: set[VarSpec] = set()

        # Units and scaling
        self.equations_units: dict[EquationBase, tuple[str, ...]] = {}
        self.scaled: bool = False

        # Settings
        self.fluid_settings: FluidSettings = FluidSettings(EmptyFluidModel())
        self.num_span: int = 1

        # Thermo update arguments
        self.thermo_updt_args: list[VarSpec] = []

        # Build status
        self.built: bool = False


class EquationRegistry:
    """Manages equations, nodes, and their argument mappings"""

    def __init__(self, data: SystemSharedData):
        self.data = data

    def add_equation(
        self,
        equation: EquationBase,
        abs_position: int | Iterable[int],
    ):
        """Add an equation to the system in the specified position"""
        abs_position = ensure_tuple(abs_position)

        # Check that the provided position has the same length as equation arguments
        if len(equation.arg_nodes) != len(abs_position):
            raise ValueError(
                f'Detected indices in the definition of '
                f'`{equation.__class__.__name__}` {equation.arg_nodes} '
                f'is not equal to the length of the prescribed '
                f'nodal position {abs_position}'
            )

        # Check that an equation of the same type does not exist at the same location
        if self.contains(equation.__class__, abs_position):
            raise ExistingEquationError(
                f'Duplicate equation entry for {equation.__class__.__name__}'
                f' at position {abs_position}'
            )

        self.data.equations[equation] = tuple(abs_position)

        logger.debug(
            f'Added equation {equation.__class__.__name__} to system '
            f'in position {abs_position}'
        )

    def contains(
        self,
        eq_class: Type[EquationBase],
        abs_position: int | list[int] | tuple[int, ...],
    ):
        abs_position = ensure_tuple(abs_position)
        for eq_instance, eq_pos in self.data.equations.items():
            is_same_pos = set(eq_pos) == set(abs_position)
            if isinstance(eq_instance, eq_class) and is_same_pos:
                return True

        return False

    def remove_equation_type(self, *equation_class: Type[EquationBase]):
        """Remove all equations of the specified types"""
        cleaned_equations = self.data.equations.copy()
        for eq in self.data.equations:
            if isinstance(eq, equation_class):
                cleaned_equations.pop(eq)
        self.data.equations = cleaned_equations

    def remove_equation(
        self,
        equation: Type[EquationBase],
        abs_position: int | tuple[int, ...],
    ):
        """Remove equation from the system"""
        if isinstance(abs_position, int):
            abs_position = (abs_position,)

        to_remove = None
        for eq, pos in self.data.equations.items():
            if isinstance(eq, equation) and set(pos) == set(abs_position):
                to_remove = eq
                break

        if to_remove is None:
            logger.warning(f'No equation {equation} found to remove at {abs_position}')
        else:
            self.data.equations.pop(to_remove)

    def _read_decl_args(self) -> tuple[VarSpec, ...]:
        decl_args: list[VarSpec] = []
        for eq in self.data.equations:
            for arg in eq.arg_specs:
                abs_node = self.data._arg_maps[eq][arg.node]
                abs_arg = arg._at_node(abs_node)
                decl_args.append(abs_arg)

        return tuple(decl_args)

    def _build_argument_maps(
        self,
    ) -> dict[EquationBase, dict[int, int]]:
        arg_maps = {}
        logger.debug('Reading all the equation arguments...')

        for eq, eq_position in self.data.equations.items():
            arg_maps[eq] = dict(
                zip(
                    eq.arg_nodes,
                    eq_position,
                    strict=True,
                ),
            )

        return arg_maps


class ConstraintManager:
    """Handles all forms of constraints and boundary conditions"""

    def __init__(self, data: SystemSharedData):
        self.data = data

    def _check_arg_declaration(
        self,
        spec: VarSpec,
        caller_msg: str = '',
    ):
        caller = f'from {caller_msg}' if caller_msg else ''
        if spec not in self.data.decl_args:
            logger.warning(
                f'Imposing a condition {caller} on `{spec.full_symbol(True)}`'
                f', but it does not appear in any equation'
            )

    def add_boundary_conditions(
        self, bnd_cond: dict[VarSpec, AdetArray | PlainQuantity]
    ):
        """Add boundary conditions for a specific node"""
        for spec, val in bnd_cond.items():
            if isinstance(val, PlainQuantity):
                mag = val.to_base_units().magnitude
            else:
                mag = val

            mag = np.atleast_1d(mag)

            if len(mag) != self.data.num_span:
                if len(mag) == 1:
                    mag_valid = mag * np.ones(self.data.num_span)
                else:
                    raise ValueError(f'Length mismatch {spec}')

            self._check_arg_declaration(spec, 'boundary conditions')
            self.data.boun_cond[spec] = mag_valid

    def add_equalities(self, *equalities: tuple[VarSpec, ...]):
        """
        Each tuple represents a set of variables treated as equal by the system.
        Adding ('a', 'b', 'c') adds the equations: a-b=0, a-c=0
        """
        for args in equalities:
            if len(args) < 2:
                logger.warning(f'Single variable equality detected for {args}')
            if set(args) not in self.data.equalities:
                self.data.equalities.append(set(args))

    def add_spanwise_constants(self, *arguments: VarSpec):
        for arg in arguments:
            self.data.spanwise_constants.add(arg)

    def _validate_units(self) -> None:
        """Write the stored boundary conditions to the nodes"""
        logger.debug('Checking boundary conditions...')
        for spec, value in self.data.boun_cond.items():
            if isinstance(value, PlainQuantity):
                def_unit = Unit(spec.unit)
                if not value.units.is_compatible_with(def_unit):
                    raise ValueError(
                        f'{def_unit} is not compatible with {value.units}'
                        f' prescribed in the boundary conditions for {spec.symbol}'
                    )

                self.data.boun_cond[spec] = value.to_base_units()

    def check_constraints_effectiveness(self) -> None:
        """
        Check if the arguments used in equalities and spanwise_constants
        appear as declared arguments
        """
        # TODO: Could even check if they are free
        args_to_check: set[VarSpec] = set()

        for equality in self.data.equalities:
            args_to_check.update(equality)

        for arg in self.data.spanwise_constants:
            args_to_check.add(arg)

        for arg in args_to_check:
            self._check_arg_declaration(arg, 'equalities/spanwise constants')


class ArgumentResolver:
    """Resolves free arguments vs followers, handles EoS logic"""

    def __init__(self, data: SystemSharedData):
        self.data = data

    def identify_free_arguments(self) -> tuple[VarSpec, ...]:
        """
        Get the real thermodynamic and kinematic arguments needed to complete
        the different states of the node.
        """
        if isinstance(self.data.fluid_settings.model, EmptyFluidModel):
            return tuple(set(self.data.decl_args) - set(self.data.boun_cond))
        else:
            return tuple(self._get_effective_arguments())

    def _get_effective_arguments(self):
        """
        Get the thermo that act on thermodynamic state updates.
        e.g. if hmass, smass, p, T appear on the same state,
        only two variables are effective (pure substance + phase),
        while the other two are followers
        """
        # Non thermodynamic arguments
        nonthermo_args = [arg for arg in self.data.decl_args if not arg.state]

        # Get what variables will be used for state updates
        self.data.thermo_updt_args = []
        prescr_upd_vars = self.data.fluid_settings.update_variables

        first_node = min(arg.node for arg in self.data.decl_args)
        max_node = max(arg.node for arg in self.data.decl_args)

        for node in range(first_node, max_node + 1):
            for st in NodeStates:
                upd_args = [v._at_node(node)._with_state(st) for v in prescr_upd_vars]
                self.data.thermo_updt_args.extend(upd_args)

        return set(self.data.thermo_updt_args + nonthermo_args).difference(
            self.data.boun_cond
        )

    def get_discarded_thermo_args(self) -> list[VarSpec]:
        """
        Retrieve the discarded thermo arguments a.k.a. the ones
        that were declared in the equations but have become extractions
        from the equations of state
        """
        all_discarded = (
            set(self.data.decl_args)
            - set(self.data.thermo_updt_args)
            - set(self.data.free_args)
        )

        discarded = [arg for arg in all_discarded if arg.state]

        return discarded

    # ********************  TODO: Restore introspection START
    def make_arg_structure(self, arguments: Sequence[str]):
        """Detect the argument structure of a sequence of arguments"""
        arguments_struct = []
        for arg in arguments:
            num_span = 1 if arg in self.data.scalar_arguments else self.data.num_span
            branch = num_span * [0]
            arguments_struct.append(branch)

        return arguments_struct

    def get_args_coordinates(self):
        arg_struct = self.make_arg_structure(self.data.free_args)
        arg_lengths = [len(arg) for arg in arg_struct]
        acc_lengths = [0] + list(accumulate(arg_lengths[:-1]))
        return arg_lengths, acc_lengths

    def position_from_arg(self, arg_index: int) -> tuple[int, int]:
        """
        Return the start and end index of an argument, given its index
        in the free_args attribute
        """
        arg_lengths, acc_lengths = self.get_args_coordinates()
        start_index = acc_lengths[arg_index]
        arg_len = arg_lengths[arg_index]
        if arg_len == 1:
            end_index = start_index + 1
        else:
            end_index = start_index + arg_lengths[arg_index] - 1

        return (start_index, end_index)

    # ********************  TODO: Restore introspection END


class UnitScalingManager:
    """Handles unit checking and scaling factors"""

    def __init__(self, data: SystemSharedData):
        self.data = data

    def check_equations_units(self):
        self.data.equations_units = {}

        """Check units for all equations"""
        for eq in self.data.equations:
            self.data.equations_units[eq] = self._test_eq_units(eq)

        logger.debug('Units for the residual equations successfully verified')

    def _test_eq_units(self, equation: EquationBase) -> tuple[str, ...]:
        probe_args = []
        if equation.config.manual_units:
            return equation.config.manual_units

        for spec in equation.arg_specs:
            if spec.scalar:
                dummy_arg = Quantity(np.nan, spec.unit)
            else:
                dummy_arg = Quantity(self.data.num_span * [np.nan], spec.unit)

            probe_args.append(dummy_arg)

        residuals = cast(
            Quantity | tuple[Quantity, ...],
            equation.residual(*probe_args),
        )

        if not isinstance(residuals, tuple):
            residuals = (residuals,)

        return tuple(res.to_base_units().units.__str__() for res in residuals)

    def get_free_args_scaling(self):
        """Build multi-span scaling factors for free arguments"""
        return self._argument_scaling_helper(self.data.free_args)

    def get_constraints_scaling(self):
        """Build multi-span scaling factors for constraints"""
        bc_specs = list(self.data.boun_cond.keys())
        return self._argument_scaling_helper(bc_specs)

    def _argument_scaling_helper(self, arguments: Sequence[VarSpec]) -> list[float]:
        """Returns the arguments scales for a sequence of system arguments"""
        if not self.data.scaled:
            scales = len(arguments) * [1.0]
        else:
            scales = [_scale_reg.get(spec.unit) for spec in arguments]

        return scales

    def get_arguments_bounds(
        self, custom_bounds: dict[VarSpec, tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """The custom bounds are to be provided dimensionally"""
        bounds = []
        for spec in self.data.free_args:
            if spec in custom_bounds:
                lower_bound, upper_bound = custom_bounds[spec]
            elif spec.Plain in custom_bounds:
                lower_bound, upper_bound = custom_bounds[spec]
            else:
                if not spec.bounds:
                    lower_bound = -1e20
                    upper_bound = 1e20
                else:
                    lower_bound, upper_bound = spec.bounds

            scaling = _scale_reg.get(spec.unit)
            bounds.append(
                (
                    lower_bound / scaling,
                    upper_bound / scaling,
                )
            )

        return bounds

    def get_equations_scaling(self):
        """Build multi-span scaling factors for equations"""
        eq_scales: list[float] = []

        num_equations = jax.tree.leaves(self.data.equations_units).__len__()
        if not self.data.scaled:
            scales = np.ones(num_equations)
        else:
            for eq, units in self.data.equations_units.items():
                if eq._scaling_factor:
                    logger.debug(
                        f'Custom scaling factor found for '
                        f'{eq.__class__.__name__}, '
                        f'{eq._scaling_factor}'
                    )
                    for scl, u in zip(eq._scaling_factor, units):
                        if scl is None:
                            eq_scales.append(_scale_reg.get(u))
                        else:
                            eq_scales.append(scl)
                else:
                    eq_scales += [_scale_reg.get(u) for u in units]

            scales = np.array(eq_scales)

        return scales


class SystemAssembler(ABC):
    """
    Class for assembling a system of equations, gathering its arguments
    and returning the residuals dispatched to the equations it is made
    out of.

    This class uses a composition-based architecture with specialized
    managers handling different responsibilities.
    """

    def __init__(self, num_span: int) -> None:
        # Initialize shared context
        self.data = SystemSharedData()
        self.num_span = num_span

        # Initialize managers
        self._scaling_manager = UnitScalingManager(self.data)
        self._equation_registry = EquationRegistry(self.data)
        self._argument_resolver = ArgumentResolver(self.data)
        self._constraint_manager = ConstraintManager(self.data)

    @property
    def num_span(self):
        return self.data.num_span

    @num_span.setter
    def num_span(self, num_span):
        if num_span % 2 == 0:
            raise ValueError(
                f'Provide an odd number of spanwise station, currently = {num_span}'
            )
        else:
            self.data.num_span = num_span

    @property
    def fluid_settings(self):
        return self.data.fluid_settings

    @fluid_settings.setter
    def fluid_settings(self, settings: FluidSettings) -> None:
        self.data.fluid_settings = settings

    @property
    def num_equations(self):
        return jax.tree.leaves(self.data.equations_units).__len__()

    @property
    def first_node(self) -> int:
        return min(arg.node for arg in self.data.decl_args)

    @property
    def last_node(self) -> int:
        return max(arg.node for arg in self.data.decl_args)

    def reset(self) -> None:
        old_settings = self.data.fluid_settings
        self.__init__(self.data.num_span)
        self.fluid_settings = old_settings

    def copy(self) -> Self:
        new_instance = self.__class__(self.data.num_span)
        new_instance.from_dict(self.to_dict())
        return new_instance

    def to_dict(self):
        """Serialize the context to a dictionary"""
        FIELDS_TO_SAVE = [
            'equations',
            'boundary_conditions',
            'fluid_settings',
            'global_constraints',
            'equalities',
        ]

        out_dict = {}
        for field in FIELDS_TO_SAVE:
            attr = getattr(self.data, field)

            out_dict[field] = deepcopy(attr)

        return out_dict

    def from_dict(self, data_dict):
        """Load the context from a dictionary"""
        for attr, value in data_dict.items():
            setattr(self.data, attr, value)

    def add_boundary_conditions(self, bnd_cond: dict):
        """Delegate to constraint manager"""
        self._constraint_manager.add_boundary_conditions(bnd_cond)

    def add_equation(
        self,
        equation: EquationBase,
        nodal_position: int | list[int] | tuple[int, ...],
    ):
        self._equation_registry.add_equation(equation, nodal_position)

    def remove_equation_type(self, *eq_child_class: Type[EquationBase]):
        """Delegate to equation registry"""
        self._equation_registry.remove_equation_type(*eq_child_class)

    def contains_eq(
        self,
        eq_class: Type[EquationBase],
        abs_position: int | list[int] | tuple[int, ...],
    ):
        return self._equation_registry.contains(eq_class, abs_position)

    def remove_equation(
        self,
        equation_class: Type[EquationBase],
        nodal_position: int | tuple[int, ...],
    ):
        """
        Remove equation from the system

        Parameters
        ----------
        equation_class:
            Class of the equation instance to be removed
        nodal_position:
            Position where the equation is located
        """
        self._equation_registry.remove_equation(equation_class, nodal_position)

    def add_equalities(self, *equalities: tuple[VarSpec, ...]):
        """
        Each tuple in the list represent a set of variables that are treated
        as equal by the system. This is achieved by adding trivial residual
        equation to the system.
        Adding ('a', 'b', 'c') adds the equations
        a - b = 0
        a - c = 0
        b - c = 0
        """
        self._constraint_manager.add_equalities(*equalities)

    def add_spanwise_constants(self, *arguments: VarSpec):
        """
        Add arguments that are constant along the span
        """
        self._constraint_manager.add_spanwise_constants(*arguments)

    def build(self, scaled: bool):
        """
        Build the system of equations:

        - Create the nodes
        - Write b.c. to the nodes
        - Identify the arguments of the nonlinear problem
        - Create gather matrices to dispatch the arguments
        - Build compiled residual functions
        - Check well posedness and unit consistency of equations
        """
        logger.debug('Building the system of equations...')

        if scaled:
            logger.info('Using automatic scaling of variables and equations')
            self.data.scaled = True

        # Delegate to managers
        self.data._arg_maps = self._equation_registry._build_argument_maps()
        self.data.decl_args = self._equation_registry._read_decl_args()

        self._constraint_manager._validate_units()
        self._constraint_manager.check_constraints_effectiveness()

        # Arguments manipulation
        self.data.free_args = self._argument_resolver.identify_free_arguments()

        # Validity checks and scaling
        self._scaling_manager.check_equations_units()

        self.data.built = True
        logger.info('System assembled successfully')

    def _check_built(self) -> None:
        """Check that the system is flagged as built"""
        if not self.data.built:
            raise RuntimeError(
                'The system is not built or failed to do so, build it `build()`'
            )

    def get_arg_position(self, arg_index: int):
        return self._argument_resolver.position_from_arg(arg_index)

    @property
    def equations_indices(self):
        """
        Return a list of each single equations,
        splitting also multi-residual objects
        Needed to identify problematic equations
        """
        eq_identifiers = []
        for eq, pos in self.data.equations.items():
            eq_identifiers += [
                f'{eq.__class__.__name__} EQ#{idx} @NODES{pos}'
                for idx in range(eq.num_equations)
            ]

        return eq_identifiers

    @property
    def free_args_scaling(self):
        """Build multi-span scaling factors for arguments"""
        return self._scaling_manager.get_free_args_scaling()

    @property
    def constraints_scaling(self):
        """Build multi-span scaling factors for constraints"""
        return self._scaling_manager.get_constraints_scaling()

    @property
    def equations_scaling(self):
        """Build multi-span scaling factors for equations"""
        return self._scaling_manager.get_equations_scaling()

    @abstractmethod
    def make_residual_function(self):
        self._check_built()

    def get_arguments_bounds(self, custom_bounds={}):
        self._scaling_manager.get_arguments_bounds(custom_bounds)

    def get_scaled_constraints(self) -> list[NDArray]:
        return jax.tree.map(
            lambda x, y: x / y,
            list(self.data.boun_cond.values()),
            self.constraints_scaling,
        )

    def get_scaled_guess(
        self,
        manual_values: dict[VarSpec, AdetArray] = {},
        fallback: float | None = None,
    ) -> list[NDArray]:
        """Generate initial guesses for free arguments"""
        guesses = []

        for spec in self.data.free_args:
            # If there is a manual value, overwrite the guess registry

            if spec in manual_values:
                guess_value = manual_values[spec]
                logger.debug(f'Using manual value {guess_value} for {spec}')
            elif spec.Plain in manual_values:
                guess_value = manual_values[spec.Plain]
                logger.debug(f'Using manual value {guess_value} for {spec}')
            # If a guess is available in the registry
            elif spec.guess:
                guess_value = spec.guess

                # Vary the total and static values, avoid singularities
                if spec.state == NodeStates.STATIC:
                    guess_value *= 0.95
                elif spec.state in (NodeStates.TOTAL, NodeStates.RELTOT):
                    guess_value *= 1.05

            # If there is no guess and no manual value
            else:
                # If the user has defined a fallback, use that
                if fallback:
                    guess_value = fallback
                # Otherwise ask for user input
                else:
                    input_msg = f'INPUT >>> DIMENSIONAL guess for {spec} [1.0] = '
                    guess_value = float(input(input_msg) or 1.0)
                    manual_values[spec.Plain] = guess_value

            guess_value = np.atleast_1d(guess_value)

            if max(guess_value.shape) != self.data.num_span:
                logger.debug(
                    f'Length mismatch in guess for {spec}, using linear resampling'
                )
                # This simply repeats the single value when 1 -> N
                guess_value = resample_linear(
                    guess_value.flatten(),
                    self.data.num_span,
                )

                if spec.scalar:
                    guess_value = np.array([guess_value[0]])

            # Scale
            scaling_factor = _scale_reg.get(spec.unit)
            guess_value_scaled = guess_value / scaling_factor
            guesses.append(guess_value_scaled)

        return guesses


class CasadiSystem(SystemAssembler):
    def __init__(self, num_span: int = 1, *, scale_suffix: str = '__SCL') -> None:
        super().__init__(num_span)
        self.scale_suffix = scale_suffix

    def _reset_symbols(self):
        logger.debug('Resetting all CasADi symbolics...')
        self.const_sym: dict[VarSpec, cs.MX] = {}
        self.free_args_sym: dict[VarSpec, cs.MX] = {}

        self.scales_const_sym: dict[VarSpec, cs.MX] = {}
        self.scales_free_args_sym: dict[VarSpec, cs.MX] = {}

        self._eq_scales_sym: list[cs.MX] = []
        self.residual_expr: list[cs.MX]

    def build(self, scaled: bool = True):
        super().build(scaled)
        logger.info('Building CasADi backend...')
        self._reset_symbols()
        self._build_base_symbols()
        self._build_products()
        self._build_residual_expressions()

    def _create_symbols(
        self, arg_specs: Sequence[VarSpec], num_span: int, scale_suffix: str
    ) -> tuple[
        dict[VarSpec, cs.MX],
        dict[VarSpec, cs.MX],
    ]:
        """Helper to create symbols and their scaled versions."""
        symbols = {}
        scales = {}

        for spec in arg_specs:
            symbol_name = spec.full_symbol(index=True)
            symbol_length = 1 if spec.scalar else num_span
            symbols[spec] = cs.MX.sym(symbol_name, symbol_length)
            scales[spec] = cs.MX.sym(symbol_name + scale_suffix, 1)

        return symbols, scales

    def _build_base_symbols(self):
        """
        Convert the free arguments and constraints of the system to
        lists of symbols, together with a symbolic representation of
        their scaling values
        """

        # Create symbols
        self.free_args_sym, self.scales_free_args_sym = self._create_symbols(
            self.data.free_args,
            self.num_span,
            self.scale_suffix,
        )
        bc_specs = list(self.data.boun_cond)
        self.const_sym, self.scales_const_sym = self._create_symbols(
            bc_specs,
            self.num_span,
            self.scale_suffix,
        )

    def _build_products(self):
        """
        Loop through the system's equations giving as arguments symbolic
        MX representation of each argument, mapped from the relative equation
        indices to the absolute system indices.
        """
        # === Build the product of each symbolic variable for their scaling value
        # 1. For free arguments
        free_args_products = {
            spec: self.free_args_sym[spec] * self.scales_free_args_sym[spec]
            for spec in self.data.free_args
        }

        constraints_products = {
            spec: self.const_sym[spec] * self.scales_const_sym[spec]
            for spec in self.data.boun_cond
        }

        all_args_products = {**free_args_products, **constraints_products}

        # 3. Build equation of state symbols
        casadi_eos_symbols = self._build_equations_of_state(all_args_products)

        self._all_symbols = {**all_args_products, **casadi_eos_symbols}

    def _build_equations_of_state(
        self, all_args_products: dict[VarSpec, cs.MX]
    ) -> dict[VarSpec, cs.MX]:
        fl_model = self.fluid_settings.model

        # TODO: Fix typing here for analytical eos
        self._eos_callbacks: dict[
            int, dict[NodeStates, cs.Function | CasadiEos | Any]
        ] = {
            n_idx: dict.fromkeys(
                NodeStates,
                None,
            )
            for n_idx in range(self.first_node, self.last_node + 1)
        }

        self._eos_factory = EosFactory(fl_model)

        # Add inter-node eos
        for eq in self.data.equations:
            eq_conf = eq.config
            if eq_conf.input_pair:
                eq.eos = self._eos_factory.make_eos(
                    eq_conf.input_pair,
                    eq_conf.out_properties,
                    self.num_span,
                    f'multi_{eq.__class__.__name__}',
                )

        # TODO: Refactor this, messy
        discarded_vars = self._argument_resolver.get_discarded_thermo_args()
        out_syms: dict[VarSpec, cs.MX] = {}

        sorted_discarded: dict[
            int,
            dict[NodeStates, list[VarSpec]],
        ] = {}
        # build output properties
        for spec in discarded_vars:
            if spec.node not in sorted_discarded:
                sorted_discarded[spec.node] = dict.fromkeys(NodeStates, [])
            if spec.state:
                sorted_discarded[spec.node][spec.state].append(spec)

        for node_idx in range(self.first_node, self.last_node + 1):
            for state, out_specs in sorted_discarded[node_idx].items():
                pair_id = self.data.fluid_settings.input_pair

                eos_caller = self._eos_factory.make_eos(
                    pair_id,
                    out_specs,
                    self.num_span,
                    f'nodeCb_{state.value}N{node_idx}',
                )

                # This is to keep references alive
                self._eos_callbacks[node_idx][state] = eos_caller

                upd_specs = [
                    spec._with_state(state)._at_node(node_idx)
                    for spec in self.data.fluid_settings.update_variables
                ]

                # Symbolic representation of the input pair properties
                symbolic_pair = [all_args_products[spec] for spec in upd_specs]

                # Symbolic representation of the output properties
                out_props_syms = eos_caller(*symbolic_pair)

                if not isinstance(out_props_syms, tuple):
                    out_props_syms = (out_props_syms,)

                # Make a dictionary of symbols that are ouputs from the eos callbacks
                for pr_spec, pr_sym in zip(out_specs, out_props_syms):
                    if pr_spec in self.data.boun_cond:
                        pr_spec = pr_spec._with_symbol(
                            pr_spec.symbol + THERMO_CONST_SUFFIX
                        )
                    out_syms[pr_spec] = pr_sym

        return out_syms

    def _build_equalities_expr(self) -> list[cs.MX]:
        """
        Build expressions for constant quantities, across nodes,
        that can be part of a single or multiple components
        """
        equalities_expressions = []
        for equal_args in self.data.equalities:
            eq_args_ls = list(equal_args)
            arg_couples = [(eq_args_ls[0], arg) for arg in eq_args_ls[1:]]
            for arg_tuple in arg_couples:
                # If both argument do not appear in the equations, skip to next couple
                # if one of them is unused by other eqns. it is useless to add it
                if not set(arg_tuple).issubset(self._all_symbols):
                    continue

                sym0 = self._all_symbols[arg_tuple[0]]
                sym1 = self._all_symbols[arg_tuple[1]]

                # NOTE: We don't care about scaling, they
                # are just identities, either on scaled or unscaled vars
                equalities_expressions.append(sym1 - sym0)

        return equalities_expressions

    def _build_spanwise_constants(self) -> list[cs.MX]:
        """
        Build expressions for imposing variables to be constant
        along the span of a certain station
        """
        spanwise_expressions = []
        for arg in self.data.spanwise_constants:
            if arg not in self._all_symbols:
                continue

            arg_sym = self._all_symbols[arg]

            if max(arg_sym.shape) == 1:
                continue
            else:
                expression = arg_sym[1:] - arg_sym[:-1]
                spanwise_expressions.append(expression)

        return spanwise_expressions

    def _build_thermo_constraints(self) -> list[cs.MX]:
        """
        Build equations for imposed constraint that do not
        end up being part of the equations of state update
        variables
        """
        constraints_eqs = []
        for spec in self.data.boun_cond:
            if not spec.state:
                continue

            if spec not in self.data.thermo_updt_args:
                constraints_eqs.append(
                    self._all_symbols[spec]
                    - self._all_symbols[
                        spec._with_symbol(spec.symbol + THERMO_CONST_SUFFIX)
                    ]
                )

        return constraints_eqs

    def _build_residual_expressions(self):
        # Build scaling symbols for all equations
        self._eq_scales_sym = [
            cs.MX.sym(f'eq{idx}{self.scale_suffix}', 1)
            for idx in range(self.num_equations)
        ]

        # Build and concatenate residual equations
        logger.info('Building residual equation symbolics (this may take a while)...')

        residuals: list[Any | tuple[Any, ...]] = []
        for eq in self.data.equations:
            args = []
            for spec in eq.arg_specs:
                arg_map = self.data._arg_maps[eq]
                abs_arg = spec._at_node(arg_map[spec.node])
                args.append(self._all_symbols[abs_arg])

            # NOTE: No need to override the operators for now,
            # just use numpy operations compatible with casadi

            # overridden_eq = override_operators(eq.residual, 'numpy', cs)
            overridden_eq = eq.residual
            res_syms = overridden_eq(*args)
            if eq.config.manual_units:
                self._manual_units_check(eq, res_syms)

            residuals.append(res_syms)

        # Divide each residual expression by its scaling symbol
        self.residual_expr = list(
            map(
                lambda X, Y: X / Y,
                jax.tree.leaves(residuals),
                self._eq_scales_sym,
            )
        )

        self.residual_expr += self._build_equalities_expr()
        self.residual_expr += self._build_spanwise_constants()
        self.residual_expr += self._build_thermo_constraints()

        num_vars = max(cs.vertcat(*self.free_args_sym.values()).shape)
        num_residuals = max(cs.vertcat(*self.residual_expr).shape)
        logger.info(
            f'System info: {num_residuals} total equations, {num_vars} total variables'
        )

        if num_vars != num_residuals:
            answer = input(
                f'*** WARNING: Mismatch in number of equations {num_residuals}'
                f' and variables {num_vars}, continue anyway? [y/n] '
            )
            if answer not in ('y', 'Y', 'yes'):
                sys.exit()

    def get_residual_indices(self):
        idx = 0
        res_indices = {}
        for r_expr in self.residual_expr:
            n_eqs = max(r_expr.shape)

            if n_eqs == 1:
                final_idx = idx
                res_indices[r_expr] = final_idx
            else:
                final_idx = idx + n_eqs - 1
                res_indices[r_expr] = (idx, final_idx)

            idx = final_idx + 1

        return res_indices

    def _manual_units_check(
        self,
        equation: EquationBase,
        residual_symbols: cs.MX | tuple[cs.MX, ...] | list[cs.MX],
    ):
        """
        Check the matching between residual and manual units
        """

        if not isinstance(residual_symbols, (tuple, list)):
            residual_symbols = (residual_symbols,)
        eq_name = equation.__class__.__name__
        eq_conf = equation.config
        if len(eq_conf.manual_units) != len(residual_symbols):
            raise ValueError(
                f'Mismatch in eq_conf `{eq_name}` between manual '
                f'units length ({len(eq_conf.manual_units)}) '
                f'{eq_conf.manual_units} and number of eq_confs '
                f'({len(residual_symbols)})'
            )

    def make_residual_function(self):
        """
        Build a function object for the full residual, with the scaling
        factors as arguments, and create an MX expression for the full
        function with the actual scaling values assigned
        """
        super().make_residual_function()  # This only checks if built for now

        FULL_ARGUMENTS = [
            list(self.free_args_sym.values()),
            list(self.scales_free_args_sym.values()),
            list(self.const_sym.values()),
            list(self.scales_const_sym.values()),
            self._eq_scales_sym,
        ]

        full_arguments_cat = [cs.vertcat(*args) for args in FULL_ARGUMENTS]

        # Symbolic free arguments and constraints
        free_args = full_arguments_cat[0]
        constraints = full_arguments_cat[2]

        res_full_func = cs.Function(
            'res_func', full_arguments_cat, [cs.vertcat(*self.residual_expr)]
        )

        # Numerical values for scaling of variables and equations
        free_args_scaling_values = self.free_args_scaling
        const_scaling_values = self.constraints_scaling
        eq_scaling_values = self.equations_scaling

        res_part_expr = res_full_func(
            free_args,
            free_args_scaling_values,
            constraints,
            const_scaling_values,
            eq_scaling_values,
        )

        return cs.Function(
            'res_func',
            [free_args, constraints],
            [res_part_expr],
            ['free_args', 'constraints'],
            ['residuals'],
        )

    def make_rootfinder(
        self,
        root_method: Literal['newton', 'ipopt', 'lstsq', 'kinsol'],
        opts={},
    ) -> Callable[[AdetArray, AdetArray], cs.DM]:
        """
        Create a rootfinder callable object that takes as a first input the
        initial guess and as second input the values of the constraints and
        returns the solution.
        """
        res_func = self.make_residual_function()

        args_sym = list(self.free_args_sym.values())
        cons_sym = list(self.const_sym.values())

        free_args_symbols = cs.vertcat(*args_sym)
        constraints_symbols = cs.vertcat(*cons_sym)

        res_expr_partial = res_func(
            free_args_symbols,
            constraints_symbols,
        )

        if root_method == 'lstsq':
            func_spec = {
                'f': cs.norm_fro(res_expr_partial),
            }
        else:
            func_spec = {
                'g': res_expr_partial,
            }

        # CasADi boilerplate
        rootfind_problem = {
            'x': free_args_symbols,
            'p': constraints_symbols,
            **func_spec,
        }

        # TODO: remove hardcoded options
        if root_method == 'newton':
            # Newton-Raphson solver -> Fast but unstable w/o good guess
            rootfinder = cs.rootfinder(
                'newton_rootfinder',
                'newton',
                rootfind_problem,
                {
                    'error_on_fail': True,
                    'print_iteration': False,
                    **opts,
                },
            )

        elif root_method == 'ipopt' or root_method == 'lstsq':
            # IPOPT solver
            rootfinder = cs.nlpsol(
                'ipopt_rootfinder',
                'ipopt',
                rootfind_problem,
                {
                    'error_on_fail': True,
                    # Reasonable defaults for IPOPT, overwritten by user
                    'ipopt.print_level': 3,
                    'ipopt.max_iter': 6000,
                    'ipopt.max_wall_time': 60,
                    'ipopt.tol': 1e-8,
                    'ipopt.acceptable_constr_viol_tol': 1e-10,
                    'ipopt.bound_frac': 0.1,  # Relative initial push < 0.5
                    'ipopt.mu_init': 0.3,  # Initial barrier param
                    'ipopt.mu_strategy': 'adaptive',
                    'ipopt.linear_solver': 'spral',
                    # Lower = stricter restoration (def = 100 * tol)
                    'ipopt.resto_failure_feasibility_threshold': 1e-7,
                    'ipopt.expect_infeasible_problem': 'yes',
                    # 'ipopt.hessian_approximation': 'limited-memory',  # Less updates
                    **opts,
                },
            )
        elif root_method == 'kinsol':
            # kinsol rootfinder
            rootfinder = cs.rootfinder(
                'kinsol_rootfinder',
                'kinsol',
                rootfind_problem,
                {
                    'error_on_fail': True,
                    **opts,
                },
            )
        return rootfinder

    def get_arguments_bounds(
        self,
        custom_bounds: dict[VarSpec, tuple[float, float]] = {},
    ):
        bounds_by_arg = self._scaling_manager.get_arguments_bounds(custom_bounds)
        lbx = []
        ubx = []

        for arg, scales in zip(self.data.free_args, bounds_by_arg):
            arg_size = max(self._all_symbols[arg].shape)
            lbx += arg_size * [scales[0]]
            ubx += arg_size * [scales[1]]

        return cs.vertcat(*lbx), cs.vertcat(*ubx)

    # def write_solution_to_nodes(self, solution_values: NDArray):
    #     solution_dict = super().write_solution_to_nodes(solution_values)
    #
    #     for node_idx, cb_specs in self._eos_callbacks.items():
    #         for state_id, eos_cb in cb_specs.items():
    #             eos_name: str = eos_cb.name()
    #             if not eos_name.startswith('nodeCb'):
    #                 continue
    #
    #             # Full CoolProp
    #             input_args = [INVERSE_CP_NAMES_MAP[arg] for arg in eos_cb.name_in()]
    #             out_props = eos_cb.name_out()
    #
    #             state_obj = self.nodes[node_idx].fetch_state(state_id)
    #
    #             inputs = [
    #                 state_obj.get(arg).to_base_units().magnitude for arg in input_args
    #             ]
    #             output_values = eos_cb(*inputs)
    #
    #             if not output_values:
    #                 raise ValueError(f'{eos_cb} returned no output values')
    #
    #             thermo_dict = {}
    #             for prop, val in zip(out_props, output_values):
    #                 thermo_dict[f'{state_id}_{prop}{node_idx}'] = (
    #                     val.toarray().flatten()
    #                 )
    #
    #             self.nodes[node_idx].write_to_node(thermo_dict, False)
    #
    #             solution_dict.update(thermo_dict)
    #
    #     return solution_dict

    def to_jax(self):
        """
        Convert system to casadi
        """
        jax_sys = JaxSystem(self.num_span)
        jax_sys.from_dict(self.to_dict())
        return jax_sys


class JaxSystem(SystemAssembler):
    """
    Build a jax compatible assembled system, with stacked arguments input shapes,
    no need for flat vectors.
    """

    def __init__(self, num_span: int) -> None:
        logger.warning('*** jax backend is not maintained, proceed with caution ***')
        input('I understand the jax interface is not working, press enter to proceed')

        super().__init__(num_span)

    def _build_stack_composer(self):
        """
        Build a function that merges the stack of free arguments
        of shape (# free args, # span stations) to a stack of constraints
        of shape shape (# constraints, # span station).
        The full stack is then fed to a residual function.
        """

        all_args = self.data.decl_args

        free_args = self.data.free_args
        declared_constraints = self.data.boun_cond

        num_free = len(free_args)
        num_const = len(declared_constraints)

        # Get the position of arguments and knowns in the
        # overall stack of arguments
        free_stack_pos = jnp.array([all_args.index(arg) for arg in free_args])
        const_stack_pos = jnp.array(
            [all_args.index(knw) for knw in declared_constraints if knw in all_args]
        )

        def stack_composer(arguments_stack, constraints_stack):
            full_stack = jnp.zeros(num_free + num_const)
            full_stack = full_stack.at[free_stack_pos].set(arguments_stack)
            full_stack = full_stack.at[const_stack_pos].set(constraints_stack)

            return full_stack

        stack_composer = jax.vmap(stack_composer, (1, 1), 1)

        return stack_composer

    def _create_equation_stack(self):
        """
        Create a stack of equations, overriding numpy
        with jax.numpy and making them return an
        array of residuals instead of tuples
        """

        def make_return_array(res_func):
            def wrapped_func(*args):
                return jnp.array(res_func(*args))

            return wrapped_func

        return [
            make_return_array(
                override_operators(eq.residual, 'numpy', jnp),
            )
            for eq in self.data.equations
        ]

    def _get_residual_positions(self):
        """
        Get where residual equations are placed in the stack
        of residuals
        """
        curr_index = 0
        residual_indices = []
        for eq in self.data.equations:
            num_eqs = eq.num_equations
            residual_indices.append(tuple(curr_index + j for j in range(num_eqs)))
            curr_index += num_eqs

        return residual_indices

    def _get_equation_lines(self, residuals_name: str):
        """
        Make the equation lines, each writing the residuals
        array at the correct residual indices
        """
        residual_indices = self._get_residual_positions()
        eq_lines = []
        for idx, eq in enumerate(self.data.equations):
            int_map = self.data._arg_maps[eq]

            mapped_args = [get_absolute_arg(int_map, arg) for arg in eq.arguments]

            eq_lines.append(
                f'{residuals_name} = {residuals_name}.at'
                f'[jnp.array( {residual_indices[idx]} )].'
                f'set( eq{idx}( {", ".join(mapped_args)} ) )'
            )

        return eq_lines

    def _generate_residual_code(
        self,
        func_name: str,
        residuals_name: str,
    ):
        """
        Generate residuals code that takes as input a list of equations objects
        and all arguments declared by all equations in the stack
        """
        DIVIDER = ', '
        eq_name_unpacker = (
            DIVIDER.join(f'eq{idx}' for idx, _ in enumerate(self.data.equations))
            + DIVIDER
        )
        eq_lines = self._get_equation_lines(residuals_name)
        eq_stack = '\n    '.join(eq_lines)

        codegen = f"""
def {func_name}(equations, {', '.join(self.data.decl_args)}):
    {residuals_name} = jnp.zeros({(self.num_equations, self.num_span)})

    {eq_name_unpacker} = equations
    {eq_stack}

    return {residuals_name}"""

        return codegen

    def _make_arg_dispatcher(self):
        """
        Create a dispatcher function
        """
        FUNC_NAME = 'unified_func'
        RESIDUALS_NAME = 'residuals'

        eqs = self._create_equation_stack()
        codegen = self._generate_residual_code(FUNC_NAME, RESIDUALS_NAME)

        # Create isolated namespace
        nspace: dict[str, Callable] = {}
        # Execute the custom code in this space
        exec(codegen, None, nspace)
        # Get the callable
        res_dispatcher = nspace[FUNC_NAME]

        # Build a stack
        stack_composer = self._build_stack_composer()

        def argument_dispatcher(arg_stack, knowns_stack):
            full_stack = stack_composer(arg_stack, knowns_stack)
            return res_dispatcher(eqs, *full_stack)

        return argument_dispatcher

    def make_residual_function(self):
        """
        Build compiled version of residual, jacobian and hessian
        functions for the assembled system
        """

        super().make_residual_function()
        num_args = len(self.data.free_args)
        num_const = len(self.data.boun_cond)

        argument_scaler = self.free_args_scaling
        constraint_scaler = self.constraints_scaling

        eq_scaling_stack = self.equations_scaling

        compute_residuals = self._make_arg_dispatcher()

        def residual_function(args_stack, const_stack):
            arg_dimensional = args_stack * argument_scaler
            const_dimensional = const_stack * constraint_scaler

            # Do I need the reshape
            arg_dimensional.reshape(num_args, -1)
            const_dimensional.reshape(num_const, -1)

            res = compute_residuals(arg_dimensional, const_dimensional)

            return res / eq_scaling_stack

        return residual_function

    def to_casadi(self):
        """Convert system to casadi backend"""
        cas_sys = CasadiSystem()
        cas_sys.from_dict(self.to_dict())
        return cas_sys


if __name__ == '__main__':
    nls = CasadiSystem(5)

    setup_logger(logger)
    n0 = NodeVariables(0)
    n1 = NodeVariables(1)

    import CoolProp as cp

    thrm = ThermoVariables()

    class TestEquation(EquationBase):
        config = EquationConfig(
            manual_units=('J / kg', 'J / kg / K'),
            input_pair=cp.PT_INPUTS,
            out_properties=(thrm.Entropy, thrm.SpeedSound, thrm.Viscosity),
        )

        def residual(
            self,
            h0: n0.tot.Enthalpy.Hint,
            h1: n1.tot.Enthalpy.Hint,
            p: n0.tot.Pressure.Hint,
            t: n1.tot.Temperature.Hint,
            s: n1.stc.Entropy.Hint,
        ):
            r1 = h0 - h1
            r2 = s - self.eos(p, t)[0]

            return r1, r2

    class EulerEquation(EquationBase):
        def residual(
            self,
            ht0: n0.tot.Enthalpy.Hint,
            ht1: n1.tot.Enthalpy.Hint,
            u0: n0.kin.BladeSpeed.Hint,
            u1: n1.kin.BladeSpeed.Hint,
            vt0: n0.kin.V_tan.Hint,
            vt1: n1.kin.V_tan.Hint,
        ):
            return (ht1 - ht0) - (u1 * vt1 - u0 * vt0)

    # +++ Fluid settings
    fluid_model_real = ExternalFluidModel(
        DebugAbstractState('HEOS', 'Air'),  # This just counts the number of updates
    )

    fluid_settings = FluidSettings(
        model=fluid_model_real,
        update_variables=(
            thrm.Pressure,
            thrm.Temperature,
        ),
    )
    nls.fluid_settings = fluid_settings

    n0 = NodeVariables(0)
    n1 = NodeVariables(1)

    nls.add_equation(TestEquation(), (3, 4))
    nls.add_boundary_conditions(
        {
            n0.kin.V_tan: 10,
            n1.kin.BladeSpeed: 10,
        },
    )

    nls.add_spanwise_constants(
        NodeVariables(3).tot.Enthalpy,
    )
    nls.build()
    res_func = nls.make_residual_function()
    nls.get_scaled_constraints()
    nls.get_scaled_guess()
    nls.make_rootfinder('ipopt')
