"""
Module that contains the tools to define equations, build systems made
up of assembly of such residual equations and routines to generate reasonable
initial guesses based on the available thermodynamic, geometric and kinematic
data.
Sometimes the CasADi api is slightly cryptic, sorry.
"""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import accumulate
from typing import Any, Callable, Iterable, Literal, Mapping, Self, Sequence, Type, cast

import casadi as cs
import jax as jax
import numpy as np
from numpy.typing import NDArray
from pint import Quantity, Unit
from pint.facets.plain import PlainQuantity

from adet.constants import AdetArray
from adet.equations.base_equation import EquationBase
from adet.errors import ExistingEquationError
from adet.fluid.casadi_eos import CasadiEos
from adet.fluid.eos_factory import EosFactory
from adet.fluid.settings import FluidSettings
from adet.registries import ScalingRegistry
from adet.tools.interpolation import resample_linear
from adet.tools.iter import ensure_tuple, leaves
from adet.variables import ThermoVariables
from adet.varspec import NodeStates, VarSpec

logger = logging.getLogger(__name__)

THERMO_CONST_SUFFIX = '__thrmCNS'

_scale_reg = ScalingRegistry()


IPOPT_DEFAULTS = {
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
    'ipopt.hessian_approximation': 'limited-memory',  # Less updates
}


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
        self.fluid_settings: None | FluidSettings = None
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
        abs_position: int | Sequence[int],
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
                abs_arg = arg.at_node(abs_node)
                decl_args.append(abs_arg)

        return tuple(decl_args)

    def _build_argument_maps(self) -> dict[EquationBase, dict[int, int]]:
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
        flag_unused = spec not in self.data.decl_args

        if flag_unused:
            logger.warning(
                f'Imposing a condition {caller} on `{spec.full_symbol(True)}`'
                f', but it does not appear in any equation'
            )

        return flag_unused

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
                    if spec.scalar:
                        mag_valid = mag * np.ones(1)
                    else:
                        mag_valid = mag * np.ones(self.data.num_span)
                else:
                    raise ValueError(f'Length mismatch {spec}')
            else:
                mag_valid = mag

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

    def _validate_length(self, spec: VarSpec, val: AdetArray | PlainQuantity):
        if isinstance(val, PlainQuantity):
            val = val.to_base_units().magnitude

        val_array = np.atleast_1d(val)

        if len(val_array) == 1:
            if spec.scalar:
                val_valid = val_array
            else:
                val_valid = np.repeat(val_array, self.data.num_span)
        elif len(val_array) != self.data.num_span:
            raise ValueError(
                f'Length mismatch for boundary condition {spec.full_symbol(True)}'
            )
        else:
            val_valid = val_array

        return val_valid

    def _validate_all_units(self) -> None:
        """Write the stored boundary conditions"""
        logger.debug('Checking boundary conditions units...')
        for spec, value in self.data.boun_cond.items():
            if isinstance(value, PlainQuantity):
                def_unit = Unit(spec.unit)
                if not value.units.is_compatible_with(def_unit):
                    raise ValueError(
                        f'{def_unit} is not compatible with {value.units}'
                        f' prescribed in the boundary conditions for {spec.symbol}'
                    )

                # Do not convert yet
                self.data.boun_cond[spec] = value
            else:
                self.data.boun_cond[spec] = value

    def check_constraints_effectiveness(self) -> None:
        """
        Check if the arguments used in equalities and spanwise_constants
        appear as declared arguments
        """
        # TODO: Could check if they are free (?)
        for equality in self.data.equalities:
            for arg in equality:
                self._check_arg_declaration(arg, 'equalities')

        for arg in self.data.spanwise_constants:
            self._check_arg_declaration(arg, 'spanwise constants')

        for arg in self.data.boun_cond:
            # Skip update arguments
            if arg in self.data.thermo_updt_args:
                continue
            self._check_arg_declaration(arg, 'boundary conditions')

    def get_array_boun_conds(self) -> list[NDArray]:
        bc_values = []
        for spec, val in self.data.boun_cond.items():
            val_valid = self._validate_length(spec, val)
            bc_values.append(val_valid)
        return bc_values

    def get_plain_bc_dict(self) -> dict[VarSpec, NDArray]:
        values = self.get_array_boun_conds()
        plain_bcs = {}
        for spec, val in zip(self.data.boun_cond.keys(), values):
            plain_bcs[spec] = val

        return plain_bcs


class ArgumentResolver:
    """Resolves free arguments vs followers, handles EoS logic"""

    def __init__(self, data: SystemSharedData):
        self.data = data

    def identify_free_arguments(self) -> tuple[VarSpec, ...]:
        """
        Get the real thermodynamic and kinematic arguments needed to complete
        the different states of the node.
        """
        return tuple(
            self._get_effective_arguments(),
        )

    def _get_effective_arguments(self) -> set[VarSpec]:
        """
        Get the thermo that act on thermodynamic state updates.
        e.g. if hmass, smass, p, T appear on the same state,
        only two variables are effective (pure substance + phase),
        while the other two are followers
        """
        if self.data.fluid_settings is None:
            return set(self.data.decl_args) - set(self.data.boun_cond)

        # Non thermodynamic arguments
        nonthermo_args = [arg for arg in self.data.decl_args if not arg.state]

        # Get what variables will be used for state updates
        self.data.thermo_updt_args = []
        prescr_upd_vars = self.data.fluid_settings.update_variables

        nodes = [arg.node for arg in self.data.decl_args]
        first_node = min(nodes)
        last_node = max(nodes)

        for node in range(first_node, last_node + 1):
            for st in NodeStates:
                upd_args = [v.at_node(node)._with_state(st) for v in prescr_upd_vars]
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
        discarded_args = (
            set(self.data.decl_args)
            - set(self.data.thermo_updt_args)
            - set(self.data.free_args)
        )

        for v_spec in self.data.boun_cond:
            if self.data.thermo_updt_args:
                discarded_args.add(v_spec)

        return [arg for arg in discarded_args if arg.state]

    # *** System introspection
    def make_arg_structure(self, arguments: Sequence[VarSpec]):
        """Detect the argument structure of a sequence of arguments"""
        arguments_struct = []
        for arg in arguments:
            num_span = 1 if arg.scalar else self.data.num_span
            branch = num_span * [0]
            arguments_struct.append(branch)

        return arguments_struct

    def get_args_coordinates(self):
        arg_struct = self.make_arg_structure(self.data.free_args)
        arg_lengths = [len(arg) for arg in arg_struct]
        acc_lengths = [0] + list(accumulate(arg_lengths[:-1]))
        return arg_lengths, acc_lengths

    def position_from_idx(self, arg_index: int) -> tuple[int, int]:
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


class UnitScalingManager:
    """Handles unit checking and scaling factors"""

    def __init__(self, data: SystemSharedData):
        self.data = data

    def _check_all_eqs_units(self):
        self.data.equations_units = {}

        """Check units for all equations"""
        for eq in self.data.equations:
            self.data.equations_units[eq] = self._test_singl_eq_units(eq)

        logger.debug('Units for the residual equations successfully verified')

    def _test_singl_eq_units(self, equation: EquationBase) -> tuple[str, ...]:
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

        if not isinstance(residuals, (tuple, list)):
            residuals = (residuals,)

        for r in residuals:
            if r is None:
                raise ValueError(f'No residuals return found for equation {equation}')

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
        self,
        custom_bounds: dict[VarSpec, tuple[float, float]],
        ignore_defaults: bool = False,
    ) -> list[tuple[float, float]]:
        """The custom bounds are to be provided dimensionally"""
        bounds = []
        for spec in self.data.free_args:
            if spec in custom_bounds:
                lower_bound, upper_bound = custom_bounds[spec]
            else:
                if spec.Glob in custom_bounds:
                    lower_bound, upper_bound = custom_bounds[spec.Glob]
                else:
                    if spec.bounds is None or ignore_defaults:
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

        num_equations = leaves(self.data.equations_units.values()).__len__()
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
    def num_equations(self) -> int:
        return leaves(self.data.equations_units.values()).__len__()

    @property
    def first_node(self) -> int:
        return min(arg.node for arg in self.data.decl_args)

    @property
    def last_node(self) -> int:
        return max(arg.node for arg in self.data.decl_args)

    def reset(self) -> None:
        old_settings = self.data.fluid_settings
        self.__init__(self.data.num_span)
        self.data.fluid_settings = old_settings

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

    def add_boundary_conditions(self, bnd_cond: dict[VarSpec, Any]):
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

        # Build the free arguments
        self.data.free_args = self._argument_resolver.identify_free_arguments()

        # Check thast boundary conditions respect their declared units
        self._constraint_manager._validate_all_units()
        self._constraint_manager.check_constraints_effectiveness()

        # Check the equations units and build their scaling factors
        self._scaling_manager._check_all_eqs_units()

        self.data.built = True
        logger.info('Parent system assembled successfully')

    def _check_built(self) -> None:
        """Check that the system is flagged as built"""
        if not self.data.built:
            raise RuntimeError(
                'The system is not built or failed to do so, build it `build()`'
            )

    def get_arg_position(self, arg_index: int):
        return self._argument_resolver.position_from_idx(arg_index)

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

    def sol_to_dict(self, solution: NDArray) -> dict[VarSpec, NDArray]:
        sol_dict = {}
        curr_idx = 0
        scales = self.free_args_scaling
        solution = solution.flatten()
        for arg_pos, spec in enumerate(self.data.free_args):
            arg_length = 1 if spec.scalar else self.num_span
            scl = scales[arg_pos]
            sol_dict[spec] = solution[curr_idx : curr_idx + arg_length] * scl
            curr_idx += arg_length

        bc_dict = self._constraint_manager.get_plain_bc_dict()
        thrm_dict = self._compute_secondary_thermo({**sol_dict, **bc_dict})

        all_data = {**sol_dict, **thrm_dict, **bc_dict}

        def sort_func(x: tuple[VarSpec, Any]):
            name = x[0].symbol + str(x[0])
            return name.lower()

        return dict(
            sorted(all_data.items(), key=sort_func),
        )

    def _compute_secondary_thermo(
        self, sol_data: dict[VarSpec, NDArray]
    ) -> dict[VarSpec, NDArray]:

        if self.data.fluid_settings is None:
            return {}

        thrm_data = {}

        # TODO: Choose variables to write
        _thrm = ThermoVariables()
        TO_WRITE = [
            _thrm.Entropy,
            _thrm.Density,
            _thrm.Enthalpy,
            _thrm.SpeedSound,
            _thrm.GasConstant,
            _thrm.Viscosity,
            _thrm.MolarMass,
        ]

        # Extract fluid settings data
        fluid_settings = self.data.fluid_settings
        abs_state = fluid_settings.fluid_state
        input_pair = fluid_settings.input_pair
        # Global version of updated variables
        var0_glb = fluid_settings.update_variables[0].Glob
        var1_glb = fluid_settings.update_variables[1].Glob

        # Write the specs at each node
        for spec in TO_WRITE:
            for state in NodeStates:
                for node in range(self.last_node + 1):
                    upd_var0 = var0_glb.at_node(node)._with_state(state)
                    upd_var1 = var1_glb.at_node(node)._with_state(state)

                    v0_values = sol_data[upd_var0]
                    v1_values = sol_data[upd_var1]

                    pty_arr = []
                    for v0, v1 in zip(v0_values, v1_values):
                        abs_state.update(input_pair, v0, v1)
                        pty_meth = getattr(abs_state, spec.symbol)
                        try:
                            pty_arr.append(pty_meth())
                        except Exception:  # Catch property extraction failure
                            pty_arr.append(np.nan)

                    spec = spec.at_node(node)._with_state(state)
                    thrm_data[spec] = np.array(pty_arr)

        return thrm_data

    def get_bounds(self, custom_bounds={}):
        self._check_built()
        self._scaling_manager.get_arguments_bounds(custom_bounds)

    def get_boundary_conds(self) -> list[NDArray]:
        self._check_built()
        dimensional_constr = self._constraint_manager.get_array_boun_conds()
        return jax.tree.map(
            lambda x, y: x / y,
            dimensional_constr,
            self.constraints_scaling,
        )

    def get_guess(
        self,
        manual_values: Mapping[VarSpec, AdetArray] = {},
        fallback: float | None = None,
    ) -> list[NDArray]:
        """Generate initial guesses for free arguments"""
        self._check_built()
        guesses = []
        manual_values = dict(manual_values)

        for spec in self.data.free_args:
            # If there is a manual value, overwrite the guess registry

            if spec in manual_values:
                guess_value = manual_values[spec]
                logger.debug(f'Using manual value {guess_value} for {spec}')
            elif spec.Glob in manual_values:
                guess_value = manual_values[spec.Glob]
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
                if fallback is not None:
                    guess_value = fallback
                # Otherwise ask for user input
                else:
                    input_msg = (
                        f'INPUT >>> DIMENSIONAL guess for {spec.symbol} [1.0] = '
                    )
                    guess_value = float(input(input_msg) or 1.0)
                    manual_values[spec.Glob] = np.atleast_1d(guess_value)

            guess_value = np.atleast_1d(guess_value)

            if max(guess_value.shape) != self.data.num_span:
                logger.debug(
                    f'Length mismatch in guess for {spec}, using linear resampling'
                )
                # This simply repeats the single value when num_span=1
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

        self._eos_callbacks: dict[
            int,
            dict[
                NodeStates,
                cs.Function | CasadiEos | Any,
            ],
        ] = {}

    def build(self, scaled: bool = True):
        super().build(scaled)
        logger.info('Building CasADi backend...')
        self._reset_symbols()
        self._build_base_symbols()  # Create symbols and their scaler
        self._build_products()  # Multiply them
        self._build_residual_expressions()  # Plug them into residual expressions
        logger.info('Backend built successfully')
        self._count_variables()

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

        # TODO: Refactor this whole method, it is quite messy
        if self.data.fluid_settings is None:
            return {}

        fl_state = self.data.fluid_settings.fluid_state

        self._eos_callbacks = {
            n_idx: dict.fromkeys(
                NodeStates,
                None,
            )
            for n_idx in range(self.first_node, self.last_node + 1)
        }

        self._eos_factory = EosFactory(fl_state)

        # Add inter-node eos
        for eq, eq_pos in self.data.equations.items():
            eq_conf = eq.config
            pos_str = '_'.join(str(p) for p in eq_pos)
            if eq_conf.input_pair:
                eq.eos = self._eos_factory.make_eos(
                    eq_conf.input_pair,
                    eq_conf.out_properties,
                    self.num_span,
                    f'multi_{eq.__class__.__name__}_{pos_str}',
                )

        discarded_vars = self._argument_resolver.get_discarded_thermo_args()
        out_syms: dict[VarSpec, cs.MX] = {}

        sorted_discarded: dict[
            int,
            dict[NodeStates, list[VarSpec]],
        ] = {}
        # build output properties
        for spec in discarded_vars:
            if spec.node not in sorted_discarded:
                sorted_discarded[spec.node] = {state: [] for state in NodeStates}
            if spec.state and spec not in self.data.boun_cond:
                sorted_discarded[spec.node][spec.state].append(spec)

        for node_idx in range(self.first_node, self.last_node + 1):
            # If there is no need for any update variables, move to next node
            if node_idx not in sorted_discarded:
                continue

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
                    spec._with_state(state).at_node(node_idx)
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
            for two_args in arg_couples:
                # If both argument do not appear in the equations, skip to next couple
                # if one of them is unused by other eqns. it is useless to add it
                if set(two_args).intersection(self._all_symbols):
                    # TODO: Make this fail more gracefully
                    # or add the equalities to free arguments
                    sym0 = self._all_symbols[two_args[0]]
                    sym1 = self._all_symbols[two_args[1]]

                    # NOTE: We don't care about scaling, they are just identities
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
        logger.info('Building residual equation symbolics...')

        residuals: list[Any | tuple[Any, ...]] = []
        for eq in self.data.equations:
            args = []
            # Transpose to absolute node indices
            for spec in eq.arg_specs:
                arg_map = self.data._arg_maps[eq]
                abs_arg = spec.at_node(arg_map[spec.node])
                args.append(self._all_symbols[abs_arg])

            res_syms = eq.residual(*args)

            # Check that the manual units have the correct length
            if eq.config.manual_units:
                self._man_units_len_check(eq, res_syms)

            residuals.append(res_syms)

        # Divide each residual expression by its scaling symbol
        self.residual_expr = list(
            map(
                lambda X, Y: X / Y,
                leaves(residuals),
                self._eq_scales_sym,
            )
        )

        # Add secondary expressions
        self.residual_expr += self._build_equalities_expr()
        self.residual_expr += self._build_spanwise_constants()
        self.residual_expr += self._build_thermo_constraints()

    def _count_variables(self):
        # Count variables and residuals -> Warn user for mismatch
        num_vars = max(cs.vertcat(*self.free_args_sym.values()).shape)
        num_residuals = max(cs.vertcat(*self.residual_expr).shape)
        logger.info(
            f'System info: {num_residuals} total equations, {num_vars} total variables'
        )

        if num_vars != num_residuals:
            mismatch = abs(num_vars - num_residuals)
            if num_vars > num_residuals:
                logger.warning(f'Add {mismatch} conditions for a square system.')
            else:
                logger.warning(f'Release {mismatch} conditions for a square system.')

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

    def _man_units_len_check(
        self,
        equation: EquationBase,
        residual_symbols: cs.MX | tuple[cs.MX, ...] | list[cs.MX],
    ):
        """Check the matching between residual and manual units"""

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

        # NOTE:
        # CasADi boilerplate:
        # minimize f(x,p)
        # subj. to:
        # --- lbg <= g(x,p) <= lbg
        # --- lbx <= x <= ubx

        rootfind_problem = {
            'x': free_args_symbols,
            'p': constraints_symbols,
            **func_spec,
        }

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

        elif root_method == 'ipopt' or root_method == 'lstsq':
            # IPOPT solver
            rootfinder = cs.nlpsol(
                'ipopt_rootfinder',
                'ipopt',
                rootfind_problem,
                {
                    **IPOPT_DEFAULTS,
                    **opts,
                },
            )
        return rootfinder

    def get_bounds(
        self,
        custom_bounds: dict[VarSpec, tuple[float, float]] = {},
        ignore_defaults: bool = False,
    ):
        bounds_by_arg = self._scaling_manager.get_arguments_bounds(
            custom_bounds, ignore_defaults
        )
        lbx = []
        ubx = []

        for arg, scales in zip(self.data.free_args, bounds_by_arg):
            arg_size = max(self._all_symbols[arg].shape)
            lbx += arg_size * [scales[0]]
            ubx += arg_size * [scales[1]]

        return cs.vertcat(*lbx), cs.vertcat(*ubx)
