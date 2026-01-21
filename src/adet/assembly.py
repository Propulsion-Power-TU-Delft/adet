"""
Module that contains the tools to define equations, build systems made
up of assembly of such residual equations and routines to generate reasonable
initial guesses based on the available thermodynamic, geometric and kinematic
data.
"""

from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from copy import deepcopy
from itertools import combinations
import logging
from typing import Callable, Mapping, Self, Sequence, Type, Literal, Any, cast

from numpy.typing import NDArray
from pint import Quantity

import numpy as np
from pint.facets.plain import PlainQuantity
import sympy as sp
import casadi as cs

import jax as jax
import jax.numpy as jnp

from adet.equations.base_equation import EquationBase
from adet.errors import ExistingEquationError
from adet.fluid.casadi_eos import CasadiEos, CasadiEosFactory
from adet.fluid.settings import (
    AnalyticalFluidModel,
    EmptyFluidModel,
    ExternalFluidModel,
    FluidSettings,
)
from adet.registries import (
    GuessRegistry,
    ScalingRegistry,
    ScalarsRegistry,
    VariableBoundsRegistry,
)
from adet.tools.coolprop_utils import (
    pair_based_sorting,
    pair_id_from_tuple,
    pair_tuple_from_id,
)
from adet.tools.strings import get_arg_state, rm_end_digits, get_index, get_arg_type
from adet.node import FlowNode
from adet.constants import NodeStatesNames, ArrayLike
from adet.tools.context import dummy_context, override_operators, output_suppression


logger = logging.getLogger(__name__)


def get_units_string(var):
    return str(var.to_base_units().units)


_scale_reg = ScalingRegistry()
_scalars_reg = ScalarsRegistry()
_bounds_reg = VariableBoundsRegistry()
_guess_reg = GuessRegistry()


class SystemSharedData:
    """
    Shared data container accessible to all managers.
    This holds all the state that needs to be shared between different
    components of the system assembler.
    """

    def __init__(self):
        # Core structures
        self.equations: OrderedDict[EquationBase, tuple[int, ...]] = OrderedDict()
        self.nodes: tuple[FlowNode, ...] = tuple()
        self._arg_maps: dict[EquationBase, dict[str, str]] = {}

        # Arguments
        self.declared_arguments: tuple[str, ...] = tuple()
        self.free_args: tuple[str, ...] = tuple()
        self.constraints: tuple[str, ...] = tuple()
        self.constraints_values: list[NDArray] = []
        self.scalar_arguments: tuple[str, ...] = tuple()

        # Boundary conditions and constraints
        self.boundary_conditions: defaultdict[
            int,
            defaultdict[
                NodeStatesNames,
                dict[str, ArrayLike | PlainQuantity],
            ],
        ] = defaultdict(lambda: defaultdict(dict))
        self.global_constraints: defaultdict[
            NodeStatesNames, dict[str, ArrayLike | PlainQuantity]
        ] = defaultdict(dict)
        self.invariants: list[set[str]] = []
        self.spanwise_constants: set[str] = set()

        # Units and scaling
        self.arguments_units: dict[str, str] = {}
        self.equations_units: list[list[str]] = []
        self.scaled: bool = False

        # Settings
        self.fluid_settings: FluidSettings = FluidSettings(EmptyFluidModel())
        self.num_span: int = 1

        # EoS tracking
        self.analytic_eos_equations: list[EquationBase] = []

        # Build status
        self.built: bool = False


class EquationRegistry:
    """Manages equations, nodes, and their argument mappings"""

    def __init__(self, data: SystemSharedData):
        self.data = data

    def add_equation(
        self,
        equation: EquationBase,
        nodal_position: int | list[int] | tuple[int, ...],
    ):
        """Add an equation to the system in the specified position"""
        if isinstance(nodal_position, int):
            nodal_position = [nodal_position]

        # Check that the provided position has the same length as equation arguments
        local_indices = {get_index(arg) for arg in equation.arguments}
        if len(local_indices) != len(nodal_position):
            raise ValueError(
                f'Detected indices of in the definition of '
                f'`{equation.__class__.__name__}` {tuple(local_indices)} '
                f'is not equal to length of the prescribed '
                f'absolute nodal position {nodal_position}'
            )

        # Check that an equation of the same type does not exist at the same location
        for eq_instance, eq_nodes in self.data.equations.items():
            if isinstance(eq_instance, equation.__class__) and (
                set(eq_nodes) == set(nodal_position)
            ):
                raise ExistingEquationError(
                    f'Duplicate equation entry for {equation.__class__.__name__}'
                    f' at position {nodal_position}'
                )

        self.data.equations[equation] = tuple(nodal_position)

        logger.debug(
            f'Added equation {equation.__class__.__name__} to system '
            f'in position {nodal_position}'
        )

    def remove_equation_type(self, *equation_class: Type[EquationBase]):
        """Remove all equations of the specified types"""
        cleaned_equations = self.data.equations.copy()
        for eq in self.data.equations:
            if isinstance(eq, equation_class):
                cleaned_equations.pop(eq)
        self.data.equations = cleaned_equations

    def remove_equation(
        self,
        equation_class: Type[EquationBase],
        nodal_position: int | tuple[int, ...],
    ):
        """Remove equation from the system"""
        if isinstance(nodal_position, int):
            nodal_position = tuple([nodal_position])

        equation_found = False
        for eq_instance, eq_position in self.data.equations.copy().items():
            if isinstance(eq_instance, equation_class) and (
                set(eq_position) == set(nodal_position)
            ):
                equation_found = True
                self.data.equations.pop(eq_instance)

        if not equation_found:
            logger.warning(
                f'No equation {equation_class} found to remove in {nodal_position}'
            )

    def create_nodes(self):
        """Create nodes based on the nodal positions of the equations"""
        logger.debug('Creating nodes for the equation system')

        node_indices = set()
        for nodal_pos in self.data.equations.values():
            node_indices.update(nodal_pos)

        if not node_indices:
            raise RuntimeError(
                'Cannot build an empty system! Please add some equations'
            )

        if min(node_indices) > 0:
            logger.warning(
                'Minimum node index for equation system is '
                'greater than 0, consider shifting your nodal positions. '
                'This may produce unexpected behaviours.'
            )

        self.data.nodes = tuple(
            FlowNode(self.data.fluid_settings, self.data.num_span)
            for _ in range(0, 1 + max(node_indices))
        )

        logger.debug(f'Successfully created {len(self.data.nodes)} nodes')

    def add_analytical_eos(self):
        """Add analytical equations of state to the system if the model calls for it"""
        fl_model = self.data.fluid_settings.model

        if isinstance(fl_model, AnalyticalFluidModel):
            for node_idx, _ in enumerate(self.data.nodes):
                for eq in fl_model.get_equations():
                    try:
                        self.add_equation(eq, node_idx)
                        logger.debug(
                            f'Added EoS equation {eq.__class__.__name__} at {node_idx}'
                        )
                        self.data.analytic_eos_equations.append(eq)
                    except ExistingEquationError:
                        # If the equation of state is already there just keep going
                        pass

    def build_argument_maps(self) -> tuple[str, ...]:
        """
        1. Get all the available arguments and assign them the correct absolute index.
        2. Creates the keyword argument map for each equation, which maps the relative
           argument of that equation to the absolute system arguments.
        """
        system_arguments = []
        scalar_arguments = []
        self.data._arg_maps = {}

        logger.debug('Reading all the equation arguments...')

        for eq, eq_position in self.data.equations.items():
            local_indices: set[int] = {get_index(arg) for arg in eq.arguments}

            index_map = {
                rel_pos: abs_pos
                for abs_pos, rel_pos in zip(eq_position, sorted(local_indices))
            }
            self.data._arg_maps[eq] = {}

            for arg in eq.arguments:
                arg_rel_idx = get_index(arg)
                arg_abs_idx = index_map[arg_rel_idx]

                arg_type = get_arg_type(arg)
                arg_no_digit = rm_end_digits(arg)

                system_arg = arg_no_digit + str(eq_position[arg_rel_idx])
                system_arguments.append(system_arg)

                if arg_type in _scalars_reg:
                    scalar_arguments.append(system_arg)

                # Create the variable in the node
                self.data.nodes[arg_abs_idx].create_vars(arg_no_digit)
                self.data._arg_maps[eq][arg] = arg_no_digit + str(arg_abs_idx)

        system_arguments = sorted(set(system_arguments))
        self.data.scalar_arguments = tuple(scalar_arguments)

        logger.debug(f'Arguments detected are: {", ".join(system_arguments)}')

        return tuple(system_arguments)

    @property
    def num_equations(self):
        return sum(eq.num_equations for eq in self.data.equations)


class ConstraintManager:
    """Handles all forms of constraints and boundary conditions"""

    def __init__(self, data: SystemSharedData):
        self.data = data

    def add_boundary_conditions(self, bnd_cond: dict, node_idx: int):
        """Add boundary conditions for a specific node"""
        for state_id, state_bnd_cond in bnd_cond.items():
            self.data.boundary_conditions[node_idx][state_id].update(state_bnd_cond)

    def add_global_constraints(self, bnd_cond: dict):
        """Add global constraints that apply to all nodes"""
        self.data.global_constraints.update(bnd_cond)

    def add_invariants(self, *invariants: tuple[str, ...]):
        """
        Each tuple represents a set of variables treated as equal by the system.
        Adding ('a', 'b', 'c') adds the equations: a-b=0, a-c=0, b-c=0
        """
        for args in invariants:
            if set(args) not in self.data.invariants:
                self.data.invariants.append(set(args))

    def add_spanwise_constants(self, *arguments: str):
        for arg in arguments:
            self.data.spanwise_constants.add(arg)

    def write_to_nodes(self):
        """Write the stored boundary conditions to the nodes"""
        logger.debug('Writing boundary conditions to nodes...')

        def to_base_units(var) -> NDArray:
            if isinstance(var, PlainQuantity):
                mag = var.to_base_units().magnitude
            else:
                mag = var
            return np.atleast_1d(mag)

        for node_idx, node in enumerate(self.data.nodes):
            # Write global constraints
            self.add_boundary_conditions(self.data.global_constraints, node_idx)

            # Convert to base units arrays
            bc_arrays = jax.tree.map(
                to_base_units,
                self.data.boundary_conditions[node_idx],
                is_leaf=lambda x: isinstance(x, (list, tuple)),
            )

            for state_id, constraints in bc_arrays.items():
                state_obj = node.fetch_state(state_id)
                # Add them to the node
                for var, val in constraints.items():
                    state_obj.set_value(var, val)
                    state_obj.change_status(var, fixed=True)

    def extract_constraints(self) -> tuple[tuple[str, ...], list[NDArray]]:
        """Get all the constraints defined by the nodes taking part in the system"""
        constraint_names = []
        constraint_values = []

        for node_idx, node in enumerate(self.data.nodes):
            for arg_no_idx, value in node.get_constraints().items():
                arg = arg_no_idx + str(node_idx)
                if arg not in self.data.declared_arguments:
                    logger.warning(f'Unused constraint {arg}')
                constraint_names.append(arg)
                constr_value = value.to_base_units().magnitude

                if arg in self.data.scalar_arguments:
                    constr_value = np.atleast_1d(constr_value[0])

                constraint_values.append(constr_value)

        return tuple(constraint_names), constraint_values


class ArgumentResolver:
    """Resolves free arguments vs followers, handles EoS logic"""

    def __init__(self, data: SystemSharedData):
        self.data = data

    def identify_free_arguments(self) -> tuple[str, ...]:
        """
        Get the real thermodynamic and kinematic arguments needed to complete
        the different states of the node.
        """
        if isinstance(self.data.fluid_settings.model, AnalyticalFluidModel):
            # Just a difference between sets, sorted
            return tuple(
                sorted(
                    set(self.data.declared_arguments) - set(self.data.constraints),
                )
            )
        else:
            return tuple(
                sorted(self._get_effective_arguments()),
            )

    def _get_effective_arguments(self):
        """
        Get the thermo that act on thermodynamic state updates.
        e.g. if hmass, smass, p, T appear on the same state,
        only two variables are effective (pure substance + phase),
        while the other two are followers
        """
        # Non thermodynamic arguments
        nonthermo_args = [
            arg
            for arg in self.data.declared_arguments
            if not arg.startswith(('stc', 'tot', 'rlt'))
        ]

        # Get what variables will be used for state updates
        update_args = []
        for node_idx, node in enumerate(self.data.nodes):
            updt_vars = node.get_update_variables()

            for state, updt_pair in updt_vars.items():
                update_args += [f'{state}_{var}{node_idx}' for var in updt_pair]

        return set(update_args + nonthermo_args).difference(self.data.constraints)

    def get_discarded_thermo_args(
        self,
    ) -> defaultdict[int, defaultdict[str, list[str]]]:
        """
        Retrieve the discarded thermo arguments a.k.a. the ones
        that were declared in the equations but have become extractions
        from the equations of state
        """
        all_discarded = (
            set(self.data.declared_arguments)
            - set(self.data.constraints)
            - set(self.data.free_args)
        )

        discarded = defaultdict(lambda: defaultdict(list))

        for arg in all_discarded:
            arg_state = get_arg_state(arg)
            arg_idx = get_index(arg)
            arg_type = get_arg_type(arg)

            discarded[arg_idx][arg_state].append(arg_type)

        return discarded

    def make_arg_structure(self, arguments: Sequence[str]):
        """Detect the argument structure of a sequence of arguments"""
        arguments_struct = []
        for arg in arguments:
            num_span = 1 if arg in self.data.scalar_arguments else self.data.num_span
            branch = num_span * [0]
            arguments_struct.append(branch)

        return jax.tree.structure(arguments_struct)


class UnitScalingManager:
    """Handles unit checking and scaling factors"""

    def __init__(self, data: SystemSharedData):
        self.data = data

    def extract_args_units(self):
        self.data.arguments_units = {}

        for idx, node in enumerate(self.data.nodes):
            node_arguments = {
                f'{arg}{idx}': var for arg, var in node.get_all_quantities().items()
            }

            self.data.arguments_units.update(
                {arg: get_units_string(var) for arg, var in node_arguments.items()}
            )

    def check_equations_units(self):
        self.data.equations_units = []

        """Check units for all equations"""
        for eq, kwmap in self.data._arg_maps.items():
            self.data.equations_units.append(self._get_eq_units(eq, kwmap))

        logger.debug('Units for the residual equations successfully verified')

    def _get_eq_units(self, equation: EquationBase, kwmap: dict[str, str]) -> list[str]:
        """Get units for a single equation"""
        if equation.manual_units:
            return list(equation.manual_units)

        args = []
        for arg in equation.arguments:
            absolute_argument = kwmap[arg]
            units = self.data.arguments_units[absolute_argument]
            dummy_value = Quantity([np.nan], units)
            args.append(dummy_value)

        res = equation.residual(*args)

        if not isinstance(res, (list, tuple)):
            res = (res,)

        return [get_units_string(r) for r in res]

    def _assign_scaling_factor(self, units: str):
        """Assign scaling factor based on units"""
        if not self.data.scaled:
            return 1.0
        return _scale_reg[units]

    def get_free_args_scaling(self):
        """Build multi-span scaling factors for free arguments"""
        return self._argument_scaling_helper(self.data.free_args)

    def get_constraints_scaling(self):
        """Build multi-span scaling factors for constraints"""
        return self._argument_scaling_helper(self.data.constraints)

    def _argument_scaling_helper(self, arguments: Sequence[str]) -> list[float]:
        """Returns the arguments scales for a sequence of system arguments"""
        if not self.data.scaled:
            scales = len(arguments) * [1.0]
        else:
            all_args_scales: dict[str, float] = jax.tree.map(
                self._assign_scaling_factor,
                self.data.arguments_units,
            )
            scales = [all_args_scales[arg] for arg in arguments]

        return scales

    def get_equations_scaling(self):
        """Build multi-span scaling factors for equations"""
        eq_scales = []

        num_equations = sum(eq.num_equations for eq in self.data.equations)

        if not self.data.scaled:
            scales = np.ones(num_equations)
        else:
            for eq, units in zip(self.data.equations.keys(), self.data.equations_units):
                if eq.scaling_factor:
                    logging.debug(
                        f'Custom scaling factor found for {eq.__class__.__name__}, '
                        f'{eq.scaling_factor}'
                    )
                    eq_scales += eq.scaling_factor
                else:
                    eq_scales += [self._assign_scaling_factor(u) for u in units]

            scales = np.array(eq_scales)

        return scales


class SolutionDispatcher:
    """Handles solution formatting and node updates"""

    def __init__(self, data: SystemSharedData):
        self.data = data

    def unflatten_solution(self, solution_values: NDArray) -> list[NDArray]:
        """Unflatten the solution based on a PyTree definition"""
        arg_resolver = ArgumentResolver(self.data)
        free_args_struct = arg_resolver.make_arg_structure(self.data.free_args)
        solution_values_inflated = jax.tree.unflatten(
            free_args_struct, solution_values.flatten().tolist()
        )

        return list(map(np.array, solution_values_inflated))

    def solution_to_dict(
        self, solution_values: list[NDArray] | NDArray, scaling: list[float]
    ) -> dict[str, NDArray]:
        """
        Map argument values to a dict pointing to which argument they belong,
        useful for passing information between systems for initialization
        """
        if isinstance(solution_values, np.ndarray):
            solution_values = self.unflatten_solution(solution_values)

        if self.data.scaled:
            solution_values = jax.tree.map(
                lambda x, y: x * y,
                solution_values,
                scaling,
            )

        # Arguments in absolute indices
        return {
            arg: solution_values[idx] for idx, arg in enumerate(self.data.free_args)
        }

    def split_arguments_by_node(self, arguments: Sequence[str]):
        """
        Split the arguments, in the form of <state>_<var_type><abs_node_idx>
        in their respective FlowNode, making them easier to read/write
        """
        node_by_args: dict[FlowNode, list[str]] = {node: [] for node in self.data.nodes}

        for arg in set(arguments):
            node = self.data.nodes[get_index(arg)]
            node_by_args[node].append(rm_end_digits(arg))

        return node_by_args

    def write_solution_to_nodes(self, solution_values: NDArray, scaling: list[float]):
        """Dispatch the provided values to the correct node"""
        split_arg_dictionaries = self.split_arguments_by_node(self.data.free_args)
        solution_dict = self.solution_to_dict(solution_values, scaling)

        arg_print = ', '.join(solution_dict)
        logger.debug(f'Writing arguments to node: {arg_print}')

        for node, args_to_write in split_arg_dictionaries.items():
            node_idx = str(self.data.nodes.index(node))
            node.write_to_node(
                {arg: solution_dict[arg + node_idx] for arg in args_to_write},
                fixed=False,
            )

        return solution_dict


class SystemAssembler(ABC):
    """
    Class for assembling a system of equations, gathering its arguments
    and returning the residuals dispatched to the equations it is made
    out of.

    This class now uses a composition-based architecture with specialized
    managers handling different responsibilities.
    """

    def __init__(self, num_span: int) -> None:
        # Initialize shared context
        self.data = SystemSharedData()
        self.num_span = num_span

        # Initialize managers
        self._equation_registry = EquationRegistry(self.data)
        self._constraint_manager = ConstraintManager(self.data)
        self._argument_resolver = ArgumentResolver(self.data)
        self._scaling_manager = UnitScalingManager(self.data)
        self._solution_dispatcher = SolutionDispatcher(self.data)

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
        return self._equation_registry.num_equations

    # Expose context properties for backward compatibility
    @property
    def equations(self):
        return self.data.equations

    @property
    def nodes(self):
        return self.data.nodes

    @property
    def free_args(self):
        return self.data.free_args

    @property
    def constraints(self):
        return self.data.constraints

    @property
    def constraints_values(self):
        return self.data.constraints_values

    @property
    def boundary_conditions(self):
        return self.data.boundary_conditions

    @property
    def _analytic_eos_equations(self):
        return self.data.analytic_eos_equations

    @property
    def _declared_arguments(self):
        return self.data.declared_arguments

    @property
    def _scalar_arguments(self):
        return self.data.scalar_arguments

    @property
    def _arg_maps(self):
        return self.data._arg_maps

    @property
    def _arguments_units(self):
        return self.data.arguments_units

    @property
    def _equations_units(self):
        return self.data.equations_units

    @property
    def _built(self):
        return self.data.built

    @property
    def _scaled(self):
        return self.data.scaled

    @property
    def _global_constraints(self):
        return self.data.global_constraints

    @property
    def _invariants(self):
        return self.data.invariants

    @property
    def _spanwise_constants(self):
        return self.data.spanwise_constants

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
            'invariants',
        ]

        out_dict = {}
        for field in FIELDS_TO_SAVE:
            try:
                attr = getattr(self.data, field)
            except AttributeError as e:
                raise AttributeError(
                    f'{e} encountered while trying to deepcopy attribute {field}'
                )

            out_dict[field] = deepcopy(attr)

        return out_dict

    def from_dict(self, data_dict):
        """Load the context from a dictionary"""
        for attr, value in data_dict.items():
            setattr(self.data, attr, value)

    def add_boundary_conditions(self, bnd_cond: dict, node_idx: int):
        """Delegate to constraint manager"""
        self._constraint_manager.add_boundary_conditions(bnd_cond, node_idx)

    def add_global_constraints(self, bnd_cond: dict):
        """Delegate to constraint manager"""
        self._constraint_manager.add_global_constraints(bnd_cond)

    def add_equation(
        self,
        equation: EquationBase,
        nodal_position: int | list[int] | tuple[int, ...],
    ):
        self._equation_registry.add_equation(equation, nodal_position)

    def remove_equation_type(self, *eq_child_class: Type[EquationBase]):
        """Delegate to equation registry"""
        self._equation_registry.remove_equation_type(*eq_child_class)

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

    def add_invariants(self, *invariants: tuple[str, ...]):
        """
        Each tuple in the list represent a set of variables that are treated
        as equal by the system. This is achieved by adding trivial residual
        equation to the system.
        Adding ('a', 'b', 'c') adds the equations
        a - b = 0
        a - c = 0
        b - c = 0
        """
        self._constraint_manager.add_invariants(*invariants)

    def add_spanwise_constants(self, *arguments: str):
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
        self._equation_registry.create_nodes()
        self.data.declared_arguments = self._equation_registry.build_argument_maps()

        self._equation_registry.add_analytical_eos()
        self._constraint_manager.write_to_nodes()
        (
            self.data.constraints,
            self.data.constraints_values,
        ) = self._constraint_manager.extract_constraints()

        # Arguments manipulation
        self.data.free_args = self._argument_resolver.identify_free_arguments()

        # Validity checks and scaling
        self._scaling_manager.extract_args_units()
        self._scaling_manager.check_equations_units()

        self.data.built = True
        logger.info('System assembled successfully')

    def _check_built(self) -> None:
        """Check that the system is flagged as built"""
        if not self.data.built:
            raise RuntimeError(
                'The system is not built or failed to do so, build it `build()`'
            )

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

    def solution_to_dict(
        self, solution_values: list[NDArray] | NDArray
    ) -> dict[str, NDArray]:
        """
        Simple utility method for mapping some argument values to a dict pointing
        to which argument they belong, useful for passing information between
        systems for initialization
        """
        return self._solution_dispatcher.solution_to_dict(
            solution_values, self.free_args_scaling
        )

    def write_solution_to_nodes(self, solution_values: NDArray):
        """
        Dispatch the provided values to the correct node and return
        the arguments in absolute indices
        """
        self._check_built()

        return self._solution_dispatcher.write_solution_to_nodes(
            solution_values, self.free_args_scaling
        )

    def to_symbolic(self) -> tuple[sp.Expr, ...]:
        """
        Convert the symbolic representation of each equation
        into absolute arguments
        """
        # Symbolic equation in the relative indices
        sym_eqs_rel = []
        for eq in self.equations.keys():
            sym_eqs_rel.append(eq.to_symbolic())

        # Symbolic equation in the absolute indices
        sym_eqs_abs = []
        for s_eq, eq in zip(sym_eqs_rel, self.equations.keys()):
            kwarg_map = self._arg_maps[eq]
            # Create a map of Relative -> Absolute arguments
            arg_map = {arg: kwarg_map[arg] for arg in eq.arguments}

            # Convert to symbols (not strictly necessary, but suggested)
            arg_map = {sp.symbols(k): sp.symbols(v) for k, v in arg_map.items()}

            if isinstance(s_eq, sp.Expr):
                sym_eqs_abs.append(s_eq.subs(arg_map))
            elif isinstance(s_eq, (list, tuple)):
                for expr in s_eq:
                    abs_expression = expr.subs(arg_map)
                    sym_eqs_abs.append(abs_expression)
            elif isinstance(s_eq, str):
                sym_eqs_abs.append(s_eq)
            else:
                raise TypeError(
                    f'Unknown type received while converting {self} to symbolic'
                )

        sym_eqs_abs = jax.tree.leaves(sym_eqs_abs)

        return tuple(sym_eqs_abs)

    @abstractmethod
    def make_residual_function(self):
        self._check_built()

    def get_scaled_constraints(self) -> list[NDArray]:
        return jax.tree.map(
            lambda x, y: x / y,
            self.constraints_values,
            self.constraints_scaling,
        )

    def get_initial_guess(
        self, manual_values: Mapping[str, ArrayLike] = {}
    ) -> list[NDArray]:
        """Generate initial guesses for free arguments"""
        guesses = []
        scaling = self.free_args_scaling

        for idx, arg in enumerate(self.data.free_args):
            arg_type = get_arg_type(arg)
            arg_state = get_arg_state(arg)

            # If there is a manual value, overwrite the guess registry
            if arg in manual_values:
                guess_value = manual_values[arg]
                logger.debug(f'Using manual value {guess_value} for {arg}')

            # If a guess is available in the registry
            elif arg_type in _guess_reg:
                guess_value = _guess_reg[arg_type]

                # Vary the total and static values, avoid singularities
                if arg_state == 'stc':
                    guess_value *= 0.95
                elif arg_state in ('rlt', 'tot'):
                    guess_value *= 1.05
                else:
                    pass

            # If there is no guess and no manual value
            else:
                # If the registry has defined a fallback, use that
                if _guess_reg._fallback_value:
                    guess_value = _guess_reg.get(arg_type)
                # Otherwise ask for user input
                else:
                    input_msg = f'INPUT >>> DIMENSIONAL guess for {arg} [1.0] = '
                    guess_value = float(input(input_msg) or 1.0)

                    # Add the input value to the registry
                    _guess_reg[arg_type] = guess_value

            guess_value = np.atleast_1d(guess_value)

            if max(guess_value.shape) != self.data.num_span:
                if max(guess_value.shape) != 1:
                    logger.warning(
                        f'Length mismatch in guess for {arg},'
                        f' using the first element as guess'
                    )
                    guess_value = guess_value[0]

                if arg not in self.data.scalar_arguments:
                    guess_value = np.repeat(guess_value, self.data.num_span)

            # Scale
            scaling_factor = scaling[idx]
            guess_value_scaled = guess_value / scaling_factor
            guesses.append(guess_value_scaled)

        return guesses


class CasadiSystem(SystemAssembler):
    def __init__(self, num_span: int = 1, *, scale_suffix: str = '__SCALER') -> None:
        super().__init__(num_span)
        self.scale_suffix = scale_suffix

    def _reset_symbols(self):
        logger.debug('Resetting all CasADi symbolics...')
        self.const_sym: list[cs.MX] = []
        self.free_args_sym: list[cs.MX] = []

        self.scales_const_sym: list[cs.MX] = []
        self.scales_free_args_sym: list[cs.MX] = []

        self._eq_scales_sym: list[cs.MX] = []
        self._res_expr_scaled: list[cs.MX] = []

    def build(self, scaled: bool = True, throw: bool = True):
        super().build(scaled)
        logger.info('Building CasADi backend...')
        self._reset_symbols()
        self._build_base_symbols()
        self._build_composed_symbols()
        self._build_residual_expressions()

    def _create_symbols(
        self, arg_names: Sequence[str], num_span: int, scale_suffix: str
    ) -> tuple[list[cs.MX], list[cs.MX]]:
        """Helper to create symbols and their scaled versions."""
        symbols = []
        scales = []

        for arg in arg_names:
            symbol_length = 1 if arg in self._scalar_arguments else num_span
            symbols.append(cs.MX.sym(arg, symbol_length))  # pyright:ignore
            # -> Choice: Scales are all single dimensional
            scales.append(cs.MX.sym(arg + scale_suffix, 1))  # pyright:ignore

        return symbols, scales

    def _build_base_symbols(self):
        """
        Convert the free arguments and constraints of the system to
        lists of symbols, together with a symbolic representation of
        their scaling values
        """

        # Create symbols
        self.free_args_sym, self.scales_free_args_sym = self._create_symbols(
            self.free_args,
            self.num_span,
            self.scale_suffix,
        )
        self.const_sym, self.scales_const_sym = self._create_symbols(
            self.constraints,
            self.num_span,
            self.scale_suffix,
        )

    def _build_equations_of_state(
        self, all_args_products: dict[str, cs.MX]
    ) -> dict[str, cs.MX]:
        fl_model = self.fluid_settings.model

        if not isinstance(fl_model, ExternalFluidModel):
            raise NotImplementedError

        self._eos_callbacks: list[CasadiEos] = []
        self._eos_factory = CasadiEosFactory(fl_model)

        # Get discarded thermo args => They become eos outputs
        discarded_thermo = self._argument_resolver.get_discarded_thermo_args()

        # Add inter-node eos
        for eq in self.equations:
            if eq.input_pair:
                if eq._eos and eq.eos._num_span == self.num_span:
                    continue
                eq.eos = self._eos_factory.make_eos(
                    eq.input_pair,
                    eq.output_quantities,
                    self.num_span,
                    f'multi_{eq.__class__.__name__}',
                )

        out_syms = {}
        for node_idx, discarded_vars in discarded_thermo.items():
            node_inp_pairs = self.nodes[node_idx].get_update_variables()

            for state_name, out_props in discarded_vars.items():
                pair_tuple = node_inp_pairs[state_name]
                pair_id = pair_id_from_tuple(pair_tuple)
                sorted_pair_tuple = pair_based_sorting(*pair_tuple)

                casadi_eos_cb = self._eos_factory.make_eos(
                    pair_id,
                    out_props,
                    self.num_span,
                    f'nodeCb_{state_name}_N{node_idx}',
                )

                # This is to keep references alive
                self._eos_callbacks.append(casadi_eos_cb)

                # Symbolic representation of the input pair properties
                symbolic_pair = [
                    all_args_products[f'{state_name}_{var}{node_idx}']
                    for var in sorted_pair_tuple
                ]

                # Symbolic representation of the output properties
                out_props_syms = casadi_eos_cb(*symbolic_pair)

                if not isinstance(out_props_syms, tuple):
                    out_props_syms = [out_props_syms]

                # Make a dictionary of symbols that are ouputs from the eos callbacks
                for pr_name, pr_sym in zip(out_props, out_props_syms):
                    out_syms[f'{state_name}_{pr_name}{node_idx}'] = pr_sym

        return out_syms

    def _build_composed_symbols(self):
        """
        Loop through the system's equations giving as arguments symbolic
        MX representation of each argument, mapped from the relative equation
        indices to the absolute system indices.
        """
        # === Build the product of each symbolic variable for their scaling value
        # 1. For free arguments
        free_args_products: dict[str, cs.MX] = {
            sym.name(): sym * scale
            for sym, scale in zip(self.free_args_sym, self.scales_free_args_sym)
        }

        # 2. For constrained arguments
        constraints_products: dict[str, cs.MX] = {
            sym.name(): sym * scale
            for sym, scale in zip(self.const_sym, self.scales_const_sym)
        }

        all_args_products = {**free_args_products, **constraints_products}

        # === Build equation of state symbols
        casadi_eos_symbols = self._build_equations_of_state(all_args_products)

        self._all_symbols = {**all_args_products, **casadi_eos_symbols}

    def _build_invariant_expressions(self) -> list[cs.MX]:
        """
        Build expressions for constant quantities, either across nodes
        or across components
        """
        invariant_expressions = []
        for args in self._invariants:
            for arg_couples in combinations(args, 2):
                # If both argument do not appear in the equations, skip to next couple
                # if one of them is unused by other eqns. it is useless to add it
                if not set(arg_couples).issubset(self._all_symbols):
                    continue

                sym0 = self._all_symbols[arg_couples[0]]
                sym1 = self._all_symbols[arg_couples[1]]

                # NOTE: We don't care about scaling, they
                # are just identities, either on scaled or unscaled vars
                invariant_expressions.append(sym1 - sym0)

        return invariant_expressions

    def _build_spanwise_constants(self) -> list[cs.MX]:
        spanwise_expressions = []
        for arg in self._spanwise_constants:
            if arg not in self._all_symbols:
                continue

            arg_sym = self._all_symbols[arg]

            if max(arg_sym.shape) == 1:
                continue
            else:
                expression = arg_sym[1:] - arg_sym[:-1]
                spanwise_expressions.append(expression)

        return spanwise_expressions

    def _build_residual_expressions(self):
        # Build scaling symbols for all equations
        self._eq_scales_sym = [
            cs.MX.sym(f'eq{idx}{self.scale_suffix}', 1)  # pyright:ignore
            for idx in range(self.num_equations)
        ]

        # Build and concatenate residual equations
        logger.info('Building residual equation symbolics (this may take a while)...')

        residuals = []
        for eq in self.equations:
            kwmap = self._arg_maps[eq]  # Convert to abs args
            args = [self._all_symbols[kwmap[k]] for k in eq.arguments]

            # NOTE: No need to override the operators for now,
            # just use numpy operations in a way compatible with
            # casadi symbolics

            # overridden_eq = override_operators(eq.residual, 'numpy', cs)
            overridden_eq = eq.residual
            residuals.append(overridden_eq(*args))

        # Divide each residual expression by its scaling symbol
        self._res_expr_scaled = list(
            map(
                lambda X, Y: X / Y,
                jax.tree.leaves(residuals),
                self._eq_scales_sym,
            )
        )

        self._res_expr_scaled += self._build_invariant_expressions()
        self._res_expr_scaled += self._build_spanwise_constants()

        num_vars = max(cs.vertcat(*self.free_args_sym).shape)
        num_residuals = max(cs.vertcat(*self._res_expr_scaled).shape)
        logger.info(
            f'System info: {num_vars} total variables, {num_residuals} total equations'
        )

        if num_vars != num_residuals:
            input(
                f'*** WARNING: Mismatch in number of equations {num_residuals}'
                f' and variables {num_vars}, press ENTER to continue anyway'
            )

    def make_residual_function(self):
        """
        Build a function object for the full residual, with the scaling
        factors as arguments, and create an MX expression for the full
        function with the actual scaling values assigned
        """
        super().make_residual_function()

        FULL_ARGUMENTS = [
            self.free_args_sym,
            self.scales_free_args_sym,
            self.const_sym,
            self.scales_const_sym,
            self._eq_scales_sym,
        ]

        full_arguments_cat = [cs.vertcat(*args) for args in FULL_ARGUMENTS]

        # Symbolic free arguments and constraints
        free_args = full_arguments_cat[0]
        constraints = full_arguments_cat[2]

        res_full_func = cs.Function(
            'res_full_func', full_arguments_cat, [cs.vertcat(*self._res_expr_scaled)]
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
        root_method: Literal['newton', 'ipopt', 'kinsol'],
        opts={},
    ) -> Callable[[ArrayLike, ArrayLike], cs.DM]:
        """
        Create a rootfinder callable object that takes as a first input the
        initial guess and as second input the values of the constraints and
        returns the solution.
        """
        res_func = self.make_residual_function()

        free_args_symbols = cs.vertcat(*self.free_args_sym)
        constraints_symbols = cs.vertcat(*self.const_sym)

        res_expr_partial = res_func(
            free_args_symbols,
            constraints_symbols,
        )

        rootfind_problem = {
            'x': free_args_symbols,
            'p': constraints_symbols,
            'g': res_expr_partial,
        }

        # TODO: remove hardcoded options
        match root_method:
            case 'newton':
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

            case 'ipopt':
                # IPOPT solver
                rootfinder = cs.nlpsol(
                    'ipopt_rootfinder',
                    'ipopt',
                    rootfind_problem,
                    {
                        'error_on_fail': True,
                        # Reasonable defaults for IPOPT, overwritten by user
                        'ipopt.print_level': 5,
                        'ipopt.max_iter': 3000,
                        'ipopt.tol': 1e-12,
                        # NOTE: Superseeded by new implementations, thermo
                        # derivatives available up to the 3rd order (null)
                        # 'ipopt.hessian_approximation': 'limited-memory',
                        **opts,
                    },
                )
            case 'kinsol':
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

    def get_arguments_bounds(self):
        lbx = []
        ubx = []
        for arg, scaling in zip(self.free_args_sym, self.free_args_scaling):
            arg_type = get_arg_type(arg.name())

            if arg_type not in _bounds_reg:
                lower_bound = -1e20
                upper_bound = 1e20
            else:
                lower_bound, upper_bound = _bounds_reg.get(arg_type)

            arg_size = max(arg.shape)

            lbx += arg_size * [lower_bound / scaling]
            ubx += arg_size * [upper_bound / scaling]

        return cs.vertcat(*lbx), cs.vertcat(*ubx)

    def write_solution_to_nodes(self, solution_values: NDArray):
        solution_dict = super().write_solution_to_nodes(solution_values)

        for eos_cb in self._eos_callbacks:
            eos_name: str = eos_cb.name()
            if not eos_name.startswith('nodeCb'):
                continue

            eos_name_fields = eos_name.split('_')

            state_id = eos_name_fields[1]
            state_id = cast(NodeStatesNames, state_id)

            node_idx = get_index(eos_name_fields[2])
            input_args = pair_tuple_from_id(eos_cb._input_pair)

            state_obj = self.nodes[node_idx].fetch_state(state_id)

            inputs = [
                state_obj.get(arg).to_base_units().magnitude for arg in input_args
            ]
            output_values = eos_cb(*inputs)

            thermo_dict = {}
            for prop, val in zip(eos_cb._output_props, output_values):
                thermo_dict[f'{state_id}_{prop}{node_idx}'] = val.toarray().flatten()

            self.nodes[node_idx].write_to_node(thermo_dict, False)

            solution_dict.update(thermo_dict)

        return solution_dict

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
        super().__init__(num_span)

    def _build_stack_composer(self):
        """
        Build a function that merges the stack of free arguments
        of shape (# free args, # span stations) to a stack of constraints
        of shape shape (# constraints, # span station).
        The full stack is then fed to a residual function.
        """

        all_args = self._declared_arguments

        free_args = self.free_args
        declared_constraints = self.constraints

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
            for eq in self.equations
        ]

    def _get_residual_positions(self):
        """
        Get where residual equations are placed in the stack
        of residuals
        """
        curr_index = 0
        residual_indices = []
        for eq in self.equations:
            residual_indices.append(
                tuple(curr_index + j for j in range(eq.num_equations))
            )
            curr_index += eq.num_equations

        return residual_indices

    def _get_equation_lines(self, residuals_name: str):
        """
        Make the equation lines, each writing the residuals
        array at the correct residual indices
        """
        residual_indices = self._get_residual_positions()
        eq_lines = []
        for idx, eq in enumerate(self.equations):
            kwmap = self._arg_maps[eq]

            mapped_args = [kwmap[arg] for arg in eq.arguments]

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
            DIVIDER.join(f'eq{idx}' for idx, _ in enumerate(self.equations)) + DIVIDER
        )
        eq_lines = self._get_equation_lines(residuals_name)
        eq_stack = '\n    '.join(eq_lines)

        codegen = f"""
def {func_name}(equations, {', '.join(self._declared_arguments)}):
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
        num_args = len(self.free_args)
        num_const = len(self.constraints)

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


def solve_root_problem(
    rootfinder: Any,
    guess: list[NDArray],
    knowns: list[NDArray],
    arg_bounds: tuple[cs.DM, cs.DM] | None = None,
    suppress_output: bool = True,
    perturbate: bool = False,
    delta_pert: float = 0.05,
):
    if suppress_output:
        output_manipulator = output_suppression
    else:
        output_manipulator = dummy_context
    """Simple utility function for solving rootfinding problems"""

    with output_manipulator():
        logger.info('Solving the system...')

        def perturb_func(x):
            return x + x * delta_pert * (-1 + 2 * np.random.ranf(1))

        if perturbate:
            guess = jax.tree.map(perturb_func, guess)

        guess_cat = np.concatenate(guess)
        knowns_cat = np.concatenate(knowns)

        extra_args = {}
        if rootfinder.n_in() > 2:
            extra_args.update({'lbg': 0, 'ubg': 0})
            if arg_bounds:
                extra_args['lbx'], extra_args['ubx'] = arg_bounds

        sol = rootfinder(
            x0=guess_cat,
            p=knowns_cat,
            **extra_args,
        )

        if isinstance(sol, dict):
            sol = sol['x']

        return np.array(sol)
