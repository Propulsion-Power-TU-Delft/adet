"""
Module that contains the tools to define equations, build systems made
up of assembly of such residual equations and routines to generate reasonable
initial guesses based on the available thermodynamic, geometric and kinematic
data.
"""

from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from copy import deepcopy
import logging
from typing import Callable, Self, Sequence, Type

from numpy.typing import NDArray
from pint import Quantity

import numpy as np
from pint.facets.plain import PlainQuantity
import sympy as sp
import casadi as cs

import jax as jax
import jax.numpy as jnp

from adet.errors import ConstraintError, ExistingEquationError
from adet.fluid.casadi_eos import CasadiEoS
from adet.fluid.settings import (
    AnalyticalFluidModel,
    EmptyFluidModel,
    ExternalFluidModel,
    FluidSettings,
)
from adet.registries import GuessRegistry, ScalingRegistry
from adet.tools.coolprop_utils import (
    pair_based_sorting,
    pair_id_from_name,
    pair_name_from_tuple,
)
from adet.tools.strings import get_arg_state, rm_digits, get_index, get_arg_type
from adet.equations import EquationBase
from adet.node import FlowNode
from adet.constants import NodeStatesNames, ArrayLike
from adet.tools.context import override_operators


logger = logging.getLogger(__name__)


def get_units_string(var):
    return str(var.to_base_units().units)


_scale_reg = ScalingRegistry()
_guess_reg = GuessRegistry()


class SystemAssembler(ABC):
    """
    Class for assembling a system of equations, gathering its arguments
    and returning the residuals dispatched to the equations it is made
    out of
    """

    # TODO: This class is quite heavy, it is probably a good idea
    # to break it down into more manageable components, for now
    # it is fine

    def __init__(self, spanwise_stations: int) -> None:
        # Initialize with empty settings
        self._fluid_settings = FluidSettings(EmptyFluidModel())

        self.spanwise_stations = spanwise_stations
        self.equations: OrderedDict[EquationBase, tuple[int, ...]] = OrderedDict()
        """
        All the equations, in the form of a dictionary:
            {
                EquationBase: Instance of the equation/system of eqns
                tuple[int, ...]: absolute nodes involved in the equation
            }
        """

        self._eos_equations: list[EquationBase] = []

        self.nodes: tuple[FlowNode, ...] = tuple()
        """All the nodes called by the equations"""

        self._declared_arguments: tuple[str, ...] = tuple()
        """All the arguments called by the equations, unfiltered"""

        self.free_args: tuple[str, ...] = tuple()
        """The arguments modified during quasi-Newton iterations"""

        self.constraints: tuple[str, ...] = tuple()
        """All the constraints"""

        self._global_constraints: defaultdict[
            NodeStatesNames, dict[str, ArrayLike | PlainQuantity]
        ] = defaultdict(dict)

        self._arg_maps: dict[EquationBase, dict[str, str]] = {}
        """
        Mapping of keyword arguments to move between relative and absolute arguments in
        each equation:
            {
                EquationBase: Instance of the equation
                dict[str, str]: { Relative argument: Absolute argument }
            }
        """

        self.constraints_values: NDArray = np.array([])
        """All the constraints defined by the FlowNodes"""

        self.boundary_conditions: defaultdict[
            int,
            defaultdict[
                NodeStatesNames,
                dict[str, ArrayLike | PlainQuantity],
            ],
        ] = defaultdict(lambda: defaultdict(dict))

        self._equations_units: list[list[str]] = []
        """
        Units of the equation, with the output equation structure
        """

        self._arguments_units: dict[str, str] = {}
        """ Units of all the free arguments """

        self._built: bool = False
        self._scaled: bool = False

    @property
    def fluid_settings(self):
        return self._fluid_settings

    @fluid_settings.setter
    def fluid_settings(self, settings: FluidSettings) -> None:
        self._fluid_settings = settings

    @property
    def num_equations(self):
        return sum(eq.num_equations for eq in self.equations)

    def reset(self) -> None:
        old_settings = self._fluid_settings
        self.__init__(self.spanwise_stations)
        self.fluid_settings = old_settings

    def copy(self) -> Self:
        new_instance = self.__class__(self.spanwise_stations)
        new_instance.from_dict(self.to_dict())
        # Remove EoS equaitons
        return new_instance

    def to_dict(self):
        FIELDS_TO_SAVE = [
            'equations',
            'boundary_conditions',
            '_fluid_settings',
            '_global_constraints',
        ]

        out_dict = {}
        for field in FIELDS_TO_SAVE:
            try:
                attr = getattr(self, field)
            except AttributeError as e:
                raise AttributeError(
                    f'{e} encountered while trying to deepcopy attribute {attr}'
                )

            out_dict[field] = deepcopy(attr)

        return out_dict

    def from_dict(self, data_dict):
        for attr, value in data_dict.items():
            setattr(self, attr, value)

    def add_boundary_conditions(self, bnd_cond: dict, node_idx: int):
        for state_id, state_bnd_cond in bnd_cond.items():
            self.boundary_conditions[node_idx][state_id].update(state_bnd_cond)

    def add_global_constraints(self, bnd_cond: dict):
        self._global_constraints.update(bnd_cond)

    def _write_bc_to_nodes(self):
        """Write the stored boundary conditions to the nodes"""
        logger.debug('Writing boundary conditions to nodes...')

        def to_base_units(var) -> NDArray:
            if not isinstance(var, PlainQuantity):
                mag = var
            else:
                mag = var.to_base_units().magnitude

            return np.atleast_1d(mag)

        for node_idx, node in enumerate(self.nodes):
            # Add virtual constraints (e.g. cp, cv for ideal gas)
            self.add_boundary_conditions(self._global_constraints, node_idx)

            # Convert to base units arrays
            bc_arrays = jax.tree.map(to_base_units, self.boundary_conditions[node_idx])

            for state_id, constraints in bc_arrays.items():
                state_obj = node.fetch_state(state_id)
                # Add them to the node
                for var, val in constraints.items():
                    state_obj.set_value(var, val)
                    state_obj.change_status(var, fixed=True)

    def add_equation(
        self,
        equation: EquationBase,
        nodal_position: int | list[int] | tuple[int, ...],
    ):
        """
        Add an equation to the system in the specified position.
        Node order is preserved (0,1) != (1,0)

        Parameters
        ----------------------------------------------
        equation: EquationBase
            instance of the equation to add
        nodal_position: int | tuple[int, ...]
            Where in the global system this equation should be added
        """
        if isinstance(nodal_position, int):
            nodal_position = [nodal_position]

        # 1. Check that the provided position has the same length as the
        # equation arguments
        local_indices = {get_index(arg) for arg in equation.arguments}
        if len(local_indices) != len(nodal_position):
            raise ValueError(
                f'Detected indices of in the definition of '
                f'`{equation.__class__.__name__}` {tuple(local_indices)} '
                f'is not equal to length of the prescribed '
                f'absolute nodal position {nodal_position}'
            )

        # 2. Check that an equation of the same type does not exist at the same location
        for eq_instance, eq_nodes in self.equations.items():
            if isinstance(eq_instance, equation.__class__) and (
                set(eq_nodes) == set(nodal_position)
            ):
                raise ExistingEquationError(
                    f'Duplicate equation entry for {equation.__class__.__name__}'
                    f' at position {nodal_position}'
                )

        self.equations[equation] = tuple(nodal_position)

        logger.debug(
            f'Added equation {equation.__class__.__name__} to system {id(self)}'
            f' in position {nodal_position}'
        )

    def remove_equation_type(self, *eq_child_class: Type[EquationBase]):
        cleaned_equations = self.equations.copy()
        for eq in self.equations:
            if isinstance(eq, eq_child_class):
                cleaned_equations.pop(eq)

        self.equations = cleaned_equations

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
        if isinstance(nodal_position, int):
            nodal_position = tuple([nodal_position])

        for eq_instance, eq_position in self.equations.copy().items():
            equation_found = False

            if isinstance(eq_instance, equation_class) and (
                set(eq_position) == set(nodal_position)
            ):
                equation_found = True
                self.equations.pop(eq_instance)

        if not equation_found:
            logger.warning(
                f'No equation {equation_class} found to remove in {nodal_position}'
            )

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
            self._scaled = True

        self._create_nodes()
        self._add_analytical_eos()
        self._write_bc_to_nodes()
        self.constraints, self.constraints_values = self._get_constraints()

        # Arguments manipulation
        self._declared_arguments = self._ingest_eqs_arguments()
        self.free_args = self._identify_free_arguments()

        # Validity checks and scaling
        self._get_args_units()
        self._check_equations_units()

        # TODO: Well posedness check suspended for now
        # >>> self._check_well_posedness(throw)
        # Reasons:
        # 1. Difficult to make consistent with dynamic argument choice of update pairs
        # 2. Solvers throw an error for shape mismatch anyway, although more cryptic

        self._built = True
        logger.info('System assembled succesfully')

    def _create_nodes(self):
        """
        Create nodes based on the nodal positions of the equations
        defined by the user
        """
        logger.debug('Creating nodes for the equation system')

        node_indices = set()
        for nodal_pos in self.equations.values():
            node_indices.update(nodal_pos)

        if not node_indices:
            raise RuntimeError(
                'Cannot build an empty system! Please add some equations'
            )

        if min(node_indices) > 0:
            logger.warning(
                f'Minimum node index for equation system {self} is '
                f'greater than 0, consider shifiting your nodal positions. '
                f'This may produce unexpected behaviours.'
            )

        self.nodes = tuple(
            FlowNode(self._fluid_settings, self.spanwise_stations)
            for _ in range(0, 1 + max(node_indices))
        )

        logger.debug(f'Successfully created {len(self.nodes)} nodes')

    def _add_analytical_eos(self):
        """
        Add analytical equations of state to the system
        if the model calls for it. They are removed when
        copied
        """

        fl_model = self.fluid_settings.model

        # Equations of state
        if isinstance(fl_model, AnalyticalFluidModel):
            for node_idx, _ in enumerate(self.nodes):
                for eq in fl_model.get_equations():
                    try:
                        self.add_equation(eq, node_idx)
                        logger.debug(
                            f'Added EoS equation {eq.__class__.__name__} at {node_idx}'
                        )
                        self._eos_equations.append(eq)
                    except ExistingEquationError:
                        pass

    def _get_constraints(self) -> tuple[tuple[str, ...], NDArray]:
        """
        Get all the constraints defined by the nodes taking part in
        the system of equations
        """
        constraint_names = []
        constraint_values = []

        for node_idx, node in enumerate(self.nodes):
            for key, value in node.get_constraints().items():
                constraint_names.append(key + str(node_idx))
                constraint_values.append(value.to_base_units().magnitude)

        if constraint_values:
            const_values_ret = np.vstack(constraint_values)
        else:
            # Catch the case with no constraints
            const_values_ret = np.array([])

        return tuple(constraint_names), const_values_ret

    def _ingest_eqs_arguments(self) -> tuple[str, ...]:
        """
        Get all the available arguments and assign them the correct absolute
        index.
        This also create the keyword argument map for each equation, which
        maps the relative argument of that equation to the absolute system arguments.
        """
        system_arguments = []

        logger.debug('Reading all the equation arguments...')

        for eq, eq_position in self.equations.items():
            local_indices: set[int] = {get_index(arg) for arg in eq.arguments}

            index_map = {
                rel_pos: abs_pos
                for abs_pos, rel_pos in zip(eq_position, sorted(local_indices))
            }
            self._arg_maps[eq] = {}

            for arg in eq.arguments:
                arg_rel_idx = get_index(arg)
                arg_abs_idx = index_map[arg_rel_idx]

                arg_no_digit = rm_digits(arg)

                system_arguments.append(
                    arg_no_digit + str(eq_position[arg_rel_idx]),
                )

                # Create the variable in the node
                self.nodes[arg_abs_idx].create_vars(arg_no_digit)
                self._arg_maps[eq][arg] = arg_no_digit + str(arg_abs_idx)

        system_arguments = sorted(set(system_arguments))

        logger.debug(f'Arguments detected are: {", ".join(system_arguments)}')

        return tuple(system_arguments)

    def _identify_free_arguments(self) -> tuple[str, ...]:
        """
        Get the real thermodynamic and kinematic arguments needed to complete the
        different states of the node.
        If an ideal gas state is detected, it is assumed that the velocity triangle
        and equations of state are solved in a coupled manner with the rest of
        the system
        """

        if isinstance(self._fluid_settings.model, AnalyticalFluidModel):
            # Just a difference between sets, sorted
            return tuple(
                sorted(
                    set(self._declared_arguments) - set(self.constraints),
                )
            )
        else:
            return tuple(
                sorted(self._get_effective_arguments()),
            )

    def _get_effective_arguments(self):
        """
        Get the thermo that act on thermodynamic state updates
        e.g. if hmass, smass, p, T appear on the same state,
        only two variables are effective (pure subtance + phase),
        while the other two are followers
        """
        # Non thermodynamic arguments
        nonthermo_args = [
            arg
            for arg in self._declared_arguments
            if not arg.startswith(('stc', 'tot', 'rlt'))
        ]

        # Get what variables will be used for state updates
        update_args = []
        for node_idx, node in enumerate(self.nodes):
            updt_vars = node.get_update_variables()

            for state, updt_pair in updt_vars.items():
                update_args += [f'{state}_{var}{node_idx}' for var in updt_pair]

        return set(update_args + nonthermo_args).difference(self.constraints)

    def _get_discarded_thermo_args(
        self,
    ) -> defaultdict[int, defaultdict[str, list[str]]]:
        """
        Retrieve the discarded thermo arguments a.k.a. the ones
        that were declared in the equations but have become extractions
        from the equations of state
        """
        all_discarded = (
            set(self._declared_arguments) - set(self.constraints) - set(self.free_args)
        )

        discarded = defaultdict(lambda: defaultdict(list))

        for arg in all_discarded:
            arg_state = get_arg_state(arg)
            arg_idx = get_index(arg)
            arg_type = get_arg_type(arg)

            discarded[arg_idx][arg_state].append(arg_type)

        return discarded

    def _check_built(self) -> None:
        """Check that the system is flagged as built"""
        if not self._built:
            raise RuntimeError(
                'The system is not built or failed to do so, build it `build()`'
            )

    def _check_well_posedness(self, throw: bool) -> None:
        num_args = len(self.free_args)
        if self.num_equations != num_args:
            arguments_str = '\n'.join(self.free_args)
            ERR_MSG = (
                'Badly posed system detected, the number of residual equations '
                + f'({self.num_equations}) is not equal to the number of system '
                f'arguments ({num_args}). '
                f'The free system arguments are:\n{arguments_str}'
            )

            if throw:
                raise ConstraintError(ERR_MSG)
            else:
                logger.warning(ERR_MSG)
                return

        logger.info(
            f'System is well posed, with {self.num_equations} equations '
            f'and {num_args} arguments'
        )

    def _get_args_units(self):
        """
        Check that the units of the equations are consistent, if not
        a dimensionality error is raised to the user
        """
        self._arguments_units = {}

        for idx, node in enumerate(self.nodes):
            node_arguments = {
                f'{arg}{idx}': var for arg, var in node.get_all_quantities().items()
            }

            self._arguments_units.update(jax.tree.map(get_units_string, node_arguments))
            self._arguments_units.update(
                {arg: get_units_string(var) for arg, var in node_arguments.items()}
            )

    def _check_equations_units(self):
        for eq, kwmap in self._arg_maps.items():
            self._equations_units.append(self._get_eq_units(eq, kwmap))

        logger.debug('Units for the residual equations succesfully verified')

    def _get_eq_units(self, equation: EquationBase, kwmap: dict[str, str]) -> list[str]:
        """Get units for a single equation"""
        if equation.skip_unit_check:
            return list(equation.manual_units)

        args = []
        for arg in equation.arguments:
            absolute_argument = kwmap[arg]
            units = self._arguments_units[absolute_argument]
            dummy_value = Quantity([np.nan], units)

            args.append(dummy_value)

        res = equation.residual(*args)

        if not isinstance(res, (list, tuple)):
            res = (res,)

        return [get_units_string(r) for r in res]

    def _assign_scaling_factor(self, units: str):
        if not self._scaled:
            factor = 1.0

        factor = _scale_reg[units]

        return factor

    @property
    def free_args_scaling(self):
        """Build multi-span scaling factors for arguments"""
        return self._argument_scaling_helper(self.free_args)

    @property
    def constraints_scaling(self):
        """Build multi-span scaling factors for constraints"""
        return self._argument_scaling_helper(self.constraints)

    def _argument_scaling_helper(self, arguments: Sequence[str]):
        """Returns the arguments scales for a sequence of system arguments"""
        if not self._scaled:
            scales = np.ones(len(arguments))
        else:
            all_args_scales = jax.tree.map(
                self._assign_scaling_factor,
                self._arguments_units,
            )

            scales = np.array([all_args_scales[arg] for arg in arguments])
        return np.tile(scales, (self.spanwise_stations, 1)).T

    @property
    def equations_indices(self):
        """
        Return a list of each single equations,
        splitting also multi-residual objects
        Needed to identify problematic equations
        """

        eq_identifiers = []
        for eq, pos in self.equations.items():
            eq_identifiers += [
                f'{eq.__class__.__name__} EQ#{idx} @NODES{pos}'
                for idx in range(eq.num_equations)
            ]

        return eq_identifiers

    @property
    def equations_scaling(self):
        """
        Build multi-span scaling factors for equations
        """

        eq_scales = []

        if not self._scaled:
            scales = np.ones(self.num_equations)
        else:
            for eq, units in zip(self.equations.keys(), self._equations_units):
                if eq.scaling_factor:
                    logging.debug(
                        f'Custom scaling factor found for {eq.__class__.__name__}, '
                        f'{eq.scaling_factor}'
                    )
                    eq_scales += eq.scaling_factor
                else:
                    eq_scales += [self._assign_scaling_factor(u) for u in units]

            scales = np.array(eq_scales)

        return np.tile(scales, (self.spanwise_stations, 1)).T

    def _split_arguments_by_node(self, arguments: Sequence[str]):
        """
        Split the arguments, in the form of <state>_<var_type><abs_node_idx>
        in their respective FlowNode, making them easier to read/write
        using the `FlowNode`'s method
        """
        node_by_args: dict[FlowNode, list[str]] = {node: [] for node in self.nodes}

        for arg in set(arguments):
            node = self.nodes[get_index(arg)]
            node_by_args[node].append(rm_digits(arg))

        return node_by_args

    def solution_to_dict(self, argument_values: NDArray) -> dict[str, NDArray]:
        """
        Simple utility method for mapping some argument values to a dict pointing
        to which argument they belong, useful for passing information between
        systems for initialization
        """

        argument_values = argument_values.reshape(
            len(self.free_args),
            self.spanwise_stations,
        )

        if self._scaled:
            argument_values = argument_values * self.free_args_scaling

        # Arguments in absolute indices
        return {arg: argument_values[idx] for idx, arg in enumerate(self.free_args)}

    def write_solution_to_nodes(self, free_argument_values: NDArray):
        """
        Dispatch the provided values to the correct node and return
        the arguments in absolute indices
        """

        free_argument_values = np.array(free_argument_values).reshape(
            len(self.free_args), -1
        )
        self._check_built()

        split_arg_dictionaries = self._split_arguments_by_node(self.free_args)

        args_values = self.solution_to_dict(free_argument_values)

        arg_print = ', '.join(args_values)
        logger.debug(f'Writing arguments to node: {arg_print}')

        for node, args_to_write in split_arg_dictionaries.items():
            node_idx = str(self.nodes.index(node))
            node.write_to_node(
                {arg: args_values[arg + node_idx] for arg in args_to_write},
                fixed=False,
            )

        return args_values

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

    def get_scaled_constraints(self) -> NDArray:
        return self.constraints_values / self.constraints_scaling

    def get_initial_guess(
        self, manual_values: dict[str, jax.Array] | None = None
    ) -> NDArray:
        self._check_built()

        guesses = []

        def span_expander(value):
            return value * np.ones(self.spanwise_stations)

        for idx, arg in enumerate(self.free_args):
            arg_type = get_arg_type(arg)
            arg_state = get_arg_state(arg)

            # If there is a manual value, overwrite the guess registry
            if manual_values is not None and arg in manual_values:
                guess_value = span_expander(manual_values[arg])
                logger.debug(f'Using manual value {guess_value} for {arg}')
            # Else, If a guess is available in the registry
            elif arg_type in _guess_reg:
                guess_value = span_expander(_guess_reg[arg_type])

                # Vary the total and static values, avoid singularities
                if arg_state == 'stc':
                    guess_value *= 0.95
                elif arg_state in ('rlt', 'tot'):
                    guess_value *= 1.05
                else:
                    pass

            # If there is no guess and no manual value
            else:
                # If the registry has defined a fallbak, use that
                if _guess_reg._fallback_value:
                    guess_value = _guess_reg.get(arg_type)
                # Otherwise ask for user input
                else:
                    input_msg = f'INPUT >>> DIMENSIONAL guess for {arg} [1.0] = '
                    guess_value = float(input(input_msg) or 1.0)

                    # Add the input value to the registry
                    _guess_reg[arg_type] = guess_value

            # Scale
            scaling_factor = self.free_args_scaling[idx]
            guess_value_scaled = guess_value / scaling_factor
            guesses.append(guess_value_scaled)

        return np.array(guesses)


class CasadiSystem(SystemAssembler):
    """
    Build a system using CasADi, good for CPU computations
    and code generation. Good direct integration with numpy
    """

    def __init__(
        self, spanwise_stations: int = 1, *, scale_suffix: str = '__SCALER'
    ) -> None:
        super().__init__(spanwise_stations)
        self.scale_suffix = scale_suffix

    def build(self, scaled: bool, throw: bool = True):
        super().build(scaled)
        logger.info('Building CasADi backend')
        self._build_base_symbols()
        self._build_composed_symbols()
        self._build_residual_expressions()

    @staticmethod
    def _create_symbols(names: Sequence[str], num_span: int, scale_suffix: str):
        """Helper to create symbols and their scaled versions."""
        symbols = [cs.SX.sym(name, num_span) for name in names]  # pyright:ignore
        scales = [cs.SX.sym(name + scale_suffix, num_span) for name in names]  # pyright:ignore
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
            self.spanwise_stations,
            self.scale_suffix,
        )
        self.const_sym, self.scales_const_sym = self._create_symbols(
            self.constraints,
            self.spanwise_stations,
            self.scale_suffix,
        )

    def _build_equations_of_state(
        self, all_args_products: dict[str, cs.SX]
    ) -> dict[str, cs.SX]:
        fl_model = self._fluid_settings.model

        if not isinstance(fl_model, ExternalFluidModel):
            return {}

        eos_obj = fl_model.eos_object

        self._casadi_eos_callbacks = []

        discarded_thermo = self._get_discarded_thermo_args()

        out_syms = {}
        for node_idx, discarded_vars in discarded_thermo.items():
            node_inp_pairs = self.nodes[node_idx].get_update_variables()

            for state_name, out_props in discarded_vars.items():
                pair_tuple = node_inp_pairs[state_name]

                pair_name = pair_name_from_tuple(pair_tuple)
                pair_id = pair_id_from_name(pair_name)

                casadi_eos_cb = CasadiEoS(
                    f'casadi_eos_{state_name}{node_idx}',
                    eos_obj,
                    pair_id,
                    out_props,
                    self.spanwise_stations,
                )

                # This is to keep references
                self._casadi_eos_callbacks.append(casadi_eos_cb)

                sorted_pair_tuple = pair_based_sorting(*pair_tuple)

                symbolic_pair = [
                    all_args_products[f'{state_name}_{var}{node_idx}']
                    for var in sorted_pair_tuple
                ]

                # Symbolic representation of the casadi_eos
                out_props_syms = casadi_eos_cb(*symbolic_pair)

                if not isinstance(out_props_syms, tuple):
                    out_props_syms = [out_props_syms]

                for pr_name, pr_sym in zip(out_props, out_props_syms):
                    out_syms[f'{state_name}_{pr_name}{node_idx}'] = pr_sym

        return out_syms

    def _build_composed_symbols(self):
        """
        Loop through the system's equations giving as arguments symbolic
        MX representation of each argument, mapped from the relative equation
        indices to the absolute system indices.
        """
        # Build the product of each symbolic variable for their scaling value
        free_args_products: dict[str, cs.SX] = {
            sym.name(): sym * scale
            for sym, scale in zip(self.free_args_sym, self.scales_free_args_sym)
        }

        constraints_products: dict[str, cs.SX] = {
            sym.name(): sym * scale
            for sym, scale in zip(self.const_sym, self.scales_const_sym)
        }

        all_args_products = {**free_args_products, **constraints_products}
        casadi_eos_symbols = self._build_equations_of_state(all_args_products)

        self._all_symbols = {**all_args_products, **casadi_eos_symbols}

    def _build_residual_expressions(self):
        num_span = self.spanwise_stations

        # Build scaling symbols for all equations
        self._eq_scales_sym = [
            cs.SX.sym(f'eq{idx}{self.scale_suffix}', num_span)  # pyright:ignore
            for idx in range(self.num_equations)
        ]

        logger.info(
            f'System info: {num_span * len(self.free_args)} total variables, '
            f'{num_span * self.num_equations} total equations'
        )
        logger.info('Building residual equation symbolics (this may take a while)...')
        # Build and concatenate residual equations
        # (no need to override numpy)
        residuals = []
        for eq in self.equations:
            kwmap = self._arg_maps[eq]  # Convert to abs args
            args = [self._all_symbols[kwmap[k]] for k in eq.arguments]

            overridden_eq = override_operators(eq.residual, 'numpy', cs)
            residuals.append(overridden_eq(*args))

        self._res_expr_scaled = list(
            map(
                lambda X, Y: X / Y,
                jax.tree.leaves(residuals),
                self._eq_scales_sym,
            )
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
        free_args_scaling_values = self.free_args_scaling.flatten()
        const_scaling_values = self.constraints_scaling.flatten()
        eq_scaling_values = self.equations_scaling.flatten()

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

    def to_jax(self):
        """
        Convert system to casadi
        """
        jax_sys = JaxSystem(self.spanwise_stations)
        jax_sys.from_dict(self.to_dict())
        return jax_sys


class JaxSystem(SystemAssembler):
    """
    Build a jax compatible assembled system, with stacked arguments input shapes,
    no need for flat vectors.
    """

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
    {residuals_name} = jnp.zeros({(self.num_equations, self.spanwise_stations)})

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
