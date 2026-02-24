from abc import ABC
import inspect
from collections import defaultdict
import logging
from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias, Type, Any

from adet.constants import ArrayLike
from pint.facets.plain import PlainQuantity

from adet.assembly import CasadiSystem
from adet.equations import EquationBase, UniqueEquation
from adet.equations.base_equation import LossApplier
from adet.node import FlowNode
from adet.tools.iter import ensure_tuple
from adet.tools.strings import get_arg_specs, get_arg_state, get_arg_type

if TYPE_CHECKING:
    from adet.assembly import CasadiSystem
    from adet.components.network import ComponentNetwork

BaseEquationsFormat: TypeAlias = list[
    tuple[
        Type[EquationBase],
        tuple[int, ...] | int,
    ]
]

logger = logging.getLogger(__name__)


class BaseComponent(ABC):
    # I use a tuple because EquationBase
    # is not hashable
    base_equations: ClassVar[
        list[
            tuple[
                Type[EquationBase],
                int | tuple[int, ...],
            ]
        ]
    ]
    """
    Base equations that link the inlet and outlet nodes
    of a component
    """

    from_previous_node: ClassVar[list[str]] = []
    """
    Variables that are inherited from the previous node
    """

    constant_variables: ClassVar[list[str]] = []
    """
    Variables that are treated as invariant between inlet
    and outlet
    """

    def __init__(
        self,
        name: str,
        inlet_bc: dict[
            str,
            dict[str, Any],
        ] = {},
        outlet_bc: dict[
            str,
            dict[str, Any],
        ] = {},
        extra_equations: dict[
            EquationBase,
            int | tuple[int, ...],
        ] = {},
        from_previous_node: list[str] = [],
        constant_variables: list[str] = [],
    ):
        self.name = name

        # === Network syncronization
        self._attached_networks: set[ComponentNetwork[CasadiSystem]] = set()
        self._network_maps: dict['ComponentNetwork', dict[int, int]] = {}

        # === Store
        self._spanwise_constants: set[str] = set()

        # === Get all the variables to copy from previous node
        self._from_prev_node: set[str] = set(
            self.__class__.from_previous_node + from_previous_node
        )
        # === Write the constant variables
        self._const_variables: set[str] = set(
            self.__class__.constant_variables + constant_variables
        )

        # === Constraint dictionaries
        self._boundary_conditions = {0: defaultdict(dict), 1: defaultdict(dict)}
        self.inlet_bc.update(inlet_bc)
        self.outlet_bc.update(outlet_bc)

        # === Equation management
        base_eqs_instances = {eq(): pos for eq, pos in self.base_equations}
        # Superseed base equations of the component with user-defined
        self._equations = self._merge_unique_equations(
            base_eqs_instances, extra_equations
        )

        # Check for duplicates
        self._equation_checks()

        # === Convenient node access
        self.inlet_node: FlowNode | None = None
        self.outlet_node: FlowNode | None = None

        # Post-init for child classes
        self._post_init()

    def _post_init(self):
        pass

    def _write_equalities(self, copy_from_prev: list[str], comp_const: list[str]):
        for arg in copy_from_prev:
            self.copy_from_previous(arg)
        for arg in comp_const:
            self.set_component_constant(arg)

    def attach_network(self, network: 'ComponentNetwork'):
        logger.debug(f'Attached network {network} to {self}')
        self._attached_networks.add(network)

    @property
    def inlet_bc(self):
        return self._boundary_conditions[0]

    @property
    def outlet_bc(self):
        return self._boundary_conditions[1]

    @property
    def network_maps(self):
        if not self._attached_networks.issubset(self._network_maps):
            self._build_network_maps()
        return self._network_maps

    def _check_attached_network(self, *, strict: bool = True):
        if not self._attached_networks:
            message = f'Modifying {self}, `{self.name}` with no networks attached'
            if strict:
                raise AttributeError(message)
            logger.warning(message)

    def _build_network_maps(self):
        for ntw in self._attached_networks:
            for eq in self._equations:
                abs_pos = self.get_absolute_eq_position(eq, ntw)
                if len(abs_pos) == 2:
                    rel_pos = ensure_tuple(self._equations[eq])
                    break
            else:
                raise RuntimeError(
                    f'No two-node equation found in {self} '
                    f'to build network map for {ntw}'
                )
            self._network_maps[ntw] = dict(zip(rel_pos, abs_pos))

    def _equation_checks(self):
        # 1. This checks that the user has not defined multiple
        # incompatible unique equations
        self._check_duplicate_equations()
        # 2. Check that at least 1 LossModel was added
        # you can use DummyLoss to forecefully pass this check
        # self._check_loss_model()

    def _merge_unique_equations(
        self,
        base_eqs: dict[EquationBase, int | tuple[int, ...]],
        user_eqs: dict[EquationBase, int | tuple[int, ...]],
    ):
        """
        Intended behaviour: If the user does not specify a unique
        equation, the one in the base is used (e.g. a camberline
        parametrization), but if the user specifies it, the new
        camberline equations should substitute the existing one.
        """
        # WARN: This intentionally only merges the base and user
        # equations, but not an arbitrary single dictionary, because
        # it is not clear which of two instances to keep

        # > Loop over the base equations
        for base_eq, base_pos in base_eqs.copy().items():
            # > If one of the base equations is a unique eq.
            if isinstance(base_eq, UniqueEquation):
                # > Get its parent class and position
                base_eq_parent = base_eq.__class__.__base__
                base_pos = set(ensure_tuple(base_pos))

                # > Check that there are no user equations
                # that superseed that unique equation
                for user_eq, user_pos in user_eqs.items():
                    user_pos = set(ensure_tuple(user_pos))
                    user_eq_parent = user_eq.__class__.__base__
                    # > If there are
                    # > AND they are in the same position
                    if user_eq_parent == base_eq_parent and user_pos == base_pos:
                        logger.warning(
                            f'Overwriting {base_eq.__class__} with {user_eq.__class__} '
                            f'in position {user_pos}'
                        )
                        # > Remove the equation instance from the base
                        base_eqs.pop(base_eq)

        return {**base_eqs, **user_eqs}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Force children to define these class attributes
        if not hasattr(cls, 'base_equations'):
            raise TypeError(f'{cls.__name__} must define `base_equations`')

        cls._verify_base_equation_format()

    @classmethod
    def _verify_base_equation_format(cls):
        """Verify that the base equations are entered in the correct format"""
        # Check types
        if not isinstance(cls.base_equations, list):
            raise TypeError('Equations must be supplied as a list')

        for eq_class, position in cls.base_equations:
            if not inspect.isclass(eq_class):
                raise TypeError(
                    f'`{eq_class.__class__.__name__}` in `{cls.__name__}` is not an'
                    f' EquationBase class type. Please provide a class object,'
                    f' not an instance'
                )
            if not isinstance(position, (int, tuple)):
                raise TypeError(
                    f'Base equation position for `{eq_class.__name__}` '
                    f'in `{cls.__name__}`, must either be an integer '
                    f'or a tuple of integers. Found `{type(position)}`, {position}.'
                )

            if isinstance(position, tuple):
                if not all(type(x) is int for x in position):
                    raise TypeError(
                        f'Ambiguous equation position `{position}` in `{cls.__name__}` '
                        f'for `{eq_class.__name__}`. Please provide a tuple of '
                        f'integers. '
                    )

        logger.debug(f'Base equations format in {cls.__name__} verifiedx')

    def _check_duplicate_equations(self):
        unique_types_seen = []
        logger.debug(f'Checking for duplicate equations in {self}')
        for eq, eq_pos in self._equations.items():
            if isinstance(eq, UniqueEquation):
                eq_pos = set(ensure_tuple(eq_pos))

                eq_base_cls = eq.__class__.__base__

                type_key = (eq_base_cls, eq_pos)
                if type_key not in unique_types_seen:
                    unique_types_seen.append(type_key)
                else:
                    raise KeyError(
                        f'Duplicate equation type for {eq_base_cls} in {self}'
                    )

    def _check_loss_model(self):
        # The duplicate instances of LossApplier should
        # be caught by duplicate eqution checking
        loss_model_seen = False
        for eq in self._equations:
            if isinstance(eq, LossApplier):
                loss_model_seen = True
                break

        if not loss_model_seen:
            raise AttributeError(
                f'No loss applier function for `{self.name}` component instance'
            )

    # ================== Interaction with the system
    def get_absolute_eq_position(
        self, equation: EquationBase, network: 'ComponentNetwork'
    ):
        rel_position = self._equations[equation]
        rel_position = ensure_tuple(rel_position)

        inl_idx, out_idx = network._get_abs_indices(self)
        index_map = {0: inl_idx, 1: out_idx}
        return tuple(index_map[idx] for idx in rel_position)

    def add_equation(
        self,
        equation: EquationBase,
        rel_position: int | tuple[int, ...],
    ):
        # Add to self (component)
        self._equations[equation] = rel_position
        self._equation_checks()
        # Add to attached networks
        self._check_attached_network(strict=False)
        for ntw in self._attached_networks:
            abs_position = self.get_absolute_eq_position(equation, ntw)
            logger.debug(
                f'Adding {equation} in position {abs_position} '
                f'to network {ntw} attached to {self} '
            )
            ntw.system.add_equation(equation, abs_position)

    def equations_from_dict(self, equations: dict[EquationBase, int | tuple[int, ...]]):
        """Simple method for multiple equations"""
        for eq, pos in equations.items():
            self.add_equation(eq, pos)

    def remove_equation(
        self, equation_class: Type[EquationBase], rel_position: int | tuple[int, ...]
    ):
        logger.debug(f'Requested removal of {equation_class} from {self}')

        rel_position = ensure_tuple(rel_position)
        equation_found = None
        for eq, pos in self._equations.copy().items():
            pos = ensure_tuple(pos)
            if isinstance(eq, equation_class) and pos == rel_position:
                logger.debug('Instance found, removing from component')
                equation_found = eq
                break

        if equation_found is not None:
            # Remove it to all attached networks
            for ntw in self._attached_networks:
                abs_position = self.get_absolute_eq_position(equation_found, ntw)
                logger.debug(
                    f'Removing {equation_found} in position {abs_position} '
                    f'from network {ntw} attached to {self} '
                )
                ntw.system.remove_equation(equation_class, abs_position)
            # Remove it from the component itself as a final step
            self._equations.pop(equation_found)
        else:
            logger.warning(
                f'No equation {equation_class} found in'
                f' {self} at position {rel_position}'
            )
            pass

    def set_boundary_cond(self, argument: str, value: ArrayLike | PlainQuantity):
        self._system_interaction_helper('set_bc', argument, value)

    def rm_boundary_cond(self, argument: str):
        self._system_interaction_helper('rm_bc', argument)

    def set_component_constant(self, argument: str):
        self._system_interaction_helper('const', argument)

    def set_spanwise_constant(self, *arguments: str):
        for arg in arguments:
            self._system_interaction_helper('span', arg)

    def copy_from_previous(self, *arguments: str):
        for arg in arguments:
            self._system_interaction_helper('prev', arg)

    def _system_interaction_helper(
        self,
        mode: Literal['set_bc', 'rm_bc', 'span', 'const', 'prev'],
        argument: str,
        value: ArrayLike | PlainQuantity = -1,
    ):
        # Only `set` and `rm` you can work unattached
        self._check_attached_network(strict=mode not in ('set', 'rm'))
        for ntw in self._attached_networks:
            if mode in ('const', 'prev'):
                arg_state = get_arg_state(argument)
                arg_type = get_arg_type(argument)
                abs_indices = self.network_maps[ntw].values()
                arg_no_idx = f'{arg_state}_{arg_type}'
                if mode == 'const':
                    abs_equality = tuple(f'{arg_no_idx}{i}' for i in abs_indices)
                    ntw.system.add_equalities(abs_equality)  # Add to system
                    self._const_variables.add(arg_no_idx)  # Add to self
                elif mode == 'prev':
                    inl_idx = min(abs_indices)
                    ntw.system.add_equalities(
                        (
                            f'{arg_state}_{arg_type}{inl_idx - 1}',  # Out previous
                            f'{arg_state}_{arg_type}{inl_idx}',  # Inlet self
                        )
                    )
                    self._from_prev_node.add(arg_no_idx)

            else:
                arg_state, arg_type, rel_idx = get_arg_specs(argument)
                abs_idx = self.network_maps[ntw][rel_idx]
                if mode == 'set_bc':
                    ntw.system.boundary_conditions[abs_idx][arg_state][arg_type] = value
                    self._boundary_conditions[rel_idx][arg_state][arg_type] = value
                elif mode == 'rm_bc':
                    ntw.system.boundary_conditions[abs_idx][arg_state].pop(arg_type)
                    self._boundary_conditions[rel_idx][arg_state].pop(arg_type)
                elif mode == 'span':
                    ntw.system.add_spanwise_constants(
                        f'{arg_state}_{arg_type}{abs_idx}'
                    )
                    self._spanwise_constants.add(f'{arg_state}_{arg_type}{rel_idx}')
