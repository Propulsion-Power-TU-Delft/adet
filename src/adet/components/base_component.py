import inspect
import logging
from abc import ABC
from typing import ClassVar, Literal, Mapping, Type, TypeAlias

from pint.facets.plain import PlainQuantity

from adet.assemblers import SystemAssembler
from adet.constants import AdetArray
from adet.equations import EquationBase, UniqueEquation
from adet.equations.base_equation import LossApplier
from adet.tools.iter import ensure_tuple
from adet.varspec import VarSpec

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
    base_equations: ClassVar[BaseEquationsFormat]
    """
    Base equations that link the inlet and outlet nodes
    of a component
    """

    from_previous_node: ClassVar[list[VarSpec]] = []
    """
    Variables that are inherited from the previous node
    """

    constant_variables: ClassVar[list[VarSpec]] = []
    """
    Variables that are treated as invariant between inlet
    and outlet
    """

    from_next_node: ClassVar[list[VarSpec]] = []
    """
    Variables that are inherited from the next node
    """

    def __init__(
        self,
        name: str,
        bound_cond: dict[VarSpec, PlainQuantity | AdetArray] = {},
        extra_equations: dict[
            EquationBase,
            int | tuple[int, ...],
        ] = {},
        constant_variables: list[VarSpec] = [],
        spanwise_constants: list[VarSpec] = [],
        from_prev_node: list[VarSpec] = [],
        from_next_node: list[VarSpec] = [],
    ):
        self.name = name

        # === Network syncronization
        self._attached_systems: set[SystemAssembler] = set()
        self._systems_maps: dict[SystemAssembler, dict[int, int]] = {}

        # === Store
        self._spanwise_constants: set[VarSpec] = set(spanwise_constants)

        # === Get all the variables to copy from previous node
        self._from_prev_node: set[VarSpec] = set(
            self.__class__.from_previous_node + from_prev_node
        )
        # === Get all the variables to copy from previous node
        self._from_next_node: set[VarSpec] = set(
            self.__class__.from_next_node + from_next_node
        )
        # === Write the constant variables
        self._const_variables: set[VarSpec] = set(
            self.__class__.constant_variables + constant_variables
        )

        # === Boundary conditions dictionaries
        self._boundary_conditions = bound_cond

        # === Equation management
        base_eqs_instances = {eq(): pos for eq, pos in self.base_equations}
        # Superseed base equations of the component with user-defined
        self._equations = self._merge_unique_equations(
            base_eqs_instances, extra_equations
        )

        # Check for duplicates
        self._equation_checks()

        # Post-init for child classes
        self._post_init()

    def _post_init(self):
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Force children to define these class attributes
        if not hasattr(cls, 'base_equations'):
            raise TypeError(f'{cls.__name__} must define `base_equations`')

        cls._verify_base_equation_format()

    def attach_system(self, system: SystemAssembler):
        logger.debug(f'Attached network {system} to {self}')
        self._attached_systems.add(system)

    @property
    def inlet_bc(self) -> list[VarSpec]:
        return [spec for spec in self._boundary_conditions if spec.node == 0]

    @property
    def outlet_bc(self) -> list[VarSpec]:
        return [spec for spec in self._boundary_conditions if spec.node == 1]

    @property
    def system_maps(self) -> dict[SystemAssembler, dict[int, int]]:
        """{0: ABS_IN, 1: ABS_OUT}"""
        if not self._attached_systems.issubset(self._systems_maps):
            self._build_network_maps()
        return self._systems_maps

    def _check_attached_system(self, *, strict: bool = True):
        if not self._attached_systems:
            message = f'Modifying {self}, `{self.name}` with no networks attached'
            if strict:
                raise AttributeError(message)
            logger.warning(message)

    def get_absolute_eq_position(self, equation: EquationBase, system: SystemAssembler):
        rel_position = self._equations[equation]
        rel_position = ensure_tuple(rel_position)

        index_map = self.system_maps[system]
        return tuple(index_map[idx] for idx in rel_position)

    def _build_network_maps(self):
        for system in self._attached_systems:
            seen_couples: set[tuple[int, ...]] = set()
            for eq in self._equations:
                abs_pos = system.data.equations[eq]
                # Find a two node equation and use it for mapping
                if len(abs_pos) == 2:
                    rel_pos = ensure_tuple(self._equations[eq])
                    seen_couples.add(rel_pos)
                    ntw_map = dict(zip(rel_pos, abs_pos))

            if not seen_couples:
                raise RuntimeError(
                    f'No two-node equation found in {self} '
                    f'to build network map for {system}'
                )

            elif len(seen_couples) > 1:
                raise RuntimeError(
                    f'Multiple incompatible equation positions for {system} in {self}'
                )

            self._systems_maps[system] = ntw_map

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
    ) -> dict[EquationBase, int | tuple[int, ...]]:
        """
        Intended behaviour: If the user does not specify a unique
        equation, the one in the base is used (e.g. a camberline
        parametrization), but if the user specifies it, the new
        camberline equations should substitute the existing one.
        """
        # NOTE: This intentionally only merges the base and user
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
                            f'Overwriting {base_eq.__class__} with '
                            f'{user_eq.__class__} in position {user_pos}'
                        )
                        # > Remove the equation instance from the base
                        base_eqs.pop(base_eq)

        return {**base_eqs, **user_eqs}

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

    # ================== Interactions with the system ==================
    def add_equation(
        self,
        equation: EquationBase,
        rel_position: int | tuple[int, ...],
    ):
        # Add to self (component)
        self._equations[equation] = rel_position
        self._equation_checks()
        # Add to attached networks
        self._check_attached_system(strict=False)
        for system in self._attached_systems:
            abs_position = self.get_absolute_eq_position(equation, system)
            logger.debug(
                f'Adding {equation} in position {abs_position} '
                f'to system {system} attached to {self} '
            )
            # TODO: Use merge logic here?
            system.add_equation(equation, abs_position)

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
            for system in self._attached_systems:
                abs_position = self.get_absolute_eq_position(equation_found, system)
                logger.debug(
                    f'Removing {equation_found} in position {abs_position} '
                    f'from network {system} attached to {self} '
                )
                system.remove_equation(equation_class, abs_position)
            # Remove it from the component itself as a final step
            self._equations.pop(equation_found)
        else:
            logger.warning(
                f'No equation {equation_class} found in'
                f' {self} at position {rel_position}'
            )
            pass

    def set_boundary_cond(self, argument: VarSpec, value: AdetArray | PlainQuantity):
        self._bound_cond_helper('add', argument, value)

    def rm_boundary_cond(self, argument: VarSpec):
        self._bound_cond_helper('rm', argument)

    def copy_from_previous(self, *arguments: VarSpec):
        self._equalities_helper('prev', *arguments)

    def copy_from_next(self, *arguments: VarSpec):
        self._equalities_helper('next', *arguments)

    def set_constants(self, *arguments: VarSpec):
        self._equalities_helper('const', *arguments)

    def set_bc_from_dict(
        self, bound_conds: Mapping[VarSpec, PlainQuantity | AdetArray]
    ):
        for spec, value in bound_conds.items():
            self.set_boundary_cond(spec, value)

    def _bound_cond_helper(
        self,
        mode: Literal['add', 'rm'],
        spec: VarSpec,
        value: AdetArray | PlainQuantity | None = None,
    ):
        if mode == 'add':
            if value is None:
                raise ValueError(f'Missing value to set {spec}')
            self._boundary_conditions[spec] = value

            for system in self._attached_systems:
                abs_idx = self.system_maps[system][spec.node]
                system.data.boun_cond[spec.at_node(abs_idx)] = value
        else:
            self._boundary_conditions.pop(spec)
            for system in self._attached_systems:
                abs_idx = self.system_maps[system][spec.node]
                system.data.boun_cond.pop(spec.at_node(abs_idx))

    def _equalities_helper(
        self, mode: Literal['const', 'prev', 'next'], *arguments: VarSpec
    ):
        for spec in arguments:
            # Add to self
            if mode == 'const':
                self._const_variables.add(spec)
            elif mode == 'prev':
                self._from_prev_node.add(spec)
            elif mode == 'next':
                self._from_next_node.add(spec)

            # Add to networks
            for system in self._attached_systems:
                if mode == 'const':
                    equality = tuple(
                        spec.at_node(i) for i in self.system_maps[system].values()
                    )
                elif mode == 'prev':
                    inl_idx = min(self.system_maps[system].values())
                    equality = (
                        spec.at_node(inl_idx - 1),  # outlet of prev
                        spec.at_node(inl_idx),  # inlet of self
                    )
                elif mode == 'next':
                    out_idx = max(self.system_maps[system].values())
                    equality = (
                        spec.at_node(out_idx),  # outlet of self
                        spec.at_node(out_idx + 1),  # inlet of next
                    )

                system.add_equalities(equality)

    def set_spanwise_constant(self, *arguments: VarSpec):
        for spec in arguments:
            if spec in self._boundary_conditions:
                continue
            self._spanwise_constants.add(spec)
            for system in self._attached_systems:
                abs_spec = spec.at_node(self.system_maps[system][spec.node])
                system.add_spanwise_constants(abs_spec)
