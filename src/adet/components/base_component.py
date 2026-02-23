from abc import ABC
import inspect
from collections import defaultdict
import logging
from typing import TYPE_CHECKING, ClassVar, TypeAlias, Type, Any

from adet.assembly import CasadiSystem
from adet.equations import EquationBase, UniqueEquation
from adet.equations.base_equation import LossApplier
from adet.node import FlowNode
from adet.tools.iter import ensure_tuple

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
        in_constraints: dict[
            str,
            dict[str, Any],
        ] = {},
        out_constraints: dict[
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

        self._from_previous_node = self.__class__.from_previous_node.copy()
        self._constant_variables = self.__class__.constant_variables.copy()
        if from_previous_node:
            self._from_previous_node += from_previous_node
        if constant_variables:
            self._constant_variables += constant_variables

        # Update the constraint dictionaries
        self.in_constraints = defaultdict(dict)
        self.in_constraints.update(in_constraints)
        self.out_constraints = defaultdict(dict)
        self.out_constraints.update(out_constraints)

        # Careful, This makes equations non-reusable, because when
        # added to a system the instance is used for dictionary keys
        base_eqs_instances = self._create_base_instances()

        # Superseed base equations of the component with user
        # defined ones
        self._equations = self._merge_unique_equations(
            base_eqs_instances, extra_equations
        )
        self._equation_checks()

        self._attached_networks: set[ComponentNetwork[CasadiSystem]] = set()
        self.inlet_node: FlowNode | None = None
        self.outlet_node: FlowNode | None = None

        # Post-init for child classes
        self._post_init()

    def attach_network(self, network: 'ComponentNetwork'):
        self._attached_networks.add(network)

    def _create_base_instances(self):
        base_eqs_instances: dict[EquationBase, int | tuple[int, ...]] = {}
        for eq, pos in self.base_equations:
            base_eqs_instances[eq()] = pos
        return base_eqs_instances

    def _equation_checks(self):
        # 1. This checks that the user has not defined multiple
        # incompatible unique equations
        self._check_duplicate_equations()
        # 2. Check that at least 1 LossModel was added
        # you can use DummyLoss to forecefully pass this check
        # self._check_loss_model()

    def _post_init(self):
        pass

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
        unique_types_seen = {}
        logger.debug(f'Checking for duplicate equations in {self}')
        for eq, eq_pos in self._equations.items():
            if not isinstance(eq, UniqueEquation):
                pass
            else:
                eq_pos = ensure_tuple(eq_pos)

                eq_base_cls = eq.__class__.__base__

                if eq_base_cls not in unique_types_seen:
                    unique_types_seen[eq_base_cls] = set(eq_pos)
                else:
                    if set(eq_pos) != unique_types_seen[eq_base_cls]:
                        pass
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

    def get_absolute_eq_position(
        self, equation: EquationBase, network: 'ComponentNetwork'
    ):
        rel_position = self._equations[equation]
        rel_position = ensure_tuple(rel_position)

        inl_idx, out_idx = network._get_abs_indices(self)
        index_map = {0: inl_idx, 1: out_idx}
        return tuple(index_map[idx] for idx in rel_position)

    def add_equation(self, equation: EquationBase, rel_position: int | tuple[int, ...]):
        self._equations[equation] = rel_position
        self._equation_checks()
        for ntw in self._attached_networks:
            abs_position = self.get_absolute_eq_position(equation, ntw)
            ntw.system.add_equation(equation, abs_position)

    def remove_equation(
        self,
        equation_class: Type[EquationBase],
        rel_position: int | tuple[int, ...],
    ):
        rel_position = ensure_tuple(rel_position)

        logger.debug(f'Requested removal of {equation_class} from {self}')
        equation_found = None
        for eq, pos in self._equations.copy().items():
            pos = ensure_tuple(pos)
            if isinstance(eq, equation_class) and pos == rel_position:
                logger.debug('Instance found, removing from component')
                equation_found = eq
                break

        if equation_found is not None:
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
