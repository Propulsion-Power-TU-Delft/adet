from abc import ABC
import inspect
from collections import defaultdict
import logging
from typing import ClassVar, TypeAlias, Type, Any

from adet.equations import EquationBase, UniqueEquation
from adet.losses import LossModel
from adet.node import FlowNode
from adet.tools.iter import ensure_iterable

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
        base_equation_instances = {
            equation(): position for equation, position in self.base_equations
        }

        # Superseed base equations of the component with user
        # defined ones
        self._equations = self._merge_unique_equations(
            base_equation_instances, extra_equations
        )
        self._equation_checks()

        self.inlet_node: FlowNode | None = None
        self.outlet_node: FlowNode | None = None

    def _equation_checks(self):
        # This checks that the user has not defined multiple
        # incompatible unique equations
        self._check_duplicate_equations()
        # Check that at least 1 LossModel was added
        # you can use DummyLoss to forecefully pass this check
        self._check_loss_model()

        # Post-init for child classes
        self._post_init()

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
        base_eqs_orig = base_eqs.copy()

        # > Loop over the base equations
        for base_eq, base_pos in base_eqs_orig.items():
            # > If one of the base equations is a unique eq.
            if isinstance(base_eq, UniqueEquation):
                # > Get its parent class and position
                base_eq_parent = base_eq.__class__.__base__
                base_pos = set(ensure_iterable(base_pos))

                # > Check that there are no user equations
                # that superseed that unique equation
                for user_eq, user_pos in user_eqs.items():
                    user_pos = set(ensure_iterable(user_pos))
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

        # if not hasattr(cls, 'from_previous_node'):
        #     raise TypeError(
        #         f'{cls.__name__} must define `from_previous_node` '
        #         f'for component interface'
        #     )

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
        for eq, eq_pos in self._equations.items():
            if not isinstance(eq, UniqueEquation):
                pass
            else:
                eq_pos = ensure_iterable(eq_pos)

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
        loss_model_seen = False
        for eq in self._equations:
            if isinstance(eq, LossModel):
                loss_model_seen = True
                break

        if not loss_model_seen:
            raise AttributeError(
                f'No loss model found for `{self.name}` component instance'
            )

    def add_equation(self, equation: EquationBase, position: int | tuple[int, ...]):
        self._equations[equation] = position
        self._equation_checks()

    def remove_equation(
        self,
        equation_class: Type[EquationBase],
        position: int | tuple[int, ...],
    ):
        if isinstance(position, int):
            position = (position,)

        for eq, pos in self._equations.items():
            if isinstance(pos, int):
                pos = (pos,)

            if isinstance(eq.__class__, equation_class) and set(pos) == set(position):
                pass

            self._equations.pop(eq)
