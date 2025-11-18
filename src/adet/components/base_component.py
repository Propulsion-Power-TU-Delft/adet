from abc import ABC
import inspect
from collections import defaultdict
import logging
from typing import ClassVar, TypeAlias, Type, Any

from adet.equations import EquationBase

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
        ],
        out_constraints: dict[
            str,
            dict[str, Any],
        ],
        extra_equations: dict[
            EquationBase,
            int | tuple[int, ...],
        ] = {},
    ):
        self.name = name

        self.in_constraints = defaultdict(dict)
        self.in_constraints.update(in_constraints)

        self.out_constraints = defaultdict(dict)
        self.out_constraints.update(out_constraints)

        # Careful, This makes equations non-reusable, because when
        # added to a system the instance is used for dictionary keys
        base_equation_instances = {
            equation(): position for equation, position in self.base_equations
        }

        self._equations = {**base_equation_instances, **extra_equations}

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
