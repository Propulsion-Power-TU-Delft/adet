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

    # These describe extra links between the outlet
    # inlet node of the component and the previous one
    linker_equations: ClassVar[list[Type[EquationBase]]]

    def __init__(
        self,
        name: str,
        boundary_conditions: dict[
            str,
            dict[str, Any],
        ],
        extra_equations: dict[
            EquationBase,
            int | tuple[int, ...],
        ] = {},
    ):
        self.name = name
        self.boundary_conditions = defaultdict(dict)
        self.boundary_conditions.update(boundary_conditions)

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

        if not hasattr(cls, 'linker_equations'):
            raise TypeError(
                f'{cls.__name__} must define `linker_equations` for component interface'
            )

        cls._verify_base_equation_format()

    @classmethod
    def _verify_base_equation_format(cls):
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
