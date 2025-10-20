from abc import ABC
from collections import defaultdict
from typing import ClassVar, TypeAlias, Type, Any

from adet.equations import EquationBase

BoundaryConditions: TypeAlias = dict[str, dict[str, Any]]

BaseEquationsFormat: TypeAlias = list[
    tuple[
        Type[EquationBase],
        tuple[int, ...] | int,
    ]
]

ExtraEquationsFormat: TypeAlias = dict[
    EquationBase,
    int | tuple[int, ...],
]


class BaseComponent(ABC):
    # Force children to define these class attributes with specific types
    base_equations: ClassVar[BaseEquationsFormat]

    def __init__(
        self,
        boundary_conditions: BoundaryConditions,
        extra_equations: ExtraEquationsFormat,
    ):
        self.boundary_conditions = defaultdict(dict)
        self.boundary_conditions.update(boundary_conditions)

        base_equation_instances = {
            equation(): position for equation, position in self.base_equations
        }

        self._equations = {**base_equation_instances, **extra_equations}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Check if required attributes are defined
        if not hasattr(cls, 'base_equations'):
            raise TypeError(f'{cls.__name__} must define `base_equations`')

        # Check types
        if not isinstance(cls.base_equations, list):
            raise TypeError('Equations must be supplied as a list')

        # TODO Validate structure
