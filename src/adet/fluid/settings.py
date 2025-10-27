from abc import ABC  # , abstractmethod
from dataclasses import dataclass
import logging
from typing import Generic, TypeVar


logger = logging.getLogger(__name__)

T = TypeVar('T')


class FluidModel(ABC, Generic[T]):
    """Abstract base for any fluid model backend."""

    def __init__(self, eos: T, is_analytic: bool) -> None:
        self.eos = eos
        self._is_analytic = is_analytic
        super().__init__()

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj

        # Just copy the same object
        # Abstract state has problems being deepcopied
        new_obj.eos = self.eos
        new_obj._is_analytic = self._is_analytic

        return new_obj


class EmptyFluidModel(FluidModel):
    def __init__(self) -> None:
        super().__init__(None, True)


# - - - - - - - - - - - - - - - FLUID SETTINGS
@dataclass
class FluidSettings:
    model: FluidModel
    update_variables: tuple[str, ...] = ()
    update_length: int = 2


if __name__ == '__main__':
    import CoolProp as cp
    from copy import deepcopy

    eos = cp.AbstractState('HEOS', 'R134a')

    sett = FluidSettings(
        FluidModel(eos, False),
        ('p', 'T'),
        2,
    )

    new_sett = deepcopy(sett)
