import copy
import inspect
import logging
from functools import wraps
from inspect import Parameter, Signature
from math import log
from typing import Annotated

from casadi import MX
from pint import Quantity

from adet.equations.base_equation import EquationBase
from adet.variables import ThermoVariables
from adet.varspec import NodeStates, VarSpec

logger = logging.getLogger(__name__)

trm = ThermoVariables(0)


def override_state_signature(func, state: NodeStates):
    @wraps(func)
    def with_state(*args, **kwargs):
        return func(*args, **kwargs)

    sign = inspect.signature(func)
    new_params = [Parameter('self', Parameter.POSITIONAL_ONLY)]
    for name, param in sign.parameters.items():
        # Extract the varspec metadata
        spec: VarSpec = param.annotation.__metadata__[0]
        # Apply state and node transformations
        spec = spec._with_state(state)
        spec = spec.at_node(0)

        new_param = Parameter(
            name,
            kind=param.kind,
            annotation=Annotated[MX | Quantity, spec],
        )
        new_params.append(new_param)

    with_state.__signature__ = Signature(new_params)  # ty: ignore

    return with_state


class AnalyticalFluidState:
    def __init__(self, eos_eq: EquationBase):
        self.eos_eq = eos_eq

    def _make_eq_with_state(self, state: NodeStates):
        new_equation_obj = copy.deepcopy(self.eos_eq)
        new_equation_obj.residual = override_state_signature(
            self.eos_eq.residual, state
        )
        return new_equation_obj


class IdealEos(EquationBase):
    def __init__(self, gamma, gas_constant, viscosity):
        self._gamma: float = gamma
        self.gas_constant: float = gas_constant

        self.viscosity = viscosity
        self.cvmass = self.gas_constant / (self._gamma - 1)
        self.cpmass = self.cvmass * self._gamma
        super().__init__()

    def residual(
        self,
        p: trm.Pressure.Hint,
        T: trm.Temperature.Hint,
        rhomass: trm.Density.Hint,
        hmass: trm.Enthalpy.Hint,
        umass: trm.IntEnergy.Hint,
        smass: trm.Entropy.Hint,
        speed_sound: trm.SpeedSound.Hint,
    ):
        """Docstring test"""
        r1 = p - self.gas_constant * rhomass * T
        r2 = hmass - self.cpmass * T
        r3 = umass - self.cvmass * T
        r4 = speed_sound - (self._gamma * self.gas_constant * T) ** 0.5
        r5 = smass - self.cpmass * log(T) + self.gas_constant * log(p)

        return r1, r2, r3, r4, r5


if __name__ == '__main__':
    a = AnalyticalFluidState(IdealEos(1.4, 287, 2e-5))
