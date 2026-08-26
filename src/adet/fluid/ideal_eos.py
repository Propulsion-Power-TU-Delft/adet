import copy
import inspect
import logging
from abc import ABC, abstractmethod
from functools import wraps
from inspect import Parameter, Signature
from typing import Annotated

import numpy as np
from casadi import MX
from pint import Quantity

from adet.constants import UNIVERSAL_GAS_CONSTANT
from adet.equations.base_equation import EquationBase
from adet.variables import ThermoVariables
from adet.varspec import NodeStates, VarSpec

logger = logging.getLogger(__name__)

trm = ThermoVariables(0)


def override_state_signature(func, state: NodeStates):
    """Override the signature of a NON-STATIC method"""

    @wraps(func)
    def with_state(*args, **kwargs):
        return func(*args, **kwargs)

    sign = inspect.signature(func)
    # WARN: Manually add the 'self' parameter
    new_params = [Parameter('self', Parameter.POSITIONAL_ONLY)]
    for name, param in sign.parameters.items():
        # Extract the varspec metadata
        spec: VarSpec = param.annotation.__metadata__[0]
        # Apply state and node transformations
        spec = spec.with_state(state).at_node(0)

        new_param = Parameter(
            name,
            kind=param.kind,
            annotation=Annotated[MX | Quantity, spec],
        )
        new_params.append(new_param)

    with_state.__signature__ = Signature(new_params)  # ty: ignore

    return with_state


class AnalyticalFluidState(EquationBase, ABC):
    def __init__(self):
        self.state: None | NodeStates = None
        super().__init__()

    def get_eos(self, state: NodeStates):
        new_equation_obj = copy.deepcopy(self)
        new_equation_obj.residual = override_state_signature(self.residual, state)
        new_equation_obj.state = state
        return new_equation_obj

    @abstractmethod
    def eos_params(self) -> dict[VarSpec, float]:
        """
        This must as varspecs all the parameters passed from the single equation object
        as boundary conditions for the problem.
        """
        raise NotImplementedError


class IdealGasState(AnalyticalFluidState):
    def __init__(self, gamma, sp_gas_constant, viscosity):
        self.gamma: float = gamma
        self.gas_constant = sp_gas_constant

        self.viscosity = viscosity
        self.cvmass = self.gas_constant / (self.gamma - 1)
        self.cpmass = self.cvmass * self.gamma
        self.molar_mass = UNIVERSAL_GAS_CONSTANT / sp_gas_constant

        super().__init__()

    def eos_params(self) -> dict[VarSpec, float]:
        return {
            trm.Cp: self.cpmass,
            trm.Cv: self.cvmass,
            trm.Viscosity: self.viscosity,
            trm.GasConstant: UNIVERSAL_GAS_CONSTANT,
            trm.MolarMass: self.molar_mass,
            trm.RefPress: 1,
            trm.RefTemp: 1,
        }

    def residual(
        self,
        p: trm.Pressure.Hint,
        T: trm.Temperature.Hint,
        p_ref: trm.RefPress.Hint,
        T_ref: trm.RefTemp.Hint,
        rhomass: trm.Density.Hint,
        hmass: trm.Enthalpy.Hint,
        umass: trm.IntEnergy.Hint,
        smass: trm.Entropy.Hint,
        speed_sound: trm.SpeedSound.Hint,
        gas_constant: trm.GasConstant.Hint,
        molar_mass: trm.MolarMass.Hint,
        cpmass: trm.Cp.Hint,
        cvmass: trm.Cv.Hint,
    ):
        """Ideal gas law, calorifically perfect gas"""
        gamma = cpmass / cvmass
        specific_gas_const = gas_constant / molar_mass

        r1 = p - specific_gas_const * rhomass * T
        r2 = hmass - cpmass * (T - T_ref)
        r3 = umass - cvmass * (T - T_ref)
        r4 = speed_sound - (gamma * specific_gas_const * T) ** 0.5
        r5 = smass - cpmass * np.log(T / T_ref) + specific_gas_const * np.log(p / p_ref)

        return r1, r2, r3, r4, r5


if __name__ == '__main__':
    a = IdealGasState(1.4, 287, 2e-5)
