import logging
from abc import ABC, abstractmethod
from inspect import signature
from typing import Annotated, Any, get_args, get_origin, get_type_hints

import casadi as cs
import numpy as np
from pint import DimensionalityError, Quantity
from pint.facets.plain import PlainQuantity

from adet.tools.coolprop_utils import pair_tuple_from_id
from adet.variables import ThermoVariables
from adet.varspec import VarSpec

logger = logging.getLogger(__name__)

_thrm = ThermoVariables()

# Cache key: (class, input pair, gamma, gas constant)
# Cache value: scalar rootfinder, input variables, (ordered) unknown variables
# and the initial guess of the unknowns
_CacheValue = tuple[cs.Function, list[VarSpec], list[VarSpec], list[float]]


class AnalyticalFluidState(ABC):
    solution_cache: dict[int, _CacheValue] = {}

    def __init__(self):
        self.current_state: dict[VarSpec, Any] = {}

    @abstractmethod
    def eos(self, *args):
        """
        Residuals of the equation of state. The arguments must be annotated
        with `<ThermoVariable>.Hint`, which defines the variable each
        argument stands for. Returns a sequence of residuals, equal in number
        to the arguments, which are zero at a thermodynamically consistent state.
        """

    @property
    def arguments(self) -> list[VarSpec]:
        """Variables of the eos, in the order of the signature hints"""
        hints = get_type_hints(self.eos, include_extras=True)
        specs = []
        for name in signature(self.eos).parameters:
            hint = hints.get(name)
            if get_origin(hint) is not Annotated:
                raise TypeError(f'Argument {name} of eos is missing a variable hint')

            spec = get_args(hint)[1]
            if spec is None or not isinstance(spec, VarSpec):
                raise TypeError(f'Hint of argument {name} contains no VarSpec')
            specs.append(spec)

        return specs

    @staticmethod
    def _as_row(value, length: int):
        """Convert a value to a 1 x length row, broadcasting scalars"""
        if not isinstance(value, cs.MX):
            value = cs.DM(np.asarray(value, dtype=float).ravel())
        value = cs.vec(value).T
        return cs.repmat(value, 1, length) if value.numel() == 1 else value

    def check_units(self):
        """
        Evaluate the eos once with unit-carrying quantities, so that any
        dimensionally inconsistent residual raises a pint DimensionalityError.
        """
        quantities = [Quantity(1.0, spec.unit) for spec in self.arguments]
        try:
            self.eos(*quantities)
        except DimensionalityError as err:
            raise TypeError(
                f'Unit check of {type(self).__name__}.eos failed: {err}'
            ) from err

    def _build_rootfinder(
        self, input_vars: list[VarSpec]
    ) -> tuple[cs.Function, list[VarSpec]]:
        self.check_units()
        arguments = self.arguments
        unknowns = [spec for spec in arguments if spec not in input_vars]

        syms = {spec: cs.MX.sym(spec.symbol) for spec in arguments}
        residuals = self.eos(*(syms[spec] for spec in arguments))

        problem = {
            'p': cs.vertcat(*(syms[spec] for spec in input_vars)),
            'x': cs.vertcat(*(syms[spec] for spec in unknowns)),
            'g': cs.vertcat(*residuals),
        }

        rtfn = cs.rootfinder(
            'analytical_eos',
            'newton',
            problem,
            {'error_on_fail': False},
        )
        return rtfn, unknowns

    def _build_entry(self, input_pair: int) -> _CacheValue:
        spec_by_symbol = {spec.symbol: spec for spec in self.arguments}

        input_vars = [spec_by_symbol[name] for name in pair_tuple_from_id(input_pair)]

        rtfn, unknowns = self._build_rootfinder(input_vars)

        # Variable guesses are just unity
        # guess = [2.0 if s.guess is None else s.guess for s in unknowns]
        guess = [2.0 for _ in unknowns]

        return rtfn, input_vars, unknowns, guess

    def update(self, input_pair: int, value0, value1):
        entry = self.solution_cache.get(input_pair)

        if entry is None:
            logger.debug('Cache miss for %s, building rootfinder', input_pair)
            entry = self.solution_cache[input_pair] = self._build_entry(input_pair)

        rtfn, input_vars, unknowns, guess = entry

        if isinstance(value0, cs.MX):
            rtfn = rtfn.map(max(value0.shape), [True, False])
            knowns = cs.horzcat(value0, value1).T
        elif isinstance(value0, PlainQuantity):
            # rtfn = rtfn.map(max(value0.shape), [False, False])
            knowns = [value0, value1]
        else:
            knowns = [value0, value1]

        solution = rtfn(guess, knowns)

        for spec, value in zip(input_vars, (value0, value1)):
            self.current_state[spec] = (
                cs.vec(value) if isinstance(value, cs.MX) else value
            )

        for i, spec in enumerate(unknowns):
            self.current_state[spec] = solution[i, :].T

    def constraints(self) -> dict[VarSpec, float]:
        """Properties which are constant for the fluid, keyed by global spec"""
        return {}

    def get_property(self, spec: VarSpec):
        """Value of a property, regardless of the node and state of the spec"""
        key = spec.Glob
        if key in self.current_state:
            return self.current_state[key]

        constants = self.constraints()
        if key in constants:
            return constants[key]

        method = getattr(self, spec.symbol, None)
        if callable(method):
            return method()

        raise KeyError(f'Property {spec.symbol} not available in {type(self).__name__}')

    def p(self):
        return self.current_state[_thrm.Pressure]

    def T(self):
        return self.current_state[_thrm.Temperature]

    def rhomass(self):
        return self.current_state[_thrm.Density]

    def hmass(self):
        return self.current_state[_thrm.Enthalpy]

    def smass(self):
        return self.current_state[_thrm.Entropy]

    def p_critical(self):
        return 1

    def T_critical(self):
        return 1

    def speed_sound(self):
        return self.current_state[_thrm.SpeedSound]

    def gas_constant(self):
        return 8.31451

    def molar_mass(self):
        return 0.0287


class IdealGasState(AnalyticalFluidState):
    def __init__(self, gamma: float, sp_gas_constant: float, viscosity: float):
        super().__init__()
        self._gamma: float = gamma
        self._sp_gas_constant: float = sp_gas_constant

        self.viscosity = viscosity
        self.cvmass = self._sp_gas_constant / (self._gamma - 1)
        self.cpmass = self.cvmass * self._gamma

    def constraints(self):
        return {
            _thrm.Cp: self.cpmass,
            _thrm.Cv: self.cvmass,
            _thrm.Viscosity: self.viscosity,
        }

    def eos(
        self,
        p: _thrm.Pressure.Hint,
        T: _thrm.Temperature.Hint,
        rhomass: _thrm.Density.Hint,
        hmass: _thrm.Enthalpy.Hint,
        umass: _thrm.IntEnergy.Hint,
        smass: _thrm.Entropy.Hint,
        speed_sound: _thrm.SpeedSound.Hint,
    ):
        # Constants carry units only when called with quantities (unit check)
        def const(value: float, unit: str):
            return Quantity(value, unit) if isinstance(T, PlainQuantity) else value

        gas_constant = const(self._sp_gas_constant, 'J / kg / K')
        gamma = self._gamma
        cp = const(self.cpmass, 'J / kg / K')
        cv = const(self.cvmass, 'J / kg / K')
        T_ref = const(1.0, 'K')
        p_ref = const(1.0, 'Pa')

        r0 = p - gas_constant * rhomass * T
        r1 = hmass - cp * T
        r2 = umass - cv * T
        r3 = speed_sound**2 - (gamma * gas_constant * T)
        r4 = smass - cp * np.log(T / T_ref) + gas_constant * np.log(p / p_ref)

        return r0, r1, r2, r3, r4


if __name__ == '__main__':
    import CoolProp as cp

    ideal_state = IdealGasState(
        gamma=1.4,
        sp_gas_constant=287.0,
        viscosity=2e-5,
    )

    # Numeric update
    abs_state = cp.AbstractState('HEOS', 'Air')
    abs_state.update(cp.PSmass_INPUTS, 3e5, 1000)
    ideal_state.update(cp.PSmass_INPUTS, 3e5, 1000)

    print(f'Hmass is {ideal_state.hmass()}')
    print(f'Temperature is {ideal_state.T()}')

    # Symbolic (differentiable) update
    p = cs.MX.sym('p', 3)
    T = cs.MX.sym('T', 3)
    ideal_state.update(cp.PT_INPUTS, p, T)

    func = cs.Function('func', [p, T], [ideal_state.hmass(), ideal_state.rhomass()])
    jac_func = cs.Function('drho_dT', [p, T], [cs.jacobian(ideal_state.rhomass(), T)])

    print(func([1e5, 2e5, 3e5], [300, 400, 500]))
    print(jac_func([1e5, 2e5, 3e5], [300, 400, 500]))
