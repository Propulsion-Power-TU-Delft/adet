import logging
import threading
from abc import ABC, abstractmethod
from inspect import signature
from types import ModuleType
from typing import (
    Annotated,
    Any,
    Callable,
    NamedTuple,
    get_args,
    get_origin,
    get_type_hints,
)

import casadi as cs
import numpy as np
import sympy
from pint import DimensionalityError, Quantity
from pint.facets.plain import PlainQuantity

from adet.constants import COOLPROP_PAIRS
from adet.tools.coolprop_utils import pair_tuple_from_id
from adet.variables import ThermoVariables
from adet.varspec import VarSpec

logger = logging.getLogger(__name__)

_thrm = ThermoVariables()


class _Solver(NamedTuple):
    # Maps a (num_inputs + num_params, n) matrix to a (num_unknowns, n) matrix
    get_sol: Callable[[Any], Any]
    inputs: list[VarSpec]
    params: list[VarSpec]
    unknowns: list[VarSpec]


# Seconds before a sympy solve is abandoned in favour of the rootfinder
SYMBOLIC_SOLVE_TIMEOUT = 2.0


def _run_with_timeout(fn: Callable[[], Any], timeout: float) -> Any | None:
    """
    Run fn in a daemon thread. Returns None on timeout; re-raises fn's exception.
    The thread cannot be killed, so a timed-out call keeps running in the background
    but no longer blocks the caller or interpreter exit.
    """
    result: dict[str, Any] = {}

    def target():
        try:
            result['value'] = fn()
        except BaseException as err:  # re-raised in the caller
            result['error'] = err

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        logger.debug(f'Call timed out after {timeout} s')
        return None
    if 'error' in result:
        raise result['error']
    return result['value']


def _length(value) -> int:
    return value.numel() if isinstance(value, cs.MX) else np.size(value)


def _as_row(value, length: int):
    """Convert a value to a 1 x length row, broadcasting scalars"""
    if not isinstance(value, cs.MX):
        value = cs.DM(np.asarray(value, dtype=float).ravel())
    value = cs.vec(value).T
    return cs.repmat(value, 1, length) if value.numel() == 1 else value


def _random_probe(symbols) -> dict:
    """Fixed random point in (0.5, 2) for each symbol, for numeric checks"""
    values = np.random.default_rng(0).uniform(0.5, 2.0, len(symbols))
    return {sym: float(v) for sym, v in zip(symbols, values)}


def _with_units(value: float, unit: str, like):
    """Attach units to a constant only when the eos is evaluated with quantities"""
    return Quantity(value, unit) if isinstance(like, PlainQuantity) else value


class AlgebraicFluidState(ABC):
    # Constants are eos parameters, so solvers only depend on the class and input pair
    solution_cache: dict[tuple[type, int], _Solver] = {}
    # Input pairs that skip the symbolic solve and always use the rootfinder
    rootfinder_pairs: frozenset[int] = frozenset()

    def __init_subclass__(cls, rootfinder_pairs: tuple[int, ...] = (), **kwargs):
        """
        Subclass option,
        e.g. `class MyState(AlgebraicFluidState, rootfinder_pairs=(pair_id,))`

        The rootfinder pairs will skip the symbolic attempt and directly be
        instantiated as a N-R rootfinder
        """
        super().__init_subclass__(**kwargs)
        cls.rootfinder_pairs = frozenset(rootfinder_pairs)

    def __init__(self):
        self.current_state: dict[VarSpec, Any] = {}

    @abstractmethod
    def eos(self, pkg: ModuleType, *args):
        """
        Residuals of the equation of state, one per unknown. `pkg` is the math
        module to use (`sympy`, `numpy` or `casadi`). Arguments must be annotated
        with `<ThermoVariable>.Hint`. Constants are arguments too, and must be
        keys of `constraints()`.
        """

    def constraints(self) -> dict[VarSpec, float]:
        """Constant parameters of the eos, keyed by global spec"""
        return {}

    @property
    def arguments(self) -> list[VarSpec]:
        """Variables of the eos (without `pkg`), in signature order"""
        hints = get_type_hints(self.eos, include_extras=True)
        specs = []
        for name in list(signature(self.eos).parameters)[1:]:
            hint = hints.get(name)
            if get_origin(hint) is not Annotated:
                raise TypeError(f'Argument {name} of eos is missing a variable hint')

            spec = get_args(hint)[1]
            if not isinstance(spec, VarSpec):
                raise TypeError(f'Hint of argument {name} contains no VarSpec')
            specs.append(spec)

        return specs

    def check_units(self):
        """
        Evaluate the eos once with unit-carrying quantities, so that any
        dimensionally inconsistent residual raises a TypeError.
        """
        quantities = [Quantity(1.0, spec.unit) for spec in self.arguments]
        try:
            self.eos(np, *quantities)
        except DimensionalityError as err:
            raise TypeError(
                f'Unit check of {type(self).__name__}.eos failed: {err}'
            ) from err

    def _symbolic_solve(
        self,
        inputs: list[VarSpec],
        params: list[VarSpec],
        unknowns: list[VarSpec],
    ) -> Callable | None:
        """
        Explicit solution of the eos as a differentiable casadi function.
        Returns None when sympy does not find exactly one explicit solution,
        e.g. when there is none (degenerate or inconsistent pair), when there
        are several (branch ambiguity), or when the result is not closed form.
        """
        syms = {spec: sympy.Symbol(spec.symbol) for spec in self.arguments}
        residuals = self.eos(sympy, *(syms[spec] for spec in self.arguments))
        unknown_syms = [syms[spec] for spec in unknowns]

        try:
            solutions = _run_with_timeout(
                lambda: sympy.solve(list(residuals), unknown_syms, dict=True),
                SYMBOLIC_SOLVE_TIMEOUT,
            )
        except NotImplementedError:
            return None

        # Timed out: treat as no explicit solution
        if solutions is None:
            return None

        # Failure due to solution size
        if len(solutions) != 1 or set(solutions[0]) != set(unknown_syms):
            return None

        known = [syms[spec] for spec in inputs + params]

        # Expanding logs avoids evaluating powers such as cv**cv, which overflow
        exprs = [
            sympy.expand_log(
                solutions[0][sym],
                force=True,
            )
            for sym in unknown_syms
        ]

        for expr in exprs:
            if not expr.free_symbols <= set(known):
                return None
            # Discard imaginary, piecewise, complex roots
            if expr.has(sympy.I, sympy.Piecewise, sympy.CRootOf):
                return None

        # The solution must satisfy the residuals (wrong branch, nan, inf)
        probe = _random_probe(syms.values())
        substituted = [r.subs(dict(zip(unknown_syms, exprs))) for r in residuals]

        # Failure due to residual mismatch
        # np.isclose is False for nan, so non-finite solutions are rejected
        if not all(
            np.isclose(float(r.subs(probe)), 0.0, atol=1e-8) for r in substituted
        ):
            return None

        # Make casadi function
        x = cs.MX.sym('x', len(known))

        # numpy ufuncs dispatch correctly to MX
        body = sympy.lambdify(known, exprs, 'numpy')(*cs.vertsplit(x))
        func = cs.Function('symbolic_eos', [x], [cs.vertcat(*body)])
        return lambda X: func.map(X.shape[1])(X)

    def _is_well_posed(self, unknowns: list[VarSpec]) -> bool:
        """
        A pair determines the state when the residuals are square in the unknowns
        and their Jacobian w.r.t. the unknowns is full rank. Checked at a random
        point: a rank deficiency does not depend on the particular state.
        """
        syms = {spec: sympy.Symbol(spec.symbol) for spec in self.arguments}
        residuals = self.eos(sympy, *(syms[spec] for spec in self.arguments))
        unknown_syms = [syms[spec] for spec in unknowns]
        if len(residuals) != len(unknown_syms):
            return False

        jac = sympy.Matrix(residuals).jacobian(unknown_syms)
        numeric = np.array(jac.subs(_random_probe(syms.values())), dtype=float)
        return bool(np.linalg.matrix_rank(numeric) == len(unknown_syms))

    def _rootfinder_solve(
        self,
        inputs: list[VarSpec],
        params: list[VarSpec],
        unknowns: list[VarSpec],
    ) -> Callable:
        """Newton rootfinder on the residuals, as a differentiable casadi function"""
        syms = {spec: cs.MX.sym(spec.symbol) for spec in self.arguments}
        residuals = self.eos(cs, *(syms[spec] for spec in self.arguments))

        problem = {
            'p': cs.vertcat(*(syms[spec] for spec in inputs + params)),
            'x': cs.vertcat(*(syms[spec] for spec in unknowns)),
            'g': cs.vertcat(*residuals),
        }
        rtfn = cs.rootfinder(
            'analytical_eos',
            'newton',
            problem,
            {'error_on_fail': False},
        )
        guess = cs.DM.ones(len(unknowns), 1) * 2.0  # Unity guess

        return lambda X: rtfn.map(X.shape[1], [True, False])(guess, X)

    def _build_solver(self, input_pair: int) -> _Solver:
        self.check_units()

        by_symbol = {spec.symbol: spec for spec in self.arguments}
        inputs = [by_symbol[name] for name in pair_tuple_from_id(input_pair)]
        constants = self.constraints()
        params = [spec for spec in self.arguments if spec in constants]
        unknowns = [spec for spec in self.arguments if spec not in inputs + params]

        if not self._is_well_posed(unknowns):
            pair_name = COOLPROP_PAIRS[input_pair]
            raise ValueError(
                f'{type(self).__name__} does not determine the state from {pair_name}'
            )

        # Try symbolic solve first, unless the subclass opted this pair out
        solve = None
        if input_pair not in self.rootfinder_pairs:
            solve = self._symbolic_solve(inputs, params, unknowns)
        if solve is None:
            logger.debug(f'No explicit solution for {input_pair}, using rootfinder')
            # Fallback to rootfinder
            solve = self._rootfinder_solve(inputs, params, unknowns)

        return _Solver(solve, inputs, params, unknowns)

    def update(self, input_pair: int, value0, value1):
        key = (type(self), input_pair)
        if key not in self.solution_cache:
            logger.debug(f'Cache miss for {input_pair}, building solver')
            self.solution_cache[key] = self._build_solver(input_pair)
        solver = self.solution_cache[key]

        constants = self.constraints()
        n = max(_length(value0), _length(value1))
        rows = [_as_row(value0, n), _as_row(value1, n)]
        rows += [cs.repmat(cs.DM(constants[spec]), 1, n) for spec in solver.params]

        solution = solver.get_sol(cs.vertcat(*rows))

        for spec, value in zip(solver.inputs, (value0, value1)):
            self.current_state[spec] = (
                cs.vec(value) if isinstance(value, cs.MX) else value
            )

        for i, spec in enumerate(solver.unknowns):
            self.current_state[spec] = solution[i, :].T

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

    def speed_sound(self):
        return self.current_state[_thrm.SpeedSound]

    def p_critical(self):
        return 1

    def T_critical(self):
        return 1

    def gas_constant(self):
        return 8.31451

    def molar_mass(self):
        return 0.0287


class IdealGasState(AlgebraicFluidState):
    def __init__(self, gamma: float, sp_gas_constant: float, viscosity: float):
        super().__init__()
        self.viscosity = viscosity
        self.cvmass = sp_gas_constant / (gamma - 1)
        self.cpmass = self.cvmass * gamma

    def constraints(self):
        return {
            _thrm.Cp: self.cpmass,
            _thrm.Cv: self.cvmass,
            _thrm.Viscosity: self.viscosity,
        }

    def eos(
        self,
        pkg: ModuleType,
        p: _thrm.Pressure.Hint,
        T: _thrm.Temperature.Hint,
        rhomass: _thrm.Density.Hint,
        hmass: _thrm.Enthalpy.Hint,
        umass: _thrm.IntEnergy.Hint,
        smass: _thrm.Entropy.Hint,
        speed_sound: _thrm.SpeedSound.Hint,
        cp: _thrm.Cp.Hint,
        cv: _thrm.Cv.Hint,
    ):
        gas_constant = cp - cv
        gamma = cp / cv
        T_ref = _with_units(1.0, 'K', T)
        p_ref = _with_units(1.0, 'Pa', p)

        r0 = p - gas_constant * rhomass * T
        r1 = hmass - cp * T
        r2 = umass - cv * T
        r3 = speed_sound - pkg.sqrt(gamma * gas_constant * T)
        r4 = smass - cp * pkg.log(T / T_ref) + gas_constant * pkg.log(p / p_ref)

        return r0, r1, r2, r3, r4


# Virial coefficients: B(T) = B0 + B1 / T, C constant
_B0 = VarSpec('B0', 'm**3 / kg')
_B1 = VarSpec('B1', 'm**3 * K / kg')
_C0 = VarSpec('C0', 'm**6 / kg**2')


class VirialState(
    AlgebraicFluidState,
    rootfinder_pairs=tuple(COOLPROP_PAIRS),
):
    """
    Truncated virial eos:
    p = rho R T (1 + B(T) rho + C rho^2),
    with R = cp - cv.

    Departure functions follow from the residual Helmholtz energy
    a_r = R T (B rho + C rho^2 / 2).
    The ideal part is the same as ideal gas
    """

    def __init__(
        self,
        gamma: float,
        sp_gas_constant: float,
        viscosity: float,
        b0: float,
        b1: float,
        c0: float,
    ):
        super().__init__()
        self.viscosity = viscosity
        self.b0 = b0
        self.b1 = b1
        self.c0 = c0
        self.cvmass = sp_gas_constant / (gamma - 1)
        self.cpmass = self.cvmass * gamma

    def constraints(self):
        return {
            _thrm.Cp: self.cpmass,
            _thrm.Cv: self.cvmass,
            _thrm.Viscosity: self.viscosity,
            _B0: self.b0,
            _B1: self.b1,
            _C0: self.c0,
        }

    def eos(
        self,
        pkg: ModuleType,
        p: _thrm.Pressure.Hint,
        T: _thrm.Temperature.Hint,
        rhomass: _thrm.Density.Hint,
        hmass: _thrm.Enthalpy.Hint,
        umass: _thrm.IntEnergy.Hint,
        smass: _thrm.Entropy.Hint,
        speed_sound: _thrm.SpeedSound.Hint,
        cp: _thrm.Cp.Hint,
        cv: _thrm.Cv.Hint,
        b0: _B0.Hint,
        b1: _B1.Hint,
        c0: _C0.Hint,
    ):
        gas_constant = cp - cv
        T_ref = _with_units(1.0, 'K', T)
        p_ref = _with_units(1.0, 'Pa', p)

        B = b0 + b1 / T  # Second virial coefficient, dB/dT = -b1 / T**2 used below
        Z = 1 + B * rhomass + c0 * rhomass**2  # Compressibility factor p / (rho R T)

        # Residual departure functions, from a_r = R T (B rho + C rho^2 / 2)
        u_dep = gas_constant * b1 * rhomass  # -R T^2 d(a_r/RT)/dT
        h_dep = u_dep + gas_constant * T * (B * rhomass + c0 * rhomass**2)
        s_dep = -gas_constant * (b0 * rhomass + c0 * rhomass**2 / 2)  # b1 cancels

        # Entropy as s_ig(T, p) - R ln Z + s_dep, so no log of R is taken
        r0 = p - gas_constant * rhomass * T * Z
        r1 = hmass - cp * T - h_dep
        r2 = umass - cv * T - u_dep
        r3 = smass - (
            cp * pkg.log(T / T_ref)
            - gas_constant * pkg.log(p / p_ref)
            - gas_constant * pkg.log(Z)
            + s_dep
        )

        # Isentropic speed of sound: c^2 = (dp/drho)_T + T (dp/dT)_rho^2 / (rho^2 cv)
        # cv_dep = 0 since u_dep does not depend on T, so cv is the real cv
        dp_drho = gas_constant * T * (1 + 2 * B * rhomass + 3 * c0 * rhomass**2)
        dp_dT = gas_constant * (rhomass + b0 * rhomass**2 + c0 * rhomass**3)
        c2 = dp_drho + T * dp_dT**2 / (rhomass**2 * cv)
        r4 = speed_sound - pkg.sqrt(c2)

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
