"""
Humid air as a mixture of three pure fluids (dry air, water
vapour, liquid water), each evaluated with its own CoolProp ``AbstractState``.

Composition is specified per kg of dry air through

- the humidity ratio ``omega = m_v / m_a``
- the wetness ``beta = m_l / (m_l + m_v)``

The gas phase obeys Dalton's law (``p = p_a + p_v``). The liquid is a
separate phase with no partial pressure: it is in mechanical and thermal
equilibrium with the gas, i.e. at the mixture pressure ``p`` and temperature
``T`` (no surface tension), and travels with the gas (no slip). Mixture
properties are mass-fraction weighted.

Phase equilibrium is a condition on the *vapour*, ``p_v = p_sat(T)``
(relative humidity ``phi = 1``). Since ``p > p_sat``, the liquid is slightly
compressed rather than saturated: its properties at ``(p, T)`` differ from
the saturated-liquid ones at ``(p_sat, T)``.
The corresponding Poynting correction to the equilibrium vapour pressure,
``p_v = p_sat exp(v_l (p - p_sat) / (R_w T))``, is neglected (< 0.2% up to 2 bar).
This is the first-order (incompressible liquid) approximation to the Poynting
correction; see A. H. Harvey (2017), "Accuracy of Approximations to the
Poynting Correction for Ice and Liquid Water", J. Res. NIST 122:41,
https://doi.org/10.6028/jres.122.041.

All properties are CasADi expressions and are fully differentiable: the
pure-fluid states go through :class:`adet.fluid.casadi_eos.CasadiEos`
(analytical first/second derivatives from the EOS) and the saturation
pressure through :class:`SaturationPressure` (exact Clapeyron derivative).

Notes
-----
- The vapour state is pinned to the gas phase and the liquid state to the
  liquid phase, so metastable (supersaturated) vapour and subcooled liquid
  are returned instead of the stable phase. The metastable extension of the
  vapour EOS is reliable for moderate supersaturation only (S = p_v/p_sat up
  to roughly 2 near ambient temperature).
- The condensate is always liquid. Below the water triple point (273.16 K)
  it is *supercooled* liquid, which the liquid-water EOS cannot represent
  (REFPROP returns erratic values tens of kelvin below the triple point):

  - the saturation pressure is that of supercooled liquid water from
    Murphy & Koop (2005), eq. 10 (valid 123-332 K), not REFPROP's
    sublimation pressure (vapour over ice);
  - the liquid enthalpy and entropy follow from the same correlation
    through Clapeyron (ideal vapour, negligible liquid volume),
    ``h_l = h_v(p_sat, T) - L``, ``s_l = s_v(p_sat, T) - L / T`` with
    ``L = R_w T^2 dln(p_sat)/dT``, plus constant offsets that make them
    continuous with REFPROP at the triple point. This Clapeyron enthalpy is
    that of saturated liquid at ``p_sat``; the term ``(p - p_sat) / rho_l``
    corrects it to the mixture pressure, as the EOS liquid above the triple
    point is evaluated at ``(p, T)``. The liquid density is frozen at its
    triple-point value.

  Physically, liquid water cannot be supercooled below about 235 K; deeper
  states are an extrapolation.
  ``HAPropsSI`` instead switches to saturation
  over ice below the triple point (and carries no condensate).
- Each fluid uses its own EOS reference state, so absolute ``h`` and ``s`` are
  not comparable with other humid-air libraries (e.g. ``HAPropsSI``). Only
  differences at conserved dry-air and total-water mass are meaningful.
- No vapour-pressure enhancement factor is applied (pure Dalton, no
  Poynting correction, no dissolved air), so saturation humidity ratios are
  ~0.4% below ``HAPropsSI`` near ambient.

Use in ADeT systems
-------------------
:class:`HumidAirMixture` is a :class:`~adet.fluid.compositional.CompositionalFluid`:
pass it as ``FluidSettings(fluid_state=mixture, update_variables=(p, T))``.
Every node then carries the composition variables
``NodeVariables(n).mix.HumidityRatio`` and ``.Wetness``, and node thermo
properties (``h, s, rho, a, cp, cv, viscosity, molar_mass``, plus
``RelHumidity`` and ``EqSpeedSound``) are evaluated with that composition.
The viscosity is that of the gas phase, mixing dry air and water vapour at
their partial pressures with Wilke's rule (C. R. Wilke (1950), "A Viscosity
Equation for Gas Mixtures", J. Chem. Phys. 18(4):517-519,
https://doi.org/10.1063/1.1747673); the liquid is not included. Below the
triple point the water-vapour viscosity is evaluated at the triple-point
temperature (approximation).
"""

import logging
from typing import Any, Sequence

import casadi as cs
import CoolProp as cp

from adet.fluid.casadi_eos import CasadiEos
from adet.fluid.compositional import CompositionalFluid
from adet.tools.coolprop_utils import configure_refprop, inames_from_id
from adet.variables import MixtureVariables

logger = logging.getLogger(__name__)

PURE_PROPS = ('hmass', 'smass', 'rhomass')

MIXTURE_OUTPUTS = (
    'Y_a',  # dry-air mass fraction
    'Y_v',  # vapour mass fraction
    'Y_l',  # liquid mass fraction
    'W',  # total water / dry air
    'x_v',  # vapour mole fraction in the gas phase
    'p_a',  # dry-air partial pressure
    'p_v',  # vapour partial pressure
    'psat',  # water saturation pressure at T
    'phi',  # relative humidity p_v / psat
    'h',  # mixture specific enthalpy
    's',  # mixture specific entropy
    'rho',  # mixture density
    'a',  # frozen speed of sound
)

# Properties available to ADeT node / equation EOS calls (VarSpec symbols)
EOS_OUTPUTS = (
    'p',
    'T',
    'hmass',
    'smass',
    'rhomass',
    'speed_sound',
    'cpmass',
    'cvmass',
    'viscosity',
    'gas_constant',
    'molar_mass',
    'rel_humidity',
    'eq_speed_sound',
)
# CoolProp input names -> property symbols
_PAIR_SYMBOLS = {
    'P': 'p',
    'T': 'T',
    'Hmass': 'hmass',
    'Smass': 'smass',
    'Dmass': 'rhomass',
}
R_UNIVERSAL = 8.314462618  # J / (mol K)
T_TRIPLE = 273.16  # water triple-point temperature [K]
# The vapour EOS is evaluated at p_v >= P_V_MIN, so that dry gas (omega = 0,
# p_v = 0) is defined; its terms are weighted by the vapour fraction anyway
P_V_MIN = 1e-6  # [Pa]


def ln_psat_supercooled(T: Any) -> Any:
    """
    ``ln(p_sat [Pa])`` of (supercooled) liquid water, Murphy & Koop (2005),
    eq. 10, valid for 123 K < T < 332 K. Accepts numbers or CasADi
    expressions.
    """
    return (
        54.842763
        - 6763.22 / T
        - 4.210 * cs.log(T)
        + 0.000367 * T
        + cs.tanh(0.0415 * (T - 218.8))
        * (53.878 - 1331.22 / T - 9.44523 * cs.log(T) + 0.014025 * T)
    )


class _SaturationPressureSecondDerivative(cs.Callback):
    """
    d2p_sat/dT2 by finite differences of the Clapeyron slope, never evaluated
    below the triple point: central with step ``DT`` where ``T - DT >=
    T_TRIPLE``, second-order forward otherwise. Below the triple point the
    EOS liquid-vapour saturation does not exist (REFPROP returns the
    sublimation pressure for the vapour and a melting-line state for the
    liquid), so a central difference at ``T_TRIPLE`` would mix the two curves.
    """

    DT = 1e-3  # [K]

    def __init__(self, name: str, parent: 'SaturationPressure', opts=None):
        cs.Callback.__init__(self)
        self._parent = parent
        self.construct(name, {'enable_fd': True, **(opts or {})})

    def get_n_in(self):
        return 3  # T, nominal p_sat, nominal dp_sat/dT

    def get_n_out(self):
        return 2  # d(dp_sat/dT)/dT, d(dp_sat/dT)/d(nominal p_sat) = 0

    def eval(self, args):
        T, h = float(args[0]), self.DT
        slope = self._parent.evaluate
        if T - h >= T_TRIPLE:
            d2 = (slope(T + h)[1] - slope(T - h)[1]) / (2 * h)
        else:
            d0, d1, d2_ = (slope(T + k * h)[1] for k in (0, 1, 2))
            d2 = (-3 * d0 + 4 * d1 - d2_) / (2 * h)
        return [d2, 0.0]


class _SaturationPressureDerivative(cs.Callback):
    """dp_sat/dT from Clapeyron: (s_v - s_l) / (v_v - v_l)."""

    def __init__(self, name: str, parent: 'SaturationPressure', opts=None):
        cs.Callback.__init__(self)
        self._parent = parent
        self.construct(name, opts or {})

    def get_n_in(self):
        return 2  # T, nominal output p_sat

    def get_n_out(self):
        return 1

    def eval(self, args):
        return [self._parent.evaluate(float(args[0]))[1]]

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, _inames, _onames, opts={}):
        # Keep a reference alive, CasADi does not own Python callbacks
        self._jac = _SaturationPressureSecondDerivative(name, self._parent, opts)
        return self._jac


class SaturationPressure(cs.Callback):
    """Water saturation pressure ``p_sat(T)`` as a differentiable CasADi function."""

    def __init__(self, name: str, state: Any, opts=None):
        cs.Callback.__init__(self)
        self._state = state
        self.construct(name, opts or {})

    def evaluate(self, T: float) -> tuple[float, float]:
        """
        Return ``(p_sat, dp_sat/dT)`` at temperature ``T``; NaN where the
        backend cannot evaluate the saturated vapour, so iterative solvers fail
        cleanly instead of raising.

        The derivative is exact (Clapeyron, ``(s_v - s_l) / (v_v - v_l)``) where
        the backend returns a saturated-liquid state, and a central finite
        difference of the vapour saturation pressure where the liquid update
        raises.

        Only valid at and above the triple point (``T >= T_TRIPLE``), where both
        backends give the liquid-vapour saturation curve. Below it the result
        depends on the backend and is *not* the supercooled-liquid saturation
        pressure (use :class:`LiquidSaturationPressure`, which switches to
        :func:`ln_psat_supercooled` there and never calls this below
        ``T_TRIPLE``):

        - HEOS extrapolates the liquid-vapour curve (0.14% below Murphy & Koop
          at 240 K) and fails (NaN) by 200 K;
        - REFPROP returns the sublimation pressure (vapour over ice, 12% below
          the supercooled-liquid value at 260 K). Its saturated-liquid state is
          on the melting line (MPa pressures), so down to 251.165 K (the ice
          Ih-III-liquid triple point, end of the melting line) the Clapeyron
          slope mixes two unrelated states and is meaningless; further down
          the liquid update raises and the finite difference gives the slope of
          the sublimation curve.
        """
        st = self._state
        try:
            st.update(cp.QT_INPUTS, 1.0, T)
        except ValueError:
            return float('nan'), float('nan')
        p, s_v, v_v = st.p(), st.smass(), 1 / st.rhomass()
        try:
            st.update(cp.QT_INPUTS, 0.0, T)
            s_l, v_l = st.smass(), 1 / st.rhomass()
            return p, (s_v - s_l) / (v_v - v_l)
        except ValueError:
            dT = 1e-3
            try:
                st.update(cp.QT_INPUTS, 1.0, T + dT)
                p_hi = st.p()
                st.update(cp.QT_INPUTS, 1.0, T - dT)
                p_lo = st.p()
            except ValueError:
                return p, float('nan')
            return p, (p_hi - p_lo) / (2 * dT)

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def eval(self, args):
        return [self.evaluate(float(args[0]))[0]]

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, _inames, _onames, opts={}):
        # Keep a reference alive, CasADi does not own Python callbacks
        self._jac = _SaturationPressureDerivative(name, self, opts)
        return self._jac


class LiquidSaturationPressure:
    """
    Saturation pressure of water over *liquid* water: the EOS saturation curve
    (:class:`SaturationPressure`) above the triple point, the supercooled
    liquid correlation :func:`ln_psat_supercooled` below it. The switch is
    not exactly smooth: at the triple point the two branches differ by ~4e-6
    relative in ``p_sat``, ~6e-5 in ``dp_sat/dT`` and ~3e-4 in
    ``d2p_sat/dT2``, negligible against the model accuracy. Callable on
    numbers or CasADi expressions; :meth:`evaluate` returns
    ``(p_sat, dp_sat/dT)``.
    """

    def __init__(self, eos_psat: SaturationPressure) -> None:
        self.eos_psat = eos_psat
        T = cs.MX.sym('T')
        p = cs.if_else(T >= T_TRIPLE, eos_psat(T), cs.exp(ln_psat_supercooled(T)), True)
        self.function = cs.Function('psat_liquid', [T], [p], ['T'], ['psat'])
        self._dfunction = cs.Function('dpsat_liquid', [T], [cs.jacobian(p, T)])

    def __call__(self, T: Any) -> Any:
        return self.function(T)

    def evaluate(self, T: float) -> tuple[float, float]:
        return float(self.function(T)), float(self._dfunction(T))


class HumidAirMixture(CompositionalFluid):
    """
    Differentiable humid-air mixture built from separate air and water EOSs.

    Parameters
    ----------
    backend:
        CoolProp backend for all pure-fluid states (``'REFPROP'`` or
        ``'HEOS'``).
    refprop_path:
        REFPROP installation folder; if omitted it is located with
        :func:`adet.tools.coolprop_utils.configure_refprop` (``RPPREFIX``,
        then common install locations).
    air, water:
        Fluid names for the backend.
    """

    def __init__(
        self,
        backend: str = 'REFPROP',
        air: str = 'Air',
        water: str = 'Water',
        refprop_path: str | None = None,
    ) -> None:
        if backend == 'REFPROP':
            configure_refprop(refprop_path)

        self.air_state = cp.AbstractState(backend, air)
        self.vap_state = cp.AbstractState(backend, water)
        self.vap_state.specify_phase(cp.iphase_gas)
        self.liq_state = cp.AbstractState(backend, water)
        self.liq_state.specify_phase(cp.iphase_liquid)
        self.sat_state = cp.AbstractState(backend, water)

        self.M_a = self.air_state.molar_mass()
        self.M_w = self.vap_state.molar_mass()
        self.R_a = self.air_state.gas_constant() / self.M_a

        self.air_eos = CasadiEos('air_eos', self.air_state, cp.PT_INPUTS, PURE_PROPS)
        self.vap_eos = CasadiEos('vap_eos', self.vap_state, cp.PT_INPUTS, PURE_PROPS)
        self.liq_eos = CasadiEos('liq_eos', self.liq_state, cp.PT_INPUTS, PURE_PROPS)
        self.air_visc = CasadiEos(
            'air_visc', self.air_state, cp.PT_INPUTS, ('viscosity',)
        )
        self.vap_visc = CasadiEos(
            'vap_visc', self.vap_state, cp.PT_INPUTS, ('viscosity',)
        )
        self.eos_psat = SaturationPressure('psat', self.sat_state)
        self.psat = LiquidSaturationPressure(self.eos_psat)
        self._init_supercooled_liquid()

        self.function = self._build()
        self.equilibrium_sound_speed = self._build_equilibrium_sound_speed()
        self._props_cache: dict[tuple[str, ...], cs.Function] = {}
        self._eos_cache: dict[tuple, cs.Function] = {}

    # ------------------------------------------------------------------
    def _init_supercooled_liquid(self) -> None:
        """
        Constants of the supercooled-liquid model: water gas constant, liquid
        density at the triple point (``rho_l_tp``) and the constant offsets
        ``_dh_l``, ``_ds_l`` (EOS saturated liquid minus
        :meth:`_supercooled_hs` at ``T_TRIPLE``), added in :meth:`_liquid` to
        make the Clapeyron-based liquid continuous with the EOS saturated
        liquid at the triple point.
        """
        self.R_w = R_UNIVERSAL / self.M_w
        st = self.sat_state
        st.update(cp.QT_INPUTS, 0.0, T_TRIPLE)
        h_l, s_l, self.rho_l_tp = st.hmass(), st.smass(), st.rhomass()
        h_l_sc, s_l_sc = (float(v) for v in self._supercooled_hs(T_TRIPLE))
        self._dh_l, self._ds_l = h_l - h_l_sc, s_l - s_l_sc

    def _supercooled_hs(self, T) -> tuple[Any, Any]:
        """
        Raw Clapeyron ``(h, s)`` of saturated supercooled liquid at
        ``p_sat(T)`` from :func:`ln_psat_supercooled`:
        ``h = h_v - L``, ``s = s_v - L / T``, ``L = R_w T^2 dln(p_sat)/dT``.

        No offsets and no pressure correction: :meth:`_liquid` adds the
        constant offsets ``_dh_l``, ``_ds_l`` (from
        :meth:`_init_supercooled_liquid`) and the compression to the mixture
        pressure.
        """
        if not hasattr(self, '_sc_hs'):
            T_sym = cs.MX.sym('T')
            ln_ps = ln_psat_supercooled(T_sym)
            L = self.R_w * T_sym**2 * cs.jacobian(ln_ps, T_sym)
            h_v, s_v, _ = self.vap_eos(cs.exp(ln_ps), T_sym)
            self._sc_hs = cs.Function(
                'supercooled_liquid_hs', [T_sym], [h_v - L, s_v - L / T_sym]
            )
        h, s = self._sc_hs(T)
        return h, s

    def _liquid(self, p, T) -> tuple[Any, Any, Any]:
        """
        Liquid ``(h, s, rho)`` at the mixture pressure and temperature.

        - ``T >= T_TRIPLE``: liquid-water EOS at ``(p, T)``.
        - ``T < T_TRIPLE``: supercooled-liquid model (see module notes),

          - ``h = h_raw + _dh_l + (p - p_sat) / rho_l_tp``
          - ``s = s_raw + _ds_l`` (no pressure correction)
          - ``rho = rho_l_tp``

          with ``(h_raw, s_raw)`` from :meth:`_supercooled_hs` and ``p_sat``
          from :func:`ln_psat_supercooled`.
        """
        h_eos, s_eos, rho_eos = self.liq_eos(p, T)
        h_sc, s_sc = self._supercooled_hs(T)
        p_s = cs.exp(ln_psat_supercooled(T))
        h_sc = h_sc + self._dh_l + (p - p_s) / self.rho_l_tp
        s_sc = s_sc + self._ds_l
        out = cs.if_else(
            T >= T_TRIPLE,
            cs.vertcat(h_eos, s_eos, rho_eos),
            cs.vertcat(h_sc, s_sc, self.rho_l_tp),
            True,  # short-circuit: the EOS liquid is erratic far below T_TRIPLE
        )
        return out[0], out[1], out[2]

    def _expressions(self, p, T, omega, beta) -> dict[str, cs.MX]:
        # Composition, per kg of dry air
        m_l = omega * beta / (1 - beta)
        W = omega + m_l
        Y_a, Y_v, Y_l = 1 / (1 + W), omega / (1 + W), m_l / (1 + W)

        # Dalton's law on the gas phase
        x_v = omega / (omega + self.M_w / self.M_a)
        p_v = x_v * p
        p_a = p - p_v
        psat = self.psat(T)

        h_a, s_a, rho_a = self.air_eos(p_a, T)
        h_v, s_v, _ = self.vap_eos(cs.fmax(p_v, P_V_MIN), T)
        h_l, s_l, rho_l = self._liquid(p, T)

        h = Y_a * h_a + Y_v * h_v + Y_l * h_l
        s = Y_a * s_a + Y_v * s_v + Y_l * s_l
        # Gas volume = volume of the dry air at its partial pressure
        rho = (1 + W) / (1 / rho_a + m_l / rho_l)

        return {
            'Y_a': Y_a, 'Y_v': Y_v, 'Y_l': Y_l, 'W': W, 'x_v': x_v,
            'p_a': p_a, 'p_v': p_v, 'psat': psat, 'phi': p_v / psat,
            'h': h, 's': s, 'rho': rho,
        }  # fmt: skip

    def _build(self) -> cs.Function:
        p, T, omega, beta = (cs.MX.sym(n) for n in ('p', 'T', 'omega', 'beta'))
        out = self._expressions(p, T, omega, beta)

        # Frozen speed of sound (fixed composition, phases in thermal and
        # mechanical equilibrium): a^2 = 1 / (d rho / d p)_s
        rho_p, rho_T = cs.jacobian(out['rho'], p), cs.jacobian(out['rho'], T)
        s_p, s_T = cs.jacobian(out['s'], p), cs.jacobian(out['s'], T)
        out['a'] = cs.sqrt(1 / (rho_p - rho_T * s_p / s_T))

        return cs.Function(
            'humid_air_mixture',
            [p, T, omega, beta],
            [out[k] for k in MIXTURE_OUTPUTS],
            ['p', 'T', 'omega', 'beta'],
            list(MIXTURE_OUTPUTS),
        )

    def _build_equilibrium_sound_speed(self) -> cs.Function:
        """
        Equilibrium speed of sound of a saturated mixture: ``a_e^2 = (dp/drho)``
        along an isentrope on which the phases stay in thermodynamic
        equilibrium (``phi = 1``) and total water ``W`` per kg dry air is
        conserved, so vapour condenses/evaporates with the perturbation.
        """
        p, T, W = (cs.MX.sym(n) for n in ('p', 'T', 'W'))
        omega = self.omega_from_phi(p, T, 1.0)
        beta = 1 - omega / W
        out = self._expressions(p, T, omega, beta)

        # Total derivatives along the saturated manifold (omega, beta = f(p, T))
        rho_p, rho_T = cs.jacobian(out['rho'], p), cs.jacobian(out['rho'], T)
        s_p, s_T = cs.jacobian(out['s'], p), cs.jacobian(out['s'], T)
        a_e = cs.sqrt(1 / (rho_p - rho_T * s_p / s_T))

        return cs.Function(
            'equilibrium_sound_speed', [p, T, W], [a_e], ['p', 'T', 'W'], ['a_e']
        )

    def eos_function(self, names: Sequence[str] = EOS_OUTPUTS) -> cs.Function:
        """
        Scalar ``(pressure, temperature, omega, beta) -> names``, cached. Only
        the sub-graph of the requested properties is built and evaluated (e.g.
        the equilibrium speed of sound, meaningless away from saturation, is
        not evaluated unless requested).
        """
        names = tuple(names)
        if names not in self._props_cache:
            self._props_cache[names] = self._build_eos_function(names)
        return self._props_cache[names]

    def _gas_viscosity(self, p_a, p_v, T, x_v) -> Any:
        """
        Viscosity of the gas phase (dry air + water vapour), Wilke's mixing
        rule on the pure-component viscosities at their partial pressures:

            mu = sum_i x_i mu_i / sum_j x_j Phi_ij,
            Phi_ij = (1 + (mu_i/mu_j)^(1/2) (M_j/M_i)^(1/4))^2
                     / (8 (1 + M_i/M_j))^(1/2),   Phi_ii = 1

        C. R. Wilke (1950), "A Viscosity Equation for Gas Mixtures", J. Chem.
        Phys. 18(4):517-519, https://doi.org/10.1063/1.1747673. The liquid is
        not included (no-slip droplets of negligible volume fraction).

        Below the triple point the water-vapour viscosity is evaluated at
        ``T_TRIPLE`` (approximation): the REFPROP water viscosity correlation
        is not defined below it. The vapour partial pressure is floored at
        ``P_V_MIN`` so that dry gas (``x_v = 0``) is defined.
        """
        mu_a = self.air_visc(p_a, T)
        mu_v = self.vap_visc(cs.fmax(p_v, P_V_MIN), cs.fmax(T, T_TRIPLE))
        x_a = 1 - x_v

        def phi(mu_i, mu_j, M_i, M_j):
            num = (1 + cs.sqrt(mu_i / mu_j) * (M_j / M_i) ** 0.25) ** 2
            return num / cs.sqrt(8 * (1 + M_i / M_j))

        phi_av = phi(mu_a, mu_v, self.M_a, self.M_w)
        phi_va = phi(mu_v, mu_a, self.M_w, self.M_a)
        return x_a * mu_a / (x_a + x_v * phi_av) + x_v * mu_v / (x_v + x_a * phi_va)

    def _build_eos_function(self, names: tuple[str, ...]) -> cs.Function:
        p, T, omega, beta = (cs.MX.sym(n) for n in ('p', 'T', 'omega', 'beta'))
        out = self._expressions(p, T, omega, beta)
        rho, s, h = out['rho'], out['s'], out['h']

        def speed_sound():
            rho_p, rho_T = cs.jacobian(rho, p), cs.jacobian(rho, T)
            s_p, s_T = cs.jacobian(s, p), cs.jacobian(s, T)
            return cs.sqrt(1 / (rho_p - rho_T * s_p / s_T))

        def cvmass():
            # cv = cp - T (dv/dT)_p^2 / (-(dv/dp)_T), with v = 1 / rho
            rho_p, rho_T = cs.jacobian(rho, p), cs.jacobian(rho, T)
            return cs.jacobian(h, T) - T * rho_T**2 / (rho**2 * rho_p)

        builders = {
            'p': lambda: p,
            'T': lambda: T,
            'hmass': lambda: h,
            'smass': lambda: s,
            'rhomass': lambda: rho,
            'speed_sound': speed_sound,
            'cpmass': lambda: cs.jacobian(h, T),  # frozen composition, const p
            'cvmass': cvmass,
            'viscosity': lambda: self._gas_viscosity(
                out['p_a'], out['p_v'], T, out['x_v']
            ),
            'gas_constant': lambda: R_UNIVERSAL + 0 * p,
            # gas-phase molar mass (mol of gas per kg dry air in denominator)
            'molar_mass': lambda: (1 + omega) / (1 / self.M_a + omega / self.M_w),
            'rel_humidity': lambda: out['phi'],
            'eq_speed_sound': lambda: self.equilibrium_sound_speed(
                p, T, omega / (1 - beta)
            ),
        }
        return cs.Function(
            f'humid_air_props_{len(self._props_cache)}',
            [p, T, omega, beta],
            [builders[k]() for k in names],
            ['pressure', 'temperature', 'omega', 'beta'],
            list(names),
        )

    # *** CompositionalFluid interface ---------------------------------
    @property
    def composition_variables(self) -> tuple:
        mv = MixtureVariables
        return (mv.HumidityRatio.Glob, mv.Wetness.Glob)

    def make_eos(
        self,
        input_pair: int,
        out_properties: Sequence[str],
        length: int,
        name: str,
    ) -> cs.Function:
        """
        ``(v0, v1, omega, beta) -> out_properties`` for a CoolProp input pair.
        Pairs other than ``(p, T)`` are inverted with an embedded Newton
        solve (differentiable through the implicit function theorem).
        Supported pairs: PT, PSmass, SmassT, HmassP, HmassSmass, DmassP,
        DmassT.
        """
        out_properties = tuple(out_properties)
        unknown = [o for o in out_properties if o not in EOS_OUTPUTS]
        if unknown:
            raise NotImplementedError(
                f'{self.__class__.__name__} cannot compute {unknown}; '
                f'available properties: {EOS_OUTPUTS}'
            )

        key = (input_pair, out_properties)
        if key not in self._eos_cache:
            self._eos_cache[key] = self._make_scalar_eos(input_pair, out_properties)
        scalar = self._eos_cache[key]

        n_in = scalar.n_in()
        xs = [cs.MX.sym(f'in{i}', length) for i in range(n_in)]
        if length == 1:
            outs = scalar(*xs)
        else:
            outs = scalar.map(length)(*[x.T for x in xs])
        outs = [outs] if not isinstance(outs, (tuple, list)) else list(outs)
        if length > 1:
            outs = [o.T for o in outs]
        return cs.Function(name, xs, outs, [*scalar.name_in()], list(out_properties))

    def _make_scalar_eos(
        self, input_pair: int, out_properties: tuple[str, ...]
    ) -> cs.Function:
        in_names = inames_from_id(input_pair)
        try:
            in_syms = [_PAIR_SYMBOLS[n] for n in in_names]
        except KeyError as err:
            raise NotImplementedError(
                f'Input pair {in_names} not supported by {self.__class__.__name__}'
            ) from err

        v0, v1, omega, beta = (cs.MX.sym(n) for n in (*in_syms, 'omega', 'beta'))
        p, T = self._solve_pT(dict(zip(in_syms, (v0, v1))), omega, beta)
        props = self.eos_function(out_properties)(
            pressure=p, temperature=T, omega=omega, beta=beta
        )
        return cs.Function(
            f'humid_air_{"".join(in_names)}',
            [v0, v1, omega, beta],
            [props[k] for k in out_properties],
            [*(f'in_{n}' for n in in_syms), 'omega', 'beta'],
            list(out_properties),
        )

    def _solve_pT(self, known: dict[str, cs.MX], omega: cs.MX, beta: cs.MX):
        """Pressure and temperature from any supported property pair."""
        if 'p' in known and 'T' in known:
            return known['p'], known['T']

        P_REF, T_REF = 1e5, 300.0
        cp_ref = 1005.0 + 1880.0 * omega
        R_ref = self.R_a

        def props(p, T):
            return self.eos_function(('hmass', 'smass'))(
                pressure=p, temperature=T, omega=omega, beta=beta
            )

        unknowns = [x for x in ('p', 'T') if x not in known]
        targets = [k for k in known if k not in ('p', 'T')]

        # Ideal-gas-like initial guesses, expressed in the known properties
        if unknowns == ['T']:
            p = known['p']
            tg = targets[0]
            if tg == 'smass':
                T_g = T_REF * cs.exp(
                    (known['smass'] - props(p, T_REF)['smass']) / cp_ref
                )
            elif tg == 'hmass':
                T_g = T_REF + (known['hmass'] - props(p, T_REF)['hmass']) / cp_ref
            else:  # rhomass
                T_g = p / (known['rhomass'] * R_ref)
            guess = [T_g / T_REF]
        elif unknowns == ['p']:
            T = known['T']
            tg = targets[0]
            if tg == 'smass':
                p_g = P_REF * cs.exp(
                    -(known['smass'] - props(P_REF, T)['smass']) / R_ref
                )
            elif tg == 'rhomass':
                p_g = known['rhomass'] * R_ref * T
            else:
                raise NotImplementedError('(h, T) does not determine the pressure')
            guess = [p_g / P_REF]
        else:
            if set(targets) != {'hmass', 'smass'}:
                raise NotImplementedError(f'Input pair {targets} not supported')
            T_g = T_REF + (known['hmass'] - props(P_REF, T_REF)['hmass']) / cp_ref
            p_g = P_REF * cs.exp(-(known['smass'] - props(P_REF, T_g)['smass']) / R_ref)
            guess = [p_g / P_REF, T_g / T_REF]

        # Implicit solve on scaled unknowns z; parameters are the target
        # property values, the composition and the known p or T (if any)
        fixed = [k for k in ('p', 'T') if k in known]
        n_t = len(targets)
        par = cs.vertcat(
            *[known[k] for k in targets], omega, beta, *[known[k] for k in fixed]
        )
        par_sym = cs.MX.sym('par', par.shape[0])
        z = cs.MX.sym('z', len(unknowns))

        vals = {
            u: z[i] * (P_REF if u == 'p' else T_REF) for i, u in enumerate(unknowns)
        }
        vals |= {k: par_sym[n_t + 2 + i] for i, k in enumerate(fixed)}
        om_s, be_s = par_sym[n_t], par_sym[n_t + 1]
        pr = self.eos_function(tuple(targets))(
            pressure=vals['p'], temperature=vals['T'], omega=om_s, beta=be_s
        )
        scales = {'hmass': R_ref * T_REF, 'smass': R_ref}
        res = []
        for i, k in enumerate(targets):
            if k == 'rhomass':
                res.append(pr[k] / par_sym[i] - 1)
            else:
                res.append((pr[k] - par_sym[i]) / scales[k])
        g = cs.Function('pT_residual', [z, par_sym], [cs.vertcat(*res)])
        rf = cs.rootfinder(
            f'pT_from_{"_".join(targets)}',
            'newton',
            g,
            {'abstol': 1e-12, 'max_iter': 50},
        )
        z_sol = rf(cs.vertcat(*guess), par)
        out = dict(known)
        for i, u in enumerate(unknowns):
            out[u] = z_sol[i] * (P_REF if u == 'p' else T_REF)
        return out['p'], out['T']

    # ------------------------------------------------------------------
    def __call__(self, p: Any, T: Any, omega: Any, beta: Any = 0.0) -> dict[str, Any]:
        """
        Mixture properties at ``(p, T, omega, beta)``; accepts numbers or
        CasADi expressions. Returns a dict keyed by :data:`MIXTURE_OUTPUTS`.
        """
        return self.function(p=p, T=T, omega=omega, beta=beta)

    def evaluate(
        self, p: float, T: float, omega: float, beta: float = 0.0
    ) -> dict[str, float]:
        """Numerical evaluation, returning plain floats."""
        return {k: float(v) for k, v in self(p, T, omega, beta).items()}

    def omega_from_phi(self, p: Any, T: Any, phi: Any) -> Any:
        """Humidity ratio of a gas at relative humidity ``phi`` (no liquid)."""
        p_v = phi * self.psat(T)
        return self.M_w / self.M_a * p_v / (p - p_v)
