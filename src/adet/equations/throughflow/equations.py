"""Custom equations shared by the quasi-3D throughflow examples: the
hub-to-shroud force balance + mass conservation (Van den Braembussche,
*Design and Analysis of Centrifugal Compressors*, 2019, Section 3.1.1,
Eqs. 3.3/3.6) coupled to the blade-to-blade loading (Section 3.1.2, Eqs.
3.13-3.15), both evaluated with a directly-called ``IdealGasState``
(rothalpy conservation + constant inlet entropy) rather than ADeT's
``FluidSettings``/``CasadiEos`` wiring, to avoid adding per-node P/T
variables across a full streamwise x spanwise grid.
"""

import CoolProp as cp
import numpy as np

from adet.equations.base_equation import EquationBase, EquationConfig
from adet.equations.throughflow.variables import (
    BladeLoadingVariables,
    MeridionalFlowVariables,
)
from adet.fluid.ideal_eos import IdealGasState
from adet.variables import NodeVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)
mf0 = MeridionalFlowVariables(0)
mf1 = MeridionalFlowVariables(1)
bl0 = BladeLoadingVariables(0)
bl1 = BladeLoadingVariables(1)


class LocalThermoState(EquationBase):
    """Static density from rothalpy conservation (Eq. 1.68) and the
    ideal-gas EOS at constant (inlet) entropy."""

    config = EquationConfig(manual_units=('kg / m**3',))

    def __init__(self, rothalpy: float, entropy: float, gas_state: IdealGasState, **kw):
        super().__init__(**kw)
        self.rothalpy = rothalpy
        self.entropy = entropy
        self.gas_state = gas_state

    def residual(
        self,
        w0: n0.kin.W_mag.Hint,
        u0: n0.kin.BladeSpeed.Hint,
        rho0: n0.stc.Density.Hint,
    ):
        h_static = self.rothalpy + u0**2 / 2 - w0**2 / 2
        self.gas_state.update(cp.HmassSmass_INPUTS, h_static, self.entropy)
        return rho0 - self.gas_state.rhomass()


class MeridionalMomentumMarch(EquationBase):
    """Trapezoidal finite-difference form of the hub-to-shroud force
    balance, Eq. (3.3), Coriolis-sign-corrected against Katsanis (1964)."""

    config = EquationConfig(manual_units=('m / s',))

    def __init__(
        self,
        delta_n: float,
        r0: float,
        r1: float,
        gamma: float,
        omega: float,
        inv_rn0: float,
        inv_rn1: float,
        **kw,
    ):
        super().__init__(**kw)
        self.delta_n = delta_n
        self.r0, self.r1 = r0, r1
        self.gamma = gamma
        self.omega = omega
        self.inv_rn0, self.inv_rn1 = inv_rn0, inv_rn1

    def _rhs(self, w, beta, inv_rn: float, r: float):
        return w * np.cos(beta) ** 2 * inv_rn - np.cos(self.gamma) * np.sin(beta) * (
            2 * self.omega + w * np.sin(beta) / r
        )

    def residual(
        self,
        w0: n0.kin.W_mag.Hint,
        beta0: n0.kin.FlowAngleRel.Hint,
        w1: n1.kin.W_mag.Hint,
        beta1: n1.kin.FlowAngleRel.Hint,
    ):
        rhs0 = self._rhs(w0, beta0, self.inv_rn0, self.r0)
        rhs1 = self._rhs(w1, beta1, self.inv_rn1, self.r1)
        return (w1 - w0) - self.delta_n * 0.5 * (rhs0 + rhs1)


class MassFluxAccumulation(EquationBase):
    """Trapezoidal, chained form of Eq. (3.6): cumulative meridional mass
    flow from the hub up to this spanwise station."""

    config = EquationConfig(manual_units=('kg / s',))

    def __init__(self, delta_n: float, r0: float, r1: float, **kw):
        super().__init__(**kw)
        self.delta_n = delta_n
        self.r0, self.r1 = r0, r1

    def residual(
        self,
        w0: n0.kin.W_mag.Hint,
        beta0: n0.kin.FlowAngleRel.Hint,
        rho0: n0.stc.Density.Hint,
        cf0: mf0.CumFlow.Hint,
        w1: n1.kin.W_mag.Hint,
        beta1: n1.kin.FlowAngleRel.Hint,
        rho1: n1.stc.Density.Hint,
        cf1: mf1.CumFlow.Hint,
    ):
        flux0 = rho0 * w0 * np.cos(beta0) * self.r0
        flux1 = rho1 * w1 * np.cos(beta1) * self.r1
        return cf1 - cf0 - 2 * np.pi * 0.5 * (flux0 + flux1) * self.delta_n


class BladeToBladeLoading(EquationBase):
    """Blade-to-blade suction-to-pressure side velocity difference (Eq.
    3.13), from Stokes' theorem applied to the relative-flow curl (Eq.
    1.71). ``rotation_sign=-1`` flips only the Coriolis/rotation term
    (Eq. 3.14), not the camber-turning term (Eq. 3.15) -- the radial
    turbine convention (radially-inward flow puts the Coriolis
    contribution on the opposite blade face); ``+1`` is the compressor
    convention."""

    config = EquationConfig(manual_units=('m / s',))

    def __init__(self, delta_s: float, rotation_sign: float = 1.0, **kw):
        super().__init__(**kw)
        self.delta_s = delta_s
        self.rotation_sign = rotation_sign

    def residual(
        self,
        omega0: n0.kin.Omega.Hint,
        r0: n0.geo.RDistr.Hint,
        wm0: n0.kin.W_mer.Hint,
        beta0: n0.kin.FlowAngleRel.Hint,
        omega1: n1.kin.Omega.Hint,
        r1: n1.geo.RDistr.Hint,
        wm1: n1.kin.W_mer.Hint,
        beta1: n1.kin.FlowAngleRel.Hint,
        z_r1: n1.geo.NumBlades.Hint,
        delta_bl1: n1.geo.BldThick.Hint,
        delta_w1: bl1.DeltaW.Hint,
    ):
        rotation_term = (omega1 * r1**2 - omega0 * r0**2) / self.delta_s  # Eq. (3.14)
        turning_term = (
            wm1 * r1 * np.tan(beta1) - wm0 * r0 * np.tan(beta0)
        ) / self.delta_s  # Eq. (3.15)

        pitch_minus_thickness = 2 * np.pi / z_r1 - delta_bl1 / (r1 * np.cos(beta1))

        loading = pitch_minus_thickness * (
            self.rotation_sign * rotation_term - turning_term
        )

        return delta_w1 - loading


class SlipTransition(EquationBase):
    """Stand-in for the empirical slip correlation, Eqs. (3.9)-(3.12):
    downstream of a transition point :math:`s^*`, the flow angle follows
    a quadratic blend continuous in value and slope with the blade metal
    angle at :math:`s^*`, with the remaining free coefficient implicitly
    set elsewhere (typically by a Kutta condition at the trailing edge)."""

    config = EquationConfig(manual_units=('rad',))

    def __init__(
        self,
        s_i: float,
        s_star: float,
        s_end: float,
        beta_bl_star: float,
        slope_star: float,
        **kw,
    ):
        super().__init__(**kw)
        self.s_i = s_i
        self.s_star = s_star
        self.s_end = s_end
        self.beta_bl_star = beta_bl_star
        self.slope_star = slope_star

    def residual(
        self,
        beta0: n0.kin.FlowAngleRel.Hint,
        beta_slip1: n1.kin.FlowAngleRel.Hint,
    ):
        span = self.s_end - self.s_star
        a_coeff = (beta_slip1 - self.beta_bl_star - self.slope_star * span) / span**2

        beta_target = (
            a_coeff * (self.s_i - self.s_star) ** 2
            + self.slope_star * (self.s_i - self.s_star)
            + self.beta_bl_star
        )
        return beta0 - beta_target


class CarryLoadingToLE(EquationBase):
    """Alternative to the zero-loading LE boundary condition: no equation
    otherwise ties the leading-edge loading to anything (no edge exists
    upstream of it), so carry the first interval's loading back to the
    LE (zero-gradient extrapolation) instead."""

    def residual(self, delta_w0: bl0.DeltaW.Hint, delta_w1: bl1.DeltaW.Hint):
        return delta_w0 - delta_w1


class SuctionPressureVelocities(EquationBase):
    """Superpose the blade-to-blade loading on the pitchwise-averaged
    relative velocity to recover SS/PS velocities."""

    def residual(
        self,
        w_mean0: n0.kin.W_mag.Hint,
        delta_w0: bl0.DeltaW.Hint,
        w_ss0: bl0.W_ss.Hint,
        w_ps0: bl0.W_ps.Hint,
    ):
        r1 = w_ss0 - (w_mean0 + delta_w0 / 2)
        r2 = w_ps0 - (w_mean0 - delta_w0 / 2)
        return r1, r2


class BladeSurfacePressures(EquationBase):
    """Static pressure on the SS/PS blade surfaces, from rothalpy
    conservation plus the shared static entropy."""

    config = EquationConfig(manual_units=('Pa', 'Pa'))

    def __init__(self, rothalpy: float, entropy: float, gas_state: IdealGasState, **kw):
        super().__init__(**kw)
        self.rothalpy = rothalpy
        self.entropy = entropy
        self.gas_state = gas_state

    def residual(
        self,
        w_ss0: bl0.W_ss.Hint,
        w_ps0: bl0.W_ps.Hint,
        u0: n0.kin.BladeSpeed.Hint,
        p_ss0: bl0.P_ss.Hint,
        p_ps0: bl0.P_ps.Hint,
    ):
        h_ss = self.rothalpy + u0**2 / 2 - w_ss0**2 / 2
        h_ps = self.rothalpy + u0**2 / 2 - w_ps0**2 / 2

        self.gas_state.update(cp.HmassSmass_INPUTS, h_ss, self.entropy)
        p_ss_eos = self.gas_state.p()
        self.gas_state.update(cp.HmassSmass_INPUTS, h_ps, self.entropy)
        p_ps_eos = self.gas_state.p()

        return p_ss0 - p_ss_eos, p_ps0 - p_ps_eos
