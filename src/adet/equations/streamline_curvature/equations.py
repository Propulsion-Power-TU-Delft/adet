"""
Custom equations for quasi-3D streamline-curvature modeling solving the isentropic
radial equilibrium and mass conservation in the meridional plane (Van den
Braembussche, *Design and Analysis of Centrifugal Compressors*, 2019, Section
3.1.1, Eqs. 3.3/3.6), and the blade-to-blade flow (Section 3.1.2, Eqs.
3.13-3.15).

The flow is isentropic (globally uniform entropy, legitimate given uniform
upstream total conditions and no loss model) but *not* homorothalpic: rothalpy
:math:`I = h + W^2/2 - U^2/2` (Eq. 1.68) is only conserved *along* each
streamline (:class:`RothalpyConservation`), anchored at its own leading-edge
velocity triangle (:class:`AbsoluteEnergyClosure`) -- it generally differs
*between* streamlines (spanwise stations), and must not be approximated as a
single constant shared by the whole domain.
"""

import CoolProp as cp
import numpy as np

from adet.equations.base_equation import EquationBase, EquationConfig
from adet.equations.streamline_curvature.variables import (
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
    """Static density from the local static enthalpy -- a free, solved
    quantity (see :class:`RothalpyConservation`/:class:`AbsoluteEnergyClosure`
    for how it's tied to the velocity triangle) -- and the shared
    (globally uniform, isentropic) entropy."""

    config = EquationConfig(manual_units=('kg / m**3',))

    def __init__(
        self,
        entropy: float,
        gas_state: IdealGasState,
        **kw,
    ):
        super().__init__(**kw)
        self.entropy = entropy
        self.gas_state = gas_state

    def residual(
        self,
        h0: n0.stc.Enthalpy.Hint,
        rho0: n0.stc.Density.Hint,
    ):
        self.gas_state.update(cp.HmassSmass_INPUTS, h0, self.entropy)
        return rho0 - self.gas_state.rhomass()


class RothalpyConservation(EquationBase):
    """Conservation of rothalpy (Eq. 1.68), :math:`I = h + W^2/2 - U^2/2`,
    along a streamline (same spanwise station, between two neighboring
    streamwise stations): with no external work or heat addition outside
    the blade passage, :math:`I` cannot change along ``s`` -- but it
    generally differs *between* streamlines (spanwise stations), since
    each one's own leading-edge velocity triangle differs (see
    :class:`AbsoluteEnergyClosure`, which anchors it there). It must
    **not** be approximated as a single global constant shared by the
    whole domain."""

    def residual(
        self,
        h0: n0.stc.Enthalpy.Hint,
        w0: n0.kin.W_mag.Hint,
        u0: n0.kin.BladeSpeed.Hint,
        h1: n1.stc.Enthalpy.Hint,
        w1: n1.kin.W_mag.Hint,
        u1: n1.kin.BladeSpeed.Hint,
    ):
        rothalpy0 = h0 + w0**2 / 2 - u0**2 / 2
        rothalpy1 = h1 + w1**2 / 2 - u1**2 / 2
        return rothalpy1 - rothalpy0


class AbsoluteEnergyClosure(EquationBase):
    """Anchors the local static enthalpy at the leading edge of a
    streamline from the (globally uniform) upstream absolute total
    enthalpy and the *local* absolute velocity magnitude:
    :math:`h = H_{0,abs} - V^2/2`. ``total_enthalpy`` is the same
    upstream absolute stagnation enthalpy at every span position (e.g.
    ``gas_state.hmass()`` at the inlet total pressure/temperature); the
    local static enthalpy -- and hence the rothalpy
    (:class:`RothalpyConservation`) that gets conserved from here on --
    still varies with span through the local, solved ``V_mag``."""

    config = EquationConfig(manual_units=('J / kg',))

    def __init__(
        self,
        total_enthalpy: float,
        **kw,
    ):
        super().__init__(**kw)
        self.total_enthalpy = total_enthalpy

    def residual(
        self,
        h0: n0.stc.Enthalpy.Hint,
        v0: n0.kin.V_mag.Hint,
    ):
        return h0 - (self.total_enthalpy - v0**2 / 2)


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

    def _rhs(
        self,
        w,
        beta,
        inv_rn: float,
        r: float,
    ):
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

    def __init__(
        self,
        delta_n: float,
        r0: float,
        r1: float,
        **kw,
    ):
        super().__init__(**kw)
        self.delta_n = delta_n  # spanwise spacing between two subsequent stremlines
        self.r0, self.r1 = r0, r1  # local radii of the two subsequent streamlines

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
    3.13). ``rotation_sign=-1`` flips the Coriolis/rotation term. Use
    ``rotation_sign=-1`` for turbines, and ``rotation_sign=+1`` for
    compressors."""

    config = EquationConfig(manual_units=('m / s',))

    def __init__(
        self,
        delta_s: float,
        rotation_sign: float = 1.0,
        **kw,
    ):
        super().__init__(**kw)
        self.delta_s = delta_s  # s coordinate increment for two points along streamline
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
    """Empirical slip correlation by Stanitz, Eqs. (3.9)-(3.12):
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
    upstream of it), so with this class the loading of the first node
    downstream of the LE is carried back to the LE itself (zero-gradient
    extrapolation)."""

    def residual(
        self,
        delta_w0: bl0.DeltaW.Hint,
        delta_w1: bl1.DeltaW.Hint,
    ):
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
    """Static pressure on the SS/PS blade surfaces, from the local
    static enthalpy/entropy plus the blade loading's kinetic-energy
    split. At fixed rothalpy (hence fixed ``U``), Eq. 1.68 gives
    :math:`h_{ss/ps} = I + U^2/2 - W_{ss/ps}^2/2 = h + (W^2 -
    W_{ss/ps}^2)/2` -- ``U`` cancels out, so only the local static
    enthalpy ``h`` and pitchwise-mean relative velocity ``W`` (both
    already solved quantities) are needed, not the rothalpy itself."""

    config = EquationConfig(manual_units=('Pa', 'Pa'))

    def __init__(
        self,
        entropy: float,
        gas_state: IdealGasState,
        **kw,
    ):
        super().__init__(**kw)
        self.entropy = entropy
        self.gas_state = gas_state

    def residual(
        self,
        h0: n0.stc.Enthalpy.Hint,
        w0: n0.kin.W_mag.Hint,
        w_ss0: bl0.W_ss.Hint,
        w_ps0: bl0.W_ps.Hint,
        p_ss0: bl0.P_ss.Hint,
        p_ps0: bl0.P_ps.Hint,
    ):
        h_ss = h0 + (w0**2 - w_ss0**2) / 2
        h_ps = h0 + (w0**2 - w_ps0**2) / 2

        self.gas_state.update(cp.HmassSmass_INPUTS, h_ss, self.entropy)
        p_ss_eos = self.gas_state.p()
        self.gas_state.update(cp.HmassSmass_INPUTS, h_ps, self.entropy)
        p_ps_eos = self.gas_state.p()

        return p_ss0 - p_ss_eos, p_ps0 - p_ps_eos
