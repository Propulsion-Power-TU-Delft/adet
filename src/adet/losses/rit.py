import CoolProp as cp
import numpy as np

from adet.equations.base_equation import DeviationModel, EquationConfig
from adet.equations.utils import safe_abs, safe_if_else, safe_max, safe_min_clip
from adet.losses.base_loss import LossModel
from adet.variables import NodeVariables, ThermoVariables, VarSpec

n0 = NodeVariables(0)
n1 = NodeVariables(1)
thrm = ThermoVariables()


def computePrustHerman(Q):
    den = 1 / 1.68 + Q / 2.88 + Q**2 / 4.4 + Q**3 / 6.24
    # Energy factor
    numE = 2 * (1 / 1.92 + Q / 3.2 + Q**2 / 4.8 + Q**3 / 6.72)
    E = numE / den
    # Form factor
    numH = 1 / 1.2 + 3 * Q / 1.6 + 5 * Q**2 / 2 + 7 * Q**3 / 2.4 + 9 * Q**4 / 2.8
    H = numH / den

    return E, H


# *** Stator
class StatorProfileLoss(LossModel):
    """
    From `COMPUTER PROGRAM FOR DESIGN ANALYSIS OF RADIAL-INFLOW TURBINES`
    Arthur J. Glassman
    """

    config = EquationConfig(
        input_pair=cp.HmassP_INPUTS,
        out_properties=(thrm.Entropy,),
    )
    # on both nodes. For ideal gas need to use

    def residual(
        self,
        angle_out: n1.kin.FlowAngleRel.Hint,
        bld_thick: n1.geo.BldThick.Hint,
        mom_th: n1.oth.MomThick.Hint,
        pitch: n1.geo.Pitch.Hint,
        s0: n0.stc.Entropy.Hint,
        h0: n0.stc.Enthalpy.Hint,
        W1: n1.kin.W_mag.Hint,
        Tt0: n0.tot.Temperature.Hint,
        gas_const: n1.stc.GasConstant.Hint,
        mmass: n1.stc.MolarMass.Hint,
        h1_is: n1.oth.Enthalpy_Is.Hint,
        W0: n0.kin.W_mag.Hint,
        U0: n0.kin.BladeSpeed.Hint,
        U1: n1.kin.BladeSpeed.Hint,
        p1: n1.stc.Pressure.Hint,
        gPv0: n0.oth.GammaPV.Hint,
        gPv1: n1.oth.GammaPV.Hint,
        Ds_prof: n1.loss.Ds_profile.Hint,
    ):
        R = gas_const / mmass
        gPv = (gPv0 + gPv1) / 2
        W_cr = (2 * gPv / (gPv + 1) * R * Tt0) ** 0.5

        # Loss coefficient computation
        Q = (gPv - 1) / (gPv + 1) * (W1 / W_cr) ** 2
        E, H = computePrustHerman(Q)
        t_s = bld_thick / pitch
        theta_s = mom_th / pitch
        zeta_2D = (E * theta_s) / (np.cos(angle_out) - t_s - H * theta_s)

        Roth0 = h0 + W0**2 / 2 - U0**2 / 2
        # Avoid negative square argument
        roth_minus_his = safe_max(Roth0 - h1_is, 0.0 * h1_is)
        W1_is = (2 * roth_minus_his + U1**2) ** 0.5
        W1_lss = W1_is * (1 - zeta_2D) ** 0.5
        h1_lss = Roth0 - W1_lss**2 / 2 + U1**2 / 2
        s1_profile = self.eos(h1_lss, p1)

        return Ds_prof - (s1_profile - s0)


class ShockLoss(DeviationModel):
    config = EquationConfig(
        input_pair=cp.HmassSmass_INPUTS,
        out_properties=(thrm.Pressure, thrm.Density, thrm.SpeedSound),
    )

    def residual(
        self,
        # *** Shock properties
        mach0: n0.kin.MachPresh.Hint,
        h0: n0.oth.EnthalpyPresh.Hint,
        s0: n0.oth.EntropyPresh.Hint,
        s1: n0.stc.Entropy.Hint,
        shock_angle: n0.oth.ShockAngle.Hint,
        defl_angle: n0.oth.ShockDeflection.Hint,
        metal_ang: n0.geo.MetalAngle.Hint,
        beta1: n0.kin.FlowAngleRel.Hint,
        # *** Outlet
        p1: n0.stc.Pressure.Hint,
        rho1: n0.stc.Density.Hint,
        h1: n0.stc.Enthalpy.Hint,
        W1: n0.kin.W_mag.Hint,
        gPv: n0.oth.GammaPV.Hint,
        ds_shock: n0.loss.Ds_shock.Hint,
    ):
        p0, rho0, a0 = self.eos(h0, s0)
        W0 = a0 * mach0

        w0 = W0 * np.cos(shock_angle)
        u0 = W0 * np.sin(shock_angle)

        w1 = W1 * np.cos(shock_angle - defl_angle)
        u1 = W1 * np.sin(shock_angle - defl_angle)

        # Continuity
        r1 = (rho1 * u1) - (rho0 * u0)
        # Tangential momentum
        r2 = (rho1 * u1 * w1) - (rho0 * u0 * w0)
        # Normal momentum
        r3 = (p1 + rho1 * u1**2) - (p0 + rho0 * u0**2)
        # Energy
        r4 = (h1 + u1**2 / 2) - (h0 + u0**2 / 2)

        # Shock angle (arcsin is defined only below 1)
        mach0 = safe_abs(W0 / a0)
        r5 = shock_angle - (
            np.arcsin(1 / mach0)
            + (gPv + 1) / 4 * mach0**2 / (mach0**2 - 1) * defl_angle
        )

        r6 = ds_shock - (s1 - s0)
        r7 = beta1 - (metal_ang - defl_angle)

        return r1, r2, r3, r4, r5, r6, r7


# *** Shared (stator + rotor)
def _bezier_wall_stations(z_in, z_out, r_in, r_out, angle_in, angle_out, t_values):
    """
    Cubic-Bezier meridional wall curve, tangent-matched at both ends (same
    construction as ``adet.geometry.BezierCurve``: control points placed at
    1/3-chord along each end's tangent line), evaluated at the given list
    of Python-float parameter values in [0, 1]. Unlike ``BezierCurve``,
    this is built from plain arithmetic (+ ``np.cos``/``np.sin``, which the
    ``adet`` numpy shim overloads for ``cs.MX``/``Quantity``), so it is
    symbolically traceable for use inside an equation's residual.

    Returns ``(z_stations, r_stations)``, each a list of length
    ``len(t_values)``.
    """
    chord = ((z_out - z_in) ** 2 + (r_out - r_in) ** 2) ** 0.5
    chord_third = chord / 3

    p1z = z_in + chord_third * np.cos(angle_in)
    p1r = r_in + chord_third * np.sin(angle_in)
    p2z = z_out - chord_third * np.cos(angle_out)
    p2r = r_out - chord_third * np.sin(angle_out)

    z_stations, r_stations = [], []
    for t in t_values:
        mt = 1 - t
        b0, b1, b2, b3 = mt**3, 3 * mt**2 * t, 3 * mt * t**2, t**3
        z_stations.append(b0 * z_in + b1 * p1z + b2 * p2z + b3 * z_out)
        r_stations.append(b0 * r_in + b1 * p1r + b2 * p2r + b3 * r_out)

    return z_stations, r_stations


def _wall_segment_area(z_stations, r_stations, i):
    """
    Wetted (surface-of-revolution) area of one wall curve's interval
    ``[i-1, i]``: ``dA = 2*pi*r_avg*ds``, with ``ds`` the meridional-plane
    chord length between the two stations (not ``dr`` alone -- the wall is
    curved, so radius and axial position both change along the interval).
    """
    dz = z_stations[i] - z_stations[i - 1]
    dr = r_stations[i] - r_stations[i - 1]

    # NOTE: floor away from exactly 0, not just >= 0 -- sqrt's derivative
    # diverges at 0, which would otherwise hand the solver a NaN Jacobian
    # the moment two consecutive stations coincide, e.g. at the initial
    # guess
    raw = dz**2 + dr**2
    floor = 1e-12 * raw.units if hasattr(raw, 'units') else 1e-12
    ds = safe_max(raw, floor) ** 0.5

    r_avg = (r_stations[i - 1] + r_stations[i]) / 2
    return 2 * np.pi * r_avg * ds


# |> Endwall-loss march stations (internal to EndwallLoss below): entropy
# rise at each intermediate radial station. These are bookkeeping unknowns
# of a single loss model, not physical flow stations, so they are all
# co-located at the row's outlet node rather than given their own nodes.
_ds_ew1 = VarSpec('ds_ew_stn1', 'J / kg / K', guess=1.0, bounds=(0.0, 500.0), node=1)
_ds_ew2 = VarSpec('ds_ew_stn2', 'J / kg / K', guess=1.0, bounds=(0.0, 500.0), node=1)
_ds_ew3 = VarSpec('ds_ew_stn3', 'J / kg / K', guess=1.0, bounds=(0.0, 500.0), node=1)
_ds_ew4 = VarSpec('ds_ew_stn4', 'J / kg / K', guess=1.0, bounds=(0.0, 500.0), node=1)


class EndwallLoss(LossModel):
    """
    Row endwall (hub + shroud) boundary-layer entropy generation, from
    Denton's dissipation-coefficient closure [Ref 7] (see
    ``calculateEndwallLosses`` in ``docs/rit_losses_turbosim.py`` for the
    original space-marching reference implementation). Applies to any
    radial-inflow-turbine row -- stator (stationary) or rotor (rotating) --
    with the row's meridional passage shape and rotation rate as the only
    inputs that differ between the two.

    The row is discretized into ``N_STATIONS`` fixed radial intervals
    between inlet and outlet. Unlike the reference implementation -- a
    nested numerical march with an inner Picard loop per station to
    reconcile velocity/entropy/density -- the per-station entropy rise is
    exposed here as ``N_STATIONS`` new system-level unknowns/residuals
    (``Ds_endwall`` at the final station being the row's public endwall
    loss), so the whole march is solved jointly with the rest of the
    network by the same Newton/IPOPT/KINSOL rootfinder: no separate/nested
    solve is needed.

    The hub and shroud walls are each fit with a cubic-Bezier curve in the
    meridional (axial, radius) plane, tangent-matched to the row's inlet
    and outlet meridional angles -- the same construction
    ``adet.geometry.BezierCurve``/``RowGeometry`` already use for plotting
    a row's meridional profile (see ``components/blade_row.py``), just
    re-implemented here with plain arithmetic so it is CasADi-traceable.
    The local blade height (``r_tip(m) - r_hub(m)``) therefore varies
    continuously along the march rather than being assumed constant, and
    the wetted area element is the curved-wall ``dA = 2*pi*r(m)*dm`` (``m``
    = meridional arc length), not the flat-disk ``dA = 2*pi*r*dr`` -- the
    latter would incorrectly vanish wherever ``dr -> 0`` while the wetted
    path length ``dm`` stays finite (e.g. near a rotor's axial exducer). A
    stator's flat radial endwalls (purely radial meridional angle, constant
    between inlet and outlet) are just the degenerate case of this same
    curve, so no separate flat-disk formulation is needed.

    The static-pressure profile between inlet and outlet is assumed linear
    in radius for now (as in the reference implementation). Once a
    blade-loading discretization is available for this row, its per-station
    pressures should replace this placeholder.

    The relevant kinetic energy scale is the *relative* velocity
    (rothalpy-based, since rothalpy -- not absolute total enthalpy -- is
    the quantity conserved along an adiabatic passage, rotating or not),
    and the local blade speed is evaluated at each station's mean radius
    ``(r_hub(m) + r_tip(m)) / 2``. For a stationary row (``omega0 = 0``),
    rothalpy collapses to ordinary total enthalpy and relative velocity
    collapses to absolute velocity, recovering the stator's physics
    exactly.
    """

    N_STATIONS = 5

    config = EquationConfig(
        input_pair=cp.PSmass_INPUTS,
        out_properties=(thrm.Temperature, thrm.Density, thrm.Enthalpy),
    )

    def __init__(self, dissipation_coeff: float = 0.002, **kwargs):
        super().__init__(**kwargs)
        self.cd = dissipation_coeff

    def residual(
        self,
        rr_hub0: n0.geo.Rhub.Hint,
        rr_tip0: n0.geo.Rtip.Hint,
        rr_hub1: n1.geo.Rhub.Hint,
        rr_tip1: n1.geo.Rtip.Hint,
        mer_angle0: n0.geo.MeridionalAngle.Hint,
        mer_angle1: n1.geo.MeridionalAngle.Hint,
        chord_ax1: n1.geo.ChordAx.Hint,
        omega0: n0.kin.Omega.Hint,
        p0: n0.stc.Pressure.Hint,
        p1: n1.stc.Pressure.Hint,
        s0: n0.stc.Entropy.Hint,
        h0: n0.stc.Enthalpy.Hint,
        T0: n0.stc.Temperature.Hint,
        rho0: n0.stc.Density.Hint,
        W0: n0.kin.W_mag.Hint,
        U0: n0.kin.BladeSpeed.Hint,
        mf0: n0.oth.StreamMassFlow.Hint,
        ds_ew1: _ds_ew1.Hint,
        ds_ew2: _ds_ew2.Hint,
        ds_ew3: _ds_ew3.Hint,
        ds_ew4: _ds_ew4.Hint,
        Ds_endwall1: n1.loss.Ds_endwall.Hint,
    ):
        n_stations = self.N_STATIONS
        t_values = [i / n_stations for i in range(n_stations + 1)]

        z_hub, r_hub = _bezier_wall_stations(
            0.0 * chord_ax1,
            chord_ax1,
            rr_hub0,
            rr_hub1,
            mer_angle0,
            mer_angle1,
            t_values,
        )
        z_tip, r_tip = _bezier_wall_stations(
            0.0 * chord_ax1,
            chord_ax1,
            rr_tip0,
            rr_tip1,
            mer_angle0,
            mer_angle1,
            t_values,
        )

        # ds_stations[0] is the (non-free, identically zero) entropy rise at
        # the row inlet; ds_stations[i] for i>=1 are the new free unknowns
        ds_stations = [0.0 * s0, ds_ew1, ds_ew2, ds_ew3, ds_ew4, Ds_endwall1]
        pressures = [p0 + (p1 - p0) * i / n_stations for i in range(n_stations + 1)]

        # Rothalpy (conserved along an adiabatic rotating passage, playing
        # the role total enthalpy plays in the stationary stator)
        rothalpy0 = h0 + W0**2 / 2 - U0**2 / 2
        kinetic_floor = 1.0 * rothalpy0.units if hasattr(rothalpy0, 'units') else 1.0

        U_prev = omega0 * (rr_hub0 + rr_tip0) / 2
        W_prev = safe_max(2 * (rothalpy0 - h0) + U_prev**2, kinetic_floor) ** 0.5
        T_prev, rho_prev = T0, rho0

        residuals = []
        for i in range(1, n_stations + 1):
            s_i = s0 + ds_stations[i]
            T_i, rho_i, h_i = self.eos(pressures[i], s_i)

            U_i = omega0 * (r_hub[i] + r_tip[i]) / 2
            W_i = safe_max(2 * (rothalpy0 - h_i) + U_i**2, kinetic_floor) ** 0.5

            # Station-averaged properties (implicit/trapezoidal-like
            # closure): accurate without needing a nested Picard iteration,
            # since the whole chain is solved simultaneously by the outer
            # rootfinder
            T_avg = (T_prev + T_i) / 2
            rho_avg = (rho_prev + rho_i) / 2
            W_avg = (W_prev + W_i) / 2

            area_i = _wall_segment_area(z_hub, r_hub, i) + _wall_segment_area(
                z_tip, r_tip, i
            )

            entropy_gen = self.cd * rho_avg * W_avg**3 / T_avg * area_i
            residuals.append(ds_stations[i] - (ds_stations[i - 1] + entropy_gen / mf0))

            T_prev, rho_prev, W_prev = T_i, rho_i, W_i

        return tuple(residuals)


# *** Impeller
class ImpellerIncidenceLoss(LossModel):
    """
    Rotor incidence loss model for radial-inflow turbines from `Small High
    Pressure Ratio Turbines`, D. G. Wilson / NASA CR-based correlation, as
    reproduced in `COMPUTER PROGRAM FOR DESIGN ANALYSIS OF RADIAL-INFLOW
    TURBINES` [Ref 1] (see ``rotorIncidenceLosses_NASA`` in
    ``docs/rit_losses_turbosim.py``, [Ref 5]).
    """

    config = EquationConfig(
        input_pair=cp.HmassP_INPUTS,
        out_properties=(thrm.Entropy,),
    )

    def residual(
        self,
        beta0: n0.kin.FlowAngleRel.Hint,
        metal_ang0: n0.geo.MetalAngle.Hint,
        W0: n0.kin.W_mag.Hint,
        s0: n0.stc.Entropy.Hint,
        p1: n1.stc.Pressure.Hint,
        h1_is: n1.oth.Enthalpy_Is.Hint,
        Ds_inc1: n1.loss.Ds_incidence.Hint,
    ):
        incidence = beta0 - metal_ang0
        cos_incidence = np.cos(incidence)

        # Negative (under-turning) incidence is less dissipative than positive
        # (over-turning) incidence, hence the different exponent
        Deltah_neg = W0**2 / 2 * (1 - cos_incidence**2.5)
        Deltah_pos = W0**2 / 2 * (1 - cos_incidence**1.75)
        Deltah = safe_if_else(incidence <= 0, Deltah_neg, Deltah_pos)

        h1_lss = h1_is + Deltah
        s1_incidence = self.eos(h1_lss, p1)

        return Ds_inc1 - (s1_incidence - s0)


class ImpellerPassageLoss(LossModel):
    """
    Radial-inflow turbine impeller passage loss (friction, blade loading
    and secondary flow) from `A MEANLINE PREDICTION METHOD FOR RADIAL
    TURBINE EFFICIENCY`, Baines 1998 [Ref 8] (see ``rotorPassageLoss_Baines``
    in ``docs/rit_losses_turbosim.py``).
    """

    config = EquationConfig(
        input_pair=cp.HmassP_INPUTS,
        out_properties=(thrm.Entropy,),
    )

    def residual(
        self,
        r2m: n0.geo.Rmid.Hint,
        beta2g: n0.geo.MetalAngle.Hint,
        h2: n0.geo.Height.Hint,
        r3m: n1.geo.Rmid.Hint,
        r3h: n1.geo.Rhub.Hint,
        r3s: n1.geo.Rtip.Hint,
        beta3g: n1.geo.MetalAngle.Hint,
        h3: n1.geo.Height.Hint,
        chord_ax1: n1.geo.ChordAx.Hint,
        n_bl1: n1.geo.NumBlades.Hint,
        w0: n0.kin.W_mag.Hint,
        w1: n1.kin.W_mag.Hint,
        s0: n0.stc.Entropy.Hint,
        p1: n1.stc.Pressure.Hint,
        h1_is: n1.oth.Enthalpy_Is.Hint,
        Ds_prof1: n1.loss.Ds_profile.Hint,
    ):
        # Blade chord along the mean camber line, from the axial chord
        mean_tan_beta = (np.tan(beta2g) + np.tan(beta3g)) / 2
        chord = chord_ax1 / np.cos(np.arctan(mean_tan_beta))

        hyd_len = np.pi / 4 * ((chord_ax1 - h2 / 2) + (r2m - r3s - h3 / 2))

        hyd_diam = 0.5 * (
            (4 * np.pi * r2m * h2) / (n_bl1 * h2 + 2 * np.pi * r2m)
            + (2 * np.pi * (r3s**2 - r3h**2)) / (np.pi * (r3s - r3h) + n_bl1 * h3)
        )

        loading_term = 0.68 * (1 - (r3m / r2m) ** 2) * np.cos(beta3g) / h3 * chord

        Deltah_base = 0.1 * (hyd_len / hyd_diam + loading_term) * 0.5 * (w0**2 + w1**2)

        # Higher loss coefficient for a shallow (mostly axial) exducer
        Deltah = safe_if_else((r2m - r3s) / h3 >= 0.2, Deltah_base, 2 * Deltah_base)

        h1_lss = h1_is + Deltah
        s1_profile = self.eos(h1_lss, p1)

        return Ds_prof1 - (s1_profile - s0)


class ImpellerLeakageLoss(LossModel):
    """
    Radial-inflow turbine impeller tip leakage loss from `A MEANLINE
    PREDICTION METHOD FOR RADIAL TURBINE EFFICIENCY`, Baines 1998 [Ref 8]
    (see ``rotorTipLeakageLoss_Baines`` in ``docs/rit_losses_turbosim.py``).

    Parameters
    ----------
    Kx, Kr, Kxr:
        Axial, radial and mixed axial-radial tip clearance discharge
        coefficients from [Ref 8].
    """

    config = EquationConfig(
        input_pair=cp.HmassP_INPUTS,
        out_properties=(thrm.Entropy,),
    )

    def __init__(self, Kx: float = 0.4, Kr: float = 0.75, Kxr: float = -0.3, **kwargs):
        super().__init__(**kwargs)
        self.Kx = Kx
        self.Kr = Kr
        self.Kxr = Kxr

    def residual(
        self,
        u0: n0.kin.BladeSpeed.Hint,
        vm0: n0.kin.V_mer.Hint,
        vm1: n1.kin.V_mer.Hint,
        r2m: n0.geo.Rmid.Hint,
        h2: n0.geo.Height.Hint,
        cl0: n0.geo.TipClearance.Hint,
        r3m: n1.geo.Rmid.Hint,
        r3s: n1.geo.Rtip.Hint,
        h3: n1.geo.Height.Hint,
        chord_ax1: n1.geo.ChordAx.Hint,
        n_bl1: n1.geo.NumBlades.Hint,
        cl1: n1.geo.TipClearance.Hint,
        s0: n0.stc.Entropy.Hint,
        p1: n1.stc.Pressure.Hint,
        h1_is: n1.oth.Enthalpy_Is.Hint,
        Ds_leak1: n1.loss.Ds_leakage.Hint,
    ):
        gx_H = cl0 / h2
        gr_H = cl1 / h3

        Cx = (1 - (r3s / r2m)) / (vm0 * h2)
        Cr = (r3s / r2m) * (chord_ax1 - h2) / (vm1 * r3m * h3)

        # Cross term (sqrt of a product that can wander through zero away
        # from the converged solution, e.g. during the solver's initial
        # iterations): floor it away from 0 to keep its derivative bounded,
        # since d(sqrt(x))/dx diverges as x -> 0
        cross_term_sq = safe_min_clip(safe_abs(gx_H * h2 * Cx * gr_H * h3 * Cr), 1e-9)

        # Clearance loss coefficient (corrected from [Ref 8]: u0 ** 5 -->
        # u0 ** 3 to obtain a dimensionally correct coefficient)
        Deltah = (
            (u0**3 * n_bl1)
            / (8 * np.pi)
            * (
                self.Kx * gx_H * h2 * Cx
                + self.Kr * gr_H * h3 * Cr
                + self.Kxr * cross_term_sq**0.5
            )
        )

        h1_lss = h1_is + Deltah
        s1_leakage = self.eos(h1_lss, p1)

        return Ds_leak1 - (s1_leakage - s0)


class GlassmanMeitnerBlayer(LossModel):
    def residual(self): ...


class RadialGapLoss(LossModel):
    def residual(self): ...
