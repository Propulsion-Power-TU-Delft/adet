import CoolProp as cp
import numpy as np

from adet.equations.base_equation import DeviationModel, EquationConfig
from adet.equations.utils import safe_abs, safe_if_else, safe_max
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

    def __init__(
        self, dissipation_coeff: float = 0.002, row_fraction: float = 1.0, **kwargs
    ):
        """``row_fraction``: the march covers the walls from the row inlet
        up to this fraction of the wall curves (e.g. a stator mixing row
        whose outlet is a mixed-out plane past the trailing edge, with the
        walls beyond the trailing edge left to a gap model); the pressure is
        interpolated to it."""
        super().__init__(**kwargs)
        self.cd = dissipation_coeff
        self.row_fraction = row_fraction

    def residual(
        self,
        rr_hub0: n0.geo.Rhub.Hint,
        rr_tip0: n0.geo.Rtip.Hint,
        rr_hub1: n1.geo.Rhub.Hint,
        rr_tip1: n1.geo.Rtip.Hint,
        mer_angle0: n0.geo.MeridionalAngle.Hint,
        mer_angle1: n1.geo.MeridionalAngle.Hint,
        mer_chord1: n1.geo.MerChord.Hint,
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
        # NOTE: The number of stations influence the number of residuals
        # we might be able to make this flexible (also the arguments would need
        # to change

        n_stations = self.N_STATIONS
        t_values = [self.row_fraction * i / n_stations for i in range(n_stations + 1)]

        # mer_chord1 is the row's meridional-plane chord magnitude (axial
        # for an axial row, radial for a radial row, per MeridionalAngle);
        # only its axial (z) projection is a literal z-span here -- e.g.
        # zero for a purely radial row (MeridionalAngle = -90 deg), where
        # the wall curve stays flat in z and the whole chord shows up as
        # the (already-correct, independent) r_hub/r_tip radial span.
        mean_mer_angle = (mer_angle0 + mer_angle1) / 2
        z_span = mer_chord1 * np.cos(mean_mer_angle)

        z_hub, r_hub = _bezier_wall_stations(
            0.0 * z_span,
            z_span,
            rr_hub0,
            rr_hub1,
            mer_angle0,
            mer_angle1,
            t_values,
        )
        z_tip, r_tip = _bezier_wall_stations(
            0.0 * z_span,
            z_span,
            rr_tip0,
            rr_tip1,
            mer_angle0,
            mer_angle1,
            t_values,
        )

        # ds_stations[0] is the (non-free, identically zero) entropy rise at
        # the row inlet; ds_stations[i] for i>=1 are the new free unknowns
        ds_stations = [0.0 * s0, ds_ew1, ds_ew2, ds_ew3, ds_ew4, Ds_endwall1]
        pressures = [p0 + (p1 - p0) * t for t in t_values]

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
class ChenSlip(DeviationModel):
    """
    Rotor-inlet flow slip for radial/mixed-flow turbines, from H. Chen and
    N. C. Baines, "The aerodynamic loading of radial and mixed-flow
    turbines," Int. J. Mech. Sci. 36(1), pp. 63-79 (1994), Eq. (8) (eddy
    contained within the blade passage). Meant to be attached directly to
    a rotor ``BladeRow``'s own inlet node (position 0, i.e. ``ChenSlip(): 0``
    in its ``extra_equations``) -- see ``StanitzSlip`` for the alternate
    Stanitz (Eq. 4) correlation.
    """

    def residual(
        self,
        beta0: n0.kin.FlowAngleRel.Hint,
        metal_ang0: n0.geo.MetalAngle.Hint,
        mer_angle0: n0.geo.MeridionalAngle.Hint,
        n_bl0: n0.geo.NumBlades.Hint,
        wm0: n0.kin.W_mer.Hint,
        u0: n0.kin.BladeSpeed.Hint,
        slip0: n0.oth.SlipFactor.Hint,
    ):
        sin_cone = safe_abs(np.sin(mer_angle0))  # sin(gamma), Fig. 2
        cos_metal = np.cos(metal_ang0)
        sin_pitch = np.sin(np.pi / n_bl0)

        # Eq. (8)
        slip_target = (
            1 - (2 * sin_pitch / (np.pi * (1 + sin_pitch))) * sin_cone * cos_metal
        )
        r_slip = slip0 - slip_target

        # Eq. (5), reworked for the actual tangential velocity: ideal
        # relative Wt minus the Wiesner-style slip deficit (1 - mu) * U.
        wt_ideal = wm0 * np.tan(metal_ang0)
        wt_actual = wt_ideal - (1 - slip0) * u0
        r_incidence = wm0 * np.tan(beta0) - wt_actual

        return r_slip, r_incidence


class StanitzSlip(DeviationModel):
    """
    Rotor-inlet flow slip for radial/mixed-flow turbines, from the
    original centrifugal-compressor Stanitz correlation, Eq. (4) in H.
    Chen and N. C. Baines, "The aerodynamic loading of radial and
    mixed-flow turbines," Int. J. Mech. Sci. 36(1), pp. 63-79 (1994).
    Meant to be attached directly to a rotor ``BladeRow``'s own inlet node
    (position 0, i.e. ``StanitzSlip(): 0`` in its ``extra_equations``) --
    see ``ChenSlip`` for the alternate Chen & Baines (Eq. 8) correlation.
    """

    def residual(
        self,
        beta0: n0.kin.FlowAngleRel.Hint,
        metal_ang0: n0.geo.MetalAngle.Hint,
        mer_angle0: n0.geo.MeridionalAngle.Hint,
        n_bl0: n0.geo.NumBlades.Hint,
        wm0: n0.kin.W_mer.Hint,
        u0: n0.kin.BladeSpeed.Hint,
        slip0: n0.oth.SlipFactor.Hint,
    ):
        sin_cone = safe_abs(np.sin(mer_angle0))  # sin(gamma), Fig. 2
        cos_metal = np.cos(metal_ang0)

        # Eq. (4)
        slip_target = 1 - 0.63 * np.pi / n_bl0 * sin_cone * cos_metal
        r_slip = slip0 - slip_target

        # Eq. (5), reworked for the actual tangential velocity: ideal
        # relative Wt minus the Wiesner-style slip deficit (1 - mu) * U.
        wt_ideal = wm0 * np.tan(metal_ang0)
        wt_actual = wt_ideal - (1 - slip0) * u0
        r_incidence = wm0 * np.tan(beta0) - wt_actual

        return r_slip, r_incidence


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
        # Beyond 90 deg incidence (e.g. a strongly over-speeding rotor, flow
        # onto the blade suction side) the whole relative kinetic energy is
        # lost: cos clipped at 0 (continuous, and smooth since cos^n has zero
        # slope there), instead of a fractional power of a negative number
        cos_incidence = safe_max(np.cos(incidence), 0.0 * np.cos(incidence))

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
        mer_chord1: n1.geo.MerChord.Hint,
        n_bl1: n1.geo.NumBlades.Hint,
        w0: n0.kin.W_mag.Hint,
        w1: n1.kin.W_mag.Hint,
        s0: n0.stc.Entropy.Hint,
        p1: n1.stc.Pressure.Hint,
        h1_is: n1.oth.Enthalpy_Is.Hint,
        ds_prof1: n1.loss.Ds_profile.Hint,
    ):
        # Blade chord along the mean camber line, from the axial chord
        mean_tan_beta = (np.tan(beta2g) + np.tan(beta3g)) / 2
        chord = mer_chord1 / np.cos(np.arctan(mean_tan_beta))

        hyd_len = np.pi / 4 * ((mer_chord1 - h2 / 2) + (r2m - r3s - h3 / 2))

        hyd_diam = 0.5 * (
            (4 * np.pi * r2m * h2) / (n_bl1 * h2 + 2 * np.pi * r2m)
            + (2 * np.pi * (r3s**2 - r3h**2)) / (np.pi * (r3s - r3h) + n_bl1 * h3)
        )

        loading_term = 0.68 * (1 - (r3m / r2m) ** 2) * np.cos(beta3g) / h3 * chord

        deltah_base = 0.1 * (hyd_len / hyd_diam + loading_term) * 0.5 * (w0**2 + w1**2)

        # Higher loss coefficient for a shallow (mostly axial) exducer
        deltah = safe_if_else((r2m - r3s) / h3 >= 0.2, deltah_base, 2 * deltah_base)

        h1_lss = h1_is + deltah
        s1_profile = self.eos(h1_lss, p1)

        return ds_prof1 - (s1_profile - s0)


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
        mer_chord1: n1.geo.MerChord.Hint,
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
        Cr = (r3s / r2m) * (mer_chord1 - h2) / (vm1 * r3m * h3)

        # Cross term (sqrt of a product that can wander through zero away
        # from the converged solution, e.g. during the solver's initial
        # iterations): floor it away from 0 to keep its derivative bounded,
        # since d(sqrt(x))/dx diverges as x -> 0
        cross_term_sq = safe_max(gx_H * h2 * Cx * gr_H * h3 * Cr, 1e-9 / vm0**2)

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


# |> Radial-gap march stations (internal to RadialGapLoss below): static
# pressure, tangential velocity and entropy rise at each radial station
# past the gap inlet. Bookkeeping unknowns of a single loss model,
# co-located at the gap outlet node (as for EndwallLoss).
def _gap_station_specs(name: str, unit: str, guess: float, bounds) -> list[VarSpec]:
    return [
        VarSpec(f'{name}_stn{i}', unit, guess=guess, bounds=bounds, node=1)
        for i in range(1, 6)
    ]


# (pressures relative to the gap inlet one: the bounds keep the march off the
# meridionally supersonic root of the isentropic step, about 0.2 at low
# meridional Mach numbers)
_p_gap = _gap_station_specs('p_ratio_gap', '', 0.98, (0.5, 2.0))
_vt_gap = _gap_station_specs('vt_gap', 'm / s', 200.0, (1.0, 2e3))
_ds_gap = _gap_station_specs('ds_gap', 'J / kg / K', 0.1, (0.0, 500.0))[:4]
_p_gap_start = VarSpec('p_ratio_gap_stn0', '', 1.0, bounds=(0.5, 2.0), node=1)
GAP_START_RADIUS_RATIO = VarSpec('r_ratio_gap_start', '', 1.0, node=0)
"""Radius where the ``RadialGapLoss`` friction march starts, over the gap
inlet radius (boundary condition at the gap inlet)"""


class RadialGapLoss(LossModel):
    """
    Wall-friction entropy rise across the vaneless gap between the stator
    and the rotor of a radial-inflow turbine (node 0: gap inlet, node 1: gap
    outlet):
    the one-zone vaneless-space model of Stanitz with Japikse's
    friction coefficient, marched inwards over ``N_STATIONS`` equal radial
    steps. At each step from radius R_in to R:

    - isentropic step: free-vortex V_t and continuity (constant width b)
      at the inlet entropy give the static pressure p;
    - friction: Re = m / (2 mu R), C_f = k (1.8e5 / Re)^0.2 and the
      tangential momentum with the wall shear on both walls,

          V_t,in / V_t = R / R_in + 2 pi C_f rho V_t R_in (R_in - R) / m;

    - V_m = m / (2 pi R b rho) and the total enthalpy is conserved, so the
      reduced V_t raises the static enthalpy at p, hence the entropy.

    ``GAP_START_RADIUS_RATIO`` (R_start / R_0, boundary condition at the gap
    inlet, 1: from the gap inlet radius) starts the friction
    march at R_start instead of the gap inlet radius, the gap inlet state
    being carried there by one frictionless isentropic step (free vortex,
    continuity): e.g. at the stator trailing edge when the gap inlet node is
    a mixed-out plane inside the gap. The march then covers the walls from
    the trailing edge to the gap outlet.
    """

    N_STATIONS = 5
    # Station unknowns (at the gap outlet node), e.g. for seeding guesses;
    # the last station's entropy rise is ``Ds_endwall``
    STATION_PRESSURES = tuple(_p_gap)
    STATION_V_TANS = tuple(_vt_gap)
    STATION_ENTROPIES = tuple(_ds_gap)
    START_PRESSURE = _p_gap_start

    config = EquationConfig(
        input_pair=cp.PSmass_INPUTS,
        out_properties=(thrm.Enthalpy, thrm.Density),
    )

    def residual(
        self,
        rr0: n0.geo.Rmid.Hint,
        rr1: n1.geo.Rmid.Hint,
        hgt1: n1.geo.Height.Hint,
        p0: n0.stc.Pressure.Hint,
        s0: n0.stc.Entropy.Hint,
        h0: n0.stc.Enthalpy.Hint,
        rho0: n0.stc.Density.Hint,
        vt0: n0.kin.V_tan.Hint,
        vm0: n0.kin.V_mer.Hint,
        mf0: n0.oth.TotMassFlow.Hint,
        k_fric0: n0.oth.DiffFricConst.Hint,
        mu0: n0.oth.BLViscosity.Hint,
        start_ratio0: GAP_START_RADIUS_RATIO.Hint,
        p_start: _p_gap_start.Hint,
        p_1: _p_gap[0].Hint,
        p_2: _p_gap[1].Hint,
        p_3: _p_gap[2].Hint,
        p_4: _p_gap[3].Hint,
        p_5: _p_gap[4].Hint,
        vt_1: _vt_gap[0].Hint,
        vt_2: _vt_gap[1].Hint,
        vt_3: _vt_gap[2].Hint,
        vt_4: _vt_gap[3].Hint,
        vt_5: _vt_gap[4].Hint,
        ds_1: _ds_gap[0].Hint,
        ds_2: _ds_gap[1].Hint,
        ds_3: _ds_gap[2].Hint,
        ds_4: _ds_gap[3].Hint,
        Ds_endwall1: n1.loss.Ds_endwall.Hint,
    ):
        n_st = self.N_STATIONS
        ht0 = h0 + (vt0**2 + vm0**2) / 2

        # Gap inlet state carried to the march start radius (isentropic)
        r_start = rr0 * start_ratio0
        p_st = p0 * p_start
        h_st, rho_st = self.eos(p_st, s0)
        vt_st = vt0 * rr0 / r_start
        vm_st = vm0 * rho0 / rho_st * rr0 / r_start
        residuals = [h_st + (vt_st**2 + vm_st**2) / 2 - ht0]

        radii = [r_start + (rr1 - r_start) * i / n_st for i in range(n_st + 1)]
        pressures = [p0 * ratio for ratio in (p_start, p_1, p_2, p_3, p_4, p_5)]
        v_tans = [vt_st, vt_1, vt_2, vt_3, vt_4, vt_5]
        ds_st = [0.0 * s0, ds_1, ds_2, ds_3, ds_4, Ds_endwall1]
        rho_prev, vm_prev = rho_st, vm_st

        for i in range(1, n_st + 1):
            r_in, r_i = radii[i - 1], radii[i]
            p_i, vt_i = pressures[i], v_tans[i]

            # Isentropic step from the previous station: static pressure
            h_is, rho_is = self.eos(p_i, s0 + ds_st[i - 1])
            vt_is = v_tans[i - 1] * r_in / r_i
            vm_is = vm_prev * rho_prev / rho_is * r_in / r_i
            residuals.append(h_is + (vt_is**2 + vm_is**2) / 2 - ht0)

            # Friction: tangential momentum and energy at that pressure
            h_i, rho_i = self.eos(p_i, s0 + ds_st[i])
            reynolds = mf0 / (2 * mu0 * r_i)
            if hasattr(reynolds, 'to'):
                # (unit check) reduce before the fractional power
                reynolds = reynolds.to('')
            c_fric = k_fric0 * (1.8e5 / reynolds) ** 0.2
            shear = 2 * np.pi * c_fric * rho_i * vt_i * r_in * (r_in - r_i) / mf0
            residuals.append(v_tans[i - 1] - vt_i * (r_i / r_in + shear))

            vm_i = mf0 / (2 * np.pi * r_i * hgt1 * rho_i)
            residuals.append(h_i + (vt_i**2 + vm_i**2) / 2 - ht0)

            rho_prev, vm_prev = rho_i, vm_i

        return tuple(residuals)
