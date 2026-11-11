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
        t_values = [i / n_stations for i in range(n_stations + 1)]

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
        sin_cone = safe_abs(np.sin(mer_angle0))  # ty: ignore
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
        sin_cone = safe_abs(np.sin(mer_angle0))  # ty: ignore
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


def stodola_opt_incidence(alpha_in, n_blades):
    """Optimal relative inflow angle from the absolute inflow angle, Persky &
    Sauret (2019) Eq. A.2b (Stodola-type correlation, ``beta_in,opt,2``).
    """
    return np.arctan(-1.98 * np.tan(alpha_in) / (n_blades * (1 - 1.98 / n_blades)))


class ImpellerIncidenceLoss(LossModel):
    """
    Rotor incidence loss for radial-inflow turbines, Persky & Sauret (2019)
    Table 1 II.1 with the optimal relative angle ``beta_in,opt,2`` (Eq. A.2b):

    C.A. Wasserbauer, A.J. Glassman, Fortran Program for Predicting Oﬀ-design
    Performance of Radial-inﬂow Turbines, Tech. Rep., National Aeronautics and Space
    Administration, 1975.

        Delta h_i = 1/2 W_in^2 |sin(beta_in - beta_in,opt)|^n

    The exponent switches with the sign of the incidence, as in the NASA-based
    correlation of [Ref 1]: positive (over-turning) incidence uses
    ``exponent_pos`` and negative (under-turning) incidence uses
    ``exponent_neg``. Positive incidence is the more dissipative case.


    Parameters
    ----------
    exponent_neg : float
        Exponent for negative incidence. Persky & Sauret: 2.
    exponent_pos : float
        Exponent for positive incidence. Persky & Sauret: 3.
    """

    config = EquationConfig(
        input_pair=cp.HmassP_INPUTS,
        out_properties=(thrm.Entropy,),
    )

    def __init__(
        self, exponent_neg: float = 2.0, exponent_pos: float = 3.0, scaling_factor=None
    ):
        self.exponent_neg = exponent_neg
        self.exponent_pos = exponent_pos
        super().__init__(scaling_factor)

    def residual(
        self,
        beta0: n0.kin.FlowAngleRel.Hint,
        alpha0: n0.kin.FlowAngleAbs.Hint,
        n_bl1: n1.geo.NumBlades.Hint,
        W0: n0.kin.W_mag.Hint,
        s0: n0.stc.Entropy.Hint,
        p1: n1.stc.Pressure.Hint,
        h1_is: n1.oth.Enthalpy_Is.Hint,
        Ds_inc1: n1.loss.Ds_incidence.Hint,
    ):
        beta_opt = stodola_opt_incidence(alpha0, n_bl1)
        incidence = beta0 - beta_opt
        sin_incidence = safe_abs(np.sin(incidence))

        # Sign of the incidence selects the exponent (not the loss value's
        # sign), so that unit probing works
        Deltah_neg = W0**2 / 2 * sin_incidence**self.exponent_neg
        Deltah_pos = W0**2 / 2 * sin_incidence**self.exponent_pos
        Deltah = safe_if_else(incidence <= 0, Deltah_neg, Deltah_pos)

        h1_lss = h1_is + Deltah
        s1_incidence = self.eos(h1_lss, p1)

        return Ds_inc1 - (s1_incidence - s0)


class ImpellerPassageLoss(LossModel):
    """
    Radial-inflow turbine impeller passage loss (friction, blade loading
    and secondary flow) from `A MEANLINE PREDICTION METHOD FOR RADIAL
    TURBINE EFFICIENCY`, Baines 1998 [Ref 8]"""

    config = EquationConfig(
        input_pair=cp.HmassP_INPUTS,
        out_properties=(thrm.Entropy,),
    )

    def __init__(self, passage_coeff: float = 0.1, scaling_factor=None):
        self.passage_coeff = passage_coeff
        super().__init__(scaling_factor)

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
        ax_len1: n1.geo.AxialLength.Hint,
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
        chord = ax_len1 / np.cos(np.arctan(mean_tan_beta))

        hyd_len = np.pi / 4 * ((ax_len1 - h2 / 2) + (r2m - r3s - h3 / 2))

        hyd_diam = 0.5 * (
            (4 * np.pi * r2m * h2) / (n_bl1 * h2 + 2 * np.pi * r2m)
            + (2 * np.pi * (r3s**2 - r3h**2)) / (np.pi * (r3s - r3h) + n_bl1 * h3)
        )

        loading_term = 0.68 * (1 - (r3m / r2m) ** 2) * np.cos(beta3g) / h3 * chord
        viscous_term = hyd_len / hyd_diam

        deltah_base = (
            self.passage_coeff * (viscous_term + loading_term) * (w0**2 + w1**2) / 2
        )

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
        Cx = (1 - (r3s / r2m)) / (vm0 * h2)
        Cr = (r3s / r2m) * (mer_chord1 - h2) / (vm1 * r3m * h3)

        # Cross term (sqrt of a product that can wander through zero away
        # from the converged solution, e.g. during the solver's initial
        # iterations): floor it away from 0 to keep its derivative bounded,
        # since d(sqrt(x))/dx diverges as x -> 0
        cross_term_sq = safe_max(cl0 * Cx * cl1 * Cr, 1e-9 / vm0**2)

        # Clearance loss coefficient (corrected from [Ref 8]: u0 ** 5 -->
        # u0 ** 3 to obtain a dimensionally correct coefficient)
        Deltah = (
            (u0**3 * n_bl1)
            / (8 * np.pi)
            * (self.Kx * cl0 * Cx + self.Kr * cl1 * Cr + self.Kxr * cross_term_sq**0.5)
        )

        h1_lss = h1_is + Deltah
        s1_leakage = self.eos(h1_lss, p1)

        return Ds_leak1 - (s1_leakage - s0)


class QiClearanceLoss(LossModel):
    """Tip clearance loss, Qi et al. (2017), Eq. (14), the response-surface fit
    to Futral et al. data that Persky & Sauret (2019) label I.4:

        Delta h_tip = Delta h_0 * (0.09678 A + 1.69997 R - 0.096844 R^2
                                   + 0.03379 A R) / 100

    with A and R the axial and radial clearance ratios in %, and Delta h_0 the
    total-to-total enthalpy drop, h_0in - h_tt,is,out.

    Normalisation (from Qi et al., Sec. 2.2.1 and Fig. 3):
    - A = 100 * eps_a / b_in, the axial gap over the rotor inlet blade height.
      Qi's example (A = 9 %, eps_a = 0.1 mm, b_4 = 1.11 mm) confirms this.
    - R = 100 * eps_r / b_out, the radial gap over the rotor exit blade height.
      Qi's text does not give this denominator explicitly; b_out is my reading.

    Note: the Persky & Sauret Table 1 coefficients (1.355 - 0.0837 A - ...) do
    not match Qi's Eq. (14), so Qi's equation is used here."""

    config = EquationConfig(
        input_pair=cp.HmassP_INPUTS,
        out_properties=(thrm.Entropy,),
    )

    def residual(
        self,
        h0_tot: n0.tot.Enthalpy.Hint,
        eps_ax0: n0.geo.TipClearance.Hint,
        b_in0: n0.geo.Height.Hint,
        eps_rad1: n1.geo.TipClearance.Hint,
        b_out1: n1.geo.Height.Hint,
        h1_tt_is: n1.oth.Enthalpy_totIs.Hint,
        s0: n0.stc.Entropy.Hint,
        p1: n1.stc.Pressure.Hint,
        h1_is: n1.oth.Enthalpy_Is.Hint,
        Ds_leak1: n1.loss.Ds_leakage.Hint,
    ):
        A = 100 * eps_ax0 / b_in0
        R = 100 * eps_rad1 / b_out1
        frac = (0.09678 * A + 1.69997 * R - 0.096844 * R**2 + 0.03379 * A * R) / 100
        Deltah = frac * (h0_tot - h1_tt_is)

        s1_clearance = self.eos(h1_is + Deltah, p1)
        return Ds_leak1 - (s1_clearance - s0)


class SauretWindageLoss(LossModel):
    """Disc windage (rotor back-face friction), Persky & Sauret (2019) Table 1
    I.1 with Daily & Nece coefficients Cm.1 (Table A.6), Eqs. A.7 and A.8c:

        Cm = C (eps / r)^m / Re^n,    Re = rho U r / mu
        Delta h_w = Cm rho U^3 r^2 / (4 mdot)

    Laminar coefficients (3.7, 0.1, 0.5) apply below Re = 1e5, turbulent
    coefficients (0.0102, 0.2, 0.2) above it.

    TODO(windage clearance): the back-face clearance ``eps`` is taken as the
    rotor-inlet tip clearance. The paper gives no separate value for this turbine.
    TODO(windage factor): Eq. A.8a gives M = 0.5 Cm rho w^2 r^5 (two faces), but
    Eq. A.8b and A.8c use 0.25. The 0.25 of A.8c is used here; confirm it.
    """

    config = EquationConfig(
        input_pair=cp.HmassP_INPUTS,
        out_properties=(thrm.Entropy,),
    )

    RE_CROSSOVER = 1e5

    def residual(
        self,
        rho0: n0.stc.Density.Hint,
        mu0: n0.stc.Viscosity.Hint,
        u0: n0.kin.BladeSpeed.Hint,
        r_in0: n0.geo.Rmid.Hint,
        eps0: n0.geo.TipClearance.Hint,
        mdot0: n0.oth.TotMassFlow.Hint,
        s0: n0.stc.Entropy.Hint,
        p1: n1.stc.Pressure.Hint,
        h1_is: n1.oth.Enthalpy_Is.Hint,
        Ds_wind1: n1.loss.Ds_secondary.Hint,
    ):
        Re = rho0 * u0 * r_in0 / mu0
        clearance_ratio = eps0 / r_in0

        cm_laminar = 3.7 * clearance_ratio**0.1 / Re**0.5
        cm_turbulent = 0.0102 * clearance_ratio**0.2 / Re**0.2
        cm = safe_if_else(Re < self.RE_CROSSOVER, cm_laminar, cm_turbulent)

        Deltah = cm * rho0 * u0**3 * r_in0**2 / (2 * mdot0)

        s1_windage = self.eos(h1_is + Deltah, p1)
        return Ds_wind1 - (s1_windage - s0)


class GlassmanMeitnerBlayer(LossModel):
    def residual(self): ...
