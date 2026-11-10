"""
Rotor loss models of Persky & Sauret (2019), "Loss models for on and off-design
performance of radial inflow turbomachinery", Applied Thermal Engineering 150,
1066-1077 (``2019_Sauret_LossModelsRadial.pdf``).

The paper enumerates ~1.5 million loss-model configurations, scores each one
against CFD over several operating conditions (O1-O4 error metrics, Eqs. 2-4),
and selects the best. The best configuration (Table 5, "Best Model") is:

    Incidence   II.1, beta_in,opt,2     -> SauretIncidenceLoss
    Passage     I.2,  sigma_3           -> SauretPassageLoss
    Clearance   I.4                     -> SauretClearanceLoss
    Exit        I.1                     -> not applied (total-to-total efficiency)
    Trailing    II.2                    -> not implemented (see below)
    Windage     I.1, Cm.1               -> SauretWindageLoss

Clearance I.4 is implemented from Qi et al. (2017), Eq. (14), with A and R the
axial and radial clearance ratios in %. The R denominator is an inference (see
the class).

TODO(trailing edge II.2): not implemented. Its relative total pressure is
referenced to the inlet entropy S_0,in (Eq. A.5b). Over the whole rotor that
counts every other entropy source a second time, so the reference state is
undefined here. The paper's best model gives it 1-2 % of the losses.

TODO(exit I.1): not applied. It is a loss of the kinetic energy at exit, which
is outside the total-to-total efficiency used here. The exit kinetic energy is
shown separately in the example's loss breakdown.

All models return the entropy rise in J/kg/K, following the other ``adet``
loss models: the enthalpy loss Delta h is added to the isentropic exit state
at p1 and the resulting entropy is compared with the inlet entropy.
"""

import CoolProp as cp
import numpy as np

from adet.equations.base_equation import EquationConfig, LossApplier
from adet.equations.utils import safe_abs, safe_if_else
from adet.losses.base_loss import LossModel
from adet.variables import NodeVariables, ThermoVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)
thrm = ThermoVariables()


def sauret_beta_opt_incidence(alpha_in, n_blades):
    """Optimal relative inflow angle from the absolute inflow angle, Eq. A.2b
    (Stodola-type correlation, ``beta_in,opt,2``).

    tan(beta_opt) = -1.98 tan(alpha_in) / (Z_r (1 - 1.98 / Z_r))
    """
    return np.arctan(-1.98 * np.tan(alpha_in) / (n_blades * (1 - 1.98 / n_blades)))


class AddSauretRotorLosses(LossApplier):
    """Sum the rotor loss mechanisms of the Sauret best model onto the static
    entropy rise. The loss slots are reused from the existing rotor
    (passage -> profile, clearance -> leakage, windage -> secondary)."""

    def residual(
        self,
        s0: n0.stc.Entropy.Hint,
        s1: n1.stc.Entropy.Hint,
        ds_passage1: n1.loss.Ds_profile.Hint,
        ds_incidence1: n1.loss.Ds_incidence.Hint,
        ds_clearance1: n1.loss.Ds_leakage.Hint,
        ds_windage1: n1.loss.Ds_secondary.Hint,
        ds_mixing1: n1.loss.Ds_mixing.Hint,
    ):
        return s1 - (
            s0 + ds_passage1 + ds_incidence1 + ds_clearance1 + ds_windage1 + ds_mixing1
        )


class SauretIncidenceLoss(LossModel):
    """Incidence loss, Table 1 II.1 with ``beta_in,opt,2`` (Eq. A.2b):

        Delta h_i = 1/2 W_in^2 (beta_in - beta_in,opt)^2

    The relative inflow angle is measured from radial, as in the paper.
    """

    config = EquationConfig(
        input_pair=cp.HmassP_INPUTS,
        out_properties=(thrm.Entropy,),
    )

    def residual(
        self,
        beta0: n0.kin.FlowAngleRel.Hint,
        alpha0: n0.kin.FlowAngleAbs.Hint,
        metal_ang0: n0.geo.MetalAngle.Hint,
        W0: n0.kin.W_mag.Hint,
        n_bl1: n1.geo.NumBlades.Hint,
        s0: n0.stc.Entropy.Hint,
        p1: n1.stc.Pressure.Hint,
        h1_is: n1.oth.Enthalpy_Is.Hint,
        Ds_inc1: n1.loss.Ds_incidence.Hint,
    ):
        beta_opt = sauret_beta_opt_incidence(alpha0, n_bl1)
        inc = beta0 - metal_ang0 - beta_opt
        # Table 1 II.2 exponent: n = 3 for positive incidence, 2 for negative.
        # Select between the loss values (not the exponent) so that unit probing
        # works, as in ImpellerIncidenceLoss.
        Deltah_neg = W0**2 / 2 * np.sin(beta0 - beta_opt) ** 2
        Deltah_pos = W0**2 / 2 * np.sin(beta0 - beta_opt) ** 3
        Deltah = safe_if_else(inc > 0, Deltah_pos, Deltah_neg)

        s1_incidence = self.eos(h1_is + Deltah, p1)
        return Ds_inc1 - (s1_incidence - s0)


class SauretPassageLoss(LossModel):
    """Passage loss, Table 1 I.2 with the slip-factor optimal angle sigma_3:

        Delta h_p = 1/2 K (W_in^2 cos^2(beta_in - beta_in,opt) + W_out^2)

    The slip factor is sigma_3 = 1 - sqrt(beta_blade) / Z_r^0.7 (Eq. A.1c).
    The optimal flow angle follows from the slip relation C_theta = sigma U,
    i.e. W_theta,opt = (sigma - 1) U and tan(beta_in,opt) = W_theta,opt / C_m.
    For a radial blade (beta_blade = 0) this gives sigma = 1 and beta_opt = 0.

    Parameters
    ----------
    K : float
        Passage loss coefficient.

    TODO(passage K): the paper does not tabulate K for Passage I.2. The default
    0.1 is the value of the existing ``ImpellerPassageLoss``; it needs calibration.
    TODO(slip-to-angle): the mapping from sigma_3 to beta_in,opt above is my
    reading of the paper's "procedure from Whitfield". Verify it against Whitfield.
    """

    config = EquationConfig(
        input_pair=cp.HmassP_INPUTS,
        out_properties=(thrm.Entropy,),
    )

    def __init__(self, K: float = 0.1, scaling_factor=None):
        self.K = K
        super().__init__(scaling_factor)

    def residual(
        self,
        beta0: n0.kin.FlowAngleRel.Hint,
        metal0: n0.geo.MetalAngle.Hint,
        alpha0: n0.kin.FlowAngleAbs.Hint,
        n_bl: n0.geo.NumBlades.Hint,
        u0: n0.kin.BladeSpeed.Hint,
        vm0: n0.kin.V_mer.Hint,
        W0: n0.kin.W_mag.Hint,
        W1: n1.kin.W_mag.Hint,
        s0: n0.stc.Entropy.Hint,
        p1: n1.stc.Pressure.Hint,
        h1_is: n1.oth.Enthalpy_Is.Hint,
        Ds_prof1: n1.loss.Ds_profile.Hint,
    ):
        # Small offset keeps d(sqrt)/d(beta) finite at a radial blade (beta = 0)
        sigma3 = 1 - np.sqrt(safe_abs(metal0) + 1e-12) / n_bl**0.7
        beta_opt = np.arctan((sigma3 - 1) * u0 / vm0)
        # beta_opt = sauret_beta_opt_incidence(alpha0, n_bl)

        Deltah = 0.5 * self.K * (W0**2 * np.cos(beta0 - beta_opt) ** 2 + W1**2)

        s1_passage = self.eos(h1_is + Deltah, p1)
        return Ds_prof1 - (s1_passage - s0)


class SauretClearanceLoss(LossModel):
    """Tip clearance loss, Qi et al. (2017), Eq. (14), the response-surface fit
    to Futral et al. data that Persky & Sauret label I.4:

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
    """Disc windage (rotor back-face friction), Table 1 I.1 with Daily & Nece
    coefficients Cm.1 (Table A.6), Eqs. A.7 and A.8c:

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


class SauretMixingLoss(LossModel):
    """Trailing-edge loss from the rotor exit blockage, Qi et al. (2017),
    Eqs. (18)-(19). Persky & Sauret label this term Trailing Edge II.2 and it
    is the mixing slot of the rotor here (the two names are used
    interchangeably).

        dP_0,rel = rho W^2 / (2 g_c) * (Z t_t / (pi (r_6t + r_6h) cos beta_6))^2
        dh_te    = dP_0,rel / rho

    so that dh_te = W^2/2 * (Z t_t / (pi (r_6t + r_6h) cos beta_6))^2. The
    closed form needs no reference pressure, so the reference-state problem of
    the earlier version (rotor-inlet relative total against exit absolute total)
    does not arise.

    Parameters
    ----------
    thickness_factor : float
        t_t / t, ratio of the trailing-edge thickness to the blade thickness.
        Qi et al. taper the trailing edge to t_t = t / 2 (their Fig. 2).

    TODO(trailing-edge thickness): the T-100 trailing edge thickness is not
    given in Sauret's tables. It is taken as thickness_factor * BldThick at
    the rotor exit, with factor 0.5 from Qi. Check against the geometry.
    TODO(exit relative angle): beta_6 is taken as the rotor exit relative flow
    angle (n1.kin.FlowAngleRel), and W as the exit relative velocity
    (n1.kin.W_mag). Qi use the span-RMS values (beta_6rms, W_6rms). The T-100
    model is mid-span only, so these may differ from the RMS values.
    TODO(mixing reference, superseded): an earlier form, dP = p_rlt0 -
    P(H = ht1, S = s0), gave about 287 J/kg/K because it compared the
    rotor-inlet relative total with the exit absolute total at inlet entropy,
    so the expansion pressure ratio was counted as loss. Removed in favour of
    the closed form above.
    """

    config = EquationConfig(
        input_pair=cp.HmassP_INPUTS,
        out_properties=(thrm.Entropy,),
    )

    def __init__(self, thickness_factor: float = 0.5, scaling_factor=None):
        self.thickness_factor = thickness_factor
        super().__init__(scaling_factor)

    def residual(
        self,
        W1: n1.kin.W_mag.Hint,
        beta1: n1.kin.FlowAngleRel.Hint,
        r_tip1: n1.geo.Rtip.Hint,
        r_hub1: n1.geo.Rhub.Hint,
        n_bl1: n1.geo.NumBlades.Hint,
        bld_thick1: n1.geo.BldThick.Hint,
        s0: n0.stc.Entropy.Hint,
        p1: n1.stc.Pressure.Hint,
        h1_is: n1.oth.Enthalpy_Is.Hint,
        Ds_mix1: n1.loss.Ds_mixing.Hint,
    ):
        t_te = self.thickness_factor * bld_thick1
        blockage = n_bl1 * t_te / (np.pi * (r_tip1 + r_hub1) * np.cos(beta1))
        Deltah = W1**2 / 2 * blockage**2

        s1_te = self.eos(h1_is + Deltah, p1)
        return Ds_mix1 - (s1_te - s0)


def comparison_metrics(ua0_model, eta_model, ua0_ref, eta_ref) -> dict[str, float]:
    """Error metrics of the paper (Eqs. 2a-2d) for one reference data set.

    The model speedline is interpolated at the reference abscissae; reference
    points outside the converged model range are discarded. Returned values:

    - ``O1``: integral of |error| over U/a0 [eta points * U/a0]
    - ``O2``: mean error [eta points]
    - ``O3``: RMSE [eta points]
    - ``O4``: mean |d(error)/d(U/a0)| [eta points per U/a0], a trend error

    Efficiencies are fractions, so ``O2``, ``O3`` and ``O4`` are multiplied by
    100 to read in percentage points. ``O1`` is multiplied by 100 as well.
    """
    ua0_model = np.asarray(ua0_model, dtype=float)
    eta_model = np.asarray(eta_model, dtype=float)
    ua0_ref = np.asarray(ua0_ref, dtype=float)
    eta_ref = np.asarray(eta_ref, dtype=float)

    order = np.argsort(ua0_model)
    ua0_model, eta_model = ua0_model[order], eta_model[order]
    inside = (ua0_ref >= ua0_model[0]) & (ua0_ref <= ua0_model[-1])
    ua0_ref, eta_ref = ua0_ref[inside], eta_ref[inside]

    # Repeated abscissae (e.g. two measurements at the same U/a0) are averaged,
    # so the trend metric O4 is well defined
    ua0_ref, inverse = np.unique(ua0_ref, return_inverse=True)
    eta_ref = np.bincount(inverse, weights=eta_ref) / np.bincount(inverse)
    error = np.interp(ua0_ref, ua0_model, eta_model) - eta_ref

    trapezoid = getattr(np, 'trapezoid', None) or np.trapz
    return {
        'O1': 100 * float(trapezoid(np.abs(error), ua0_ref)),
        'O2': 100 * float(np.mean(error)),
        'O3': 100 * float(np.sqrt(np.mean(error**2))),
        'O4': 100 * float(np.mean(np.abs(np.gradient(error, ua0_ref)))),
        'n_points': len(ua0_ref),
    }
