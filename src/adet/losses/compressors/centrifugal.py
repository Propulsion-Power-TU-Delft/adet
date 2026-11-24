"""Compressor loss models for centrifugal impellers: incidence, blade loading, clearance, recirculation, leakage, friction and mixing."""

import numpy as np

from adet.equations.base_equation import (
    EquationBase,
)
from adet.equations.utils import (
    safe_abs,
    safe_if_else,
    safe_max,
    safe_mean,
    safe_min,
    safe_sign,
)
from adet.losses.base_loss import LossModel
from adet.variables import NodeVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)


class CaseyRushInletFunc(EquationBase):
    # TODO: This can be generalized instead of using the
    # ideal gas expressions with gamma(pv)
    def residual(
        self,
        m_tip0: n0.kin.RelMach_tip.Hint,
        beta_tip0: n0.kin.Beta_tip.Hint,
        cp1: n1.stc.Cp.Hint,
        cv1: n1.stc.Cv.Hint,
    ):
        gamma = cp1 / cv1
        first_term = 3 + gamma * m_tip0**2 + 2 * m_tip0
        second_term = 3 + gamma * m_tip0**2 - 2 * m_tip0
        rhs = (first_term**0.5 - second_term**0.5) / (2 * m_tip0)

        return np.cos(beta_tip0) - rhs


class CompressorShapeFactor(EquationBase):
    def residual(
        self,
        rhub0: n0.geo.Rhub.Hint,
        rtip0: n0.geo.Rtip.Hint,
        k_shape0: n0.geo.ShapeCoeff.Hint,
    ):
        return k_shape0 - (1 - (rhub0 / rtip0) ** 2)


class BladeLoadingCoppage(LossModel):
    def residual(
        self,
        ht0: n0.tot.Enthalpy.Hint,
        ht1: n1.tot.Enthalpy.Hint,
        rtip0: n0.geo.Rtip.Hint,
        rmid1: n1.geo.Rmid.Hint,
        u1: n1.kin.BladeSpeed.Hint,
        w1: n1.kin.W_mag.Hint,
        w_tip0: n0.kin.W_tip.Hint,
        n_bl_eff1: n1.geo.NumBladesEff.Hint,
        bl_coeff1: n1.oth.BlLoadingCoeff.Hint,
        dht_load1: n1.loss.Dht_loading.Hint,
    ):
        w_ratio = w1 / w_tip0
        work = ht1 - ht0
        r_ratio = rtip0 / rmid1
        diff_fact = (
            1
            - w_ratio
            + bl_coeff1
            * (work / u1**2)
            * w_ratio
            / (n_bl_eff1 / np.pi * (1 - r_ratio) + 2 * r_ratio)
        )

        return dht_load1 - 0.05 * (diff_fact * u1) ** 2


class ClearanceJansen(LossModel):
    def residual(
        self,
        vm0: n0.kin.V_mer.Hint,
        vt1: n1.kin.V_tan.Hint,
        hgt1: n1.geo.Height.Hint,
        rhub0: n0.geo.Rhub.Hint,
        rtip0: n0.geo.Rtip.Hint,
        rr1: n1.geo.RDistr.Hint,
        n_bl_eff1: n1.geo.NumBladesEff.Hint,
        cl0: n0.geo.TipClearance.Hint,
        cl1: n1.geo.TipClearance.Hint,
        rho0: n0.stc.Density.Hint,
        rho1: n1.stc.Density.Hint,
        dht_cl1: n1.loss.Dht_clearance.Hint,
    ):
        clearance = (cl0 + cl1) / 2
        K = safe_abs((rtip0**2 - rhub0**2) / ((rr1 - rtip0) * (1 + rho1 / rho0)))
        abs_vt1 = safe_abs(vt1)

        return dht_cl1 - (
            0.6
            * clearance
            / hgt1
            * abs_vt1
            * ((4 * np.pi / (hgt1 * n_bl_eff1)) * K * abs_vt1 * vm0) ** 0.5
        )


class ClearanceBrasz(LossModel):
    def residual(
        self,
        rr1: n1.geo.RDistr.Hint,
        rhub0: n0.geo.Rhub.Hint,
        rtip0: n0.geo.Rtip.Hint,
        hgt1: n1.geo.Height.Hint,
        cl0: n0.geo.TipClearance.Hint,
        cl1: n1.geo.TipClearance.Hint,
        n_bl_eff1: n1.geo.NumBladesEff.Hint,
        vt1: n1.kin.V_tan.Hint,
        vm0: n0.kin.V_mer.Hint,
        rho0: n0.stc.Density.Hint,
        rho1: n1.stc.Density.Hint,
        dht_cl1: n1.loss.Dht_clearance.Hint,
    ):
        clearance = (cl0 + cl1) / 2

        K = (rtip0**2 - rhub0**2) / ((rr1 - rtip0) * (1 + rho1 / rho0))
        abs_vt1 = safe_abs(vt1)

        return dht_cl1 - (
            0.6
            * clearance
            * abs_vt1
            / (hgt1 + clearance / 2)
            * np.sqrt(
                4 * np.pi * abs_vt1 * vm0 * K / ((hgt1 + clearance / 2) * n_bl_eff1)
            )
        )


class IncidenceVDB(LossModel):
    """Van den Braembussche incidence model"""

    def residual(
        self,
        beta_opt0: n0.kin.BetaOpt.Hint,
        beta0: n0.kin.FlowAngleRel.Hint,
        metal_ang0: n0.geo.MetalAngle.Hint,
        m_rel0: n0.kin.RelMach.Hint,
        w_tip0: n0.kin.W_tip.Hint,
        dht_inc1: n1.loss.Dht_incidence.Hint,
    ):
        incidence = beta0 - metal_ang0
        incidence_opt = beta_opt0 - metal_ang0

        incidence *= safe_sign(metal_ang0)
        incidence_opt *= safe_sign(metal_ang0)

        C_te = safe_if_else(incidence > incidence_opt, 2.5, 2.0)

        delta_i = 2.5 + 0.15 * (12.5 - 0.1 * beta0) * (m_rel0 - 1.2) ** 2 / 2

        incidence_diff = incidence - incidence_opt
        dht = (
            0.833 * (incidence_diff / (C_te * delta_i)) ** 2
            + 0.1667 * incidence_diff / (C_te * delta_i)
        ) * w_tip0**2

        return dht_inc1 - dht


class IncidenceGalvas(LossModel):
    def residual(
        self,
        w0: n0.kin.W_mag.Hint,
        beta0: n0.kin.FlowAngleRel.Hint,
        beta_opt0: n0.kin.BetaOpt.Hint,
        inc_coeff0: n0.oth.IncCoeff.Hint,
        dht_inc1: n1.loss.Dht_incidence.Hint,
    ):
        lost_angle = safe_abs(beta_opt0 - beta0)
        lost_wt = w0 * np.sin(lost_angle)

        return dht_inc1 - inc_coeff0 * lost_wt**2 / 2


class MixingJohnstonDean(LossModel):
    """
    Johnston and Dean (1966) - Losses in Vaneless Diffusers of Centrifugal
    Compressors and Pumps: Analysis, Experiment, and Design.

    Sudden-expansion mixing loss of the impeller discharge wakes at the
    diffuser inlet: a square-wave jet-wake profile leaving the blade passages
    mixes out suddenly to a uniform, axisymmetric flow (Sudden Expansion
    Analysis). Only the meridional velocity component is mixed out, the
    tangential component is preserved.

    Notation, Johnston and Dean -> adet:
    - C_i         -> V_mag, impeller outlet (diffuser inlet) velocity
    - lambda_i    -> tan(FlowAngleAbs), swirl parameter C_theta / C_m
    - eps         -> WakeFrac, fraction of the blade-to-blade spacing
                     occupied by the wake
    - B = b_i/b_t -> MixDepthRatio, diffuser inlet depth / impeller tip depth

    Assumptions of the paper: no relative flow in the wake (W_w = 0), constant
    static pressure around the impeller rim, Z >= 10 blades, incompressible
    flow, B >= 1 (footnote 12). The loss is applied as an enthalpy loss.

    NB: Wake fraction computation is NOT from Johnston and Dean. The paper
    takes eps as an input and states that its prediction is intentionally
    ignored (footnote 23); its only indications relate significant wakes
    (eps > 0.25) to low flow, near surge (p. 58), and mixing losses to a
    swirl parameter above a critical value (pp. 59-60).

    Single-node equation at the impeller outlet, e.g. `MixingJohnstonDean(): 1`
    or `MixingJohnstonDean(ramp='geometric'): 1`.
    """

    def __init__(self, ramp: str = 'linear', **kwargs):
        super().__init__(**kwargs)
        if ramp not in ('linear', 'geometric'):
            raise ValueError(f"ramp must be 'linear' or 'geometric', got {ramp!r}")
        self.ramp = ramp

    def residual(
        self,
        v0: n0.kin.V_mag.Hint,
        alpha0: n0.kin.FlowAngleAbs.Hint,
        cum_mf0: n0.oth.TotMassFlow.Hint,
        mf_choke0: n0.oth.ChokeMassflow.Hint,
        wake_frac0: n0.oth.WakeFrac.Hint,
        min_wake0: n0.oth.MinWakeFrac.Hint,
        max_wake0: n0.oth.MaxWakeFrac.Hint,
        mf_thres0: n0.oth.WakeFracThres.Hint,
        depth_ratio0: n0.oth.MixDepthRatio.Hint,
        dht_mix0: n0.loss.Dht_mixing.Hint,
    ):
        # Wake fraction ramp, MinWakeFrac at WakeFracThres ChokeMassflow (ramp
        # position 0) and MaxWakeFrac at ChokeMassflow (ramp position 1)
        ramp_pos = (cum_mf0 - mf_thres0 * mf_choke0) / ((1 - mf_thres0) * mf_choke0)
        if self.ramp == 'geometric':
            # Ramp position limited to [0, 1]: both if_else branches are evaluated,
            # the power would otherwise overflow far from the solution
            ramp_pos_lim = safe_max(0.0, safe_min(ramp_pos, 1.0))
            ramp_wake_frac = min_wake0 * (max_wake0 / min_wake0) ** ramp_pos_lim
        else:
            ramp_wake_frac = min_wake0 + (max_wake0 - min_wake0) * ramp_pos

        wake_frac_lo = min_wake0
        wake_frac_hi = safe_min(ramp_wake_frac, max_wake0)

        r_wake = wake_frac0 - safe_if_else(
            cum_mf0 < mf_thres0 * mf_choke0,
            wake_frac_lo,
            wake_frac_hi,
        )

        k1 = (1 - wake_frac0 - depth_ratio0) / (1 - wake_frac0)
        k2 = 1 + np.tan(alpha0) ** 2
        dht = (1 / k2) * k1**2 * v0**2 / 2

        r_dht = dht_mix0 - dht

        return r_wake, r_dht


class RecirculationCoppage(LossModel):
    """
    Coppage et al. (1956) - Study of Supersonic Radial Compressors for
    Refrigeration and Pressurization Systems, WADC TR 55-257, Eq. (2.36).

            dh_rc = 0.02 sqrt(tan(alpha_2)) D_f U_2^2

    tan(alpha_2): no loss for a radial discharge, infinite loss at zero
    through-flow (Coppage, Section 2.4.5); tentative expression, "can only be
    regarded as hypotheses" (Coppage, Section 2.4).

    D_f: diffusion factor of the blade loading loss, Coppage Eq. (2.33), as in
    Oh et al. (1997) Table 5. Galvas (1972) - Fortran program for predicting
    off-design performance of centrifugal compressors- Eq. (B76)

    Notation as in RecirculationOh:
    - alpha_2    -> FlowAngleAbs at the impeller outlet, from the meridional
                    direction
    - k          -> RecircDiffCoeff, 0.75 in the report; separate from the
                    BlLoadingCoeff of BladeLoadingCoppage, so that the blade
                    loading loss can be calibrated without changing this one
                    0.60 when accounting for splitter blades

    Two-node equation across the impeller, e.g. `RecirculationCoppage(): (0, 1)`.
    """

    def residual(
        self,
        dht_recirc1: n1.loss.Dht_recirculation.Hint,
        alpha: n1.kin.FlowAngleAbs.Hint,
        ht0: n0.tot.Enthalpy.Hint,
        ht1: n1.tot.Enthalpy.Hint,
        rtip0: n0.geo.Rtip.Hint,
        rmid1: n1.geo.Rmid.Hint,
        u1: n1.kin.BladeSpeed.Hint,
        w1: n1.kin.W_mag.Hint,
        w_tip0: n0.kin.W_tip.Hint,
        n_bl_eff1: n1.geo.NumBladesEff.Hint,
        rc_coeff1: n1.oth.RecircDiffCoeff.Hint,
    ):
        w_ratio = w1 / w_tip0
        work = ht1 - ht0
        r_ratio = rtip0 / rmid1
        diff_fact = (
            1
            - w_ratio
            + rc_coeff1
            * (work / u1**2)
            * (1 / w_ratio)
            * (((n_bl_eff1 / np.pi) * (1 - r_ratio)) + (2 * r_ratio)) ** (-1)
        )
        dht = 0.02 * np.sqrt(np.tan(alpha)) * diff_fact * u1**2

        return dht_recirc1 - safe_mean(dht)


class RecirculationOh(LossModel):
    """
    Oh, Yoon and Chung (1997) - An Optimum Set of Loss Models for Performance
    Prediction of Centrifugal Compressors, Table 5 (set E-3) and Table 6.

    Diffusion factor of the Coppage et al. (1956) blade loading loss, with a
    hyperbolic function of the exit flow angle instead of Coppage's tan(alpha_2),
    so that the loss lies between those of Coppage and Aungier at low flow (Fig. 6):

        dh_rc = 8e-5 sinh(3.5 alpha_2^3) D_f^2 U_2^2

    Notation, Oh et al. -> adet:
    - alpha_2    -> FlowAngleAbs at the impeller outlet, from the meridional
                    direction
    - W_1t       -> W_tip, inlet relative velocity at the tip
    - W_2, U_2   -> W_mag and BladeSpeed at the impeller outlet
    - dh_Euler   -> ht1 - ht0, the Euler work (no parasitic loss in ht1)
    - D_1t / D_2 -> Rtip at the inlet / Rmid at the outlet
    - Z          -> NumBladesEff
    - k          -> RecircDiffCoeff, 0.75 in the paper; separate from the
                    BlLoadingCoeff of BladeLoadingCoppage, so that the blade
                    loading loss can be calibrated without changing this one

    Two-node equation across the impeller, e.g. `RecirculationOh(): (0, 1)`.
    """

    def residual(
        self,
        w_tip0: n0.kin.W_tip.Hint,
        w1: n1.kin.W_mag.Hint,
        u1: n1.kin.BladeSpeed.Hint,
        n_bl_eff1: n1.geo.NumBladesEff.Hint,
        ht0: n0.tot.Enthalpy.Hint,
        ht1: n1.tot.Enthalpy.Hint,
        rtip0: n0.geo.Rtip.Hint,
        rmid1: n1.geo.Rmid.Hint,
        alpha1: n1.kin.FlowAngleAbs.Hint,
        rc_coeff1: n1.oth.RecircDiffCoeff.Hint,
        dht_recirc1: n1.loss.Dht_recirculation.Hint,
    ):
        w_ratio = w1 / w_tip0
        work = ht1 - ht0
        r_ratio = rtip0 / rmid1
        diff_fact = (
            1
            - w_ratio
            + rc_coeff1
            * (work / u1**2)
            * (1 / w_ratio)
            * (((n_bl_eff1 / np.pi) * (1 - r_ratio)) + (2 * r_ratio)) ** (-1)
        )
        dht = 8e-5 * np.sinh(3.5 * alpha1**3) * (diff_fact * u1) ** 2

        return dht_recirc1 - safe_mean(dht)


class LeakageAungier(LossModel):
    def residual(
        self,
        rtip0: n0.geo.Rtip.Hint,
        rr1: n1.geo.RDistr.Hint,
        hgt0: n0.geo.Height.Hint,
        hgt1: n1.geo.Height.Hint,
        vt1: n1.kin.V_tan.Hint,
        vt0: n0.kin.V_tan.Hint,
        n_bl1: n1.geo.NumBlades.Hint,
        n_split1: n1.geo.NumSplitters.Hint,
        mf0: n0.oth.StreamMassFlow.Hint,
        rho1: n1.stc.Density.Hint,
        u1: n1.kin.BladeSpeed.Hint,
        hyd_len1: n1.geo.HydLen.Hint,
        cl0: n0.geo.TipClearance.Hint,
        cl1: n1.geo.TipClearance.Hint,
        dht_leak1: n1.loss.Dht_leakage.Hint,
    ):
        num_blades = n_bl1 + n_split1
        clearance = (cl0 + cl1) / 2

        R_mean = (rtip0 + rr1) / 2
        H_mean = (hgt0 + hgt1) / 2
        Dp_cl = (mf0 * ((rr1 * vt1) - (rtip0 * vt0))) / (
            num_blades * R_mean * H_mean * hyd_len1
        )
        U_cl = 0.816 * (2 * Dp_cl / rho1) ** 0.5
        m_cl = rho1 * num_blades * clearance * hyd_len1 * U_cl
        return dht_leak1 - m_cl * U_cl * u1 / (2 * mf0)


class LeakageLostWork(LossModel):
    def residual(
        self,
        work_loss_coeff: n1.oth.WorkLossCoeff.Hint,
        ht0: n0.tot.Enthalpy.Hint,
        ht1: n1.tot.Enthalpy.Hint,
        cl0: n0.geo.TipClearance.Hint,
        cl1: n1.geo.TipClearance.Hint,
        hgt1: n1.geo.Height.Hint,
        dht_lost1: n1.loss.Dht_lost.Hint,
    ):
        work = ht1 - ht0
        clearance = (cl0 + cl1) / 2

        dht_leakage_lost = work_loss_coeff * work * clearance / hgt1

        return dht_lost1 - dht_leakage_lost


class AungierChoking(LossModel):
    def residual(
        self,
        area_chk: n0.geo.ChokeArea.Hint,
        area_thr: n0.geo.ThroatArea.Hint,
        Dht_chk: n1.loss.Dht_choking.Hint,
        w0: n0.kin.W_mag.Hint,
    ):
        X = 11 - 10 * area_thr / area_chk
        loss = safe_max(
            0.0,
            0.5 * (0.05 * X + X**7),
        )

        return Dht_chk - w0**2 * loss
