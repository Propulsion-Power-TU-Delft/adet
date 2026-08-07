import CoolProp as cp
import numpy as np

from adet.equations.base_equation import DeviationModel, EquationBase, EquationConfig
from adet.equations.utils import (
    safe_abs,
    safe_if_else,
    safe_max,
    safe_mean,
    safe_min,
    safe_sign,
)
from adet.losses.base_loss import LossModel
from adet.variables import NodeVariables, ThermoVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)
thrm = ThermoVariables()


# TODO: This can be generalized instead of using the
# ideal gas expressions with gamma(pv)
class CaseyRushInletFunc(EquationBase):
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


class BackstromSlip(DeviationModel):
    def residual(
        self,
        slip1: n1.oth.SlipFactor.Hint,
        slip_coeff1: n1.oth.SlipFactCoeff.Hint,
        rr0: n0.geo.RDistr.Hint,
        rr1: n1.geo.RDistr.Hint,
        n_bl1: n1.geo.NumBlades.Hint,
        metal_ang1: n1.geo.MetalAngle.Hint,
        eff_solid1: n1.geo.EffSolidity.Hint,
        n_split1: n1.geo.NumSplitters.Hint,
        u1: n1.kin.BladeSpeed.Hint,
        wm1: n1.kin.W_mer.Hint,
        wt1: n1.kin.W_tan.Hint,
    ):
        tot_blades = n_bl1 + n_split1

        radius_ratio = rr0 / rr1
        r1 = eff_solid1 - (
            (1 - radius_ratio) * tot_blades / (2 * np.pi * np.cos(metal_ang1))
        )

        r2 = slip1 - (
            1 - 1 / (1 + slip_coeff1 * eff_solid1 * np.cos(metal_ang1) ** 0.5)
        )

        slip_velocity = u1 * (1 - slip1)

        wt_noslip = wm1 * np.tan(metal_ang1)

        r3 = wt1 - (wt_noslip - slip_velocity)

        return r1, r2, r3


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


class HydraulicQuantities(EquationBase):
    def residual(
        self,
        rr1: n1.geo.RDistr.Hint,
        rtip0: n0.geo.Rtip.Hint,
        rhub0: n0.geo.Rhub.Hint,
        hgt1: n1.geo.Height.Hint,
        chord_ax1: n1.geo.ChordAx.Hint,
        metal_hub0: n0.geo.MetalAngleHub.Hint,
        metal_tip0: n0.geo.MetalAngleTip.Hint,
        metal_ang1: n1.geo.MetalAngle.Hint,
        n_bl_eff1: n1.geo.NumBladesEff.Hint,
        hyd_diam1: n1.geo.HydDiam.Hint,
        hyd_len1: n1.geo.HydLen.Hint,
    ):
        hyd_L = (
            np.pi
            / 8
            * (2 * rr1 - (rtip0 - rhub0) - hgt1 + 2 * chord_ax1)
            * (4 / ((np.cos(metal_tip0) + np.cos(metal_hub0)) + 2 * np.cos(metal_ang1)))
        )

        metal_cos_sum = np.cos(metal_tip0) + np.cos(metal_hub0)
        hyd_D = (
            2
            * rr1
            * (
                (
                    np.cos(metal_ang1)
                    / (n_bl_eff1 / np.pi + 2 * rr1 * np.cos(metal_ang1) / hgt1)
                )
                + (0.5 * (rtip0 / rr1 + rhub0 / rr1) * (metal_cos_sum / 2))
                / (
                    n_bl_eff1 / np.pi
                    + ((2 * (rtip0 + rhub0)) / (2 * (rtip0 - rhub0)))
                    * (metal_cos_sum / 2)
                )
            )
        )

        r1 = hyd_len1 - hyd_L
        r2 = hyd_diam1 - hyd_D

        return r1, r2


class SkinFrictionJansen(LossModel):
    def residual(
        self,
        cum_mf0: n0.oth.TotMassFlow.Hint,
        hyd_len1: n1.geo.HydLen.Hint,
        hyd_diam1: n1.geo.HydDiam.Hint,
        v0: n0.kin.V_mag.Hint,
        v1: n1.kin.V_mag.Hint,
        w1: n1.kin.W_mag.Hint,
        w_tip0: n0.kin.W_tip.Hint,
        w_hub0: n0.kin.W_hub.Hint,
        visc0: n0.stc.Viscosity.Hint,
        visc1: n1.stc.Viscosity.Hint,
        abs_rough1: n1.geo.AbsRoughness.Hint,
        cf_smooth1: n1.oth.Cf_smooth.Hint,
        cf_rough1: n1.oth.Cf_rough.Hint,
        dht_skin1: n1.loss.Dht_skin.Hint,
    ):
        w_mean = (v0[-1] + v1 + w_tip0 + 2 * w_hub0 + 3 * w1) / 8
        mu_mean = (visc0 + visc1) / 2

        Re = cum_mf0 / (mu_mean * hyd_diam1)
        Re_e = (Re - 2000) * abs_rough1 / hyd_diam1

        r1 = 4 * cf_smooth1 - (
            (1 / np.log10((2.51 / (Re * np.sqrt(4 * cf_smooth1))) ** -2)) ** 2
        )
        r2 = (
            4 * cf_rough1 - (1 / np.log10((abs_rough1 / (3.71 * hyd_diam1)) ** -2)) ** 2
        )

        Cf = cf_smooth1 + (cf_rough1 - cf_smooth1) * (1 - (60 / Re_e))
        r3 = dht_skin1 - 2 * safe_abs(Cf) * (hyd_len1 / hyd_diam1) * (w_mean**2)

        return r1, r2, r3


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
    def residual(
        self,
        v0: n0.kin.V_mag.Hint,
        alpha0: n0.kin.FlowAngleAbs.Hint,
        cum_mf0: n0.oth.TotMassFlow.Hint,
        mf_choke0: n0.oth.ChokeMassflow.Hint,
        wake_frac0: n0.oth.WakeFrac.Hint,
        min_wake0: n0.oth.MinWakeFrac.Hint,
        max_wake0: n0.oth.MaxWakeFrac.Hint,
        dht_mix0: n0.loss.Dht_mixing.Hint,
    ):
        MF_THRES = 0.75
        B = 1

        slope_eps_mf = (max_wake0 - min_wake0) / ((1 - MF_THRES) * mf_choke0)
        offset_eps_mf = max_wake0 - slope_eps_mf * mf_choke0
        linear_wake_frac = offset_eps_mf + slope_eps_mf * cum_mf0

        wake_frac_lo = min_wake0
        wake_frac_hi = safe_min(linear_wake_frac, max_wake0)

        r_wake = wake_frac0 - safe_if_else(
            cum_mf0 < MF_THRES * mf_choke0,
            wake_frac_lo,
            wake_frac_hi,
        )
        k1 = (1 - wake_frac0 - B) / (1 - wake_frac0)
        k2 = 1 + np.tan(alpha0) ** 2
        dht = (1 / k2) * k1**2 * v0**2 / 2

        r_dht = dht_mix0 - dht

        return r_wake, r_dht


class DiskFricDailyNece(LossModel):
    def residual(
        self,
        rho0: n0.stc.Density.Hint,
        rho1: n1.stc.Density.Hint,
        u1: n1.kin.BladeSpeed.Hint,
        rr1: n1.geo.RDistr.Hint,
        hgt1: n1.geo.Height.Hint,
        visc1: n1.stc.Viscosity.Hint,
        cum_mf0: n0.oth.TotMassFlow.Hint,
        dht_disk1: n1.loss.Dht_disk.Hint,
        back_cl1: n1.geo.BackClearance.Hint,
    ):
        rho_mean = (rho0 + rho1) / 2
        Re1 = (u1 * rr1 * rho1) / visc1

        cl_ratio = back_cl1 / hgt1
        f_df_lo = 3.700 * cl_ratio**0.1 / (Re1**0.5)
        f_df_hi = 0.102 * cl_ratio**0.1 / (Re1**0.2)

        f_df = safe_if_else(Re1 <= 3e5, f_df_lo, f_df_hi)

        return dht_disk1 - 0.25 * (f_df * rho_mean * rr1**2 * u1**3) / cum_mf0


class RecirculationCoppage(LossModel):
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
        bl_coeff1: n1.oth.BlLoadingCoeff.Hint,
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
        return dht_recirc1 - 0.02 * np.tan(alpha) * (diff_fact * u1) ** 2


class RecirculationOh(LossModel):
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
        bl_coeff1: n1.oth.BlLoadingCoeff.Hint,
        dht_recirc1: n1.loss.Dht_recirculation.Hint,
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
            / (((n_bl_eff1 / np.pi) * (1 - r_ratio)) + (2 * r_ratio))
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


class AmiranteDiffuserMomentum(EquationBase):
    config = EquationConfig(
        input_pair=cp.PSmass_INPUTS,
        out_properties=(thrm.Enthalpy,),
        manual_units=('m^2 / s', 'K'),
    )

    def residual(
        self,
        alpha1: n1.kin.FlowAngleAbs.Hint,
        hgt1: n1.geo.Height.Hint,
        rr1: n1.geo.RDistr.Hint,
        rr0: n0.geo.RDistr.Hint,
        v0: n0.kin.V_mag.Hint,
        v1: n1.kin.V_mag.Hint,
        vt0: n0.kin.V_tan.Hint,
        p0: n0.stc.Pressure.Hint,
        p1: n1.stc.Pressure.Hint,
        rho1: n1.stc.Density.Hint,
        cp1: n1.stc.Cp.Hint,
        cv1: n1.stc.Cv.Hint,
        T0: n0.stc.Temperature.Hint,
        T1: n1.stc.Temperature.Hint,
        vt1: n1.kin.V_tan.Hint,
        visc1: n1.stc.Viscosity.Hint,
        wake_frac0: n0.oth.WakeFrac.Hint,
    ):
        FRIC_CONST = 0.02
        ETA_POLY = 0.85

        delta_rad = safe_max(0.001 * rr0, rr1 - rr0)
        x_log = delta_rad / np.cos(alpha1)
        Re = (rho1 * v1 * x_log) / visc1
        Cf = FRIC_CONST * (1.8e5 / Re) ** 0.2
        Wf = (Cf * (v1 * rr1) ** 2 * delta_rad) / (hgt1 * rr0 * rr1 * np.cos(alpha1))

        r1 = rr0 * vt0 - rr1 * vt1 * (1 + Wf / (wake_frac0 * v1 * v0))

        p_ratio = p1 / p0
        gamma = cp1 / cv1

        exponent = (gamma - 1) / (ETA_POLY * gamma)

        r2 = T1 - T0 * (p_ratio) ** exponent

        return r1, r2


class JansenDiffuserLoss(LossModel):
    config = EquationConfig(
        manual_units=('dimensionless', 'J/kg'),
    )

    def residual(
        self,
        rr0: n0.geo.Rmid.Hint,
        rr1: n1.geo.Rmid.Hint,
        hh0: n0.geo.Height.Hint,
        hh1: n1.geo.Height.Hint,
        rho0: n0.stc.Density.Hint,
        vt0: n0.kin.V_tan.Hint,
        v0: n0.kin.V_mag.Hint,
        alpha0: n0.kin.FlowAngleAbs.Hint,
        vt1: n1.kin.V_tan.Hint,
        cum_mf: n0.oth.TotMassFlow.Hint,
        visc0: n0.stc.Viscosity.Hint,
        visc1: n1.stc.Viscosity.Hint,
        gas_const: n0.stc.GasConstant.Hint,
        s0: n0.stc.Entropy.Hint,
        s1: n1.stc.Entropy.Hint,
    ):
        h_mean = (hh0 + hh1) / 2
        visc_mean = (visc0 + visc1) / 2
        hyd_D = 2 * h_mean

        Re = cum_mf / (visc_mean * hyd_D)
        FRIC_CONST = 0.01
        Cf = FRIC_CONST * (1.8e5 / Re) ** 0.2

        r1 = (
            vt0 / vt1
            - rr1 / rr0
            - (2 * np.pi * Cf * rho0 * vt0 * (rr1**2 - rr0 * rr1)) / cum_mf
        )

        loss = (Cf * (1 - (rr0 / rr1) ** 1.5) * v0**2) / (1.5 * hh0 * np.cos(alpha0))

        r2 = s1 - (s0 + np.exp(-loss / gas_const))

        return r1, r2
