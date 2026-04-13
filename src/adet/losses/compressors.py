import numpy as np
import CoolProp as cp

from adet.equations.base_equation import DeviationModel, EquationBase
from adet.equations.utils import (
    safe_abs,
    safe_if_else,
    safe_max,
    safe_mean,
    safe_min,
    safe_sign,
)
from adet.losses.base_loss import LossModel


class WorkCoefficientEstimate(EquationBase):
    def residual(
        self,
        tot_p0,
        tot_p1,
        kin_machU1,
        oth_workCoeff1,
        # oth_isEfficiency1,
        oth_gamma_pv1,
    ):
        lhs = (tot_p1 / tot_p0) ** ((oth_gamma_pv1 - 1) / oth_gamma_pv1)
        rhs = 1 + (oth_gamma_pv1 - 1) * oth_workCoeff1 * kin_machU1**2
        return lhs - rhs


# TODO: This can be generalized instead of using the
# ideal gas expressions with gamma(pv)
class CaseyRushInletFunc(EquationBase):
    def residual(
        self,
        kin_relmach_tip0,
        kin_beta_tip0,
        stc_cpmass1,
        stc_cvmass1,
    ):
        gamma = stc_cpmass1 / stc_cvmass1
        # Equation 18 in Casey-Rush
        first_term = 3 + gamma * kin_relmach_tip0**2 + 2 * kin_relmach_tip0
        second_term = 3 + gamma * kin_relmach_tip0**2 - 2 * kin_relmach_tip0
        rhs = (first_term**0.5 - second_term**0.5) / (2 * kin_relmach_tip0)

        return np.cos(kin_beta_tip0) - rhs


class CompressorShapeFactor(EquationBase):
    # shape factor  k = 1 - (Rh1/Rs1)^2
    def residual(
        self,
        geo_rr_hub0,
        geo_rr_tip0,
        geo_shapeKCoeff0,
    ):
        return geo_shapeKCoeff0 - (1 - (geo_rr_hub0 / geo_rr_tip0) ** 2)


class BackstromSlip(DeviationModel):
    def residual(
        self,
        oth_slip_factor1,
        oth_slip_factCoeff1,
        #
        geo_rr0,
        geo_rr1,
        geo_num_blades1,
        geo_metal_angle1,
        geo_eff_solidity1,
        geo_num_splitters1,
        #
        kin_U1,
        kin_Wm1,
        kin_Wt1,
    ):
        # Use actual number of blades at the outlet
        tot_blades = geo_num_blades1 + geo_num_splitters1

        radius_ratio = geo_rr0 / geo_rr1
        r1 = geo_eff_solidity1 - (
            (1 - radius_ratio) * tot_blades / (2 * np.pi * np.cos(geo_metal_angle1))
        )

        r2 = oth_slip_factor1 - (
            1
            - 1
            / (
                1
                + oth_slip_factCoeff1
                * geo_eff_solidity1
                * np.cos(geo_metal_angle1) ** 0.5
            )
        )

        slip_velocity = kin_U1 * (1 - oth_slip_factor1)  # Sign depends on U

        Wt1_noslip = kin_Wm1 * np.tan(geo_metal_angle1)

        r3 = kin_Wt1 - (Wt1_noslip - slip_velocity)

        return r1, r2, r3


class BladeLoadingCoppage(LossModel):
    def residual(
        self,
        # Enthalpies for work
        tot_hmass0,
        tot_hmass1,
        # Geometry
        geo_rr_tip0,
        geo_rr_midspan1,
        # Kinematics
        kin_U1,
        kin_W1,
        kin_W_tip0,
        geo_num_blades_eff1,
        # Coefficients
        oth_bl_loadingCoeff1,  # 0.75
        oth_delta_hmass_loading1,
    ):

        W_ratio = kin_W1 / kin_W_tip0
        work = tot_hmass1 - tot_hmass0
        r0_by_r1 = geo_rr_tip0 / geo_rr_midspan1
        diff_fact = (
            1
            - W_ratio
            + oth_bl_loadingCoeff1
            * (work / kin_U1**2)
            * W_ratio
            / (geo_num_blades_eff1 / np.pi * (1 - r0_by_r1) + 2 * r0_by_r1)
        )

        return oth_delta_hmass_loading1 - 0.05 * (diff_fact * kin_U1) ** 2


class ClearanceJansen(LossModel):
    def residual(
        self,
        # Kine
        kin_Vm0,
        kin_Vt1,
        # Geo
        geo_height1,
        geo_rr_hub0,
        geo_rr_tip0,
        geo_rr1,
        geo_num_blades_eff1,
        geo_tip_clearance0,
        geo_tip_clearance1,
        # Thermo
        stc_rhomass0,
        stc_rhomass1,
        # Enthalpy prod
        oth_delta_hmass_clearance1,
    ):
        clearance = (geo_tip_clearance0 + geo_tip_clearance1) / 2
        K = safe_abs(
            (geo_rr_tip0**2 - geo_rr_hub0**2)
            / ((geo_rr1 - geo_rr_tip0) * (1 + stc_rhomass1 / stc_rhomass0))
        )
        abs_Vt1 = safe_abs(kin_Vt1)

        return oth_delta_hmass_clearance1 - (
            0.6
            * clearance
            / geo_height1
            * abs_Vt1
            * (
                (4 * np.pi / (geo_height1 * geo_num_blades_eff1))
                * K
                * abs_Vt1
                * kin_Vm0
            )
            ** 0.5
        )


class ClearanceBrasz(LossModel):
    def residual(
        self,
        geo_rr1,
        geo_rr_hub0,
        geo_rr_tip0,
        geo_height1,
        geo_tip_clearance0,
        geo_tip_clearance1,
        geo_num_blades_eff1,
        kin_Vt1,
        kin_Vm0,
        stc_rhomass0,
        stc_rhomass1,
        oth_delta_hmass_clearance1,
    ):
        clearance = (geo_tip_clearance0 + geo_tip_clearance1) / 2

        K = (geo_rr_tip0**2 - geo_rr_hub0**2) / (
            (geo_rr1 - geo_rr_tip0) * (1 + stc_rhomass1 / stc_rhomass0)
        )
        abs_Vt1 = safe_abs(kin_Vt1)

        return oth_delta_hmass_clearance1 - (
            0.6
            * clearance
            * abs_Vt1
            / (geo_height1 + clearance / 2)
            * np.sqrt(
                4
                * np.pi
                * abs_Vt1
                * kin_Vm0
                * K
                / ((geo_height1 + clearance / 2) * geo_num_blades_eff1)
            )
        )


class HydraulicQuantities(EquationBase):
    def residual(
        self,
        geo_rr1,
        geo_rr_tip0,
        geo_rr_hub0,
        geo_height1,
        geo_chord_ax1,
        geo_metal_angle_hub0,
        geo_metal_angle_tip0,
        geo_metal_angle1,
        geo_num_blades_eff1,
        geo_hyd_diam1,
        geo_hyd_len1,
    ):
        L_hyd = (
            np.pi
            / 8
            * (
                2 * geo_rr1
                - geo_rr_tip0
                - geo_rr_hub0
                - geo_height1
                + 2 * geo_chord_ax1
            )
            * (
                4
                / (
                    (np.cos(geo_metal_angle_hub0) + np.cos(geo_metal_angle_tip0))
                    + 2 * np.cos(geo_metal_angle1)
                )
            )
        )

        # D_hyd = (
        #     np.pi
        #     * ((2 * geo_rr_tip0) ** 2 - (2 * geo_rr_hub0) ** 2)
        #     / (
        #         (4 * np.pi * geo_rr0)
        #         + geo_num_blades_eff1 * 2 * (geo_rr_tip0 - geo_rr_hub0)
        #     )
        # )

        metal_cosine_sum0 = np.cos(geo_metal_angle_tip0) + np.cos(geo_metal_angle_hub0)
        D_hyd = (
            2
            * geo_rr1
            * (
                (
                    np.cos(geo_metal_angle1)
                    / (
                        geo_num_blades_eff1 / np.pi
                        + 2 * geo_rr1 * np.cos(geo_metal_angle1) / geo_height1
                    )
                )
                + (
                    0.5
                    * (geo_rr_tip0 / geo_rr1 + geo_rr_hub0 / geo_rr1)
                    * (metal_cosine_sum0 / 2)
                )
                / (
                    geo_num_blades_eff1 / np.pi
                    + (
                        (2 * (geo_rr_tip0 + geo_rr_hub0))
                        / (2 * (geo_rr_tip0 - geo_rr_hub0))
                    )
                    * (metal_cosine_sum0 / 2)
                )
            )
        )

        r1 = geo_hyd_len1 - L_hyd
        r2 = geo_hyd_diam1 - D_hyd

        return r1, r2


class SkinFrictionJansen(LossModel):
    def residual(
        self,
        oth_cum_massflow0,
        geo_hyd_len1,
        geo_hyd_diam1,
        # Kinematics
        kin_V0,
        kin_V1,
        kin_W1,
        kin_W_tip0,
        kin_W_hub0,
        # Thermo
        stc_viscosity0,
        stc_viscosity1,
        # Roughness params
        oth_abs_roughness1,
        oth_Cf_smooth1,
        oth_Cf_rough1,
        oth_delta_hmass_skin1,
    ):
        w_mean = (kin_V0[-1] + kin_V1 + kin_W_tip0 + 2 * kin_W_hub0 + 3 * kin_W1) / 8
        mu_mean = (stc_viscosity0 + stc_viscosity1) / 2

        Re = oth_cum_massflow0 / (mu_mean * geo_hyd_diam1)
        Re_e = (Re - 2000) * oth_abs_roughness1 / geo_hyd_diam1

        r1 = 4 * oth_Cf_smooth1 - (
            (1 / np.log10((2.51 / (Re * np.sqrt(4 * oth_Cf_smooth1))) ** -2)) ** 2
        )
        r2 = (
            4 * oth_Cf_rough1
            - (1 / np.log10((oth_abs_roughness1 / (3.71 * geo_hyd_diam1)) ** -2)) ** 2
        )

        Cf = oth_Cf_smooth1 + (oth_Cf_rough1 - oth_Cf_smooth1) * (1 - (60 / Re_e))
        r3 = oth_delta_hmass_skin1 - 2 * safe_abs(Cf) * (
            geo_hyd_len1 / geo_hyd_diam1
        ) * (w_mean**2)

        return r1, r2, r3

        # TODO: More readable formulation (singular like this)
        # # 1. Cf_smooth equation
        # r1 = 1 / sqrt_smooth - 2 * np.log10(2.51 / (Re * sqrt_smooth))
        # # 2. Cf_rough equation
        # r2 = 1 / sqrt_rough - 2 * np.log10(oth_abs_roughness1 / (3.71 * D_hyd))


class IncidenceVDB(LossModel):
    """
    Van den Braembussche incidence model
    """

    def residual(
        self,
        kin_beta_opt0,
        kin_beta0,
        geo_metal_angle0,
        kin_relmach0,
        kin_W_tip0,
        oth_delta_hmass_incidence1,
    ):
        incidence = kin_beta0 - geo_metal_angle0
        incidence_opt = kin_beta_opt0 - geo_metal_angle0

        incidence *= safe_sign(geo_metal_angle0)
        incidence_opt *= safe_sign(geo_metal_angle0)

        C_te = safe_if_else(incidence > incidence_opt, 2.5, 2.0)

        delta_i = 2.5 + 0.15 * (12.5 - 0.1 * kin_beta0) * (kin_relmach0 - 1.2) ** 2 / 2

        incidence_diff = incidence - incidence_opt
        dht = (
            0.833 * (incidence_diff / (C_te * delta_i)) ** 2
            + 0.1667 * incidence_diff / (C_te * delta_i)
        ) * kin_W_tip0**2

        return oth_delta_hmass_incidence1 - dht


class IncidenceGalvas(LossModel):
    def residual(
        self,
        kin_W0,
        kin_beta0,
        kin_beta_opt0,
        oth_incCoeff0,
        oth_delta_hmass_incidence1,
    ):

        lost_angle = safe_abs(kin_beta_opt0 - kin_beta0)
        lost_Wt = kin_W0 * np.sin(lost_angle)

        return oth_delta_hmass_incidence1 - oth_incCoeff0 * lost_Wt**2 / 2


class MixingJohnstonDean(LossModel):
    def residual(
        self,
        kin_V0,
        kin_alpha0,
        oth_minWake_frac0,
        oth_maxWake_frac0,
        oth_cum_massflow0,
        oth_massflow_choke0,
        oth_wake_frac0,
        oth_delta_hmass_mixing0,
    ):
        #        ^ eps
        #        |        <--hi------>
        #        |          /````````` eps_max
        #        | <--lo-> /
        #        | _______/ eps_min
        #        |
        #        |-------------> m / m_choke
        #                 |  |
        #         MF_THRES   1

        MF_THRES = 0.75  # Ramp up wake frac from MF_THRES * m_choke
        B = 1  # hypothesis! No sudden area change after impeller

        slope_eps_mf = (oth_maxWake_frac0 - oth_minWake_frac0) / (
            (1 - MF_THRES) * oth_massflow_choke0
        )
        offset_eps_mf = oth_maxWake_frac0 - slope_eps_mf * oth_massflow_choke0
        linear_wake_frac = offset_eps_mf + slope_eps_mf * oth_cum_massflow0

        wake_frac_lo = oth_minWake_frac0
        wake_frac_hi = safe_min(linear_wake_frac, oth_maxWake_frac0)

        r_wake = oth_wake_frac0 - safe_if_else(
            oth_cum_massflow0 < MF_THRES * oth_massflow_choke0,
            wake_frac_lo,
            wake_frac_hi,
        )
        k1 = (1 - oth_wake_frac0 - B) / (1 - oth_wake_frac0)  # pyright: ignore
        k2 = 1 + np.tan(kin_alpha0) ** 2
        dht = (1 / k2) * k1**2 * kin_V0**2 / 2

        r_dht = oth_delta_hmass_mixing0 - dht

        return r_wake, r_dht


class DiskFricDailyNece(LossModel):
    def residual(
        self,
        stc_rhomass0,
        stc_rhomass1,
        kin_U1,
        geo_rr1,
        geo_height1,
        stc_viscosity1,
        oth_cum_massflow0,
        oth_delta_hmass_disk1,
        geo_back_clearance1,
    ):
        rho_mean = (stc_rhomass0 + stc_rhomass1) / 2
        Re1 = (kin_U1 * geo_rr1 * stc_rhomass1) / stc_viscosity1

        cl_ratio = geo_back_clearance1 / geo_height1
        f_df_lo = 3.700 * cl_ratio**0.1 / (Re1**0.5)
        f_df_hi = 0.102 * cl_ratio**0.1 / (Re1**0.2)

        f_df = safe_if_else(Re1 < 3e5, f_df_lo, f_df_hi)

        return (
            oth_delta_hmass_disk1
            - 0.25 * (f_df * rho_mean * geo_rr1**2 * kin_U1**3) / oth_cum_massflow0
        )


class RecirculationOh(LossModel):
    def residual(
        self,
        kin_W_tip0,
        kin_W1,
        kin_U1,
        geo_num_blades_eff1,
        tot_hmass0,
        tot_hmass1,
        geo_rr_tip0,
        geo_rr_midspan1,
        kin_alpha1,
        oth_bl_loadingCoeff1,
        oth_delta_hmass_recirc1,
    ):
        W_ratio = kin_W1 / kin_W_tip0
        work = tot_hmass1 - tot_hmass0
        r0_by_r1 = geo_rr_tip0 / geo_rr_midspan1
        diff_fact = (
            1
            - W_ratio
            + oth_bl_loadingCoeff1
            * (work / kin_U1**2)
            * W_ratio
            / (((geo_num_blades_eff1 / np.pi) * (1 - r0_by_r1)) + (2 * r0_by_r1))
        )
        dht = 8e-5 * np.sinh(3.5 * kin_alpha1**3) * (diff_fact * kin_U1) ** 2

        return oth_delta_hmass_recirc1 - safe_mean(dht)

        # return (
        #     oth_delta_hmass_recirc1
        #     - 0.02 * np.tan(kin_alpha1) * (diff_fact * kin_U1) ** 2
        # )


class LeakageAungier(LossModel):
    def residual(
        self,
        geo_rr_tip0,
        geo_rr1,
        geo_height0,
        geo_height1,
        kin_Vt1,
        kin_Vt0,
        geo_num_blades1,
        geo_num_splitters1,
        oth_massflow0,
        stc_rhomass1,
        kin_U1,
        geo_hyd_len1,
        geo_tip_clearance0,
        geo_tip_clearance1,
        oth_delta_hmass_leakage1,
    ):
        num_blades = geo_num_blades1 + geo_num_splitters1
        clearance = (geo_tip_clearance0 + geo_tip_clearance1) / 2

        R_mean = (geo_rr_tip0 + geo_rr1) / 2
        H_mean = (geo_height0 + geo_height1) / 2
        Dp_cl = (oth_massflow0 * ((geo_rr1 * kin_Vt1) - (geo_rr_tip0 * kin_Vt0))) / (
            num_blades * R_mean * H_mean * geo_hyd_len1
        )
        U_cl = 0.816 * (2 * Dp_cl / stc_rhomass1) ** 0.5
        m_cl = stc_rhomass1 * num_blades * clearance * geo_hyd_len1 * U_cl
        return oth_delta_hmass_leakage1 - m_cl * U_cl * kin_U1 / (2 * oth_massflow0)


class LeakageLostWork(LossModel):
    def residual(
        self,
        oth_worklossCoeff1,
        tot_hmass0,
        tot_hmass1,
        geo_tip_clearance0,
        geo_tip_clearance1,
        geo_height1,
        oth_delta_hmass_leakage1,
    ):

        work = tot_hmass1 - tot_hmass0
        clearance = (geo_tip_clearance0 + geo_tip_clearance1) / 2

        # LEAKAGE WORK LOSS!
        dht_leakage_lost = oth_worklossCoeff1 * work * clearance / geo_height1

        return oth_delta_hmass_leakage1 - dht_leakage_lost


class AmiranteDiffuserMomentum(EquationBase):
    input_pair = cp.PSmass_INPUTS
    manual_units = ('m^2 / s', 'K')
    output_quantities = ('hmass',)

    def residual(
        self,
        kin_alpha1,
        geo_height1,
        geo_rr1,
        geo_rr0,
        kin_V0,
        kin_V1,
        kin_Vt0,
        stc_p0,
        stc_p1,
        stc_rhomass1,
        stc_cpmass1,
        stc_cvmass1,
        stc_T0,
        stc_T1,
        kin_Vt1,
        stc_viscosity1,
        oth_wake_frac0,
    ):
        # constants TODO: unhardcode
        FRIC_CONST = 0.01
        ETA_POLY = 0.93

        delta_rad = safe_max(0.001 * geo_rr0, geo_rr1 - geo_rr0)
        x_log = delta_rad / np.cos(kin_alpha1)
        Re = (stc_rhomass1 * kin_V1 * x_log) / stc_viscosity1
        Cf = FRIC_CONST * (1.8e5 / Re) ** 0.2
        # Dissipation work
        Wf = (Cf * (kin_V1 * geo_rr1) ** 2 * delta_rad) / (
            geo_height1 * geo_rr0 * geo_rr1 * np.cos(kin_alpha1)
        )

        r1 = geo_rr0 * kin_Vt0 - geo_rr1 * kin_Vt1 * (
            1 + Wf / (oth_wake_frac0 * kin_V1 * kin_V0)
        )

        p_ratio = stc_p1 / stc_p0
        gamma = stc_cpmass1 / stc_cvmass1

        exponent = (gamma - 1) / (ETA_POLY * gamma)

        r2 = stc_T1 - stc_T0 * (p_ratio) ** exponent

        return r1, r2
