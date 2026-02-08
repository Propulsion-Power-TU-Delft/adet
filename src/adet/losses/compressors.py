import numpy as np

from adet.equations.utils import safe_abs
from adet.losses.base_loss import LossModel
from adet.equations.base_equation import DeviationModel, EquationBase


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
# ideal gas expressions with gamma_pv
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
        geo_hh0,
        geo_rr0,
        geo_rr_tip0,
        geo_rr_midspan1,
        # Kinematics
        kin_U1,
        kin_W1,
        kin_W_tip0,
        kin_Wm0,
        kin_Wt0,
        kin_omega0,
        geo_num_blades_eff1,
        # Coefficients
        oth_bl_loadingCoeff1,  # 0.75
        oth_delta_hmass_loading1,
    ):
        num_span = max(geo_rr0.shape)
        if num_span == 1:
            W_loc_tip = kin_W_tip0
            r_loc_tip = geo_rr_tip0
        else:
            W_loc_tip = (kin_Wm0**2 + (kin_Wt0 + kin_omega0 * geo_hh0 / 2) ** 2) ** 0.5
            r_loc_tip = geo_rr0 + geo_hh0 / 2

        W_ratio = kin_W1 / W_loc_tip

        work = tot_hmass1 - tot_hmass0

        r0_by_r1 = r_loc_tip / geo_rr_midspan1

        diffusion_coeff = (
            1
            - W_ratio
            + oth_bl_loadingCoeff1
            * (work / kin_U1**2)
            * W_ratio
            / (((geo_num_blades_eff1 / np.pi) * (1 - r0_by_r1)) + (2 * r0_by_r1))
        )

        return oth_delta_hmass_loading1 - 0.05 * (diffusion_coeff * kin_U1) ** 2


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
        geo_rr_midspan1,
        geo_num_blades_eff1,
        geo_tip_clearance0,
        # Thermo
        stc_rhomass0,
        stc_rhomass1,
        # Enthalpy prod
        oth_delta_hmass_clearance1,
    ):
        K = safe_abs(
            (geo_rr_tip0**2 - geo_rr_hub0**2)
            / ((geo_rr_midspan1 - geo_rr_tip0) * (1 + stc_rhomass1 / stc_rhomass0))
        )
        return oth_delta_hmass_clearance1 - (
            0.6
            * geo_tip_clearance0
            / geo_height1
            * kin_Vt1
            * (
                safe_abs(
                    (4 * np.pi / (geo_height1 * geo_num_blades_eff1))
                    * K
                    * kin_Vt1
                    * kin_Vm0
                )
            )
            ** 0.5
        )


class SkinFrictionJansen(LossModel):
    def residual(
        self,
        oth_cum_massflow0,
        # Geometry
        geo_rr_tip0,
        geo_rr_hub0,
        geo_rr0,
        geo_rr_midspan1,
        geo_height1,
        geo_chord_ax1,
        geo_num_blades_eff1,
        geo_metal_angle0,
        geo_metal_angle1,
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
        L_hyd = (
            np.pi
            / 8
            * (
                2 * geo_rr_midspan1
                - geo_rr_tip0
                - geo_rr_hub0
                - geo_height1
                + 2 * geo_chord_ax1
            )
            * (
                2
                / (
                    (np.cos(geo_metal_angle0[0]) + np.cos(geo_metal_angle0[-1])) / 2
                    + np.cos(geo_metal_angle1)
                )
            )
        )

        D_hyd = (
            np.pi
            * ((2 * geo_rr_tip0) ** 2 - (2 * geo_rr_hub0) ** 2)
            / (
                (4 * np.pi * geo_rr0)
                + geo_num_blades_eff1 * 2 * (geo_rr_tip0 - geo_rr_hub0)
            )
        )

        w_mean = (kin_V0 + kin_V1 + kin_W_tip0 + 2 * kin_W_hub0 + 3 * kin_W1) / 8

        Re = oth_cum_massflow0 / ((stc_viscosity0 + stc_viscosity1) / 2 * D_hyd)
        Re_e = (Re - 2000) * oth_abs_roughness1 / D_hyd

        r1 = 4 * oth_Cf_smooth1 - (
            (1 / np.log10((2.51 / (Re * np.sqrt(4 * oth_Cf_smooth1))) ** -2)) ** 2
        )
        r2 = (
            4 * oth_Cf_rough1
            - (1 / np.log10((oth_abs_roughness1 / (3.71 * D_hyd)) ** -2)) ** 2
        )

        Cf = oth_Cf_smooth1 + (oth_Cf_rough1 - oth_Cf_smooth1) * (1 - (60 / Re_e))
        r3 = oth_delta_hmass_skin1 - 2 * safe_abs(Cf) * (L_hyd / D_hyd) * (w_mean**2)

        return r1, r2, r3


class LossAdder(EquationBase):
    def residual(
        self,
        tot_hmass0,
        oth_tot_hmass_is0,
        oth_delta_hmass_skin0,
        oth_delta_hmass_loading,
        oth_delta_hmass_clearance0,
    ):
        return tot_hmass0 - (
            oth_tot_hmass_is0
            + oth_delta_hmass_skin0
            + oth_delta_hmass_loading
            + oth_delta_hmass_clearance0
        )
