from adet.losses.base_loss import LossModel
import numpy as np
import casadi as cs
import CoolProp as cp

# - * - * - * - * FROM TURBOSIM - * - * - * - *
# NOTE:
# - computed with flow properties at midspan and spread along the blade span
# - it includes endwall boundary layer losses'
# CR = NP.cos(flow_angle_in) / NP.cos(
#     flow_angle_out
# )  # Convergence ratio (?): flow acceleration
# H_c = H_Cax * NP.cos(blade_stagger)
#
# if H_c < 2.0:
#     Y = (0.038 + 0.41 * NP.tanh(1.20 * self.delta_star_H)) / (
#         NP.sqrt(NP.cos(blade_stagger))
#         * CR
#         * (H_c * NP.cos(flow_angle_out) / NP.cos(blade_stagger)) ** 0.55
#     )
# else:
#     Y = (0.052 + 0.56 * NP.tanh(1.20 * self.delta_star_H)) / (
#         NP.sqrt(NP.cos(blade_stagger))
#         * CR
#         * H_c
#         * (NP.cos(flow_angle_out) / NP.cos(blade_stagger)) ** 0.55
#     )
#
# Pt_out = (Pt_in + Y * P_out) / (Y + 1)
#
# self.flow.fluid.EoS.update(fld.CoolProp.HmassP_INPUTS, ht_out, Pt_out)
# self.ds_secondary[flag_row, :] = self.flow.fluid.EoS.smass() - s_in
# - * - * - * - * - * - * - * - * - * - * - * - *


class SecondaryBSM(LossModel):
    """
    Benner, M. W., Sjolander, S. A., and Moustapha, S. H., 2006, “An Empirical
    Prediction Method for Secondary Losses in Turbines—Part II: A New Second-
    ary Loss Correlation,” ASME J. Turbomach., 128(2), pp. 281–291.
    """

    manual_units = ('J / kg / K',)
    input_pair = cp.HmassP_INPUTS
    output_quantities = ('smass',)

    def residual(
        self,
        geo_height1,
        geo_chord1,
        kin_beta0,
        kin_beta1,
        geo_stagger1,
        oth_disp_thick_ew1,
        # Thermo
        rlt_p0,
        stc_p1,
        stc_smass0,
        rlt_hmass1,
        oth_delta_smass_secondary1,
    ):
        hgt_by_ch = geo_height1 / geo_chord1
        cos_ratio = np.cos(kin_beta0) / np.cos(kin_beta1)
        disp_by_H = oth_disp_thick_ew1 / geo_height1

        Y_min2 = (0.038 + 0.41 * np.tanh(1.2 * disp_by_H)) / (
            np.cos(geo_stagger1) ** 0.5
            * cos_ratio
            * (hgt_by_ch * np.cos(kin_beta1) / np.cos(geo_stagger1)) ** 0.55
        )

        Y_gtr2 = (0.052 + 0.56 * np.tanh(1.2 * disp_by_H)) / (
            np.cos(geo_stagger1) ** 0.5
            * cos_ratio
            * hgt_by_ch
            * (np.cos(kin_beta1) / np.cos(geo_stagger1)) ** 0.55
        )

        loss_coeffY = cs.if_else(hgt_by_ch < 2.0, Y_min2, Y_gtr2)

        rlt_p_out = (rlt_p0 + loss_coeffY * stc_p1) / (loss_coeffY + 1)

        return oth_delta_smass_secondary1 - (
            self.eos(rlt_hmass1, rlt_p_out) - stc_smass0
        )
