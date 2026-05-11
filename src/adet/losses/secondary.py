from adet.equations.variables import NodeVariables
from adet.losses.base_loss import LossModel
import numpy as np
import casadi as cs
import CoolProp as cp

n0 = NodeVariables(0)
n1 = NodeVariables(1)

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
        h1: n1.geo.Height.Hint,
        chord1: n1.geo.Chord.Hint,
        beta0: n0.kin.FlowAngleRel.Hint,
        beta1: n1.kin.FlowAngleRel.Hint,
        stag1: n1.geo.Stagger.Hint,
        disp_ew1: n1.oth.DispThickEW.Hint,
        p_rlt0: n0.rlt.Pressure.Hint,
        p1: n1.stc.Pressure.Hint,
        s0: n0.stc.Entropy.Hint,
        h_rlt0: n0.rlt.Enthalpy.Hint,
        ds_sec1: n1.loss.Ds_secondary.Hint,
    ):
        hgt_by_ch = h1 / chord1
        cos_ratio = np.cos(beta0) / np.cos(beta1)
        disp_by_H = disp_ew1 / h1

        Y_min2 = (0.038 + 0.41 * np.tanh(1.2 * disp_by_H)) / (
            np.cos(stag1) ** 0.5
            * cos_ratio
            * (hgt_by_ch * np.cos(beta1) / np.cos(stag1)) ** 0.55
        )

        Y_gtr2 = (0.052 + 0.56 * np.tanh(1.2 * disp_by_H)) / (
            np.cos(stag1) ** 0.5
            * cos_ratio
            * hgt_by_ch
            * (np.cos(beta1) / np.cos(stag1)) ** 0.55
        )

        loss_coeffY = cs.if_else(hgt_by_ch < 2.0, Y_min2, Y_gtr2)
        # Bound the loss coefficient to be >= 0
        loss_coeffY = cs.if_else(loss_coeffY > 0.0, loss_coeffY, 0.0)

        rlt_p_out = (p_rlt0 + loss_coeffY * p1) / (loss_coeffY + 1)

        return ds_sec1 - (self.eos(h_rlt0, rlt_p_out) - s0)
