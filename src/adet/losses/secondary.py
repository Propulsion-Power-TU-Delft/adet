import CoolProp as cp
import numpy as np

from adet.equations.base_equation import EquationConfig
from adet.equations.utils import safe_if_else
from adet.losses.base_loss import LossModel
from adet.variables import NodeVariables, ThermoVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)

thrm = ThermoVariables()


class SecondaryBSM(LossModel):
    """
    Benner, M. W., Sjolander, S. A., and Moustapha, S. H., 2006, “An Empirical
    Prediction Method for Secondary Losses in Turbines—Part II: A New Second-
    ary Loss Correlation,” ASME J. Turbomach., 128(2), pp. 281–291.
    """

    config = EquationConfig(
        manual_units=('J / kg / K',),
        input_pair=cp.HmassP_INPUTS,
        out_properties=(thrm.Entropy,),
    )

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

        loss_coeffY = safe_if_else(hgt_by_ch < 2.0, Y_min2, Y_gtr2)
        # Bound the loss coefficient to be >= 0
        loss_coeffY = safe_if_else(loss_coeffY > 0.0, loss_coeffY, 0.0)

        rlt_p_out = (p_rlt0 + loss_coeffY * p1) / (loss_coeffY + 1)

        return ds_sec1 - (self.eos(h_rlt0, rlt_p_out) - s0)
