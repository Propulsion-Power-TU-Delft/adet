"""Compressor loss models shared across centrifugal, axial and mixed rotors."""

import CoolProp as cp
import numpy as np

from adet.equations.base_equation import (
    EquationConfig,
)
from adet.equations.utils import (
    safe_abs,
    safe_if_else,
)
from adet.losses.base_loss import LossModel
from adet.variables import NodeVariables, ThermoVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)
thrm = ThermoVariables()


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


class AungierSimpleMixLoss(LossModel):
    config = EquationConfig(
        input_pair=cp.HmassP_INPUTS,
        out_properties=(thrm.Entropy,),
    )

    def residual(
        self,
        stc_rhomass0,
        kin_W0,
        geo_pitch0,
        geo_metal_angle0,
        geo_bld_thick0,
        rlt_p0,
        rlt_hmass0,
        oth_disp_thick0,
        stc_smass0,
        oth_delta_smass_mixing0,
    ):
        opening = geo_pitch0 * np.cos(geo_metal_angle0)
        delta_pt = (
            0.5
            * stc_rhomass0
            * kin_W0**2
            * (opening / (opening - geo_bld_thick0 - oth_disp_thick0) - 1) ** 2
        )

        smass1 = self.eos(rlt_hmass0, rlt_p0 - delta_pt)

        return oth_delta_smass_mixing0 - (stc_smass0 - smass1)
