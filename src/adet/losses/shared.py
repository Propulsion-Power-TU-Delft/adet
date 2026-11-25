import CoolProp as cp
import numpy as np

from adet.equations.base_equation import (
    EquationConfig,
)
from adet.equations.utils import (
    safe_if_else,
)
from adet.losses.base_loss import LossModel
from adet.variables import NodeVariables, ThermoVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)
thrm = ThermoVariables()


class AbruptExpansion(LossModel):
    """
    Expansion loss due to abrupt enlargement
    Benedict et al.
    Aungier page 81 of 'Turbine Aerodynamics'
    """

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
