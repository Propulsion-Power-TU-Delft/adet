from adet.equations.base_equation import EquationBase
from .base_loss import LossModel
import numpy as np
import CoolProp as cp


class EndWallVelocities(EquationBase):
    """Computation of the velocities at the endwall, also for single span cases"""

    def residual(
        self,
        kin_Wt1,
        kin_Wm1,
        geo_meridional_angle1,
        kin_omega1,
        geo_rr1,
        geo_height1,
        kin_W_hub1,
        kin_W_shroud1,
    ):
        deltaW = kin_omega1 * geo_height1 * np.cos(geo_meridional_angle1) / 2
        Wt_hub = kin_Wt1 - deltaW
        Wt_shroud = kin_Wt1 + deltaW

        r1 = kin_W_hub1 - np.sqrt(kin_Wm1**2 + Wt_hub**2)
        r2 = kin_W_shroud1 - np.sqrt(kin_Wm1**2 + Wt_shroud**2)

        return r1, r2


class TotalTotalCompressionEfficiency(EquationBase):
    input_pair = cp.PSmass_INPUTS
    output_quantities = ('hmass',)
    manual_units = ('J / kg',)

    def residual(
        self,
        tot_p1,
        stc_smass0,
        tot_hmass0,
        tot_hmass1,
        oth_eta_tt1,
    ):
        return oth_eta_tt1 * (tot_hmass1 - tot_hmass0) - (
            self.eos(tot_p1, stc_smass0) - tot_hmass0
        )


class CoppageBladeLoading(LossModel):
    def residual(
        self,
        tot_hmass0,
        tot_hmass1,
        kin_W1,
        kin_W_shroud1,
        kin_U1,
        geo_n_blades,
        Ratio_D1sD2,
        Df,
        K,
    ):
        work = tot_hmass1 - tot_hmass0
        Df = (
            1
            - kin_W1 / kin_W_shroud1
            + K
            * (abs(work) / kin_U1 ^ 2)
            * (kin_W1 / kin_W_shroud1)
            / (((geo_n_blades / np.pi) * (1 - Ratio_D1sD2)) + (2 * Ratio_D1sD2))
        )
        Dht_bl = 0.05 * (Df * kin_U1) ^ 2


class ClearanceJansen(LossModel):
    def residual(self, R1s, R1h, R2, rho_out, rho_in, eps, H2, vt2, N_blades, vm1):
        K = abs((R1s ^ 2 - R1h ^ 2) / ((R2 - R1s) * (1 + rho_out / rho_in)))
        Dht_cl = (
            0.6
            * eps
            / H2
            * vt2
            * np.sqrt(np.abs((4 * np.pi / (H2 * N_blades)) * K * vt2 * vm1))
        )


class SkinFrictionJansesn(LossModel):
    def residual(
        self,
        R1,
        R2,
        R1s,
        R1h,
        H2,
        L_ax,
        beta1s,
        beta1h,
        beta2_blade,
        N_blades,
        v1s,
        v2,
        w1s,
        w1h,
        w2,
        mu_in,
        mu_out,
        Ra,
        Cf_smooth,
        Cf_rough,
        m,
    ):
        L_hyd = (
            np.pi
            / 8
            * (2 * R2 - R1s - R1h - H2 + 2 * L_ax)
            * (2 / ((np.cos(beta1s) + np.cos(beta1h)) / 2 + np.cos(beta2_blade)))
        )

        D_hyd = (
            np.pi
            * ((2 * R1s) ^ 2 - (2 * R1h) ^ 2)
            / ((4 * np.pi * R1) + N_blades * ((2 * R1s) - (2 * R1h)))
        )

        w_mean = (v1s + v2 + w1s + 2 * w1h + 3 * w2) / 8

        Re = m / (((mu_in + mu_out) / 2) * D_hyd)
        Re_e = (Re - 2000) * Ra / D_hyd

        Cf_smooth = (
            1 / (np.log10((2.51 / (Re * np.sqrt((4 * Cf_smooth)))) ^ (-2)))
        ) ^ 2 / 4
        Cf_rough = (1 / (np.log10((Ra / (3.71 * D_hyd)) ^ (-2)))) ^ 2 / 4
        Cf = Cf_smooth + (Cf_rough - Cf_smooth) * (1 - (60 / Re_e))

        Dht_sf = 2 * abs(Cf) * (L_hyd / D_hyd) * (w_mean ^ 2)
