from adet.equations.base_equation import EquationBase
from .base_loss import LossModel
import numpy as np
import CoolProp as cp


class EndWallVelocities(EquationBase):
    """Computation of the velocities at the endwall, also for single span cases"""

    def residual(
        self,
        kin_Wt0,
        kin_Wm0,
        geo_meridional_angle0,
        kin_omega0,
        geo_rr0,
        geo_height0,
        kin_W_hub0,
        kin_W_shroud0,
    ):
        num_span = max(kin_Wt0.shape)
        if num_span == 1:
            midspan = 0
        else:
            midspan = num_span // 2

        deltaW = kin_omega0 * geo_height0 * np.cos(geo_meridional_angle0) / 2
        Wt_hub = kin_Wt0[midspan] - deltaW
        Wt_shroud = kin_Wt0[midspan] + deltaW

        r1 = kin_W_hub0 - np.sqrt(kin_Wm0[midspan] ** 2 + Wt_hub**2)
        r2 = kin_W_shroud0 - np.sqrt(kin_Wm0[midspan] ** 2 + Wt_shroud**2)

        return r1, r2


class CoppageBladeLoading(LossModel):
    def residual(
        self,
        tot_hmass0,
        tot_hmass1,
        #
        geo_rmid0,
        geo_rmid1,
        geo_height0,
        geo_meridional_angle0,
        Ratio_D1sD2,
        #
        kin_U1,
        kin_W1,
        kin_W_shroud0,
        geo_num_blades1,
        oth_delta_hmass_loading,
        oth_K_loading,  # 0.75
    ):
        work = tot_hmass1 - tot_hmass0
        r_shroud0 = geo_rmid0 + geo_height0 * np.cos(geo_meridional_angle0) / 2

        Df = (
            1
            - kin_W1 / kin_W_shroud0
            + oth_K_loading
            * (abs(work) / kin_U1 ^ 2)
            * (kin_W1 / kin_W_shroud0)
            / (
                ((geo_num_blades1 / np.pi) * (1 - r_shroud0 / geo_rmid1))
                + (2 * Ratio_D1sD2)
            )
        )
        return oth_delta_hmass_loading - 0.05 * (Df * kin_U1) ^ 2


class ClearanceJansen(LossModel):
    def residual(
        self,
        R1s,
        R1h,
        R2,
        rho_out,
        rho_in,
        eps,
        H2,
        vt2,
        N_blades,
        vm1,
        oth_delta_hmass_clearance1,
    ):
        K = abs((R1s ^ 2 - R1h ^ 2) / ((R2 - R1s) * (1 + rho_out / rho_in)))
        return oth_delta_hmass_clearance1 - (
            0.6
            * eps
            / H2
            * vt2
            * np.sqrt(np.abs((4 * np.pi / (H2 * N_blades)) * K * vt2 * vm1))
        )


class SkinFrictionJansen(LossModel):
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
