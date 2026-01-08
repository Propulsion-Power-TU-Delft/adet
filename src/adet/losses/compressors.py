import numpy as np

from adet.equations.utils import safe_abs
from adet.losses.base_loss import LossModel
from adet.equations.base_equation import DeviationModel


class BackstromSlip(DeviationModel):
    def residual(
        self,
        oth_slip_factor1,
        oth_slip_factCoeff1,
        geo_rr0,
        geo_rr1,
        geo_num_blades1,
        geo_metal_angle1,
        oth_eff_solidity1,
        kin_U1,
        kin_Wm1,
        kin_Wt1,
    ):
        radius_ratio = geo_rr0 / geo_rr1
        r0 = oth_eff_solidity1 - (
            (1 - radius_ratio)
            * geo_num_blades1
            / (2 * np.pi * np.cos(geo_metal_angle1))
        )

        r1 = oth_slip_factor1 - (
            1
            - 1
            / (
                1
                + oth_slip_factCoeff1
                * oth_eff_solidity1
                * np.cos(geo_metal_angle1) ** 0.5
            )
        )

        slip_velocity = kin_U1 * (1 - oth_slip_factor1)  # Sign depends on U

        Wt1_noslip = kin_Wm1 * np.tan(geo_metal_angle1)

        r2 = kin_Wt1 - (Wt1_noslip - slip_velocity)

        return r0, r1, r2


class CoppageBladeLoading(LossModel):
    def residual(
        self,
        # Enthalpies for work
        tot_hmass0,
        tot_hmass1,
        # Geometry
        geo_rmid0,
        geo_rmid1,
        geo_height0,
        geo_meridional_angle0,
        # Kinematics
        kin_U1,
        kin_W1,
        kin_W_shroud0,
        geo_num_blades1,
        # Coefficients
        oth_delta_hmass_loading1,
        oth_bl_loadingCoeff1,  # 0.75
    ):
        work_abs = safe_abs(tot_hmass1 - tot_hmass0)
        r0s_by_r1 = (geo_rmid0 + geo_height0 / 2) / geo_rmid1
        diff_coeff = (
            1
            - kin_W1 / kin_W_shroud0
            + oth_bl_loadingCoeff1
            * (work_abs / kin_U1**2)
            * (kin_W1 / kin_W_shroud0)
            / (((geo_num_blades1 / np.pi) * (1 - r0s_by_r1)) + (2 * r0s_by_r1))
        )

        return oth_delta_hmass_loading1 - 0.05 * (diff_coeff * kin_U1) ** 2


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
