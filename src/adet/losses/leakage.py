import CoolProp as cp

from adet.equations.utils import trapezoid2
from adet.losses.base_loss import LossModel
from adet.losses.profile import trapezoidal_vel_profile


class DentonLeakageLoss(LossModel):
    manual_units = ('J / kg',)
    input_pair = cp.HmassSmass_INPUTS
    output_quantities = ('p', 'rhomass')

    def residual(
        self,
        # Thermo
        rlt_hmass0,
        stc_smass0,
        # stc_smass1,
        oth_stc_T_is1,
        # Kine
        kin_W0,
        kin_W1,
        # Geo
        geo_hh1,
        geo_camb_len1,
        # Misc
        oth_ch_massflow1,
        # Loss dependencies
        oth_xi_by_camb_len_A1,
        oth_xi_by_camb_len_B1,
        oth_k_prof1,
        oth_dischCoeff1,  # 0.3 - 0.4 for rotating cascades
        geo_tip_clearance1,
        oth_delta_smass_leakage1,
    ):
        xi_by_camb_len, W_distr_ss, W_distr_ps = trapezoidal_vel_profile(
            oth_xi_by_camb_len_A1, oth_xi_by_camb_len_B1, oth_k_prof1, kin_W0, kin_W1
        )

        p_ss, _ = self.eos(rlt_hmass0 - W_distr_ss**2 / 2, stc_smass0)
        p_ps, rho_ps = self.eos(rlt_hmass0 - W_distr_ps**2 / 2, stc_smass0)
        xi_dimensional = xi_by_camb_len * geo_camb_len1

        dm_by_dxi = (
            oth_dischCoeff1 * geo_tip_clearance1 * (2 * rho_ps * (p_ps - p_ss)) ** 0.5
        )
        leak_integral = trapezoid2(
            W_distr_ss**2 * (1 - W_distr_ps / W_distr_ss) * dm_by_dxi, xi_dimensional
        )

        return oth_delta_smass_leakage1 - leak_integral / (
            oth_ch_massflow1 * oth_stc_T_is1
        )
