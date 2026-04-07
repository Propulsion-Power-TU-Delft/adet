import CoolProp as cp
import numpy as np

from adet.equations.base_equation import EquationBase


class FullIncidence(EquationBase):
    input_pair = cp.HmassSmass_INPUTS
    output_quantities = ('rhomass',)
    manual_units = ('kg / s', 'rad')

    def residual(
        self,
        stc_rhomass0,
        kin_Wm0,
        geo_bld_thick0,
        geo_pitch0,
        geo_metal_angle0,
        stc_smass0,
        rlt_hmass0,
        kin_W_th0,
        geo_hh0,
        kin_beta_opt0,
    ):
        hmass_th = rlt_hmass0 - kin_W_th0**2 / 2
        Wt_th = kin_W_th0 * np.sin(geo_metal_angle0)
        Wm_th = kin_W_th0 * np.cos(geo_metal_angle0)

        rho_th = self.eos(hmass_th, stc_smass0)  # Isentropic

        original_area = geo_hh0 * geo_pitch0
        restrict_area = geo_hh0 * (
            geo_pitch0 - geo_bld_thick0 / np.cos(geo_metal_angle0)
        )

        r1 = stc_rhomass0 * kin_Wm0 * original_area - rho_th * Wm_th * restrict_area
        r2 = kin_beta_opt0 - np.atan2(Wt_th, kin_Wm0)

        return r1, r2


# NOTE: This is an experimental equation to compute the choking
# conditions in parallel to any row, it does not enforce anything
# for now but it is an accurate physical choking prediction that
# does not add overhead. In the future we could do something for
# massflow maximization using Lagrange multipliers like turboflow
class ChokingCriterion(EquationBase):
    manual_units = ('kg / s', 'm / s', 'Pa')
    input_pair = cp.HmassSmass_INPUTS
    output_quantities = ('rhomass', 'speed_sound', 'p')

    def residual(
        self,
        tot_hmass0,
        stc_smass0,
        geo_eff_area0,
        geo_eff_area1,
        geo_metal_angle1,
        kin_U0,
        geo_metal_angle0,
        kin_U1,
        # Outputs
        kin_W_choke0,
        kin_W_choke1,
        oth_p_choke1,
    ):
        Wt_in = kin_W_choke0 * np.sin(geo_metal_angle0)
        Wt_th = kin_W_choke1 * np.sin(geo_metal_angle1)

        Vt_in = Wt_in + kin_U0
        Vt_th = Wt_th + kin_U1

        Vm_in = kin_W_choke0 * np.cos(geo_metal_angle0)
        Vm_th = kin_W_choke1 * np.cos(geo_metal_angle1)

        tot_hmass_th = tot_hmass0 + (kin_U1 * Vt_th - kin_U0 * Vt_in)
        stc_smass_th = stc_smass0

        kin_W_choke0 = minmax_bound(kin_W_choke0, 0.1, 300)
        kin_W_choke1 = minmax_bound(kin_W_choke1, 0.1, 300)

        stc_hmass_in = tot_hmass0 - kin_W_choke0**2 / 2
        stc_hmass_th = tot_hmass_th - kin_W_choke1**2 / 2

        stc_rhomass_in, _, _ = self.eos(stc_hmass_in, stc_smass0)
        stc_rhomass_th, stc_speed_sound_th, stc_p_th = self.eos(
            stc_hmass_th, stc_smass_th
        )

        r1 = (
            stc_rhomass_in * Vm_in * geo_eff_area0
            - stc_rhomass_th * Vm_th * geo_eff_area1
        )

        # Assume velocity perpendicular to blade
        r2 = kin_W_choke1 - stc_speed_sound_th

        r3 = oth_p_choke1 - stc_p_th

        return r1, r2, r3
