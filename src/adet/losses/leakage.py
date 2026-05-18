from adet.equations.base_equation import EquationConfig
import casadi as cs
import CoolProp as cp

from adet.equations.utils import trapezoid2
from adet.variables import NodeVariables, ThermoVariables
from adet.losses.base_loss import LossModel
from adet.losses.profile import rectangular_vel_profile, trapezoidal_vel_profile

n0 = NodeVariables(0)
n1 = NodeVariables(1)
thrm = ThermoVariables()


class DentonTrapLeakage(LossModel):
    config = EquationConfig(
        manual_units=('J / kg / K',),
        input_pair=cp.HmassSmass_INPUTS,
        out_properties=(thrm.Pressure, thrm.Density),
    )

    def residual(
        self,
        h_rlt0: n0.rlt.Enthalpy.Hint,
        s0: n0.stc.Entropy.Hint,
        T_is1: n1.oth.Tis_stc.Hint,
        W0: n0.kin.W_mag.Hint,
        W1: n1.kin.W_mag.Hint,
        camb_len1: n1.geo.CamberLength.Hint,
        ch_mf1: n1.oth.ChMassflow.Hint,
        xi_A1: n1.oth.XiCambLenA.Hint,
        xi_B1: n1.oth.XiCambLenB.Hint,
        k_prof1: n1.oth.ProfileLoading.Hint,
        disch_coeff1: n1.oth.DischCoeff.Hint,
        tip_clr1: n1.geo.TipClearance.Hint,
        ds_leak1: n1.loss.Ds_leakage.Hint,
    ):
        xi_by_camb_len, W_distr_ss, W_distr_ps = trapezoidal_vel_profile(
            xi_A1, xi_B1, k_prof1, W0, W1
        )

        p_ss, _ = self.eos(h_rlt0 - W_distr_ss**2 / 2, s0)
        p_ps, rho_ps = self.eos(h_rlt0 - W_distr_ps**2 / 2, s0)
        xi_dimensional = xi_by_camb_len * camb_len1

        delta_p = cs.fmax(p_ps - p_ss, 0.1)  # Avoid NaN in root
        # [kg / s / m]
        dm_by_dxi = disch_coeff1 * tip_clr1 * (2 * rho_ps * delta_p) ** 0.5

        # [ J / kg ] * [ kg / s / m ] * [m] = [ J / s ]
        leak_integral = trapezoid2(
            W_distr_ss**2 * (1 - W_distr_ps / W_distr_ss) * dm_by_dxi, xi_dimensional
        )

        # [ J / s ] * [ s / kg ] * [ 1 / K ] = [ J / kg / K ] OK
        return ds_leak1 - leak_integral / (ch_mf1 * T_is1)


class DentonRectLeakage(LossModel):
    config = EquationConfig(
        manual_units=('J / kg / K',),
        input_pair=cp.HmassSmass_INPUTS,
        out_properties=(thrm.Pressure, thrm.Density),
    )

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
        geo_camb_len1,
        # Misc
        oth_ch_massflow1,
        oth_k_prof1,
        # Loss dependencies
        oth_dischCoeff1,  # 0.3 - 0.4 for rotating cascades
        geo_tip_clearance1,
        oth_delta_smass_leakage1,
    ):
        W_ss, W_ps = rectangular_vel_profile(kin_W0, kin_W1, oth_k_prof1)

        p_ss, _ = self.eos(rlt_hmass0 - W_ss**2 / 2, stc_smass0)
        p_ps, rho_ps = self.eos(rlt_hmass0 - W_ps**2 / 2, stc_smass0)

        delta_p = cs.fmax(p_ps - p_ss, 0.1)  # Avoid NaN in root derivatives
        # [kg / s / m]
        dm_by_dxi = oth_dischCoeff1 * geo_tip_clearance1 * (2 * rho_ps * delta_p) ** 0.5

        # [ J / kg ] * [ kg / s / m ] * [m] = [ J / s ]
        leak_integral = W_ss**2 * (1 - W_ps / W_ss) * dm_by_dxi * geo_camb_len1

        # [ J / s ] * [ s / kg ] * [ 1 / K ] = [ J / kg / K ] OK
        return oth_delta_smass_leakage1 - leak_integral / (
            oth_ch_massflow1 * oth_stc_T_is1
        )
