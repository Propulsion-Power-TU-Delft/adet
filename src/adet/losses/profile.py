from typing import Callable, cast
from adet.equations import EquationBase

from adet.fluid.casadi_eos import CasadiEoS
import CoolProp as cp
import casadi as cs
import numpy as np

from adet.fluid.settings import ExternalFluidModel
from adet.tools.coolprop_utils import DebugAbstractState


# Greitzer model
# ==============
#                               Trailing edge
#  ^                  +---------------+--------  kin_W1
#  | Velocity         |               |
#  |                  | - - - - - - - |- W_mean
#  |                  |               |
#  |    kin_W0  ------+---------------+
#  |             Leading edge
#  |___________________________> Distance along blade
#
class RectVelocityIncompressible(EquationBase):
    """
    References
    ----------
    - Section 5.4.3 of Greitzer's `Internal Flows`
    - Section 7.1 of Denton's 1993 hit paper `Loss Mechanisms in Turbomachines`
    """

    def residual(
        self,
        kin_W0,
        kin_W1,
        oth_Cd_profile1,
        geo_camb_len1,
        geo_pitch1,
        tot_p0,
        tot_p1,
        stc_rhomass1,
    ):
        delta_W = kin_W1 - kin_W0
        W_mean = (kin_W1 + kin_W0) / 2

        loss_coeff = (
            2
            * oth_Cd_profile1
            * (geo_camb_len1 / geo_pitch1)
            * (
                2 * (W_mean / kin_W0) ** 3
                + 6 * (W_mean / kin_W1) * (delta_W / kin_W1) ** 2
            )
        )

        return (tot_p0 - tot_p1) - loss_coeff * 0.5 * stc_rhomass1 * kin_W1**2


# Denton Model
# ============
#
#  ^              Suction
#  | Velocity     side
#  |            ___________ _ 2 k kin_W1 + delta_W
#  |           /           \
#  |          /             \ _ kin_W1
#  |       kin_W0           /
#  |            ___________/ _ k kin_W0 - delta_W
#  |           /  Pressure
#  |       0  /   side
#  |____________________________> Distance along blade
#               |          |
#            x_by_camb_len1   x_by_camb_len2


class DentonProfileLoss(EquationBase):
    """
    Axial blade profile losses based on simplified pressure distribution.
    It should be able to be used for axial compresstore and turbine blades,
    so far it has been tested for turbines (check the integral signs mainly)


    Warning
    -------
    - This function is ONLY compatible with CasADi
    - This assumes an external gas model
    """

    skip_unit_check = True
    manual_units = ('N', 'J / kg / K')

    def __init__(
        self,
        fluid_model: ExternalFluidModel,
        scaling_factor: list[float] | None = None,
    ):
        """
        This requires intermediate state updates, meaning ad eos object has to be
        provided manually
        """
        self._fluid_model = fluid_model
        self._eos_callback = None
        super().__init__(scaling_factor)

    @staticmethod
    def _build_velocity_profile(
        x_by_camb_len_A, x_by_camb_len_B, k_prof, kin_W0, kin_W1
    ) -> tuple[cs.DM, cs.DM, cs.DM]:
        """Build the velocity proile"""
        # Positions
        x_by_camb_len = cs.horzcat(
            0.0 * x_by_camb_len_A,
            x_by_camb_len_A,
            x_by_camb_len_B,
            x_by_camb_len_A / x_by_camb_len_A,
        )

        # Velocity values
        CLIP_RATIO = 5  # Clip pressure side to inlet W / ratio
        delta_W = kin_W1 - kin_W0

        # NOTE: Nothing prevents k_prof from being negative, because
        # the values used in the residual formulation to compute the pressure
        # only use W**2.
        # The problem arises when you try to use the clip ratio, because you
        # are assuming the value of the sign by using the max function
        # => Solution = Take the absolute value of k_prof

        k_prof = cs.fabs(k_prof)  # Take the absolute value
        W_mid_ps = cs.fmax(k_prof * kin_W0 - delta_W, kin_W0 / CLIP_RATIO)
        W_mid_ss = 2 * k_prof * kin_W1 + delta_W

        # Full velocity distribution
        W_distr_ss = cs.horzcat(1.0 * kin_W0, W_mid_ss, W_mid_ss, kin_W1)  # Suction
        W_distr_ps = cs.horzcat(0.0 * kin_W0, W_mid_ps, W_mid_ps, kin_W1)  # Pressure

        return x_by_camb_len, W_distr_ss, W_distr_ps

    def _compute_thermo_distributions(self, rlt_hmass0, stc_smass0, W_distr):
        """
        Compute the pressure distribution from total enthalpy and entorpy
        at the inlet

        Note
        ----
        This approximates the flow as isentropic along the blade itself
        and is valid only for axial machines where total relative enthalpy
        is conserved
        """
        num_span = max(rlt_hmass0.shape)
        NUM_STREAM = 4

        if self._eos_callback is None:
            _eos_callback = CasadiEoS(
                f'Denton_HS_{id(self)}',
                self._fluid_model.eos_object,
                cp.HmassSmass_INPUTS,
                ['p', 'rhomass', 'T'],
                num_span,
            )
            # ! Manual typing annotation !
            self._eos_callback = cast(
                Callable[..., tuple[cs.DM, cs.DM, cs.DM]],
                _eos_callback,
            )

        stc_hmass_dst = [rlt_hmass0 - W_distr[:, i] ** 2 / 2 for i in range(NUM_STREAM)]

        # Extract p, T, and density distributions from abstract state
        p_list, rho_list, T_list = [], [], []
        for h in stc_hmass_dst:
            p, rho, T = self._eos_callback(h, stc_smass0)
            p_list.append(p)
            rho_list.append(rho)
            T_list.append(T)

        p_cat = cs.horzcat(*p_list)
        rho_cat = cs.horzcat(*rho_list)
        T_cat = cs.horzcat(*T_list)

        return p_cat, rho_cat, T_cat

    @staticmethod
    def _trapezoid(y, x):
        """Trapezoidal rule"""
        dx = x[:, 1:] - x[:, :-1]
        integrand = (y[:, :-1] + y[:, 1:]) * dx / 2
        return cs.sum2(integrand)

    def residual(
        self,
        # Thermo
        rlt_hmass0,
        stc_smass0,
        stc_smass1,
        # Kine
        kin_W0,
        kin_W1,
        kin_Vt0,
        kin_Vt1,
        # Misc
        oth_ch_massflow1,
        oth_x_by_camb_len_A1,
        oth_x_by_camb_len_B1,
        oth_k_prof1,
        oth_Cd_profile1,
        # Geo
        geo_hh1,
        geo_chord_ax1,
        geo_camb_len1,
        geo_stagger1,
        geo_n_blades1,
    ):
        x_by_camb_len, W_distr_ss, W_distr_ps = self._build_velocity_profile(
            oth_x_by_camb_len_A1, oth_x_by_camb_len_B1, oth_k_prof1, kin_W0, kin_W1
        )

        # TODO: Idea, make smass1 also an input and distribute
        # entropy (linearly?) between inlet and outlet
        p_ss, rho_ss, T_ss = self._compute_thermo_distributions(
            rlt_hmass0, stc_smass0, W_distr_ss
        )
        p_ps, rho_ps, T_ps = self._compute_thermo_distributions(
            rlt_hmass0, stc_smass0, W_distr_ps
        )

        # X is the along the chord
        x_dimensional = x_by_camb_len * geo_camb_len1

        # Trapezoidal integration (can't use np.trapezoidal for differentiability)
        # (trapezoidal rule is exact because everything is linear)
        # [Pa * m = N / m]
        pressure_integral = self._trapezoid(p_ps - p_ss, x_dimensional)

        # Entropy generation from 2D viscous dissipation
        # [ kg / m**3 ] * [ m**3 / s**3 ] / [K] * [m]
        # = [ kg * m / s**3 / K ] = [ N / s / K ]
        entropy_integral_ps = self._trapezoid(
            oth_Cd_profile1 * rho_ps * W_distr_ps**3 / T_ps,
            x_dimensional,
        )

        entropy_integral_ss = self._trapezoid(
            oth_Cd_profile1 * rho_ss * W_distr_ss**3 / T_ss,
            x_dimensional,
        )

        entropy_integral = entropy_integral_ps + entropy_integral_ss
        # NOTE: Integral = Entropy production (NOT SPECIFIC)
        # per unit length (in height direction) per unit time
        # To convert to specific (see residual definition below):
        #    |> multiply by height sector
        #    |> divide my massflow of each channel

        # 1. Tangential momentum balance [N]
        delta_Vt = cs.fabs(kin_Vt1 - kin_Vt0)
        r1 = oth_ch_massflow1 * delta_Vt - pressure_integral * geo_hh1 * np.cos(
            geo_stagger1
        )

        # 2. SPECIFIC entropy generation [J / kg / K]
        r2 = stc_smass1 - stc_smass0 - entropy_integral * geo_hh1 / oth_ch_massflow1

        return r1, r2


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import casadi as cs
    import numpy as np

    # NOTE: You can mix and match cs.DM and
    # numpy array, but remember casadi is consistent
    # with shape manipulation and extraction, and every
    # array is AT LEAST 2D
    N_SPAN = 11
    eos = DebugAbstractState('HEOS', 'Air')
    model = ExternalFluidModel(eos)

    # Define example values
    W0 = np.linspace(100, 300, N_SPAN)
    W1 = np.linspace(175, 400, N_SPAN)

    Vt0 = np.linspace(40, 50, N_SPAN)
    Vt1 = np.linspace(40, 50, N_SPAN)

    x_by_camb_len_A = cs.linspace(0.3, 0.3, N_SPAN)
    x_by_camb_len_B = np.linspace(0.6, 0.6, N_SPAN)

    dummy_hh = cs.DM.ones(N_SPAN) * 0.05  # pyright:ignore

    dummy_ht = cs.DM.ones(N_SPAN) * 5e5  # pyright:ignore
    dummy_st = np.ones(N_SPAN) * 4000
    dummy_mass_flow = np.ones(N_SPAN) * 10

    Cd = 0.002 * np.ones(N_SPAN)
    chord_ax = 0.1 * np.ones(N_SPAN)
    camb_len = 0.15 * np.ones(N_SPAN)
    pitch = 0.2 * np.ones(N_SPAN)
    k_prof = 0.6 * np.ones(N_SPAN)

    # Test
    dl = DentonProfileLoss(model)

    # Plots to check pressure and velocity distro
    # => First station should clip to W_1 / 5
    x_by_camb_len, V_ss, V_ps = dl._build_velocity_profile(
        x_by_camb_len_A, x_by_camb_len_B, k_prof, W0, W1
    )
    p_ss, rho_ss, T_ss = dl._compute_thermo_distributions(dummy_ht, dummy_st, V_ss)
    p_ps, rho_ps, T_ps = dl._compute_thermo_distributions(dummy_ht, dummy_st, V_ps)
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    cmap = plt.get_cmap('viridis')
    for i in range(N_SPAN):
        color = cmap(i / (N_SPAN))
        ax[0].plot(np.array(x_by_camb_len)[i], np.array(V_ss)[i], color=color)
        ax[0].plot(np.array(x_by_camb_len)[i], np.array(V_ps)[i], color=color)

        ax[1].plot(np.array(x_by_camb_len)[i], np.array(p_ss)[i], color=color)
        ax[1].plot(np.array(x_by_camb_len)[i], np.array(p_ps)[i], color=color)

        ax[0].grid(True)
        ax[1].grid(True)

    fig.show()
    plt.close('all')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   ____   _     _     __  __          _       _        #
#  / __ \ | |   | |   |  \/  |        | |     | |       #
# | |  | || | __| |   | \  / | ___  __| | ___ | | ___   #
# | |  | || |/ _  |   | |\/| |/ _ \/ _  |/ _ \| |/ __|  #
# | |__| || ||(_| |   | |  | ||(_)||(_| || __/| |\__ \  #
#  \____/ |_|\____|   |_|  |_|\___/\____|\___||_||___/  #
#                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# NOTE: These below are model of previous implementations
# available in turbosim, they are not in a working state and only
# act as reference for the implementations


# class OLD_DentonProfileLosses(EquationBase):
#     def BBL_loop(self, p, *data):
#         """
#         Internal loop of BBL loss (rectangular V profile) used to match tangential
#         momentum balance while considering
#         flow compressibility
#         """
#
#         delta_Vt, V_mean, Vax_mean, D_mean, Cax_s, delta_V, ht_in, s_in, den = data
#         V_mean = p
#
#         Vss = V_mean + delta_V
#         Vps = V_mean - delta_V
#
#         # Limiter?
#         # if Vps < 0:
#         #     residual = 100
#         # else:
#
#         hps = ht_in - Vps**2 / 2
#         hss = ht_in - Vss**2 / 2
#         self.flow.fluid.EoS.update(fld.CoolProp.HmassSmass_INPUTS, hps, s_in)
#         Pps = self.flow.fluid.EoS.p()
#         self.flow.fluid.EoS.update(fld.CoolProp.HmassSmass_INPUTS, hss, s_in)
#         Pss = self.flow.fluid.EoS.p()
#         residual = np.abs(D_mean * Vax_mean * delta_Vt - (Pps - Pss) * Cax_s) / den
#
#         return residual
#
#     def residual(self, *args):
#         if row == 'stator':
#             flag_row = 0
#             D_in = self.flow.D[0, :]
#             D_out = self.flow.D[1, :]
#             V_in = self.flow.V[0, :]
#             V_out = self.flow.V[1, :]
#             flow_angle_in = np.deg2rad(self.flow.alpha[0, :])
#             flow_angle_out = np.deg2rad(self.flow.alpha[1, :])
#             ht_in = self.flow.ht[0, :]
#             s_in = self.flow.s[0, :]
#         elif row == 'rotor':
#             flag_row = 1
#             D_in = self.flow.D[2, :]
#             D_out = self.flow.D[3, :]
#             V_in = self.flow.W[2, :]
#             V_out = self.flow.W[3, :]
#             flow_angle_in = np.deg2rad(self.flow.beta[2, :])
#             flow_angle_out = np.deg2rad(self.flow.beta[3, :])
#             ht_in = self.flow.htr[2, :]
#             s_in = self.flow.s[2, :]
#
#         delta_s_ps = np.zeros(self.flow.Nslices)
#         delta_s_ss = np.zeros(self.flow.Nslices)
#
#         Vt_in = V_in * np.sin(flow_angle_in)
#         Vt_out = V_out * np.sin(flow_angle_out)
#         Vax_in = V_in * np.cos(flow_angle_in)
#         Vax_out = V_out * np.cos(flow_angle_out)
#
#         V_mean = (V_in + V_out) / 2
#         Vax_mean = (Vax_in + Vax_out) / 2
#         D_mean = (D_in + D_out) / 2
#
#         Cs_s = Cs_c * C_s
#         delta_Vt = np.abs(Vt_out - Vt_in)
#         delta_V = delta_Vt / (2 * Cs_s)
#         den = D_mean * Vax_mean * delta_Vt  # used for BBL_loop if Vprofile == 'simple'
#         V_mean_incompr = (
#             Vax_mean * delta_Vt / (2.0 * Cax_s * delta_V)
#         )  # used for BBL_loop if Vprofile == 'simple'
#
#         self.flow.k[flag_row, :] = 0.4 * np.ones(self.flow.Nslices)  # first guess
#         x1 = np.linspace(0, 0.375, int(self.Nstream / 3))
#         x2 = np.linspace(0.375, 0.625, int(self.Nstream / 3))
#         x3 = np.linspace(0.625, 1.0, int(self.Nstream / 3))
#
#         for ii in range(self.flow.Nslices):
#             if Vprofile == 'simple':
#                 self.flow.k[flag_row, ii] = 100
#                 try:
#                     data = (
#                         delta_Vt[ii],
#                         V_mean[ii],
#                         Vax_mean[ii],
#                         D_mean[ii],
#                         Cax_s,
#                         delta_V[ii],
#                         ht_in[ii],
#                         s_in[ii],
#                         den[ii],
#                     )
#                     optim = opt.minimize(
#                         self.BBL_loop, V_mean_incompr[ii], args=data, tol=0.01
#                     )
#                     V_mean[ii] = optim.x
#                 except:
#                     V_mean[ii] = V_mean_incompr[ii]
#                     self.flag_profile[flag_row, ii] = 1.0
#                     if warning == 1:
#                         print(
#                             'Error in Blade Boundary Layer Loss Model: going back to Incompressible Loss Model'
#                         )
#
#                 if (V_mean[ii] - delta_V[ii]) > 0:
#                     self.Vps[flag_row, ii, :] = V_mean[ii] - delta_V[ii]
#                 else:
#                     self.Vps[flag_row, ii, :] = 5  # impose a minimum value of V_PS
#                     self.flag_profile[flag_row, ii] = 1.0
#
#                 self.Vss[flag_row, ii, :] = V_mean[ii] + delta_V[ii]
#                 self.hss[flag_row, ii, :] = (
#                     ht_in[ii] - self.Vss[flag_row, ii, :] ** 2 / 2
#                 )
#                 self.hps[flag_row, ii, :] = (
#                     ht_in[ii] - self.Vps[flag_row, ii, :] ** 2 / 2
#                 )
#
#                 self.flow.fluid.EoS.update(
#                     fld.CoolProp.HmassSmass_InpUTS, self.hps[flag_row, ii, 0], s_in[ii]
#                 )
#                 self.Pps[flag_row, ii, :] = self.flow.fluid.EoS.p()
#                 self.Dps[flag_row, ii, :] = self.flow.fluid.EoS.rhomass()
#                 self.Tps[flag_row, ii, :] = self.flow.fluid.EoS.T()
#
#                 self.flow.fluid.EoS.update(
#                     fld.CoolProp.HmassSmass_InpUTS, self.hss[flag_row, ii, 0], s_in[ii]
#                 )
#                 self.Pss[flag_row, ii, :] = self.flow.fluid.EoS.p()
#                 self.Dss[flag_row, ii, :] = self.flow.fluid.EoS.rhomass()
#                 self.Tss[flag_row, ii, :] = self.flow.fluid.EoS.T()
#             else:
#                 res = 10
#                 toll = 0.05
#                 while abs(res) > toll:
#                     if (self.flow.k[flag_row, ii] * V_in[ii] - delta_V[ii]) > (
#                         V_in[ii] / 5
#                     ):
#                         self.Vss1[ii, :] = (
#                             V_in[ii]
#                             + (
#                                 (
#                                     2 * self.flow.k[flag_row, ii] * V_out[ii]
#                                     + delta_V[ii]
#                                 )
#                                 - V_in[ii]
#                             )
#                             * x1
#                             / x1[-1]
#                         )
#                         self.Vps1[ii, :] = (
#                             (self.flow.k[flag_row, ii] * V_in[ii] - delta_V[ii])
#                             * x1
#                             / x1[-1]
#                         )
#                         self.Vss2[ii, :] = (
#                             2 * self.flow.k[flag_row, ii] * V_out[ii] + delta_V[ii]
#                         ) * np.ones(len(x2))
#                         self.Vps2[ii, :] = (
#                             self.flow.k[flag_row, ii] * V_in[ii] - delta_V[ii]
#                         ) * np.ones(len(x2))
#                         self.Vss3[ii, :] = (
#                             2 * self.flow.k[flag_row, ii] * V_out[ii] + delta_V[ii]
#                         ) + (
#                             V_out[ii]
#                             - (2 * self.flow.k[flag_row, ii] * V_out[ii] + delta_V[ii])
#                         ) * (x3 - x3[0]) / (x3[-1] - x3[0])
#                         self.Vps3[ii, :] = (
#                             self.flow.k[flag_row, ii] * V_in[ii] - delta_V[ii]
#                         ) + (
#                             V_out[ii]
#                             - (self.flow.k[flag_row, ii] * V_in[ii] - delta_V[ii])
#                         ) * (x3 - x3[0]) / (x3[-1] - x3[0])
#                     else:
#                         self.Vss1[ii, :] = (
#                             V_in[ii]
#                             + (
#                                 (
#                                     2 * self.flow.k[flag_row, ii] * V_out[ii]
#                                     + delta_V[ii]
#                                 )
#                                 - V_in[ii]
#                             )
#                             * x1
#                             / x1[-1]
#                         )
#                         self.Vps1[ii, :] = (V_in[ii] / 5) * x1 / x1[-1]
#                         self.Vss2[ii, :] = (
#                             2 * self.flow.k[flag_row, ii] * V_out[ii] + delta_V[ii]
#                         ) * np.ones(len(x2))
#                         self.Vps2[ii, :] = (V_in[ii] / 5) * np.ones(len(x2))
#                         self.Vss3[ii, :] = (
#                             2 * self.flow.k[flag_row, ii] * V_out[ii] + delta_V[ii]
#                         ) + (
#                             V_out[ii]
#                             - (2 * self.flow.k[flag_row, ii] * V_out[ii] + delta_V[ii])
#                         ) * (x3 - x3[0]) / (x3[-1] - x3[0])
#                         self.Vps3[ii, :] = (V_in[ii] / 5) + (
#                             V_out[ii] - (V_in[ii] / 5)
#                         ) * (x3 - x3[0]) / (x3[-1] - x3[0])
#
#                     hss1 = ht_in[ii] - self.Vss1[ii, :] ** 2 / 2
#                     hps1 = ht_in[ii] - self.Vps1[ii, :] ** 2 / 2
#                     hss2 = ht_in[ii] - self.Vss2[ii, :] ** 2 / 2
#                     hps2 = ht_in[ii] - self.Vps2[ii, :] ** 2 / 2
#                     hss3 = ht_in[ii] - self.Vss3[ii, :] ** 2 / 2
#                     hps3 = ht_in[ii] - self.Vps3[ii, :] ** 2 / 2
#
#                     for jj in range(int(self.Nstream / 3)):
#                         self.flow.fluid.EoS.update(
#                             fld.CoolProp.HmassSmass_InpUTS, hss1[jj], s_in[ii]
#                         )
#                         self.Pss1[ii, jj] = self.flow.fluid.EoS.p()
#                         self.flow.fluid.EoS.update(
#                             fld.CoolProp.HmassSmass_InpUTS, hps1[jj], s_in[ii]
#                         )
#                         self.Pps1[ii, jj] = self.flow.fluid.EoS.p()
#                         self.flow.fluid.EoS.update(
#                             fld.CoolProp.HmassSmass_InpUTS, hss2[jj], s_in[ii]
#                         )
#                         self.Pss2[ii, jj] = self.flow.fluid.EoS.p()
#                         self.flow.fluid.EoS.update(
#                             fld.CoolProp.HmassSmass_InpUTS, hps2[jj], s_in[ii]
#                         )
#                         self.Pps2[ii, jj] = self.flow.fluid.EoS.p()
#                         self.flow.fluid.EoS.update(
#                             fld.CoolProp.HmassSmass_InpUTS, hss3[jj], s_in[ii]
#                         )
#                         self.Pss3[ii, jj] = self.flow.fluid.EoS.p()
#                         self.flow.fluid.EoS.update(
#                             fld.CoolProp.HmassSmass_InpUTS, hps3[jj], s_in[ii]
#                         )
#                         self.Pps3[ii, jj] = self.flow.fluid.EoS.p()
#
#                     int1 = fld.integrate.trapz(
#                         (self.Pps1[ii, :] - self.Pss1[ii, :]), x=x1
#                     )
#                     int2 = fld.integrate.trapz(
#                         (self.Pps2[ii, :] - self.Pss2[ii, :]), x=x2
#                     )
#                     int3 = fld.integrate.trapz(
#                         (self.Pps3[ii, :] - self.Pss3[ii, :]), x=x3
#                     )
#                     P_int = int1 + int2 + int3
#
#                     res_new = (
#                         D_mean[ii] * Vax_mean[ii] * delta_Vt[ii] - P_int * Cax_s
#                     ) / (D_mean[ii] * Vax_mean[ii] * delta_Vt[ii])
#
#                     if res_new > 0:
#                         self.flow.k[flag_row, ii] += 0.01
#                     else:
#                         self.flow.k[flag_row, ii] -= 0.01
#
#                     if abs(res) < abs(res_new):
#                         self.flag_profile[flag_row, ii] = 1
#                         break
#
#                     res = res_new
#
#             if Vprofile == 'simple':
#                 delta_s_ps[ii] = (
#                     self.Cd
#                     * Cs_s[ii]
#                     * (
#                         self.Dps[flag_row, ii, 0]
#                         * self.Vps[flag_row, ii, 0] ** 3
#                         / self.Tps[flag_row, ii, 0]
#                     )
#                     / (D_mean[ii] * Vax_mean[ii])
#                 )
#                 delta_s_ss[ii] = (
#                     self.Cd
#                     * Cs_s[ii]
#                     * (
#                         self.Dss[flag_row, ii, 0]
#                         * self.Vss[flag_row, ii, 0] ** 3
#                         / self.Tss[flag_row, ii, 0]
#                     )
#                     / (D_mean[ii] * Vax_mean[ii])
#                 )
#             else:
#                 for jj in range(int(self.Nstream / 3)):
#                     self.flow.fluid.EoS.update(
#                         fld.CoolProp.PSmass_InpUTS, self.Pss1[ii, jj], s_in[ii]
#                     )
#                     self.Dss1[ii, jj] = self.flow.fluid.EoS.rhomass()
#                     self.Tss1[ii, jj] = self.flow.fluid.EoS.T()
#                 for jj in range(int(self.Nstream / 3)):
#                     self.flow.fluid.EoS.update(
#                         fld.CoolProp.PSmass_InpUTS, self.Pps1[ii, jj], s_in[ii]
#                     )
#                     self.Dps1[ii, jj] = self.flow.fluid.EoS.rhomass()
#                     self.Tps1[ii, jj] = self.flow.fluid.EoS.T()
#                 for jj in range(int(self.Nstream / 3)):
#                     self.flow.fluid.EoS.update(
#                         fld.CoolProp.PSmass_InpUTS, self.Pss2[ii, jj], s_in[ii]
#                     )
#                     self.Dss2[ii, jj] = self.flow.fluid.EoS.rhomass()
#                     self.Tss2[ii, jj] = self.flow.fluid.EoS.T()
#                 for jj in range(int(self.Nstream / 3)):
#                     self.flow.fluid.EoS.update(
#                         fld.CoolProp.PSmass_InpUTS, self.Pps2[ii, jj], s_in[ii]
#                     )
#                     self.Dps2[ii, jj] = self.flow.fluid.EoS.rhomass()
#                     self.Tps2[ii, jj] = self.flow.fluid.EoS.T()
#                 for jj in range(int(self.Nstream / 3)):
#                     self.flow.fluid.EoS.update(
#                         fld.CoolProp.PSmass_InpUTS, self.Pss3[ii, jj], s_in[ii]
#                     )
#                     self.Dss3[ii, jj] = self.flow.fluid.EoS.rhomass()
#                     self.Tss3[ii, jj] = self.flow.fluid.EoS.T()
#                 for jj in range(int(self.Nstream / 3)):
#                     self.flow.fluid.EoS.update(
#                         fld.CoolProp.PSmass_InpUTS, self.Pps3[ii, jj], s_in[ii]
#                     )
#                     self.Dps3[ii, jj] = self.flow.fluid.EoS.rhomass()
#                     self.Tps3[ii, jj] = self.flow.fluid.EoS.T()
#
#                 if (np.isinf(self.Dps1[ii, :])).any():
#                     good = [e for e in self.Dps1[ii, :] if not (np.isinf(e))]
#                     self.Dps1[ii, np.isinf(self.Dps1[ii, :])] = good[0]
#
#                 intss1 = fld.integrate.trapz(
#                     (self.Dss1[ii, :] * self.Vss1[ii, :] ** 3 / self.Tss1[ii, :]), x=x1
#                 )
#                 intps1 = fld.integrate.trapz(
#                     (self.Dps1[ii, :] * self.Vps1[ii, :] ** 3 / self.Tps1[ii, :]), x=x1
#                 )
#                 intss2 = fld.integrate.trapz(
#                     (self.Dss2[ii, :] * self.Vss2[ii, :] ** 3 / self.Tss2[ii, :]), x=x2
#                 )
#                 intps2 = fld.integrate.trapz(
#                     (self.Dps2[ii, :] * self.Vps2[ii, :] ** 3 / self.Tps2[ii, :]), x=x2
#                 )
#                 intss3 = fld.integrate.trapz(
#                     (self.Dss3[ii, :] * self.Vss3[ii, :] ** 3 / self.Tss3[ii, :]), x=x3
#                 )
#                 intps3 = fld.integrate.trapz(
#                     (self.Dps3[ii, :] * self.Vps3[ii, :] ** 3 / self.Tps3[ii, :]), x=x3
#                 )
#
#                 self.Vss[flag_row, ii, :] = np.concatenate(
#                     (self.Vss1[ii, :], self.Vss2[ii, :], self.Vss3[ii, :])
#                 )
#                 self.Vps[flag_row, ii, :] = np.concatenate(
#                     (self.Vps1[ii, :], self.Vps2[ii, :], self.Vps3[ii, :])
#                 )
#                 self.Pss[flag_row, ii, :] = np.concatenate(
#                     (self.Pss1[ii, :], self.Pss2[ii, :], self.Pss3[ii, :])
#                 )
#                 self.Pps[flag_row, ii, :] = np.concatenate(
#                     (self.Pps1[ii, :], self.Pps2[ii, :], self.Pps3[ii, :])
#                 )
#                 self.Dss[flag_row, ii, :] = np.concatenate(
#                     (self.Dss1[ii, :], self.Dss2[ii, :], self.Dss3[ii, :])
#                 )
#                 self.Dps[flag_row, ii, :] = np.concatenate(
#                     (self.Dps1[ii, :], self.Dps2[ii, :], self.Dps3[ii, :])
#                 )
#                 self.Tss[flag_row, ii, :] = np.concatenate(
#                     (self.Dss1[ii, :], self.Dss2[ii, :], self.Dss3[ii, :])
#                 )
#                 self.Tps[flag_row, ii, :] = np.concatenate(
#                     (self.Dps1[ii, :], self.Dps2[ii, :], self.Dps3[ii, :])
#                 )
#
#                 delta_s_ps[ii] = (
#                     self.Cd
#                     * Cs_s[ii]
#                     * (intps1 + intps2 + intps3)
#                     / (D_mean[ii] * Vax_mean[ii])
#                 )
#                 delta_s_ss[ii] = (
#                     self.Cd
#                     * Cs_s[ii]
#                     * (intss1 + intss2 + intss3)
#                     / (D_mean[ii] * Vax_mean[ii])
#                 )
#
#             self.ds_profile[flag_row, ii] = delta_s_ps[ii] + delta_s_ss[ii]
