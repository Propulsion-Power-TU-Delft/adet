from typing import Any
from adet.equations import EquationBase
import numpy as np

from adet.fluid.eos import CasadiEoS
import CoolProp as cp


class DentonProfileLoss(EquationBase):
    # SKETCH
    # ======
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
    #            x_by_cs1   x_by_cs2

    def __init__(
        self,
        eos: Any,
        scaling_factor: float | tuple[float] | None = None,
    ):
        """
        This requires intermediate state updates, meaning ad eos object has to be
        provided manually
        """
        self.eos = eos
        super().__init__(scaling_factor)

    @staticmethod
    def _get_velocity_profile(x_by_cs1, x_by_cs2, k_prof, kin_W0, kin_W1):
        # Positions
        x_by_cs = np.array(
            [
                0.0 * x_by_cs1,
                x_by_cs1,
                x_by_cs2,
                x_by_cs1 / x_by_cs1,  # = 1 (with the correct shape)
            ]
        ).T

        # Velocity difference (out - in)
        delta_W = kin_W1 - kin_W0

        # Pressure values, suction and pressure side
        W_mid_ss = 2 * k_prof * kin_W1 + delta_W
        W_mid_ps = k_prof * kin_W0 - delta_W

        # Full velocity distribution
        # -> Suction Side
        W_distr_ss = np.array(
            [
                kin_W0,
                W_mid_ss,
                W_mid_ss,
                kin_W1,
            ]
        ).T
        # -> Pressure Side
        W_distr_ps = np.array(
            [
                0.0 * kin_W0,
                W_mid_ps,
                W_mid_ps,
                kin_W1,
            ]
        ).T
        return x_by_cs, W_distr_ss, W_distr_ps

    def residual(self, kin_W0, kin_W1, oth_x_by_cs1, oth_x_by_cs2, oth_k_prof):
        num_span = max(kin_W0.shape)

        _eos_callback = CasadiEoS(
            f'Denton_HS_{id(self)}',
            self.eos,
            cp.HmassSmass_INPUTS,
            ['p'],
            num_span,
        )


class RectVelProfile(EquationBase):
    """
    References
    ----------
    - Section 5.4.3 of Greitzer's `Internal Flows`
    - Section 7.1 of Denton's 1993 hit paper `Loss Mechanisms in Turbomachines`
    """

    # Sketch
    #                               Trailing edge
    #  ^                  +---------------+--------  kin_W1
    #  | Velocity         |               |
    #  |                  | - - - - - - - |- W_mean
    #  |                  |               |
    #  |    kin_W0  ------+---------------+
    #  |             Leading edge
    #  |___________________________> Distance along blade
    #

    def residual(
        self,
        kin_W0,
        kin_W1,
        oth_Cd_profile1,
        oth_bld_len1,
        oth_bld_spacing1,
        tot_p0,
        tot_p1,
        stc_rhomass1,
    ):
        delta_W = kin_W1 - kin_W0
        W_mean = (kin_W1 + kin_W0) / 2

        loss_coeff = (
            2
            * oth_Cd_profile1
            * (oth_bld_len1 / oth_bld_spacing1)
            * (
                2 * (W_mean / kin_W0) ** 3
                + 6 * (W_mean / kin_W1) * (delta_W / kin_W1) ** 2
            )
        )

        return (tot_p0 - tot_p1) - loss_coeff * 0.5 * stc_rhomass1 * kin_W1**2


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import casadi as cs

    N_SPAN = 5
    dl = DentonProfileLoss(None)
    x, V_ss, V_ps = dl._get_velocity_profile(
        np.linspace(0.3, 0.4, N_SPAN),
        np.linspace(0.6, 0.7, N_SPAN),
        np.ones(N_SPAN) * 0.6,
        np.linspace(100, 200, N_SPAN),
        np.linspace(150, 225, N_SPAN),
    )

    x1 = cs.MX.sym('x1')  # type:ignore
    x2 = cs.MX.sym('x2')  # type:ignore
    k = cs.MX.sym('k')  # type:ignore
    W0 = cs.MX.sym('W0')  # type:ignore
    W1 = cs.MX.sym('W1')  # type:ignore

    dl._get_velocity_profile(x1, x2, k, W0, W1)

    # Plots
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('Dark2')
    for i in range(N_SPAN):
        color = cmap(i / (N_SPAN - 1))
        ax.plot(x[i], V_ss[i], color=color)
        ax.plot(x[i], V_ps[i], color=color)

    ax.grid(True)

    fig.show()
    # plt.close('all')


# This is being ported
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
