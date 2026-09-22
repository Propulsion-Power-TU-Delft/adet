# -*- coding: utf-8 -*-
from copy import deepcopy
from src.wrapper_condensation import update_props

#####################################################################
# TurboSim - Turbomachinery design program
# Content: Turbomachinery loss models
# 2025 - TU Delft - All rights reserved
#####################################################################
# References
# [1] Glassman 1976 - COMPUTER PROGRAM FOR DESIGN ANALYSIS OF RADIAL-INFLOW TURBINES
# [2] Rodgers 1987 - Small High Pressure Ratio Turbines (Lecture Series 1987-07)
# [3] Rene Van den Braembussche. Design and Analysis of Centrifugal Compressors. Wiley, 2019. url: https://learning-oreilly-com.tudelft.idm.oclc.org/library/view/design-and-analysis/9781119424093/.
# [4] Anand 2020 - Design guidelines for supersonic stators operating with fluids made of complex molecules
# [5] Baines 1998 - Prediction of the steady and non-steady flow performance of a highly loaded mixed flow turbine
# [6] Whitfield-Baines 1990 - Design of Radial Turbomachines
# [7] Denton 1993 - Loss Mechanisms in Turbomachines
# [8] Baines 1998 - A MEANLINE PREDICTION METHOD FOR RADIAL TURBINE EFFICIENCY
# [9] Rodgers 1967
# [10] Coull 2017 - Endwall Loss in Turbine Cascades
# [11] Lampart, Tip Leakage Flows in Turbines, 2007
# [12] Coull, Blade Loading and its Application in the Mean-Line Design of Low Pressure Turbines, 2013
# [13] Mee, An Examination to the Contributions to Loss on a Transonic Turbine Blade in Cascade, 1990
# [14] Sieverding, The Base Pressure Problem in Transonic Turbine Cascades, 1980
# [15] Majer, Design guidelines for high-pressure ratio supersonic radial-inflow turbines of organic rankine cycle systems, 2025

from turbosim.src.utils import *
from turbosim.src.interp import *
import turbosim.src.flow_model as fld

class Loss:
    def __init__(self,
                 flow,
                 Cd=0.002,
                 C_cs=0.6,
                 C_cu=0.4,
                 delta_star_H = 0.05,
                 delta_star_theta = 2.0,
                 theta_t = 0.075,
                 Kx = 0.4,
                 Kr = 0.75,
                 Kxr = -0.3,
                 k = 0.0035,
                 **kwargs):
        """
        :param Cd:                  Blade boundary layer dissipation coefficient from Ref. [7]
        :param C_cs:                Discharge coefficient for shrouded blade from Ref. [7]
        :param C_cu:                Discharge coefficient for unshrouded blade (relative motion) from Ref. [11]
        :param delta_star_H:        Blade boundary layer displacement thickness on endwalls from Ref. [12]
        :param delta_star_theta:    Blade boundary layer shape factor in transonic turbines from Ref. [13]
        :param theta_t:             Blade boundary layer momentum thickness from Ref. [14]
        :param Kx:                  Axial tip clearance discharge coefficient from Ref. [8]
        :param Kr:                  Radial tip clearance discharge coefficient from Ref. [8]
        :param Kxr:                 Mixed radial-axial tip clearance discharge coefficient from Ref. [8]
        :param k:                   Friction loss factor from Ref. [6] tuned as per [15]
        """
        self.flow = flow
        self.Nstream = 15

        # Model parameters
        self.Cd = Cd
        self.C_cs = C_cs
        self.C_cu = C_cu
        self.delta_star_H = delta_star_H
        self.delta_star_theta = delta_star_theta
        self.theta_t = theta_t
        self.Kx = Kx
        self.Kr = Kr
        self.Kxr = Kxr
        self.k = k

        # Flow properties along blade surfaces
        self.Vss1 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Vps1 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Vss2 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Vps2 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Vss3 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Vps3 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Pss1 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Pps1 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Pss2 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Pps2 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Pss3 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Pps3 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Dss1 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Dps1 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Dss2 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Dps2 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Dss3 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Dps3 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Tss1 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Tps1 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Tss2 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Tps2 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Tss3 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Tps3 = np.ndarray((self.flow.Nslices, int(self.Nstream / 3)), float) * 0.0
        self.Vss = np.ndarray((3, self.flow.Nslices, self.Nstream), float) * 0.0
        self.Vps = np.ndarray((3, self.flow.Nslices, self.Nstream), float) * 0.0
        self.Dss = np.ndarray((3, self.flow.Nslices, self.Nstream), float) * 0.0
        self.Dps = np.ndarray((3, self.flow.Nslices, self.Nstream), float) * 0.0
        self.Pss = np.ndarray((3, self.flow.Nslices, self.Nstream), float) * 0.0
        self.Pps = np.ndarray((3, self.flow.Nslices, self.Nstream), float) * 0.0
        self.Tss = np.ndarray((3, self.flow.Nslices, self.Nstream), float) * 0.0
        self.Tps = np.ndarray((3, self.flow.Nslices, self.Nstream), float) * 0.0
        self.hss = np.ndarray((3, self.flow.Nslices, self.Nstream), float) * 0.0
        self.hps = np.ndarray((3, self.flow.Nslices, self.Nstream), float) * 0.0

        # Load dataset to evaluate base pressure through Sieverding correlation
        script_dir = os.path.dirname(__file__)
        self.xq_conv = np.load(script_dir + '/data/Sieverding_conv_xq.npy')
        self.yq_conv = np.load(script_dir + '/data/Sieverding_conv_yq.npy')
        self.zq_conv = np.load(script_dir + '/data/Sieverding_conv_zq.npy')
        self.xq_conv_div = np.load(script_dir + '/data/Sieverding_conv-div_xq.npy')
        self.yq_conv_div = np.load(script_dir + '/data/Sieverding_conv-div_yq.npy')
        self.zq_conv_div = np.load(script_dir + '/data/Sieverding_conv-div_zq.npy')

        # Loss coefficients
        self.ds_profile = np.zeros([3, self.flow.Nslices])
        self.ds_mixing = np.zeros([3, self.flow.Nslices])
        self.ds_shock = np.zeros([3, self.flow.Nslices])
        self.ds_secondary = np.zeros([3, self.flow.Nslices])
        self.ds_leakage = np.zeros([3, self.flow.Nslices])
        self.ds_incidence = np.zeros([3, self.flow.Nslices])
        self.ds_endwall = np.zeros([3, self.flow.Nslices])
        self.ds = np.zeros([3, self.flow.Nslices])

        self.flag_profile = np.zeros([3, self.flow.Nslices])
        self.flag_endwall = np.zeros([3, self.flow.Nslices])
        self.flag_incidence = np.zeros([3, self.flow.Nslices])
        self.flag_mixing = np.zeros([3, self.flow.Nslices])
        self.flag_shock = np.zeros([3, self.flow.Nslices])
        self.flag_leakage = np.zeros([3, self.flow.Nslices])

        self.zeta_profile = np.zeros(3)
        self.zeta_mixing = np.zeros(3)
        self.zeta_shock_ps = np.zeros(3)
        self.zeta_shock_ss = np.zeros(3)
        self.zeta_shock = np.zeros(3)
        self.zeta_secondary = np.zeros(3)
        self.zeta_leakage = np.zeros(3)
        self.zeta_incidence = np.zeros(3)
        self.zeta_endwall = np.zeros(3)
        self.zeta = np.zeros(3)

        # additional quantities
        self._Cpb = np.zeros([3, self.flow.Nslices])
        self.dh_windage = 0

        # condensation
        self.ds_cond = np.zeros([3, self.flow.Nslices])

    def ComputeTotalLosses(self, flow):
        self.ds_profile[self.ds_profile < 0] = 0.0
        self.ds_mixing[self.ds_mixing < 0] = 0.0
        self.ds_shock[self.ds_shock < 0] = 0.0
        self.ds_secondary[self.ds_secondary < 0] = 0.0
        self.ds_leakage[self.ds_leakage < 0] = 0.0
        self.ds_incidence[self.ds_incidence < 0] = 0.0
        self.ds_endwall[self.ds_endwall < 0] = 0.0
        self.ds = self.ds_profile + self.ds_mixing + self.ds_shock + self.ds_secondary + \
                  self.ds_leakage + self.ds_incidence + self.ds_endwall

        self.zeta_profile[0] = flow.massFlowWAv(self.ds_profile[0, :] * flow.T1_is / (flow.V1_is ** 2 / 2), 'stator')
        self.zeta_profile[1] = flow.massFlowWAv(self.ds_profile[1, :] * flow.T2_is / (flow.V2_is ** 2 / 2), 'vaneless')
        self.zeta_profile[2] = flow.massFlowWAv(self.ds_profile[2, :] * flow.T3_is / (flow.W3_is ** 2 / 2), 'rotor')
        self.zeta_mixing[0] = flow.massFlowWAv(self.ds_mixing[0, :] * flow.T1_is / (flow.V1_is ** 2 / 2), 'stator')
        self.zeta_mixing[1] = flow.massFlowWAv(self.ds_mixing[1, :] * flow.T2_is / (flow.V2_is ** 2 / 2), 'vaneless')
        self.zeta_mixing[2] = flow.massFlowWAv(self.ds_mixing[2, :] * flow.T3_is / (flow.W3_is ** 2 / 2), 'rotor')
        self.zeta_shock[0] = flow.massFlowWAv(self.ds_shock[0, :] * flow.T1_is / (flow.V1_is ** 2 / 2), 'stator')
        self.zeta_shock[1] = flow.massFlowWAv(self.ds_shock[1, :] * flow.T2_is / (flow.V2_is ** 2 / 2), 'vaneless')
        self.zeta_shock[2] = flow.massFlowWAv(self.ds_shock[2, :] * flow.T3_is / (flow.W3_is ** 2 / 2), 'rotor')
        self.zeta_secondary[0] = flow.massFlowWAv(self.ds_secondary[0, :] * flow.T1_is / (flow.V1_is ** 2 / 2), 'stator')
        self.zeta_secondary[1] = flow.massFlowWAv(self.ds_secondary[1, :] * flow.T2_is / (flow.V2_is ** 2 / 2), 'vaneless')
        self.zeta_secondary[2] = flow.massFlowWAv(self.ds_secondary[2, :] * flow.T3_is / (flow.W3_is ** 2 / 2), 'rotor')
        self.zeta_leakage[0] = flow.massFlowWAv(self.ds_leakage[0, :] * flow.T1_is / (flow.V1_is ** 2 / 2), 'stator')
        self.zeta_leakage[1] = flow.massFlowWAv(self.ds_leakage[1, :] * flow.T2_is / (flow.V2_is ** 2 / 2), 'vaneless')
        self.zeta_leakage[2] = flow.massFlowWAv(self.ds_leakage[2, :] * flow.T3_is / (flow.W3_is ** 2 / 2), 'rotor')
        self.zeta_incidence[0] = flow.massFlowWAv(self.ds_incidence[0, :] * flow.T1_is / (flow.V1_is ** 2 / 2), 'stator')
        self.zeta_incidence[1] = flow.massFlowWAv(self.ds_incidence[1, :] * flow.T2_is / (flow.V2_is ** 2 / 2), 'vaneless')
        self.zeta_incidence[2] = flow.massFlowWAv(self.ds_incidence[2, :] * flow.T3_is / (flow.W3_is ** 2 / 2), 'rotor')
        self.zeta_endwall[0] = flow.massFlowWAv(self.ds_endwall[0, :] * flow.T1_is / (flow.V1_is ** 2 / 2), 'stator')
        self.zeta_endwall[1] = flow.massFlowWAv(self.ds_endwall[1, :] * flow.T2_is / (flow.V2_is ** 2 / 2), 'vaneless')
        self.zeta_endwall[2] = flow.massFlowWAv(self.ds_endwall[2, :] * flow.T3_is / (flow.W3_is ** 2 / 2), 'rotor')
        self.zeta = self.zeta_profile + self.zeta_mixing + self.zeta_shock + \
                    self.zeta_secondary + self.zeta_leakage + self.zeta_incidence + self.zeta_endwall

        return

    # Blade Boundary Layers' Loss Model
    ###########################################################################################################
    def BBL_loop(self, p, *data):
        """Internal loop of BBL loss (rectangular V profile) used to match tangential momentum balance while considering
        flow compressibility"""
        delta_Vt, V_mean, Vax_mean, D_mean, Cax_s, delta_V, ht_in, s_in, den = data
        V_mean = p

        Vss = V_mean + delta_V
        Vps = V_mean - delta_V
        if Vps < 0:
            residual = 100
        else:
            hps = ht_in - Vps ** 2 / 2
            hss = ht_in - Vss ** 2 / 2
            self.flow.fluid.EoS.update(CoolProp.HmassSmass_INPUTS, hps, s_in)
            Pps = self.flow.fluid.EoS.p()
            self.flow.fluid.EoS.update(CoolProp.HmassSmass_INPUTS, hss, s_in)
            Pss = self.flow.fluid.EoS.p()
            residual = np.abs(D_mean * Vax_mean * delta_Vt - (Pps - Pss) * Cax_s) / den

        return residual

    def ComputeBoundaryLayerLosses_Denton(self, row, C_s, Cax_s, Cs_c, Vprofile, warning):
        "BBL loss for compressible flow reproducing a realistic velocity distribution along blade surfaces"

        # mean-line variables assignment
        if row == 'stator':
            flag_row = 0
            D_in = self.flow.D[0, :]
            D_out = self.flow.D[1, :]
            V_in = self.flow.V[0, :]
            V_out = self.flow.V[1, :]
            flow_angle_in = np.radians(self.flow.alpha[0, :])
            flow_angle_out = np.radians(self.flow.alpha[1, :])
            ht_in = self.flow.ht[0, :]
            s_in = self.flow.s[0, :]

        elif row == 'rotor':
            flag_row = 2
            D_in = self.flow.D[2, :]
            D_out = self.flow.D[3, :]
            V_in = self.flow.W[2, :]
            V_out = self.flow.W[3, :]
            flow_angle_in = np.radians(self.flow.blade_angle[2, :])
            flow_angle_out = np.radians(self.flow.blade_angle[3, :])
            ht_in = self.flow.htr[2, :]
            s_in = self.flow.s[2, :]

        delta_s_ps = np.zeros(self.flow.Nslices)
        delta_s_ss = np.zeros(self.flow.Nslices)

        Vt_in = V_in * np.sin(flow_angle_in)
        Vt_out = V_out * np.sin(flow_angle_out)
        Vax_in = V_in * np.cos(flow_angle_in)
        Vax_out = V_out * np.cos(flow_angle_out)

        V_mean = (V_in + V_out) / 2
        Vax_mean = (Vax_in + Vax_out) / 2
        D_mean = (D_in + D_out) / 2

        Cs_s = Cs_c * C_s
        delta_Vt = np.abs(Vt_out - Vt_in)  # generalization of control volume: R_out/R_in
        delta_V = delta_Vt / (2 * Cs_s)
        den = D_mean * Vax_mean * delta_Vt  # used for BBL_loop if Vprofile == 'simple'
        V_mean_incompr = Vax_mean * delta_Vt / (2.0 * Cax_s * delta_V)  # used for BBL_loop if Vprofile == 'simple'

        self.flow.k[flag_row, :] = 0.4 * np.ones(self.flow.Nslices)  # first guess
        x1 = np.linspace(0, 0.375, int(self.Nstream / 3))
        x2 = np.linspace(0.375, 0.625, int(self.Nstream / 3))
        x3 = np.linspace(0.625, 1.0, int(self.Nstream / 3))

        for ii in range(self.flow.Nslices):
            if Vprofile == 'simple':
                self.flow.k[flag_row, ii] = 100
                try:
                    data = (delta_Vt[ii], V_mean[ii], Vax_mean[ii], D_mean[ii], Cax_s, delta_V[ii], ht_in[ii],
                            s_in[ii], den[ii])
                    optim = opti.minimize(self.BBL_loop, V_mean_incompr[ii], args=data, tol=0.01)
                    V_mean[ii] = optim.x
                except:
                    V_mean[ii] = V_mean_incompr[ii]
                    self.flag_profile[flag_row, ii] = 1.0
                    if warning == 1:
                        print('Error in Blade Boundary Layer Loss Model: going back to Incompressible Loss Model')

                if (V_mean[ii] - delta_V[ii]) > 0:
                    self.Vps[flag_row, ii, :] = V_mean[ii] - delta_V[ii]
                else:
                    self.Vps[flag_row, ii, :] = 5  # impose a minimum value of V_PS
                    self.flag_profile[flag_row, ii] = 1.0

                self.Vss[flag_row, ii, :] = V_mean[ii] + delta_V[ii]
                self.hss[flag_row, ii, :] = ht_in[ii] - self.Vss[flag_row, ii, :] ** 2 / 2
                self.hps[flag_row, ii, :] = ht_in[ii] - self.Vps[flag_row, ii, :] ** 2 / 2

                self.flow.fluid.EoS.update(CoolProp.HmassSmass_INPUTS, self.hps[flag_row, ii, 0], s_in[ii])
                self.Pps[flag_row, ii, :] = self.flow.fluid.EoS.p()
                self.Dps[flag_row, ii, :] = self.flow.fluid.EoS.rhomass()
                self.Tps[flag_row, ii, :] = self.flow.fluid.EoS.T()

                self.flow.fluid.EoS.update(CoolProp.HmassSmass_INPUTS, self.hss[flag_row, ii, 0], s_in[ii])
                self.Pss[flag_row, ii, :] = self.flow.fluid.EoS.p()
                self.Dss[flag_row, ii, :] = self.flow.fluid.EoS.rhomass()
                self.Tss[flag_row, ii, :] = self.flow.fluid.EoS.T()
            else:
                res = 10
                toll = 0.05
                while abs(res) > toll:
                    if (self.flow.k[flag_row, ii] * V_in[ii] - delta_V[ii]) > (V_in[ii] / 5):
                        self.Vss1[ii, :] = V_in[ii] + ((2 * self.flow.k[flag_row, ii] * V_out[ii] + delta_V[ii]) -
                                                       V_in[ii]) * x1 / x1[-1]
                        self.Vps1[ii, :] = (self.flow.k[flag_row, ii] * V_in[ii] - delta_V[ii]) * x1 / x1[-1]
                        self.Vss2[ii, :] = (2 * self.flow.k[flag_row, ii] * V_out[ii] + delta_V[ii]) * np.ones(
                            len(x2))
                        self.Vps2[ii, :] = (self.flow.k[flag_row, ii] * V_in[ii] - delta_V[ii]) * np.ones(len(x2))
                        self.Vss3[ii, :] = (2 * self.flow.k[flag_row, ii] * V_out[ii] + delta_V[ii]) + \
                                           (V_out[ii] - (2 * self.flow.k[flag_row, ii] * V_out[ii] +
                                                         delta_V[ii])) * (x3 - x3[0]) / (x3[-1] - x3[0])
                        self.Vps3[ii, :] = (self.flow.k[flag_row, ii] * V_in[ii] - delta_V[ii]) + \
                                           (V_out[ii] - (self.flow.k[flag_row, ii] * V_in[ii] -
                                                         delta_V[ii])) * (x3 - x3[0]) / (x3[-1] - x3[0])
                    else:
                        self.Vss1[ii, :] = V_in[ii] + ((2 * self.flow.k[flag_row, ii] * V_out[ii] + delta_V[ii]) -
                                                       V_in[ii]) * x1 / x1[-1]
                        self.Vps1[ii, :] = (V_in[ii] / 5) * x1 / x1[-1]
                        self.Vss2[ii, :] = (2 * self.flow.k[flag_row, ii] * V_out[ii] + delta_V[ii]) * np.ones(
                            len(x2))
                        self.Vps2[ii, :] = (V_in[ii] / 5) * np.ones(len(x2))
                        self.Vss3[ii, :] = (2 * self.flow.k[flag_row, ii] * V_out[ii] + delta_V[ii]) + \
                                           (V_out[ii] - (2 * self.flow.k[flag_row, ii] * V_out[ii] +
                                                         delta_V[ii])) * (x3 - x3[0]) / (x3[-1] - x3[0])
                        self.Vps3[ii, :] = (V_in[ii] / 5) + (V_out[ii] - (V_in[ii] / 5)) * (x3 - x3[0]) / (
                                    x3[-1] - x3[0])

                    hss1 = ht_in[ii] - self.Vss1[ii, :] ** 2 / 2
                    hps1 = ht_in[ii] - self.Vps1[ii, :] ** 2 / 2
                    hss2 = ht_in[ii] - self.Vss2[ii, :] ** 2 / 2
                    hps2 = ht_in[ii] - self.Vps2[ii, :] ** 2 / 2
                    hss3 = ht_in[ii] - self.Vss3[ii, :] ** 2 / 2
                    hps3 = ht_in[ii] - self.Vps3[ii, :] ** 2 / 2

                    for jj in range(int(self.Nstream / 3)):
                        self.flow.fluid.EoS.update(CoolProp.HmassSmass_INPUTS, hss1[jj], s_in[ii])
                        self.Pss1[ii, jj] = self.flow.fluid.EoS.p()
                        self.flow.fluid.EoS.update(CoolProp.HmassSmass_INPUTS, hps1[jj], s_in[ii])
                        self.Pps1[ii, jj] = self.flow.fluid.EoS.p()
                        self.flow.fluid.EoS.update(CoolProp.HmassSmass_INPUTS, hss2[jj], s_in[ii])
                        self.Pss2[ii, jj] = self.flow.fluid.EoS.p()
                        self.flow.fluid.EoS.update(CoolProp.HmassSmass_INPUTS, hps2[jj], s_in[ii])
                        self.Pps2[ii, jj] = self.flow.fluid.EoS.p()
                        self.flow.fluid.EoS.update(CoolProp.HmassSmass_INPUTS, hss3[jj], s_in[ii])
                        self.Pss3[ii, jj] = self.flow.fluid.EoS.p()
                        self.flow.fluid.EoS.update(CoolProp.HmassSmass_INPUTS, hps3[jj], s_in[ii])
                        self.Pps3[ii, jj] = self.flow.fluid.EoS.p()

                    int1 = fld.integrate.trapz((self.Pps1[ii, :] - self.Pss1[ii, :]), x=x1)
                    int2 = fld.integrate.trapz((self.Pps2[ii, :] - self.Pss2[ii, :]), x=x2)
                    int3 = fld.integrate.trapz((self.Pps3[ii, :] - self.Pss3[ii, :]), x=x3)
                    P_int = int1 + int2 + int3

                    res_new = (D_mean[ii] * Vax_mean[ii] * delta_Vt[ii] - P_int * Cax_s) / \
                              (D_mean[ii] * Vax_mean[ii] * delta_Vt[ii])

                    if res_new > 0:
                        self.flow.k[flag_row, ii] += 0.01
                    else:
                        self.flow.k[flag_row, ii] -= 0.01

                    if abs(res) < abs(res_new):
                        self.flag_profile[flag_row, ii] = 1
                        break

                    res = res_new

            if Vprofile == 'simple':
                delta_s_ps[ii] = self.Cd * Cs_s[ii] * (self.Dps[flag_row, ii, 0] * self.Vps[flag_row, ii, 0] ** 3 /
                                                       self.Tps[flag_row, ii, 0]) / (D_mean[ii] * Vax_mean[ii])
                delta_s_ss[ii] = self.Cd * Cs_s[ii] * (self.Dss[flag_row, ii, 0] * self.Vss[flag_row, ii, 0] ** 3 /
                                                       self.Tss[flag_row, ii, 0]) / (D_mean[ii] * Vax_mean[ii])
            else:
                for jj in range(int(self.Nstream / 3)):
                    self.flow.fluid.EoS.update(CoolProp.PSmass_INPUTS, self.Pss1[ii, jj], s_in[ii])
                    self.Dss1[ii, jj] = self.flow.fluid.EoS.rhomass()
                    self.Tss1[ii, jj] = self.flow.fluid.EoS.T()
                for jj in range(int(self.Nstream / 3)):
                    self.flow.fluid.EoS.update(CoolProp.PSmass_INPUTS, self.Pps1[ii, jj], s_in[ii])
                    self.Dps1[ii, jj] = self.flow.fluid.EoS.rhomass()
                    self.Tps1[ii, jj] = self.flow.fluid.EoS.T()
                for jj in range(int(self.Nstream / 3)):
                    self.flow.fluid.EoS.update(CoolProp.PSmass_INPUTS, self.Pss2[ii, jj], s_in[ii])
                    self.Dss2[ii, jj] = self.flow.fluid.EoS.rhomass()
                    self.Tss2[ii, jj] = self.flow.fluid.EoS.T()
                for jj in range(int(self.Nstream / 3)):
                    self.flow.fluid.EoS.update(CoolProp.PSmass_INPUTS, self.Pps2[ii, jj], s_in[ii])
                    self.Dps2[ii, jj] = self.flow.fluid.EoS.rhomass()
                    self.Tps2[ii, jj] = self.flow.fluid.EoS.T()
                for jj in range(int(self.Nstream / 3)):
                    self.flow.fluid.EoS.update(CoolProp.PSmass_INPUTS, self.Pss3[ii, jj], s_in[ii])
                    self.Dss3[ii, jj] = self.flow.fluid.EoS.rhomass()
                    self.Tss3[ii, jj] = self.flow.fluid.EoS.T()
                for jj in range(int(self.Nstream / 3)):
                    self.flow.fluid.EoS.update(CoolProp.PSmass_INPUTS, self.Pps3[ii, jj], s_in[ii])
                    self.Dps3[ii, jj] = self.flow.fluid.EoS.rhomass()
                    self.Tps3[ii, jj] = self.flow.fluid.EoS.T()

                if (np.isinf(self.Dps1[ii, :])).any():
                    good = [e for e in self.Dps1[ii, :] if not (np.isinf(e))]
                    self.Dps1[ii, np.isinf(self.Dps1[ii, :])] = good[0]

                intss1 = fld.integrate.trapz((self.Dss1[ii, :] * self.Vss1[ii, :] ** 3 / self.Tss1[ii, :]), x=x1)
                intps1 = fld.integrate.trapz((self.Dps1[ii, :] * self.Vps1[ii, :] ** 3 / self.Tps1[ii, :]), x=x1)
                intss2 = fld.integrate.trapz((self.Dss2[ii, :] * self.Vss2[ii, :] ** 3 / self.Tss2[ii, :]), x=x2)
                intps2 = fld.integrate.trapz((self.Dps2[ii, :] * self.Vps2[ii, :] ** 3 / self.Tps2[ii, :]), x=x2)
                intss3 = fld.integrate.trapz((self.Dss3[ii, :] * self.Vss3[ii, :] ** 3 / self.Tss3[ii, :]), x=x3)
                intps3 = fld.integrate.trapz((self.Dps3[ii, :] * self.Vps3[ii, :] ** 3 / self.Tps3[ii, :]), x=x3)

                self.Vss[flag_row, ii, :] = np.concatenate((self.Vss1[ii, :], self.Vss2[ii, :], self.Vss3[ii, :]))
                self.Vps[flag_row, ii, :] = np.concatenate((self.Vps1[ii, :], self.Vps2[ii, :], self.Vps3[ii, :]))
                self.Pss[flag_row, ii, :] = np.concatenate((self.Pss1[ii, :], self.Pss2[ii, :], self.Pss3[ii, :]))
                self.Pps[flag_row, ii, :] = np.concatenate((self.Pps1[ii, :], self.Pps2[ii, :], self.Pps3[ii, :]))
                self.Dss[flag_row, ii, :] = np.concatenate((self.Dss1[ii, :], self.Dss2[ii, :], self.Dss3[ii, :]))
                self.Dps[flag_row, ii, :] = np.concatenate((self.Dps1[ii, :], self.Dps2[ii, :], self.Dps3[ii, :]))
                self.Tss[flag_row, ii, :] = np.concatenate((self.Dss1[ii, :], self.Dss2[ii, :], self.Dss3[ii, :]))
                self.Tps[flag_row, ii, :] = np.concatenate((self.Dps1[ii, :], self.Dps2[ii, :], self.Dps3[ii, :]))

                delta_s_ps[ii] = self.Cd * Cs_s[ii] * (intps1 + intps2 + intps3) / (D_mean[ii] * Vax_mean[ii])
                delta_s_ss[ii] = self.Cd * Cs_s[ii] * (intss1 + intss2 + intss3) / (D_mean[ii] * Vax_mean[ii])

            self.ds_profile[flag_row, ii] = delta_s_ps[ii] + delta_s_ss[ii]

        self.ds[flag_row,:] = self.ds[flag_row,:] + self.ds_profile[flag_row,:]

        return

    def ComputeBoundaryLayerLosses_VDB(self, row, C_s, Cs_c, t_s, z, R, **kwargs):
        "BBL loss for compressible flow reproducing a realistic velocity distribution along blade surfaces"

        # variables initialization
        delta_s_ps = np.zeros(self.flow.Nslices)
        delta_s_ss = np.zeros(self.flow.Nslices)
        P_int = np.zeros(self.flow.Nslices)
        delta_RVt = np.zeros(self.flow.Nslices)
        flow_angle = np.zeros([self.flow.Nslices, self.Nstream])
        V = np.zeros([self.flow.Nslices, self.Nstream])
        dflowangleds = np.zeros([self.flow.Nslices, self.Nstream])
        dtanflowangleds = np.zeros([self.flow.Nslices, self.Nstream])
        dRds = np.zeros([self.flow.Nslices, self.Nstream])
        dVaxRds = np.zeros([self.flow.Nslices, self.Nstream])
        delta_V = np.zeros([self.flow.Nslices, self.Nstream])

        # mean-line variables assignment
        if row == 'stator':
            flag_row = 0
            D_in = self.flow.D[0, :]
            D_out = self.flow.D[1, :]
            V_in = self.flow.V[0, :]
            V_out = self.flow.V[1, :]
            U_in = self.flow.U[0, :]
            flow_angle_in = np.radians(self.flow.blade_angle[0, :])
            flow_angle_out = np.radians(self.flow.blade_angle[1, :])
            Roth = self.flow.ht[0, :]
            s_in = self.flow.s[0, :]
            R_in = self.flow.R_ad[0, :]
            R_out = self.flow.R_ad[1, :]
            z_in = self.flow.z_ad[0, :]
            z_out = self.flow.z_ad[1, :]
            H_in = self.flow.H_Rm[0]
            H_out = self.flow.H_Rm[1]
            H_in_distr = np.concatenate((H_in / (2 * (self.flow.Nslices - 1)), [H_in / (self.flow.Nslices - 1)] *
                                         np.ones(self.flow.Nslices - 2), H_in / (2 * (self.flow.Nslices - 1))), axis=None)
            H_out_distr = np.concatenate((H_out / (2 * (self.flow.Nslices - 1)), [H_out / (self.flow.Nslices - 1)] *
                                         np.ones(self.flow.Nslices - 2), H_out / (2 * (self.flow.Nslices - 1))), axis=None)
            H_distr = np.zeros((self.flow.Nslices, self.Nstream))
            for ii in range(self.flow.Nslices):
                H_distr[ii] = np.ones((1, self.Nstream)) * H_in_distr[ii]
            Z = self.flow.Zs

        elif row == 'rotor':
            flag_row = 2
            D_in = self.flow.D[2, :]
            D_out = self.flow.D[3, :]
            V_in = self.flow.W[2, :]
            V_out = self.flow.W[3, :]
            U_in = self.flow.U[2, :]
            flow_angle_in = np.radians(self.flow.blade_angle[2, :])
            flow_angle_out = np.radians(self.flow.blade_angle[3, :])
            Roth = self.flow.Roth[2, :]
            s_in = self.flow.s[2, :]
            R_in = self.flow.R_ad[2, :]
            R_out = self.flow.R_ad[3, :]
            z_in = self.flow.z_ad[2, :]
            z_out = self.flow.z_ad[3, :]
            H_in = self.flow.H_Rm[2]
            H_out = self.flow.H_Rm[3]
            H_in_distr = np.concatenate((H_in / (2 * (self.flow.Nslices - 1)), [H_in / (self.flow.Nslices - 1)] *
                                         np.ones(self.flow.Nslices - 2), H_in / (2 * (self.flow.Nslices - 1))), axis=None)
            H_out_distr = np.concatenate((H_out / (2 * (self.flow.Nslices - 1)), [H_out / (self.flow.Nslices - 1)] *
                                         np.ones(self.flow.Nslices - 2), H_out / (2 * (self.flow.Nslices - 1))), axis=None)
            H_distr = self.bladeHeightDiscretization(z, R)
            Z = self.flow.Z

        # calculation of necessary quantities
        Vax_in = V_in * np.cos(flow_angle_in)
        Vax_out = V_out * np.cos(flow_angle_out)
        Vax_mean = (Vax_in + Vax_out) / 2
        Cs_s = Cs_c * C_s
        m_in = D_in * Vax_in * 2 * np.pi * H_in_distr * R_in
        m_out = D_out * Vax_out * 2 * np.pi * H_out_distr * R_out
        m = np.array([np.linspace(m_in[ii], m_out[ii], self.Nstream) for ii in range(self.flow.Nslices)])
        D = np.array([np.linspace(D_in[ii], D_out[ii], self.Nstream) for ii in range(self.flow.Nslices)])
        # arm for torque balance
        arm = (R_in + R_out) / 2
        # streamwise coordinate array
        s = [self.streamwiseCoordinate(z[ii, :], R[ii, :]) for ii in range(self.flow.Nslices)]
        # axial velocity distribution
        Vax = m / (D * 2 * np.pi * H_distr * R)

        # resolving tangential momentum balance for each blade slice span-wise
        res = 10 * np.ones(self.flow.Nslices)
        toll = 0.01
        k_gain = 1 * np.ones(self.flow.Nslices)
        iter = 1

        while np.any(abs(res) > toll) and iter < 20:
            for ii in range(self.flow.Nslices):

                # flow angle along the stream-wise direction
                (flow_angle[ii, :], _) = self.continuousBladeAngleDistribution([flow_angle_in[ii], R_in[ii]],
                                                                               [flow_angle_out[ii], R_out[ii]],
                                                                               self.Nstream)

                # calculation of derivatives
                dflowangleds[ii, :] = self.calculateDerivative(s[ii], flow_angle[ii, :])
                dtanflowangleds[ii, :] = self.calculateDerivative(s[ii], np.tan(flow_angle[ii, :]))
                dRds[ii, :] = self.calculateDerivative(s[ii], R[ii, :])
                dVaxRds[ii,:] = self.calculateDerivative(s[ii], (Vax[ii,:] * R[ii, :]))

                # calculation of delta_V
                delta_V[ii] = self.calculateDeltaVSSPS(row, Z, t_s, R[ii], flow_angle[ii], U_in[ii], Vax[ii],
                                                       dRds[ii], dtanflowangleds[ii], dVaxRds[ii],
                                                       dflowangleds[ii], k_gain[ii], flag_convergence_control=1)

                # calculation of meridional flow velocity distribution
                V[ii, :] = Vax[ii] / np.cos(flow_angle[ii, :])

                # calculation of pressure side and suction side velocity distributions
                self.Vss[flag_row,ii,:] = V[ii,:] + np.abs(delta_V[ii,:]) / 2
                self.Vps[flag_row,ii,:] = V[ii,:] - np.abs(delta_V[ii,:]) / 2
                for jj in range(self.Nstream):
                    if self.Vps[flag_row, ii, jj] < 5:
                        self.Vps[flag_row, ii, jj] = 5
                    self.flow.fluid.EoS.update(CoolProp.HmassSmass_INPUTS, self.flow.h[2,ii], s_in[ii])
                    if self.Vss[flag_row, ii, jj] > 1.2 * self.flow.fluid.EoS.speed_sound():
                        self.Vss[flag_row, ii, jj] = 1.2 * self.flow.fluid.EoS.speed_sound()
                    V[ii,jj] = np.mean([self.Vss[flag_row, ii, jj],self.Vps[flag_row, ii, jj]])

                # calculation of the static enthalpy distribution from the definition of Rothalpy
                self.hss[flag_row,ii,:] = Roth[ii] - self.Vss[flag_row,ii,:] ** 2 / 2 + (U_in[ii] * R[ii,:]) ** 2 / 2
                self.hps[flag_row,ii,:] = Roth[ii] - self.Vps[flag_row,ii,:] ** 2 / 2 + (U_in[ii] * R[ii,:]) ** 2 / 2

                # calculation of the pressure distribution using the equation of state
                for jj in range(self.Nstream):
                    self.flow.fluid.EoS.update(CoolProp.HmassSmass_INPUTS, self.hps[flag_row, ii, jj], s_in[ii])
                    self.Pps[flag_row, ii, jj] = self.flow.fluid.EoS.p()
                    self.Dps[flag_row, ii, jj] = self.flow.fluid.EoS.rhomass()
                    self.Tps[flag_row, ii, jj] = self.flow.fluid.EoS.T()
                    self.flow.fluid.EoS.update(CoolProp.HmassSmass_INPUTS, self.hss[flag_row, ii, jj], s_in[ii])
                    self.Pss[flag_row, ii, jj] = self.flow.fluid.EoS.p()
                    self.Dss[flag_row, ii, jj] = self.flow.fluid.EoS.rhomass()
                    self.Tss[flag_row, ii, jj] = self.flow.fluid.EoS.T()

                # integration of pressure distribution
                P_int[ii] = integrate.simpson((self.Pps[flag_row, ii, :] - self.Pss[flag_row, ii, :]) *
                                              H_distr[ii, :], s[ii])

                # mass flow conservation
                Vax[ii,:] = m[ii] / ((self.Dps[flag_row,ii,:] + self.Dss[flag_row,ii,:]) * np.pi * H_distr[ii,:] * R[ii,:])

                # recalculate the angular momentum difference
                delta_RVt[ii] = (Vax[ii, self.Nstream - 1] * np.tan(flow_angle[ii, self.Nstream - 1])
                                 + U_in[ii] * R_out[ii] / R_in[ii]) * R_out[ii] \
                                - (Vax[ii, 0] * np.tan(flow_angle[ii, 0]) + U_in[ii]) * R_in[ii]

                # calculation of the entropy generation
                delta_s_ps[ii] = self.Cd * Cs_s[ii] * integrate.simpson(self.Dps[flag_row,ii,:] *
                                                                     self.Vps[flag_row,ii,:] ** 3 /
                                                                     self.Tps[flag_row,ii,:], s[ii]) / \
                                 (np.mean([D_in[ii], D_out[ii]]) * Vax_mean[ii])
                delta_s_ss[ii] = self.Cd * Cs_s[ii] * integrate.simpson(self.Dss[flag_row,ii,:] *
                                                                     self.Vss[flag_row,ii,:] ** 3 /
                                                                     self.Tss[flag_row,ii,:], s[ii]) / \
                                 (np.mean([D_in[ii], D_out[ii]]) * Vax_mean[ii])
                self.ds_profile[flag_row, ii] = delta_s_ps[ii] + delta_s_ss[ii]

            # calculating the residual of the tangential momentum equation

            res_new = np.array([(m_in[ii] * abs(delta_RVt[ii]) - Z * P_int[ii] * arm[ii]) / (m_in[ii] * abs(delta_RVt[ii]))
                                for ii in range(self.flow.Nslices)])

            for ii in range(self.flow.Nslices):
                if res_new[ii] < 0 and abs(res_new[ii]) > toll:
                    if abs(res_new[ii]) > 0.5: k_gain[ii] -= 0.1
                    if abs(res_new[ii]) > 0.2 and abs(res_new[ii]) < 0.5: k_gain[ii] -= 0.05
                    if abs(res_new[ii]) < 0.2: k_gain[ii] -= 0.025
                elif res_new[ii] > 0 and abs(res_new[ii]) > toll:
                    if abs(res_new[ii]) > 0.5: k_gain[ii] += 0.1
                    if abs(res_new[ii]) > 0.2 and abs(res_new[ii]) < 0.5: k_gain[ii] += 0.05
                    if abs(res_new[ii]) < 0.2: k_gain[ii] += 0.025

            res = res_new
            iter += 1

        if kwargs.get('verbosity'):
            print('Total iterations rotor BL loss model: ' + str(iter))
        # check convergence for simulation report
        for ii in range(self.flow.Nslices):
            if abs(res[ii]) > toll or iter > 25:
                self.flag_profile[flag_row, ii] = 1

        self.ds[flag_row,:] = self.ds[flag_row,:] + self.ds_profile[flag_row,:]

        return

    def ComputeBoundaryLayerLosses_VDB_v2(self, row, C_s, Cs_c, t_s, z, R):
        "BBL loss for compressible flow reproducing a realistic velocity distribution along blade surfaces"

        # variables initialization
        H_distr = np.zeros((self.flow.Nslices, self.Nstream))
        s = np.zeros((self.flow.Nslices, self.Nstream))

        # mean-line variables assignment
        if row == 'stator':
            flag_row = 0
            D_in = self.flow.D[0, :]
            D_out = self.flow.D[1, :]
            V_in = self.flow.V[0, :]
            V_out = self.flow.V[1, :]
            U_in = self.flow.U[0, :]
            flow_angle_in = np.radians(self.flow.blade_angle[0, :])
            flow_angle_out = np.radians(self.flow.blade_angle[1, :])
            Roth = self.flow.ht[0, :]
            s_in = self.flow.s[0, :]
            R_in = self.flow.R_ad[0, :]
            R_out = self.flow.R_ad[1, :]
            z_in = self.flow.z_ad[0, :]
            z_out = self.flow.z_ad[1, :]
            H_in = self.flow.H_Rm[0]
            H_out = self.flow.H_Rm[1]
            H_in_distr = np.concatenate((H_in / (2 * (self.flow.Nslices - 1)), [H_in / (self.flow.Nslices - 1)] *
                                         np.ones(self.flow.Nslices - 2), H_in / (2 * (self.flow.Nslices - 1))), axis=None)
            H_out_distr = np.concatenate((H_out / (2 * (self.flow.Nslices - 1)), [H_out / (self.flow.Nslices - 1)] *
                                         np.ones(self.flow.Nslices - 2), H_out / (2 * (self.flow.Nslices - 1))), axis=None)
            for ii in range(self.flow.Nslices):
                H_distr[ii] = np.ones((1, self.Nstream)) * H_in_distr[ii]
            Z = self.flow.Zs

        elif row == 'rotor':
            flag_row = 2
            D_in = self.flow.D[2, :]
            D_out = self.flow.D[3, :]
            V_in = self.flow.W[2, :]
            V_out = self.flow.W[3, :]
            U_in = self.flow.U[2, :]
            flow_angle_in = np.radians(self.flow.blade_angle[2, :])
            flow_angle_out = np.radians(self.flow.blade_angle[3, :])
            Roth = self.flow.Roth[2, :]
            s_in = self.flow.s[2, :]
            R_in = self.flow.R_ad[2, :]
            R_out = self.flow.R_ad[3, :]
            z_in = self.flow.z_ad[2, :]
            z_out = self.flow.z_ad[3, :]
            H_in = self.flow.H_Rm[2]
            H_out = self.flow.H_Rm[3]
            H_in_distr = np.concatenate((H_in / (2 * (self.flow.Nslices - 1)), [H_in / (self.flow.Nslices - 1)] *
                                         np.ones(self.flow.Nslices - 2), H_in / (2 * (self.flow.Nslices - 1))), axis=None)
            H_out_distr = np.concatenate((H_out / (2 * (self.flow.Nslices - 1)), [H_out / (self.flow.Nslices - 1)] *
                                         np.ones(self.flow.Nslices - 2), H_out / (2 * (self.flow.Nslices - 1))), axis=None)
            H_distr = self.bladeHeightDiscretization(z, R)
            Z = self.flow.Z

        # calculation of necessary quantities
        Vax_in = V_in * np.cos(flow_angle_in)
        Vax_out = V_out * np.cos(flow_angle_out)
        Vax_mean = (Vax_in + Vax_out) / 2
        Cs_s = Cs_c * C_s
        m_in = D_in * Vax_in * 2 * np.pi * H_in_distr * R_in
        m_out = D_out * Vax_out * 2 * np.pi * H_out_distr * R_out
        m = np.array([np.linspace(m_in[ii], m_out[ii], self.Nstream) for ii in range(self.flow.Nslices)])
        D = np.array([np.linspace(D_in[ii], D_out[ii], self.Nstream) for ii in range(self.flow.Nslices)])
        # arm for torque balance
        arm = (R_in + R_out) / 2
        # streamwise coordinate array
        s = [self.streamwiseCoordinate(z[ii, :], R[ii, :]) for ii in range(self.flow.Nslices)]
        # axial velocity distribution
        Vax = m / (D * 2 * np.pi * H_distr * R)


        for ii in range(self.flow.Nslices):
            data = (C_s[ii], Cs_s[ii], Cs_c[ii], t_s, z[ii, :], R[ii, :], row, flag_row, D_in[ii], D_out[ii], V_in[ii], V_out[ii], U_in[ii],
                    flow_angle_in[ii], flow_angle_out[ii], Roth[ii], s_in[ii], R_in[ii], R_out[ii], z_in[ii], z_out[ii],
                    H_in, H_out, H_in_distr[ii], H_out_distr[ii], H_distr[ii, :], Z, arm[ii], s[ii], Vax[ii, :],
                    m[ii, :], m_in[ii], Vax_mean[ii], ii)

            output = opti.minimize(self.angularMomentumBalance_VDB, [0.2, 0.7], args=data, method='Nelder-Mead')

        self.ds[flag_row,:] = self.ds[flag_row,:] + self.ds_profile[flag_row,:]

        return

    def angularMomentumBalance_VDB(self, p, *data):

        # initialize quantities
        flow_angle = np.zeros([self.Nstream])
        V = np.zeros([self.Nstream])
        dflowangleds = np.zeros([self.Nstream])
        dtanflowangleds = np.zeros([self.Nstream])
        dRds = np.zeros([self.Nstream])
        dVaxRds = np.zeros([self.Nstream])
        delta_V = np.zeros([self.Nstream])

        C_s, Cs_s, Cs_c, t_s, z, R, row, flag_row, D_in, D_out, V_in, V_out, U_in, flow_angle_in, flow_angle_out, \
        Roth, s_in, R_in, R_out, z_in, z_out, H_in, H_out, H_in_distr, H_out_distr, H_distr, Z, arm, \
        s, Vax, m, m_in, Vax_mean, span = data

        # flow angle along the stream-wise direction
        (flow_angle[:], _) = self.continuousBladeAngleDistribution_opti([flow_angle_in, R_in], [flow_angle_out, R_out],
                                                                        p[0], p[1], 0, 0, self.Nstream)

        # calculation of derivatives
        dflowangleds[:] = self.calculateDerivative(s, flow_angle[:])
        dtanflowangleds[:] = self.calculateDerivative(s, np.tan(flow_angle[:]))
        dRds[:] = self.calculateDerivative(s, R[:])
        dVaxRds[:] = self.calculateDerivative(s, (Vax[:] * R[:]))

        # calculation of delta_V
        delta_V = self.calculateDeltaVSSPS(row, Z, t_s, R, flow_angle, U_in, Vax, dRds, dtanflowangleds, dVaxRds,
                                           dflowangleds, 1, flag_convergence_control=0)

        # calculation of meridional flow velocity distribution
        V[:] = Vax / np.cos(flow_angle[:])

        # calculation of pressure side and suction side velocity distributions
        self.Vss[flag_row, span, :] = V[:] + np.abs(delta_V[:]) / 2
        self.Vps[flag_row, span, :] = V[:] - np.abs(delta_V[:]) / 2
        for jj in range(self.Nstream):
            if self.Vps[flag_row, span, jj] < 5:
                self.Vps[flag_row, span, jj] = 5
            self.flow.fluid.EoS.update(CoolProp.HmassSmass_INPUTS, self.flow.h[2,span], s_in)
            if self.Vss[flag_row, span, jj] > 1.2 * self.flow.fluid.EoS.speed_sound():
                self.Vss[flag_row, span, jj] = 1.2 * self.flow.fluid.EoS.speed_sound()
            V[jj] = np.mean([self.Vss[flag_row, span, jj],self.Vps[flag_row, span, jj]])

        # calculation of the static enthalpy distribution from the definition of Rothalpy
        self.hss[flag_row, span, :] = Roth - self.Vss[flag_row, span, :] ** 2 / 2 + (U_in * R[:]) ** 2 / 2
        self.hps[flag_row, span, :] = Roth - self.Vps[flag_row, span, :] ** 2 / 2 + (U_in * R[:]) ** 2 / 2

        # calculation of the pressure distribution using the equation of state
        for jj in range(self.Nstream):
            self.flow.fluid.EoS.update(CoolProp.HmassSmass_INPUTS, self.hps[flag_row, span, jj], s_in)
            self.Pps[flag_row, span, jj] = self.flow.fluid.EoS.p()
            self.Dps[flag_row, span, jj] = self.flow.fluid.EoS.rhomass()
            self.Tps[flag_row, span, jj] = self.flow.fluid.EoS.T()
            self.flow.fluid.EoS.update(CoolProp.HmassSmass_INPUTS, self.hss[flag_row, span, jj], s_in)
            self.Pss[flag_row, span, jj] = self.flow.fluid.EoS.p()
            self.Dss[flag_row, span, jj] = self.flow.fluid.EoS.rhomass()
            self.Tss[flag_row, span, jj] = self.flow.fluid.EoS.T()

        # integration of pressure distribution
        DPhR_int = integrate.simpson((self.Pps[flag_row, span, :] - self.Pss[flag_row, span, :]) *
                                      H_distr[:] * R[:], s)

        # mass flow conservation
        Vax = m / ((self.Dps[flag_row, span, :] + self.Dss[flag_row, span, :]) * np.pi * H_distr[:] * R[:])

        # recalculate the angular momentum difference
        delta_RVt = (Vax[self.Nstream - 1] * np.tan(flow_angle[self.Nstream - 1])
                         + U_in * R_out / R_in) * R_out \
                        - (Vax[0] * np.tan(flow_angle[0]) + U_in) * R_in

        # calculation of the entropy generation
        delta_s_ps = self.Cd * Cs_s * integrate.simpson(self.Dps[flag_row, span, :] *
                                                             self.Vps[flag_row, span, :] ** 3 /
                                                             self.Tps[flag_row, span, :], s) / \
                         (np.mean([D_in, D_out]) * Vax_mean)
        delta_s_ss = self.Cd * Cs_s * integrate.simpson(self.Dss[flag_row, span, :] *
                                                             self.Vss[flag_row, span, :] ** 3 /
                                                             self.Tss[flag_row, span, :], s) / \
                         (np.mean([D_in, D_out]) * Vax_mean)

        self.ds_profile[flag_row, span] = delta_s_ps + delta_s_ss

        # calculating the residual of the tangential momentum equation

        res = (m_in * abs(delta_RVt) - Z * DPhR_int) / (m_in * abs(delta_RVt))

        return abs(res)

    def setMeridionalCoordinates(self, P0, P3, n_points):
        """ Creates 3rd order Bezier spline between a starting point (P0) and end point (P3) """

        t = fld.np.linspace(0, 1, n_points)
        P1 = fld.np.ndarray(fld.np.size(P0))
        P2 = fld.np.ndarray(fld.np.size(P0))

        P1[0] = P0[0] + P3[0] * 0.05
        P2[0] = P3[0] * 0.5
        P1[1] = (P0[1] + P3[1]) * 0.5
        P2[1] = P3[1] + (P0[1] - P3[1]) * 0.0

        Bz = P0[0] * (1 - t[:]) ** 3 + P1[0] * 3 * t[:] * (1 - t[:]) ** 2 + P2[0] * 3 * t[:] ** 2 * (1 - t[:]) + \
             P3[0] * t[:] ** 3
        Br = P0[1] * (1 - t[:]) ** 3 + P1[1] * 3 * t[:] * (1 - t[:]) ** 2 + P2[1] * 3 * t[:] ** 2 * (1 - t[:]) + \
             P3[1] * t[:] ** 3

        return Bz, Br

    def streamwiseCoordinate(self, z, R):
        """Function that takes 2 arrays representing the axial and radial coordinate in the meridional plane as input
        and calculates the stream-wise coordinate as output."""
        ds = [np.sqrt((R[jj + 1] - R[jj]) ** 2 + (z[jj + 1] - z[jj]) ** 2) for jj in range(self.Nstream - 1)]
        s = np.zeros(self.Nstream)
        for jj in range(1, self.Nstream): s[jj] = s[jj - 1] + ds[jj - 1]
        return s

    def parabolicBladeAngleDistribution(self, flow_angle_in, flow_angle_out, ind_var):
        """ Function that takes the inlet and outlet blade angles as input and calculates the parabolic distribution
        between the two former sections as a function of a specified independent variable """

        # calculation of parabolic distribution coefficients
        b = (flow_angle_out - flow_angle_in) / (ind_var[len(ind_var)-1] - ind_var[len(ind_var)-1] ** 2 / 2 / \
                                                ind_var[0] - ind_var[0] / 2)
        a = -b / 2 / ind_var[0]
        c = flow_angle_in - b * ind_var[0] / 2
        # calculation of angle distribution
        flow_angle = a * ind_var ** 2 + b * ind_var + c
        return flow_angle

    def polynomialBladeAngleDistribution(self, flow_angle_in, flow_angle_out, ind_var):
        """ Function that takes the inlet and outlet blade angles as input and calculates the second order polynomial
        distribution between the two former sections as a function of a specified independent variable

        flow_angle = A * ind_var ** 2 + B * ind_var + C

        Boundary conditions:
        flow_angle[ind_var[0]] = flow_angle_in
        flow_angle[ind_var[-1]] = flow_angle_out
        d_flow_angle/d_ind_var [ind_var[mid]] = 0

        """

        # calculation of polynomial coefficients
        index_mid = 6
        C = flow_angle_in
        B = (- 2 * (flow_angle_out - flow_angle_in) * ind_var[index_mid] / ind_var[-1] ** 2) / \
            (1 - 2 * ind_var[index_mid] / ind_var[-1])
        A = ((flow_angle_out - flow_angle_in) - B * ind_var[-1]) / ind_var[-1] ** 2

        # calculation of angle distribution
        flow_angle = A * ind_var ** 2 + B * ind_var + C
        return flow_angle

    def continuousBladeAngleDistribution(self, P0, P3, n_points):
        """ Creates 3rd order Bezier spline between a starting point (P0) and end point (P3) """

        t = fld.np.linspace(0, 1, n_points)
        P1 = fld.np.ndarray(fld.np.size(P0))
        P2 = fld.np.ndarray(fld.np.size(P0))

        P1[0] = P0[0] + P3[0] * 0.2
        P2[0] = P3[0] * 0.7
        P1[1] = P0[1]
        P2[1] = P3[1] + (P0[1] - P3[1]) * 0.0

        Bz = P0[0] * (1 - t[:]) ** 3 + P1[0] * 3 * t[:] * (1 - t[:]) ** 2 + P2[0] * 3 * t[:] ** 2 * (1 - t[:]) + \
             P3[0] * t[:] ** 3
        Br = P0[1] * (1 - t[:]) ** 3 + P1[1] * 3 * t[:] * (1 - t[:]) ** 2 + P2[1] * 3 * t[:] ** 2 * (1 - t[:]) + \
             P3[1] * t[:] ** 3

        return Bz, Br

    def continuousBladeAngleDistribution_opti(self, P0, P3, a, b, c, d, n_points):
        """ Creates 3rd order Bezier spline between a starting point (P0) and end point (P3) """

        t = fld.np.linspace(0, 1, n_points)
        P1 = fld.np.ndarray(fld.np.size(P0))
        P2 = fld.np.ndarray(fld.np.size(P0))

        P1[0] = P0[0] + P3[0] * a
        P2[0] = P3[0] * b
        P1[1] = P0[1] + c
        P2[1] = P3[1] + d

        Bz = P0[0] * (1 - t[:]) ** 3 + P1[0] * 3 * t[:] * (1 - t[:]) ** 2 + P2[0] * 3 * t[:] ** 2 * (1 - t[:]) + \
             P3[0] * t[:] ** 3
        Br = P0[1] * (1 - t[:]) ** 3 + P1[1] * 3 * t[:] * (1 - t[:]) ** 2 + P2[1] * 3 * t[:] ** 2 * (1 - t[:]) + \
             P3[1] * t[:] ** 3

        return Bz, Br

    def calculateDerivative(self, x, y):
        dx = np.zeros(len(x)-1)
        dy = np.zeros(len(x)-1)
        for ii in range(1,len(x)):
            dx[ii - 1] = x[ii] - x[ii - 1]
            dy[ii - 1] = y[ii] - y[ii - 1]
        dydx = np.concatenate((0, dy/dx), axis=None)
        return dydx

    def calculateDeltaVSSPS(self, row, Z, t_s, R, flow_angle, U_in, Vax, dRds, dtanflowangleds, dVaxRds, dflowangleds,
                            k_gain, flag_convergence_control=0):
        "Calculates the velocity difference between the SS and PS as proposed by R. Van Den Braembussche [Ref. 3]"

        "[*]: Resource available at: https://learning-oreilly-com.tudelft.idm.oclc.org/library/view/design-and-analysis/9781119424093/"

        "The model needs the following parameters as input:" \
        "Z:                 number of blades" \
        "t_s:               blade thickness to pitch ratio" \
        "R:                 adimnsional radius in the blade channel" \
        "flow_angle:        flow angle distribution in the blade channel" \
        "U_in:              inlet peripheral velocity" \
        "Vax:               meridional velocity distribution in the blade channel" \
        "dRds:              first derivative of the adimensional radius distribution in the blade channel" \
        "dtanflowangleds:   first derivative of the flow angle distribution in the blade channel" \
        "dVaxds:            first derivative of the meridional flow velocity distribution in the blade channel" \
        "k_gain:            scaling parameter for convergence control"

        if row == 'rotor':
            delta_V = np.zeros(self.Nstream)
            delta_V[:] = 2 * np.pi / Z * (1 - t_s / R[:] / np.cos(flow_angle[:])) * \
                          (2 * U_in * R[:] * dRds[:]
                           - Vax[:] * R[:] * dtanflowangleds[:]
                           - np.tan(flow_angle[:]) * dVaxRds[:])
            delta_V[len(delta_V)-1] = 0
            delta_V[0] = 0
            if flag_convergence_control == 1:
                delta_V = delta_V * k_gain
        elif row == 'stator':
            delta_V = np.zeros(self.Nstream)
            delta_V[:] = (2 * np.pi * R[:] / Z - t_s / np.cos(flow_angle[:])) * Vax[:] / np.cos(flow_angle[:]) * \
                         dflowangleds[:]
            delta_V[len(delta_V)-1] = 0
            delta_V[0] = 0
            if flag_convergence_control == 1:
                delta_V = delta_V * k_gain
        return delta_V

    def bladeHeightDiscretization(self, z, R):
        "Function that computes the blade height distribution given the meridional plane coordinates as input"
        H = np.array([np.sqrt((z[self.flow.Nslices - 1, jj] - z[0, jj]) ** 2 + (R[self.flow.Nslices - 1, jj] - R[0, jj]) ** 2)
             for jj in range(self.Nstream)])
        H_distr = np.array([np.concatenate((H[jj] / (2 * (self.flow.Nslices - 1)),
                                            [H[jj] / (self.flow.Nslices - 1)] * np.ones(self.flow.Nslices - 2),
                                            H[jj] / (2 * (self.flow.Nslices - 1))), axis=None)
                            for jj in range(self.Nstream)]).transpose()
        return H_distr

    def bladeLoading(self, row, solidity):
        """ This function calculates the blade loading according to a simplified model of incompressible flow
        [Ref. 7]
         Assumptions:
         - Incompressible flow
         - No external entropy generation sources: T*Ds = 0 --> Dh = 1/rho * DP
         - Velocity circulation is proportional to blade loading (net pressure force on the blade) """

        if row == 'stator':
            flag_row = 0
            index_in = 0
            index_out = 1
            V_out_is = self.flow.V1_is
            DVtR = np.abs(self.flow.Vt[1] - self.flow.Vt[0] * self.flow.R_ad[0] / self.flow.R_ad[1])
            Vm_ave = [np.mean([self.flow.Vm[0,ii], self.flow.Vm[1,ii]]) for ii in range(self.flow.Nslices)]
        elif row == 'rotor':
            flag_row = 2
            index_in = 2
            index_out = 3
            V_out_is = self.flow.W3_is
            DVtR = np.abs(self.flow.Wt[3] * self.flow.R_ad[3] / self.flow.R_ad[2] - self.flow.Wt[2])
            Vm_ave = [np.mean([self.flow.Wm[2, ii], self.flow.Wm[3, ii]]) for ii in range(self.flow.Nslices)]

        Dh_PS = np.array(self.flow.massFlowWAv(solidity ** -1 * Vm_ave * DVtR, row))

        # loading factor
        LF = 2 * Dh_PS / (V_out_is ** 2)

        return LF

    # Empirical and semi-empirical passage loss models
    ###########################################################################################################
    def rotorFrictionLoss_RG(self, t_s, Re=1e5):
        """ This function calculates the rotor friction loss by an analogy with a flow in a curved, rotating pipe,

        Inputs: inlet thickness to pitch ratio

        Hp: trailing edge thickness to pitch ratio recommended to be smaller than 2%

        Source: Rodgers [Ref. 2] """

        # midchannel streamline meridional coordinates
        [z, R] = self.setMeridionalCoordinates([self.flow.z_ad[2, self.flow.Nmid], self.flow.R_ad[2, self.flow.Nmid]],
                                               [self.flow.z_ad[3, self.flow.Nmid], self.flow.R_ad[3, self.flow.Nmid]],
                                               self.Nstream)

        # midchannel streamline axial streamwise coordinate
        sx = self.streamwiseCoordinate(z, R)

        # flow angle along the stream-wise direction
        metal_angle_distr = self.parabolicBladeAngleDistribution(
            self.flow.blade_angle[2, self.flow.Nmid], self.flow.blade_angle[3, self.flow.Nmid], R)

        # midchannel streamline streamwise coordinate --> sx / cos(metal_angle_distr)
        s = sx / np.cos(np.radians(metal_angle_distr))

        # blade length
        L = s[-1]

        # calculate the average velocity between the inlet and outlet of the passage
        Wmean = np.mean([self.flow.massFlowWAv(self.flow.W[2], 2), self.flow.massFlowWAv(self.flow.W[3], 3)])

        # calculate the inlet and outlet dynamic viscosities
        self.flow.fluid.EoS.update(CoolProp.DmassP_INPUTS,
                                   self.flow.D[2, self.flow.Nmid],
                                   self.flow.P[2, self.flow.Nmid])
        nu_in = self.flow.fluid.EoS.viscosity() / self.flow.D[2, self.flow.Nmid]
        self.flow.fluid.EoS.update(CoolProp.DmassP_INPUTS,
                                   self.flow.D[3, self.flow.Nmid],
                                   self.flow.P[3, self.flow.Nmid])
        nu_out = self.flow.fluid.EoS.viscosity() / self.flow.D[3, self.flow.Nmid]

        # calculate the friction coefficient for smooth walls
        Cf = 0.05 / (Re ** 0.2)

        # inlet hydraulic diameter
        pitch_in_hub = 2 * np.pi * self.flow.R_ad[2, 0] / self.flow.Z
        pitch_in_tip = 2 * np.pi * self.flow.R_ad[2, -1] / self.flow.Z
        area_in = (pitch_in_hub + pitch_in_tip) * (1 - t_s) / 2 * self.flow.H_Rm[2]
        perim_in = (pitch_in_hub + pitch_in_tip) * (1 - t_s) + 2 * self.flow.H_Rm[2]
        Dh_in = 4 * area_in / perim_in

        # outlet hydraulic diameter
        pitch_out_hub = 2 * np.pi * self.flow.R_ad[3, 0] / self.flow.Z
        pitch_out_tip = 2 * np.pi * self.flow.R_ad[3, -1] / self.flow.Z
        area_out = (pitch_out_hub + pitch_out_tip) * (1 - t_s) / 2 * self.flow.H_Rm[3]
        perim_out = (pitch_out_hub + pitch_out_tip) * (1 - t_s) + 2 * self.flow.H_Rm[3]
        Dh_out = 4 * area_out / perim_out

        # average hydraulic diameter
        Dh = (Dh_in + Dh_out) / 2

        # curvature factor
        Kr = (self.flow.H_Rm[2] + self.flow.H_Rm[3]) \
             / (self.flow.R_ad[2, self.flow.Nmid] - self.flow.R_ad[3, self.flow.Nmid])

        # calculation of the enthalpy loss
        Deltah = (2 * Cf * L / Dh + Kr) * (Wmean) ** 2 / 9.81

        # Span-wise distribution of the enthalpy
        mass_distr = [(np.mean([self.flow.Vm[3, ii], self.flow.Vm[3, ii + 1]]) *
                       np.mean([self.flow.D[3, ii], self.flow.D[3, ii + 1]]) *
                       (self.flow.R_ad[3, ii + 1] - self.flow.R_ad[3, ii]) *
                       np.mean([self.flow.R_ad[3, ii], self.flow.R_ad[3, ii + 1]])) /\
                      self.flow.m_red for ii in range(self.flow.Nslices - 1)]

        Deltah_span = np.ndarray(self.flow.Nslices, float)

        Deltah_span[0] = Deltah * mass_distr[0] / 2
        Deltah_span[1] = Deltah * (mass_distr[0] / 2 + mass_distr[1] / 2)
        Deltah_span[2] = Deltah * (mass_distr[1] / 2 + mass_distr[2] / 2)
        Deltah_span[3] = Deltah * (mass_distr[2] / 2 + mass_distr[3] / 2)
        Deltah_span[4] = Deltah * mass_distr[3] / 2

        # determination of outlet thermodynamic conditions and calculation of the entropy generation
        ht_out = Deltah + self.flow.massFlowWAv(self.flow.ht3_is, 3)
        self.flow.fluid.EoS.update(CoolProp.HmassP_INPUTS, ht_out, self.flow.massFlowWAv(self.flow.Pt[3], 3))
        s_out = self.flow.fluid.EoS.smass()
        self.ds_profile[2] = s_out - self.flow.massFlowWAv(self.flow.s[2], 2)

    def bladeLoadingLosses_RG(self):
        """ This function calculates the rotor blade loading loss according to a simplified model by Rodgers [Ref.2] """

        ht_out_is = self.flow.ht3_is

        sigma = np.ndarray(self.flow.Nslices, float)

        # calculate blade solidity
        for ii in range(self.flow.Nslices):
            # midchannel streamline meridional coordinates
            [z, R] = self.setMeridionalCoordinates(
                [self.flow.z_ad[2, ii], self.flow.R_ad[2, ii]],
                [self.flow.z_ad[3, ii], self.flow.R_ad[3, ii]],
                self.Nstream)

            # midchannel streamline axial streamwise coordinate
            sx = self.streamwiseCoordinate(z, R)

            # flow angle along the stream-wise direction
            metal_angle_distr = self.parabolicBladeAngleDistribution(
                self.flow.blade_angle[2, ii], self.flow.blade_angle[3, ii], R)

            # midchannel streamline streamwise coordinate --> sx / cos(metal_angle_distr)
            s = sx / np.cos(np.radians(metal_angle_distr))

            # blade length and solidity
            L = s[-1]
            sigma[ii] = self.flow.Z * L / 2 / self.flow.R_ad[2, ii]

        Deltah = sigma ** -1 * self.flow.Vt[2] ** 2

        # determination of outlet thermodynamic conditions and calculation of the entropy generation
        ht_out = Deltah + ht_out_is
        for ii in range(self.flow.Nslices):
            self.flow.fluid.EoS.update(CoolProp.HmassP_INPUTS, ht_out[ii], self.flow.Pt[1, ii])
            s_out = self.flow.fluid.EoS.smass()
            self.ds_profile[2, ii] = self.ds_profile[1, ii] + s_out - self.flow.s[2, ii]

    def bladeProfileLosses_Glassman(self, row, t_s, t_te, solidity):
        """ Blade loading loss model based on boundary layer theory developed by Glassman [Ref. 1] """

        # initialize quantities
        if row == 'stator':
            flag_row = 0
            ref_station = 1
            gPv_in = deepcopy(self.flow.gamma[0, :])
            gPv_out = (self.flow.gamma[1, :])
            Tt_in = deepcopy(self.flow.Tt[0, :])
            ht_in = deepcopy(self.flow.ht[0, :])
            V_out_is = deepcopy(self.flow.V1_is)
            angle_out = deepcopy(np.radians(self.flow.alpha[1]))
            P_out = deepcopy(self.flow.P[1, :])
            U_out = deepcopy(self.flow.U[1, :]) * 0.0
            s_out = deepcopy(self.flow.s[1, :])
            D_out = deepcopy(self.flow.D[1, :])
        elif row == 'rotor':
            flag_row = 2
            ref_station = 3
            gPv_in = deepcopy(self.flow.gamma[2, :])
            gPv_out = deepcopy(self.flow.gamma[3, :])
            Tt_in = deepcopy(self.flow.Ttr[2, :])
            ht_in = deepcopy(self.flow.Roth[2, :])
            V_out_is = deepcopy(self.flow.W3_is)
            angle_out = deepcopy(np.radians(self.flow.beta[3, :]))
            P_out = deepcopy(self.flow.P[3, :])
            U_out = deepcopy(self.flow.U[3, :])
            s_out = deepcopy(self.flow.s[3, :])
            D_out = deepcopy(self.flow.D[3, :])

        Re_theta = np.zeros(self.flow.Nslices)
        theta_c = np.zeros(self.flow.Nslices)
        theta = self.theta_t * t_te
        for ii in range(self.flow.Nslices):
            self.flow.fluid.EoS.update(CoolProp.PSmass_INPUTS, P_out[ii], s_out[ii])
            mu_out = self.flow.fluid.EoS.viscosity()
            Re_theta[ii] = D_out[ii] * V_out_is[ii] * theta / mu_out
            theta_c[ii] = 0.001 * (Re_theta[ii] / 500) ** (-0.2)

        theta_s = theta_c * solidity

        # calculate loss iteratively for each section span-wise
        for ii in range(self.flow.Nslices):

            # while loop control
            tol = 1e-3
            res = 10
            iter = 0
            V_out = V_out_is[ii] # initialization of V_out with ideal flow speed

            while res > tol:
                # calculate the average gamma_Pv along the expansion
                gPv = (gPv_in[ii] + gPv_out[ii]) / 2

                # calculate the critical stagnation velocity
                V_cr = np.sqrt(2 * gPv / (gPv + 1) * self.flow.R * Tt_in[ii])

                # calculate the Q parameter for the stator
                Q = (gPv - 1) / (gPv + 1) * (V_out / V_cr) ** 2

                # determine the E (energy factor) and H (form factor) parameters for the BL
                E, H = self.calculatePrustHermanParameters(Q)

                # calculate the internal flow loss for a 2D section in the blade-to-blade plane
                e_2D = (E * theta_s[ii]) / (np.cos(angle_out[ii]) - t_s - H * theta_s[ii])

                # determine the real, outlet flow conditions after losses
                V_out = np.sqrt(1 - e_2D) * V_out_is[ii]
                h_out = ht_in[ii] - V_out ** 2 / 2 + U_out[ii] ** 2 / 2
                self.flow.fluid.EoS.update(CoolProp.CoolProp.HmassP_INPUTS, h_out, P_out[ii])
                D_out = self.flow.fluid.EoS.rhomass()
                try:
                    gPv_out[ii] = self.flow.fluid.EoS.gamma_pv()
                except:
                    dP_dv_T = (- 1 / (D_out ** 2) *
                               self.flow.fluid.EoS.first_partial_deriv(CoolProp.iDmass, CoolProp.iP, CoolProp.iT)) ** (-1)
                    gPv_out[ii] = - 1 / (P_out[ii] * D_out) * self.flow.fluid.EoS.cpmass() / \
                                           self.flow.fluid.EoS.cvmass() * dP_dv_T
                s_out_old = s_out[ii]
                s_out[ii] = self.flow.fluid.EoS.smass()
                res = (s_out[ii] - s_out_old) / s_out_old
                iter += 1

            self.ds_profile[flag_row, ii] = s_out[ii] - self.flow.s[ref_station, ii]
            if self.ds_profile[flag_row, ii] < 0:
                self.ds_profile[flag_row, ii] = 0

        return

    def bladeProfileLosses_GlassmanModified(self, row, t_s, t_te, solidity):
        """ Blade loading loss model based on boundary layer theory developed by Glassman [Ref. 1] """

        if row == 'stator':
            flag_row = 0
            gPv_in = deepcopy(self.flow.gamma[0, :])
            gPv_out = deepcopy(self.flow.gamma[1, :])
            Tt_in = deepcopy(self.flow.Tt[0, :])
            ht_in = deepcopy(self.flow.ht[0, :])
            V_out_is = deepcopy(self.flow.V1_is)
            angle_out = deepcopy(np.radians(self.flow.alpha[1]))
            P_out = deepcopy(self.flow.P[1, :])
            U_out = deepcopy(self.flow.U[1, :]) * 0.0
            s_out = deepcopy(self.flow.s[1, :])
            D_out = deepcopy(self.flow.D[1, :])
            WAR_out = deepcopy(self.flow.WAR[1, :])
        elif row == 'rotor':
            flag_row = 2
            gPv_in = deepcopy(self.flow.gamma[2, :])
            gPv_out = (self.flow.gamma[3, :])
            Tt_in = deepcopy(self.flow.Ttr[2, :])
            ht_in = deepcopy(self.flow.Roth[2, :])
            V_out_is = deepcopy(self.flow.W3_is)
            angle_out = deepcopy(np.radians(self.flow.beta[3, :]))
            P_out = deepcopy(self.flow.P[3, :])
            U_out = deepcopy(self.flow.U[3, :])
            s_out = deepcopy(self.flow.s[3, :])
            D_out = deepcopy(self.flow.D[3, :])

        # calculate loading factor
        LF = self.bladeLoading(row, solidity)

        # calculate theta_c based on Reynolds number
        Re_theta = np.zeros(self.flow.Nslices)
        theta_c = np.zeros(self.flow.Nslices)
        theta = self.theta_t * t_te

        for ii in range(self.flow.Nslices):
            self.flow.fluid.EoS.update(CoolProp.PSmass_INPUTS, P_out[ii], s_out[ii])
            mu_out = self.flow.fluid.EoS.viscosity()
            Re_theta[ii] = D_out[ii] * V_out_is[ii] * theta / mu_out
            theta_c[ii] = 0.001 * (Re_theta[ii] / 500) ** (-0.2)

        theta_s = theta_c * solidity

        # calculate loss iteratively for each section span-wise
        for ii in range(self.flow.Nslices):

            # while loop control
            tol = 1e-6
            res = 10
            iter = 0
            V_out = V_out_is[ii] # initialization of V_out with ideal flow speed
            s_old = s_out[ii]

            while res > tol:
                # calculate the average gamma_Pv along the expansion
                gPv = (gPv_in[ii] + gPv_out[ii]) / 2

                # calculate the critical stagnation velocity
                V_cr = np.sqrt(2 * gPv / (gPv + 1) * self.flow.R * Tt_in[ii])

                # calculate the Q parameter for the stator
                Q = (gPv - 1) / (gPv + 1) * (V_out / V_cr) ** 2

                # determine the E (energy factor) and H (form factor) parameters for the BL
                E, H = self.calculatePrustHermanParameters(Q)

                # calculate the internal flow loss for a 2D section in the blade-to-blade plane
                e_2D = (E * theta_s[ii]) / (np.cos(angle_out[ii]) - t_s - H * theta_s[ii]) * (1 + LF[ii])

                # determine the real, outlet flow conditions after losses
                V_out = np.sqrt(1 - e_2D) * V_out_is[ii]
                h_out = ht_in[ii] - V_out ** 2 / 2 + U_out[ii] ** 2 / 2

                if self.flow._condensation:
                    self.flow.fluid.EoS.set_humidity('W', WAR_out[ii])

                self.flow.fluid.EoS.update(CoolProp.CoolProp.HmassP_INPUTS, h_out, P_out[ii])
                D_out[ii] = self.flow.fluid.EoS.rhomass()
                s = self.flow.fluid.EoS.smass()

                try:
                    gPv_out[ii] = self.flow.fluid.EoS.gamma_pv()
                except:
                    dP_dv_T = (- 1 / (D_out[ii] ** 2) *
                               self.flow.fluid.EoS.first_partial_deriv(CoolProp.iDmass, CoolProp.iP, CoolProp.iT)) ** (-1)
                    gPv_out[ii] = - 1 / (P_out[ii] * D_out[ii]) * self.flow.fluid.EoS.cpmass() / \
                                           self.flow.fluid.EoS.cvmass() * dP_dv_T

                res = abs(s - s_old) / s_old
                s_old = s
                iter += 1

            self.ds_profile[flag_row, ii] = s - s_out[ii]
            if self.ds_profile[flag_row, ii] < 0:
                self.ds_profile[flag_row, ii] = 0

            # ensure non-negative entropy generation before accumulating
            if self.ds_profile[flag_row, ii] >= - 0.01:
                self.ds[flag_row, ii] = self.ds[flag_row, ii] + self.ds_profile[flag_row, ii]
            else:
                self.ds_profile[flag_row, ii] = 0.0

        return

    def calculatePrustHermanParameters(self, Q):

        den = 1 / 1.68 + Q / 2.88 + Q ** 2 / 4.4 + Q ** 3 / 6.24
        # Energy factor
        numE = 2 * (1 / 1.92 + Q / 3.2 + Q ** 2 / 4.8 + Q ** 3 / 6.72)
        E = numE / den
        # Form factor
        numH = 1 / 1.2 + 3 * Q / 1.6 + 5 * Q ** 2 / 2 + 7 * Q ** 3 / 2.4 + 9 * Q ** 4 / 2.8
        H = numH / den

        return E, H

    def rotorIncidenceLosses_NASA(self):
        """ Rotor incidence loss model from [Ref 5] """

        # incidence is determined as the difference between the blade angle and the optimal flow angle for slip
        i = np.radians(self.flow.beta2_opt - self.flow.blade_angle[2, self.flow.Nmid])

        if i <= 0: n = 2.5
        else: n = 1.75

        Deltah = self.flow.W[2, self.flow.Nmid] ** 2 / 2 * (1 - np.cos(i) ** n)

        # determination of outlet thermodynamic conditions and calculation of the entropy generation
        h_out = self.flow.massFlowWeighting(Deltah, row='rotor', split=1) + self.flow.h3_is
        for ii in range(self.flow.Nslices):

            if self.flow._condensation:
                self.flow.fluid.EoS.set_humidity('W', self.flow.WAR[3, ii])

            self.flow.fluid.EoS.update(CoolProp.HmassP_INPUTS, h_out[ii], self.flow.P[3, ii])
            s_out = self.flow.fluid.EoS.smass()

            self.ds_incidence[2, ii] = s_out - self.flow.s[2, ii]

            # ensure non-negative entropy generation before accumulating
            if self.ds_incidence[2, ii] >= - 0.01:
                self.ds[2, ii] = self.ds[2, ii] + self.ds_incidence[2, ii]
            else:
                self.ds_incidence[2, ii] = 0.0

        return

    def rotorIncidenceLosses_Baines(self):
        """ Rotor incidence loss model from [Ref 5] """

        if np.sin(np.radians(self.flow.beta[2, self.flow.Nmid])) * np.sin(
                np.radians(self.flow.blade_angle[2, self.flow.Nmid])) > 0:
            k = -1
        elif np.sin(np.radians(self.flow.beta[2, self.flow.Nmid])) * np.sin(
                np.radians(self.flow.blade_angle[2, self.flow.Nmid])) <= 0:
            k = 1

        Deltah = self.flow.W[2, self.flow.Nmid] ** 2 / 2 * \
                 np.abs(np.sin(np.radians(self.flow.beta[2, self.flow.Nmid])) ** 2 +
                        k * np.sin(np.radians(self.flow.blade_angle[2, self.flow.Nmid])))

        # determination of outlet thermodynamic conditions and calculation of the entropy generation
        h_out = Deltah + self.flow.h3_is
        for ii in range(self.flow.Nslices):
            self.flow.fluid.EoS.update(CoolProp.HmassP_INPUTS, h_out[ii], self.flow.P[3, ii])
            s_out = self.flow.fluid.EoS.smass()
            self.ds_incidence[2, ii] = s_out - self.flow.s[2, ii]

            # ensure non-negative entropy generation before accumulating
            if self.ds_incidence[2, ii] >= - 0.01:
                self.ds[2, ii] = self.ds[2, ii] + self.ds_incidence[2, ii]
            else:
                self.ds_incidence[2, ii] = 0.0

        return

    def rotorPassageLoss_Baines(self):
        """ Passage loss model as per [Ref. 8] accounting for friction, blade loading and secondary flow loss """

        # initialize quantities
        R2m = self.flow.R_ad[2, self.flow.Nmid]
        R3m = self.flow.R_ad[3, self.flow.Nmid]
        R3h = self.flow.R_ad[3, 0]
        R3s = self.flow.R_ad[3, -1]
        z2h = self.flow.z_ad[2, 0]
        z3m = self.flow.z_ad[3, self.flow.Nmid]
        beta2g = np.radians(self.flow.blade_angle[2, self.flow.Nmid])
        beta3g = np.radians(self.flow.blade_angle[3, self.flow.Nmid])
        H2 = self.flow.H_Rm[2]
        H3 = self.flow.H_Rm[3]
        Zb = self.flow.Z
        W2_ave = self.flow.massFlowWAv(self.flow.W[2, :], 2)
        W3_ave = self.flow.massFlowWAv(self.flow.W[3, :], 3)

        # setting Kp coefficient
        if (R2m - R3s) / H3 >= 0.2:
            Kp = 1
        elif (R2m - R3s) / H3 < 0.2:
            Kp = 2

        # blade chord according to definition in [Ref. 8]
        c = (z3m - z2h) / np.cos(np.arctan(np.mean([np.tan(beta2g), np.tan(beta3g)])))

        # Hydraulic length according to definition in [Ref. 8]
        Lh = np.pi / 4 * ((z3m - z2h - H2 / 2) + (R2m - R3s - H3 / 2))

        # Hydraulic diameter according to definition in [Ref. 8] - corrected denominator first addendum
        # Dh = 0.5 * ((2 * np.pi * R2m * H2 / Zb) / (2 * (H2 + 2 * np.pi * R2m / Zb)) +
        #             (np.pi * (R3s ** 2 - R3h ** 2) / Zb) / (2 * np.pi / Zb * (R3s + R3h) + 2 * H3))
        Dh = 0.5 * ((4 * np.pi * R2m * H2) / (Zb * H2 + 2 * np.pi * R2m) +
                    (2 * np.pi * (R3s ** 2 - R3h ** 2)) / (np.pi * (R3s - R3h) + Zb * H3))

        # calculate loss coefficient
        Lp = Kp * 0.1 * (Lh / Dh + 0.68 * (1 - (R3m / R2m) ** 2) * np.cos(beta3g) / H3 * c) * \
             0.5 * (W2_ave ** 2 + W3_ave ** 2)

        # determination of outlet thermodynamic conditions and calculation of the entropy generation
        h_out = self.flow.massFlowWeighting(Lp, row='rotor', split=1) + self.flow.h3_is
        for ii in range(self.flow.Nslices):

            if self.flow._condensation:
                self.flow.fluid.EoS.set_humidity('W', self.flow.WAR[3, ii])

            self.flow.fluid.EoS.update(CoolProp.HmassP_INPUTS, h_out[ii], self.flow.P[3, ii])
            s_out = self.flow.fluid.EoS.smass()

            self.ds_profile[2, ii] = s_out - self.flow.s[2, ii]

            # ensure non-negative entropy generation before accumulating
            if self.ds_profile[2, ii] >= - 0.01:
                self.ds[2, ii] = self.ds[2, ii] + self.ds_profile[2, ii]
            else:
                self.ds_profile[2, ii] = 0.0

        return

    def calculateSupersonicFlowConditions(self, p, *data):
        """ Compute the real flow conditions based on specified Mach number and initial conditions """
        P = p
        (ht_in, s_in, Ma_tg) = data

        try:
            self.flow.fluid.EoS.update(CoolProp.CoolProp.PSmass_INPUTS, P, s_in)
            a = self.flow.fluid.EoS.speed_sound()
            h = self.flow.fluid.EoS.hmass()
            Ma = fld.np.sqrt(2 * (ht_in - h)) / a
            res = (Ma - Ma_tg) / Ma_tg
        except ValueError:
            res = 1

        return res

    # Mixing Loss Models todo: implement condensation check and thermo props update
    ###########################################################################################################
    def ComputeMixingLosses(self, row, t_s, warning, RR, blade_parameter=2):    # what is blade parameter?
        "Mixing loss due to wake and BBL mixing"
        if row == 'stator':
            flag_row = 0
            angle_out = np.radians(self.flow.alpha[1, :])
            P_out = self.flow.P[1, :]
            D_out = self.flow.D[1, :]
            V_out = self.flow.V[1, :]
            M_out = self.flow.MachAbs[1, :]
            s_in = self.flow.s[0, :]
            Pt_in = self.flow.Pt[0, :]
            ht_in = self.flow.ht[0, :]
            U = self.flow.U[1, :] * 0.0
            s_a = self.flow.s[0, :] + self.ds_profile[0, :] + self.ds_secondary[0, :] + self.ds_incidence[0, :] + self.ds_leakage[0, :]
            Rmx_Rout = RR * np.ones([self.flow.Nslices])
            
        elif row == 'rotor':
            flag_row = 2
            angle_out = np.radians(self.flow.beta[3, :])
            P_out = self.flow.P[3, :]
            D_out = self.flow.D[3, :]
            V_out = self.flow.W[3, :]
            M_out = self.flow.MachRel[3, :]
            Pt_in = self.flow.Ptr[2, :]
            ht_in = self.flow.Roth[2, :]
            U = self.flow.U[3, :]
            s_a = self.flow.s[2, :] + self.ds_profile[2, :] + self.ds_secondary[2, :] + self.ds_incidence[2, :] + self.ds_leakage[2, :]
            Rmx_Rout = RR * np.ones(self.flow.Nslices)

        for ii in range(self.flow.Nslices):

            theta_s = self.theta_t * t_s
            delta_star_s = self.delta_star_theta * theta_s

            try:
                if (row == 'stator' and (M_out[ii] >= 0.95 and M_out[ii] <= 1.05)) or (row == 'rotor' and M_out[ii] >= 0.95):

                    # Purely convergent blade channel
                    Pb = Pt_in[ii] * w_extrap(P_out[ii] / Pt_in[ii], blade_parameter, self.xq_conv, self.yq_conv,
                                              self.zq_conv)

                    # sonic loop to compute field quantities at throat section
                    data = (ht_in[ii], s_a[ii], U[ii])
                    P_a = opti.fsolve(self.sonicLoop, P_out[ii], args=data, full_output=False, xtol=1.0e-06)
                    self.flow.post_expansion[flag_row, ii] = P_a / P_out[ii]
                    self.flow.fluid.EoS.update(CoolProp.PSmass_INPUTS, P_a, s_a[ii])
                    V_a = self.flow.fluid.EoS.speed_sound()
                    D_a = self.flow.fluid.EoS.rhomass()
                    h_a = self.flow.fluid.EoS.hmass()
                    self.flow._Vstar_Dstar[flag_row, ii] = V_a * D_a
                    self.flow._Vstar[flag_row, ii] = V_a
                    self.flow._Dstar[flag_row, ii] = D_a
                    self.flow._Pstar[flag_row, ii] = P_a

                    # loop to compute s_out
                    errV = 10
                    iter_max = 50
                    iter = 0

                    while (errV > 1.0e-9) and (iter < iter_max):

                        # energy balance
                        h_out = h_a + V_a ** 2 / 2 - V_out[ii] ** 2 / 2
                        self.flow.fluid.EoS.update(CoolProp.HmassP_INPUTS, h_out, P_out[ii])
                        D_out[ii] = self.flow.fluid.EoS.rhomass()

                        # mass balance
                        angle_a = np.arccos(t_s + delta_star_s + (D_out[ii] * V_out[ii] * np.cos(angle_out[ii])) /
                                            (D_a * V_a) * Rmx_Rout[ii])

                        if np.isnan(angle_a):
                            raise ValueError

                        # calculate deviation
                        if (angle_out[ii] * angle_a) == - np.abs(angle_out[ii] * angle_a):
                            angle_a = - angle_a

                        if np.sign(angle_out[ii]) == - 1:
                            self.flow.deviation[flag_row, ii] = np.degrees(angle_out[ii] - angle_a)
                        else:
                            self.flow.deviation[flag_row, ii] = np.degrees(angle_a - angle_out[ii])

                        if self.flow.deviation[flag_row, ii] < 0:
                            self.flow.deviation[flag_row, ii] = 0

                        # axial momentum balance
                        V_out_new = (D_a * V_a ** 2 * (np.cos(angle_a) - t_s - delta_star_s - theta_s) +
                                     P_a * (np.cos(angle_a) - t_s) + Pb * t_s - P_out[ii] * np.cos(angle_a)
                                     * Rmx_Rout[ii]) / \
                                    (D_a * V_a * (np.cos(angle_a) - t_s - delta_star_s) *
                                     np.cos(np.radians(self.flow.deviation[flag_row, ii])))

                        errV = np.abs(V_out_new - V_out[ii]) / V_out[ii]
                        V_out[ii] = V_out_new

                        iter += 1

                    self.flow.fluid.EoS.update(CoolProp.HmassP_INPUTS, h_out, P_out[ii])
                    s_out = self.flow.fluid.EoS.smass()
                    self.ds_mixing[flag_row, ii] = s_out - s_a[ii]

                    self.ds[flag_row, ii] = self.ds[flag_row, ii] + self.ds_mixing[flag_row, ii]

                elif M_out[ii] > 1.05:
                    # Convergent-divergent blade channel - only stator
                    Pb = Pt_in[ii] * w_extrap(P_out[ii] / Pt_in[ii], blade_parameter, self.xq_conv_div,
                                              self.yq_conv_div, self.zq_conv_div) / 10

                    s_e = s_a  # profile loss imposed from throat to exit
                    s_a = s_in  # isentropic flow until throat
                    data = (ht_in[ii], s_a[ii])
                    P_a = opti.fsolve(self.sonicLoop, P_out[ii], args=data, full_output=False, xtol=1.0e-06)
                    self.flow.fluid.EoS.update(CoolProp.PSmass_INPUTS, P_a, s_a[ii])

                    # setting optimal post expansion ratio according to Anand 2020 [Ref. 4]
                    self.flow.post_expansion[flag_row, ii] = 0.9964 + 0.0596 * self.flow.gamma_id
                    P_e = P_out[ii] * self.flow.post_expansion[flag_row, ii]

                    self.flow.fluid.EoS.update(CoolProp.PSmass_INPUTS, P_e, s_e[ii])
                    D_e = self.flow.fluid.EoS.rhomass()
                    h_e = self.flow.fluid.EoS.hmass()
                    V_e = np.sqrt(2 * (ht_in[ii] - h_e))
                    self.flow._Vstar_Dstar[flag_row, ii] = V_e * D_e
                    self.flow._Vstar[flag_row, ii] = V_e
                    self.flow._Dstar[flag_row, ii] = D_e
                    self.flow._Pstar[flag_row, ii] = P_e

                    # loop to compute s_out
                    errV = 10
                    iter_max = 50
                    iter = 0

                    while (errV > 1.0e-9) and (iter < iter_max):

                        # energy balance
                        h_out = h_e + V_e ** 2 / 2 - V_out[ii] ** 2 / 2
                        self.flow.fluid.EoS.update(CoolProp.HmassP_INPUTS, h_out, P_out[ii])
                        D_out[ii] = self.flow.fluid.EoS.rhomass()

                        # mass balance
                        angle_e = np.arccos(t_s + delta_star_s + (D_out[ii] * V_out[ii] * np.cos(angle_out[ii])) /
                                            (D_e * V_e) * Rmx_Rout[ii])

                        if np.isnan(angle_e):
                            raise ValueError

                        # calculate deviation
                        if (angle_out[ii] * angle_e) == - np.abs(angle_out[ii] * angle_e):
                            angle_e = - angle_e

                        if np.sign(angle_out[ii]) == - 1:
                            self.flow.deviation[flag_row, ii] = np.degrees(angle_out[ii] - angle_e)
                        else:
                            self.flow.deviation[flag_row, ii] = np.degrees(angle_e - angle_out[ii])

                        if self.flow.deviation[flag_row, ii] < 0:
                            self.flow.deviation[flag_row, ii] = 0

                        # calculate deviation
                        self.flow.deviation[flag_row, ii] = np.degrees(angle_out[ii] - angle_e)

                        # axial momentum balance - radial control volume approach with linear pitch
                        V_out_new = (D_e * V_e ** 2 * (np.cos(angle_e) - t_s - delta_star_s - theta_s) +
                                     P_e * (np.cos(angle_e) - t_s) + Pb * t_s - P_out[ii] * np.cos(angle_e)
                                     * Rmx_Rout[ii]) / \
                                    (D_e * V_e * (np.cos(angle_e) - t_s - delta_star_s) *
                                     np.cos(np.radians(self.flow.deviation[flag_row, ii])))

                        errV = np.abs(V_out_new - V_out[ii]) / V_out[ii]
                        V_out[ii] = V_out_new

                        iter += 1

                    self.flow.fluid.EoS.update(CoolProp.HmassP_INPUTS, h_out, P_out[ii])
                    s_out = self.flow.fluid.EoS.smass()
                    self.ds_mixing[flag_row, ii] = s_out - s_e[ii]

                    self.ds[flag_row, ii] = self.ds[flag_row, ii] + self.ds_mixing[flag_row, ii]

                else:
                    # Purely convergent blade channel
                    self.flow.deviation[flag_row, ii] = 0.0
            except:
                # Purely convergent blade channel
                if warning == 1:
                    print('Control Volume Loop failed in Mixing Loss Model: going back to Simplified '
                          'Denton\'s Model')

                self.flow.deviation[flag_row, ii] = 0.0

    def ComputeTrailingEdgeWakeMixingLoss(self, *data, flag_slice=0, blade_parameter=2):
        if flag_slice == 1:

            Pt_in, P_out, blade_parameter, flag_row, g_Pv_out, \
            M_out, t_s, angle_out, delta_star_s, T_out_is, V_out_is, \
            V_out, D_out, ii = data[0]

            Pb = Pt_in * w_extrap(P_out / Pt_in, blade_parameter, self.xq_conv, self.yq_conv,
                                      self.zq_conv)

            # Cpb = 2 * (Pb - P_out) / (D_out * V_out ** 2)
            Cpb = 2 * (Pb / P_out - 1) * (2 / (g_Pv_out * M_out ** 2) * t_s) # From Baumgartner et al. (2020)

            # Hp: deviation = 0 --> angle_a = angle_out
            t_a = t_s / np.cos(angle_out)
            theta_a = self.theta_t * t_a
            delta_star_a = delta_star_s / np.cos(angle_out)

            self.ds_mixing[flag_row, ii] = (- Cpb * t_a + 2 * theta_a + (delta_star_a + t_a) ** 2) / \
                                           T_out_is * (V_out_is ** 2 / 2)

            # mass flux product at the throat
            self.flow._Vstar_Dstar[flag_row, ii] = (V_out * D_out) / (np.cos(angle_out) - t_s)
            self._Cpb[flag_row, ii] = Cpb

            # ensure non-negative entropy generation before accumulating
            if self.ds_mixing[flag_row, ii] >= - 0.01:
                self.ds[flag_row, ii] = self.ds[flag_row, ii] + self.ds_mixing[flag_row, ii]
            else:
                self.ds_mixing[flag_row, ii] = 0.0
        else:
            row, t_s = data
            if row == 'stator':
                flag_row = 0
                angle_out = np.radians(self.flow.blade_angle[1, :])
                P_out = self.flow.P[1, :]
                D_out = self.flow.D[1, :]
                V_out = self.flow.V[1, :]
                M_out = self.flow.MachAbs[1, :]
                Pt_in = self.flow.Pt[0, :]
                T_out_is = self.flow.T1_is
                V_out_is = self.flow.V1_is
                g_Pv_out = self.flow.gamma[1, :]
            elif row == 'rotor':
                flag_row = 2
                angle_out = np.radians(self.flow.blade_angle[3, :])
                P_out = self.flow.P[3, :]
                D_out = self.flow.D[3, :]
                V_out = self.flow.W[3, :]
                M_out = self.flow.MachRel[3, :]
                Pt_in = self.flow.Ptr[2, :]
                T_out_is = self.flow.T3_is
                V_out_is = self.flow.W3_is
                g_Pv_out = self.flow.gamma[3, :]

            for ii in range(self.flow.Nslices):

                theta_s = self.theta_t * t_s
                delta_star_s = self.delta_star_theta * theta_s
                Pb = Pt_in[ii] * w_extrap(P_out[ii] / Pt_in[ii], blade_parameter, self.xq_conv, self.yq_conv,
                                          self.zq_conv)

                Cpb = 2 * (Pb / P_out[ii] - 1) * (2 / (g_Pv_out[ii] * M_out[ii] ** 2) * t_s) # From Baumgartner et al. (2020)

                # Hp: deviation = 0 --> angle_a = angle_out
                t_a = t_s / np.cos(angle_out[ii])
                theta_a = self.theta_t * t_a
                delta_star_a = delta_star_s / np.cos(angle_out[ii])

                self.ds_mixing[flag_row, ii] = (- Cpb * t_a + 2 * theta_a + (delta_star_a + t_a) ** 2) / \
                                               T_out_is[ii] * (V_out_is[ii] ** 2 / 2) + self.ds_mixing[flag_row, ii]

                # mass flux product at the throat
                self.flow._Vstar_Dstar[flag_row, ii] = (V_out[ii] * D_out[ii]) / (np.cos(angle_out[ii]) - t_s)
                self._Cpb[flag_row, ii] = Cpb

                # ensure non-negative entropy generation before accumulating
                if self.ds_mixing[flag_row, ii] >= - 0.01:
                    self.ds[flag_row, ii] = self.ds[flag_row, ii] + self.ds_mixing[flag_row, ii]
                else:
                    self.ds_mixing[flag_row, ii] = 0.0

    def sonicLoop(self, p, *data):
        "Compute sonic conditions in the throat section"
        ht_in, s_a, U = data
        try:
            len(p)
            P_a = p[0]
        except:
            P_a = p

        # sonic conditions
        self.flow.fluid.EoS.update(CoolProp.PSmass_INPUTS, P_a, s_a)
        h_a = self.flow.fluid.EoS.hmass()
        c_a = self.flow.fluid.EoS.speed_sound()

        # energy balance
        res = 1 - h_a / ht_in - (c_a ** 2 - U ** 2) / (2 * ht_in)

        return res

    def annular_sonic_loop(self, p, *data):
        """ Compute sonic conditions in the annular section """

        ht_in, s_a, U, Vt = data
        try:
            len(p)
            P_a = p[0]
        except:
            P_a = p

        # sonic conditions
        self.flow.fluid.EoS.update(CoolProp.PSmass_INPUTS, P_a, s_a)
        h_a = self.flow.fluid.EoS.hmass()
        c_a = self.flow.fluid.EoS.speed_sound()

        # energy balance
        res = 1 - h_a / ht_in - (c_a ** 2 + Vt ** 2 - U ** 2) / (2 * ht_in)

        return res

    def massBalanceOutlet(self, p, *data):
        D_mix, V_mix, D_a, V_a, Rmix_R, angle_mix, t_s, delta_star_s = data
        A = (D_mix * V_mix) / (D_a * V_a) * Rmix_R
        res = abs(1 - A * (np.cos(angle_mix) + np.sin(angle_mix) * np.tan(p)) +
                  (t_s + delta_star_s) / np.cos(p))
        return res

    # Shocks-Induced Loss Models todo: implement condensation check and thermo props update
    ###########################################################################################################
    def RankineHugoniotReal(self, p, *data):
        "Solve a non-linear system of 4 equations to find the post-shock conditions and the deviation angle [rad], "
        "once the pre-shock conditions and the shock angle [rad] are known"

        P1, h1, D1, V1, beta = data
        P2, h2, V2, theta = p

        self.flow.fluid.EoS.update(CoolProp.HmassP_INPUTS, h2, P2)
        D2 = self.flow.fluid.EoS.rhomass()
        v2 = 1 / D2
        v1 = 1 / D1

        dP = (P2 - P1) + (v2 - v1) * (D1 * V1 * np.sin(beta)) ** 2
        dh = (h2 - h1) - 0.5 * (P2 - P1) * (v1 + v2)
        dV = V1 * np.cos(beta) - V2 * np.cos(beta - theta)
        dtheta = D1 * np.tan(beta) - D2 * np.tan(beta - theta)

        return dP, dh, dV, dtheta

    def RankineHugoniotPerfect(self, R, Cp, gamma, M1, beta):
        "Compute the post-shock conditions and the deviation angle [rad] under the assumption of perfect gas, once the"
        "pre-shock conditions and the shock angle [rad] are known"
        if M1 * np.sin(beta) >= 1.0:
            M1_normal = M1 * np.sin(beta)
            theta = np.arctan(2 / np.tan(beta) * (M1 ** 2 * np.sin(beta) ** 2 - 1) /
                              (M1 ** 2 * (gamma + np.cos(2 * beta) + 2)))
            M2_normal = np.sqrt((1 + (gamma - 1) / 2 * M1_normal ** 2) / (gamma * M1_normal ** 2 - (gamma - 1) / 2))
            M2 = M2_normal / np.sin(beta)
            P_ratio = 1 + (2 * gamma) / (gamma + 1) * (M1_normal ** 2 - 1)
            D_ratio = (1 + (gamma + 1) / (gamma - 1) * P_ratio) / ((gamma + 1) / (gamma - 1) + P_ratio)
            T_ratio = P_ratio / D_ratio
            delta_s = Cp * np.log(T_ratio) - R * np.log(P_ratio)
        else:
            M2 = M1
            P_ratio = 1
            T_ratio = 1
            delta_s = 0
            theta = 0

        return theta, P_ratio, T_ratio, M2, delta_s

    def ComputeShockLosses(self, row, switch, warning):
        "Determine the dissipation induced by an oblique shock with pre-defined angle"
        # AG 13/05/20: SHOCK LOSS DATA (only ss)
        R = self.flow.R
        Cp_id = self.flow.cp0
        gamma_id = self.flow.gamma_id
        if row == 'stator':
            flag_row = 0
            gamma_Pv = self.flow.gamma_Pv[1, :]
            M_preShock = self.flow.MachAbs[1, :] * 1.12
            shock_angle = np.radians(55.0) * np.ones(self.flow.Nslices)     # only used as initial value
            P_preShock = self.flow.P[1, :] * 0.85
            s_preShock = self.flow.s[0, :]
        elif row == 'rotor':
            flag_row = 2
            gamma_Pv = self.flow.gamma_Pv[3, :]
            M_preShock = self.flow.MachRel[3, :] * 1.06
            shock_angle = np.radians(55.0) * np.ones(self.flow.Nslices)     # only used as initial value
            P_preShock = self.flow.P[3, :] * 0.92
            s_preShock = self.flow.s[2, :]

        for ii in range(self.flow.Nslices):
            # perfect gas: Cp and gamma computed with (Pc, Tc)
            theta_tmp, P_ratio_tmp, T_ratio_tmp, M_postShock_tmp, delta_s_tmp = \
                self.RankineHugoniotPerfect(R, Cp_id, gamma_id, M_preShock[ii], shock_angle[ii])

            if theta_tmp != 0:
                shock_angle[ii] = np.arcsin(1 / M_preShock[ii]) + (gamma_Pv[ii] + 1) / 4 * \
                                  M_preShock[ii] ** 2 / (M_preShock[ii] ** 2 - 1) * theta_tmp

            theta = theta_tmp
            if switch == 'real':
                if P_ratio_tmp != 1:
                    try:
                        # real gas
                        self.flow.fluid.EoS.update(CoolProp.PSmass_INPUTS, P_preShock[ii], s_preShock[ii])
                        V_preShock = M_preShock[ii] * self.flow.fluid.EoS.speed_sound()
                        h_preShock = self.flow.fluid.EoS.hmass()
                        D_preShock = self.flow.fluid.EoS.rhomass()
                        P_guess = P_preShock[ii] * P_ratio_tmp
                        T_guess = self.flow.fluid.EoS.T() * T_ratio_tmp
                        self.flow.fluid.EoS.update(CoolProp.PT_INPUTS, P_guess, T_guess)
                        h_guess = self.flow.fluid.EoS.hmass()
                        V_guess = M_postShock_tmp * self.flow.fluid.EoS.speed_sound()

                        dshock_angle = 10
                        while dshock_angle > 1e-3:
                            data = (P_preShock[ii], h_preShock, D_preShock, V_preShock, shock_angle[ii])
                            guess = (P_guess, h_guess, V_guess, theta)
                            P_postShock, h_postShock, V_postShock, theta = \
                                opti.fsolve(self.RankineHugoniotReal, guess, args=data, full_output=False, xtol=1.0e-06)
                            shock_angle_new = np.arcsin(1 / M_preShock[ii]) + (gamma_Pv[ii] + 1) / 4 * \
                                              M_preShock[ii] ** 2 / (M_preShock[ii] ** 2 - 1) * theta
                            dshock_angle = (shock_angle[ii] - shock_angle_new) / shock_angle[ii]
                            shock_angle[ii] = shock_angle_new

                        self.flow.fluid.EoS.update(CoolProp.HmassP_INPUTS, h_postShock, P_postShock)
                        self.ds_shock[flag_row, ii] = self.flow.fluid.EoS.smass() - s_preShock[ii]
                        if self.ds_shock[flag_row, ii] <= 0:
                            raise Exception('delta_s <= 0 in real gas Rankine-Hugoniot')
                    except:
                        self.ds_shock[flag_row, ii] = delta_s_tmp
                        self.flag_shock[flag_row, ii] = 1
                        if warning == 1:
                            print('Error in Real Gas Shock Model: going back to Ideal Gas Model')
                else:
                    self.ds_shock[flag_row, ii] = 0
            elif switch == 'ideal':
                self.ds_shock[flag_row, ii] = delta_s_tmp

            # ensure non-negative entropy generation before accumulating
            if self.ds_shock[flag_row, ii] >= - 0.01:
                self.ds[flag_row, ii] = self.ds[flag_row, ii] + self.ds_shock[flag_row, ii]
            else:
                self.ds_shock[flag_row, ii] = 0.0

        return

    # Secondary Flow Loss Models
    ###########################################################################################################
    def ComputeSecondaryFlowsLosses_BSM(self, row, H_Cax, blade_stagger):
        """ Determine the dissipation due to the mixing of the secondary flows according to Benner & Sjolander & Moustapha"
        'NB: it includes endwall boundary layer losses'
        'NB: computed with flow properties at midspan and spread along the blade span' """

        if row == 'stator':
            flag_row = 0
            flow_angle_in = np.radians(self.flow.alpha[0, self.flow.Nmid])
            flow_angle_out = np.radians(self.flow.alpha[1, self.flow.Nmid])
            Pt_in = self.flow.Pt[0, self.flow.Nmid]
            P_out = self.flow.P[1, self.flow.Nmid]
            ht_out = self.flow.ht[1, self.flow.Nmid]
            s_in = self.flow.s[0, self.flow.Nmid]
        elif row == 'rotor':
            flag_row = 2
            flow_angle_in = np.radians(self.flow.beta[2, self.flow.Nmid])
            flow_angle_out = np.radians(self.flow.beta[3, self.flow.Nmid])
            Pt_in = self.flow.Ptr[2, self.flow.Nmid]
            P_out = self.flow.P[3, self.flow.Nmid]
            ht_out = self.flow.htr[3, self.flow.Nmid]
            s_in = self.flow.s[2, self.flow.Nmid]

        CR = np.cos(flow_angle_in) / np.cos(flow_angle_out)     # Convergence ratio: flow acceleration
        H_c = H_Cax * np.cos(blade_stagger)

        if H_c < 2.0:
            Y = (0.038 + 0.41 * np.tanh(1.20 * self.delta_star_H)) / \
                (np.sqrt(np.cos(blade_stagger)) * CR * (H_c * np.cos(flow_angle_out) / np.cos(blade_stagger)) ** 0.55)
        else:
            Y = (0.052 + 0.56 * np.tanh(1.20 * self.delta_star_H)) / \
                (np.sqrt(np.cos(blade_stagger)) * CR * H_c * (np.cos(flow_angle_out) / np.cos(blade_stagger)) ** 0.55)

        Pt_out = (Pt_in + Y * P_out) / (Y + 1)

        self.flow.fluid.EoS.update(CoolProp.HmassP_INPUTS, ht_out, Pt_out)
        self.ds_secondary[flag_row, :] = self.flow.fluid.EoS.smass() - s_in

        return

    def ComputeSecondaryFlowsLosses_RG(self, chord):
        """ Determine the secondary flow losses occurring in the rotor of radial turbines as per [Ref. 9] """

        # average velocity across the channel
        Wmean = np.mean([np.mean(self.flow.W[2]), np.mean(self.flow.W[3])])

        # calculation of the enthalpy drop
        eps_secondary = 0.01 * 2 * self.flow.R_ad[2, self.flow.Nmid] / self.flow.H_Rm[2] \
                        * 2 * np.pi * self.flow.R_ad[2, self.flow.Nmid] / self.flow.Z / chord + \
                        (1 - np.cos(np.mean(np.radians(self.flow.alpha[3]))))

        Deltah = eps_secondary * Wmean ** 2 / 2

        # determination of outlet static thermodynamic conditions and calculation of the entropy generation
        h_out = self.flow.h3_is + Deltah
        for ii in range(self.flow.Nslices):
            self.flow.fluid.EoS.update(CoolProp.HmassP_INPUTS, h_out[ii], self.flow.P[3, ii])
            s_out = self.flow.fluid.EoS.smass()
            self.ds_secondary[2, ii] = s_out - self.flow.s[2, ii]

        self.ds[2,:] = self.ds[2,:] + self.ds_secondary[2,:]

        return

    # Tip-leakage Loss Models
    ###########################################################################################################
    def ComputeUnshroudedTipLeakageLosses(self, row, g_H, solidity):
        "Determine the dissipation due to the leakage flow for unshrouded blade according to Denton"
        "Hp: incompressible flow"
        if row == 'stator':
            flag_row = 0
            D_in = self.flow.D[0, 0]
            D_out = self.flow.D[1, 0]
            Vax_in = self.flow.Vm[0, 0]
            Vax_out = self.flow.Vm[1, 0]
            T_out_is = self.flow.T1_is[:]
        elif row == 'rotor':
            flag_row = 2
            D_in = self.flow.D[2, -1]
            D_out = self.flow.D[3, -1]
            Vax_in = self.flow.Vm[2, -1]
            Vax_out = self.flow.Vm[3, -1]
            T_out_is = self.flow.T3_is[:]
            gx_H = g_H[0]
            gr_H = g_H[1]

        D_mean = (D_in + D_out) / 2
        Vax_mean = (Vax_in + Vax_out) / 2
        x = np.linspace(0, 1.0, self.Nstream)
        if row == 'stator':
            int = fld.integrate.trapz((self.Vss[flag_row, 0, :] ** 2 *
                                       (1 - self.Vps[flag_row, 0, :] / self.Vss[flag_row, 0, :]) *
                                       np.sqrt(2 * self.Dps[flag_row, 0, :] * abs(self.Pps[flag_row, 0, :] -
                                                                                  self.Pss[flag_row, 0, :]))), x=x)
            delta_h = self.C_cu * g_H * solidity / (D_mean * Vax_mean) * int
        elif row == 'rotor':
            # Modified tip leakage model according to [Ref. 8]
            Cd_x = 0.4
            Cd_r = 0.75
            Lh = np.pi / 4 * ((self.flow.z_ad[3, self.flow.Nmid] - self.flow.z_ad[2, 0]) +
                              (self.flow.R_ad[2, self.flow.Nmid] - self.flow.z_ad[3, -1] -
                               self.flow.H_Rm[3] / 2))

            L_gr = Lh - self.flow.H_Rm[2]
            perc_gx = 1 - L_gr / Lh
            index_gx = np.int64(np.round(self.Nstream * perc_gx, 0))

            int_gx = fld.integrate.trapz((self.Vss[flag_row, -1, :index_gx] ** 2 *
                                          (1 - self.Vps[flag_row, -1, :index_gx] / self.Vss[flag_row, -1, :index_gx]) *
                                          np.sqrt(2 * self.Dps[flag_row, -1, :index_gx] * abs(self.Pps[flag_row, -1, :index_gx] -
                                                                                              self.Pss[flag_row, -1, :index_gx]))), x=x[:index_gx])
            delta_h_gx = Cd_x * gx_H * solidity / (D_mean * Vax_mean) * int_gx

            int_gr = fld.integrate.trapz((self.Vss[flag_row, -1, index_gx:] ** 2 *
                                          (1 - self.Vps[flag_row, -1, index_gx:] / self.Vss[flag_row, -1, index_gx:]) *
                                          np.sqrt(2 * self.Dps[flag_row, -1, index_gx:] * abs(self.Pps[flag_row, -1, index_gx:] -
                                                                                              self.Pss[flag_row, -1, index_gx:]))), x=x[index_gx:])
            delta_h_gr = Cd_r * gr_H * solidity / (D_mean * Vax_mean) * int_gr

            delta_h = delta_h_gx + delta_h_gr

        self.ds_leakage[flag_row, :] = np.array(self.flow.massFlowWeighting(delta_h,row)) / T_out_is

        self.ds[flag_row,:] = self.ds[flag_row,:] + self.ds_leakage[flag_row,:]

        return

    def computeTipLeakageLoss_Jansen(self, g_H):
        """ Determine the loss due to leakage for unshrouded radial impellers according to Jansen """

        delta_h = 0.6 * g_H * self.flow.Vt[2, self.flow.Nmid] / self.flow.U[2, self.flow.Nmid] * \
                  (4 * np.pi / self.flow.H_Rm[2] / self.flow.Z *
                   ((self.flow.R_ad[3, -1] ** 2 - self.flow.R_ad[3, 0] ** 2) /
                    (self.flow.R_ad[2, self.flow.Nmid] - self.flow.R_ad[3, -1]) / (1 + self.flow.D[2, self.flow.Nmid] /
                                                                                   self.flow.D[3, self.flow.Nmid])) *
                   self.flow.Vt[2, self.flow.Nmid] / self.flow.U[2, self.flow.Nmid] *
                   self.flow.Vm[3, self.flow.Nmid] / self.flow.U[2, self.flow.Nmid]) ** 0.5

        h_out = delta_h + self.flow.h3_is
        for ii in range(self.flow.Nslices):
            self.flow.fluid.EoS.update(CoolProp.HmassP_INPUTS, h_out[ii], self.flow.P[3, ii])
            s_out = self.flow.fluid.EoS.smass()
            self.ds_leakage[2, ii] = s_out - self.flow.s[2, ii]

        self.ds[2,:] = self.ds[2,:] + self.ds_leakage[2,:]

        return

    def rotorTipLeakageLoss_Baines(self, gr_H, gx_H):
        """ Determine the loss due to leakage for unshrouded radial impellers according to [Ref. 8] """

        U2s = self.flow.U[2, -1]
        Vm2 = self.flow.Vm[2, self.flow.Nmid]
        Vm3 = self.flow.Vm[3, self.flow.Nmid]
        R2m = self.flow.R_ad[2, self.flow.Nmid]
        R3m = self.flow.R_ad[3, self.flow.Nmid]
        R3s = self.flow.R_ad[3, -1]
        H2 = self.flow.H_Rm[2]
        H3 = self.flow.H_Rm[3]
        z2h = self.flow.z_ad[2, 0]
        z3m = self.flow.z_ad[3, self.flow.Nmid]
        Zb = self.flow.Z

        Kx = self.Kx
        Kr = self.Kr
        Kxr = self.Kxr

        z = (z3m - z2h)

        Cx = (1 - (R3s / R2m)) / (Vm2 * H2)
        Cr = (R3s / R2m) * (z - H2) / (Vm3 * R3m * H3)

        # clearance loss coefficient - corrected from [Ref. 8]: U2s ** 5 --> U2s ** 3 to obtain dimensionally correct coefficient
        Lc = (U2s ** 3 * Zb) / (8 * np.pi) * \
             (Kx * gx_H * H2 * Cx + Kr * gr_H * H3 * Cr + Kxr * np.sqrt(gx_H * H2 * Cx * gr_H * H3 * Cr))

        # determination of outlet thermodynamic conditions and calculation of the entropy generation
        h_out = self.flow.massFlowWeighting(Lc, row='rotor', split=1) + self.flow.h3_is
        for ii in range(self.flow.Nslices):

            if self.flow._condensation:
                self.flow.fluid.EoS.set_humidity('W', self.flow.WAR[3, ii])

            self.flow.fluid.EoS.update(CoolProp.HmassP_INPUTS, h_out[ii], self.flow.P[3, ii])
            s_out = self.flow.fluid.EoS.smass()

            self.ds_leakage[2, ii] = s_out - self.flow.s[2, ii]

            if self.ds_leakage[2,ii] >= - 0.01:
                self.ds[2,ii] = self.ds[2,ii] + self.ds_leakage[2,ii]
            else:
                self.ds_leakage[2,ii] = 0.0
        
        return

    # Friction Loss Models
    ###########################################################################################################
    def calculateBackfaceWindageLoss(self, eps_b, R_tip, b_in, mass_flow): # todo: implement condensation check and thermo props update
        """ Function that calculates the external loss due to impeller back-face flow shearing from [Ref. 5] """

        D_tip = self.flow.D[2, 0]
        P_tip = self.flow.P[2, 0]
        D_in_mfave = self.flow.massFlowWAv(self.flow.D[2, :], 2)
        D_out_mfave = self.flow.massFlowWAv(self.flow.D[3, :], 3)
        D_ave = np.mean([D_in_mfave, D_out_mfave])
        s_tip = self.flow.s[2, 0]
        U_tip = self.flow.U[2, 0]
        W_out = self.flow.massFlowWAv(self.flow.U[3, :], 3)

        eps = eps_b * b_in
        self.flow.fluid.EoS.update(CP.PSmass_INPUTS, P_tip, s_tip)
        dyn_vis = self.flow.fluid.EoS.viscosity()
        Re_y = D_tip * U_tip * R_tip / dyn_vis

        if Re_y <= 3e5: Kf = 3.7 * (eps / R_tip) ** 0.1 / Re_y ** 0.5
        else: Kf = 0.102 * (eps / R_tip) ** 0.1 / Re_y ** 0.2

        self.dh_windage = 0.25 * Kf * D_ave * U_tip ** 5 * R_tip ** 2 / mass_flow / (W_out ** 2)

        return

    def calculateEndwallLosses(self, row, massflow, R1, R2):
        """ Calculate the endwall loss of the stator and vaneless space.

         Stator: flow loss according to [Ref. 10] using Denton's dissipation coefficient

         Vaneless: flow loss according to Stanitz vaneless flow model from [Ref. 6] using Japikse's
                   vaneless flow friction coefficient calculation
        """

        span = self.flow.Nmid

        if self.flow._condensation:
            flow = deepcopy(self.flow)
        else:
            flow = None

        if row == 'stator':

            flag_row = 0

            s0 = self.flow.s[0, span]
            ht0 = self.flow.ht[0, span]
            V0 = self.flow.V[0, span]
            T0 = self.flow.T[0, span]
            D0 = self.flow.D[0, span]
            P0 = self.flow.P[0, span]
            WAR0 = self.flow.WAR[0, span]
            P1 = self.flow.P[1, span]
            Cd = self.Cd

            Rin = R1
            sin = s0
            Vin = V0
            Tin = T0
            Din = D0
            Pin = P0
            WARin = WAR0

            dR = (R2 - R1) / 20
            R = Rin + dR

            dP = (P1 - P0) / 20
            P = Pin + dP

            while R > R2:

                WAR = WARin
                if self.flow._condensation:
                    self.flow.fluid.EoS.set_humidity('W', WAR)

                self.flow.fluid.EoS.update(CP.PSmass_INPUTS, P, sin)
                T = self.flow.fluid.EoS.T()
                D = self.flow.fluid.EoS.rhomass()
                h = self.flow.fluid.EoS.hmass()

                V = np.sqrt(2 * (ht0 - h))
                A = np.pi * (Rin ** 2 - R ** 2) * 2

                res_V = 10
                max_it = 50
                it = 0
                while res_V > 1e-6 and it <= max_it:
                    V_ave = np.mean([Vin, V])
                    D_ave = np.mean([Din, D])
                    T_ave = np.mean([Tin, T])

                    S_gen = Cd * D_ave * V_ave ** 3 / T_ave * A
                    s = sin + S_gen / massflow

                    self.flow.fluid.EoS.update(CP.PSmass_INPUTS, P, s)

                    if self.flow._condensation:
                        self.flow.fluid.EoS.set_humidity('W', WAR)
                        self.flow.WAR[1, self.flow.Nmid] = WAR
                        self.flow.T[1, self.flow.Nmid] = self.flow.fluid.EoS.T()
                        self.flow.P[1, self.flow.Nmid] = P
                        flow = update_props(self.flow, 1, self.flow.Nmid)
                        P, T = (flow.P[1, flow.Nmid], flow.T[1, flow.Nmid])
                        self.flow.fluid.EoS.set_humidity('W', flow.fluid.EoS.absolute_humidity())
                        self.flow.fluid.EoS.update(CoolProp.PT_INPUTS, P, T)
                        WAR = self.flow.fluid.EoS.absolute_humidity()
                    h = self.flow.fluid.EoS.hmass()
                    T = self.flow.fluid.EoS.T()
                    D = self.flow.fluid.EoS.rhomass()

                    Vold = V
                    V = np.sqrt(2 * (ht0 - h))
                    res_V = abs(V - Vold) / Vold

                    it += 1

                Rin = R
                sin = s
                Vin = V
                Tin = T
                Din = D
                Pin = P
                WARin = WAR

                R = Rin + dR
                P = Pin + dP

        if row == 'vaneless':
            flag_row = 1

            s1 = self.flow.s[1, span]
            D1 = self.flow.D[1, span]
            h1 = self.flow.h[1, span]
            V1 = self.flow.V[1, span]
            Vm1 = self.flow.Vm[1, span]
            Vt1 = self.flow.Vt[1, span]
            WAR1 = self.flow.WAR[1, span]
            k = self.k
            b = self.flow.H_Rm[2] * R2

            mf = massflow
            Vin = V1
            Vtin = Vt1
            Vmin = Vm1
            Rin = R1
            Din = D1
            hin = h1
            sin = s1
            WARin = WAR1

            dR = (R1 - R2) / 20
            R = R1 - dR
            Vt_old = Vt1 * R1 / R

            while R > R2:
                WAR = WARin
                P, D, mu, Vm = self.isentropicStaticPressureLoop(Din * 0.99, Vin, Vtin, Vmin, Rin, R, Din, hin, sin, span)
                res = 10
                max_it = 50
                it = 0
                while res >= 1E-6 and it <= max_it:
                    Re = mf / mu / (2 * R)
                    Cf = k * (1.8 * 10**5 / Re) ** 0.2
                    Vtin_Vt = R / Rin + (2 * np.pi * Cf * D * Vt_old * (Rin ** 2 - Rin * R)) / mf
                    Vt = Vtin / Vtin_Vt
                    res = abs(Vt - Vt_old) / Vt_old
                    Vt_old = Vt
                    V = np.sqrt(Vm ** 2 + Vt ** 2)
                    h = h1 + V1 ** 2 / 2 - V ** 2 / 2

                    self.flow.fluid.EoS.update(CoolProp.HmassP_INPUTS, h, P)
                    if self.flow._condensation:
                        self.flow.fluid.EoS.set_humidity('W', WAR)
                        self.flow.WAR[2, self.flow.Nmid] = WAR
                        self.flow.T[2, self.flow.Nmid] = self.flow.fluid.EoS.T()
                        self.flow.P[2, self.flow.Nmid] = P
                        flow = update_props(self.flow, 2, self.flow.Nmid)
                        P, T = (flow.P[2, flow.Nmid], flow.T[2, flow.Nmid])
                        self.flow.fluid.EoS.set_humidity('W', flow.fluid.EoS.absolute_humidity())
                        self.flow.fluid.EoS.update(CoolProp.PT_INPUTS, P, T)
                        WAR = self.flow.fluid.EoS.absolute_humidity()
                    s = self.flow.fluid.EoS.smass()
                    mu = self.flow.fluid.EoS.viscosity()
                    D = self.flow.fluid.EoS.rhomass()

                    Vm = mf / (2 * np.pi * R * b * D)

                    it += 1

                Vin = V
                Vtin = Vt
                Vmin = Vm
                Rin = R
                Din = D
                hin = h
                sin = s
                WARin = WAR

                R = R - dR

        self.flow.fluid.EoS.update(CoolProp.HmassP_INPUTS, h, P)
        s = self.flow.fluid.EoS.smass()
        self.ds_endwall[flag_row, :] = self.flow.massFlowWeighting(s - self.flow.s[flag_row, span], row='rotor', split=1)

        if np.all(self.ds_endwall[flag_row,:] >= - 0.01):
            self.ds[flag_row,:] = self.ds[flag_row,:] + self.ds_endwall[flag_row,:]
        else:
            self.ds_endwall[flag_row,:] = 0.0

        return

    def isentropicStaticPressureLoop(self, *data):
        D, V1, Vt1, Vm1, R1, R, D1, h1, s1, span = data
        Vt = Vt1 * R1 / R

        res = 10
        while res > 1E-3:
            Vm = Vm1 * D1 / D * R1 / R
            V = np.sqrt(Vt ** 2 + Vm ** 2)
            h = h1 + V1 ** 2 / 2 - V ** 2 / 2

            self.flow.fluid.EoS.update(CoolProp.HmassSmass_INPUTS, h, s1)
            D_new = self.flow.fluid.EoS.rhomass()

            res = abs(D_new - D) / D
            D = D_new

        self.flow.fluid.EoS.update(CoolProp.HmassSmass_INPUTS, h, s1)
        mu = self.flow.fluid.EoS.viscosity()
        P = self.flow.fluid.EoS.p()

        return P, D, mu, Vm