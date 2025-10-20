#############################################################################
# TurboSim - Integrated Turbomachinery Design Suite
# Authors: Dr. A. Giuffre', ir. M. Majer, ir. F. Sinopoli, Prof. M. Pini
# Content: Computation of diffusers design and off-design performance
# 2019 - 2025 - Delft University of Technology - All rights reserved
#############################################################################

import CoolProp
import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import scipy.linalg as linalg
import TurboSim.geometry as geo
import TurboSim.flow_model as flow


class AnnularDiffuser:
    """ Annular diffuser model based on control volume approach with straight inner and outer walls """
    def __init__(self, EoS, library, fluid, verbose, q_w=0.0, Cf=0.0, inlet=None, outlet=None, geometry=None):
        """
        :param EoS: instance of CoolProp.AbstractState class
        :param library: string identifying the thermodynamic model used in the EoS object
        :param fluid: string identifying the prescribed working fluid
        :param verbose: 0 --> minimal print to screen
                        1 --> print to screen also intermediate results
                        2 --> print to screen all the warnings and intermediate results (debugging)
        :param q_w: wall heat flux (float)
            default value is 0 --> adiabatic wall
        :param Cf: wall friction factor (float)
            default value is 0 --> calculation based on Re
        :param inlet: instance of FlowNode class identifying the inlet port
        :param outlet: instance of FlowNode class identifying the outlet port
        :param geometry: instance of AnnularDiffuser class defined in geometry module

        """
        self.error = False
        self.verbose = verbose
        self.Cf = Cf
        self.q_w = q_w

        self.dPt = 0.0
        self.dht = 0.0
        self.ds = 0.0
        self.ht = 0.0
        self.Re = 0.0
        self.Cp = 0.0
        self.Cp_id = 0.0
        self.k = 0.0
        self.eps = 0.0

        # initialize quantities as ndarray (length depends on time marching process)
        self.Vm = np.ndarray((), float)
        self.Vt = np.ndarray((), float)
        self.alpha = np.ndarray((), float)
        self.V = np.ndarray((), float)
        self.D = np.ndarray((), float)
        self.P = np.ndarray((), float)
        self.s = np.ndarray((), float)
        self.h = np.ndarray((), float)
        self.h_is = np.ndarray((), float)
        self.Pt = np.ndarray((), float)

        if inlet is None:
            self.inlet = flow.FlowNode(EoS, library, fluid)
        else:
            self.inlet = inlet

        if outlet is None:
            self.outlet = flow.FlowNode(EoS, library, fluid)
        else:
            self.outlet = outlet

        if geometry is None:
            self.geometry = geo.AnnularDiffuser(phi=30, div=5, AR=5, R_inner_in=1.3, R_outer_in=0.8, z_inner_in=1.2, z_outer_in=1.1)
        else:
            self.geometry = geometry

    def set_diffuser_friction_factor(self):
        if self.Cf == 0:
            raise NotImplementedError("Calculation of the friction factor based on correlations not implemented yet, "
                                      "please specify a value for the friction factor")
        else:
            pass

    def compute_annular_diffuser_flow(self):
        """ 1D annular diffuser model from [Ref. 1] """

        def r_fun(r_in, phi, m):
            r = r_in + np.sin(phi) * m
            return r

        def z_fun(z_in, phi, m):
            z = z_in + np.cos(phi) * m
            return z

        def b_fun(b_in, div, m):
            b = b_in + 2 * np.tan(div) * m
            return b

        def ODE_Diffuser(m, U, *data):

            _, phi, div, r_in, b_in, z_in, Cf, q_w = data

            # Rename variables
            v_m = U[0]
            v_t = U[1]
            d = U[2]
            p = U[3]
            alpha = np.arctan(v_t / v_m)
            v = np.sqrt(v_m ** 2 + v_t ** 2)

            # Increment for finite differences
            delta = 1e-5

            # Local geometry
            r = r_fun(r_in, phi, m)     # Radius as a function of m
            z = z_fun(z_in, phi, m)     # Axial distance as a function of m
            b = b_fun(b_in, div, m)     # Channel width as a function of m

            # Derivative of the area change(forward finite differences)
            diff_br = (b_fun(b_in, div, m + delta) * r_fun(r_in, phi, m + delta) - b * r) / delta

            # Derivative of internal energy with respect to pressure (constant density)
            self.outlet.EoS.update(CoolProp.DmassP_INPUTS, d, p - delta)
            e_1 = self.outlet.EoS.umass()
            self.outlet.EoS.update(CoolProp.DmassP_INPUTS, d, p + delta)
            e_2 = self.outlet.EoS.umass()
            diff_e = (e_2 - e_1) / (2 * delta)

            # Speed of sound (avoid computations in the two phase region
            self.outlet.EoS.update(CoolProp.DmassP_INPUTS, d, p)
            a = self.outlet.EoS.speed_sound()

            # Stress at the wall
            tau_w = Cf * d * v ** 2 / 2         # Skin friction coefficient

            # Coefficient matrix A(pressure conversion from kPa to Pa)
            A = np.ndarray((4, 4), float)
            A[0, :] = np.array([d,             0,               v_m,                        0])
            A[1, :] = np.array([d * v_m,       0,               0,                          1])
            A[2, :] = np.array([0,             d * v_m,         0,                          0])
            A[3, :] = np.array([0,             0,               - d * v_m * a ** 2,         d * v_m])

            # Source term vector
            S = np.zeros(4, float)
            S[0] = -d * v_m / (b * r) * diff_br
            S[1] = +d * v_t * v_t / r * np.sin(phi) - 2 * tau_w / b * np.cos(alpha)
            S[2] = -d * v_t * v_m / r * np.sin(phi) - 2 * tau_w / b * np.sin(alpha)
            S[3] = 2 * (tau_w * v + q_w) / b / diff_e

            # Obtain the slope of the solution by Gaussian elimination
            dUdm = np.zeros((5), float)
            dUdm[0:4] = linalg.solve(A, S)

            # Check entropy generation
            T = self.outlet.EoS.T()
            sigma = 2 / b * (tau_w * v)
            dUdm[4] = sigma / (d * v_m) / T         # ds / dm

            return dUdm

        def area_ratio_lsq(m, AR_prescribed, phi, div, r_in, b_in):

            # Geometry
            r = r_fun(r_in, phi, m) # Radius as a function of m
            b = b_fun(b_in, div, m) # Channel width as a function of m
            AR_current = (b * r) / (b_in * r_in) # Current area ratio

            # Stopping criterion
            AR_check = (AR_prescribed - AR_current) / AR_prescribed
            isterminal = True
            direction = -1

            return AR_check, isterminal, direction

        def findIntegrationInterval(p, *data):
            mf = p[0]
            AR_prescribed, phi, div, r_in, b_in, z_in, Cf, q_w = data
            output = integrate.solve_ivp(ODE_Diffuser, [0, mf], U0, args=data,
                                         method='RK45', atol=1e-6, rtol=1e-6)
            m = output.get('t')
            AR_check,_ ,_ = area_ratio_lsq(m[-1], AR_prescribed, phi, div, r_in, b_in)
            return AR_check

        ### ------------------------------------ Solution of ODE system ------------------------------------ ###

        # Initial conditions of the ODE
        U0 = [self.inlet.Vm, self.inlet.Vt, self.inlet.D, self.inlet.P, self.inlet.s]

        # Data of the ODE system
        b_in = self.geometry.H_in / np.cos(self.geometry.phi)
        data = (self.geometry.AR, self.geometry.phi, self.geometry.div, self.geometry.R_mid_in, b_in, self.geometry.z_mid_in, self.Cf,
                self.q_w)

        # Find upper bound of integration interval with least_squares by minimizing AR_check
        mf = opt.least_squares(findIntegrationInterval, self.geometry.z_mid_in + 0.03,
                                args=data, ftol=1e-3).get('x')

        # Integrate the ode system using RK45
        output = integrate.solve_ivp(ODE_Diffuser, [0, mf], U0, args=data, method='RK45', atol=1e-6, rtol=1e-6)
        [m, U] = [output.get('t'), output.get('y')]

        # Save thermo-flow solution
        self.D = U[2, :]
        self.P = U[3, :]
        self.s = U[4, :]
        self.Vm = U[0, :]
        self.Vt = U[1, :]
        self.V = np.sqrt(self.Vm ** 2 + self.Vt ** 2)
        self.alpha = np.rad2deg(np.arctan(self.Vt / self.Vm))
        # Compute additional flow quantities
        self.h = self.inlet.ht - self.V ** 2 / 2
        self.Pt = self.P + self.D * self.V ** 2 / 2

        # save outlet node thermo-flow properties
        self.outlet.P = self.P[-1]
        self.outlet.s = self.s[-1]
        self.outlet.update_static_thermodynamic_state('Ps')
        self.outlet.EoS.update(CoolProp.PSmass_INPUTS, self.outlet.P, self.inlet.s)
        self.outlet.h_is = self.outlet.EoS.hmass()
        self.outlet.Pt = self.Pt[-1]
        self.outlet.s = self.s[-1]
        self.outlet.update_total_thermodynamic_state('Ps')

        self.outlet.Vm = self.Vm[-1]
        self.outlet.Vt = self.Vt[-1]
        self.outlet.V = self.V[-1]
        self.outlet.alpha = self.alpha[-1]

        # Retrieve geometry
        self.geometry.R_mid = r_fun(self.geometry.R_mid_in, self.geometry.phi, m)
        self.geometry.z_mid = z_fun(self.geometry.z_mid_in, self.geometry.phi, m)
        self.geometry.H = b_fun(b_in, self.geometry.div, m)
        self.geometry.AR = (self.geometry.H * self.geometry.R_mid) / (self.geometry.H_in * self.geometry.R_mid_in)

        # Define the inner and outer surface coordinates
        self.geometry.z_outer = self.geometry.z_mid - self.geometry.H / 2 * np.sin(self.geometry.phi)
        self.geometry.z_inner = self.geometry.z_mid + self.geometry.H / 2 * np.sin(self.geometry.phi)
        self.geometry.R_outer = self.geometry.R_mid + self.geometry.H / 2 * np.cos(self.geometry.phi)
        self.geometry.R_inner = self.geometry.R_mid - self.geometry.H / 2 * np.cos(self.geometry.phi)

        # Performance
        # Cp for incompressible flow
        self.Cp_inc = (self.outlet.P - self.inlet.P) / (1 / 2 * self.inlet.D * self.inlet.V ** 2)
        # Cp for compressible flow
        self.Cp = (self.outlet.P - self.inlet.P) / (self.inlet.Pt - self.inlet.P)
        # ideal Cp
        self.Cp_id = 1 - (self.geometry.R_mid_in / self.geometry.R_mid[-1]) ** 2 * \
                     ((self.geometry.H_in / self.geometry.H[-1]) ** 2 + self.inlet.Vt / self.inlet.Vm ** 2) / \
                     (1 + self.inlet.Vt / self.inlet.Vm ** 2)
        # total pressure loss coefficient
        self.k = (self.inlet.Pt - self.outlet.Pt) / (self.inlet.Pt - self.inlet.P)
        # kinetic energy loss coefficient
        self.eps = (self.outlet.Pt - self.outlet.P) / (self.inlet.Pt - self.inlet.P)


class ConicalDiffuser:
    """ Conical diffuser used at the outlet of axial machines or radial inflow machines """
    def __init__(self):
        self.ciao = 0


class WedgeDiffuser:
    """ Wedge diffuser used in radial outflow machines """
    def __init__(self):
        self.ciao = 0


class CascadeDiffuser:
    """ Cascade diffuser used in radial outflow machines """
    def __init__(self):
        self.ciao = 0


class PlanarDiffuser:
    """ Planar vaneless diffuser used in radial outflow machines """
    def __init__(self, n_stream, loss_model, verbose, geometry: geo.PlanarDiffuser,
                 inlet: flow.FlowNode, outlet: flow.FlowNode):
        """
        :param loss_model: string identifying the loss model to be used
        :param verbose: 0 --> minimal print to screen
                        1 --> print to screen also intermediate results
                        2 --> print to screen all the warnings and intermediate results (debugging)
        :param inlet: instance of FlowNode class identifying the inlet port
        :param outlet: instance of FlowNode class identifying the outlet port
        :param geometry: instance of PlanarDiffuser class defined in geometry module
        """
        self.error = False
        self.inlet = inlet
        self.outlet = outlet
        self.geometry = geometry
        self.loss_model = loss_model
        self.verbose = verbose
        self.n_stream = n_stream
        self.dPt = 0.0
        self.dht = 0.0
        self.ds = 0.0
        self.Cp = 0.0
        self.Re = np.zeros(self.n_stream)
        self.Cf = np.zeros(self.n_stream)
        self.P_vec = np.zeros(self.n_stream)
        self.h_vec = np.zeros(self.n_stream)
        self.D_vec = np.zeros(self.n_stream)
        self.s_vec = np.zeros(self.n_stream)
        self.Pt_vec = np.zeros(self.n_stream)
        self.ht_vec = np.zeros(self.n_stream)
        self.mu_vec = np.zeros(self.n_stream)
        self.Vm_vec = np.zeros(self.n_stream)
        self.Vt_vec = np.zeros(self.n_stream)
        self.V_vec = np.zeros(self.n_stream)
        self.alpha_vec = np.zeros(self.n_stream)

        if self.inlet.n_mid != self.outlet.n_mid:
            raise ValueError("The number of span-wise sections at the inlet and outlet nodes must match!")
        else:
            self.n_mid = self.inlet.n_mid

    def update_outlet_flow_properties(self):
        """
        Update the flow properties at the outlet of the diffuser.
        Physically the total enthalpy should be constant throughout the diffuser;
        however, there is some artificial loss of total enthalpy due to the discretization error of the forward
        differencing scheme.
        This issue should be solved by implementing a more advanced ODE solver, e.g., scipy.integrate.solve_ivp.
        For the moment, the total enthalpy at the outlet node is set equal to the one of the inlet node.
        """
        self.outlet.P[self.n_mid] = self.P_vec[-1]
        self.outlet.h[self.n_mid] = self.h_vec[-1]
        self.outlet.Pt[self.n_mid] = self.Pt_vec[-1]
        self.outlet.ht[self.n_mid] = self.ht_vec[0]
        self.outlet.update_static_thermodynamic_state('Ph_const')
        self.outlet.update_total_thermodynamic_state('Ph_const')

        self.outlet.V[self.n_mid] = self.V_vec[-1]
        self.outlet.Vm[self.n_mid] = self.Vm_vec[-1]
        self.outlet.Vt[self.n_mid] = self.Vt_vec[-1]
        self.outlet.alpha[self.n_mid] = self.alpha_vec[-1]
        self.outlet.MachAbs[self.n_mid] = self.outlet.V[self.n_mid] / self.outlet.c[self.n_mid]
        self.outlet.MachMer[self.n_mid] = self.outlet.Vm[self.n_mid] / self.outlet.c[self.n_mid]

    def compute_performance_metrics(self):
        """
        Compute the total pressure loss, total enthalpy loss, entropy generation, and pressure recovery factor
        """
        self.dPt = self.Pt_vec[0] - self.Pt_vec[-1]
        self.ds = self.s_vec[-1] - self.s_vec[0]
        self.Cp = (self.P_vec[-1] - self.P_vec[0]) / (self.Pt_vec[0] - self.P_vec[0])

        self.inlet.EoS.update(CoolProp.PSmass_INPUTS, self.Pt_vec[-1], self.s_vec[0])
        self.dht = self.ht_vec[-1] - self.inlet.EoS.hmass()

    def compute_vaneless_diffuser_loss(self):
        """
        Compute pressure recovery and losses due to friction in a vaneless diffuser given its geometry.
        Use the physics-based model originally proposed by Stanitz [1] with the friction factor correlation based
        on the local Reynolds number reported by Dubitsky and Japikse [2].
        References:
        [1] J. D. Stanitz - One-Dimensional Compressible Flow in Vaneless Diffusers of Radial or Mixed-Flow Centrifugal
            Compressors, Including Effects of Friction, Heat Transfer and Area Change, 1952.
        [2] O. Dubitsky, D. Japikse - Vaneless Diffuser Advanced Model, Journal of Turbomachinery, 2008.
        """
        # update vectors at diffuser inlet
        self.P_vec[0] = self.inlet.P[self.n_mid]
        self.h_vec[0] = self.inlet.h[self.n_mid]
        self.D_vec[0] = self.inlet.D[self.n_mid]
        self.s_vec[0] = self.inlet.s[self.n_mid]
        self.mu_vec[0] = self.inlet.mu[self.n_mid]
        self.Vm_vec[0] = self.inlet.Vm[self.n_mid]
        self.Vt_vec[0] = self.inlet.Vt[self.n_mid]
        self.V_vec[0] = self.inlet.V[self.n_mid]
        self.alpha_vec[0] = self.inlet.alpha[self.n_mid]
        self.ht_vec[0] = self.h_vec[0] + self.V_vec[0] ** 2 / 2
        self.inlet.EoS.update(CoolProp.HmassSmass_INPUTS, self.ht_vec[0], self.s_vec[0])
        self.Pt_vec[0] = self.inlet.EoS.p()

        try:
            # iterate from station 0 to to II - 1
            for ii in range(self.n_stream - 1):
                # compute variation of radius and blade height with forward difference
                dR = self.geometry.R_vec[ii + 1] - self.geometry.R_vec[ii]
                dH = self.geometry.H_vec[ii + 1] - self.geometry.H_vec[ii]

                # first solve for the isentropic state at station ii + 1
                mass = self.D_vec[ii] * self.Vm_vec[ii] * self.geometry.R_vec[ii] * self.geometry.H_vec[ii]
                momentum = self.geometry.R_vec[ii] * self.Vt_vec[ii]
                energy = self.h_vec[ii] + self.V_vec[ii] ** 2 / 2
                P_next = opt.fsolve(
                    self.compute_vaneless_diffuser_isentropic_flow, (self.P_vec[ii]),
                    args=(mass, momentum, energy, self.geometry.R_vec[ii + 1],
                          self.s_vec[ii], self.geometry.H_vec[ii + 1]),
                    full_output=False, xtol=1.0e-06)
                
                self.inlet.EoS.update(CoolProp.PSmass_INPUTS, P_next, self.s_vec[ii])
                h_next = self.inlet.EoS.hmass()
                Vt_next = momentum / self.geometry.R_vec[ii + 1]
                V_next = np.sqrt(2 * (energy - h_next))
                Vm_next = np.sqrt(V_next ** 2 - Vt_next ** 2)
                alpha_next = np.arctan(Vt_next / Vm_next)
                self.geometry.L += dR / np.sin((self.alpha_vec[ii] + alpha_next) / 2)

                # update the distance along the flow path and the local values of Reynolds number and friction factor
                self.Re[ii] = self.D_vec[ii] * self.V_vec[ii] * self.geometry.L / self.mu_vec[ii]
                self.Cf[ii] = 0.11 / self.Re[ii] ** 0.2
                # self.Cf[ii] = 0.5 / self.Re[ii] ** 0.3

                # solve for the real flow state at station ii + 1, using the isentropic state as first guess
                if self.loss_model == 'stanitz':
                    x = opt.fsolve(
                        self.Stanitz_flow_equations, np.array([Vm_next, Vt_next, float(P_next), h_next]),
                        args=(self.Vm_vec[ii], self.Vt_vec[ii], self.D_vec[ii], self.P_vec[ii], self.h_vec[ii],
                              self.geometry.R_vec[ii + 1], dR, self.geometry.H_vec[ii + 1], dH,
                              self.geometry.phi_vec[ii + 1], self.Cf[ii]),
                        full_output=True, xtol=1.0e-06)
                    
                    self.Vm_vec[ii + 1] = x[0][0]
                    self.Vt_vec[ii + 1] = x[0][1]
                    self.P_vec[ii + 1] = x[0][2]
                    self.h_vec[ii + 1] = x[0][3]
                    
                elif self.loss_model == 'isentropic':
                    self.Vm_vec[ii + 1] = Vm_next
                    self.Vt_vec[ii + 1] = Vt_next
                    self.P_vec[ii + 1] = P_next
                    self.h_vec[ii + 1] = h_next
                    
                else:
                    raise ValueError("The available choices for diffuser loss model are 'stanitz' or 'isentropic'")

                self.V_vec[ii + 1] = np.sqrt(self.Vm_vec[ii + 1] ** 2 + self.Vt_vec[ii + 1] ** 2)
                self.alpha_vec[ii + 1] = np.arctan(self.Vt_vec[ii + 1] / self.Vm_vec[ii + 1])
                self.inlet.EoS.update(CoolProp.HmassP_INPUTS, self.h_vec[ii + 1], self.P_vec[ii + 1])
                self.D_vec[ii + 1] = self.inlet.EoS.rhomass()
                self.s_vec[ii + 1] = self.inlet.EoS.smass()
                self.mu_vec[ii + 1] = self.inlet.EoS.viscosity()
                self.ht_vec[ii + 1] = self.h_vec[ii + 1] + self.V_vec[ii + 1] ** 2 / 2
                self.inlet.EoS.update(CoolProp.HmassSmass_INPUTS, self.ht_vec[ii + 1], self.s_vec[ii + 1])
                self.Pt_vec[ii + 1] = self.inlet.EoS.p()

        except ValueError:
            self.error = True
            
            if self.verbose == 2:
                print("Error in PlanarDiffuser class, method: compute_vaneless_diffuser_loss")

    def compute_vaneless_diffuser_isentropic_flow(self, p, *data):
        """
        Non-linear system of equations for isentropic compressible flow in a vaneless diffuser.
        Given the radius, blade height and entropy of the i + 1 station, compute the correspondent pressure.
        """
        mass, momentum, energy, R, s, H = data
        P = p

        try:
            self.inlet.EoS.update(CoolProp.PSmass_INPUTS, P, s)
            h = self.inlet.EoS.hmass()
            D = self.inlet.EoS.rhomass()
            Vt = momentum / R
            V = np.sqrt(2 * (energy - h))
            Vm = np.sqrt(V ** 2 - Vt ** 2)
            mass_new = R * D * Vm * H
            res = (mass_new - mass) / mass

        except ValueError:
            res = 10

        return res

    def Stanitz_flow_equations(self, p, *data):
        """
        DAE system for non-isentropic compressible flow in a vaneless diffuser originally formulated by Stanitz [8].
        dx / dr is discretized with forward difference, reducing the problem to a system of non-linear algebraic eqs.
        Given the flow state at station i, compute the thermo-flow quantities at station i + 1.
        """
        Vm_i, Vt_i, rho_i, P_i, h_i, R_i1, dR, H_i1, dH, phi_i1, Cf = data
        Vm_i1, Vt_i1, P_i1, h_i1 = p

        try:
            V_i1 = np.sqrt(Vm_i1 ** 2 + Vt_i1 ** 2)
            alpha_i1 = np.arctan(Vt_i1 / Vm_i1)

            dVm = (Vm_i1 - Vm_i) / dR
            dVt = (Vt_i1 - Vt_i) / dR
            dP = (P_i1 - P_i) / dR
            dh = (h_i1 - h_i) / dR

            self.inlet.EoS.update(CoolProp.HmassP_INPUTS, h_i1, P_i1)
            rho_i1 = self.inlet.EoS.rhomass()
            drho = (rho_i1 - rho_i) / dR

            res1 = Vm_i1 * dVm - Vt_i1 ** 2 / R_i1 + Cf * V_i1 ** 2 / H_i1 * \
                np.cos(alpha_i1) / np.sin(phi_i1) + dP / rho_i1
            res2 = Vm_i1 * dVt + Vm_i1 * Vt_i1 / R_i1 + Cf * V_i1 ** 2 / H_i1 * \
                np.sin(alpha_i1) / np.sin(phi_i1)
            res3 = drho / rho_i1 + dVm / Vm_i1 + 1 / H_i1 * dH / dR + 1 / R_i1
            res4 = dh + Vm_i1 * dVm + Vt_i1 * dVt

        except ValueError:
            res1 = 10
            res2 = 10
            res3 = 10
            res4 = 10

        return res1, res2, res3, res4
