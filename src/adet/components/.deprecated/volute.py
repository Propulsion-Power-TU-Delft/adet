#############################################################################
# TurboSim - Integrated Turbomachinery Design Suite
# Authors: Dr. A. Giuffre', ir. M. Majer, ir. F. Sinopoli, Prof. M. Pini
# Content: Computation of volute design and off-design performance
# 2019 - 2025 - Delft University of Technology - All rights reserved
#############################################################################

import CoolProp
import numpy as np
import scipy.optimize as opt
import TurboSim.geometry as geo
import TurboSim.flow_model as flow


class Volute:
    def __init__(self, loss_model, verbose, geometry: geo.Volute, inlet: flow.FlowNode, outlet: flow.FlowNode):
        """
        :param loss_model: string identifying the loss model to be used
        :param verbose: 0 --> minimal print to screen
                        1 --> print to screen also intermediate results
                        2 --> print to screen all the warnings and intermediate results (debugging)
        :param inlet: instance of FlowNode class identifying the inlet port
        :param outlet: instance of FlowNode class identifying the outlet port
        :param geometry: instance of Volute class defined in geometry module
        """
        self.error = False
        self.inlet = inlet
        self.outlet = outlet
        self.geometry = geometry
        self.loss_model = loss_model
        self.verbose = verbose
        self.dPt = 0.0
        self.dht = 0.0
        self.ds = 0.0
        self.Cp = 0.0

        if self.inlet.n_mid != self.outlet.n_mid:
            raise ValueError("The number of span-wise sections at the inlet and outlet nodes must match!")
        else:
            self.n_mid = self.inlet.n_mid

    def reset(self):
        raise NotImplementedError("Reset method has not been implemented yet")

    def outlet_volute_design(self, p):
        """
        Design external volute, once fixed the impeller and diffuser design.
        Design strategy: no pressure gradient in the circumferential direction.
        Assumptions: no friction, no flow disturbance from tongue.
        Reference: M. Casey, and C. Robinson - Radial Flow Turbocompressors:
            Design, Analysis, and Applications, 2021., pag. 449
        """
        if p <= self.geometry.R_in:
            res = 10
        else:
            # volute geometry
            self.geometry.Rc_vol[-1] = p
            self.geometry.set_outlet_geometry()

            # conservation of moment of momentum
            self.outlet.V[self.n_mid] = self.inlet.Vt[self.n_mid] * self.geometry.R_in / self.geometry.Rc_vol[-1]

            # conservation of mass
            self.outlet.mf = self.inlet.mf
            self.outlet.D[self.n_mid] = self.outlet.mf / (self.outlet.V[self.n_mid] * self.geometry.A_vol[-1])

            # conservation of energy
            self.outlet.ht[self.n_mid] = self.inlet.ht[self.n_mid]

            if self.loss_model == 'isentropic':
                self.outlet.Pt[self.n_mid] = self.inlet.Pt[self.n_mid]
                self.outlet.s[self.n_mid] = self.inlet.s[self.n_mid]
            else:
                # compute losses in volute and total outlet thermodynamic state
                self.volute_loss('collector')
                self.outlet.Pt[self.n_mid] = self.inlet.Pt[self.n_mid] - self.dPt
                self.outlet.EoS.update(CoolProp.HmassP_INPUTS, self.outlet.ht[self.n_mid], self.outlet.Pt[self.n_mid])
                self.outlet.s[self.n_mid] = self.outlet.EoS.smass()

            # re-compute exit density and compute residual
            self.outlet.h[self.n_mid] = self.outlet.ht[self.n_mid] - self.outlet.V[self.n_mid] ** 2 / 2
            self.outlet.EoS.update(CoolProp.HmassSmass_INPUTS, self.outlet.h[self.n_mid], self.outlet.s[self.n_mid])
            res = (self.outlet.D[self.n_mid] - self.outlet.EoS.rhomass()) / self.outlet.D[self.n_mid]

        return res

    def inlet_volute_design(self, p):
        """
        Design inlet volute for radial inflow turbine.
        Design strategy: no pressure gradient in the circumferential direction. The design of the volute as a distributor
        is assumed to be iterative once the stator inlet geometry is known. The design starts from the outlet to the inlet.
        Assumptions: no friction, no flow disturbance from tongue.
        Index 0 is at volute inlet
        Reference: M. Casey, and C. Robinson - Radial Flow Turbocompressors:
            Design, Analysis, and Applications, 2021., pag. 449
        """
        if p <= self.geometry.R_out:
            res = 10
        else:
            # volute geometry
            self.geometry.Rc_vol[0] = p
            self.geometry.set_inlet_geometry()

            # conservation of moment of momentum
            self.inlet.V[self.n_mid] = self.outlet.Vt[self.n_mid] * self.geometry.R_out / self.geometry.Rc_vol[0]

            # conservation of mass
            self.inlet.mf = self.outlet.mf
            self.inlet.D[self.n_mid] = self.inlet.mf / (self.inlet.V[self.n_mid] * self.geometry.A_vol[0])

            # conservation of energy
            self.inlet.ht[self.n_mid] = self.outlet.ht[self.n_mid]

            if self.loss_model == 'isentropic':
                self.inlet.Pt[self.n_mid] = self.outlet.Pt[self.n_mid]
                self.inlet.s[self.n_mid] = self.outlet.s[self.n_mid]
            else:
                # compute losses in volute and total inlet thermodynamic state
                self.volute_loss('distributor')
                self.inlet.Pt[self.n_mid] = self.outlet.Pt[self.n_mid] + self.dPt
                self.inlet.EoS.update(CoolProp.HmassP_INPUTS, self.inlet.ht[self.n_mid], self.inlet.Pt[self.n_mid])
                self.inlet.s[self.n_mid] = self.inlet.EoS.smass()

            # re-compute inlet density and compute residual
            self.inlet.h[self.n_mid] = self.inlet.ht[self.n_mid] - self.inlet.V[self.n_mid] ** 2 / 2
            self.inlet.EoS.update(CoolProp.HmassSmass_INPUTS, self.inlet.h[self.n_mid], self.inlet.s[self.n_mid])
            res = (self.inlet.D[self.n_mid] - self.inlet.EoS.rhomass()) / self.inlet.D[self.n_mid]

        return res

    def outlet_volute_off_design(self, p):
        """
        Compute real flow quantities at volute outlet given the volute geometry.
        Same as design point computation, without using the conservation of moment of momentum.
        """
        V = p

        # conservation of energy
        self.outlet.ht[self.n_mid] = self.inlet.ht[self.n_mid]

        if self.loss_model == 'isentropic':
            self.outlet.Pt[self.n_mid] = self.inlet.Pt[self.n_mid]
            self.outlet.s[self.n_mid] = self.inlet.s[self.n_mid]
        else:
            # compute losses in volute and total outlet thermodynamic state
            self.volute_loss('collector')
            self.outlet.Pt[self.n_mid] = self.inlet.Pt[self.n_mid] - self.dPt
            self.outlet.EoS.update(CoolProp.HmassP_INPUTS, self.outlet.ht[self.n_mid], self.outlet.Pt[self.n_mid])
            self.outlet.s[self.n_mid] = self.outlet.EoS.smass()

        # conservation of mass
        self.outlet.mf = self.inlet.mf
        self.outlet.D[self.n_mid] = self.outlet.mf / (V * self.geometry.A_vol[-1])

        # re-compute exit density and compute residual
        self.outlet.h[self.n_mid] = self.outlet.ht[self.n_mid] - V ** 2 / 2
        self.outlet.EoS.update(CoolProp.HmassSmass_INPUTS, self.outlet.h[self.n_mid], self.outlet.s[self.n_mid])
        res = (self.outlet.D[self.n_mid] - self.outlet.EoS.rhomass()) / self.outlet.D[self.n_mid]

        return res

    def inlet_volute_off_design(self, p):
        """
        Compute real flow quantities at volute outlet given the volute geometry.
        Opposite of design point computation as the off-design goes from inlet to outlet, without using the conservation
        of moment of momentum.
        """
        V = p

        # conservation of energy
        self.outlet.ht[self.n_mid] = self.inlet.ht[self.n_mid]

        if self.loss_model == 'isentropic':
            self.outlet.Pt[self.n_mid] = self.inlet.Pt[self.n_mid]
            self.outlet.s[self.n_mid] = self.inlet.s[self.n_mid]
        else:
            # compute losses in volute and total outlet thermodynamic state
            self.volute_loss('distributor')
            self.outlet.Pt[self.n_mid] = self.inlet.Pt[self.n_mid] - self.dPt
            self.outlet.EoS.update(CoolProp.HmassP_INPUTS, self.outlet.ht[self.n_mid], self.outlet.Pt[self.n_mid])
            self.outlet.s[self.n_mid] = self.outlet.EoS.smass()

        # conservation of mass
        self.outlet.mf = self.inlet.mf
        self.outlet.D[self.n_mid] = self.outlet.mf / (V * self.geometry.A_vol[-1])

        # re-compute exit density and compute residual
        self.outlet.h[self.n_mid] = self.outlet.ht[self.n_mid] - V ** 2 / 2
        self.outlet.EoS.update(CoolProp.HmassSmass_INPUTS, self.outlet.h[self.n_mid], self.outlet.s[self.n_mid])
        res = (self.outlet.D[self.n_mid] - self.outlet.EoS.rhomass()) / self.outlet.D[self.n_mid]

        return res

    def outlet_volute_flow_properties(self, flag: str):
        """
        Set the volute geometry and compute the thermo-flow quantities at the outlet accounting for entropy generation
        :param flag: string used to identify design or off-design computation
        """
        self.inlet.mf = (self.inlet.D[self.n_mid] * self.inlet.Vm[self.n_mid] *
                         2 * np.pi * self.geometry.R_in * self.geometry.H)
        try:
            match flag:
                case 'design':
                    # set volute geometry
                    opt.fsolve(self.outlet_volute_design, (1.1 * self.geometry.R_in), full_output=False, xtol=1.0e-06)
                case 'off-design':
                    self.outlet.V[self.n_mid] = opt.fsolve(self.outlet_volute_off_design, (self.inlet.Vm[self.n_mid]),
                                                           full_output=False, xtol=1.0e-06)
                case _:
                    NotImplementedError("The supported entries for parameter 'flag' are 'design' "
                                        "and 'off-design'")

            # update outlet thermodynamic quantities
            self.outlet.update_total_thermodynamic_state('Ph_const')
            self.outlet.update_static_thermodynamic_state('Ds_const')

            # compute dht_volute for loss breakdown purposes and ancillary properties
            self.outlet.EoS.update(CoolProp.PSmass_INPUTS, self.outlet.Pt[self.n_mid], self.inlet.s[self.n_mid])
            self.dht = self.outlet.ht[self.n_mid] - self.outlet.EoS.hmass()

            # compute ancillary properties
            self.outlet.MachAbs[self.n_mid] = self.outlet.V[self.n_mid] / self.outlet.c[self.n_mid]
            self.outlet.MachMer[self.n_mid] = self.outlet.MachAbs[self.n_mid]
            self.ds = self.outlet.s[self.n_mid] - self.inlet.s[self.n_mid]
            self.Cp = ((self.outlet.P[self.n_mid] - self.inlet.P[self.n_mid]) /
                       (self.inlet.Pt[self.n_mid] - self.inlet.P[self.n_mid]))

            # check minimum threshold for pressure recovery in the volute
            match flag:
                case 'design':
                    if self.Cp < 0:
                        raise ValueError("Negative pressure recovery in the volute at design point")
                case 'off-design':
                    if self.Cp < -0.5:
                        raise ValueError("Pressure recovery in the volute is below the minimum allowable value")
                case _:
                    NotImplementedError("The supported entries for parameter 'flag' are 'design' "
                                        "and 'off-design'")
        except ValueError:
            self.error = True
            if self.verbose == 2:
                print("Error in Volute class, method: outlet_volute_flow_properties")

    def inlet_volute_flow_properties(self, flag: str):
        """
        Set the volute geometry and compute the thermo-flow quantities at the outlet accounting for entropy generation
        :param flag: string used to identify design or off-design computation
        """

        try:
            if flag == 'design':
                # calculate mass flow at outlet
                self.outlet.mf = (self.outlet.D[self.n_mid] * self.outlet.Vm[self.n_mid] *
                                  2 * np.pi * self.geometry.R_out * self.geometry.H)
                # set volute geometry
                opt.fsolve(self.inlet_volute_design, (1.2 * self.geometry.R_out), full_output=False, xtol=1.0e-09)

            elif flag == 'off-design':
                # calculate mass flow at inlet
                self.inlet.mf = self.inlet.D[self.n_mid] * self.inlet.V[self.n_mid] * self.geometry.A_vol[0]
                self.outlet.V[self.n_mid] = opt.fsolve(self.inlet_volute_off_design, (self.inlet.V[self.n_mid]),
                                                       full_output=False, xtol=1.0e-06)

            else:
                raise ValueError("The valid entries for flag are 'design' or 'off-design'")

            # update outlet thermodynamic quantities
            self.outlet.update_total_thermodynamic_state('Ph_const')
            self.outlet.update_static_thermodynamic_state('Ds_const')

            # compute dht_volute for loss breakdown purposes and ancillary properties
            self.outlet.EoS.update(CoolProp.PSmass_INPUTS, self.outlet.Pt[self.n_mid], self.inlet.s[self.n_mid])
            self.dht = self.outlet.ht[self.n_mid] - self.outlet.EoS.hmass()

            # compute ancillary properties
            self.outlet.MachAbs[self.n_mid] = self.outlet.V[self.n_mid] / self.outlet.c[self.n_mid]
            self.outlet.MachMer[self.n_mid] = self.outlet.MachAbs[self.n_mid]
            self.ds = self.outlet.s[self.n_mid] - self.inlet.s[self.n_mid]
            self.Cp = ((self.outlet.P[self.n_mid] - self.inlet.P[self.n_mid]) /
                       (self.inlet.Pt[self.n_mid] - self.inlet.P[self.n_mid]))

            # check minimum threshold for pressure recovery in the volute
            if flag not in ['design', 'off-design']:
                raise ValueError("The valid entries for flag are 'design' or 'off-design'")

        except ValueError:
            self.error = True
            if self.verbose == 2:
                print("Error in Volute class, method: inlet_volute_flow_properties")

    def volute_loss(self, flag: str, f1=0.8, f2=0.8):
        """
        Compute losses in an overhung collector [1] or distributor [2] volute, considering adiabatic compressible flow.
        References:
        [1] D. Japikse - Centrifugal Compressor Design and Performance, 1996.
        [2] A. Whitfield, and N. C. Baines - Design of Radial Turbomachines, 1990.
        :param flag: "collector" or "distributor"
        :param f1: coefficient used when part of the inlet meridional velocity can be used in the exit cone: [0.6, 1]
        :param f2: coefficient used for external volute featuring R_vol_center >> R2: [0.5, 1]
        :return dPt: total pressure drop across the volute
        """
        match flag:
            case 'collector':
                swirl = self.inlet.Vt[self.n_mid] / self.inlet.Vm[self.n_mid]
                k_m = f1 / (1 + swirl ** 2)
    
                if (self.geometry.vol_AR * swirl) > 1:
                    k_theta = (f2 * ((self.geometry.R_in / self.geometry.Rc_vol[-1]) ** 2) * 
                               ((swirl - 1 / self.geometry.vol_AR) ** 2) / (1 + swirl ** 2))
                else:
                    k_theta = 0.0
    
                self.dPt = (k_m + k_theta) * (self.inlet.Pt[self.n_mid] - self.inlet.P[self.n_mid])

            case 'distributor':
                Cf = 0.023  # Colebrook-White friction flow coefficient from Moody chart
                theta = np.linspace(1, 1 / len(self.geometry.Rc_vol), len(self.geometry.Rc_vol))
                L = np.sum(self.geometry.Rc_vol * np.cos(theta))
    
                # update static quantities to get the speed of sound for inlet Mach Number calculations
                self.inlet.h[self.n_mid] = self.inlet.ht[self.n_mid] - self.inlet.V[self.n_mid] ** 2 / 2
                self.inlet.update_static_thermodynamic_state('Dh_const')
                self.inlet.MachAbs[self.n_mid] = self.inlet.V[self.n_mid] / self.inlet.c[self.n_mid]
    
                self.outlet.MachAbs[self.n_mid] = self.outlet.V[self.n_mid] / self.outlet.c[self.n_mid]
    
                delta_s = self.inlet.R * ((1 / (self.inlet.gamma[self.n_mid] - 1)) *
                                          (1 / self.inlet.MachAbs[self.n_mid] ** 2 - 1 /
                                           self.outlet.MachAbs[self.n_mid] ** 2) +
                                          (1 / (self.inlet.gamma[self.n_mid] - 1)) *
                                          np.log(self.inlet.MachAbs[self.n_mid] ** 2 /
                                                 self.outlet.MachAbs[self.n_mid] ** 2) -
                                          (self.inlet.gamma[self.n_mid] / (self.inlet.gamma[self.n_mid] - 1)) *
                                          4 * Cf * L / (2 * np.mean(self.geometry.R_vol)))
    
                Pt_out = self.inlet.Pt[self.n_mid] * np.exp(-delta_s / self.inlet.R)
                self.dPt = Pt_out - self.inlet.Pt[self.n_mid]
            case _:
                raise NotImplementedError("The supported entries for parameter 'flag' are 'collector' "
                                          "and 'distributor'")
