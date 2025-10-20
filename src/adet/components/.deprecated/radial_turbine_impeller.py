# fmt: off
import os
import re
import CoolProp
import CoolProp.CoolProp as cp
import numpy as np
import scipy.optimize as opt
import time
import copy
from TurboSim.tools.interp import w_extrap

class RadialTurbineImpeller:
    def __init__(self, EoS, library, fluid, verbose,
                 massflow, psi, phi=0.35, phi_in=0.25, mu=0.0, omega=0.0,
                 alpha_out=0.0,
                 R_ratio=0.55, Rh_Rs_out=0.3, Lax_dR=0.9,
                 cone_in=90.0, cone_out=0.0,
                 t_p_in=0.015, t_p_out=0.015,
                 g_b_in=0.1, g_b_out=0.1, g_b_bf=0.1,
                 Nbl=0, Nbl_model='glassman', slip_model='chen',
                 vortex_model='free vortex',
                 pf_loss_model='baines', te_wake_loss_model='baumgartner',
                 mx_loss_model='osnaghi', sh_loss_model='rh real',
                 sc_loss_model='none', ic_loss_model='baines',tl_loss_model='baines',
                 shock_angle=None,
                 gw_coef=[0,0,0],
                 n_span=5, n_stream=10,
                 inlet=None, outlet=None, geometry=None):
        """
        :param EoS:             instance of CoolProp.AbstractState class
        :param library:         string identifying the thermodynamic model used in the EoS object
        :param fluid:           string identifying the prescribed working fluid
        :param massflow:        massflow rate in kg/s (float)
        :param psi:             real work coefficient defined as (ht_in - ht_out) / U_in^2 (float)
        :param phi:             real flow coefficient defined as Vm_out / U_in (float)
                                used by default, not used if both 'phi_in' and 'mu' are specified
                                default: 0.35
        :param phi_in:          real inlet flow coefficient defined as Vm_in / U_in (float)
                                used by default instead of 'mu', not used if 'mu' specified
                                default: 0.25
        :param mu:              real meridional velocity ratio defined as Vm_out / Vm_in (float)
                                not used if not specified; if specified, set either 'phi' or 'phi_in' to zero
                                default: 0.0
        :param omega:           rotational speed in rad/s (float)
                                not used if not specified or set = 0
                                default: 0.0
        :param alpha_out:       real absolute swirl at outlet in degrees (float)
                                default: 0.0
        :param R_ratio:         outlet / inlet radius ratio (float)
                                default: 0.55
        :param Rh_Rs_out:       hub / shroud radius ratio at exducer (float)
                                default: 0.3
        :param Lax_dR:          axial length to in-out radius change impeller (float)
                                default: 0.9
        :param cone_in:         cone angle at the inducer in degrees (float)
                                default: 90.0
        :param cone_out:        cone angle at the exducer in degrees (float)
                                default: 0.0
        :param t_p_in:          leading edge thickness / inducer pitch (float or list(n_span))
                                default: 0.015
        :param t_p_out:         trailing edge thickness / inducer pitch  (float or list(n_span))
                                default: 0.015
        :param g_b_in:          leading edge tip clearance / inducer blade height (float)
                                default: 0.1
        :param g_b_out:         trailing edge tip clearance / exducer blade height (float)
                                default: 0.1
        :param g_b_bf:          disk backface clearance / inducer blade height (float)
                                default: 0.1
        :param Nbl:             number of blades (int)
                                if not specified, the number of blades is determined by Nbl_model
                                default: 0
        :param Nbl_model:       string that identifies the model for the determination of the number of blades
                                options: 'glassman' (default), 'jamieson'
        :param slip_model:      string that identifies the model for the determination of the flow slip factor
                                options: 'chen' (default), 'stanitz'
        :param vortex_model:    string that identifies the model for the determination of the flow vortex distribution
                                options: 'free vortex' (default), 'forced vortex', 'constant angle', 'general whirl'
        :param pf_loss_model:   string that identifies the model for the computation of the blade profile losses
                                options: 'baines' (default), 'vdb', 'denton', 'glassman', 'none'
        :param te_wake_loss_model:   string that identifies the model for the computation of the trailing edge wake losses
                                options: 'baumgartner' (default), 'denton', 'none'
        :param mx_loss_model:   string that identifies the model for the computation of the cascade mixing losses
                                options: 'osnaghi' (default), 'none'
        :param sh_loss_model:   string that identifies the model for the computation of the shock losses
                                options: 'rh real' (default), 'rh perfect', 'none'
        :param sc_loss_model:   string that identifies the model for the computation of the secondary flow and endwall
                                losses
                                N.B.: if pf_loss_model = 'baines' is used, secondary flow and endwall losses are
                                included as part of the blade profile losses
                                options: 'rodgers', 'none' (default)
        :param ic_loss_model:   string that identifies the model for the computation of the incidence losses
                                options: 'baines' (default), 'nasa', 'none'
        :param tl_loss_model:   string that identifies the model for the computation of the tip leakage losses
                                options: 'baines' (default), 'denton', 'none'
        :param shock_angle:     shock wave angle in degrees between 1 and 90 deg for shock loss computation (float)
                                default: None (the angle is computed automatically)
        :param gw_coef:         coefficients for the definition of general whirl vortex distribution (list(3))
                                N.B.: if general whirl is selected and the coefficients are not specified, these are
                                automatically optimized
                                default: [0, 0, 0]
        :param verbose:         0 --> minimal print to screen
                                1 --> print to screen also intermediate results
                                2 --> print to screen all the warnings and intermediate results (debugging)
        :param n_span:          number of spanwise slices to discretize impeller fluid domain (odd int)
                                default: 5
        :param n_stream:        number of streamwise sections to discretize impeller fluid domain (int)
                                default: 10
        :param inlet:           instance of FlowNode class identifying the inlet port
        :param outlet:          instance of FlowNode class identifying the outlet port
        :param geometry:        instance of Volute class defined in geometry module
        """

        # general settings
        self.eos = EoS
        self.library = library
        self.fluid = fluid
        self.verbose = verbose
        self.n_span = n_span
        self.n_mid = int((self.n_span - 1) / 2)

        # design variables
        self.mf = massflow
        self.psi = psi
        self.phi = phi
        self.phi_in = phi_in
        self.mu = mu
        self.om = omega
        if omega != 0:
            self.om_fixed = 1
        else:
            self.om_fixed = 0

        # flow modeling
        self.slip_model = slip_model
        self.vortex_model = vortex_model
        self.pf_loss_model = pf_loss_model
        self.te_wake_loss_model = te_wake_loss_model
        self.mx_loss_model = mx_loss_model
        self.sh_loss_model = sh_loss_model
        self.sc_loss_model = sc_loss_model
        self.ic_loss_model = ic_loss_model
        self.tl_loss_model = tl_loss_model
        self.gw_coef = gw_coef  # general whirl coefficients a, b, n

        # loss components
        self.ds = np.zeros((n_span, n_stream))
        self.ds_profile = np.zeros((n_span, n_stream))
        self.ds_te_wake = np.zeros((n_span, n_stream))
        self.ds_mixing = np.zeros((n_span, n_stream))
        self.ds_shock = np.zeros((n_span, n_stream))
        self.ds_secondary = np.zeros((n_span, n_stream))
        self.ds_incidence = np.zeros((n_span, n_stream))
        self.ds_leakage = np.zeros((n_span, n_stream))
        self.ds_endwall = np.zeros((n_span, n_stream))

        # ancillary properties (only for cascades)
        self.work = np.zeros(n_span)
        self.incidence_angle = np.zeros(n_span)
        self.deviation_angle = np.zeros(n_span)
        self.post_expansion = np.zeros(n_span)
        self.slip_factor = np.zeros(n_span)
        self.slip_angle = np.zeros(n_span)
        self.flag_choking = np.zeros(n_span)
        self.shock_angle = [None] * self.n_span
        self.angle_fixed = None
        self.Pb = np.zeros(n_span)
        self.CPb = np.zeros(n_span)
        # throat properties
        self.MachThroat = np.zeros(n_span)
        self.P_th = np.zeros(n_span)
        self.s_th = np.zeros(n_span)
        self.V_th = np.zeros(n_span)
        self.h_th = np.zeros(n_span)
        self.D_th = np.zeros(n_span)

        # inlet node
        if inlet is None:
            self.inlet = flow.FlowNode(EoS, library, fluid, n_span, 'inlet')
        else:
            self.inlet = inlet

        # outlet node
        if outlet is None:
            self.outlet = flow.FlowNode(EoS, library, fluid, n_span, 'outlet')
        else:
            self.outlet = outlet

        self.outlet.alpha[self.n_mid] = alpha_out

        # geometrical variables
        if geometry is None:
            self.geometry = geo.Geometry(n_span, n_stream)
            self.R_ratio = R_ratio
            self.Rh_Rs_out = Rh_Rs_out

            # initialize dimensionless quantities
            self.geometry.R[self.n_mid, 0] = 1
            self.geometry.R[self.n_mid, -1] = self.outlet.r = self.R_ratio * self.geometry.R[self.n_mid, 0]
            self.geometry.H[-1] = self.outlet.b = 2 * (1 - self.Rh_Rs_out) / (1 + self.Rh_Rs_out) * self.R_ratio
            self.geometry.Lax = Lax_dR * (1 - self.R_ratio)
            self.geometry.t[:, 0] = t_p_in
            self.geometry.t[:, -1] = t_p_out
            self.geometry.tip_clearance_in = g_b_in
            self.geometry.tip_clearance_out = g_b_out
            self.geometry.bf_clearance = g_b_bf

            # initialize meridional plane geometry
            self.geometry.set_radial_turbine_impeller_geometry()

        else:
            self.geometry = geometry

            # initialize dimensionless geometry
            self.R_ratio = self.geometry.R[self.n_mid, -1] / self.geometry.R[self.n_mid, 0]
            self.Rh_Rs_out = self.geometry.R[0, -1] / self.geometry.R[-1, -1]

        if Nbl != 0:
            self.Nbl_fixed = 1
            self.geometry.Nbl = Nbl
            self.Nbl_model = 'none'
        else:
            self.Nbl_fixed = 0
            self.Nbl_model = Nbl_model

        self.geometry.cone_angle[0] = cone_in
        self.geometry.cone_angle[-1] = cone_out

        # loss models parameters TODO: add references for loss modeling parameters
        self.loss_model_params = \
            {
                "Cd": 0.002,                        # BL dissipation coefficient (ref)
                "Kx": 0.4,                          # axial gap discharge coefficient for unshrouded blade (ref)
                "Kr": 0.75,                         # radial gap discharge coefficient for unshrouded blade (ref)
                "Kxr": -0.3,                        # cross-coupling coefficient for unshrouded blade (ref)
                "C_cs": 0.6,                        # discharge coefficient for shrouded blade (ref)
                "C_cu": 0.4,                        # discharge coefficient for unshrouded blade (relative motion) (ref)
                "d*_H": 0.05,                       # BL displacement thickness on endwalls (ref)
                "d*_th": 2.0,                       # BL shape factor for transonic turbine (ref)
                "th_t": 0.075,                      # BL mom thickness (ref)
                "M_rt_sh": 1.06,                    # Pre shock Mach to cascade downstream Mach number ratio
                "P_rt_sh": 0.92,                    # Pre shock pressure to cascade downstream pressure ratio
                "bld_parm": 16,                     # (eps + delta) / 2 parameter - see docs/modeling_notes/RITs_modeling.pdf - used for base pressure calculation based on (ref)
                "Rmx_Rte_rt": 1.0,                  # mixing plane to trailing edge radius ratio (set 1 for axial exducer, < 1 for radial or mixed flow exducer)
             }
        if shock_angle == None:
            self.angle_fixed = False
            # set starting value for shock angle
            self.shock_angle = np.array([25] * self.n_span)
        else:
            self.angle_fixed = True
            self.shock_angle = np.array([shock_angle] * self.n_span)

        # protected attributes
        self._mode = None   # stores the run mode defined by the user ("design" or "offdesign")
        self._error = False
        self._error_msg = None
        self._Vt_in_noslip = 0.0    # inlet absolute tangential velocity BEFORE flow slip occurs - design parameter for upstream component
        self._Wt_in_noslip = 0.0    # inlet relative tangential velocity BEFORE flow slip occurs - design parameter for upstream component
        self._alpha_in_noslip = 0.0 # inlet absolute angle BEFORE flow slip occurs - design parameter for upstream component
        self._beta_in_noslip = 0.0  # inlet relative angle BEFORE flow slip occurs - design parameter for upstream component
        # loss model flags: 0 if not requested by user or no error occurred, 1 if error occurred
        self._flag_profile = 0
        self._flag_te_wake = 0
        self._flag_mixing = 0
        self._flag_shock = 0
        self._flag_secondary = 0
        self._flag_incidence = 0
        self._flag_leakage = 0

        # load dataset to evaluate base pressure
        root_dir = re.sub('TurboSim', '', os.path.dirname(os.path.abspath(__file__)))
        self.loss_model_params["xq_conv"] = np.load(root_dir + '/data/turbine_base_pressure/Sieverding_conv_xq.npy')
        self.loss_model_params["yq_conv"] = np.load(root_dir + '/data/turbine_base_pressure/Sieverding_conv_yq.npy')
        self.loss_model_params["zq_conv"] = np.load(root_dir + '/data/turbine_base_pressure/Sieverding_conv_zq.npy')
        self.loss_model_params["xq_conv_div"] = np.load(root_dir + '/data/turbine_base_pressure/Sieverding_conv-div_xq.npy')
        self.loss_model_params["yq_conv_div"] = np.load(root_dir + '/data/turbine_base_pressure/Sieverding_conv-div_yq.npy')
        self.loss_model_params["zq_conv_div"] = np.load(root_dir + '/data/turbine_base_pressure/Sieverding_conv-div_zq.npy')

    def reset(self):
        raise NotImplementedError("Reset method has not been implemented yet")

    def reset_losses(self):
        # reset entropy generation
        self.ds[:, :] = 0.0
        self.ds_profile[:, :] = 0.0
        self.ds_mixing[:, :] = 0.0
        self.ds_shock[:, :] = 0.0
        self.ds_secondary[:, :] = 0.0
        self.ds_incidence[:, :] = 0.0
        self.ds_leakage[:, :] = 0.0
        # reset flags
        self._flag_profile = 0
        self._flag_mixing = 0
        self._flag_shock = 0
        self._flag_secondary = 0
        self._flag_incidence = 0
        self._flag_leakage = 0

    def __del__(self):
        pass

    def print_summary(self):

        print("\n ------Results: meanline------- ")
        print(' Psi:\t%f' % ((self.inlet.ht[self.n_mid] - self.outlet.ht[self.n_mid]) / self.inlet.U[self.n_mid] ** 2))
        print(' Work:\t%f' % (self.work[self.n_mid]))
        print(' slip:\t%f' % (self.slip_factor))
        print(' rad/s:\t%f' % (self.om[self.n_mid]))
        print(' rpm:\t%f' % (self.om[self.n_mid] * 60 / (2 * np.pi)))
        print("\n INLET ")
        print(" Kinematic ")
        print(" U =\t%f" % (self.inlet.U[self.n_mid]))
        print(" Vm =\t%f" % (self.inlet.Vm[self.n_mid]))
        print(" Vt =\t%f" % (self.inlet.Vt[self.n_mid]))
        print(" V =\t%f" % (self.inlet.V[self.n_mid]))
        print(" alpha =\t%f" % (self.inlet.alpha[self.n_mid]))
        print(" Wm =\t%f" % (self.inlet.Wm[self.n_mid]))
        print(" Wt =\t%f" % (self.inlet.Wt[self.n_mid]))
        print(" W =\t%f" % (self.inlet.W[self.n_mid]))
        print(" beta =\t%f" % (self.inlet.beta[self.n_mid]))
        print(" angle =\t%f" % (self.geometry.blade_angle[self.n_mid, 0]))
        print("\n Thermodynamics (static) ")
        print(" P =\t%f" % (self.inlet.P[self.n_mid]))
        print(" T =\t%f" % (self.inlet.T[self.n_mid]))
        print(" s =\t%f" % (self.inlet.s[self.n_mid]))
        print(" h =\t%f" % (self.inlet.h[self.n_mid]))
        print("\n Thermodynamics (total) ")
        print(" Pt =\t%f" % (self.inlet.Pt[self.n_mid]))
        print(" Tt =\t%f" % (self.inlet.Tt[self.n_mid]))
        print(" ht =\t%f" % (self.inlet.ht[self.n_mid]))
        print("\n Thermodynamics (total relative) ")
        print(" Ptr =\t%f" % (self.inlet.Ptr[self.n_mid]))
        print(" Ttr =\t%f" % (self.inlet.Ttr[self.n_mid]))
        print(" htr =\t%f" % (self.inlet.htr[self.n_mid]))
        print(" R =\t%f" % (self.inlet.Rh[self.n_mid]))
        print("\n OUTLET ")
        print(" Kinematic ")
        print(" U =\t%f" % (self.outlet.U[self.n_mid]))
        print(" Vm =\t%f" % (self.outlet.Vm[self.n_mid]))
        print(" Vt =\t%f" % (self.outlet.Vt[self.n_mid]))
        print(" V =\t%f" % (self.outlet.V[self.n_mid]))
        print(" alpha =\t%f" % (self.outlet.alpha[self.n_mid]))
        print(" Wm =\t%f" % (self.outlet.Wm[self.n_mid]))
        print(" Wt =\t%f" % (self.outlet.Wt[self.n_mid]))
        print(" W =\t%f" % (self.outlet.W[self.n_mid]))
        print(" beta =\t%f" % (self.outlet.beta[self.n_mid]))
        print(" angle =\t%f" % (self.geometry.blade_angle[self.n_mid, -1]))
        print("\n Thermodynamics (static) ")
        print(" P =\t%f" % (self.outlet.P[self.n_mid]))
        print(" T =\t%f" % (self.outlet.T[self.n_mid]))
        print(" s =\t%f" % (self.outlet.s[self.n_mid]))
        print(" h =\t%f" % (self.outlet.h[self.n_mid]))
        print("\n Thermodynamics (total) ")
        print(" Pt =\t%f" % (self.outlet.Pt[self.n_mid]))
        print(" Tt =\t%f" % (self.outlet.Tt[self.n_mid]))
        print(" ht =\t%f" % (self.outlet.ht[self.n_mid]))
        print("\n Thermodynamics (total relative) ")
        print(" Ptr =\t%f" % (self.outlet.Ptr[self.n_mid]))
        print(" Ttr =\t%f" % (self.outlet.Ttr[self.n_mid]))
        print(" htr =\t%f" % (self.outlet.htr[self.n_mid]))
        print(" R =\t%f" % (self.outlet.Rh[self.n_mid]))
        print("\n GEOMETRY ")
        print(" Lmax =\t%f" % (np.max(self.geometry.z[:, -1])))
        print("\n Inlet ")
        print(" Rh =\t%f" % (self.geometry.R[0, 0]))
        print(" Rm =\t%f" % (self.geometry.R[self.n_mid, 0]))
        print(" Rs =\t%f" % (self.geometry.R[-1, 0]))
        print(" H =\t%f" % (self.geometry.H[0]))
        print("\n Outlet ")
        print(" Rh =\t%f" % (self.geometry.R[0, -1]))
        print(" Rm =\t%f" % (self.geometry.R[self.n_mid, -1]))
        print(" Rs =\t%f" % (self.geometry.R[-1, -1]))
        print(" H =\t%f" % (self.geometry.H[-1]))

    def set_meanline_design(self, bc_dict={}):
        """
        Set impeller design at meanline

        :param bc_dict: Keys and values to define the total thermodynamic conditions 
        :type bc_dict: dict

        See :meth:`~.set_boundary_conditions` for details on the input dictionary
        """

        if self.verbose >= 1:
            # tic
            t0 = time.time()
            error_msg = None

            print("\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ")
            print(" Radial turbine impeller design rotine")

        # set run mode
        self._mode = 'design'

        # set impeller BCs
        if bc_dict != {}:
            self.set_boundary_conditions(bc_dict)
        else:
            raise Exception ("Wrong or missing inputs. Make sure that all three values of the boundary conditions and the "
                             "corresponding keys identifiers are correctly specified.")

        # meanline computation
        try:
            self.meanline_solver()
        except:
            self._error = True
            self._error_msg = 'Error during meanline computation.'
            raise RuntimeError (self._error_msg)

        # set initial geometry
        self.set_size()

        if self.verbose >= 1:

            # toc
            t = time.time()

            print("\n ----------RunStats------------ ")
            print(" Runtime:\t\t\t%.5f s" % (t-t0))
            print(" Error:\t\t\t\t%s" % (self._error))
            print(" Error traceback:\t%s" % (self._error_msg))
            if self.verbose == 2:
                self.print_summary()
            print(" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ")

    def set_inlet(self, input_pair, values):
        """
        Set **total** conditions for the inlet :class:`~.FlowNode`

        :param input_pair: Type of input pair, e.g. ``'Ps'``, ``'Ts'``
        :type input_pair: string
        :param values: Values for the the input pair 
        :type values: list[float]
        :raises NotImplementedError: If input pair is not supported
        """

        match input_pair:
            case 'Ps':
                self.inlet.Pt[:] = values[0]
                self.inlet.s[:] = values[1]
            case 'Ts':
                self.inlet.Tt[:] = values[0]
                self.inlet.s[:] = values[1]
            case 'PT':
                self.inlet.Pt[:] = values[0]
                self.inlet.Tt[:] = values[1]
            case 'Ph':
                self.inlet.Pt[:] = values[0]
                self.inlet.ht[:] = values[1]
            case 'hs':
                self.inlet.ht[:] = values[0]
                self.inlet.s[:] = values[1]
            case 'Dh':
                self.inlet.Dt[:] = values[0]
                self.inlet.ht[:] = values[1]
            case 'Ds':
                self.inlet.Dt[:] = values[0]
                self.inlet.s[:] = values[1]
            case _:
                raise NotImplementedError("The specified input pair is not supported yet")

        # update all other total quantities on the node
        if self._mode == 'design':
            self.inlet.update_total_thermodynamic_state(input_pair + '_const')
        else:
            self.inlet.update_total_thermodynamic_state(input_pair)

    def set_outlet(self, input, value):
        """
        Set and update the **static** conditions for the outlet :class:`~.FlowNode`

        :param input: Outlet thermodynamic quantity, ``'P'``, ``'T'``, ``'h'``, ``'D'`` ( :math:`\\rho` )
        :type input: string
        :param values: Values for the the input pair 
        :type values: float
        :raises NotImplementedError: If input pair is not supported

        .. attention::

           This also sets :math:`s_{in} = s_{out}`, and uses this to set the
           thermodynamic state, together with the given variable (e.g. ``Ps``) 
        """

        self.outlet.s[:] = self.inlet.s[:]

        match input:
            case 'P':
                self.outlet.P[:] = value
                input_pair = 'Ps'
            case 'T':
                self.outlet.T[:] = value
                input_pair = 'Ts'
            case 'h':
                self.outlet.h[:] = value
                input_pair = 'hs'
            case 'D':
                self.outlet.D[:] = value
                input_pair = 'Ds'
            case _:
                raise NotImplementedError("The specified input pair is not supported yet")

        # update all other static quantities on the node
        self.outlet.update_static_thermodynamic_state(input_pair + '_const')

    def set_boundary_conditions(self, bc_dict):
        """
        Wrapper for the :meth:`set_inlet` and :meth:`set_outlet` methods.

        :param bc_dict: Dictionary of the boundary conditions
        :type: dict

        - Position 0 = Inlet (Total conditions)
        - Position 1 = Oulet (Static conditions)

        e.g.: ``bc_dict = {'Ps':[[None], [None]], 'P':[None]}``
        default = ``{}``

        .. hint::

                Allowed entries for inlet key are ``Ps``, ``Ts``, ``PT``,
                ``Ph``, ``hs``, ``Dh``, ``Ds``.

                Allowed entries for outlet key are ``P``, ``D``, ``T``, ``h``.

                Outlet values must be specified as lists of one element
        """

        keys = list(bc_dict.keys())
        values = list(bc_dict.values())

        # inlet node
        self.set_inlet(keys[0], values[0])

        # outlet node
        self.set_outlet(keys[1], values[1])

    def meanline_solver(self):
        """
        Iterative solver for the meanline problem

        The residual is checked as

        .. math::

            \\text{res} = \\frac{h_t^{(i)} - h_t^{(i-1)}}{h_t^{(i-1)}}
        """

        if self.verbose >= 1:

            # tic
            t0 = time.time()
            error_msg = None

            print("\n =============================== ")
            print("        Meanline solver          ")

            if self.verbose == 2:
                print("\n ------ConvergenceMonitor------ ")
                print(" iter\t\tres ")


        it = 0
        res = 10
        tol = 1e-6
        max_it = 20

        self.outlet.ht[self.n_mid] = self.outlet.h[self.n_mid]

        while res > tol and it < max_it:
            ht_old = self.outlet.ht[self.n_mid]

            # set peripheral speed
            self.set_peripheral_speed()

            # set meanline thermo-flow quantities
            self.set_meanline_thermoflow_quantities()

            res = abs(ht_old - self.outlet.ht[self.n_mid]) / ht_old
            it += 1

            if self.verbose == 2:
                print(" %d\t\t\t%f " % (it, res))

        if it == max_it:
            error_msg = (" Maximum iteration limit (%d) reached in meanline_solver" % (max_it))

        if self.verbose >= 1:

            # toc
            t = time.time()

            print("\n ------------------------------ ")
            print(" Runtime:\t\t\t%.5f s" % (t - t0))
            print(" Error traceback:\t%s" % (error_msg))
            print(" =============================== ")

    def set_spanwise_design(self, inlet=None):
        """ Spanwise solver: set spanwise design considering losses

        :param inlet: Inlet flow state of the impeller
        :type inlet: :class:`~.FlowNode`

        **Inlet distribution behaviour**

        .. tab:: ``inlet = None``

          Distribution of **inlet** thermodynamic quantities 
          is set equal to the one at midspan, i.e. 

          .. math::
            h_{t,j} = h_{t,mid} \\qquad
            s_{j} = s_{mid} \\qquad \\forall j

        .. tab:: ``inlet`` specified

          *Not implemented*
        """

        if self.verbose >= 1:

            # tic
            t0 = time.time()
            error_msg = None

            print("\n =============================== ")
            print("        Spanwise solver          ")

        try:
            if self._mode == 'design' and inlet == None:

                # set vortex distribution at the inlet node
                self.inlet.ht[:] = self.inlet.ht[self.n_mid]
                self.inlet.s[:] = self.inlet.s[self.n_mid]
                self.inlet.U[:] = self.inlet.U[self.n_mid] * self.geometry.R[:, 0] / self.geometry.R[self.n_mid, 0]
                self.inlet.set_vortex_distribution('free vortex', self.geometry)

                # set vortex distribution at the outlet node
                self.outlet.Rh[:] = self.inlet.Rh[:] # Rothalpy
                self.outlet.s[:] = self.inlet.s[:]
                self.outlet.U[:] = self.outlet.U[self.n_mid] * self.geometry.R[:, -1] / self.geometry.R[self.n_mid, -1]
                self.outlet.set_vortex_distribution(self.vortex_model, self.geometry,
                                                    self.gw_coef[0], self.gw_coef[1], self.gw_coef[2],
                                                    isentropic=True)
                self.outlet.update_isentropic_thermodynamic_state(self.inlet.s)

                # update geometry
                self.update_geometry()

                # duplicate instance of the impeller class to store isentropic geometry information
                self.geom_is = copy.deepcopy(self.geometry)

                # set impeller losses
                it, max_it = self.set_losses()

                if it == max_it:
                    error_msg = (" Maximum iteration limit (%d) reached in loss routine" % (max_it))

            elif self._mode == 'design' and inlet != None:
                raise NotImplementedError ('Feature not implemented yet.')

            else:
                raise NotImplementedError ('Feature not implemented yet.')

        except:
            self._error = True
            self._error_msg = 'Error during spanwise computation.'
            raise RuntimeError (self._error_msg)



        if self.verbose >= 1:

            # toc
            t = time.time()

            print("\n ------------------------------ ")
            print(" Runtime:\t\t\t%.5f s" % (t - t0))
            print(" Error traceback:\t%s" % (error_msg))
            print(" Loss error:")
            print(" Profile:\t%s"   % (self._flag_profile.__bool__()))
            print(" Mixing:\t%s"    % (self._flag_mixing.__bool__()))
            print(" Shock:\t\t%s"     % (self._flag_shock.__bool__()))
            print(" Secondary:\t%s" % (self._flag_secondary.__bool__()))
            print(" Incidence:\t%s" % (self._flag_incidence.__bool__()))
            print(" Leakage:\t%s"   % (self._flag_leakage.__bool__()))
            print(" =============================== ")
            if self.verbose == 2:
                self.print_summary()
            print(" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ")

    def set_peripheral_speed(self):
        """
        Set the peripheral speed at the rotor tip section of the machine
        """

        if self._mode == 'design':
            self.inlet.U[self.n_mid] = np.sqrt((self.inlet.ht[self.n_mid] - self.outlet.ht[self.n_mid]) / self.psi)
            self.outlet.U[self.n_mid] = self.geometry.R[self.n_mid, -1] / self.geometry.R[self.n_mid, 0] * self.inlet.U[self.n_mid]
        else:
            self.inlet.U[self.n_mid] = self.geometry.R[self.n_mid, 0] * self.omega
            self.outlet.U[self.n_mid] = self.geometry.R[self.n_mid, -1] / self.geometry.R[self.n_mid, 0] * self.inlet.U[self.n_mid]
            self.psi = (self.inlet.ht[self.n_mid] - self.outlet.ht[self.n_mid]) / self.inlet.U[self.n_mid] ** 2

    def set_flow_slip(self):
        """
        Loop for calculating the optimal rotor inlet flow angle 
        for incidence loss reduction according to slip factor theory. 
        The model has been implemented following [ref]??
        """

        # TODO: add references to description

        beta_old = 1000
        res = 10
        it = 0
        max_it = 20
        tol = 1e-6
        while res > tol and it < max_it:
            self.set_incidence_factor()
            DVt = self.inlet.U[self.n_mid] * (1 - self.slip_factor)
            DWt = DVt

            if self._mode == 'design':
                self._Vt_in_noslip = self.inlet.Vt[self.n_mid] - DVt
                self._Wt_in_noslip = self.inlet.Wt[self.n_mid] - DWt

            if self._mode == 'offdesign':
                self._Vt_in_noslip = self.inlet.Vt[self.n_mid] + DVt
                self._Wt_in_noslip = self.inlet.Wt[self.n_mid] + DWt

            self._alpha_in_noslip = np.degrees(np.arctan(self._Vt_in_noslip / self.inlet.Vm[self.n_mid]))
            self._beta_in_noslip = np.degrees(np.arctan(self._Wt_in_noslip / self.inlet.Wm[self.n_mid]))

            res = abs(self._beta_in_noslip - beta_old) / abs(beta_old)
            beta_old = self._beta_in_noslip

            it += 1

    def set_incidence_factor(self):
        """
        If the number of blades ``Nbl`` is not set, compute
        the number of blades, then compute the slip factor.
        """

        if self.Nbl_fixed == 0:

            # calculate the number of blades
            if self.Nbl_model == 'jamieson':
                self.geometry.Nbl = 2 * np.pi * np.tan(np.radians(self.inlet.alpha[self.n_mid]))

            if self.Nbl_model == 'glassman':
                self.geometry.Nbl = np.pi / 30 * (110 - self.inlet.alpha[self.n_mid]) * np.tan(np.radians(self.inlet.alpha[self.n_mid]))

        if self.slip_model == 'stanitz':
            self.slip_factor = (1 - 0.63 * np.pi / self.geometry.Nbl *
                                np.sin(np.radians(self.geometry.cone_angle[0])) * np.cos(np.radians(self.geometry.blade_angle[self.n_mid, 0])))

        if self.slip_model == 'chen':
            self.slip_factor = (1 - 2 * np.sin(np.pi / self.geometry.Nbl) / (np.pi * (1 + np.sin(np.pi / self.geometry.Nbl))) *
                                np.sin(np.radians(self.geometry.cone_angle[0])) * np.cos(np.radians(self.geometry.blade_angle[self.n_mid, 0])))

    def set_meanline_thermoflow_quantities(self):
        """
        Set the meanline thermodynamic and flow quantities under isentropic conditions
        """

        # compute inlet and outlet meridional velocities based on the chosen set of dimensionless variables
        if (self.phi_in and self.mu) != 0 and self.phi == 0:

            # user specified phi_in and mu
            self.inlet.Vm[self.n_mid] = self.phi_in * self.inlet.U[self.n_mid]
            self.outlet.Vm[self.n_mid] = self.inlet.Vm[self.n_mid] * self.mu

        elif (self.phi and self.mu) != 0 and self.phi_in == 0:

            # user specified phi and mu
            self.outlet.Vm[self.n_mid] = self.phi * self.inlet.U[self.n_mid]
            self.inlet.Vm[self.n_mid] = self.outlet.Vm[self.n_mid] / self.mu

        elif (self.phi_in and self.phi) != 0 and self.mu == 0:

            # user specified phi_in and phi
            self.inlet.Vm[self.n_mid] = self.phi_in * self.inlet.U[self.n_mid]
            self.outlet.Vm[self.n_mid] = self.phi * self.inlet.U[self.n_mid]

        elif (self.phi_in and self.phi and self.mu) != 0:
            raise Exception('User input error, too many inputs specified. Please specify ONLY TWO of the following parameters: phi_in = Vm_in / U_in, phi = Vm_out / U_out, mu = Vm_out / Vm_in.')

        else:
            raise Exception ('User input error, please specify a combination of 2 among the following parameters: phi_in = Vm_in / U_in, phi = Vm_out / U_out, mu = Vm_out / Vm_in.')

        # compute absolute swirl at inlet and outlet
        self.outlet.Vt = self.outlet.Vm * np.tan(np.radians(self.outlet.alpha))
        self.psi_out = self.outlet.Vt[self.n_mid] / self.outlet.U[self.n_mid]
        self.psi_in = self.psi + self.geometry.R[self.n_mid, -1] / self.geometry.R[self.n_mid, 0] * self.psi_out
        self.inlet.Vt = self.psi_in * self.inlet.U

        # initialize velocity triangles at the meanline
        self.inlet.update_velocity_triangles('VmVtU')
        self.outlet.update_velocity_triangles('VmVtU')

        # initialize metal angles at the meanline
        self.geometry.blade_angle[self.n_mid, 0] = self.inlet.beta[self.n_mid]
        self.geometry.blade_angle[self.n_mid, -1] = self.outlet.beta[self.n_mid]

        # include effect of flow slip
        self.set_flow_slip()

        # solve energy conservation
        self.inlet.solve_energy_conservation('stationary', 'h')
        self.outlet.solve_energy_conservation('stationary', 'ht')
        self.inlet.solve_energy_conservation('rotating', 'Rh')
        self.outlet.solve_energy_conservation('rotating', 'Rh')
        self.inlet.solve_energy_conservation('rotating', 'htr')
        self.outlet.solve_energy_conservation('rotating', 'htr')

        # initialize thermodynamic quantities at meanline
        self.inlet.update_static_thermodynamic_state('hs_const')
        self.outlet.update_static_thermodynamic_state('hs_const')
        self.inlet.update_total_thermodynamic_state('hs_const')
        self.outlet.update_total_thermodynamic_state('hs_const')
        self.inlet.update_total_relative_thermodynamic_state('hs_const')
        self.outlet.update_total_relative_thermodynamic_state('hs_const')

        # compute work
        self.work = (self.inlet.ht - self.outlet.ht)

    def set_size(self):
        """
        Sizes the turbomachinery inlet height based on mass conservation
        and specified parameters. Converts non-dimensional parameters
        to dimensional values.
        """
        # mass conservation (adimensional)
        self.inlet.mf = self.outlet.mf = self.mf_ad = self.outlet.solve_mass_conservation('m', self.geometry)
        self.geometry.H[0] = self.inlet.solve_mass_conservation('H', self.geometry)

        if self.om_fixed == 1:
            self.U = self.inlet.U * self.geometry.R
            self.geometry.R = self.U / self.om
        else:
            self.geometry.R = np.sqrt( self.mf / self.mf_ad ) * self.geometry.R
            self.om = self.inlet.U / self.geometry.R[self.n_mid, 0]

        self.geometry.z = self.geometry.z * self.geometry.R[self.n_mid, 0]
        self.geometry.H = self.geometry.H * self.geometry.R[self.n_mid, 0]
        self.geometry.t = self.geometry.t * 2 * np.pi * self.geometry.R[self.n_mid, 0] / self.geometry.Nbl
        self.geometry.tip_clearance_in = self.geometry.tip_clearance_in * self.geometry.H[0]
        self.geometry.bf_clearance = self.geometry.bf_clearance * self.geometry.H[0]
        self.geometry.tip_clearance_out = self.geometry.tip_clearance_out * self.geometry.H[-1]
        self.geometry.Lax = self.geometry.Lax * self.geometry.R[self.n_mid, 0]

        # update flow nodes
        self.inlet.mf = self.inlet.solve_mass_conservation('m', self.geometry)
        self.outlet.mf = self.outlet.solve_mass_conservation('m', self.geometry)

    def update_geometry(self):
        """ 
        Update geometry based on losses. Inlet and outlet blade
        height are retrieved using mass conservation. The blade angles are
        angles are set equal to the flow angle :math:`\\beta`

        This method calls to 
        :meth:`~.TurboSim.geometry.Geometry.set_radial_turbine_impeller_geometry`
        to compute the entire geometry of the blade row.
        """

        # update inlet blade height
        self.geometry.H[0] = self.inlet.solve_mass_conservation('H', self.geometry)

        # update outlet blade height
        self.geometry.H[-1] = self.outlet.solve_mass_conservation('H', self.geometry)

        # update blade heights and blade angles in geometry attribute
        self.geometry.blade_angle[:, 0] = self.inlet.beta[:]
        self.geometry.blade_angle[:, -1] = self.outlet.beta[:]

        # update geometry
        self.geometry.set_radial_turbine_impeller_geometry()

    def set_losses(self, tol=1e-8, max_it=10):
        """
        Computation of the impeller losses and geometry update
        """


        if self._mode == 'design':

            if self.verbose == 2:
                print("\n ------ConvergenceMonitor------ ")
                print(" iter\t\tres\t\t\tres_m\t\tres_e ")

            it = 0
            res = 1000 * np.ones(self.n_span)
            den_old = 1000 * np.ones(self.n_span)

            while (res >= tol).any() and it < max_it:

                # loss calculation
                self.set_stage_loss_coefficients()

                # update thermodynamic conditions after losses and compute mass averaged outlet total enthalpy
                self.set_real_velocity_triangles()

                # update geometry after losses
                self.update_geometry()

                # update flow field
                self.inlet.set_vortex_distribution('free vortex', self.geometry)
                self.outlet.set_vortex_distribution(self.vortex_model, self.geometry,
                                                    self.gw_coef[0], self.gw_coef[1], self.gw_coef[2])

                # compute outlet density and residual
                res = abs(self.outlet.D - den_old) / den_old
                den_old[:] = self.outlet.D[:]

                it += 1

                if self.verbose == 2:
                    mass_in = self.inlet.solve_mass_conservation('m', self.geometry)
                    mass_out = self.outlet.solve_mass_conservation('m', self.geometry)
                    res_m = abs(mass_out - mass_in) / mass_in
                    res_e = abs(self.outlet.Rh - self.inlet.Rh) / self.inlet.Rh
                    print(" %d\t\t\t%f\t%f\t%f " % (it, max(res), res_m, max(res_e)))

            return it, max_it

        if self._mode == 'off-design':
            raise NotImplementedError ("Feature not implemented yet.")

    def set_stage_loss_coefficients(self):
        """ 
        Compute the loss coefficients, calls to the various 
        ``set_*_losses`` methods to compute their respective 
        coefficient in each routine

        .. admonition:: Loss sources breakdown
            :class: hint

            ``*`` = ``incidence, passsage, leakage, shock, te_wake,
            mixing``
        """

        self.reset_losses()

        # set incidence loss
        if self.ic_loss_model != 'none':
            self._flag_incidence = self.set_incidence_losses(self.geometry, self.ic_loss_model, self.loss_model_params)

        # set passage loss
        if self.pf_loss_model != 'none':
            self._flag_profile = self.set_passage_losses(self.geometry, self.pf_loss_model, self.loss_model_params)

        # set secondary flow losses
        if self.sc_loss_model != 'none':
            raise NotImplementedError ("Feature not implemented yet.")

        # set leakage flow losses
        if self.tl_loss_model != 'none':
            self._flag_leakage = self.set_leakage_losses(self.geometry, self.tl_loss_model, self.loss_model_params)

        # set shock losses
        if self.sh_loss_model != 'none':
            self._flag_shock = self.set_shock_losses(self.geometry, self.sh_loss_model, self.loss_model_params)

        # set trailing edge wake losses
        if self.te_wake_loss_model != 'none':
            self._flag_te_wake = self.set_te_wake_losses(self.geometry, self.te_wake_loss_model, self.loss_model_params)

        # set mixing losses
        if self.mx_loss_model != 'none':
            self._flag_mixing = self.set_mixing_losses(self.geometry, self.mx_loss_model, self.loss_model_params)

    def set_real_velocity_triangles(self):
        """
        Computes the real velocity triangles considering 
        losses and thermodynamic properties at inlet and outlet stations.
        """

        # update all thermodynamic properties at the inlet 
        self.inlet.update_static_thermodynamic_state('hs')
        self.inlet.update_total_thermodynamic_state('hs')
        self.inlet.update_total_relative_thermodynamic_state('hs')

        # update inlet velocity triangles
        self.inlet.solve_energy_conservation('stationary','V')
        self.inlet.update_velocity_triangles('alphaVU')
        self.inlet.MachAbs = self.inlet.V / self.inlet.c
        self.inlet.MachRel = self.inlet.W / self.inlet.c
        # update rothalpy
        self.inlet.solve_energy_conservation('rotating','Rh')

        # compute outlet entropy and impose rothalpy conservation
        self.outlet.s = self.inlet.s + self.ds[:, -1]
        self.outlet.Rh[:] = self.inlet.Rh[:]
        # update static thermodynamic properties at the outlet 
        self.outlet.update_static_thermodynamic_state('Ps')

        # compute outlet relative velocity, keep absolute flow angle and recalculate relative flow angle
        self.outlet.solve_energy_conservation('rotating','W')
        self.outlet.update_velocity_triangles('alphaWU')
        self.outlet.MachAbs = self.outlet.V / self.outlet.c
        self.outlet.MachRel = self.outlet.W / self.outlet.c

        # update total and total relative thermodynamic properties at the outlet considering losses
        self.outlet.solve_energy_conservation('stationary','ht')
        self.outlet.solve_energy_conservation('rotating','htr')
        self.outlet.update_total_thermodynamic_state('hs')
        self.outlet.update_total_relative_thermodynamic_state('hs')

        # compute work 
        self.work = self.inlet.ht - self.outlet.ht

    #     LOSS MODELS

    def set_incidence_losses(self, geometry, loss_model, loss_model_params):
        """
        Compute leading edge incidence losses for radial-inflow turbine impellers.
        It calculates losses that occur when the flow angle deviates from the 
        optimal angle.

        :param geometry: Object containing the cascade geometry parameters
        :type geometry: Geometry
        :param loss_model: Loss model selection ('nasa', 'baines')
        :type loss_model: str
        :param loss_model_params: Additional parameters for the selected loss model
        :type loss_model_params: dict
        :return: Binary flag (0: no errors, 1: errors encountered)
        """

        # TODO: add refs in method description

        _flag = 0
        h_out = np.ones(self.n_span) * self.outlet.h_is[:]
        s_out = np.ones(self.n_span) * self.inlet.s[:]

        if loss_model == 'nasa':

            n = np.zeros(self.n_span)

            # incidence is determined as the difference between the blade angle and the optimal flow angle for slip
            i = np.radians(self.inlet.beta[:] - geometry.blade_angle[:, 0])

            n = [2.5 if i[span] <= 0 else 1.75 for span in range(self.n_span)]

            dh = self.inlet.W[:] ** 2 / 2 * (1 - np.cos(i) ** n)

            # determination of outlet thermodynamic conditions and calculation of the entropy generation
            h_out = self.outlet.compute_mass_flux_distribution(geometry) / self.mf * dh + self.outlet.h_is[:]
            for span in range(self.n_span):
                s_out[span] = cp.PropsSI('S', 'H', h_out[span], 'P', self.outlet.P[span], self.library + '::' + self.fluid)

            ds = s_out - self.inlet.s

        elif loss_model == 'baines':
            raise NotImplementedError("Feature not implemented yet.")

        # set loss
        if (ds < 0).any():
            _flag = 1

        self.ds_incidence[:, -1] = [ds[span] if ds[span] >= 0 else 0 for span in range(self.n_span)]
        self.ds[:, -1] = self.ds[:, -1] + self.ds_incidence[:, -1]

        return _flag

    def set_passage_losses(self, geometry, loss_model, loss_model_params):
        """
        Compute passage losses for turbine blades including profile, secondary flow and endwall effects.

        :param geometry: Object containing the cascade geometry parameters
        :type geometry: Geometry  
        :param loss_model: Loss model selection ('baines', 'rodgers', 'vdb', 'denton', 'glassman')
        :type loss_model: str
        :param loss_model_params: Additional parameters for the selected loss model
        :type loss_model_params: dict
        :return: Binary flag (0: no errors, 1: errors encountered)
        """

        # TODO: add refs in method description

        _flag = 0
        h_out = np.ones(self.n_span) * self.outlet.h_is[:]
        s_out = np.ones(self.n_span) * self.inlet.s[:]

        if loss_model == "baines":
            # compute empirical parameter Kp
            p = (geometry.R[self.n_mid, 0] - geometry.R[self.n_mid, -1]) / geometry.H[-1]
            Kp = 1 if p >= 0.2 else 2

            # compute blade chord
            tan_ave = np.mean([np.tan(np.radians(geometry.blade_angle[self.n_mid, 0])),
                               np.tan(np.radians(geometry.blade_angle[self.n_mid, -1]))])

            c = ((geometry.z[self.n_mid, -1] - geometry.z[0, 0]) / np.cos(np.arctan(tan_ave)))

            # compute hydraulic length
            Lh = np.pi / 4 * ((geometry.z[self.n_mid, -1] - geometry.z[0, 0] - geometry.H[0] / 2) +
                              (geometry.R[self.n_mid, 0] - geometry.R[-1, -1] - geometry.H[-1] / 2))

            # compute hydraulic diameter - corrected denominator first addendum with respect to todo: add ref
            Dh = 0.5 * ((4 * np.pi * geometry.R[self.n_mid, 0] * geometry.H[0]) / (geometry.Nbl * geometry.H[0] + 2 * np.pi * geometry.R[self.n_mid, 0]) +
                        (2 * np.pi * (geometry.R[-1, -1] ** 2 - geometry.R[0, -1] ** 2)) / (np.pi * (geometry.R[-1, -1] - geometry.R[0, -1]) + geometry.Nbl * geometry.H[-1]))

            # calculate loss coefficient
            Lp = Kp * 0.1 * (Lh / Dh + 0.68 * (1 - (geometry.R[self.n_mid, -1] / geometry.R[self.n_mid, 0]) ** 2) * np.cos(np.radians(geometry.blade_angle[self.n_mid, -1])) / geometry.H[-1] * c) * \
                 0.5 * (self.inlet.W[self.n_mid] ** 2 + self.outlet.compute_mass_flow_average(self.outlet.W, geometry) ** 2)

            # determination of outlet thermodynamic conditions and calculation of the entropy generation
            h_out = self.outlet.compute_mass_flux_distribution(geometry) / self.mf * Lp + self.outlet.h_is[:]
            for span in range(self.n_span):
                s_out[span] = cp.PropsSI('S', 'H', h_out[span], 'P', self.outlet.P[span], self.library + '::' + self.fluid)

            ds = s_out - self.inlet.s

        elif loss_model == 'rodgers':
            raise NotImplementedError("Feature not implemented yet.")

        elif loss_model == 'vdb':
            raise NotImplementedError("Feature not implemented yet.")

        elif loss_model == 'denton':
            raise NotImplementedError("Feature not implemented yet.")

        elif loss_model == 'glassman':
            raise NotImplementedError("Feature not implemented yet.")

        # set loss
        if (ds < 0).any():
            _flag = 1

        self.ds_profile[:, -1] = [ds[span] if ds[span] >= 0 else 0 for span in range(self.n_span)]
        self.ds[:, -1] = self.ds[:, -1] + self.ds_profile[:, -1]

        return _flag

    def set_leakage_losses(self, geometry, loss_model, loss_model_params):
        """ 
        Tip leakage loss models for radial-inflow turbine impellers: 

        :param loss_model: 'baines', 'jansen', 'denton'
        :type loss_model: str
        :param geometry: object containing the cascade geometry 
        :type geometry: Geometry

        :return _flag: binary (0 - no errors, 1 - errors) 
        """

        # TODO: add refs in method description

        _flag = 0
        h_out = np.ones(self.n_span) * self.outlet.h_is[:]
        s_out = np.ones(self.n_span) * self.inlet.s[:]

        if loss_model == 'baines':

            Cx = ((1 - (geometry.R[-1, -1] / geometry.R[self.n_mid, 0])) /
                  (self.inlet.Vm[self.n_mid] * geometry.H[0]))

            Cr = ((geometry.R[-1, -1] / geometry.R[self.n_mid, 0]) * (geometry.Lax - geometry.H[0]) /
                  (self.outlet.Vm[self.n_mid] * geometry.R[self.n_mid, -1] * geometry.H[-1]))

            # clearance loss coefficient - corrected from [Ref. 8]: U2s ** 5 --> U2s ** 3 to obtain dimensionally correct coefficient
            Lc = (self.inlet.U[-1] ** 3 * geometry.Nbl) / (8 * np.pi) * \
                 (loss_model_params['Kx'] * geometry.tip_clearance_in * Cx + loss_model_params['Kr'] * geometry.tip_clearance_out * Cr +
                  loss_model_params['Kxr'] * np.sqrt(geometry.tip_clearance_in * Cx * geometry.tip_clearance_out * Cr))

            # determination of outlet thermodynamic conditions and calculation of the entropy generation
            h_out = self.outlet.compute_mass_flux_distribution(geometry) / self.mf * Lc + self.outlet.h_is[:]
            for span in range(self.n_span):
                s_out[span] = cp.PropsSI('S', 'H', h_out[span], 'P', self.outlet.P[span], self.library + '::' + self.fluid)

            ds = s_out - self.inlet.s

        elif loss_model == 'jansen':
            raise NotImplementedError("Feature not implemented yet.")

        elif loss_model == 'denton':
            raise NotImplementedError("Feature not implemented yet.")

        # set loss
        if (ds < 0).any():
            _flag = 1

        self.ds_leakage[:, -1] = [ds[span] if ds[span] >= 0 else 0 for span in range(self.n_span)]
        self.ds[:, -1] = self.ds[:, -1] + self.ds_leakage[:, -1]

        return _flag

    def set_shock_losses(self, geometry, loss_model, loss_model_params):
        """
        Computes entropy generation due to shock waves in turbine blade passages.

        :param geometry: Object containing blade passage geometry
        :type geometry: Geometry 
        :param loss_model: Model selection ('rh real' or 'rh perfect')
        :type loss_model: str
        :param loss_model_params: Model parameters (``M_rt_sh`` and ``P_rt_sh`` ratios)
        :type loss_model_params: dict
        :return: Success (0) or error (1) ``_flag``
        """
        # TODO: add refs in method description

        _flag = 0
        h_out = np.ones(self.n_span) * self.outlet.h_is[:]
        s_out = np.ones(self.n_span) * self.inlet.s[:]

        # set pre-shock Mach and pressure
        MA = self.outlet.MachRel[:] * loss_model_params['M_rt_sh']
        PA = self.outlet.P[:] * loss_model_params['P_rt_sh']
        sA = self.outlet.s[:]
        ds = np.zeros((self.n_span))
        # convert shock angle to radians
        self.shock_angle = np.radians(self.shock_angle)

        def rankine_hugoniot_perfect(R, Cp, gamma, MA, beta):
            """ Compute the post-shock conditions and the deviation angle for a perfect gas assuming the
            pre-shock conditions and the shock angle are known
            :param R:       gas constant
            :param Cp:      specific heat at constant pressure
            :param gamma:   perfect gas ratio of specific heats
            :param MA:      pre-shock Mach number
            :param beta:    shock angle [rad] """

            if MA * np.sin(beta) >= 1.0:
                MA_normal = MA * np.sin(beta)
                theta = np.arctan(2 / np.tan(beta) * (MA ** 2 * np.sin(beta) ** 2 - 1) /
                                  (MA ** 2 * (gamma + np.cos(2 * beta) + 2)))
                MB_normal = np.sqrt((1 + (gamma - 1) / 2 * MA_normal ** 2) / (gamma * MA_normal ** 2 - (gamma - 1) / 2))
                MB = MB_normal / np.sin(beta)
                P_ratio = 1 + (2 * gamma) / (gamma + 1) * (MA_normal ** 2 - 1)
                D_ratio = (1 + (gamma + 1) / (gamma - 1) * P_ratio) / ((gamma + 1) / (gamma - 1) + P_ratio)
                T_ratio = P_ratio / D_ratio
                delta_s = Cp * np.log(T_ratio) - R * np.log(P_ratio)
            else:
                MB = MA
                P_ratio = 1
                T_ratio = 1
                delta_s = 0
                theta = 0

            return np.array([theta, P_ratio, T_ratio, MB, delta_s])

        def rankine_hugoniot_real(p, *data):
            """ Solve a non-linear system of 4 equations to find the post-shock conditions and the deviation angle
            assuming the pre-shock conditions and the shock angle are known
            :param p:       post-shock conditions PB, hB, VB and theta [rad]
            :param data:    pre-shock conditions PA, hA, DA, VA, beta [rad]
             """

            PA, hA, DA, VA, beta = data
            PB, hB, VB, theta = p

            self.eos.update(CoolProp.HmassP_INPUTS, hB, PB)
            DB = self.eos.rhomass()
            v2 = 1 / DB
            v1 = 1 / DA

            dP = (PB - PA) + (v2 - v1) * (DA * VA * np.sin(beta)) ** 2
            dh = (hB - hA) - 0.5 * (PB - PA) * (v1 + v2)
            dV = VA * np.cos(beta) - VB * np.cos(beta - theta)
            dtheta = DA * np.tan(beta) - DB * np.tan(beta - theta)

            return dP, dh, dV, dtheta

        def compute_shock_angle(MA, gamma, theta):
            """ Compute and returns the shock angle based on pre-shock conditions - see todo: add ref
            :param MA:      pre shock Mach number
            :param gamma:   specific heat ratio (perfect gas) or isentropic-pressure volume exponent (real gas)
            :param theta:   deviation angle across the shock [rad] """
            return np.arcsin(1 / MA) + (gamma + 1) / 4 * MA ** 2 / (MA ** 2 - 1) * theta

        # initialize solution with perfect gas model: cp and gamma computed with (Pc, Tc)
        theta_tmp, Pr_tmp, Tr_tmp, MB_tmp, ds_tmp = \
            [np.array([rankine_hugoniot_perfect(self.outlet.R, self.outlet.cp0, self.outlet.gamma_id, MA[span], self.shock_angle[span]) for span in range(self.n_span)])[:, col] for col in range(5)]

        self.shock_angle = [compute_shock_angle(MA[span], self.outlet.gamma_id, theta_tmp[span])
                            if theta_tmp[span] != 0 and self.angle_fixed == False and MA[span] > 1
                            else self.shock_angle[span] for span in range(self.n_span)]

        theta = theta_tmp[:]

        if loss_model == 'rh real':
            for span in range(self.n_span):
                if Pr_tmp[span] != 1:
                    try:
                        # real gas
                        self.eos.update(CoolProp.PSmass_INPUTS, PA[span], self.inlet.s[span] + self.outlet.ds[span])
                        VA = MA[span] * self.eos.speed_sound()
                        hA = self.eos.hmass()
                        DA = self.eos.rhomass()
                        P_guess = PA[span] * Pr_tmp[span]
                        T_guess = self.eos.T() * Tr_tmp[span]
                        self.eos.update(CoolProp.PT_INPUTS, P_guess, T_guess)
                        h_guess = self.eos.hmass()
                        V_guess = MB_tmp[span] * self.eos.speed_sound()

                        # iterate based on shock angle
                        if self.angle_fixed == False:
                            it = 0
                            res = 10
                            tol = 1e-3
                            max_it = 20
                            shock_angle_old = self.shock_angle[span]
                            while res > tol and it < max_it:

                                # solve R-H relations for a real gas
                                data = (PA[span], hA, DA, VA, self.shock_angle[span])
                                guess = (P_guess, h_guess, V_guess, theta[span])
                                PB, hB, VB, theta[span] = \
                                    opt.fsolve(rankine_hugoniot_real, guess, args=data, full_output=False, xtol=1.0e-06)

                                # recompute shock angle
                                self.shock_angle[span] = compute_shock_angle(MA[span], self.outlet.gamma_Pv[span],  theta[span])

                                # update residual
                                res = abs(self.shock_angle[span] - shock_angle_old) / shock_angle_old
                                shock_angle_old = self.shock_angle[span]

                                it += 1

                            if it == max_it and res > tol:
                                raise Exception('Maximum number of iterations reached in real gas Rankine-Hugoniot at span loc. %d' % (span))
                        else:
                            # solve R-H relations for a real gas
                            data = (PA[span], hA, DA, VA, self.shock_angle[span])
                            guess = (P_guess, h_guess, V_guess, theta[span])
                            PB, hB, VB, theta[span] = \
                                opt.fsolve(rankine_hugoniot_real, guess, args=data, full_output=False, xtol=1.0e-06)

                        sB = cp.PropsSI('S', 'H', hB, 'P', PB, self.library + '::' + self.fluid)
                        ds[span] = sB - sA[span]
                        if ds[span] <= 0:
                            raise Exception('ds <= 0 in real gas Rankine-Hugoniot at span loc. %d' % (span))
                    except:
                        ds[span] = ds_tmp[span]
                        _flag = 1
                else:
                    ds[span] = 0

        elif loss_model == 'rh ideal':
            try:
                # iterate based on shock angle
                if self.angle_fixed == False:
                    it = 0
                    res = 10 * np.ones((self.n_span))
                    tol = 1e-3
                    max_it = 20
                    shock_angle_old = self.shock_angle[:]
                    while (res > tol).any() and it < max_it:
                        theta, _, _, _, ds = \
                            [np.array([rankine_hugoniot_perfect(self.outlet.R, self.outlet.cp0, self.outlet.gamma_id, MA[span], self.shock_angle[span]) for span in range(self.n_span)])[:, col] for col in range(5)]

                        self.shock_angle = np.array([compute_shock_angle(MA[span], self.outlet.gamma_id, theta[span])
                                                     if theta[span] != 0 and self.angle_fixed == False and MA[span] > 1
                                                     else self.shock_angle[span] for span in range(self.n_span)])

                        res = abs(self.shock_angle - shock_angle_old) / shock_angle_old
                        shock_angle_old = self.shock_angle

                        it += 1

                    if it == max_it and np.any(res) > tol:
                        raise Exception('Maximum number of iterations reached in perfect gas Rankine-Hugoniot.')
                else:
                    ds = ds_tmp
            except:
                ds = ds_tmp
                _flag = 1

        self.shock_angle = np.degrees(self.shock_angle)
        self.ds_shock[:, -1] = [ds[span] if ds[span] >= 0 else 0 for span in range(self.n_span)]
        self.ds[:, -1] = self.ds[:, -1] + self.ds_shock[:, -1]

        return _flag

    def set_te_wake_losses(self, geometry, loss_model, loss_model_params):
        """ 
        Trailing edge wake loss models for turbine blades


        :param geometry: Object containing the cascade geometry
        :type geometry: Geometry
        :param loss_model: Loss model selection ('baumgartner' or 'denton')
        :type loss_model: str
        :param loss_model_params: Dictionary containing model parameters 
        :type loss_model_params: dict
        :return: Binary ``_flag`` indicating calculation status

        .. hint::

            Loss model parameters used by mixing models are
            ``th_t, d*_th, bld_parm, xq_conv, yq_conv, zq_conv``

        """

        # TODO: add refs in method description

        _flag = 0

        # compute boundary layer parameters
        theta = loss_model_params['th_t'] * geometry.t[:, -1]
        delta_star = loss_model_params['d*_th'] * theta

        t_a = geometry.t[:, -1] / geometry.pitch[:, -1] / np.cos(np.radians(self.outlet.beta[:]))  # Hp: deviation = 0 --> angle_a = angle_out
        theta_a = loss_model_params['th_t'] * t_a
        delta_star_a = loss_model_params['d*_th'] * theta_a

        # compute base pressure
        if np.all(self.Pb) == 0:
            self.Pb = (self.inlet.Ptr[:] *
                       np.array([w_extrap(self.outlet.P[span] / self.inlet.Ptr[span],
                                          loss_model_params['bld_parm'], loss_model_params["xq_conv"],
                                          loss_model_params["yq_conv"], loss_model_params["zq_conv"])
                                 for span in range(self.n_span)])[:, 0])

        if loss_model == 'baumgartner':

            # compute effective base pressure coefficient
            self.CPb = (2 * (self.Pb / self.outlet.P[:] - 1) *
                   (2 / (self.outlet.gamma_Pv[:] * self.outlet.MachRel[:] ** 2) * geometry.t[:, -1] / geometry.pitch[:, -1]))

            # compute wake mixing entropy generation
            ds = (- self.CPb * t_a + 2 * theta_a + (delta_star_a + t_a) ** 2) / \
                                           self.outlet.T_is[:] * (self.outlet.W_is[:] ** 2 / 2)

        elif loss_model == 'denton':

            # compute base pressure coefficient
            self.CPb = 2 * (self.Pb - self.outlet.P[:]) / (self.outlet.D[:] * self.outlet.W[:] ** 2)

            # compute wake mixing entropy generation
            ds = (- self.CPb * t_a + 2 * theta_a + (delta_star_a + t_a) ** 2) / \
                                           self.outlet.T_is[:] * (self.outlet.W_is[:] ** 2 / 2)

        # set loss
        if (ds < 0).any():
            _flag = 1

        self.ds_te_wake[:, -1] = [ds[span] if ds[span] >= 0 else 0 for span in range(self.n_span)]
        self.ds[:, -1] = self.ds[:, -1] + self.ds_te_wake[:, -1]

        return _flag

    def set_mixing_losses(self, geometry, loss_model, loss_model_params):
        """
        This method calculates entropy generation due to mixing effects in the
        blade passage.
        It considers streamlines in transonic conditions (Mach > 0.9),
        performing calculations to determine the flow properties at 
        the mixing plane using mass, energy, and momentum conservation.

        :param geometry: Cascade geometry object containing blade parameters
        :type geometry: CascadeGeometry
        :param loss_model: Selection of mixing loss model ('osnaghi')
        :type loss_model: str
        :param loss_model_params: Dictionary containing model parameters:
                    :type loss_model_params: dict
        :return: Binary flag (0: no errors, 1: errors encountered)
        """

        # TODO: add refs in method description
        # TODO: implement convergent-divergent calculation

        _flag = 0

        if loss_model == 'osnaghi':

            # initialize quantities
            angle_a = np.zeros(self.n_span)
            V_out_new = np.zeros(self.n_span)
            t_p_out = geometry.t[:, -1] / geometry.pitch[:, -1]
            dstar_p_out = loss_model_params["d*_H"] * geometry.H[-1] / geometry.pitch[:, -1]
            th_p = dstar_p_out / loss_model_params["d*_th"]
            h_out = np.ones(self.n_span) * self.outlet.h_is[:]
            s_out = np.ones(self.n_span) * self.inlet.s[:]
            V_out = np.ones(self.n_span) * self.outlet.W[:]
            D_out = np.ones(self.n_span) * self.outlet.D[:]

            # Extract all the spanwise locations indices where M_rel >= 0.9
            mask = np.where(np.array([0 if self.outlet.MachRel[span] < 0.9 else 1 for span in range(self.n_span)]) == 1)[0]

            # compute base pressure
            if np.all(self.Pb) == 0:
                self.Pb = (self.inlet.Ptr[:] *
                           np.array([w_extrap(self.outlet.P[span] / self.inlet.Ptr[span],
                                              loss_model_params['bld_parm'], loss_model_params["xq_conv"], 
                                              loss_model_params["yq_conv"], loss_model_params["zq_conv"]) 
                                     for span in range(self.n_span)])[:,0])

            # sonic loop to compute field quantities at throat section
            self.s_th = self.inlet.s[:] + self.ds[:, -1]
            self.P_th = np.array([opt.fsolve(self.compute_sonic_conditions, self.outlet.P[span], args=(self.inlet.htr[span], self.s_th[span], self.inlet.U[span], self.outlet.U[span]), full_output=False, xtol=1.0e-06)
                                  for span in range(self.n_span)])[:,0]
            self.post_expansion = self.P_th / self.outlet.P[:]
            for span in range(self.n_span):
                self.eos.update(CoolProp.PSmass_INPUTS, self.P_th[span], self.s_th[span])
                self.V_th[span] = self.eos.speed_sound()
                self.D_th[span] = self.eos.rhomass()
                self.h_th[span] = self.eos.hmass()

            # loop to compute outlet entropy
            it = 0
            res = 10 * np.ones(len(mask))
            max_it = 200
            tol = 1e-9

            while (res > tol).any() and (it < max_it):

                # energy balance
                h_out[mask] = self.h_th[mask] + self.V_th[mask] ** 2 / 2 - V_out[mask] ** 2 / 2
                for span in mask:
                    self.eos.update(cp.HmassP_INPUTS, h_out[span], self.outlet.P[span])
                    D_out[span] = self.eos.rhomass()
                    s_out[span] = self.eos.smass()

                # mass balance
                angle_a[mask] = np.arccos(t_p_out[mask] + dstar_p_out[mask] + (D_out[mask] * V_out[mask] * np.cos(np.radians(self.outlet.beta[mask]))) /
                                    (self.D_th[mask] * self.V_th[mask]) * loss_model_params["Rmx_Rte_rt"])

                if np.isnan(np.any(angle_a[mask])):
                    raise ValueError ('Exception during computation of cascade mixing losses.')

                # calculate deviation
                for span in mask:
                    if (np.radians(self.outlet.beta[span]) * angle_a[span]) == - np.abs(np.radians(self.outlet.beta[span]) * angle_a[span]):
                        angle_a[span] = - angle_a[span]

                    if np.sign(np.radians(self.outlet.beta[span])) == - 1:
                        self.deviation_angle[span] = np.degrees(np.radians(self.outlet.beta[span]) - angle_a[span])
                    else:
                        self.deviation_angle[span] = np.degrees(angle_a[span] - np.radians(self.outlet.beta[span]))

                    if self.deviation_angle[span] < 0:
                        self.deviation_angle[span] = 0

                # axial momentum balance
                V_out_new[mask] = (self.D_th[mask] * self.V_th[mask] ** 2 * (np.cos(angle_a[mask]) - t_p_out[mask] - dstar_p_out[mask] - th_p[mask]) +
                                   self.P_th[mask] * (np.cos(angle_a[mask]) - t_p_out[mask]) + self.Pb[mask] * t_p_out[mask] - self.outlet.P[mask] * np.cos(angle_a[mask])
                                   * loss_model_params["Rmx_Rte_rt"]) / \
                                  (self.D_th[mask] * self.V_th[mask] * (np.cos(angle_a[mask]) - t_p_out[mask] - dstar_p_out[mask]) *
                                   np.cos(np.deg2rad(self.deviation_angle[mask])))

                res = np.abs(V_out_new[mask] - V_out[mask]) / V_out[mask]
                V_out[mask] = V_out_new[mask]

                it += 1

            # set loss
            ds = s_out - self.inlet.s
            if (ds < 0).any():
                _flag = 1

            self.ds_mixing[:, -1] = [ds[span] if ds[span] >= 0 else 0 for span in range(self.n_span)]
            self.ds[:, -1] = self.ds[:, -1] + self.ds_mixing[:, -1]

        return _flag

    def compute_sonic_conditions(self, p, *data):
        """ Compute sonic conditions """
        htr_in, s_a, U_in, U_a = data
        P_a = p

        # sonic conditions
        self.eos.update(CoolProp.PSmass_INPUTS, P_a, s_a)
        h_a = self.eos.hmass()
        c_a = self.eos.speed_sound()

        # energy balance
        res = 1 - h_a / (htr_in - U_in ** 2 / 2) - (c_a ** 2 - U_a ** 2) / (2 * (htr_in - U_in ** 2 / 2))

        return res


if __name__ == '__main__':

    import flow_model as flow
    import geometry as geo

    # initialize impeller class object
    eos = cp.AbstractState('REFPROP', 'MM')
    impeller = RadialTurbineImpeller(eos, 'REFPROP', 'MM', 2,
                                     massflow=0.132, psi=1.3, phi=0.0, phi_in=0.3, mu=0.94,
                                     R_ratio=0.56, Rh_Rs_out=0.4, Lax_dR=0.9,
                                     t_p_in=0.015, t_p_out=0.015,
                                     g_b_in=0.1, g_b_out=0.04, g_b_bf=0.07,
                                     Nbl=16,
                                     ic_loss_model='nasa', pf_loss_model='baines',
                                     sh_loss_model='rh real', te_wake_loss_model='baumgartner',
                                     mx_loss_model='none', tl_loss_model='baines', sc_loss_model='none',
                                     shock_angle=None)

    # set BCs
    Pt_in_turb = 18.1e5
    Tt_in_turb = 573
    P_out_turb = 0.44e5

    # set impeller inlet and outlet conditions
    eos.update(CoolProp.PT_INPUTS, Pt_in_turb, Tt_in_turb)
    ht_in_turb = eos.hmass()
    s_in_turb = eos.smass()

    bc_dict = {"hs": [ht_in_turb, s_in_turb], "P": P_out_turb}

    # isentropic impeller design
    impeller.set_meanline_design(bc_dict=bc_dict)
    impeller.set_spanwise_design()
