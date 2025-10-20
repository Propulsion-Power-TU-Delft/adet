import numpy as np


class RadialCompressorStage:
    def __init__(self, n_span, verbose, Cf_imp, k_diff, eps_min=0.3, eps_max=0.65):

        # total enthalpy loss
        self.dht_incidence = np.zeros(n_span)
        self.dht_shock = np.zeros(n_span)
        self.dht_loading = np.zeros(n_span)
        self.dht_friction = np.zeros(n_span)
        self.dht_clearance = np.zeros(n_span)
        self.dht_mixing = np.zeros(n_span)
        self.dht_disk = np.zeros(n_span)
        self.dht_recirculation = np.zeros(n_span)
        self.dht_leakage = np.zeros(n_span)
        self.dht_diffuser = np.zeros(n_span)
        self.dht_volute = np.zeros(n_span)
        self.dht_cone = np.zeros(n_span)
        self.dht_int = np.zeros(n_span)
        self.dht_ext = np.zeros(n_span)

        # total pressure loss
        self.dPt_incidence = np.zeros(n_span)
        self.dPt_shock = np.zeros(n_span)
        self.dPt_loading = np.zeros(n_span)
        self.dPt_friction = np.zeros(n_span)
        self.dPt_clearance = np.zeros(n_span)
        self.dPt_mixing = np.zeros(n_span)
        self.dPt_disk = np.zeros(n_span)
        self.dPt_recirculation = np.zeros(n_span)
        self.dPt_leakage = np.zeros(n_span)
        self.dPt_diffuser = np.zeros(n_span)
        self.dPt_volute = np.zeros(n_span)
        self.dPt_cone = np.zeros(n_span)

        # entropy generation
        self.ds_incidence = np.zeros(n_span)
        self.ds_shock = np.zeros(n_span)
        self.ds_loading = np.zeros(n_span)
        self.ds_friction = np.zeros(n_span)
        self.ds_clearance = np.zeros(n_span)
        self.ds_mixing = np.zeros(n_span)
        self.ds_disk = np.zeros(n_span)
        self.ds_recirculation = np.zeros(n_span)
        self.ds_leakage = np.zeros(n_span)
        self.ds_diffuser = np.zeros(n_span)
        self.ds_volute = np.zeros(n_span)
        self.ds_cone = np.zeros(n_span)

        # ancillary properties
        self.verbose = verbose
        self.Cf_imp = Cf_imp
        self.Cf_diff = 0.0
        self.Cf_cone = 0.0
        self.k_diff = k_diff
        self.D_hyd_imp = 0.0
        self.D_hyd_diff = 0.0
        self.D_hyd_cone = 0.0
        self.L_hyd_imp = 0.0
        self.Re_imp = 0.0
        self.Re_diff = 0.0
        self.Re_cone = 0.0
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps = eps_min  # wake fraction at impeller exit [15]
        self.m_cl = 0.0

