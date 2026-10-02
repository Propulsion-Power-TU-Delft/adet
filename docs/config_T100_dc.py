# -*- coding: utf-8 -*-
#####################################################################
# TurboSim - Turbomachinery stage design program
# Authors: Dr. ir. M. Pini, ir. A. Giuffré, ir. M. Majer
# Content: Configuration file radial-inflow turbine
# 2018 - TU Delft - All rights reserved
# Version: 1.0 (07/07/2024)
#####################################################################
import numpy as np

n = 25
step_speed_lines = 0.1

config = {
    "Design":
        {
            "mode": 1,  # Required - Run mode: 1: single point design, or 2: design map
            "n_proc": 4,  # Required - Number of processors to use in run mode 2 - Default 1
            "fluid":
                {
                    "fluid": "Air",  # Required - fluid name
                    "library": "REFPROP",  # Optional - name of the thermodynamic library: REFPROP, HEOS or LUT
                },
            "P0": 580400,  # Optional - Total inlet pressure in Pa (required if Pr is not specified)
            "T0": 1056.4,  # Optional - Total inlet temperature in K (required if Tr is not specified)
            "exp_ratio": 5.73,  # Required - Expansion ratio inlet/outlet conditions
            "exp_ratio_kind": "pressure",  # Required - Expansion model: 'pressure' or 'density'
            "outlet_state": "ts",  # Required - Expansion type: 'ts' = static outlet conditions, 'tt' = total outlet conditions
            "load": 0.96,  # Optional - Load coefficient Vt2/U2 - R2/R3 * Vt3/U2 (isentropic) - Default None - ! Leave empty or None if 'Ns' is specified !
            "Ns": None,  # Optional - Specific speed ... - Default None - ! Leave empty or None if 'load' is specified !
            "flow": 0.19,  # Required - Flow coefficient Vm2/U2 (isentropic)
            "samples": 1,  # Optional - Number of samples for the maps discretization (only required if mode = 2) - Default None
            "mass_flow": 0.33,  # Optional - Massflow rate in kg/s - Default None - ! Leave empty or None if 'SP' is specified !
            "SP": None,  # Optional - Size parameter in m - Default None - ! Leave empty or None if 'mass_flow' is specified !
            "res_dir": "T100_dc",  # Optional - Default Automatic
            "verbosity": 2,  # Optional - Default 0
            "warning": 0,  # Optional - Default 0
            "save": 2,  # Optional - 0: no save of results (fastest), 1: save only output files, 2: full save including figures - Default 1
            "flag_local": 1,  # Optional - Set to 1 if running the code as a standalone - Default 0
            "Zst": 19,  # Optional - Stator vanes count: [any integer number] or 0 (vanes count calculated using solidity) - Default 0
            "Zrot": 16,  # Optional - Impeller blades count: [any integer number] or 0 (blades count calculated using a correlation) - Default 0
            "loss_impeller": 1,  # Optional - Impeller blades losses: disabled (0), enabled (1) - Default 1
            "loss_stator": 1,  # Optional - Stator vanes losses: disabled (0), enabled (1) - Default 1
            "loss_radial_gap": 1,  # Optional - Radial gap losses: disabled (0), enabled (1) - Default 1
            "loss_windage": 1,  # Optional - Impeller windage losses: disabled (0), enabled (1) - Default 1
            "vortex_distribution": 0,  # Optional - Impeller exducer vortex distribution: free vortex (0), constant angle (1), forced vortex (2), general whirl (3) - Default 0
            "alpha0": 25,  # Optional - Stator vanes inlet angle in deg - Default 10
            "alpha3": 0,  # Optional - Impeller outlet swirl in deg (isentropic) - Default 0
            "cone_angle_in": 90,  # Optional - Impeller inlet "cone angle" in deg - Default 90
            "cone_angle_out": 0,  # Optional - Impeller outlet "cone angle" in deg - Default 0
            "R3_2_R2": 0.448,  # Optional - Impeller radius ratio outlet/inlet - Default 0.5
            "R2_2_R1": 0.916,  # Optional - Radial gap radius ratio outlet/inlet - Default 0.9
            "R1_2_R0": 0.856,  # Optional - Stator vanes radius ratio outlet/inlet - Default 0.8
            "Rh_Rt": 0.414,  # Optional - Impeller exducer radius ratio hub/shroud - Default 0.4
            "Vm_ratio": 1.678,  # Optional - Impeller meridional velocity ratio outlet/inlet (isentropic) - Default 1.5
            "H_DR": 0.281,  # Optional - Stator vanes aspect ratio: (mean blade height)/(inlet radius - outlet radius) - Default 0.3
            "Lax_R": 0.67,  # Optional - Impeller aspect ratio in meridional plane: (axial length)/(inlet radius - outlet radius) - Default 0.9
            "solidity_stator": 1.5,  # Optional - Stator vanes solidity: (chord)/(outlet pitch) - Default 1.5
            "t_s_st": 0.025,  # Optional - Stator vanes trailing edge blockage: thickness / (outlet pitch) - Default 0.015
            "t_s_rot": 0.0437,  # Optional - Impeller blades trailing edge blockage: thickness / (outlet pitch) - Default 0.06
            "g_h_le": 0.04,  # Optional - Impeller blades tip normalized axial clearance (inlet section): (axial clearance) / (inlet blade height) - Default 0.1
            "g_h_te": 0.01,  # Optional - Impeller blades tip normalized radial clearance (outlet section): (radial clearance) / (outlet blade height) - Default 0.04
            "gbf_h": 0.072,  # Optional - Impeller disk normalized axial clearance: (disk axial clearance) / (inlet blade height) - Default 0.7
            "min_thickness": 0.0001,  # Optional - Minimum material thickness for impeller blades and stator vanes in m - Default None
            "min_clearance": 0.00005,  # Optional - Minimum impeller tip clearance in m - Default None
            "max_MachRel_tip": 1.4,  # Optional - Maximum impeller tip relative Mach number - Default None
            "max_U_tip": 650,  # Optional - Maximum impeller inlet peripheral speed in m/s - Default None
            "slip_model": "chen",  # Optional - Model to be used for the slip calculation (chen or stanitz) - Default chen
            "loss_coeff":
                {
                    "Kx": 0.4,
                    "Kr": 0.75,
                    "Kxr": -0.3,
                    "k": 0.005,
                },
        },
    "Analysis":
        {
            "fluid":
                {
                    "fluid": "Air",  # Required - fluid name
                    "library": "REFPROP",  # Optional - name of the thermodynamic library: REFPROP, HEOS or LUT
                },

            "Settings":
                {
                    "total_pressure_inlet": np.linspace(580400,580400, n),
                    "total_temperature_inlet": np.linspace(1056.4,1056.4, n),
                    # "pressure_ratio": [2.977,],
                    "pressure_ratio": np.linspace(1.2,6, n),
                    # "mass_flow": np.linspace(0.1,0.36, n),
                    "rotational_speed": np.arange(0.5, 1.11, step_speed_lines) * 110752,
                },
        },
}

