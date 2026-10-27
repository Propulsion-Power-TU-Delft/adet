"""
Back-to-back turbocharger assembly: compressor impeller, pump-out-vane
cavity (behind the impeller disk) and turbine rotor, each solved as a
``SurfaceFlow`` *meridional-only* (hub-to-shroud) model --
``include_blade_to_blade=False``, so there is no blade-to-blade loading,
no SlipTransition/Kutta closure, and the blade metal angle is a direct,
prescribed boundary condition rather than a solved unknown.

This also uses the corrected rothalpy treatment: rothalpy is *not* a
single constant applied uniformly over the whole domain (that was wrong --
rothalpy depends on each streamline's own local velocity triangle, which
varies across the span). Instead, ``assemble()`` takes the upstream
*absolute total enthalpy* (uniform at inlet), each streamline's local
static enthalpy is anchored at its own leading edge
(``AbsoluteEnergyClosure``) and then conserved downstream along it
(``RothalpyConservation``) -- see ``radial_impeller_blade_to_blade_flow.py``
/``radial_inflow_turbine_blade_to_blade_flow.py`` for the same pattern.

Known issue: the compressor geometry below (25-70 mm radius, 14-8 mm
height, 50 mm axial length, 58,000 rpm) -- i.e. the original numbers this
example was specified with -- does not converge with this solver, even in
this simpler meridional-only form (confirmed at both 58,000 rpm and
18,000 rpm). This is being investigated separately; the script still runs
end to end (``error_on_fail=False`` throughout) so the compressor block
can be poked at directly.
"""

import logging
from typing import Any, cast

import CoolProp as cp
import numpy as np
from matplotlib import pyplot as plt

from adet.assemblers import CasadiSystem
from adet.equations.streamline_curvature import (
    SurfaceFlow,
    build_meridional_grid,
    midline_tangent_angle,
    resample_by_arc_length,
    spanwise_normal_curvature,
)
from adet.fluid.ideal_eos import IdealGasState
from adet.geometry import BezierCurve
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger

logger = logging.getLogger(__name__)
setup_logger(logger)

GAMMA_AIR = 1.4
R_AIR = 287.05  # [J / (kg K)]
VISCOSITY_AIR = 1.8e-5  # [Pa s], unused but required by IdealGasState

RPM = 58_000.0  # [rev/min], shared shaft speed
OMEGA = RPM * 2 * np.pi / 60  # [rad/s]

N_SPAN = 7  # hub-to-shroud stations, shared by all three surfaces

IMPELLER_DISK_TIP_THICKNESS = 0.001  # [m], not specified by the user; axial gap
# left between the compressor's trailing edge and the pump-out-vane cavity in
# the combined plot (Section 4), so the two meridional grids don't overlap

# Two-slope blade-angle law parameters shared by the compressor
# (front-loaded) and the turbine (rear-loaded -- the mirror image of the
# same two-slope shape, see rear_loaded_beta_law) -- not specified by the
# user; the pump-out vane uses a trivial, constant-zero law instead.
FRONT_LOADED_KWARGS = {
    'split_frac': 0.5,
    'frac_at_split': 0.88,
    'half_width': 0.12,
}


def grid_field(surface, sol_dict, n_stream, n_span, accessor):
    return np.array(
        [
            [accessor(surface, sol_dict, i, j)[0] for j in range(n_span)]
            for i in range(n_stream)
        ]
    )


def mass_flux_weighted_mean(field_row, rho_row, vm_row, r_row, n_coord_row):
    """Mass-flux-weighted spanwise average of ``field_row`` at a fixed
    streamwise station, from that station's density/meridional-velocity/
    radius rows and its spanwise arc-length coordinate."""
    flux = rho_row * vm_row * r_row
    return np.trapezoid(field_row * flux, n_coord_row) / np.trapezoid(flux, n_coord_row)


def span_coord(delta_n):
    """Cumulative spanwise arc length (hub=0), shape (n_stream, n_span)."""
    return np.concatenate(
        [np.zeros((delta_n.shape[0], 1)), np.cumsum(delta_n, axis=1)], axis=1
    )


# ============================================================
# 1. Compressor impeller
# ============================================================
COMP_N_STREAM = 55

COMP_R_MEAN_IN = 0.035  # [m]
COMP_R_MEAN_OUT = 0.085  # [m]
COMP_H_IN = 0.016  # [m]
COMP_H_OUT = 0.006  # [m]
COMP_AXIAL_LENGTH = 0.055  # [m]

COMP_MERID_ANGLE_IN = np.radians(0.0)  # axial inflow
COMP_MERID_ANGLE_OUT = np.radians(90.0)  # radial outflow

COMP_MASS_FLOW = 0.75  # [kg/s]
COMP_T0_IN = 253.0  # [K]
COMP_P0_IN = 1.0e5  # [Pa] 1 bar total

COMP_BETA_IN = np.radians(-52.0)
COMP_BETA_OUT = np.radians(-0.0)
COMP_NUM_BLADES = 20
COMP_BLADE_THICKNESS = 0.0015  # [m], not specified by the user

COMP_Z_HUB_IN = 0.5 * COMP_H_IN * np.sin(COMP_MERID_ANGLE_IN)
COMP_R_HUB_IN = COMP_R_MEAN_IN - 0.5 * COMP_H_IN * np.cos(COMP_MERID_ANGLE_IN)
COMP_Z_SHROUD_IN = -0.5 * COMP_H_IN * np.sin(COMP_MERID_ANGLE_IN)
COMP_R_SHROUD_IN = COMP_R_MEAN_IN + 0.5 * COMP_H_IN * np.cos(COMP_MERID_ANGLE_IN)

COMP_Z_HUB_OUT = COMP_AXIAL_LENGTH + 0.5 * COMP_H_OUT * np.sin(COMP_MERID_ANGLE_OUT)
COMP_R_HUB_OUT = COMP_R_MEAN_OUT - 0.5 * COMP_H_OUT * np.cos(COMP_MERID_ANGLE_OUT)
COMP_Z_SHROUD_OUT = COMP_AXIAL_LENGTH - 0.5 * COMP_H_OUT * np.sin(COMP_MERID_ANGLE_OUT)
COMP_R_SHROUD_OUT = COMP_R_MEAN_OUT + 0.5 * COMP_H_OUT * np.cos(COMP_MERID_ANGLE_OUT)

comp_hub_curve = BezierCurve(
    z_in=COMP_Z_HUB_IN,
    z_out=COMP_Z_HUB_OUT,
    radius_in=COMP_R_HUB_IN,
    radius_out=COMP_R_HUB_OUT,
    angle_in=COMP_MERID_ANGLE_IN,
    angle_out=COMP_MERID_ANGLE_OUT,
    n_points=400,
)
comp_shroud_curve = BezierCurve(
    z_in=COMP_Z_SHROUD_IN,
    z_out=COMP_Z_SHROUD_OUT,
    radius_in=COMP_R_SHROUD_IN,
    radius_out=COMP_R_SHROUD_OUT,
    angle_in=COMP_MERID_ANGLE_IN,
    angle_out=COMP_MERID_ANGLE_OUT,
    n_points=400,
)

comp_hub_z, comp_hub_r = resample_by_arc_length(comp_hub_curve, COMP_N_STREAM)
comp_shroud_z, comp_shroud_r = resample_by_arc_length(comp_shroud_curve, COMP_N_STREAM)

COMP_Z, COMP_R = build_meridional_grid(
    comp_hub_z, comp_hub_r, comp_shroud_z, comp_shroud_r, N_SPAN
)
COMP_INV_RN = spanwise_normal_curvature(COMP_Z, COMP_R)
COMP_DELTA_N = np.hypot(np.diff(COMP_Z, axis=1), np.diff(COMP_R, axis=1))

comp_mid_z = 0.5 * (comp_hub_z + comp_shroud_z)
comp_mid_r = 0.5 * (comp_hub_r + comp_shroud_r)
COMP_GAMMA = midline_tangent_angle(comp_mid_z, comp_mid_r)

COMP_DELTA_S = np.hypot(np.diff(COMP_Z, axis=0), np.diff(COMP_R, axis=0))
COMP_S_STATIONS = np.concatenate(
    [np.zeros((1, N_SPAN)), np.cumsum(COMP_DELTA_S, axis=0)], axis=0
)
COMP_S_MAX = COMP_S_STATIONS[-1, :]

COMP_BETA_IN_SPAN = np.full(N_SPAN, COMP_BETA_IN)
COMP_BETA_OUT_SPAN = np.full(N_SPAN, COMP_BETA_OUT)

surface_comp = SurfaceFlow(
    COMP_N_STREAM,
    N_SPAN,
    beta_law='front_loaded',
    beta_law_kwargs=FRONT_LOADED_KWARGS,
    include_blade_to_blade=False,
    axial_absolute_inflow=True,
)
COMP_BETA_BL, _COMP_BETA_BL_DSHAT = surface_comp.beta_field(
    COMP_S_STATIONS, COMP_S_MAX, COMP_BETA_IN_SPAN, COMP_BETA_OUT_SPAN
)

COMP_U = OMEGA * COMP_R

COMP_GAS_STATE = IdealGasState(
    gamma=GAMMA_AIR, gas_constant=R_AIR, viscosity=VISCOSITY_AIR
)
COMP_GAS_STATE.update(cp.PT_INPUTS, COMP_P0_IN, COMP_T0_IN)
# Upstream absolute total enthalpy (uniform at inlet) and the shared,
# globally uniform isentropic entropy -- NOT a rothalpy: rothalpy depends
# on the local velocity triangle (varies with span) and is conserved
# per-streamline by SurfaceFlow, anchored at the leading edge from this.
COMP_TOTAL_ENTHALPY_IN = COMP_GAS_STATE.hmass()
COMP_ENTROPY_IN = COMP_GAS_STATE.smass()

comp_system = CasadiSystem(num_span=1)
surface_comp.assemble(
    comp_system,
    total_enthalpy=COMP_TOTAL_ENTHALPY_IN,
    entropy=COMP_ENTROPY_IN,
    gas_state=COMP_GAS_STATE,
    omega=OMEGA,
    r=COMP_R,
    gamma=COMP_GAMMA,
    inv_rn=COMP_INV_RN,
    delta_n=COMP_DELTA_N,
)

COMP_BOUNDARY_CONDITIONS = surface_comp.closure_boundary_conditions(
    beta_bl=COMP_BETA_BL, mass_flow=COMP_MASS_FLOW
)
for i in range(COMP_N_STREAM):
    for j in range(N_SPAN):
        n = surface_comp.nodes[surface_comp.node_index(i, j)]
        COMP_BOUNDARY_CONDITIONS[n.geo.RDistr] = COMP_R[i, j]
        COMP_BOUNDARY_CONDITIONS[n.kin.Omega] = OMEGA
        COMP_BOUNDARY_CONDITIONS[n.geo.NumBlades] = COMP_NUM_BLADES
        COMP_BOUNDARY_CONDITIONS[n.geo.BldThick] = COMP_BLADE_THICKNESS
        COMP_BOUNDARY_CONDITIONS[n.geo.MetalAngle] = COMP_BETA_BL[i, j]

comp_system.add_boundary_conditions(COMP_BOUNDARY_CONDITIONS)
comp_system.build()

comp_area_in_est = 2 * np.pi * COMP_R_MEAN_IN * COMP_H_IN
comp_w_guess = COMP_MASS_FLOW / (1.2 * comp_area_in_est)

comp_manual_guess = {}
for i in range(COMP_N_STREAM):
    for j in range(N_SPAN):
        k = surface_comp.node_index(i, j)
        beta_guess = COMP_BETA_BL[i, j]
        wm_guess = comp_w_guess * np.cos(beta_guess)
        wt_guess = comp_w_guess * np.sin(beta_guess)
        u_guess = COMP_U[i, j]

        comp_manual_guess[surface_comp.nodes[k].kin.W_mag] = comp_w_guess
        comp_manual_guess[surface_comp.nodes[k].kin.W_mer] = wm_guess
        comp_manual_guess[surface_comp.nodes[k].kin.W_tan] = wt_guess
        comp_manual_guess[surface_comp.nodes[k].kin.FlowAngleRel] = beta_guess
        comp_manual_guess[surface_comp.nodes[k].kin.BladeSpeed] = u_guess
        comp_manual_guess[surface_comp.nodes[k].kin.V_mer] = wm_guess
        comp_manual_guess[surface_comp.nodes[k].kin.V_tan] = wt_guess + u_guess
        comp_v_mag_guess = np.hypot(wm_guess, wt_guess + u_guess)
        comp_manual_guess[surface_comp.nodes[k].kin.V_mag] = comp_v_mag_guess
        comp_manual_guess[surface_comp.nodes[k].kin.FlowAngleAbs] = np.arctan2(
            wt_guess + u_guess, wm_guess
        )
        comp_manual_guess[surface_comp.nodes[k].stc.Density] = 1.2
        # Floored: at high blade speed this crude guess (V ~ U) can exceed
        # COMP_TOTAL_ENTHALPY_IN, giving a negative/non-physical static
        # enthalpy guess that breaks the EOS at the first solver call.
        comp_manual_guess[surface_comp.nodes[k].stc.Enthalpy] = max(
            COMP_TOTAL_ENTHALPY_IN - 0.5 * comp_v_mag_guess**2,
            0.3 * COMP_TOTAL_ENTHALPY_IN,
        )
        comp_manual_guess[surface_comp.mf_nodes[k].CumFlow] = (
            COMP_MASS_FLOW * j / (N_SPAN - 1)
        )

comp_x0 = comp_system.get_guess(comp_manual_guess, fallback=comp_w_guess)
comp_kn = comp_system.get_boundary_conds()
comp_bnd = comp_system.get_bounds()

comp_rootfinder = comp_system.make_rootfinder(
    'ipopt',
    opts={'error_on_fail': False, 'ipopt.tol': 1e-8, 'ipopt.max_iter': 2000},
)
comp_solution = solve_root_problem(
    comp_rootfinder, comp_x0, comp_kn, comp_bnd, suppress_output=False
)
COMP_STATUS = cast(Any, comp_rootfinder).stats()['return_status']
comp_rootfinder_kinsol = comp_system.make_rootfinder(
    'kinsol', opts={'error_on_fail': False}
)
comp_solution = solve_root_problem(comp_rootfinder_kinsol, comp_solution, comp_kn)
COMP_KINSOL_SUCCESS = cast(Any, comp_rootfinder_kinsol).stats()['success']
comp_sol = comp_system.sol_to_dict(comp_solution)

COMP_W = grid_field(
    surface_comp,
    comp_sol,
    COMP_N_STREAM,
    N_SPAN,
    lambda s, d, i, j: d[s.nodes[s.node_index(i, j)].kin.W_mag],
)
COMP_BETA = grid_field(
    surface_comp,
    comp_sol,
    COMP_N_STREAM,
    N_SPAN,
    lambda s, d, i, j: d[s.nodes[s.node_index(i, j)].kin.FlowAngleRel],
)
# Static enthalpy is now a per-node solved quantity (see
# RothalpyConservation/AbsoluteEnergyClosure), not recomputed from a
# single rothalpy constant -- read it straight from the solution.
COMP_H_STATIC = grid_field(
    surface_comp,
    comp_sol,
    COMP_N_STREAM,
    N_SPAN,
    lambda s, d, i, j: d[s.nodes[s.node_index(i, j)].stc.Enthalpy],
)
COMP_GAS_STATE.update(cp.HmassSmass_INPUTS, COMP_H_STATIC, COMP_ENTROPY_IN)
COMP_P = COMP_GAS_STATE.p()

COMP_V_MAG = grid_field(
    surface_comp,
    comp_sol,
    COMP_N_STREAM,
    N_SPAN,
    lambda s, d, i, j: d[s.nodes[s.node_index(i, j)].kin.V_mag],
)
COMP_V_MER = grid_field(
    surface_comp,
    comp_sol,
    COMP_N_STREAM,
    N_SPAN,
    lambda s, d, i, j: d[s.nodes[s.node_index(i, j)].kin.V_mer],
)
COMP_RHO = grid_field(
    surface_comp,
    comp_sol,
    COMP_N_STREAM,
    N_SPAN,
    lambda s, d, i, j: d[s.nodes[s.node_index(i, j)].stc.Density],
)
# Absolute total enthalpy H0 = h + V^2/2 -- used for the per-component
# power summary (Section 6), not a rothalpy (see module docstring).
COMP_H0_ABS = COMP_H_STATIC + COMP_V_MAG**2 / 2
# Rothalpy I = h + W^2/2 - U^2/2 (Eq. 1.68), recovered in post-processing
# from the solved static enthalpy/relative velocity/blade speed -- it is
# not itself a solved variable, only conserved per-streamline by the
# RothalpyConservation equation (see module docstring).
COMP_ROTHALPY = COMP_H_STATIC + COMP_W**2 / 2 - COMP_U**2 / 2

_comp_p_out_mid = COMP_P[-1, N_SPAN // 2] / 1e3
logger.info(
    f'[Compressor] Flow angle at TE (hub->shroud): {np.degrees(COMP_BETA[-1, :])} deg'
)
logger.info(f'[Compressor] Outlet static p (mid-span): {_comp_p_out_mid:.1f} kPa')

# ============================================================
# 2. Pump-out vane cavity (purely radial, back of the impeller disk)
# ============================================================
POV_N_STREAM = 25

POV_R_IN = 0.040  # [m] inner (hub-side) radius -- flow inlet
POV_R_OUT = 0.083  # [m] outer (casing-side) radius -- flow outlet
POV_H = 0.001  # [m] constant axial gap between disk and casing

POV_MERID_ANGLE = np.radians(90.0)  # purely radial meridional direction

POV_MASS_FLOW = 0.02  # [kg/s]
POV_T_IN = 300.0  # [K]
POV_P_IN = 1.0e5  # [Pa]

POV_NUM_BLADES = 10
POV_BLADE_THICKNESS = 0.0005  # [m], not specified by the user

# Hub/shroud here means "disk side" / "casing side": both run at the same
# radius at every station (purely radial duct), offset only axially by
# +-H/2, since angle_in = angle_out = 90 deg (radial) and the local height
# is constant. The disk (hub) side is the negative-z one so that, once
# placed next to the compressor in the combined plot (Section 4, which
# aligns on POV_Z.min()), it's the disk side that sits against the
# compressor and the casing (shroud) side that ends up on the far side.
POV_Z_HUB = -0.5 * POV_H * np.sin(POV_MERID_ANGLE)
POV_Z_SHROUD = 0.5 * POV_H * np.sin(POV_MERID_ANGLE)

pov_hub_curve = BezierCurve(
    z_in=POV_Z_HUB,
    z_out=POV_Z_HUB,  # no axial progression: purely radial meanline
    radius_in=POV_R_IN,
    radius_out=POV_R_OUT,
    angle_in=POV_MERID_ANGLE,
    angle_out=POV_MERID_ANGLE,
    n_points=400,
)
pov_shroud_curve = BezierCurve(
    z_in=POV_Z_SHROUD,
    z_out=POV_Z_SHROUD,
    radius_in=POV_R_IN,
    radius_out=POV_R_OUT,
    angle_in=POV_MERID_ANGLE,
    angle_out=POV_MERID_ANGLE,
    n_points=400,
)

pov_hub_z, pov_hub_r = resample_by_arc_length(pov_hub_curve, POV_N_STREAM)
pov_shroud_z, pov_shroud_r = resample_by_arc_length(pov_shroud_curve, POV_N_STREAM)

POV_Z, POV_R = build_meridional_grid(
    pov_hub_z, pov_hub_r, pov_shroud_z, pov_shroud_r, N_SPAN
)
POV_INV_RN = spanwise_normal_curvature(POV_Z, POV_R)
POV_DELTA_N = np.hypot(np.diff(POV_Z, axis=1), np.diff(POV_R, axis=1))

pov_mid_z = 0.5 * (pov_hub_z + pov_shroud_z)
pov_mid_r = 0.5 * (pov_hub_r + pov_shroud_r)
POV_GAMMA = midline_tangent_angle(pov_mid_z, pov_mid_r)

POV_DELTA_S = np.hypot(np.diff(POV_Z, axis=0), np.diff(POV_R, axis=0))
POV_S_STATIONS = np.concatenate(
    [np.zeros((1, N_SPAN)), np.cumsum(POV_DELTA_S, axis=0)], axis=0
)
POV_S_MAX = POV_S_STATIONS[-1, :]

POV_BETA_IN_SPAN = np.zeros(N_SPAN)  # straight radial blades
POV_BETA_OUT_SPAN = np.zeros(N_SPAN)

surface_pov = SurfaceFlow(
    POV_N_STREAM,
    N_SPAN,
    beta_law='linear',
    include_blade_to_blade=False,
    axial_absolute_inflow=True,
)
POV_BETA_BL, _POV_BETA_BL_DSHAT = surface_pov.beta_field(
    POV_S_STATIONS, POV_S_MAX, POV_BETA_IN_SPAN, POV_BETA_OUT_SPAN
)

POV_U = OMEGA * POV_R

POV_GAS_STATE = IdealGasState(
    gamma=GAMMA_AIR, gas_constant=R_AIR, viscosity=VISCOSITY_AIR
)
POV_GAS_STATE.update(cp.PT_INPUTS, POV_P_IN, POV_T_IN)
POV_TOTAL_ENTHALPY_IN = POV_GAS_STATE.hmass()
POV_ENTROPY_IN = POV_GAS_STATE.smass()

pov_system = CasadiSystem(num_span=1)
surface_pov.assemble(
    pov_system,
    total_enthalpy=POV_TOTAL_ENTHALPY_IN,
    entropy=POV_ENTROPY_IN,
    gas_state=POV_GAS_STATE,
    omega=OMEGA,
    r=POV_R,
    gamma=POV_GAMMA,
    inv_rn=POV_INV_RN,
    delta_n=POV_DELTA_N,
)

POV_BOUNDARY_CONDITIONS = surface_pov.closure_boundary_conditions(
    beta_bl=POV_BETA_BL, mass_flow=POV_MASS_FLOW
)
for i in range(POV_N_STREAM):
    for j in range(N_SPAN):
        n = surface_pov.nodes[surface_pov.node_index(i, j)]
        POV_BOUNDARY_CONDITIONS[n.geo.RDistr] = POV_R[i, j]
        POV_BOUNDARY_CONDITIONS[n.kin.Omega] = OMEGA
        POV_BOUNDARY_CONDITIONS[n.geo.NumBlades] = POV_NUM_BLADES
        POV_BOUNDARY_CONDITIONS[n.geo.BldThick] = POV_BLADE_THICKNESS
        POV_BOUNDARY_CONDITIONS[n.geo.MetalAngle] = POV_BETA_BL[i, j]

pov_system.add_boundary_conditions(POV_BOUNDARY_CONDITIONS)
pov_system.build()

pov_area_in_est = 2 * np.pi * POV_R_IN * POV_H
pov_w_guess = POV_MASS_FLOW / (1.2 * pov_area_in_est)

pov_manual_guess = {}
for i in range(POV_N_STREAM):
    for j in range(N_SPAN):
        k = surface_pov.node_index(i, j)
        beta_guess = POV_BETA_BL[i, j]
        wm_guess = pov_w_guess * np.cos(beta_guess)
        wt_guess = pov_w_guess * np.sin(beta_guess)
        u_guess = POV_U[i, j]

        pov_manual_guess[surface_pov.nodes[k].kin.W_mag] = pov_w_guess
        pov_manual_guess[surface_pov.nodes[k].kin.W_mer] = wm_guess
        pov_manual_guess[surface_pov.nodes[k].kin.W_tan] = wt_guess
        pov_manual_guess[surface_pov.nodes[k].kin.FlowAngleRel] = beta_guess
        pov_manual_guess[surface_pov.nodes[k].kin.BladeSpeed] = u_guess
        pov_manual_guess[surface_pov.nodes[k].kin.V_mer] = wm_guess
        pov_manual_guess[surface_pov.nodes[k].kin.V_tan] = wt_guess + u_guess
        pov_v_mag_guess = np.hypot(wm_guess, wt_guess + u_guess)
        pov_manual_guess[surface_pov.nodes[k].kin.V_mag] = pov_v_mag_guess
        pov_manual_guess[surface_pov.nodes[k].kin.FlowAngleAbs] = np.arctan2(
            wt_guess + u_guess, wm_guess
        )
        pov_manual_guess[surface_pov.nodes[k].stc.Density] = 1.2
        pov_manual_guess[surface_pov.nodes[k].stc.Enthalpy] = max(
            POV_TOTAL_ENTHALPY_IN - 0.5 * pov_v_mag_guess**2,
            0.3 * POV_TOTAL_ENTHALPY_IN,
        )
        pov_manual_guess[surface_pov.mf_nodes[k].CumFlow] = (
            POV_MASS_FLOW * j / (N_SPAN - 1)
        )

pov_x0 = pov_system.get_guess(pov_manual_guess, fallback=pov_w_guess)
pov_kn = pov_system.get_boundary_conds()
pov_bnd = pov_system.get_bounds()

pov_rootfinder = pov_system.make_rootfinder(
    'ipopt',
    opts={'error_on_fail': False, 'ipopt.tol': 1e-8, 'ipopt.max_iter': 2000},
)
pov_solution = solve_root_problem(
    pov_rootfinder, pov_x0, pov_kn, pov_bnd, suppress_output=False
)
POV_STATUS = cast(Any, pov_rootfinder).stats()['return_status']
pov_rootfinder_kinsol = pov_system.make_rootfinder(
    'kinsol', opts={'error_on_fail': False}
)
pov_solution = solve_root_problem(pov_rootfinder_kinsol, pov_solution, pov_kn)
POV_KINSOL_SUCCESS = cast(Any, pov_rootfinder_kinsol).stats()['success']
pov_sol = pov_system.sol_to_dict(pov_solution)

POV_H_STATIC = grid_field(
    surface_pov,
    pov_sol,
    POV_N_STREAM,
    N_SPAN,
    lambda s, d, i, j: d[s.nodes[s.node_index(i, j)].stc.Enthalpy],
)
POV_GAS_STATE.update(cp.HmassSmass_INPUTS, POV_H_STATIC, POV_ENTROPY_IN)
POV_P = POV_GAS_STATE.p()

POV_V_MAG = grid_field(
    surface_pov,
    pov_sol,
    POV_N_STREAM,
    N_SPAN,
    lambda s, d, i, j: d[s.nodes[s.node_index(i, j)].kin.V_mag],
)
POV_V_MER = grid_field(
    surface_pov,
    pov_sol,
    POV_N_STREAM,
    N_SPAN,
    lambda s, d, i, j: d[s.nodes[s.node_index(i, j)].kin.V_mer],
)
POV_W = grid_field(
    surface_pov,
    pov_sol,
    POV_N_STREAM,
    N_SPAN,
    lambda s, d, i, j: d[s.nodes[s.node_index(i, j)].kin.W_mag],
)
POV_RHO = grid_field(
    surface_pov,
    pov_sol,
    POV_N_STREAM,
    N_SPAN,
    lambda s, d, i, j: d[s.nodes[s.node_index(i, j)].stc.Density],
)
POV_H0_ABS = POV_H_STATIC + POV_V_MAG**2 / 2
POV_ROTHALPY = POV_H_STATIC + POV_W**2 / 2 - POV_U**2 / 2

_pov_p_out_mid = POV_P[-1, N_SPAN // 2] / 1e3
logger.info(f'[Pump-out vane] Outlet static p (mid-span): {_pov_p_out_mid:.1f} kPa')

# ============================================================
# 3. Turbine rotor (radial inflow, mounted facing the opposite direction)
# ============================================================
TURB_N_STREAM = 40
TURB_N_SPAN = 14

TURB_R_MEAN_IN = 0.063  # [m] inlet (inducer) tip radius
TURB_H_IN = 0.016  # [m] inlet blade height

TURB_R_MEAN_OUT = 0.040  # [m] exducer mean radius
TURB_H_OUT = 0.030  # [m] exducer blade height

TURB_AXIAL_LENGTH = 0.039  # [m], not specified by the user

TURB_MERID_ANGLE_IN = np.radians(-90.0)  # radial inflow
TURB_MERID_ANGLE_OUT = np.radians(0.0)  # axial outflow (exducer)

TURB_MASS_FLOW = 0.75  # [kg/s], same as the compressor
TURB_T0_IN = 363.0  # [K]
TURB_P0_IN = 1.45e5  # [Pa]
TURB_P_IN = 0.88e5

TURB_BETA_IN = np.radians(-25.0)  # radial at the inlet
TURB_BETA_OUT = np.radians(-55.0)
TURB_NUM_BLADES = 16  # not specified by the user
TURB_BLADE_THICKNESS = 0.5 * (0.00086 + 0.00076)  # [m], not specified by the user

TURB_INCIDENCE_LE = np.radians(0.0)  # perfectly radial relative flow at the LE

TURB_Z_HUB_IN = 0.5 * TURB_H_IN * np.sin(TURB_MERID_ANGLE_IN)
TURB_R_HUB_IN = TURB_R_MEAN_IN - 0.5 * TURB_H_IN * np.cos(TURB_MERID_ANGLE_IN)
TURB_Z_SHROUD_IN = -0.5 * TURB_H_IN * np.sin(TURB_MERID_ANGLE_IN)
TURB_R_SHROUD_IN = TURB_R_MEAN_IN + 0.5 * TURB_H_IN * np.cos(TURB_MERID_ANGLE_IN)

TURB_Z_HUB_OUT = TURB_AXIAL_LENGTH - 0.5 * TURB_H_OUT * np.sin(TURB_MERID_ANGLE_OUT)
TURB_R_HUB_OUT = TURB_R_MEAN_OUT - 0.5 * TURB_H_OUT * np.cos(TURB_MERID_ANGLE_OUT)
TURB_Z_SHROUD_OUT = TURB_AXIAL_LENGTH + 0.5 * TURB_H_OUT * np.sin(TURB_MERID_ANGLE_OUT)
TURB_R_SHROUD_OUT = TURB_R_MEAN_OUT + 0.5 * TURB_H_OUT * np.cos(TURB_MERID_ANGLE_OUT)

turb_hub_curve = BezierCurve(
    z_in=TURB_Z_HUB_IN,
    z_out=TURB_Z_HUB_OUT,
    radius_in=TURB_R_HUB_IN,
    radius_out=TURB_R_HUB_OUT,
    angle_in=TURB_MERID_ANGLE_IN,
    angle_out=TURB_MERID_ANGLE_OUT,
    n_points=400,
)
turb_shroud_curve = BezierCurve(
    z_in=TURB_Z_SHROUD_IN,
    z_out=TURB_Z_SHROUD_OUT,
    radius_in=TURB_R_SHROUD_IN,
    radius_out=TURB_R_SHROUD_OUT,
    angle_in=TURB_MERID_ANGLE_IN,
    angle_out=TURB_MERID_ANGLE_OUT,
    n_points=400,
)

turb_hub_z, turb_hub_r = resample_by_arc_length(turb_hub_curve, TURB_N_STREAM)
turb_shroud_z, turb_shroud_r = resample_by_arc_length(turb_shroud_curve, TURB_N_STREAM)

TURB_Z, TURB_R = build_meridional_grid(
    turb_hub_z, turb_hub_r, turb_shroud_z, turb_shroud_r, TURB_N_SPAN
)
TURB_INV_RN = spanwise_normal_curvature(TURB_Z, TURB_R)
TURB_DELTA_N = np.hypot(np.diff(TURB_Z, axis=1), np.diff(TURB_R, axis=1))

# Geometry check, plotted before the turbine system is assembled/solved.
# if __name__ == '__main__':
#     fig_turb_geom, ax_turb_geom = plt.subplots(figsize=(5.0, 4.0))
#     ax_turb_geom.plot(turb_hub_z, turb_hub_r, color='k', linewidth=1.5, label='hub')
#     ax_turb_geom.plot(
#         turb_shroud_z,
#         turb_shroud_r,
#         color='k',
#         linewidth=1.5,
#         linestyle='--',
#         label='shroud',
#     )
#     for i in range(0, TURB_N_STREAM, max(1, TURB_N_STREAM // 10)):
#         ax_turb_geom.plot(TURB_Z[i, :], TURB_R[i, :], color='tab:blue', linewidth=0.4)
#     ax_turb_geom.set_xlabel('z [m]')
#     ax_turb_geom.set_ylabel('R [m]')
#     ax_turb_geom.set_aspect('equal')
#     ax_turb_geom.set_title('Turbine meridional geometry (pre-solve)')
#     ax_turb_geom.legend()
#     fig_turb_geom.tight_layout()
#     plt.show()

TURB_DELTA_S = np.hypot(np.diff(TURB_Z, axis=0), np.diff(TURB_R, axis=0))
TURB_S_STATIONS = np.concatenate(
    [np.zeros((1, TURB_N_SPAN)), np.cumsum(TURB_DELTA_S, axis=0)], axis=0
)
TURB_S_MAX = TURB_S_STATIONS[-1, :]

TURB_BETA_IN_SPAN = np.full(TURB_N_SPAN, TURB_BETA_IN)
TURB_BETA_OUT_SPAN = np.full(TURB_N_SPAN, TURB_BETA_OUT)

surface_turb = SurfaceFlow(
    TURB_N_STREAM,
    TURB_N_SPAN,
    rotation_sign=-1.0,
    beta_law='rear_loaded',
    beta_law_kwargs=FRONT_LOADED_KWARGS,
    include_blade_to_blade=False,
    # axial_absolute_inflow=True,
)
TURB_BETA_BL, _TURB_BETA_BL_DSHAT = surface_turb.beta_field(
    TURB_S_STATIONS, TURB_S_MAX, TURB_BETA_IN_SPAN, TURB_BETA_OUT_SPAN
)

TURB_U = OMEGA * TURB_R

TURB_GAS_STATE = IdealGasState(
    gamma=GAMMA_AIR, gas_constant=R_AIR, viscosity=VISCOSITY_AIR
)
TURB_GAS_STATE.update(cp.PT_INPUTS, TURB_P0_IN, TURB_T0_IN)
TURB_TOTAL_ENTHALPY_IN = TURB_GAS_STATE.hmass()
TURB_ENTROPY_IN = TURB_GAS_STATE.smass()

TURB_GAS_STATE.update(cp.PSmass_INPUTS, TURB_P_IN, TURB_ENTROPY_IN)
TURB_RHO_GUESS = TURB_GAS_STATE.rhomass()

turb_system = CasadiSystem(num_span=1)
surface_turb.assemble(
    turb_system,
    total_enthalpy=TURB_TOTAL_ENTHALPY_IN,
    entropy=TURB_ENTROPY_IN,
    gas_state=TURB_GAS_STATE,
    omega=OMEGA,
    r=TURB_R,
    gamma=midline_tangent_angle(
        0.5 * (turb_hub_z + turb_shroud_z), 0.5 * (turb_hub_r + turb_shroud_r)
    ),
    inv_rn=TURB_INV_RN,
    delta_n=TURB_DELTA_N,
)

TURB_BOUNDARY_CONDITIONS = surface_turb.closure_boundary_conditions(
    beta_bl=TURB_BETA_BL, mass_flow=TURB_MASS_FLOW, incidence_le=TURB_INCIDENCE_LE
)
for i in range(TURB_N_STREAM):
    for j in range(TURB_N_SPAN):
        n = surface_turb.nodes[surface_turb.node_index(i, j)]
        TURB_BOUNDARY_CONDITIONS[n.geo.RDistr] = TURB_R[i, j]
        TURB_BOUNDARY_CONDITIONS[n.kin.Omega] = OMEGA
        TURB_BOUNDARY_CONDITIONS[n.geo.NumBlades] = TURB_NUM_BLADES
        TURB_BOUNDARY_CONDITIONS[n.geo.BldThick] = TURB_BLADE_THICKNESS
        TURB_BOUNDARY_CONDITIONS[n.geo.MetalAngle] = TURB_BETA_BL[i, j]

turb_system.add_boundary_conditions(TURB_BOUNDARY_CONDITIONS)
turb_system.build()

turb_area_in_est = 2 * np.pi * TURB_R_MEAN_IN * TURB_H_IN
turb_w_guess = TURB_MASS_FLOW / (TURB_RHO_GUESS * turb_area_in_est)

turb_manual_guess = {}
for i in range(TURB_N_STREAM):
    for j in range(TURB_N_SPAN):
        k = surface_turb.node_index(i, j)
        beta_guess = TURB_BETA_BL[i, j]
        wm_guess = turb_w_guess * np.cos(beta_guess)

        wt_guess = turb_w_guess * np.sin(beta_guess)
        u_guess = TURB_U[i, j]

        turb_manual_guess[surface_turb.nodes[k].kin.W_mag] = turb_w_guess
        turb_manual_guess[surface_turb.nodes[k].kin.W_mer] = wm_guess
        turb_manual_guess[surface_turb.nodes[k].kin.W_tan] = wt_guess
        turb_manual_guess[surface_turb.nodes[k].kin.FlowAngleRel] = beta_guess
        turb_manual_guess[surface_turb.nodes[k].kin.BladeSpeed] = u_guess
        turb_manual_guess[surface_turb.nodes[k].kin.V_mer] = wm_guess
        turb_manual_guess[surface_turb.nodes[k].kin.V_tan] = wt_guess + u_guess
        turb_v_mag_guess = np.hypot(wm_guess, wt_guess + u_guess)
        turb_manual_guess[surface_turb.nodes[k].kin.V_mag] = turb_v_mag_guess
        turb_manual_guess[surface_turb.nodes[k].kin.FlowAngleAbs] = np.arctan2(
            wt_guess + u_guess, wm_guess
        )
        turb_manual_guess[surface_turb.nodes[k].stc.Density] = TURB_RHO_GUESS
        turb_manual_guess[surface_turb.nodes[k].stc.Enthalpy] = (
            TURB_TOTAL_ENTHALPY_IN - 0.5 * turb_v_mag_guess**2
        )
        turb_manual_guess[surface_turb.mf_nodes[k].CumFlow] = (
            TURB_MASS_FLOW * j / (TURB_N_SPAN - 1)
        )

turb_x0 = turb_system.get_guess(turb_manual_guess, fallback=turb_w_guess)
turb_kn = turb_system.get_boundary_conds()
turb_bnd = turb_system.get_bounds()

turb_rootfinder = turb_system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        'ipopt.tol': 1e-8,
        'ipopt.max_iter': 2000,
        'ipopt.mu_strategy': 'adaptive',
    },
)
turb_solution = solve_root_problem(
    turb_rootfinder, turb_x0, turb_kn, turb_bnd, suppress_output=False
)
TURB_STATUS = cast(Any, turb_rootfinder).stats()['return_status']
turb_rootfinder_polish = turb_system.make_rootfinder(
    'kinsol', opts={'error_on_fail': False}
)
turb_solution = solve_root_problem(
    turb_rootfinder_polish, turb_solution, turb_kn, turb_bnd
)
TURB_KINSOL_SUCCESS = cast(Any, turb_rootfinder_polish).stats()['success']
turb_sol = turb_system.sol_to_dict(turb_solution)

TURB_H_STATIC = grid_field(
    surface_turb,
    turb_sol,
    TURB_N_STREAM,
    TURB_N_SPAN,
    lambda s, d, i, j: d[s.nodes[s.node_index(i, j)].stc.Enthalpy],
)
TURB_GAS_STATE.update(cp.HmassSmass_INPUTS, TURB_H_STATIC, TURB_ENTROPY_IN)
TURB_P = TURB_GAS_STATE.p()

TURB_V_MAG = grid_field(
    surface_turb,
    turb_sol,
    TURB_N_STREAM,
    TURB_N_SPAN,
    lambda s, d, i, j: d[s.nodes[s.node_index(i, j)].kin.V_mag],
)
TURB_V_MER = grid_field(
    surface_turb,
    turb_sol,
    TURB_N_STREAM,
    TURB_N_SPAN,
    lambda s, d, i, j: d[s.nodes[s.node_index(i, j)].kin.V_mer],
)
TURB_W = grid_field(
    surface_turb,
    turb_sol,
    TURB_N_STREAM,
    TURB_N_SPAN,
    lambda s, d, i, j: d[s.nodes[s.node_index(i, j)].kin.W_mag],
)
TURB_RHO = grid_field(
    surface_turb,
    turb_sol,
    TURB_N_STREAM,
    TURB_N_SPAN,
    lambda s, d, i, j: d[s.nodes[s.node_index(i, j)].stc.Density],
)
TURB_H0_ABS = TURB_H_STATIC + TURB_V_MAG**2 / 2
TURB_ROTHALPY = TURB_H_STATIC + TURB_W**2 / 2 - TURB_U**2 / 2

TURB_N_COORD = span_coord(TURB_DELTA_N)
_turb_p_out_mass_avg = mass_flux_weighted_mean(
    TURB_P[-1, :],
    TURB_RHO[-1, :],
    TURB_V_MER[-1, :],
    TURB_R[-1, :],
    TURB_N_COORD[-1, :],
)
_turb_p_bar = _turb_p_out_mass_avg / 1e5
logger.info(f'[Turbine] Mass-averaged outlet static pressure: {_turb_p_bar:.3f} bar')

# ============================================================
# 4. Plots: the compressor and the pump-out-vane cavity share one panel
# and one colorbar (the cavity sits right behind the impeller disk, so
# its local z is shifted by COMP_AXIAL_LENGTH to place it there); the
# turbine is physically a separate, unconnected flow path, so it gets
# its own panel/colorbar. One such figure is made per plotted field
# (static pressure, absolute total enthalpy, rothalpy, relative velocity).
# ============================================================
if __name__ == '__main__':
    PANEL_W, PANEL_H = 5.0, 4.0  # [in]

    # Shift the pump-out-vane grid so its leftmost point sits exactly
    # IMPELLER_DISK_TIP_THICKNESS to the right of the compressor's
    # rightmost point -- rather than COMP_AXIAL_LENGTH, which would put
    # it level with (and overlapping) the compressor's trailing edge,
    # since that also spans +-H_OUT/2 in z around COMP_AXIAL_LENGTH.
    POV_Z_OFFSET = (COMP_Z.max() - POV_Z.min()) + IMPELLER_DISK_TIP_THICKNESS
    POV_Z_GLOBAL = POV_Z + POV_Z_OFFSET
    pov_hub_z_global = pov_hub_z + POV_Z_OFFSET
    pov_shroud_z_global = pov_shroud_z + POV_Z_OFFSET

    # Give both panels the same z/R span (with a little margin), each
    # anchored on its own domain's minimum -- so, with equal aspect and
    # equal subplot size, one meter looks the same on screen in both
    # panels and the compressor/turbine are shown at true relative scale,
    # rather than each independently auto-zoomed to fill its own panel.
    comp_pov_z_min = min(COMP_Z.min(), POV_Z_GLOBAL.min())
    comp_pov_z_max = max(COMP_Z.max(), POV_Z_GLOBAL.max())
    comp_pov_r_min = min(COMP_R.min(), POV_R.min())
    comp_pov_r_max = max(COMP_R.max(), POV_R.max())

    z_span = max(comp_pov_z_max - comp_pov_z_min, TURB_Z.max() - TURB_Z.min())
    r_span = max(comp_pov_r_max - comp_pov_r_min, TURB_R.max() - TURB_R.min())
    margin = 0.05 * max(z_span, r_span)
    z_span += margin
    r_span += margin

    def plot_meridional_field(
        field_title, unit_label, comp_field, pov_field, turb_field, cmap='viridis'
    ):
        """One figure in the compressor+pump-out-vane / turbine two-panel
        layout above, for an arbitrary scalar field given as (compressor,
        pump-out-vane, turbine) grids already in the units of ``unit_label``.
        """
        f_min = min(comp_field.min(), pov_field.min())
        f_max = max(comp_field.max(), pov_field.max())
        shared_levels = np.linspace(f_min, f_max, 31)

        fig, (ax_comp_pov, ax_turb) = plt.subplots(1, 2, figsize=(2 * PANEL_W, PANEL_H))

        ax_comp_pov.contourf(
            COMP_Z, COMP_R, comp_field, levels=shared_levels, cmap=cmap
        )
        ax_comp_pov.plot(comp_hub_z, comp_hub_r, color='k', linewidth=1.5)
        ax_comp_pov.plot(
            comp_shroud_z, comp_shroud_r, color='k', linewidth=1.5, linestyle='--'
        )
        for i in range(0, COMP_N_STREAM, max(1, COMP_N_STREAM // 10)):
            ax_comp_pov.plot(COMP_Z[i, :], COMP_R[i, :], color='white', linewidth=0.4)

        contour_pov = ax_comp_pov.contourf(
            POV_Z_GLOBAL, POV_R, pov_field, levels=shared_levels, cmap=cmap
        )
        ax_comp_pov.plot(pov_hub_z_global, pov_hub_r, color='k', linewidth=1.5)
        ax_comp_pov.plot(
            pov_shroud_z_global, pov_shroud_r, color='k', linewidth=1.5, linestyle='--'
        )
        for i in range(0, POV_N_STREAM, max(1, POV_N_STREAM // 10)):
            ax_comp_pov.plot(
                POV_Z_GLOBAL[i, :], POV_R[i, :], color='white', linewidth=0.4
            )

        ax_comp_pov.set_xlabel('z [m]')
        ax_comp_pov.set_ylabel('R [m]')
        ax_comp_pov.set_aspect('equal')
        ax_comp_pov.set_title('Compressor + pump-out-vane cavity')
        fig.colorbar(contour_pov, ax=ax_comp_pov, label=f'{field_title} [{unit_label}]')

        contour_turb = ax_turb.contourf(
            TURB_Z, TURB_R, turb_field, levels=30, cmap=cmap
        )
        ax_turb.plot(turb_hub_z, turb_hub_r, color='k', linewidth=1.5)
        ax_turb.plot(
            turb_shroud_z, turb_shroud_r, color='k', linewidth=1.5, linestyle='--'
        )
        for i in range(0, TURB_N_STREAM, max(1, TURB_N_STREAM // 10)):
            ax_turb.plot(TURB_Z[i, :], TURB_R[i, :], color='white', linewidth=0.4)
        ax_turb.set_xlabel('z [m]')
        ax_turb.set_ylabel('R [m]')
        ax_turb.set_aspect('equal')
        ax_turb.set_title('Turbine')
        fig.colorbar(contour_turb, ax=ax_turb, label=f'{field_title} [{unit_label}]')

        ax_comp_pov.set_xlim(
            comp_pov_z_min - margin / 2, comp_pov_z_min - margin / 2 + z_span
        )
        ax_comp_pov.set_ylim(
            comp_pov_r_min - margin / 2, comp_pov_r_min - margin / 2 + r_span
        )
        ax_turb.set_xlim(TURB_Z.min() - margin / 2, TURB_Z.min() - margin / 2 + z_span)
        ax_turb.set_ylim(TURB_R.min() - margin / 2, TURB_R.min() - margin / 2 + r_span)

        fig.suptitle(f'Meridional-plane {field_title.lower()}')
        fig.tight_layout()
        return fig

    plot_meridional_field(
        'Static pressure $p$', 'kPa', COMP_P / 1e3, POV_P / 1e3, TURB_P / 1e3
    )
    plot_meridional_field(
        'Absolute total enthalpy $H_0$',
        'kJ/kg',
        COMP_H0_ABS / 1e3,
        POV_H0_ABS / 1e3,
        TURB_H0_ABS / 1e3,
    )
    plot_meridional_field(
        'Rothalpy $I$',
        'kJ/kg',
        COMP_ROTHALPY / 1e3,
        POV_ROTHALPY / 1e3,
        TURB_ROTHALPY / 1e3,
    )
    plot_meridional_field('Relative velocity $W$', 'm/s', COMP_W, POV_W, TURB_W)

    # Blade metal angle distribution vs. the normalized meridional (streamwise)
    # coordinate s/s_max, one subplot per component -- the boundary condition
    # each node's FlowAngleRel is pinned to (meridional-only mode), rather than
    # a field read back from the solution.
    _SPAN_SERIES = (('hub', '#2a78d6'), ('mean', '#eb6834'), ('shroud', '#1baf7a'))

    def plot_blade_angle(ax, s_stations, s_max, beta_bl, title):
        n_span_local = beta_bl.shape[1]
        span_idx = (0, (n_span_local - 1) // 2, n_span_local - 1)
        s_norm = s_stations / s_max
        for j, (label, color) in zip(span_idx, _SPAN_SERIES):
            ax.plot(s_norm[:, j], np.degrees(beta_bl[:, j]), color=color, label=label)
        ax.set_xlabel('Normalized meridional coordinate $s/s_{max}$ [-]')
        ax.set_ylabel(r'Blade metal angle $\beta_{bl}$ [deg]')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, color='#e1e0d9', linewidth=0.8)

    fig_blade, (ax_blade_comp, ax_blade_pov, ax_blade_turb) = plt.subplots(
        1, 3, figsize=(3 * PANEL_W, PANEL_H)
    )
    plot_blade_angle(
        ax_blade_comp, COMP_S_STATIONS, COMP_S_MAX, COMP_BETA_BL, 'Compressor'
    )
    plot_blade_angle(
        ax_blade_pov, POV_S_STATIONS, POV_S_MAX, POV_BETA_BL, 'Pump-out vane'
    )
    plot_blade_angle(
        ax_blade_turb, TURB_S_STATIONS, TURB_S_MAX, TURB_BETA_BL, 'Turbine'
    )
    fig_blade.suptitle('Blade metal angle distribution')
    fig_blade.tight_layout()

    plt.show()

# ============================================================
# 5. Performance summary -- total-static pressure ratio (compressor,
# pump-out vane) / total-static expansion ratio (turbine), from the
# inlet total pressure vs. the outlet static pressure at mid-span; and
# each component's power, mass_flow * the mass-flux-weighted spanwise
# average of the absolute total enthalpy change from inlet to outlet
# (positive = work added to the fluid, negative = work extracted from it).
# ============================================================
COMP_N_COORD = span_coord(COMP_DELTA_N)
COMP_H0_IN = mass_flux_weighted_mean(
    COMP_H0_ABS[0, :],
    COMP_RHO[0, :],
    COMP_V_MER[0, :],
    COMP_R[0, :],
    COMP_N_COORD[0, :],
)
COMP_H0_OUT = mass_flux_weighted_mean(
    COMP_H0_ABS[-1, :],
    COMP_RHO[-1, :],
    COMP_V_MER[-1, :],
    COMP_R[-1, :],
    COMP_N_COORD[-1, :],
)
COMP_POWER_KW = COMP_MASS_FLOW * (COMP_H0_OUT - COMP_H0_IN) / 1e3
COMP_PR_TS = COMP_P[-1, N_SPAN // 2] / COMP_P0_IN

POV_N_COORD = span_coord(POV_DELTA_N)
POV_H0_IN = mass_flux_weighted_mean(
    POV_H0_ABS[0, :], POV_RHO[0, :], POV_V_MER[0, :], POV_R[0, :], POV_N_COORD[0, :]
)
POV_H0_OUT = mass_flux_weighted_mean(
    POV_H0_ABS[-1, :],
    POV_RHO[-1, :],
    POV_V_MER[-1, :],
    POV_R[-1, :],
    POV_N_COORD[-1, :],
)
POV_POWER_KW = POV_MASS_FLOW * (POV_H0_OUT - POV_H0_IN) / 1e3
POV_PR_TS = POV_P[-1, N_SPAN // 2] / POV_P_IN

TURB_H0_IN = mass_flux_weighted_mean(
    TURB_H0_ABS[0, :],
    TURB_RHO[0, :],
    TURB_V_MER[0, :],
    TURB_R[0, :],
    TURB_N_COORD[0, :],
)
TURB_H0_OUT = mass_flux_weighted_mean(
    TURB_H0_ABS[-1, :],
    TURB_RHO[-1, :],
    TURB_V_MER[-1, :],
    TURB_R[-1, :],
    TURB_N_COORD[-1, :],
)
TURB_POWER_KW = TURB_MASS_FLOW * (TURB_H0_OUT - TURB_H0_IN) / 1e3
TURB_ER_TS = TURB_P0_IN / TURB_P[-1, TURB_N_SPAN // 2]

logger.info('=== Performance summary ===')
logger.info(
    f'Compressor: total-static PR={COMP_PR_TS:.2f}, power={COMP_POWER_KW:+.1f} kW'
)
logger.info(
    f'Pump-out vane: total-static PR={POV_PR_TS:.2f}, power={POV_POWER_KW:+.1f} kW'
)
logger.info(
    f'Turbine: total-static expansion ratio={TURB_ER_TS:.2f}, '
    f'power={TURB_POWER_KW:+.1f} kW'
)

# ============================================================
# 6. Convergence summary -- the reported verdict is KINSOL's own exit
# status on the polish step (the solution actually used downstream), with
# IPOPT's status on the main (non-polish) solve that fed it also shown in
# brackets for context, so it's obvious at a glance which block(s) need
# attention.
# ============================================================
logger.info('=== Convergence summary ===')
for _name, _status, _kinsol_ok in (
    ('Compressor', COMP_STATUS, COMP_KINSOL_SUCCESS),
    ('Pump-out vane', POV_STATUS, POV_KINSOL_SUCCESS),
    ('Turbine', TURB_STATUS, TURB_KINSOL_SUCCESS),
):
    _verdict = 'yes' if _kinsol_ok else 'NO'
    logger.info(
        f'{_name}: converged={_verdict} '
        f'(kinsol success: {_kinsol_ok}, ipopt status: {_status})'
    )
