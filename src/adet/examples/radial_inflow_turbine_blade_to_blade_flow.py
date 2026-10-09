"""
Coupled meridional (hub-to-shroud) + blade-to-blade flow of a radial-inflow
turbine (RIT) rotor.
"""

import logging

import CoolProp as cp
import numpy as np
from matplotlib import pyplot as plt

from adet.assemblers import CasadiSystem
from adet.equations.fundamental import Kinematics
from adet.equations.throughflow import (
    BladeLoadingVariables,
    BladeSurfacePressures,
    BladeToBladeLoading,
    CarryLoadingToLE,
    LocalThermoState,
    MassFluxAccumulation,
    MeridionalFlowVariables,
    MeridionalMomentumMarch,
    SlipTransition,
    SuctionPressureVelocities,
    build_meridional_grid,
    front_loaded_beta_law,
    midline_tangent_angle,
    resample_by_arc_length,
    spanwise_normal_curvature,
)
from adet.fluid.ideal_eos import IdealGasState
from adet.geometry import BezierCurve
from adet.losses.basic import ZeroDeviation
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables

logger = logging.getLogger(__name__)
setup_logger(logger)

# ============================================================
# 1. User settings and problem data
# ============================================================
R2_TIP = 0.0582  # [m] rotor inlet (inducer) tip radius
H_IN = 0.00635  # [m] rotor inlet blade height
R_HUB_IN = R2_TIP - H_IN  # [m] rotor inlet hub radius
R_MEAN_IN = 0.5 * (R2_TIP + R_HUB_IN)  # [m] mean-line radius at the LE

R3_HUB = 0.01524  # [m] exducer hub radius
R3_TIP = 0.03683  # [m] exducer tip radius
H_OUT = R3_TIP - R3_HUB  # [m] exit channel height
R_MEAN_OUT = 0.5 * (R3_HUB + R3_TIP)  # [m] mean-line radius at the exducer

AXIAL_LENGTH = 0.0389  # [m] rotor axial length

# BezierCurve convention (0 = axial +z, +90 = radial outward): purely
# radial inflow at the LE, purely axial at the exducer exit
MERID_ANGLE_IN = np.radians(-90.0)  # [rad]
MERID_ANGLE_OUT = np.radians(0.0)  # [rad]

N_STREAM = 25  # number of streamwise stations (LE = 0, TE = N_STREAM - 1)
N_SPAN = 5  # number of spanwise stations (hub to shroud)

RPM = 106_588.0  # [rev/min]
OMEGA = RPM * 2 * np.pi / 60  # [rad/s]

MASS_FLOW = 0.33  # [kg/s]

GAMMA_AIR = 1.4
R_AIR = 287.05  # [J / (kg K)]
VISCOSITY_AIR = 1.8e-5  # [Pa s], unused but required by IdealGasState

# Representative rotor-inlet velocity triangle (Jones Fig. 4/Sauret Fig. 8,
# same design values used as in T100_RIT.py initial guess) used only to
# compute the global rothalpy constant (ROTHALPY)
U2_TIP = 649.224  # [m/s] rotor inlet tip speed
VTHETA_OVER_U = 0.882  # [-] Jones Fig. 4 inlet triangle
BETA4_DESIGN = np.radians(-32.7)  # [rad] Chen & Baines optimal incidence
T_STATIC_IN_DESIGN = 906.7  # [K] rotor-inlet static temperature
P_STATIC_IN_DESIGN = 2.948e5  # [Pa] rotor-inlet static pressurei

# Blade geometry: purely radial at the LE, backswept at the exducer.
BETA_HUB_IN = np.radians(0.0)  # [rad] blade metal angle at the hub, LE
BETA_SHROUD_IN = np.radians(0.0)  # [rad] blade metal angle at the shroud, LE
BETA_HUB_OUT = np.radians(-57.3)  # [rad] blade metal angle at the hub, TE
BETA_SHROUD_OUT = np.radians(-57.3)  # [rad] blade metal angle at the shroud, TE

# FRONT-LOADED TWO-SLOPE BLADE-ANGLE LAW (see front_loaded_beta_law())
PIECEWISE_SPLIT_FRAC = 0.5  # [s_hat] where the slope changes
PIECEWISE_FRAC_AT_SPLIT = 0.88  # fraction of (beta_out - beta_in) covered by the split
PIECEWISE_BLEND_HALF_WIDTH = 0.12  # [s_hat] half-width of the smoothing window

NUM_BLADES = 16
BLADE_THICKNESS = 0.5 * (0.00086 + 0.00076)  # [m] LE/TE mean

# ------------------------------------------------------------
# LE/TE closure toggles, independently switchable at each end
# ------------------------------------------------------------
USE_SLIP_TRANSITION_LE = True  # blend a deviated LE incidence onto the law
USE_SLIP_TRANSITION_TE = True  # blend the law onto the free TE value
USE_KUTTA_LE = True  # assume zero blade loading (DeltaW=0, W_SS=W_PS) at the LE
USE_KUTTA_TE = True  # assume zero blade loading (DeltaW=0, W_SS=W_PS) at the TE

# LE incidence (deviation of the flow angle from the blade metal angle at
# s_hat=0)
INCIDENCE_LE = BETA4_DESIGN - BETA_HUB_IN  # [rad]

TRANSITION_FRAC_LE = 0.25
TRANSITION_FRAC_TE = 0.6

# Hub and shroud, offset by +-H/2 along the local normal to the prescribed
# meridional angle.
Z_HUB_IN = 0.5 * H_IN * np.sin(MERID_ANGLE_IN)
R_HUB_IN_OFF = R_MEAN_IN - 0.5 * H_IN * np.cos(MERID_ANGLE_IN)
Z_SHROUD_IN = -0.5 * H_IN * np.sin(MERID_ANGLE_IN)
R_SHROUD_IN_OFF = R_MEAN_IN + 0.5 * H_IN * np.cos(MERID_ANGLE_IN)

Z_HUB_OUT = AXIAL_LENGTH + 0.5 * H_OUT * np.sin(MERID_ANGLE_OUT)
R_HUB_OUT_OFF = R_MEAN_OUT - 0.5 * H_OUT * np.cos(MERID_ANGLE_OUT)
Z_SHROUD_OUT = AXIAL_LENGTH - 0.5 * H_OUT * np.sin(MERID_ANGLE_OUT)
R_SHROUD_OUT_OFF = R_MEAN_OUT + 0.5 * H_OUT * np.cos(MERID_ANGLE_OUT)

n0 = NodeVariables(0)
n1 = NodeVariables(1)

mf0 = MeridionalFlowVariables(0)
mf1 = MeridionalFlowVariables(1)
bl0 = BladeLoadingVariables(0)
bl1 = BladeLoadingVariables(1)

# ============================================================
# 2. Geometry, grid, and blade-angle law
# ============================================================
hub_curve = BezierCurve(
    z_in=Z_HUB_IN,
    z_out=Z_HUB_OUT,
    radius_in=R_HUB_IN_OFF,
    radius_out=R_HUB_OUT_OFF,
    angle_in=MERID_ANGLE_IN,
    angle_out=MERID_ANGLE_OUT,
    n_points=400,
)
shroud_curve = BezierCurve(
    z_in=Z_SHROUD_IN,
    z_out=Z_SHROUD_OUT,
    radius_in=R_SHROUD_IN_OFF,
    radius_out=R_SHROUD_OUT_OFF,
    angle_in=MERID_ANGLE_IN,
    angle_out=MERID_ANGLE_OUT,
    n_points=400,
)

hub_z, hub_r = resample_by_arc_length(hub_curve, N_STREAM)
shroud_z, shroud_r = resample_by_arc_length(shroud_curve, N_STREAM)

Z, R = build_meridional_grid(hub_z, hub_r, shroud_z, shroud_r, N_SPAN)
INV_RN = spanwise_normal_curvature(Z, R)
DELTA_N = np.hypot(np.diff(Z, axis=1), np.diff(R, axis=1))

mid_z, mid_r = 0.5 * (hub_z + shroud_z), 0.5 * (hub_r + shroud_r)
GAMMA = midline_tangent_angle(mid_z, mid_r)
ETA = np.linspace(0.0, 1.0, N_SPAN)

DELTA_S = np.hypot(np.diff(Z, axis=0), np.diff(R, axis=0))  # (N_STREAM-1, N_SPAN)
S_STATIONS = np.concatenate(
    [np.zeros((1, N_SPAN)), np.cumsum(DELTA_S, axis=0)], axis=0
)  # (N_STREAM, N_SPAN)
S_MAX = S_STATIONS[-1, :]  # (N_SPAN,)
S_HAT = S_STATIONS / S_MAX[None, :]  # streamwise fraction, per span

BETA_IN_SPAN = BETA_HUB_IN + (BETA_SHROUD_IN - BETA_HUB_IN) * ETA  # (N_SPAN,)
BETA_OUT_SPAN = BETA_HUB_OUT + (BETA_SHROUD_OUT - BETA_HUB_OUT) * ETA  # (N_SPAN,)

BETA_BL = np.empty((N_STREAM, N_SPAN))
_BETA_BL_DSHAT = np.empty((N_STREAM, N_SPAN))
for _j in range(N_SPAN):
    BETA_BL[:, _j], _BETA_BL_DSHAT[:, _j] = front_loaded_beta_law(
        S_HAT[:, _j],
        BETA_IN_SPAN[_j],
        BETA_OUT_SPAN[_j],
        PIECEWISE_SPLIT_FRAC,
        PIECEWISE_FRAC_AT_SPLIT,
        PIECEWISE_BLEND_HALF_WIDTH,
    )

U = OMEGA * R  # [m/s], for guesses/plots only (also enforced by Kinematics)

GAS_STATE = IdealGasState(gamma=GAMMA_AIR, gas_constant=R_AIR, viscosity=VISCOSITY_AIR)

# Global rothalpy/entropy constants, from the representative rotor-inlet
# velocity triangle (see module docstring for why the compressor file's
# zero-prewhirl shortcut doesn't apply to a pre-swirled turbine inlet).
_VU_IN_DESIGN = VTHETA_OVER_U * U2_TIP
_WU_IN_DESIGN = _VU_IN_DESIGN - U2_TIP
_W_IN_DESIGN = _WU_IN_DESIGN / np.sin(BETA4_DESIGN)

GAS_STATE.update(cp.PT_INPUTS, P_STATIC_IN_DESIGN, T_STATIC_IN_DESIGN)
_H_STATIC_IN_DESIGN = GAS_STATE.hmass()
ENTROPY_IN = GAS_STATE.smass()  # [J/(kg K)]
ROTHALPY = _H_STATIC_IN_DESIGN + _W_IN_DESIGN**2 / 2 - U2_TIP**2 / 2  # [J/kg]


def node_index(i: int, j: int) -> int:
    return i * N_SPAN + j


N_NODES = N_STREAM * N_SPAN
nodes = [NodeVariables(k) for k in range(N_NODES)]
mf_nodes = [MeridionalFlowVariables(k) for k in range(N_NODES)]
bl_nodes = [BladeLoadingVariables(k) for k in range(N_NODES)]

TRAILING_EDGE = N_STREAM - 1

TRANSITION_STAR = int(round(TRANSITION_FRAC_TE * TRAILING_EDGE))
TRANSITION_STAR_IN = max(1, int(round(TRANSITION_FRAC_LE * TRAILING_EDGE)))

LE_BLEND_END = TRANSITION_STAR_IN if USE_SLIP_TRANSITION_LE else 0

LAST_FIXED_BETA_STATION = TRAILING_EDGE - 1 if USE_KUTTA_TE else TRAILING_EDGE
TE_BLEND_START = (
    TRANSITION_STAR + 1
    if (USE_SLIP_TRANSITION_TE and USE_KUTTA_TE)
    else LAST_FIXED_BETA_STATION + 1
)

_BETA_BL_SLOPE = _BETA_BL_DSHAT[TRANSITION_STAR, :] / S_MAX
_BETA_BL_SLOPE_IN = _BETA_BL_DSHAT[TRANSITION_STAR_IN, :] / S_MAX

# ============================================================
# 3. Assemble the system
# ============================================================
system = CasadiSystem(num_span=1)

EQUATIONS = {}
for j in range(N_SPAN):
    for i in range(N_STREAM):
        k = node_index(i, j)
        EQUATIONS[Kinematics()] = k
        EQUATIONS[
            LocalThermoState(rothalpy=ROTHALPY, entropy=ENTROPY_IN, gas_state=GAS_STATE)
        ] = k
        EQUATIONS[SuctionPressureVelocities()] = k
        EQUATIONS[
            BladeSurfacePressures(
                rothalpy=ROTHALPY, entropy=ENTROPY_IN, gas_state=GAS_STATE
            )
        ] = k

        if i == 0:
            # LE flow angle is always a direct boundary condition
            # (BETA_BL[0] + INCIDENCE_LE, see BOUNDARY_CONDITIONS below).
            pass
        elif i <= LE_BLEND_END:
            EQUATIONS[
                SlipTransition(
                    s_i=S_STATIONS[i, j],
                    s_star=S_STATIONS[TRANSITION_STAR_IN, j],
                    s_end=0.0,
                    beta_bl_star=BETA_BL[TRANSITION_STAR_IN, j],
                    slope_star=_BETA_BL_SLOPE_IN[j],
                )
            ] = (k, node_index(0, j))
        elif i <= LAST_FIXED_BETA_STATION and i < TE_BLEND_START:
            EQUATIONS[ZeroDeviation()] = k
        elif i <= LAST_FIXED_BETA_STATION:
            EQUATIONS[
                SlipTransition(
                    s_i=S_STATIONS[i, j],
                    s_star=S_STATIONS[TRANSITION_STAR, j],
                    s_end=S_MAX[j],
                    beta_bl_star=BETA_BL[TRANSITION_STAR, j],
                    slope_star=_BETA_BL_SLOPE[j],
                )
            ] = (k, node_index(TRAILING_EDGE, j))

    for i in range(1, N_STREAM):
        EQUATIONS[
            BladeToBladeLoading(delta_s=DELTA_S[i - 1, j], rotation_sign=-1.0)
        ] = (
            node_index(i - 1, j),
            node_index(i, j),
        )

    if not USE_KUTTA_LE:
        EQUATIONS[CarryLoadingToLE()] = (node_index(0, j), node_index(1, j))

for i in range(N_STREAM):
    for j in range(1, N_SPAN):
        k0, k1 = node_index(i, j - 1), node_index(i, j)
        EQUATIONS[
            MeridionalMomentumMarch(
                delta_n=DELTA_N[i, j - 1],
                r0=R[i, j - 1],
                r1=R[i, j],
                gamma=GAMMA[i],
                omega=OMEGA,
                inv_rn0=INV_RN[i, j - 1],
                inv_rn1=INV_RN[i, j],
            )
        ] = (k0, k1)
        EQUATIONS[
            MassFluxAccumulation(delta_n=DELTA_N[i, j - 1], r0=R[i, j - 1], r1=R[i, j])
        ] = (k0, k1)

for eq, pos in EQUATIONS.items():
    system.add_equation(eq, pos)

# ============================================================
# 4. Boundary conditions
# ============================================================
BOUNDARY_CONDITIONS = {}
for i in range(N_STREAM):
    for j in range(N_SPAN):
        k = node_index(i, j)
        n = nodes[k]
        BOUNDARY_CONDITIONS[n.geo.RDistr] = R[i, j]
        BOUNDARY_CONDITIONS[n.kin.Omega] = OMEGA
        BOUNDARY_CONDITIONS[n.geo.NumBlades] = NUM_BLADES
        BOUNDARY_CONDITIONS[n.geo.BldThick] = BLADE_THICKNESS
        BOUNDARY_CONDITIONS[n.geo.MetalAngle] = BETA_BL[i, j]

    BOUNDARY_CONDITIONS[mf_nodes[node_index(i, 0)].CumFlow] = 0.0
    BOUNDARY_CONDITIONS[mf_nodes[node_index(i, N_SPAN - 1)].CumFlow] = MASS_FLOW

for j in range(N_SPAN):
    BOUNDARY_CONDITIONS[nodes[node_index(0, j)].kin.FlowAngleRel] = (
        BETA_BL[0, j] + INCIDENCE_LE
    )

    if USE_KUTTA_LE:
        BOUNDARY_CONDITIONS[bl_nodes[node_index(0, j)].DeltaW] = 0.0
    if USE_KUTTA_TE:
        BOUNDARY_CONDITIONS[bl_nodes[node_index(TRAILING_EDGE, j)].DeltaW] = 0.0

system.add_boundary_conditions(BOUNDARY_CONDITIONS)

system.build()

# ============================================================
# 5. Guesses and solve
# ============================================================
area_in_est = 2 * np.pi * R_MEAN_IN * H_IN

# Representative density guess from the same rotor-inlet design point used
# for ROTHALPY/ENTROPY_IN above, rather than an arbitrary constant.
GAS_STATE.update(cp.PT_INPUTS, P_STATIC_IN_DESIGN, T_STATIC_IN_DESIGN)
RHO_GUESS = GAS_STATE.rhomass()

w_guess = MASS_FLOW / (RHO_GUESS * area_in_est)

manual_guess = {}
for i in range(N_STREAM):
    for j in range(N_SPAN):
        k = node_index(i, j)
        beta_guess = BETA_BL[i, j]
        wm_guess = w_guess * np.cos(beta_guess)
        wt_guess = w_guess * np.sin(beta_guess)
        u_guess = U[i, j]

        manual_guess[nodes[k].kin.W_mag] = w_guess
        manual_guess[nodes[k].kin.W_mer] = wm_guess
        manual_guess[nodes[k].kin.W_tan] = wt_guess
        manual_guess[nodes[k].kin.FlowAngleRel] = beta_guess
        manual_guess[nodes[k].kin.BladeSpeed] = u_guess
        manual_guess[nodes[k].kin.V_mer] = wm_guess
        manual_guess[nodes[k].kin.V_tan] = wt_guess + u_guess
        manual_guess[nodes[k].kin.V_mag] = np.hypot(wm_guess, wt_guess + u_guess)
        manual_guess[nodes[k].kin.FlowAngleAbs] = np.arctan2(
            wt_guess + u_guess, wm_guess
        )
        manual_guess[nodes[k].stc.Density] = RHO_GUESS
        manual_guess[mf_nodes[k].CumFlow] = MASS_FLOW * j / (N_SPAN - 1)

x0 = system.get_guess(manual_guess, fallback=w_guess)
kn = system.get_boundary_conds()

# Bound the free-deviation regions flow angle within +-25 deg of the
# blade metal angle law
CUSTOM_BOUNDS = {}
slack = np.radians(25.0)
for j in range(N_SPAN):
    _bound_indices = list(range(1, LE_BLEND_END + 1))
    if USE_KUTTA_TE:
        _bound_indices += list(range(TE_BLEND_START, N_STREAM))
    for i in _bound_indices:
        k = node_index(i, j)
        CUSTOM_BOUNDS[nodes[k].kin.FlowAngleRel] = (
            BETA_BL[i, j] - slack,
            BETA_BL[i, j] + slack,
        )
bnd = system.get_bounds(CUSTOM_BOUNDS)

rootfinder = system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        'ipopt.tol': 1e-8,
        'ipopt.max_iter': 2000,
        'ipopt.mu_strategy': 'adaptive',
    },
)
solution = solve_root_problem(rootfinder, x0, kn, bnd, suppress_output=False)

# Polish with a second, tighter bounded IPOPT solve rather than the usual
# unconstrained KINSOL pass
rootfinder_polish = system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        'ipopt.tol': 1e-10,
        'ipopt.max_iter': 1000,
        'ipopt.mu_strategy': 'adaptive',
        'ipopt.warm_start_init_point': 'yes',
    },
)
solution = solve_root_problem(rootfinder_polish, solution, kn, bnd)

sol_dict = system.sol_to_dict(solution)

# ============================================================
# 6. Post-processing
# ============================================================


def grid_field(spec_lookup) -> np.ndarray:
    return np.array(
        [[spec_lookup(i, j)[0] for j in range(N_SPAN)] for i in range(N_STREAM)]
    )


W_grid = grid_field(lambda i, j: sol_dict[nodes[node_index(i, j)].kin.W_mag])
BETA_grid = grid_field(lambda i, j: sol_dict[nodes[node_index(i, j)].kin.FlowAngleRel])
RHO_grid = grid_field(lambda i, j: sol_dict[nodes[node_index(i, j)].stc.Density])
W_ss_grid = grid_field(lambda i, j: sol_dict[bl_nodes[node_index(i, j)].W_ss])
W_ps_grid = grid_field(lambda i, j: sol_dict[bl_nodes[node_index(i, j)].W_ps])
P_ss_grid = grid_field(lambda i, j: sol_dict[bl_nodes[node_index(i, j)].P_ss])
P_ps_grid = grid_field(lambda i, j: sol_dict[bl_nodes[node_index(i, j)].P_ps])

Wm_grid = W_grid * np.cos(BETA_grid)
Wt_grid = W_grid * np.sin(BETA_grid)

H_STATIC_grid = ROTHALPY + U**2 / 2 - W_grid**2 / 2
GAS_STATE.update(cp.HmassSmass_INPUTS, H_STATIC_grid, ENTROPY_IN)
P_grid = GAS_STATE.p()

# Euler work, from the mass-flux-weighted spanwise average of R*Vu at the
# LE and TE stations
Vu_grid = Wt_grid + U
N_COORD = np.concatenate([np.zeros((N_STREAM, 1)), np.cumsum(DELTA_N, axis=1)], axis=1)
RVu_grid = R * Vu_grid


def mass_weighted_mean(field_row: np.ndarray, i: int) -> float:
    flux = RHO_grid[i, :] * Wm_grid[i, :] * R[i, :]
    return np.trapezoid(field_row[i, :] * flux, N_COORD[i, :]) / np.trapezoid(
        flux, N_COORD[i, :]
    )


RVu_IN = mass_weighted_mean(RVu_grid, 0)
RVu_OUT = mass_weighted_mean(RVu_grid, -1)
EULER_WORK = OMEGA * (RVu_OUT - RVu_IN)  # [J/kg], negative = extracted
EULER_POWER = MASS_FLOW * EULER_WORK  # [W]

logger.info(f'Flow angle at LE (hub -> shroud): {np.degrees(BETA_grid[0, :])} deg')
logger.info(f'Flow angle at TE (hub -> shroud): {np.degrees(BETA_grid[-1, :])} deg')
logger.info(
    'Max blade loading W_ss - W_ps (hub -> shroud): '
    f'{(W_ss_grid - W_ps_grid).max(axis=0)}'
)
logger.info(f'Euler work (Omega * d(R*Vu)): {EULER_WORK / 1e3:.1f} kJ/kg')
logger.info(f'Euler power: {EULER_POWER / 1e3:.1f} kW')

# ============================================================
# 7. Plots
# ============================================================
if __name__ == '__main__':
    PANEL_W, PANEL_H = 5.0, 4.0  # [in]

    span_stations = ((0, 'Hub'), (N_SPAN // 2, 'Mid'), (N_SPAN - 1, 'Shroud'))

    # --- Figure 1: grid and the solved flow-angle law --------------------
    fig1, axs1 = plt.subplots(1, 2, figsize=(2 * PANEL_W, PANEL_H))

    ax_grid = axs1[0]
    for i in range(N_STREAM):
        ax_grid.plot(Z[i, :], R[i, :], color='tab:blue', linewidth=0.5)
    for j in range(N_SPAN):
        ax_grid.plot(Z[:, j], R[:, j], color='tab:orange', linewidth=0.5)
    ax_grid.plot(hub_z, hub_r, color='k', linewidth=1.5, label='Hub')
    ax_grid.plot(
        shroud_z, shroud_r, color='k', linestyle='--', linewidth=1.5, label='Shroud'
    )
    ax_grid.set_xlabel('z [m]')
    ax_grid.set_ylabel('R [m]')
    ax_grid.set_aspect('equal')
    ax_grid.set_title(f'Meridional grid ({N_STREAM} x {N_SPAN})')
    ax_grid.legend(loc='upper right')

    ax_beta = axs1[1]
    for j, label in span_stations:
        ax_beta.plot(
            S_HAT[:, j], np.degrees(BETA_BL[:, j]), '--', color=f'C{j}', alpha=0.5
        )
        ax_beta.plot(
            S_HAT[:, j],
            np.degrees(BETA_grid[:, j]),
            '-o',
            color=f'C{j}',
            label=f'{label} (solved)',
        )
    ax_beta.set_xlabel(r'Streamwise fraction $s/s_{max}$')
    ax_beta.set_ylabel(r'Flow angle $\beta$ [deg]')
    ax_beta.set_title('Blade metal angle (dashed) vs. solved flow angle')
    ax_beta.legend()
    ax_beta.grid(True, alpha=0.3)

    fig1.suptitle('RIT rotor: coupled grid and blade-to-blade flow angle')
    fig1.tight_layout()

    # --- Figure 2: meridional-plane velocity fields -----------------------
    fig2, axs2 = plt.subplots(
        1, 3, figsize=(3 * PANEL_W, PANEL_H), sharex=True, sharey=True
    )
    for ax, (field, title) in zip(
        axs2,
        (
            (Wm_grid, 'Meridional velocity $W_m$'),
            (Wt_grid, 'Tangential relative velocity $W_u$'),
            (W_grid, r'Pitchwise-mean relative velocity $\widetilde{W}$'),
        ),
    ):
        contour = ax.contourf(Z, R, field, levels=30, cmap='viridis')
        ax.plot(hub_z, hub_r, color='k', linewidth=1.5)
        ax.plot(shroud_z, shroud_r, color='k', linewidth=1.5, linestyle='--')
        for i in range(0, N_STREAM, max(1, N_STREAM // 10)):
            ax.plot(Z[i, :], R[i, :], color='white', linewidth=0.4)
        ax.set_xlabel('z [m]')
        ax.set_ylabel('R [m]')
        ax.set_aspect('equal')
        ax.set_title(title)
        fig2.colorbar(contour, ax=ax, label='Velocity [m/s]')

    fig2.suptitle('RIT rotor hub-to-shroud velocity distribution')
    fig2.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    # --- Figure 3: blade-to-blade velocity distribution (required) --------
    fig3, axs3 = plt.subplots(1, 3, figsize=(3 * PANEL_W, PANEL_H), sharey=True)
    for ax, (j, label) in zip(axs3, span_stations):
        ax.plot(S_HAT[:, j], W_ss_grid[:, j], '-o', label=r'$W_{SS}$')
        ax.plot(S_HAT[:, j], W_grid[:, j], '--', color='k', label=r'$\widetilde{W}$')
        ax.plot(S_HAT[:, j], W_ps_grid[:, j], '-o', label=r'$W_{PS}$')
        ax.set_xlabel(r'$s / s_{max}$')
        ax.set_title(f'{label} ($\\eta$={ETA[j]:.2f})')
        ax.grid(True, alpha=0.3)
        ax.legend()
    axs3[0].set_ylabel(r'$W$ [m/s]')
    fig3.suptitle('Blade-to-blade velocity distribution, hub to shroud')
    fig3.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    # --- Figure 4: blade surface pressure distribution (required) ---------
    fig4, axs4 = plt.subplots(1, 3, figsize=(3 * PANEL_W, PANEL_H), sharey=True)
    for ax, (j, label) in zip(axs4, span_stations):
        ax.plot(S_HAT[:, j], P_ss_grid[:, j] / 1e3, '-o', label=r'$p_{SS}$')
        ax.plot(S_HAT[:, j], P_grid[:, j] / 1e3, '--', color='k', label=r'$p$ (mean)')
        ax.plot(S_HAT[:, j], P_ps_grid[:, j] / 1e3, '-o', label=r'$p_{PS}$')
        ax.set_xlabel(r'$s / s_{max}$')
        ax.set_title(f'{label} ($\\eta$={ETA[j]:.2f})')
        ax.grid(True, alpha=0.3)
        ax.legend()
    axs4[0].set_ylabel(r'$p$ [kPa]')
    fig4.suptitle('Blade surface pressure distribution, hub to shroud')
    fig4.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    plt.show()
