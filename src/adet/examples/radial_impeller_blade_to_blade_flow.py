"""
Coupled meridional (hub-to-shroud) + blade-to-blade flow of a radial impeller.
"""

import logging

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

# ============================================================
# 1. User settings and problem data
# ============================================================
R_MEAN_IN = 0.055  # [m] mean-line radius at the channel inlet
R_MEAN_OUT = 0.130  # [m] mean-line radius at the channel outlet

H_IN = 0.026  # [m] inlet channel height
H_OUT = 0.008  # [m] exit channel height

MERID_ANGLE_IN = np.radians(0.0)  # [rad] BezierCurve convention (0 = axial)
MERID_ANGLE_OUT = np.radians(90.0)  # [rad] (90 = radial outward)

AXIAL_LENGTH = 0.08  # [m] axial length of the mean line (z=0 to z=AXIAL_LENGTH)

N_STREAM = 26  # number of streamwise stations (LE = 0, TE = N_STREAM - 1)
N_SPAN = 7  # number of spanwise stations (hub to shroud)

RPM = 50_000.0  # [rev/min]
OMEGA = RPM * 2 * np.pi / 60  # [rad/s]

# Kept modest relative to radial_impeller_meridional_flow.py: the coupled
# hub-to-shroud/blade-to-blade system is considerably more sensitive to
# aggressive loading than either sub-model alone.
MASS_FLOW = 0.75  # [kg/s]
T0_IN = 273.15  # [K] inlet total temperature
P0_IN = 101_325.0  # [Pa] inlet total pressure

GAMMA_AIR = 1.4
R_AIR = 287.05  # [J / (kg K)]
VISCOSITY_AIR = 1.8e-5  # [Pa s], unused but required by IdealGasState

# Blade geometry (Section 3.1.2): hub/shroud metal angle, LE and TE, each
# linearly interpolated across the span (ETA, below). Set hub == shroud (as
# here, by default) to recover a spanwise-uniform, lean-free law.
BETA_HUB_IN = np.radians(-48.0)  # [rad] blade metal angle at the hub, LE
BETA_SHROUD_IN = np.radians(-48.0)  # [rad] blade metal angle at the shroud, LE
BETA_HUB_OUT = np.radians(-10.0)  # [rad] blade metal angle at the hub, TE
BETA_SHROUD_OUT = np.radians(-10.0)  # [rad] blade metal angle at the shroud, TE

# Blade-angle law: one of 'linear', 'parabolic', 'circular', 'front_loaded'
# (see adet.equations.streamline_curvature.components.BETA_LAWS /
# .geometry) or a custom callable. Only 'front_loaded' takes extra
# parameters (split_frac/frac_at_split/half_width); the other three
# don't, so BETA_LAW_KWARGS is derived from BETA_LAW rather than set
# independently -- switching BETA_LAW above is then a one-line change.
BETA_LAW = 'linear'
BETA_LAW_KWARGS = (
    {
        'split_frac': 0.4,  # [s_hat] where the slope changes
        'frac_at_split': 0.48,  # fraction of (beta_out - beta_in) covered by the split
        'half_width': 0.12,  # [s_hat] half-width of the smoothing window
    }
    if BETA_LAW == 'front_loaded'
    else {}
)

NUM_BLADES = 24
BLADE_THICKNESS = 0.0015  # [m]

# Whether to solve the blade-to-blade plane at all (SurfaceFlow's
# include_blade_to_blade). With it False, the LE/TE closure toggles below
# are unused (see SurfaceFlow's docstring), and every blade-loading-only
# guess/post-processing/plot block (W_ss/W_ps/P_ss/P_ps, Figures 3-4) is
# automatically skipped instead of needing to be commented out by hand.
INCLUDE_BLADE_TO_BLADE = True

# ------------------------------------------------------------
# LE/TE closure toggles, independently switchable at each end
# ------------------------------------------------------------
USE_SLIP_TRANSITION_LE = True  # blend a deviated LE incidence onto the law
USE_SLIP_TRANSITION_TE = True  # blend the law onto the free TE value
USE_KUTTA_LE = True  # assume zero blade loading (DeltaW=0) at the LE
USE_KUTTA_TE = True  # assume zero blade loading (DeltaW=0, W_SS=W_PS) at the TE

# LE incidence (deviation of the flow angle from the blade metal angle at
# s_hat=0); 0 = true zero incidence.
INCIDENCE_LE = np.radians(0.0)  # [rad]

TRANSITION_FRAC_LE = 0.25
TRANSITION_FRAC_TE = 0.6

# Hub and shroud, offset by +-H/2 along the local normal to the
# prescribed meridional angle.
Z_HUB_IN = 0.5 * H_IN * np.sin(MERID_ANGLE_IN)
R_HUB_IN = R_MEAN_IN - 0.5 * H_IN * np.cos(MERID_ANGLE_IN)
Z_SHROUD_IN = -0.5 * H_IN * np.sin(MERID_ANGLE_IN)
R_SHROUD_IN = R_MEAN_IN + 0.5 * H_IN * np.cos(MERID_ANGLE_IN)

Z_HUB_OUT = AXIAL_LENGTH + 0.5 * H_OUT * np.sin(MERID_ANGLE_OUT)
R_HUB_OUT = R_MEAN_OUT - 0.5 * H_OUT * np.cos(MERID_ANGLE_OUT)
Z_SHROUD_OUT = AXIAL_LENGTH - 0.5 * H_OUT * np.sin(MERID_ANGLE_OUT)
R_SHROUD_OUT = R_MEAN_OUT + 0.5 * H_OUT * np.cos(MERID_ANGLE_OUT)

# ============================================================
# 2. Geometry, grid, and blade-angle law
# ============================================================
hub_curve = BezierCurve(
    z_in=Z_HUB_IN,
    z_out=Z_HUB_OUT,
    radius_in=R_HUB_IN,
    radius_out=R_HUB_OUT,
    angle_in=MERID_ANGLE_IN,
    angle_out=MERID_ANGLE_OUT,
    n_points=400,
)
shroud_curve = BezierCurve(
    z_in=Z_SHROUD_IN,
    z_out=Z_SHROUD_OUT,
    radius_in=R_SHROUD_IN,
    radius_out=R_SHROUD_OUT,
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

surface = SurfaceFlow(
    N_STREAM,
    N_SPAN,
    use_slip_transition_le=USE_SLIP_TRANSITION_LE,
    use_slip_transition_te=USE_SLIP_TRANSITION_TE,
    use_kutta_le=USE_KUTTA_LE,
    use_kutta_te=USE_KUTTA_TE,
    transition_frac_le=TRANSITION_FRAC_LE,
    transition_frac_te=TRANSITION_FRAC_TE,
    beta_law=BETA_LAW,
    beta_law_kwargs=BETA_LAW_KWARGS,
    include_blade_to_blade=INCLUDE_BLADE_TO_BLADE,
    axial_absolute_inflow=True,
)
TRAILING_EDGE = surface.trailing_edge

BETA_BL, _BETA_BL_DSHAT = surface.beta_field(
    S_STATIONS, S_MAX, BETA_IN_SPAN, BETA_OUT_SPAN
)

U = OMEGA * R  # [m/s], for guesses/plots only (also enforced by Kinematics)

GAS_STATE = IdealGasState(gamma=GAMMA_AIR, gas_constant=R_AIR, viscosity=VISCOSITY_AIR)
GAS_STATE.update(cp.PT_INPUTS, P0_IN, T0_IN)
# Upstream absolute total enthalpy (same at every span position) and the
# shared, globally uniform isentropic entropy -- NOT a rothalpy: rothalpy
# depends on the local velocity triangle (varies with span) and is
# conserved per-streamline by SurfaceFlow itself (RothalpyConservation),
# anchored here only at the leading edge (AbsoluteEnergyClosure).
TOTAL_ENTHALPY_IN = GAS_STATE.hmass()  # [J/kg]
ENTROPY_IN = GAS_STATE.smass()  # [J/(kg K)]


# ============================================================
# 3. Assemble the system
# ============================================================
system = CasadiSystem(num_span=1)
surface.assemble(
    system,
    total_enthalpy=TOTAL_ENTHALPY_IN,
    entropy=ENTROPY_IN,
    gas_state=GAS_STATE,
    omega=OMEGA,
    r=R,
    gamma=GAMMA,
    inv_rn=INV_RN,
    delta_n=DELTA_N,
    delta_s=DELTA_S,
    s_stations=S_STATIONS,
    s_max=S_MAX,
    beta_bl=BETA_BL,
    beta_bl_dshat=_BETA_BL_DSHAT,
)

# ============================================================
# 4. Boundary conditions
# ============================================================
BOUNDARY_CONDITIONS = surface.closure_boundary_conditions(
    beta_bl=BETA_BL, mass_flow=MASS_FLOW, incidence_le=INCIDENCE_LE
)
for i in range(N_STREAM):
    for j in range(N_SPAN):
        n = surface.nodes[surface.node_index(i, j)]
        BOUNDARY_CONDITIONS[n.geo.RDistr] = R[i, j]
        BOUNDARY_CONDITIONS[n.kin.Omega] = OMEGA
        BOUNDARY_CONDITIONS[n.geo.NumBlades] = NUM_BLADES
        BOUNDARY_CONDITIONS[n.geo.BldThick] = BLADE_THICKNESS
        BOUNDARY_CONDITIONS[n.geo.MetalAngle] = BETA_BL[i, j]

system.add_boundary_conditions(BOUNDARY_CONDITIONS)

system.build()

# ============================================================
# 5. Guesses and solve
# ============================================================
area_in_est = 2 * np.pi * R_MEAN_IN * H_IN
w_guess = MASS_FLOW / (1.2 * area_in_est)

manual_guess = {}
for i in range(N_STREAM):
    for j in range(N_SPAN):
        k = surface.node_index(i, j)
        beta_guess = BETA_BL[i, j]
        wm_guess = w_guess * np.cos(beta_guess)
        wt_guess = w_guess * np.sin(beta_guess)
        u_guess = U[i, j]

        manual_guess[surface.nodes[k].kin.W_mag] = w_guess
        manual_guess[surface.nodes[k].kin.W_mer] = wm_guess
        manual_guess[surface.nodes[k].kin.W_tan] = wt_guess
        manual_guess[surface.nodes[k].kin.FlowAngleRel] = beta_guess
        manual_guess[surface.nodes[k].kin.BladeSpeed] = u_guess
        manual_guess[surface.nodes[k].kin.V_mer] = wm_guess
        manual_guess[surface.nodes[k].kin.V_tan] = wt_guess + u_guess
        v_mag_guess = np.hypot(wm_guess, wt_guess + u_guess)
        manual_guess[surface.nodes[k].kin.V_mag] = v_mag_guess
        manual_guess[surface.nodes[k].kin.FlowAngleAbs] = np.arctan2(
            wt_guess + u_guess, wm_guess
        )
        manual_guess[surface.nodes[k].stc.Density] = 1.2
        # Floored: at high blade speed this crude guess (V ~ U) can
        # exceed TOTAL_ENTHALPY_IN, giving a negative/non-physical
        # static-enthalpy guess that breaks the EOS at the first solver
        # call -- the floor only matters for the initial guess itself.
        manual_guess[surface.nodes[k].stc.Enthalpy] = max(
            TOTAL_ENTHALPY_IN - 0.5 * v_mag_guess**2, 0.3 * TOTAL_ENTHALPY_IN
        )
        manual_guess[surface.mf_nodes[k].CumFlow] = MASS_FLOW * j / (N_SPAN - 1)
        if INCLUDE_BLADE_TO_BLADE:
            manual_guess[surface.bl_nodes[k].W_ss] = w_guess + 20.0
            manual_guess[surface.bl_nodes[k].W_ps] = w_guess - 20.0
            manual_guess[surface.bl_nodes[k].DeltaW] = 10.0
            manual_guess[surface.bl_nodes[k].P_ss] = 9e4
            manual_guess[surface.bl_nodes[k].P_ps] = 1.1e5

x0 = system.get_guess(manual_guess, fallback=w_guess)
kn = system.get_boundary_conds()
bnd = system.get_bounds()

rootfinder = system.make_rootfinder(
    'ipopt',
    opts={'error_on_fail': False, 'ipopt.tol': 1e-8, 'ipopt.max_iter': 2000},
)
solution = solve_root_problem(rootfinder, x0, kn, bnd, suppress_output=False)

rootfinder_kinsol = system.make_rootfinder('kinsol')
solution = solve_root_problem(rootfinder_kinsol, solution, kn)

sol_dict = system.sol_to_dict(solution)

# ============================================================
# 6. Post-processing
# ============================================================


def grid_field(spec_lookup) -> np.ndarray:
    return np.array(
        [[spec_lookup(i, j)[0] for j in range(N_SPAN)] for i in range(N_STREAM)]
    )


W_grid = grid_field(
    lambda i, j: sol_dict[surface.nodes[surface.node_index(i, j)].kin.W_mag]
)
BETA_grid = grid_field(
    lambda i, j: sol_dict[surface.nodes[surface.node_index(i, j)].kin.FlowAngleRel]
)
RHO_grid = grid_field(
    lambda i, j: sol_dict[surface.nodes[surface.node_index(i, j)].stc.Density]
)
if INCLUDE_BLADE_TO_BLADE:
    W_ss_grid = grid_field(
        lambda i, j: sol_dict[surface.bl_nodes[surface.node_index(i, j)].W_ss]
    )
    W_ps_grid = grid_field(
        lambda i, j: sol_dict[surface.bl_nodes[surface.node_index(i, j)].W_ps]
    )
    P_ss_grid = grid_field(
        lambda i, j: sol_dict[surface.bl_nodes[surface.node_index(i, j)].P_ss]
    )
    P_ps_grid = grid_field(
        lambda i, j: sol_dict[surface.bl_nodes[surface.node_index(i, j)].P_ps]
    )

Wm_grid = W_grid * np.cos(BETA_grid)
Wt_grid = W_grid * np.sin(BETA_grid)
Vm_grid = Wm_grid  # absolute = relative meridional component, always
Vt_grid = Wt_grid + U  # absolute tangential component
V_grid = np.hypot(Vm_grid, Vt_grid)

# Static enthalpy is now a per-node solved quantity (see
# RothalpyConservation/AbsoluteEnergyClosure), not recomputed from a
# single rothalpy constant -- read it straight from the solution.
H_STATIC_grid = grid_field(
    lambda i, j: sol_dict[surface.nodes[surface.node_index(i, j)].stc.Enthalpy]
)
GAS_STATE.update(cp.HmassSmass_INPUTS, H_STATIC_grid, ENTROPY_IN)
P_grid = GAS_STATE.p()

# Rothalpy (Eq. 1.68), conserved along each streamline but generally
# varying across span -- see RothalpyConservation/AbsoluteEnergyClosure.
ROTHALPY_grid = H_STATIC_grid + W_grid**2 / 2 - U**2 / 2
# Absolute total enthalpy H0 = h + V^2/2 -- uniform at the LE by
# construction (TOTAL_ENTHALPY_IN), generally not downstream of it, since
# work is done on/by the flow (Euler's turbomachinery equation).
H0_ABS_grid = H_STATIC_grid + V_grid**2 / 2

logger.info(f'Flow angle at LE (hub -> shroud): {np.degrees(BETA_grid[0, :])} deg')
logger.info(f'Flow angle at TE (hub -> shroud): {np.degrees(BETA_grid[-1, :])} deg')
if INCLUDE_BLADE_TO_BLADE:
    logger.info(
        'Max blade loading W_ss - W_ps (hub -> shroud): '
        f'{(W_ss_grid - W_ps_grid).max(axis=0)}'
    )

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
    ax_grid.legend(loc='lower right')

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

    fig1.suptitle('Coupled grid and blade-to-blade flow angle')
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

    fig2.suptitle('Hub-to-shroud velocity distribution')
    fig2.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    if INCLUDE_BLADE_TO_BLADE:
        # --- Figure 3: blade-to-blade velocity distribution ----------------
        fig3, axs3 = plt.subplots(1, 3, figsize=(3 * PANEL_W, PANEL_H), sharey=True)
        for ax, (j, label) in zip(axs3, span_stations):
            ax.plot(S_HAT[:, j], W_ss_grid[:, j], '-o', label=r'$W_{SS}$')
            ax.plot(
                S_HAT[:, j], W_grid[:, j], '--', color='k', label=r'$\widetilde{W}$'
            )
            ax.plot(S_HAT[:, j], W_ps_grid[:, j], '-o', label=r'$W_{PS}$')
            ax.set_xlabel(r'$s / s_{max}$')
            ax.set_title(f'{label} ($\\eta$={ETA[j]:.2f})')
            ax.grid(True, alpha=0.3)
            ax.legend()
        axs3[0].set_ylabel(r'$W$ [m/s]')
        fig3.suptitle('Blade-to-blade velocity distribution, hub to shroud')
        fig3.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

        # --- Figure 4: blade surface pressure distribution ------------------
        fig4, axs4 = plt.subplots(1, 3, figsize=(3 * PANEL_W, PANEL_H), sharey=True)
        for ax, (j, label) in zip(axs4, span_stations):
            ax.plot(S_HAT[:, j], P_ss_grid[:, j] / 1e3, '-o', label=r'$p_{SS}$')
            ax.plot(
                S_HAT[:, j], P_grid[:, j] / 1e3, '--', color='k', label=r'$p$ (mean)'
            )
            ax.plot(S_HAT[:, j], P_ps_grid[:, j] / 1e3, '-o', label=r'$p_{PS}$')
            ax.set_xlabel(r'$s / s_{max}$')
            ax.set_title(f'{label} ($\\eta$={ETA[j]:.2f})')
            ax.grid(True, alpha=0.3)
            ax.legend()
        axs4[0].set_ylabel(r'$p$ [kPa]')
        fig4.suptitle('Blade surface pressure distribution, hub to shroud')
        fig4.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    def plot_meridional_contour(field, title, cbar_label):
        fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
        contour = ax.contourf(Z, R, field, levels=30, cmap='viridis')
        ax.plot(hub_z, hub_r, color='k', linewidth=1.5)
        ax.plot(shroud_z, shroud_r, color='k', linewidth=1.5, linestyle='--')
        for i in range(0, N_STREAM, max(1, N_STREAM // 10)):
            ax.plot(Z[i, :], R[i, :], color='white', linewidth=0.4)
        ax.set_xlabel('z [m]')
        ax.set_ylabel('R [m]')
        ax.set_aspect('equal')
        ax.set_title(title)
        fig.colorbar(contour, ax=ax, label=cbar_label)
        fig.tight_layout()
        return fig, ax

    # --- Figure 5: meridional-plane static pressure contour ---------------
    plot_meridional_contour(
        P_grid / 1e3, 'Meridional-plane static pressure', 'Static pressure $p$ [kPa]'
    )

    # --- Figure 6: meridional-plane rothalpy contour -----------------------
    plot_meridional_contour(
        ROTHALPY_grid / 1e3,
        'Meridional-plane rothalpy',
        'Rothalpy $I = h + W^2/2 - U^2/2$ [kJ/kg]',
    )

    # --- Figure 7: meridional-plane absolute total enthalpy contour --------
    plot_meridional_contour(
        H0_ABS_grid / 1e3,
        'Meridional-plane absolute total enthalpy',
        'Absolute total enthalpy $H_0 = h + V^2/2$ [kJ/kg]',
    )

    plt.show()
