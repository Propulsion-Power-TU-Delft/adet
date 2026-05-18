"""
Regenerate meridional channel plots from saved design map data
"""

from adet.variables import NodeVariables

import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RectBivariateSpline

from adet.equations.geometrical import ParabolicCamberline

# Constants
CONTOUR_LEVELS = 50
PLOT_LOSSES = True
SPLINE_GRID_FACTOR = 3  # Refinement factor for spline grid

# Font sizes
TICK_SIZE = 24
LABEL_SIZE = 24
TITLE_SIZE = 33
AXIS_LABEL_SIZE = 27
COLORBAR_LABEL_SIZE = 26

# Line widths for profile and camber line plots
LINE_WIDTH_MAIN = 3.0
LINE_WIDTH_SECONDARY = 2.0

plt.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'serif',
    }
)

# Setup paths
data_dir = pathlib.Path(__file__).parent.parent.parent.parent / 'outputs'

node = NodeVariables()
n1 = NodeVariables(1)
n3 = NodeVariables(3)


def get_mean_radius(sol_dict):
    """Extract mean radius from solution dict for nondimensionalization."""
    radii = []
    for node_idx in [0, 1, 2, 3]:
        rr = sol_dict.get(node.geo.RDistr._at_node(node_idx))
        if rr is not None:
            val = rr[0] if hasattr(rr, '__len__') else rr
            radii.append(val)
    return np.mean(radii) if radii else 1.0


def plot_contour_quantity(
    qty,
    ax,
    design_map,
    cmap='viridis',
    levels=CONTOUR_LEVELS,
    clabel='',
    interpolate='none',
):
    """
    Plot a quantity as a 2D contour map with phi (x) and psi (y) axes.

    Parameters
    ----------
    qty : VarSpec
        Variable specification for quantity to plot in solution dicts
    ax : matplotlib.axes.Axes
        Axes to plot on
    design_map : dict
        Design map dictionary loaded from pickle file
    cmap : str
        Colormap (default: 'viridis')
    levels : int
        Number of contour levels (default: CONTOUR_LEVELS)
    clabel : str
        Colorbar label (default: qty name)
    interpolate : str
        Interpolation method: 'spline' for bicubic spline, 'none' for raw data
        (default: 'spline')
    """
    solution_dicts = design_map['solution_dicts']
    phi_vals = design_map['phi_vals']
    psi_vals = design_map['psi_vals']
    N_PTS = design_map['N_PTS']

    # Extract quantity values
    qty_values = []
    for dic in solution_dicts:
        if dic is None:
            if qty == n3.ndim.EtaTT:
                qty_values.append(0.7)
            else:
                qty_values.append(np.nan)
            continue

        val = dic.get(qty, None)
        if val is None:
            qty_values.append(np.nan)
        else:
            val = val[0] if hasattr(val, '__len__') else val
            qty_values.append(val)

    qty_values = np.array(qty_values)

    # Reshape data into 2D grids
    phi_grid = phi_vals.reshape((N_PTS, N_PTS))
    psi_grid = psi_vals.reshape((N_PTS, N_PTS))
    qty_grid = qty_values.reshape((N_PTS, N_PTS))

    # Apply interpolation based on method
    if interpolate == 'spline':
        valid_mask = ~np.isnan(qty_grid)
        if np.any(valid_mask):
            phi_min, phi_max = phi_grid.min(), phi_grid.max()
            psi_min, psi_max = psi_grid.min(), psi_grid.max()

            # Create bicubic spline (handles NaN gracefully)
            spl = RectBivariateSpline(
                phi_grid[:, 0], psi_grid[0, :], qty_grid, kx=3, ky=3
            )

            # Create finer grid for smooth contours
            n_fine = N_PTS * SPLINE_GRID_FACTOR
            phi_fine = np.linspace(phi_min, phi_max, n_fine)
            psi_fine = np.linspace(psi_min, psi_max, n_fine)
            phi_grid_fine, psi_grid_fine = np.meshgrid(phi_fine, psi_fine)

            # Evaluate spline on fine grid
            qty_grid_fine = spl(phi_fine, psi_fine, grid=True)
        else:
            phi_grid_fine = phi_grid
            psi_grid_fine = psi_grid
            qty_grid_fine = qty_grid
    else:
        # No interpolation, use raw data
        phi_grid_fine = phi_grid
        psi_grid_fine = psi_grid
        qty_grid_fine = qty_grid

    # Create contour plot
    contourf = ax.contourf(
        phi_grid_fine, psi_grid_fine, qty_grid_fine, levels=levels, cmap=cmap
    )
    contour_lines = ax.contour(
        phi_grid_fine,
        psi_grid_fine,
        qty_grid_fine,
        levels=levels,
        colors='black',
        alpha=0.3,
        linewidths=0.5,
    )
    ax.clabel(contour_lines, inline=True, fontsize=8)

    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label(
        clabel if clabel else qty,
        rotation=0,
        labelpad=20,
        fontsize=COLORBAR_LABEL_SIZE,
        ha='center',
    )

    cbar.ax.tick_params('both', labelsize=TICK_SIZE)
    ax.tick_params('both', labelsize=TICK_SIZE)
    ax.set_ylabel(r'$K_{is}$', fontdict={'fontsize': AXIS_LABEL_SIZE})
    ax.set_xlabel(r'$\phi$', fontdict={'fontsize': AXIS_LABEL_SIZE})
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    return


def plot_profile_at_point(
    phi_target,
    psi_target,
    design_map,
    plot_type='both',
    figsize=(12, 5),
    axes=None,
):
    """
    Plot blade profile (meridional channel and/or camber lines) at a specific phi/psi
    point.

    Parameters
    ----------
    phi_target : float
        Target flow coefficient (phi)
    psi_target : float
        Target loading coefficient (psi)
    design_map : dict
        Design map dictionary loaded from pickle file
    plot_type : str
        Type of plot: 'meridional', 'camber', or 'both' (default: 'both')
    figsize : tuple
        Figure size (default: (12, 5))
    axes : array-like, optional
        Pre-created axes for plotting (meridional, camber). If None, creates new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The generated figure (None if axes provided)
    sol_dict : dict
        Solution dictionary at the target point
    """
    solution_dicts = design_map['solution_dicts']
    phi_vals = design_map['phi_vals']
    psi_vals = design_map['psi_vals']

    # Find the closest point to the target phi/psi
    distances = np.sqrt((phi_vals - phi_target) ** 2 + (psi_vals - psi_target) ** 2)
    linear_idx = np.argmin(distances)

    sol_dict = solution_dicts[linear_idx]

    if sol_dict is None:
        raise ValueError(
            f'No solution found at phi={phi_target:.3f}, psi={psi_target:.3f}'
        )

    phi_actual = phi_vals[linear_idx]
    psi_actual = psi_vals[linear_idx]

    fig = None
    if axes is None:
        if plot_type == 'both':
            fig, axes = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax_single = plt.subplots(figsize=figsize)
            axes = [ax_single]

    # Plot meridional channel
    if plot_type in ('meridional', 'both'):
        ax = axes[0]
        ax.set_aspect('equal')
        r_mean = get_mean_radius(sol_dict)
        ax.set_ylabel(r'$\tilde{r}$', fontsize=AXIS_LABEL_SIZE)
        ax.set_xlabel(r'$\tilde{z}$', fontsize=AXIS_LABEL_SIZE)
        ax.tick_params('both', labelsize=TICK_SIZE)
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f'Meridional Channel: φ={phi_actual:.3f}, ψ={psi_actual:.3f}',
            fontsize=TITLE_SIZE,
        )

        offset = 0.0
        node_pairs = [(0, 1), (2, 3)]
        colors = ['steelblue', 'coral']

        for pair_idx, (node_in, node_out) in enumerate(node_pairs):
            rr_in = sol_dict.get(node.geo.RDistr._at_node(node_in), None)
            height_in = sol_dict.get(node.geo.Height._at_node(node_in), None)
            chord_ax_out = sol_dict.get(node.geo.ChordAx._at_node(node_out), None)
            rr_out = sol_dict.get(node.geo.RDistr._at_node(node_out), None)
            height_out = sol_dict.get(node.geo.Height._at_node(node_out), None)

            if (
                rr_in is None
                or height_in is None
                or rr_out is None
                or height_out is None
            ):
                continue

            rr_in_val = rr_in[0] if hasattr(rr_in, '__len__') else rr_in
            height_in_val = height_in[0] if hasattr(height_in, '__len__') else height_in
            rr_out_val = rr_out[0] if hasattr(rr_out, '__len__') else rr_out
            height_out_val = (
                height_out[0] if hasattr(height_out, '__len__') else height_out
            )
            chord_ax_val = (
                chord_ax_out[0] if hasattr(chord_ax_out, '__len__') else chord_ax_out
            )

            x_inlet = offset / r_mean
            x_outlet = (offset + chord_ax_val) / r_mean

            r_hub_inlet = (rr_in_val - height_in_val / 2) / r_mean
            r_tip_inlet = (rr_in_val + height_in_val / 2) / r_mean
            r_hub_outlet = (rr_out_val - height_out_val / 2) / r_mean
            r_tip_outlet = (rr_out_val + height_out_val / 2) / r_mean

            color = colors[pair_idx]

            ax.plot(
                [x_inlet, x_outlet],
                [r_hub_inlet, r_hub_outlet],
                color=color,
                linewidth=LINE_WIDTH_MAIN,
            )
            ax.plot(
                [x_inlet, x_outlet],
                [r_tip_inlet, r_tip_outlet],
                color=color,
                linewidth=LINE_WIDTH_MAIN,
            )
            ax.plot(
                [x_inlet, x_inlet],
                [r_hub_inlet, r_tip_inlet],
                color=color,
                linewidth=LINE_WIDTH_SECONDARY,
                linestyle='--',
                alpha=0.6,
            )
            ax.plot(
                [x_outlet, x_outlet],
                [r_hub_outlet, r_tip_outlet],
                color=color,
                linewidth=LINE_WIDTH_SECONDARY,
                linestyle='--',
                alpha=0.6,
            )

            offset += chord_ax_val * 1.05

        ax.plot(
            [0.0, offset / r_mean],
            [0.0, 0.0],
            color='r',
            linestyle='dashdot',
            linewidth=2,
        )

    # Plot camber lines
    if plot_type in ('camber', 'both'):
        ax = axes[1] if plot_type == 'both' else axes[0]
        ax.set_aspect('equal')
        r_mean = get_mean_radius(sol_dict)
        ax.set_ylabel(r'$\theta$', fontsize=AXIS_LABEL_SIZE)
        ax.set_xlabel(r'$\tilde{z}$', fontsize=AXIS_LABEL_SIZE)
        ax.tick_params('both', labelsize=TICK_SIZE)
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f'Camber Lines: phi={phi_actual:.3f}, psi={psi_actual:.3f}',
            fontsize=TITLE_SIZE,
        )

        pbl = ParabolicCamberline()
        offset = 0.0
        node_pairs = [(0, 1), (2, 3)]
        colors_blade = ['steelblue', 'coral']

        for pair_idx, (node_in, node_out) in enumerate(node_pairs):
            metal_angle_in = sol_dict.get(node.geo.MetalAngle._at_node(node_in), None)
            metal_angle_out = sol_dict.get(node.geo.MetalAngle._at_node(node_out), None)
            chord_ax = sol_dict.get(node.geo.ChordAx._at_node(node_out), None)
            pitch = sol_dict.get(node.geo.Pitch._at_node(node_out), None)

            if (
                metal_angle_in is None
                or metal_angle_out is None
                or chord_ax is None
                or pitch is None
            ):
                continue

            metal_angle_in_val = (
                metal_angle_in[0]
                if hasattr(metal_angle_in, '__len__')
                else metal_angle_in
            )
            metal_angle_out_val = (
                metal_angle_out[0]
                if hasattr(metal_angle_out, '__len__')
                else metal_angle_out
            )
            chord_ax_val = chord_ax[0] if hasattr(chord_ax, '__len__') else chord_ax
            pitch_val = pitch[0] if hasattr(pitch, '__len__') else pitch

            color = colors_blade[pair_idx]

            num_blades = 2
            for blade_num in range(num_blades):
                pbl.plot_camber_line(
                    ax,
                    metal_angle_in_val,
                    metal_angle_out_val,
                    chord_ax_val / r_mean,
                    color,
                    axial_offset=offset / r_mean,
                    tangential_offset=blade_num * pitch_val / r_mean,
                    linewidth=LINE_WIDTH_MAIN,
                )

            if hasattr(metal_angle_in_val, '__len__') and hasattr(
                metal_angle_out_val, '__len__'
            ):
                pbl.plot_camber_line(
                    ax,
                    metal_angle_in_val[0],
                    metal_angle_out_val[0],
                    chord_ax_val / r_mean,
                    'orange',
                    axial_offset=offset / r_mean,
                    linewidth=LINE_WIDTH_SECONDARY,
                    linestyle='--',
                    alpha=0.7,
                )
                pbl.plot_camber_line(
                    ax,
                    metal_angle_in_val[-1],
                    metal_angle_out_val[-1],
                    chord_ax_val / r_mean,
                    'seagreen',
                    axial_offset=offset / r_mean,
                    linewidth=LINE_WIDTH_SECONDARY,
                    linestyle='--',
                    alpha=0.7,
                )

            offset += chord_ax_val * 1.1

    if fig is not None:
        plt.tight_layout()
    return fig, sol_dict


# Load all data from pickle file
IDENTIFIER = ''
REACTION = 0.3
FILENAME = f'des_map_R{REACTION}_vr4.0_dyn_fmax30_ar3.0.pkl'

print('Loading design map data from pickle file...')
with open(data_dir / FILENAME, 'rb') as f:
    design_map = pickle.load(f)

solution_dicts = design_map['solution_dicts']
phi_vals = design_map['phi_vals']
psi_vals = design_map['psi_vals']
N_PTS = design_map['N_PTS']

print(f'Loaded {len(solution_dicts)} solution dictionaries')

# Select 4 corner points using closest point logic
# Find nearest points to target phi/psi combinations (all in row 0, cols 0-3)
corner_targets = [
    (0.4, 3, 0, 0),  # phi=0.4, K_is=3 → col 0
    (1.4, 3, 0, 1),  # phi=1.4, K_is=3 → col 1
    (0.4, 10, 0, 2),  # phi=0.4, K_is=10 → col 2
    (1.4, 10, 0, 3),  # phi=1.4, K_is=10 → col 3
]

corner_defs = []
for phi_target, psi_target, row, col in corner_targets:
    distances = np.sqrt((phi_vals - phi_target) ** 2 + (psi_vals - psi_target) ** 2)
    linear_idx = np.argmin(distances)
    phi_idx = linear_idx // N_PTS
    psi_idx = linear_idx % N_PTS
    corner_defs.append((phi_idx, psi_idx, row, col))

# 2 rows × 4 cols:
# Four design points in a row, each stacked vertically (meridional, camber)
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(
    2,
    4,
    figure=fig,
    hspace=0.4,
    wspace=0.3,
    height_ratios=[1, 1.5],
)

pbl = ParabolicCamberline()
node_pairs = [(0, 1), (2, 3)]
colors = ['steelblue', 'coral']

print('Plotting 4 corner designs...')
for phi_idx, psi_idx, row, col in corner_defs:
    linear_idx = phi_idx * N_PTS + psi_idx
    sol_dict = solution_dicts[linear_idx] if linear_idx < len(solution_dicts) else None
    phi_val = phi_vals[linear_idx]
    psi_val = psi_vals[linear_idx]
    label = rf'$\phi={phi_val:.3f}$' + '\n' + rf'$K_{{is}}={psi_val:.2f}$'

    ax_mer = fig.add_subplot(gs[row, col])
    ax_cam = fig.add_subplot(gs[row + 1, col], sharex=ax_mer)

    if sol_dict is None:
        print(f'Skipping corner ({phi_idx}, {psi_idx}) - no solution')
        for ax in (ax_mer, ax_cam):
            ax.text(
                0.5,
                0.5,
                'No solution',
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=LABEL_SIZE,
            )
            ax.set_title(label, fontsize=TITLE_SIZE, pad=12)
        continue

    # --- Pre-compute shared offsets for both subplots ---
    r_mean = get_mean_radius(sol_dict)
    OFFSET_FACTOR = 1.07
    pair_offsets = []
    offset = 0.0
    for _, node_out in node_pairs:
        chord_ax_out = sol_dict.get(node.geo.ChordAx._at_node(node_out))
        if chord_ax_out is None:
            pair_offsets.append(None)
            continue
        chord_ax_v = (
            chord_ax_out[0] if hasattr(chord_ax_out, '__len__') else chord_ax_out
        )
        pair_offsets.append(offset)
        offset += chord_ax_v * OFFSET_FACTOR

    # --- Meridional channel ---
    ax_mer.set_aspect('equal')
    ax_mer.set_ylabel(r'$\tilde{r}$', fontsize=AXIS_LABEL_SIZE)
    ax_mer.set_xlabel(r'$\tilde{z}$', fontsize=AXIS_LABEL_SIZE)
    ax_mer.tick_params('both', labelsize=TICK_SIZE)
    ax_mer.grid(True, alpha=0.3)
    ax_mer.set_title(label, fontsize=TITLE_SIZE, pad=12)

    for pair_idx, (node_in, node_out) in enumerate(node_pairs):
        if pair_offsets[pair_idx] is None:
            continue
        try:
            rr_in = sol_dict.get(node.geo.RDistr._at_node(node_in))
            height_in = sol_dict.get(node.geo.Height._at_node(node_in))
            chord_ax_out = sol_dict.get(node.geo.ChordAx._at_node(node_out))
            rr_out = sol_dict.get(node.geo.RDistr._at_node(node_out))
            height_out = sol_dict.get(node.geo.Height._at_node(node_out))
            if any(
                v is None for v in (rr_in, height_in, chord_ax_out, rr_out, height_out)
            ):
                continue
            rr_in_v = rr_in[0] if hasattr(rr_in, '__len__') else rr_in
            h_in_v = height_in[0] if hasattr(height_in, '__len__') else height_in
            rr_out_v = rr_out[0] if hasattr(rr_out, '__len__') else rr_out
            h_out_v = height_out[0] if hasattr(height_out, '__len__') else height_out
            chord_ax_v = (
                chord_ax_out[0] if hasattr(chord_ax_out, '__len__') else chord_ax_out
            )
            color = colors[pair_idx]
            offset = pair_offsets[pair_idx]
            x_in, x_out = offset / r_mean, (offset + chord_ax_v) / r_mean
            r_hub_in = (rr_in_v - h_in_v / 2) / r_mean
            r_tip_in = (rr_in_v + h_in_v / 2) / r_mean
            r_hub_out = (rr_out_v - h_out_v / 2) / r_mean
            r_tip_out = (rr_out_v + h_out_v / 2) / r_mean
            ax_mer.plot(
                [x_in, x_out],
                [r_hub_in, r_hub_out],
                color=color,
                linewidth=LINE_WIDTH_MAIN,
            )
            ax_mer.plot(
                [x_in, x_out],
                [r_tip_in, r_tip_out],
                color=color,
                linewidth=LINE_WIDTH_MAIN,
            )
            ax_mer.plot(
                [x_in, x_in],
                [r_hub_in, r_tip_in],
                color=color,
                linewidth=LINE_WIDTH_SECONDARY,
                linestyle='--',
                alpha=0.6,
            )
            ax_mer.plot(
                [x_out, x_out],
                [r_hub_out, r_tip_out],
                color=color,
                linewidth=LINE_WIDTH_SECONDARY,
                linestyle='--',
                alpha=0.6,
            )
        except Exception as e:
            print(
                f'Error in meridional pair {pair_idx} corner ({phi_idx},{psi_idx}): {e}'
            )

    # --- Camber lines ---
    ax_cam.set_aspect('equal', adjustable='box')
    ax_cam.set_ylabel(r'$\theta$', fontsize=AXIS_LABEL_SIZE)
    ax_cam.set_xlabel(r'$\tilde{z}$', fontsize=AXIS_LABEL_SIZE)
    ax_cam.tick_params('both', labelsize=TICK_SIZE)
    ax_cam.grid(True, alpha=0.3)

    for pair_idx, (node_in, node_out) in enumerate(node_pairs):
        if pair_offsets[pair_idx] is None:
            continue
        try:
            ma_in = sol_dict.get(node.geo.MetalAngle._at_node(node_in))
            ma_out = sol_dict.get(node.geo.MetalAngle._at_node(node_out))
            chord_ax = sol_dict.get(node.geo.ChordAx._at_node(node_out))
            pitch = sol_dict.get(node.geo.Pitch._at_node(node_out))
            if any(v is None for v in (ma_in, ma_out, chord_ax, pitch)):
                continue
            ma_in_v = ma_in[0] if hasattr(ma_in, '__len__') else ma_in
            ma_out_v = ma_out[0] if hasattr(ma_out, '__len__') else ma_out
            chord_ax_v = chord_ax[0] if hasattr(chord_ax, '__len__') else chord_ax
            pitch_v = pitch[0] if hasattr(pitch, '__len__') else pitch
            color = colors[pair_idx]
            offset = pair_offsets[pair_idx]
            for blade_num in range(2):
                pbl.plot_camber_line(
                    ax_cam,
                    ma_in_v,
                    ma_out_v,
                    chord_ax_v / r_mean,
                    color,
                    axial_offset=offset / r_mean,
                    tangential_offset=blade_num * pitch_v / r_mean,
                    linewidth=LINE_WIDTH_MAIN,
                )
        except Exception as e:
            print(f'Error in camber pair {pair_idx} corner ({phi_idx},{psi_idx}): {e}')
    ax_cam.set_ylim(-0.05, 0.3)
fig.savefig(
    'C:\\Users\\fvaccari\\OneDrive - Delft University of Technology\\latex'
    '\\gpps26_ADeT\\Images\\design_geometries.svg'
)

# ========================== PLOT DESIGN MAP QUANTITIES
print('Plotting design map quantities...')

# Figure 1: TT Efficiency contour map
fig, ax = plt.subplots(figsize=(8, 6))
plot_contour_quantity(
    n3.ndim.EtaTT,
    ax,
    design_map,
    clabel=r'$\eta_{tt}$',
)

# fig.savefig(
#     'C:\\Users\\fvaccari\\OneDrive - Delft University of Technology\\latex'
#     '\\gpps26_ADeT\\Images\\design_map_orc.pdf'
# )
# ax.set_title(
#     f'Total-to-Total Efficiency | identifier {IDENTIFIER} | R={REACTION}',
#     fontsize=12,
# )

plt.tight_layout()

if PLOT_LOSSES:
    # Figure 2: Stator losses
    print('Plotting stator losses...')
    fig_stator, axes_stator = plt.subplots(2, 2, figsize=(12, 10))
    axes_stator = axes_stator.flatten()

    loss_keys_stator = [
        (n1.loss.Ds_profile, 'Profile Loss'),
        (n1.loss.Ds_secondary, 'Secondary Loss'),
        (n1.loss.Ds_mixing, 'Mixing Loss'),
    ]

    for idx, (qty_key, label) in enumerate(loss_keys_stator):
        plot_contour_quantity(
            qty_key,
            axes_stator[idx],
            design_map,
            clabel=f'Stator {label}',
        )

    fig_stator.suptitle(
        f'Stator Loss Components | identifier {IDENTIFIER} | R={REACTION}',
        fontsize=16,
        y=1.00,
    )
    plt.tight_layout()

    # Figure 3: Rotor losses
    print('Plotting rotor losses...')
    fig_rotor, axes_rotor = plt.subplots(2, 2, figsize=(12, 10))
    axes_rotor = axes_rotor.flatten()

    loss_keys_rotor = [
        (n3.loss.Ds_profile, 'Profile Loss'),
        (n3.loss.Ds_secondary, 'Secondary Loss'),
        (n3.loss.Ds_mixing, 'Mixing Loss'),
        (n3.loss.Ds_leakage, 'Leakage Loss'),
    ]

    for idx, (qty_key, label) in enumerate(loss_keys_rotor):
        plot_contour_quantity(
            qty_key,
            axes_rotor[idx],
            design_map,
            clabel=f'Rotor {label}',
        )

    fig_rotor.suptitle(
        f'Rotor Loss Components | identifier {IDENTIFIER} | R={REACTION}',
        fontsize=16,
        y=1.00,
    )
    plt.tight_layout()

# plt.show(block=False)
