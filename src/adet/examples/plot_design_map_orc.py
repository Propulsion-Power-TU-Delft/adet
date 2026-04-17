"""
Regenerate meridional channel plots from saved design map data
"""

import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline

from adet.equations.geometrical import ParabolicCamberline

# Constants
CONTOUR_LEVELS = 30
SPLINE_GRID_FACTOR = 3  # Refinement factor for spline grid

# Setup paths
data_dir = pathlib.Path(__file__).parent.parent.parent.parent / 'outputs'


def plot_contour_quantity(
    qty: str,
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
    qty : str
        Key for quantity to plot in solution dicts
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
    cbar.set_label(clabel if clabel else qty, rotation=270, labelpad=20)
    ax.set_xlabel(r'Flow Coefficient $\phi$', fontsize=12)
    ax.set_ylabel(r'Loading Coefficient $\psi$', fontsize=12)
    ax.set_title(f'{clabel if clabel else qty}', fontsize=14)
    ax.grid(True, alpha=0.2)

    return


def plot_profile_at_point(
    phi_target,
    psi_target,
    design_map,
    plot_type='both',
    figsize=(12, 5),
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

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
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

    if plot_type == 'both':
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax_single = plt.subplots(figsize=figsize)
        axes = [ax_single]

    # Plot meridional channel
    if plot_type in ('meridional', 'both'):
        ax = axes[0]
        ax.set_aspect('equal')
        ax.set_ylabel('radius [m]')
        ax.set_xlabel('axial [m]')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Meridional Channel: φ={phi_actual:.3f}, ψ={psi_actual:.3f}')

        offset = 0.0
        node_pairs = [(0, 1), (2, 3)]
        colors = ['steelblue', 'coral']

        for pair_idx, (node_in, node_out) in enumerate(node_pairs):
            rr_in = sol_dict.get(f'geo_rr{node_in}', None)
            height_in = sol_dict.get(f'geo_height{node_in}', None)
            chord_ax_out = sol_dict.get(f'geo_chord_ax{node_out}', None)
            rr_out = sol_dict.get(f'geo_rr{node_out}', None)
            height_out = sol_dict.get(f'geo_height{node_out}', None)

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

            x_inlet = offset
            x_outlet = offset + chord_ax_val

            r_hub_inlet = rr_in_val - height_in_val / 2
            r_tip_inlet = rr_in_val + height_in_val / 2
            r_hub_outlet = rr_out_val - height_out_val / 2
            r_tip_outlet = rr_out_val + height_out_val / 2

            color = colors[pair_idx]

            ax.plot(
                [x_inlet, x_outlet],
                [r_hub_inlet, r_hub_outlet],
                color=color,
                linewidth=2,
            )
            ax.plot(
                [x_inlet, x_outlet],
                [r_tip_inlet, r_tip_outlet],
                color=color,
                linewidth=2,
            )
            ax.plot(
                [x_inlet, x_inlet],
                [r_hub_inlet, r_tip_inlet],
                color=color,
                linewidth=1.5,
                linestyle='--',
                alpha=0.6,
            )
            ax.plot(
                [x_outlet, x_outlet],
                [r_hub_outlet, r_tip_outlet],
                color=color,
                linewidth=1.5,
                linestyle='--',
                alpha=0.6,
            )

            offset += chord_ax_val * 1.05

        ax.plot([0.0, offset], [0.0, 0.0], color='r', linestyle='dashdot', linewidth=2)

    # Plot camber lines
    if plot_type in ('camber', 'both'):
        ax = axes[1] if plot_type == 'both' else axes[0]
        ax.set_aspect('equal')
        ax.set_ylabel('tangential [m]')
        ax.set_xlabel('axial [m]')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Camber Lines: φ={phi_actual:.3f}, ψ={psi_actual:.3f}')

        pbl = ParabolicCamberline()
        offset = 0.0
        node_pairs = [(0, 1), (2, 3)]
        colors_blade = ['steelblue', 'coral']

        for pair_idx, (node_in, node_out) in enumerate(node_pairs):
            metal_angle_in = sol_dict.get(f'geo_metal_angle{node_in}', None)
            metal_angle_out = sol_dict.get(f'geo_metal_angle{node_out}', None)
            chord_ax = sol_dict.get(f'geo_chord_ax{node_out}', None)
            pitch = sol_dict.get(f'geo_pitch{node_out}', None)

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

            num_blades = 3
            for blade_num in range(num_blades):
                pbl.plot_camber_line(
                    ax,
                    metal_angle_in_val,
                    metal_angle_out_val,
                    chord_ax_val,
                    color,
                    axial_offset=offset,
                    tangential_offset=blade_num * pitch_val,
                    linewidth=1.5,
                )

            if hasattr(metal_angle_in_val, '__len__') and hasattr(
                metal_angle_out_val, '__len__'
            ):
                pbl.plot_camber_line(
                    ax,
                    metal_angle_in_val[0],
                    metal_angle_out_val[0],
                    chord_ax_val,
                    'orange',
                    axial_offset=offset,
                    linewidth=1.5,
                    linestyle='--',
                    alpha=0.7,
                )
                pbl.plot_camber_line(
                    ax,
                    metal_angle_in_val[-1],
                    metal_angle_out_val[-1],
                    chord_ax_val,
                    'seagreen',
                    axial_offset=offset,
                    linewidth=1.5,
                    linestyle='--',
                    alpha=0.7,
                )

            offset += chord_ax_val * 1.1

    plt.tight_layout()
    return fig


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

# Select 6 representative points from the phi/psi grid
phi_indices = [0, N_PTS // 2, N_PTS - 1]
psi_indices = [0, N_PTS - 1]

selected_indices = []
for phi_idx in phi_indices:
    for psi_idx in psi_indices:
        linear_idx = phi_idx * N_PTS + psi_idx
        selected_indices.append((linear_idx, phi_idx, psi_idx))

fig, axes = plt.subplots(2, 3, figsize=(16, 12))
fig.suptitle(
    f'Meridional Channels with Camber Lines | identifier {IDENTIFIER} | R={REACTION}',
    fontsize=14,
)
axes = axes.flatten()

pbl = ParabolicCamberline()

print('Plotting meridional channels with camber lines below...')
for plot_idx, (linear_idx, phi_idx, psi_idx) in enumerate(selected_indices):
    if linear_idx >= len(solution_dicts):
        continue

    sol_dict = solution_dicts[linear_idx]

    # Skip if no solution
    if sol_dict is None:
        print(f'Skipping point ({phi_idx}, {psi_idx}) - no solution')
        axes[plot_idx].text(
            0.5,
            0.5,
            'No solution',
            ha='center',
            va='center',
            transform=axes[plot_idx].transAxes,
        )
        axes[plot_idx].set_title(
            rf'$\phi$={phi_vals[linear_idx]:.3f}, $\psi$={psi_vals[linear_idx]:.3f}'
        )
        continue

    ax = axes[plot_idx]
    ax.set_aspect('equal')
    ax.set_ylabel('radius [m]')
    ax.set_xlabel('axial [m]')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'φ={phi_vals[linear_idx]:.3f}, ψ={psi_vals[linear_idx]:.3f}')

    offset = 0.0
    node_pairs = [(0, 1), (2, 3)]
    colors = ['steelblue', 'coral']

    # First pass: collect min radius
    min_radius = float('inf')
    for pair_idx, (node_in, node_out) in enumerate(node_pairs):
        try:
            rr_in = sol_dict.get(f'geo_rr{node_in}', None)
            height_in = sol_dict.get(f'geo_height{node_in}', None)
            chord_ax_out = sol_dict.get(f'geo_chord_ax{node_out}', None)
            rr_out = sol_dict.get(f'geo_rr{node_out}', None)
            height_out = sol_dict.get(f'geo_height{node_out}', None)

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

            r_hub_inlet = rr_in_val - height_in_val / 2
            r_hub_outlet = rr_out_val - height_out_val / 2
            min_radius = min(min_radius, r_hub_inlet, r_hub_outlet)
        except Exception:
            continue

    # Plot meridional geometry
    offset = 0.0
    for pair_idx, (node_in, node_out) in enumerate(node_pairs):
        try:
            rr_in = sol_dict.get(f'geo_rr{node_in}', None)
            height_in = sol_dict.get(f'geo_height{node_in}', None)
            chord_ax_out = sol_dict.get(f'geo_chord_ax{node_out}', None)
            rr_out = sol_dict.get(f'geo_rr{node_out}', None)
            height_out = sol_dict.get(f'geo_height{node_out}', None)

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

            x_inlet = offset
            x_outlet = offset + chord_ax_val

            r_hub_inlet = rr_in_val - height_in_val / 2
            r_tip_inlet = rr_in_val + height_in_val / 2
            r_hub_outlet = rr_out_val - height_out_val / 2
            r_tip_outlet = rr_out_val + height_out_val / 2

            color = colors[pair_idx]

            ax.plot(
                [x_inlet, x_outlet],
                [r_hub_inlet, r_hub_outlet],
                color=color,
                linewidth=2,
            )
            ax.plot(
                [x_inlet, x_outlet],
                [r_tip_inlet, r_tip_outlet],
                color=color,
                linewidth=2,
            )
            ax.plot(
                [x_inlet, x_inlet],
                [r_hub_inlet, r_tip_inlet],
                color=color,
                linewidth=1.5,
                linestyle='--',
                alpha=0.6,
            )
            ax.plot(
                [x_outlet, x_outlet],
                [r_hub_outlet, r_tip_outlet],
                color=color,
                linewidth=1.5,
                linestyle='--',
                alpha=0.6,
            )

            offset += chord_ax_val * 1.07
        except Exception as e:
            print(
                f'Error plotting pair {pair_idx} for point ({phi_idx}, {psi_idx}): {e}'
            )
            continue

    # Plot camberlines below minimum radius
    if min_radius != float('inf'):
        offset = 0.0
        for pair_idx, (node_in, node_out) in enumerate(node_pairs):
            try:
                metal_angle_in = sol_dict.get(f'geo_metal_angle{node_in}', None)
                metal_angle_out = sol_dict.get(f'geo_metal_angle{node_out}', None)
                chord_ax = sol_dict.get(f'geo_chord_ax{node_out}', None)
                pitch = sol_dict.get(f'geo_pitch{node_out}', None)

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

                color = colors[pair_idx]

                # Offset camberlines below the minimum radius
                camber_spacing = pitch_val * 3
                radial_offset = min_radius - camber_spacing - 0.01

                num_blades = 3
                for blade_num in range(num_blades):
                    pbl.plot_camber_line(
                        ax,
                        metal_angle_in_val,
                        metal_angle_out_val,
                        chord_ax_val,
                        color,
                        axial_offset=offset,
                        tangential_offset=radial_offset + blade_num * pitch_val,
                        linewidth=1.5,
                    )

                offset += chord_ax_val * 1.05
            except Exception as e:
                print(
                    f'Error plotting camberlines for pair {pair_idx} at '
                    f'point ({phi_idx}, {psi_idx}): {e}'
                )
                continue

plt.tight_layout()

# ========================== PLOT DESIGN MAP QUANTITIES
print('Plotting design map quantities...')

# Figure 1: TT Efficiency contour map
fig, ax = plt.subplots(figsize=(8, 6))
plot_contour_quantity(
    'oth_eta_tt3',
    ax,
    design_map,
    clabel='Total-to-Total Efficiency',
)
ax.set_title(
    f'Total-to-Total Efficiency | identifier {IDENTIFIER} | R={REACTION}',
    fontsize=12,
)
plt.tight_layout()

# Figure 2: Stator losses
print('Plotting stator losses...')
fig_stator, axes_stator = plt.subplots(2, 2, figsize=(12, 10))
axes_stator = axes_stator.flatten()

loss_keys_stator = [
    ('oth_delta_smass_profile1', 'Profile Loss'),
    ('oth_delta_smass_secondary1', 'Secondary Loss'),
    ('oth_delta_smass_mixing1', 'Mixing Loss'),
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
    ('oth_delta_smass_profile3', 'Profile Loss'),
    ('oth_delta_smass_secondary3', 'Secondary Loss'),
    ('oth_delta_smass_mixing3', 'Mixing Loss'),
    ('oth_delta_smass_leakage3', 'Leakage Loss'),
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

plt.show(block=False)
