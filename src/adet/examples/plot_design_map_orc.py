"""
Regenerate meridional channel plots from saved design map data
"""

import pickle
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from adet.equations.geometrical import ParabolicCamberline

# Constants
CONTOUR_LEVELS = 50

# Setup paths
data_dir = pathlib.Path(__file__).parent.parent.parent.parent / 'outputs'

# Load all data from pickle file
print('Loading design map data from pickle file...')
with open(data_dir / 'design_map_orc.pkl', 'rb') as f:
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

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

print('Plotting meridional channels...')
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

    # Plot meridional geometry with straight lines
    # Solution dict has keys like 'geo_rr0', 'geo_height0', 'geo_chord_ax0', etc.
    # Nodes: 0 (inlet), 1 (stator outlet/rotor inlet), 2 (rotor outlet)

    offset = 0.0
    # Node indices for stator: 0->1, rotor: 1->2
    node_pairs = [(0, 1), (2, 3)]  # Assuming stator and rotor

    colors = ['steelblue', 'coral']

    for pair_idx, (node_in, node_out) in enumerate(node_pairs):
        try:
            # Get geometry data from solution dict
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

            # Handle both scalar and array cases
            rr_in_val = rr_in[0] if hasattr(rr_in, '__len__') else rr_in
            height_in_val = height_in[0] if hasattr(height_in, '__len__') else height_in
            rr_out_val = rr_out[0] if hasattr(rr_out, '__len__') else rr_out
            height_out_val = (
                height_out[0] if hasattr(height_out, '__len__') else height_out
            )
            chord_ax_val = (
                chord_ax_out[0] if hasattr(chord_ax_out, '__len__') else chord_ax_out
            )

            # Calculate inlet/outlet positions and radii
            x_inlet = offset
            x_outlet = offset + chord_ax_val

            r_hub_inlet = rr_in_val - height_in_val / 2
            r_tip_inlet = rr_in_val + height_in_val / 2
            r_hub_outlet = rr_out_val - height_out_val / 2
            r_tip_outlet = rr_out_val + height_out_val / 2

            color = colors[pair_idx]

            # Plot hub line
            ax.plot(
                [x_inlet, x_outlet],
                [r_hub_inlet, r_hub_outlet],
                color=color,
                linewidth=2,
            )

            # Plot tip line
            ax.plot(
                [x_inlet, x_outlet],
                [r_tip_inlet, r_tip_outlet],
                color=color,
                linewidth=2,
            )

            # Plot inlet and outlet vertical lines
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
        except Exception as e:
            print(
                f'Error plotting pair {pair_idx} for point ({phi_idx}, {psi_idx}): {e}'
            )
            continue

    # Plot centerline
    ax.plot([0.0, offset], [0.0, 0.0], color='r', linestyle='dashdot', linewidth=2)

plt.tight_layout()
print(
    f'Meridional channels plot saved to {data_dir / "meridional_channels_6points.png"}'
)

# ========================== PLOT CAMBER LINES
print('Plotting camber lines...')

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

pbl = ParabolicCamberline()

for plot_idx, (linear_idx, phi_idx, psi_idx) in enumerate(selected_indices):
    if linear_idx >= len(solution_dicts):
        continue

    sol_dict = solution_dicts[linear_idx]

    # Skip if no solution
    if sol_dict is None:
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
    ax.set_ylabel('tangential [m]')
    ax.set_xlabel('axial [m]')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'φ={phi_vals[linear_idx]:.3f}, ψ={psi_vals[linear_idx]:.3f}')

    offset = 0.0
    node_pairs = [(0, 1), (2, 3)]
    colors_blade = ['steelblue', 'coral']

    for pair_idx, (node_in, node_out) in enumerate(node_pairs):
        try:
            # Get geometry data from solution dict
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

            # Handle both scalar and array cases
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

            # Plot 3 camberlines at tangential positions
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

            # Plot hub and tip camberlines
            metal_angle_in_hub = sol_dict.get(f'geo_metal_angle{node_in}', None)
            metal_angle_out_hub = sol_dict.get(f'geo_metal_angle{node_out}', None)
            if hasattr(metal_angle_in_hub, '__len__') and hasattr(
                metal_angle_out_hub, '__len__'
            ):
                pbl.plot_camber_line(
                    ax,
                    metal_angle_in_hub[0],
                    metal_angle_out_hub[0],
                    chord_ax_val,
                    'orange',
                    axial_offset=offset,
                    linewidth=1.5,
                    linestyle='--',
                    alpha=0.7,
                )
                pbl.plot_camber_line(
                    ax,
                    metal_angle_in_hub[-1],
                    metal_angle_out_hub[-1],
                    chord_ax_val,
                    'seagreen',
                    axial_offset=offset,
                    linewidth=1.5,
                    linestyle='--',
                    alpha=0.7,
                )

            offset += chord_ax_val * 1.1
        except Exception as e:
            print(
                f'Error plotting camberlines for pair {pair_idx} at '
                f'point ({phi_idx}, {psi_idx}): {e}'
            )
            continue

plt.tight_layout()
plt.savefig(data_dir / 'camberlines_6points.png', dpi=150, bbox_inches='tight')
print(f'Camberlines plot saved to {data_dir / "camberlines_6points.png"}')

# ========================== PLOT DESIGN MAP
print('Plotting design map...')


# Extract data from the pickle
eta_tt3_values = design_map['eta_tt3_values']
massflow = design_map['massflow']
keys_loss = design_map['keys_loss']

# Reshape data into 2D grids for contour plotting
phi_grid = phi_vals.reshape((N_PTS, N_PTS))
psi_grid = psi_vals.reshape((N_PTS, N_PTS))
eta_grid = eta_tt3_values.reshape((N_PTS, N_PTS))
mf_grid = massflow.reshape((N_PTS, N_PTS))

# Create design map contour plots
fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot: Efficiency map
contour1 = ax1.contourf(
    phi_grid, psi_grid, eta_grid, levels=CONTOUR_LEVELS, cmap='viridis'
)
contour1_lines = ax1.contour(
    phi_grid,
    psi_grid,
    eta_grid,
    levels=CONTOUR_LEVELS,
    colors='black',
    alpha=0.3,
    linewidths=0.5,
)
ax1.clabel(
    contour1_lines,
    inline=True,
    fontsize=10,
)
cbar1 = plt.colorbar(contour1, ax=ax1)
cbar1.set_label('Total-to-Total Efficiency', rotation=270, labelpad=20)
ax1.set_xlabel(r'Flow Coefficient $\phi$', fontsize=14)
ax1.set_ylabel(r'Loading Coefficient $\psi$', fontsize=14)
ax1.set_title('Design Map: Total-to-Total Efficiency', fontsize=16)
ax1.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(data_dir / 'design_map_orc.png', dpi=150, bbox_inches='tight')
print(f'Design map plot saved to {data_dir / "design_map_orc.png"}')

plt.show()
