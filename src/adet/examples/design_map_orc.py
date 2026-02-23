# === IMPORTS
from copy import deepcopy
import logging
from pathlib import Path
from typing import Type

import CoolProp as cp
import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity

from adet.assembly import CasadiSystem
from adet.components import BladeRow, Inlet, Shaft
from adet.components import ComponentNetwork
from adet.components.blade_row import RowGeometry
from adet.equations.base_equation import EquationBase, LossApplier
from adet.equations.definitions import (
    BoundaryLayerRatios,
    ClearanceByHeight,
    IsentropicProperties,
    RepeatedStage,
)
from adet.equations.geometrical import MinimalCamberLine, ParabolicCamberline
from adet.equations.nondimensional import (
    FlowCoefficient,
    StaticTotalDegreeOfReaction,
    TotalStaticLoadingCoefficient,
    TotalTotalExpansionEfficiency,
    WorkCoefficient,
)
from adet.fluid.settings import ExternalFluidModel, FluidSettings
from adet.losses.basic import PercentageEntropyLoss, ZeroDeviation
from adet.losses.leakage import DentonLeakageLoss
from adet.losses.mixing import SieverdingBasePressure
from adet.losses.profile import DentonProfileLoss
from adet.losses.secondary import SecondaryBSM
from adet.registries import DefaultUnitsRegistry, GuessRegistry, VariableBoundsRegistry
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.iter import grouper
from adet.tools.loggers import setup_logger

logger = logging.getLogger(__name__)

setup_logger(
    logger,
    logging.INFO,
    logging.INFO,
    banned_keywords=['STREAM', 'findfont', 'sBIT'],
)
plt.close('all')

# === SETTINGS
NUM_SPAN = 1
SCALED = True
INITIAL_LOSS = PercentageEntropyLoss(0.0)

# ================================================
# *** Loss transition helpers


class LossMatcher(LossApplier):
    def __init__(
        self,
        tip_gap: bool,
        scaling_factor: list[float] | None = None,
    ):
        super().__init__(scaling_factor)
        self.tip_gap = tip_gap

    def residual(
        self,
        stc_smass0,
        stc_smass1,
        oth_delta_smass_leakage1,
        oth_delta_smass_profile1,
        oth_delta_smass_secondary1,
    ):
        if self.tip_gap:
            return stc_smass1 - (
                stc_smass0
                + oth_delta_smass_leakage1
                + oth_delta_smass_profile1
                + oth_delta_smass_secondary1
            )
        return stc_smass1 - (
            stc_smass0 + oth_delta_smass_profile1 + oth_delta_smass_secondary1
        )


EXTRA_EQUATIONS: dict[Type[EquationBase], int | tuple[int, ...]] = {
    ClearanceByHeight: 1,
    IsentropicProperties: (0, 1),
    BoundaryLayerRatios: 1,
    SieverdingBasePressure: (0, 1),
    SecondaryBSM: (0, 1),
    DentonProfileLoss: (0, 1),
    DentonLeakageLoss: (0, 1),
}

# === DESIGN MAP PARAMETERS
N_PHI = 20
N_PSI = 20
PHI_RANGE = np.linspace(0.4, 1.4, N_PHI)
PSI_RANGE = np.linspace(3.0, 10.0, N_PSI)

# Indices at which to save meridional channels (3x3 = 9 points)
MERID_INDICES = [0, N_PHI // 2, N_PHI - 1]

# ================================================
abs_state = DebugAbstractState('REFPROP', 'MM')
abs_state.debug_print = False

real_model = ExternalFluidModel(abs_state)
INLET_PRESSURE = 2.071 * abs_state.p_critical()
INLET_TEMPERATURE = 1.052 * abs_state.T_critical()
abs_state.update(cp.PT_INPUTS, INLET_PRESSURE, INLET_TEMPERATURE)

fluid_settings = FluidSettings(
    model=real_model,
    update_variables=('p', 'hmass', 'T'),
    update_length=2,
)

_defreg = DefaultUnitsRegistry()
_defreg.from_dict(
    {
        'xi_by_camb_.*': 'dimensionless',
        'Cd_prof': 'dimensionless',
        'k_prof': 'dimensionless',
        'Y_.*': 'dimensionless',
    }
)
_guess_reg = GuessRegistry()
_guess_reg.reset()
_guess_reg.from_dict(
    {
        'hdropCoeff': -0.8,
        'workCoeff': -0.8,
        'reactDegree_ts': 0.5,
        'p': abs_state.p(),
        'T': abs_state.T(),
        'hmass': abs_state.hmass(),
        'smass': abs_state.smass(),
        'rhomass': abs_state.rhomass(),
        'k_prof': 0.3,
    }
)

_bounds_reg = VariableBoundsRegistry()
_bounds_reg.reset()
_bounds_reg.from_dict(
    {
        'hdropCoeff': (-8.0, -0.4),
        'U': (0.0, 200.0),
    }
)
if fluid_settings.model == real_model:
    _bounds_reg.from_dict(
        {
            'p': (abs_state.p_critical(), INLET_PRESSURE),
            'T': (abs_state.T_critical(), INLET_TEMPERATURE),
            'hmass': (abs_state.hmass() - 2 * 60**2, 1.2 * abs_state.hmass()),
        }
    )

# ================================================
casing = Shaft(0.0, is_constrained=True)
shaft = Shaft(-1, is_constrained=False)

# ================================================
inlet = Inlet(
    {
        'oth': {'cum_massflow': 1},
        'kin': {'mermach': 0.1},
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'hubtipRatio': 0.7,
        },
        'tot': {
            'p': abs_state.p(),
            'hmass': abs_state.hmass(),
        },
    }
)

stator = BladeRow(
    name='Stator',
    shaft=casing,
    row_type='stator',
    in_constraints={'geo': {'thick_by_pitch': 0.04}},
    out_constraints={
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'aspRatio': 2,
            'num_blades': 25,
            'thick_by_pitch': 0.02,
            'clearance_by_height': 0.01,
        },
        'oth': {
            'mom_by_bld': 0.075,
            'disp_by_mom': 2,
            'disp_by_hgt': 0.05,
            'Cd_profile': 0.002,
            'xi_by_camb_len_A': 0.375,
            'xi_by_camb_len_B': 0.675,
            'dischCoeff': 0.35,
        },
    },
    extra_equations={
        MinimalCamberLine(): (0, 1),
        ZeroDeviation(): 0,
        ZeroDeviation(): 1,
        INITIAL_LOSS: (0, 1),
    },
    constant_variables=['geo_rr_midspan'],
)

rotor = deepcopy(stator)
rotor.shaft = shaft
rotor.row_type = 'rotor'
rotor._equations.update(
    {
        WorkCoefficient(): (0, 1),
        FlowCoefficient(): (0, 1),
    }
)

# ================================================
ntw = ComponentNetwork(
    fluid_settings,
    inlet,
    CasadiSystem(num_span=NUM_SPAN),
    components=[stator, rotor],
)

rotor.set_boundary_cond('oth_flowCoeff1', PHI_RANGE[0])
rotor.set_boundary_cond('oth_reactDegree_ts1', 0.3)
rotor.set_boundary_cond('oth_ts_loadCoeff1', PSI_RANGE[0])

final_node = ntw.num_components * 2 - 1

stator.set_spanwise_constant('geo_hh0', 'geo_chord_ax1')
rotor.set_spanwise_constant('geo_chord_ax1')
ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, final_node))
ntw.system.add_equation(StaticTotalDegreeOfReaction(), (0, 1, 2, 3))
ntw.system.add_equation(TotalStaticLoadingCoefficient(), (0, final_node))
ntw.system.add_equation(RepeatedStage(), (0, 1, 2, 3))

ntw.system.build(SCALED)

# ================================================
flow_coeff_name = f'oth_flowCoeff{final_node}'
load_coeff_name = f'oth_ts_loadCoeff{final_node}'

# Constraint indices for the isentropic system (re-fetched after loss rebuild below)
flow_coeff_idx = ntw.system.constraints.index(flow_coeff_name)
load_coeff_idx = ntw.system.constraints.index(load_coeff_name)

# ================================================
# Isentropic initial IPOPT solve
rootfinder_ipopt = ntw.system.make_rootfinder(
    'ipopt',
    opts={'error_on_fail': False, 'ipopt.max_iter': 1000},
)

x0 = ntw.system.get_scaled_guess()
kn = ntw.system.get_scaled_constraints()
bnd = ntw.system.get_arguments_bounds()

logger.info(f'Solving isentropic initial point: phi={PHI_RANGE[0]:.3f}, psi={PSI_RANGE[0]:.3f}')
solution = solve_root_problem(
    rootfinder_ipopt, x0, kn, bnd, suppress_output=False, perturbate_guess=False
)

# ================================================
# Transition to losses (mirrors axial_orc.py second phase)
sol_dict_is = ntw.system.solution_to_dict(solution)

ntw.system.remove_equation_type(LossApplier)
rotor.remove_equation(INITIAL_LOSS.__class__, (0, 1))
stator.remove_equation(INITIAL_LOSS.__class__, (0, 1))
for eq, pos in EXTRA_EQUATIONS.items():
    rotor.add_equation(eq(), pos)
    stator.add_equation(eq(), pos)
rotor.add_equation(LossMatcher(tip_gap=True), (0, 1))
stator.add_equation(LossMatcher(tip_gap=False), (0, 1))
ntw.build()

# Re-fetch constraint indices after rebuild
flow_coeff_idx = ntw.system.constraints.index(flow_coeff_name)
load_coeff_idx = ntw.system.constraints.index(load_coeff_name)
logger.info(
    f'Loss system constraint indices: {flow_coeff_name}={flow_coeff_idx}, '
    f'{load_coeff_name}={load_coeff_idx}'
)

# Solve loss initial point with IPOPT (using isentropic solution as warm start)
x0_loss = ntw.system.get_scaled_guess(sol_dict_is)
kn_loss = ntw.system.get_scaled_constraints()
bnd_loss = ntw.system.get_arguments_bounds()
rootfinder_ipopt_loss = ntw.system.make_rootfinder(
    'ipopt',
    opts={'error_on_fail': False, 'ipopt.max_iter': 1000},
)
logger.info(f'Solving loss initial point: phi={PHI_RANGE[0]:.3f}, psi={PSI_RANGE[0]:.3f}')
solution = solve_root_problem(
    rootfinder_ipopt_loss, x0_loss, kn_loss, bnd_loss, suppress_output=False, perturbate_guess=False
)

# ================================================
# Kinsol for the loss sweep
rootfinder_kinsol = ntw.system.make_rootfinder(
    'kinsol',
    opts={'error_on_fail': False},
)

# ================================================
# Meridional + camberline extraction
_pbl = ParabolicCamberline()
N_CAMBER_PTS = 50
N_BLADES_PLOT = 3


def extract_meridional():
    nodes = ntw.system.nodes
    midspan = ntw.system.num_span // 2
    x, r_hub, r_tip, r_mid = [], [], [], []
    camberlines = []  # list of dicts, one per blade row
    x_offset = 0.0

    for n0_idx, n1_idx in grouper(range(len(nodes)), 2, incomplete='ignore'):
        n0, n1 = nodes[n0_idx], nodes[n1_idx]
        chord_ax = float(n1.geo.get('chord_ax').to_base_units().magnitude[0])
        rr0 = float(n0.geo.get('rr_midspan').to_base_units().magnitude[0])
        hh0 = float(n0.geo.get('height').to_base_units().magnitude[0])
        rr1 = float(n1.geo.get('rr_midspan').to_base_units().magnitude[0])
        hh1 = float(n1.geo.get('height').to_base_units().magnitude[0])

        x += [x_offset, x_offset + chord_ax]
        r_hub += [rr0 - hh0 / 2, rr1 - hh1 / 2]
        r_tip += [rr0 + hh0 / 2, rr1 + hh1 / 2]
        r_mid += [rr0, rr1]

        # Camberline at midspan
        alpha_in = float(n0.geo.metal_angle[midspan])
        alpha_out = float(n1.geo.metal_angle[midspan])
        pitch = float(n1.geo.pitch[midspan])

        a, b, _ = _pbl._compute_parabola(alpha_in, alpha_out, chord_ax)
        xc = np.linspace(0, chord_ax, N_CAMBER_PTS)
        yc = a * xc**2 + b * xc

        mer_angle_in = float(n0.geo.get('meridional_angle').to_base_units().magnitude[0])
        mer_angle_out = float(n1.geo.get('meridional_angle').to_base_units().magnitude[0])

        camberlines.append(
            {
                'x_offset': x_offset,
                'pitch': pitch,
                'xc': xc,
                'yc': yc,
                'alpha_in': alpha_in,
                'alpha_out': alpha_out,
                'alpha_hub_in': float(n0.geo.metal_angle[0]),
                'alpha_hub_out': float(n1.geo.metal_angle[0]),
                'alpha_tip_in': float(n0.geo.metal_angle[-1]),
                'alpha_tip_out': float(n1.geo.metal_angle[-1]),
                'chord_ax': chord_ax,
                'rr0': rr0,
                'rr1': rr1,
                'hh0': hh0,
                'hh1': hh1,
                'mer_angle_in': mer_angle_in,
                'mer_angle_out': mer_angle_out,
            }
        )

        x_offset += chord_ax * 1.1

    return {
        'x': np.array(x),
        'r_hub': np.array(r_hub),
        'r_tip': np.array(r_tip),
        'r_mid': np.array(r_mid),
        'camberlines': camberlines,
    }


# ================================================
# Design map sweep (loss system)
eta_tt_map = np.full((N_PHI, N_PSI), np.nan)
converged_map = np.zeros((N_PHI, N_PSI), dtype=bool)
meridional_data = {}  # keyed by (i_phi, j_psi)

eta_tt_key = f'oth_eta_tt{final_node}'


def extract_eta_tt(sol):
    sol_dict = ntw.system.solution_to_dict(sol)
    if eta_tt_key in sol_dict:
        return float(sol_dict[eta_tt_key][0])
    return np.nan


prev_solution = solution
converged_map[0, 0] = True
eta_tt_map[0, 0] = extract_eta_tt(solution)

# Save meridional at (0,0)
if 0 in MERID_INDICES:
    ntw.system.write_solution_to_nodes(solution)
    if 0 in MERID_INDICES:
        for j_save in MERID_INDICES:
            if j_save == 0:
                meridional_data[(0, 0)] = extract_meridional()

for i, phi in enumerate(PHI_RANGE):
    for j, psi in enumerate(PSI_RANGE):
        if i == 0 and j == 0:
            continue

        ntw.system.data.constraints_values[flow_coeff_idx] = np.array([phi])
        ntw.system.data.constraints_values[load_coeff_idx] = np.array([psi])
        kn = ntw.system.get_scaled_constraints()

        logger.info(f'Solving phi={phi:.3f}, psi={psi:.3f}')
        try:
            sol = solve_root_problem(
                rootfinder_kinsol, prev_solution, kn, suppress_output=True
            )
            eta_tt_map[i, j] = extract_eta_tt(sol)
            converged_map[i, j] = True
            prev_solution = sol

            # Save meridional channel at selected grid points
            if i in MERID_INDICES and j in MERID_INDICES:
                ntw.system.write_solution_to_nodes(sol)
                meridional_data[(i, j)] = extract_meridional()

        except Exception as e:
            logger.warning(f'Failed at phi={phi:.3f}, psi={psi:.3f}: {e}')

logger.info(f'Converged {converged_map.sum()} / {N_PHI * N_PSI} points (loss sweep)')

# ================================================
# Save meridional + camberline data
merid_save = {}
for (i, j), data in meridional_data.items():
    key = f'phi{i:02d}_psi{j:02d}'
    merid_save[f'{key}_x'] = data['x']
    merid_save[f'{key}_r_hub'] = data['r_hub']
    merid_save[f'{key}_r_tip'] = data['r_tip']
    merid_save[f'{key}_r_mid'] = data['r_mid']
    merid_save[f'{key}_phi_val'] = np.array([PHI_RANGE[i]])
    merid_save[f'{key}_psi_val'] = np.array([PSI_RANGE[j]])
    for br_idx, cl in enumerate(data['camberlines']):
        merid_save[f'{key}_br{br_idx}_xc'] = cl['xc']
        merid_save[f'{key}_br{br_idx}_yc'] = cl['yc']
        merid_save[f'{key}_br{br_idx}_pitch'] = np.array([cl['pitch']])
        merid_save[f'{key}_br{br_idx}_x_offset'] = np.array([cl['x_offset']])
        merid_save[f'{key}_br{br_idx}_alpha_in'] = np.array([cl['alpha_in']])
        merid_save[f'{key}_br{br_idx}_alpha_out'] = np.array([cl['alpha_out']])

np.savez(Path(__file__).parent / 'meridional_channels.npz', **merid_save)
logger.info(f'Saved {len(meridional_data)} meridional channels with camberlines')

# ================================================
# Plot meridional channels  (axial_orc.py style: blade outlines, stator/rotor colors)
NR = len(MERID_INDICES)
row_colors = ['steelblue', 'coral']  # stator, rotor

fig_m, ax_m = plt.subplots(NR, NR, figsize=(5 * NR, 4 * NR), sharex=False, sharey=False)
fig_m.suptitle('Meridional channels', fontsize=14)

for row, i in enumerate(MERID_INDICES):
    for col, j in enumerate(MERID_INDICES):
        ax = ax_m[row, col]
        key = (i, j)
        if key in meridional_data:
            d = meridional_data[key]
            x_offset_plot = 0.0
            for br_idx, cl in enumerate(d['camberlines']):
                color = row_colors[br_idx % len(row_colors)]
                geom = RowGeometry(
                    r_in=cl['rr0'],
                    r_out=cl['rr1'],
                    height_in=cl['hh0'],
                    height_out=cl['hh1'],
                    mer_angle_in=cl['mer_angle_in'],
                    mer_angle_out=cl['mer_angle_out'],
                    axial_chord=cl['chord_ax'],
                    axial_offset=x_offset_plot,
                )
                for line in geom.plot_meridional_profile(color, ax=ax):
                    line.set_linewidth(2.5)
                # Midspan dots at LE and TE
                ax.plot(x_offset_plot, cl['rr0'], 'o', color='r', markersize=4)
                ax.plot(x_offset_plot + cl['chord_ax'], cl['rr1'], 'o', color='r', markersize=4)
                # Hub (blue) and tip (green) tick markers
                ax.plot(x_offset_plot, cl['rr0'] - cl['hh0'] / 2, '_', color='b', markersize=8)
                ax.plot(x_offset_plot, cl['rr0'] + cl['hh0'] / 2, '_', color='g', markersize=8)
                ax.plot(x_offset_plot + cl['chord_ax'], cl['rr1'] - cl['hh1'] / 2, '_', color='b', markersize=8)
                ax.plot(x_offset_plot + cl['chord_ax'], cl['rr1'] + cl['hh1'] / 2, '_', color='g', markersize=8)
                x_offset_plot += cl['chord_ax'] * 1.1
            # Axis reference line
            ax.plot([0.0, x_offset_plot], [0.0, 0.0], color='r', linestyle='dashdot', linewidth=1.5)
            ax.set_title(
                rf'$\phi$={PHI_RANGE[i]:.2f}, $\psi_{{ts}}$={PSI_RANGE[j]:.1f}',
                fontsize=9,
            )
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        if row == NR - 1:
            ax.set_xlabel('Axial [m]', fontsize=8)
        if col == 0:
            ax.set_ylabel('Radius [m]', fontsize=8)

fig_m.tight_layout()
fig_m.savefig(Path(__file__).parent / 'meridional_channels.png', dpi=150)

# ================================================
# Plot camberlines  (axial_orc.py style: midspan=black, hub=orange, tip=seagreen)
fig_c, ax_c = plt.subplots(NR, NR, figsize=(4 * NR, 3 * NR), sharex=False, sharey=False)
fig_c.suptitle('Camberlines at midspan (3 blades per row)', fontsize=14)

for row, i in enumerate(MERID_INDICES):
    for col, j in enumerate(MERID_INDICES):
        ax = ax_c[row, col]
        key = (i, j)
        if key in meridional_data:
            d = meridional_data[key]
            for br_idx, cl in enumerate(d['camberlines']):
                # Midspan blades in black
                for blade_num in range(N_BLADES_PLOT):
                    ax.plot(
                        cl['x_offset'] + cl['xc'],
                        blade_num * cl['pitch'] + cl['yc'],
                        color='k',
                        linewidth=1.5,
                    )
                # Hub camberline (orange)
                a_h, b_h, _ = _pbl._compute_parabola(cl['alpha_hub_in'], cl['alpha_hub_out'], cl['chord_ax'])
                yc_h = a_h * cl['xc'] ** 2 + b_h * cl['xc']
                ax.plot(cl['x_offset'] + cl['xc'], yc_h, color='orange', linewidth=1.5)
                # Tip camberline (seagreen)
                a_t, b_t, _ = _pbl._compute_parabola(cl['alpha_tip_in'], cl['alpha_tip_out'], cl['chord_ax'])
                yc_t = a_t * cl['xc'] ** 2 + b_t * cl['xc']
                ax.plot(cl['x_offset'] + cl['xc'], yc_t, color='seagreen', linewidth=1.5)
            ax.set_title(
                rf'$\phi$={PHI_RANGE[i]:.2f}, $\psi_{{ts}}$={PSI_RANGE[j]:.1f}',
                fontsize=9,
            )
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        if row == NR - 1:
            ax.set_xlabel('Axial [m]', fontsize=8)
        if col == 0:
            ax.set_ylabel('Tangential [m]', fontsize=8)

fig_c.tight_layout()
fig_c.savefig(Path(__file__).parent / 'camberlines.png', dpi=150)

# ================================================
# Plot design map  (phi on x-axis, psi on y-axis)
fig, ax = plt.subplots(figsize=(8, 6))

eta_masked = np.where(converged_map, eta_tt_map, np.nan)

# Transpose: eta_tt_map[i_phi, j_psi] -> contourf(phi, psi, map.T)
cf = ax.contourf(PHI_RANGE, PSI_RANGE, eta_masked.T, levels=15, cmap='viridis')
cs = ax.contour(
    PHI_RANGE,
    PSI_RANGE,
    eta_masked.T,
    levels=15,
    colors='w',
    linewidths=0.5,
    alpha=0.6,
)
ax.clabel(cs, fmt='%.3f', fontsize=8)

# Mark the meridional sample points
for i in MERID_INDICES:
    for j in MERID_INDICES:
        ax.plot(PHI_RANGE[i], PSI_RANGE[j], 'w+', markersize=8, markeredgewidth=1.5)

cbar = fig.colorbar(cf, ax=ax)
cbar.set_label(r'Total-to-total efficiency $\eta_{tt}$ [-]', fontsize=14)

ax.set_xlabel(r'Flow coefficient $\phi$ [-]', fontsize=14)
ax.set_ylabel(r'Loading coefficient $\psi_{ts}$ [-]', fontsize=14)
ax.set_title('ORC Axial Turbine — Loss-Based Design Map', fontsize=15)
ax.grid(True, alpha=0.3)
ax.plot(PHI_RANGE[0], PSI_RANGE[0], 'w*', markersize=12, label='Reference design')
ax.legend(fontsize=11)

fig.tight_layout()
fig.savefig(Path(__file__).parent / 'design_map_orc.png', dpi=150)

plt.show(block=False)
try:
    input('Press enter to close')
except EOFError:
    pass
plt.close('all')
