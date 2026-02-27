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
from adet.components.blade_row import DownstreamMixer, RowGeometry, plot_from_nodes
from adet.equations.base_equation import EquationBase, LossApplier
from adet.equations.definitions import (
    BoundaryLayerRatios,
    ClearanceByHeight,
    IsentropicProperties,
    RepeatedStage,
)
from adet.equations.fundamental import BladeBlockage, ZeroBlockage
from adet.equations.geometrical import (
    MeridionalVariable,
    MinimalCamberLine,
    ModifiedZweifel,
    ParabolicCamberline,
)
from adet.equations.nondimensional import (
    FlowCoefficient,
    StaticTotalDegreeOfReaction,
    TotalStaticLoadingCoefficient,
    TotalTotalExpansionEfficiency,
    VolumetricFlowRatio,
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


# Mixing equations: blockage and boundary layer ratios
MIXING_EQS: dict[Type[EquationBase], int | tuple[int, ...]] = {
    BladeBlockage: 1,
    BoundaryLayerRatios: 1,
    SieverdingBasePressure: (0, 1),
}

# Loss models: isentropic properties + loss correlations
LOSS_MODELS: dict[Type[EquationBase], int | tuple[int, ...]] = {
    IsentropicProperties: (0, 1),
    SecondaryBSM: (0, 1),
    DentonProfileLoss: (0, 1),
    DentonLeakageLoss: (0, 1),
    ClearanceByHeight: 1,
}

# === DESIGN MAP PARAMETERS
N_PHI = 20
N_PSI = 20
PHI_RANGE = np.linspace(0.4, 1.3, N_PHI)
PSI_RANGE = np.linspace(3.0, 9.0, N_PSI)

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
        'camberCoeff': 1.0,
        'zweifelCoeff': 0.8,
        'volflowRatio': 4.0,
    }
)

_bounds_reg = VariableBoundsRegistry()
_bounds_reg.reset()
_bounds_reg.from_dict(
    {
        'hdropCoeff': (-8.0, -0.2),
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
            'thick_by_pitch': 0.02,
            'clearance_by_height': 0.01,
            'zweifelCoeff': 0.85,
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
        ModifiedZweifel(): (0, 1),
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
rotor.add_equation(WorkCoefficient(), (0, 1))
stator.set_boundary_cond('geo_flare_angle1', Quantity(30, 'deg'))
rotor.set_boundary_cond('geo_aspRatio1', 3.0)

mixer = DownstreamMixer('mixer')

# ================================================
ntw = ComponentNetwork(
    fluid_settings,
    inlet,
    CasadiSystem(num_span=NUM_SPAN),
    components=[stator, rotor],
)

rotor.bc_from_dict(
    {
        'oth_flowCoeff1': PHI_RANGE[0],
        'oth_reactDegree_ts1': 0.3,
        'oth_ts_loadCoeff1': PSI_RANGE[0],
    }
)

final_node = ntw.num_components * 2 - 1  # = 3

stator.set_spanwise_constant('geo_hh0', 'geo_chord_ax1')
rotor.set_spanwise_constant('geo_chord_ax1')
rotor.copy_from_previous('geo_hh', 'geo_rr')
rotor.remove_equation(MeridionalVariable, 0)

ntw.system.add_equation(RepeatedStage(), (0, 1, 2, 3))
ntw.system.add_equation(StaticTotalDegreeOfReaction(), (0, 1, 2, 3))
ntw.system.add_equation(FlowCoefficient(), (0, final_node))
ntw.system.add_equation(VolumetricFlowRatio(), (0, final_node))
ntw.system.add_equation(TotalStaticLoadingCoefficient(), (0, final_node))
ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, final_node))

ntw.system.build(SCALED)

# ================================================
# Isentropic initial IPOPT solve
rootfinder_ipopt = ntw.system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        'ipopt.max_iter': 1000,
        'ipopt.hessian_approximation': 'limited-memory',
    },
)

x0 = ntw.system.get_scaled_guess()
kn = ntw.system.get_scaled_constraints()
bnd = ntw.system.get_arguments_bounds({'kin_alpha0': (-0.1, 0.1)})

logger.info(
    f'Solving isentropic initial point: phi={PHI_RANGE[0]:.3f}, psi={PSI_RANGE[0]:.3f}'
)
solution = solve_root_problem(
    rootfinder_ipopt, x0, kn, bnd, suppress_output=False, perturbate_guess=False
)

# ================================================
# Transition to mixing (add mixers + blockage/BL equations)
sol_dict_is = ntw.system.write_solution_to_nodes(solution)

rotor.remove_equation(ZeroBlockage, 1)
stator.remove_equation(ZeroBlockage, 1)
for eq, pos in MIXING_EQS.items():
    stator.add_equation(eq(), pos)
    rotor.add_equation(eq(), pos)

rot_mixer = deepcopy(mixer)
rot_mixer.bc_from_dict(
    {
        'oth_flowCoeff1': PHI_RANGE[0],
        'oth_reactDegree_ts1': 0.3,
        'oth_ts_loadCoeff1': PSI_RANGE[0],
    }
)

ntw = ComponentNetwork(
    fluid_settings,
    inlet,
    CasadiSystem(num_span=NUM_SPAN),
    components=[
        stator,
        mixer,
        rotor,
        rot_mixer,
    ],
)

# Node indices in the mixing/loss network
mix_out = 2 * ntw.components.index(mixer) + 1  # = 3
rotor_in = 2 * ntw.components.index(rotor)      # = 4
rotor_out = rotor_in + 1                         # = 5
final_node = 2 * len(ntw.components) - 1        # = 7

STAGE_POSITIONS = (0, mix_out, rotor_in, final_node)

ntw.system.add_equation(RepeatedStage(), STAGE_POSITIONS)
ntw.system.add_equation(StaticTotalDegreeOfReaction(), STAGE_POSITIONS)
ntw.system.add_equation(FlowCoefficient(), (0, final_node))
ntw.system.add_equation(VolumetricFlowRatio(), (0, final_node))
ntw.system.add_equation(TotalStaticLoadingCoefficient(), (0, final_node))
ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, final_node))

ntw.build()

# Translate isentropic solution to mixing network node indices
sol_dict_mixing_guess = {
    k.replace('2', '4').replace('3', '5'): v for k, v in sol_dict_is.items()
}

x0_mixing = ntw.system.get_scaled_guess(sol_dict_mixing_guess)
kn_mixing = ntw.system.get_scaled_constraints()
bnd_mixing = ntw.system.get_arguments_bounds()

rootfinder_mixing = ntw.system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        'ipopt.max_iter': 1000,
        'ipopt.hessian_approximation': 'limited-memory',
    },
)
logger.info(
    f'Solving mixing initial point: phi={PHI_RANGE[0]:.3f}, psi={PSI_RANGE[0]:.3f}'
)
solution = solve_root_problem(
    rootfinder_mixing,
    x0_mixing,
    kn_mixing,
    bnd_mixing,
    suppress_output=False,
    perturbate_guess=False,
)

# ================================================
# Transition to losses
sol_dict_mixing = ntw.system.write_solution_to_nodes(solution)

rotor.remove_equation(INITIAL_LOSS.__class__, (0, 1))
stator.remove_equation(INITIAL_LOSS.__class__, (0, 1))
stator.add_equation(LossMatcher(tip_gap=False), (0, 1))
rotor.add_equation(LossMatcher(tip_gap=True), (0, 1))
for eq, pos in LOSS_MODELS.items():
    stator.add_equation(eq(), pos)
    rotor.add_equation(eq(), pos)

ntw.build()

# Update constraint names and indices for the loss system
flow_coeff_name = f'oth_flowCoeff{final_node}'
load_coeff_name = f'oth_ts_loadCoeff{final_node}'

flow_coeff_idx = ntw.system.constraints.index(flow_coeff_name)
load_coeff_idx = ntw.system.constraints.index(load_coeff_name)
logger.info(
    f'Loss system constraint indices: {flow_coeff_name}={flow_coeff_idx}, '
    f'{load_coeff_name}={load_coeff_idx}'
)

x0_loss = ntw.system.get_scaled_guess(sol_dict_mixing)
kn_loss = ntw.system.get_scaled_constraints()
bnd_loss = ntw.system.get_arguments_bounds()

rootfinder_ipopt_loss = ntw.system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        'ipopt.max_iter': 1000,
        'ipopt.hessian_approximation': 'limited-memory',
    },
)
logger.info(
    f'Solving loss initial point: phi={PHI_RANGE[0]:.3f}, psi={PSI_RANGE[0]:.3f}'
)
solution = solve_root_problem(
    rootfinder_ipopt_loss,
    x0_loss,
    kn_loss,
    bnd_loss,
    suppress_output=False,
    perturbate_guess=False,
)

# ================================================
# Kinsol for the loss sweep
rootfinder_kinsol = ntw.system.make_rootfinder(
    'kinsol',
    opts={'error_on_fail': True},
)

# ================================================
# Meridional + camberline extraction
_pbl = ParabolicCamberline()
N_CAMBER_PTS = 50
N_BLADES_PLOT = 3


def extract_meridional():
    midspan = ntw.system.num_span // 2
    x, r_hub, r_tip, r_mid = [], [], [], []
    camberlines = []  # list of dicts, one per blade row
    x_offset = 0.0

    for comp in ntw.components:
        if not isinstance(comp, BladeRow):
            continue

        n0 = comp.get_inlet_node(ntw)
        n1 = comp.get_outlet_node(ntw)

        chord_ax = float(n1.geo.get('chord_ax').to_base_units().magnitude[0])
        rr0 = float(n0.geo.get('rr_midspan').to_base_units().magnitude[0])
        hh0 = float(n0.geo.get('height').to_base_units().magnitude[0])
        rr1 = float(n1.geo.get('rr_midspan').to_base_units().magnitude[0])
        hh1 = float(n1.geo.get('height').to_base_units().magnitude[0])

        x += [x_offset, x_offset + chord_ax]
        r_hub += [rr0 - hh0 / 2, rr1 - hh1 / 2]
        r_tip += [rr0 + hh0 / 2, rr1 + hh1 / 2]
        r_mid += [rr0, rr1]

        alpha_in = float(n0.geo.metal_angle[midspan])
        alpha_out = float(n1.geo.metal_angle[midspan])
        pitch = float(n1.geo.pitch[midspan])

        xc = np.linspace(0, chord_ax, N_CAMBER_PTS)

        mer_angle_in = float(
            n0.geo.get('meridional_angle').to_base_units().magnitude[0]
        )
        mer_angle_out = float(
            n1.geo.get('meridional_angle').to_base_units().magnitude[0]
        )

        camberlines.append(
            {
                'x_offset': x_offset,
                'pitch': pitch,
                'xc': xc,
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


_phi_span = PHI_RANGE[-1] - PHI_RANGE[0]
_psi_span = PSI_RANGE[-1] - PSI_RANGE[0]


def find_closest_solution(
    phi: float,
    psi: float,
    store: dict[tuple[int, int], object],
) -> object:
    """Return the solution whose (phi, psi) grid point is closest to the target."""
    best_key = min(
        store,
        key=lambda k: (
            ((PHI_RANGE[k[0]] - phi) / _phi_span) ** 2
            + ((PSI_RANGE[k[1]] - psi) / _psi_span) ** 2
        ),
    )
    return store[best_key]


solution_store: dict[tuple[int, int], object] = {}
solution_store[(0, 0)] = solution
converged_map[0, 0] = True
eta_tt_map[0, 0] = extract_eta_tt(solution)

# Save meridional at (0, 0)
ntw.system.write_solution_to_nodes(solution)
if 0 in MERID_INDICES:
    meridional_data[(0, 0)] = extract_meridional()

for i, phi in enumerate(PHI_RANGE):
    for j, psi in enumerate(PSI_RANGE):
        if i == 0 and j == 0:
            continue

        ntw.system.data.constraints_values[flow_coeff_idx] = np.array([phi])
        ntw.system.data.constraints_values[load_coeff_idx] = np.array([psi])
        kn_loss = ntw.system.get_scaled_constraints()

        warm_start = find_closest_solution(phi, psi, solution_store)
        logger.info(f'Solving phi={phi:.3f}, psi={psi:.3f}')
        try:
            sol = solve_root_problem(
                rootfinder_kinsol,
                warm_start,
                kn_loss,
                bnd_loss,
                suppress_output=True,
            )
            eta_tt_map[i, j] = extract_eta_tt(sol)
            converged_map[i, j] = True
            solution_store[(i, j)] = sol

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
        merid_save[f'{key}_br{br_idx}_pitch'] = np.array([cl['pitch']])
        merid_save[f'{key}_br{br_idx}_x_offset'] = np.array([cl['x_offset']])
        merid_save[f'{key}_br{br_idx}_alpha_in'] = np.array([cl['alpha_in']])
        merid_save[f'{key}_br{br_idx}_alpha_out'] = np.array([cl['alpha_out']])

np.savez(Path(__file__).parent / 'meridional_channels.npz', **merid_save)
logger.info(f'Saved {len(meridional_data)} meridional channels with camberlines')

# ================================================
# Plot meridional channels (NR x NR grid)
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
                ax.plot(x_offset_plot, cl['rr0'], 'o', color='r', markersize=4)
                ax.plot(
                    x_offset_plot + cl['chord_ax'],
                    cl['rr1'],
                    'o',
                    color='r',
                    markersize=4,
                )
                ax.plot(
                    x_offset_plot,
                    cl['rr0'] - cl['hh0'] / 2,
                    '_',
                    color='b',
                    markersize=8,
                )
                ax.plot(
                    x_offset_plot,
                    cl['rr0'] + cl['hh0'] / 2,
                    '_',
                    color='g',
                    markersize=8,
                )
                ax.plot(
                    x_offset_plot + cl['chord_ax'],
                    cl['rr1'] - cl['hh1'] / 2,
                    '_',
                    color='b',
                    markersize=8,
                )
                ax.plot(
                    x_offset_plot + cl['chord_ax'],
                    cl['rr1'] + cl['hh1'] / 2,
                    '_',
                    color='g',
                    markersize=8,
                )
                x_offset_plot += cl['chord_ax'] * 1.1
            ax.plot(
                [0.0, x_offset_plot],
                [0.0, 0.0],
                color='r',
                linestyle='dashdot',
                linewidth=1.5,
            )
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
# Plot camberlines (NR x NR grid, axial_orc.py style)
fig_c, ax_c = plt.subplots(NR, NR, figsize=(4 * NR, 3 * NR), sharex=False, sharey=False)
fig_c.suptitle('Camberlines at midspan (3 blades per row)', fontsize=14)

for row, i in enumerate(MERID_INDICES):
    for col, j in enumerate(MERID_INDICES):
        ax = ax_c[row, col]
        key = (i, j)
        if key in meridional_data:
            d = meridional_data[key]
            for cl in d['camberlines']:
                # Midspan blades in black
                for blade_num in range(N_BLADES_PLOT):
                    _pbl.plot_camber_line(
                        ax,
                        cl['alpha_in'],
                        cl['alpha_out'],
                        cl['chord_ax'],
                        'k',
                        axial_offset=cl['x_offset'],
                        tangential_offset=blade_num * cl['pitch'],
                    )
                # Hub camberline (orange)
                _pbl.plot_camber_line(
                    ax,
                    cl['alpha_hub_in'],
                    cl['alpha_hub_out'],
                    cl['chord_ax'],
                    'orange',
                    axial_offset=cl['x_offset'],
                )
                # Tip camberline (seagreen)
                _pbl.plot_camber_line(
                    ax,
                    cl['alpha_tip_in'],
                    cl['alpha_tip_out'],
                    cl['chord_ax'],
                    'seagreen',
                    axial_offset=cl['x_offset'],
                )
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
# Reference design point plot (axial_orc.py style: meridional + camberlines side by side)
fig_ref, (ax_merid, ax_camber) = plt.subplots(1, 2, figsize=(14, 6))
fig_ref.suptitle(
    rf'Reference design: $\phi$={PHI_RANGE[0]:.2f}, $\psi_{{ts}}$={PSI_RANGE[0]:.1f}',
    fontsize=14,
)

ax_merid.set_title('Meridional profile', fontsize=14)
ax_merid.set_ylabel('Radius [m]', fontsize=14)
ax_merid.set_xlabel('Axial [m]', fontsize=14)
ax_merid.axis('equal')
ax_merid.grid(True, alpha=0.3)

ax_camber.set_title('Camberlines at midspan', fontsize=14)
ax_camber.set_ylabel('Tangential [m]', fontsize=14)
ax_camber.set_xlabel('Axial [m]', fontsize=14)
ax_camber.axis('equal')
ax_camber.grid(True, alpha=0.3)

# Write reference solution (0, 0) to nodes for live plotting
ntw.system.data.constraints_values[flow_coeff_idx] = np.array([PHI_RANGE[0]])
ntw.system.data.constraints_values[load_coeff_idx] = np.array([PSI_RANGE[0]])
ntw.system.write_solution_to_nodes(solution)

offset = 0.0
for comp in ntw.components:
    if not isinstance(comp, BladeRow):
        continue
    idx_map = comp.network_maps[ntw]
    inl_node = ntw.system.nodes[idx_map[0]]
    out_node = ntw.system.nodes[idx_map[1]]
    ax_chord = out_node.geo.chord_ax[0]

    is_stator = comp.row_type == 'stator'
    color = 'steelblue' if is_stator else 'coral'

    plot_from_nodes(inl_node, out_node, False, offset, ax=ax_merid, color=color)
    ax_merid.plot(NUM_SPAN * [offset], inl_node.geo.rr, 'o', color='r')
    ax_merid.plot(NUM_SPAN * [offset] + ax_chord, out_node.geo.rr, 'o', color='r')
    ax_merid.plot(
        NUM_SPAN * [offset], inl_node.geo.rr + inl_node.geo.hh / 2, '_', color='g'
    )
    ax_merid.plot(
        NUM_SPAN * [offset], inl_node.geo.rr - inl_node.geo.hh / 2, '_', color='b'
    )
    ax_merid.plot(
        NUM_SPAN * [offset] + ax_chord,
        out_node.geo.rr + out_node.geo.hh / 2,
        '_',
        color='g',
    )
    ax_merid.plot(
        NUM_SPAN * [offset] + ax_chord,
        out_node.geo.rr - out_node.geo.hh / 2,
        '_',
        color='b',
    )

    midspan_idx = ntw.system.num_span // 2
    inlet_angle = inl_node.geo.metal_angle[midspan_idx]
    outlet_angle = out_node.geo.metal_angle[midspan_idx]
    chord_ax = out_node.geo.chord_ax[midspan_idx]
    pitch = out_node.geo.pitch[midspan_idx]

    for blade_num in range(N_BLADES_PLOT):
        _pbl.plot_camber_line(
            ax_camber,
            inlet_angle,
            outlet_angle,
            chord_ax,
            'k',
            axial_offset=offset,
            tangential_offset=blade_num * pitch,
        )
    _pbl.plot_camber_line(
        ax_camber,
        inl_node.geo.metal_angle[0],
        out_node.geo.metal_angle[0],
        out_node.geo.chord_ax[0],
        'orange',
        axial_offset=offset,
    )
    _pbl.plot_camber_line(
        ax_camber,
        inl_node.geo.metal_angle[-1],
        out_node.geo.metal_angle[-1],
        out_node.geo.chord_ax[-1],
        'seagreen',
        axial_offset=offset,
    )

    offset += ax_chord * 1.1

ax_merid.plot([0.0, offset], [0.0, 0.0], color='r', linestyle='dashdot', linewidth=2.5)
fig_ref.tight_layout()

# ================================================
# Plot design map (phi on x-axis, psi on y-axis)
fig, ax = plt.subplots(figsize=(8, 6))

eta_masked = np.where(converged_map, eta_tt_map, np.nan)

# Transpose: eta_tt_map[i_phi, j_psi] -> contourf(phi, psi, map.T)
cf = ax.contourf(
    PHI_RANGE, PSI_RANGE, eta_masked.T, levels=15, cmap='viridis', vmin=0.70, vmax=0.91
)
cs = ax.contour(
    PHI_RANGE,
    PSI_RANGE,
    eta_masked.T,
    levels=30,
    colors='w',
    linewidths=0.5,
    alpha=0.6,
)
ax.clabel(cs, fmt='%.3f', fontsize=8)


cbar = fig.colorbar(cf, ax=ax)
cbar.set_label(r'Total-to-total efficiency $\eta_{tt}$ [-]', fontsize=14)

ax.set_xlabel(r'Flow coefficient $\phi$ [-]', fontsize=14)
ax.set_ylabel(r'Loading coefficient $\psi_{ts}$ [-]', fontsize=14)
ax.set_title('ORC Axial Turbine — Loss-Based Design Map', fontsize=15)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

fig.tight_layout()
fig.savefig(Path(__file__).parent / 'design_map_orc.png', dpi=150)

plt.show(block=False)
try:
    input('Press enter to close')
except EOFError:
    pass
plt.close('all')
