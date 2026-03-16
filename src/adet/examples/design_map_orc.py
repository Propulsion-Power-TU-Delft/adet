# === IMPORTS
from copy import deepcopy
import logging
import sys
from typing import Type
import numpy as np

import CoolProp as cp
import matplotlib.pyplot as plt
from pint import Quantity

from adet.assembly import CasadiSystem
from adet.components import BladeRow, Inlet, Shaft
from adet.components import ComponentNetwork
from adet.equations.base_equation import EquationBase
from adet.equations.definitions import (
    BoundaryLayerRatios,
    ClearanceByHeight,
    IsentropicProperties,
    RepeatedStage,
)
from adet.equations.fundamental import (
    BladeBlockage,
    FreeVortexDistribution,
    ZeroBlockage,
)
from adet.equations.geometrical import (
    MeridionalVariable,
    MinimalCamberLine,
    ModifiedZweifel,
)
from adet.equations.nondimensional import (
    FlowCoefficient,
    StaticTotalDegreeOfReaction,
    TotalStaticLoadingCoefficient,
    TotalTotalExpansionEfficiency,
    VolumetricFlowRatio,
    WorkCoefficient,
)
from adet.examples.axial_orc import AddAxialLosses
from adet.fluid.settings import ExternalFluidModel
from adet.fluid.settings import FluidSettings
from adet.losses.basic import PercentageEntropyLoss, ZeroDeviation
from adet.losses.leakage import DentonRectLeakage
from adet.losses.mixing import (
    DentonMixingLoss,
    SieverdingBasePressure,
)
from adet.losses.profile import DentonRectProfile
from adet.losses.secondary import SecondaryBSM
from adet.registries import DefaultUnitsRegistry, GuessRegistry, VariableBoundsRegistry
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger

logger = logging.getLogger(__name__)

setup_logger(
    logger,
    logging.DEBUG,
    logging.INFO,
    banned_keywords=['STREAM', 'findfont', 'sBIT'],
)
plt.close('all')

# === SETTINGS
NUM_SPAN = 5
SCALED = True
PLOTS = True
PRINTS = True
INITIAL_LOSS = PercentageEntropyLoss(0.0)


# ================================================
def compute_design_map(
    ntw, first_sol, n_points, starter_keys=None, starter_solutions=None
):
    rtfn_kn = ntw.system.make_rootfinder(
        'kinsol', opts={'error_on_fail': False, 'max_iter': 2e9}
    )

    rtfn_ip = ntw.system.make_rootfinder(
        'ipopt',
        opts={
            'error_on_fail': False,
            'ipopt.max_wall_time': 6,
        },
    )
    keys = np.zeros((n_points**2, 2))

    solutions = np.zeros((n_points**2, len(first_sol)))
    solutions[0] = first_sol.flatten()

    curr_index = 0
    phi_idx = ntw.system.constraints.index('oth_flowCoeff3')
    psi_idx = ntw.system.constraints.index('oth_ts_loadCoeff3')
    for phi in PHI_SPAN:
        for psi in PSI_SPAN:
            curr_key = np.array([phi, psi])
            if starter_keys is not None and starter_solutions is not None:
                distances = np.linalg.norm(curr_key - starter_keys, axis=1, ord=2)
                idx = np.argmin(distances)
                x0 = ntw.system.get_scaled_guess(starter_solutions[idx])
            else:
                distances = np.linalg.norm(curr_key - keys, axis=1, ord=np.inf)
                idx = np.argmin(distances)
                x0 = solutions[idx]
                while np.isnan(x0).any():
                    idx -= 1
                    print('Walking back!')
                    x0 = solutions[idx]

            # Overwrite the knowns
            kn[phi_idx] = np.array([phi * ntw.system.constraints_scaling[phi_idx]])
            kn[psi_idx] = np.array([psi * ntw.system.constraints_scaling[psi_idx]])
            # try:
            solution = solve_root_problem(rtfn_kn, x0, kn)
            # except RuntimeError:
            #     try:
            #         solution = solve_root_problem(rtfn_ip, x0, kn)
            #         solution = solve_root_problem(rtfn_kn, solution, kn)
            #     except RuntimeError:
            #         solution = np.full(solution.shape, np.nan)

            keys[curr_index, :] = curr_key
            solutions[curr_index, :] = solution.flatten()

            curr_index += 1

    return keys, solutions


# ================================================

# This counts the number of updates in an attribute
abs_state = DebugAbstractState('REFPROP', 'MM')
abs_state.debug_print = False

N_PTS = 20

# Stable solution
DUTY_COEFFS = {
    'oth_flowCoeff1': 0.4,
    'oth_ts_loadCoeff1': 3.0,
    'oth_volflowRatio1': 4,
    'oth_reactDegree_ts1': 0.3,
}

PHI_SPAN = np.linspace(DUTY_COEFFS['oth_flowCoeff1'], 1.4, N_PTS)
PSI_SPAN = np.linspace(DUTY_COEFFS['oth_ts_loadCoeff1'], 10.0, N_PTS)

real_model = ExternalFluidModel(abs_state)
INLET_PRESSURE = 1.3 * abs_state.p_critical()
INLET_TEMPERATURE = 1.045 * abs_state.T_critical()
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
        'p_choke': 0.4 * abs_state.p(),
        'reactDegree_ts': 0.5,
        'p': abs_state.p(),
        'T': abs_state.T(),
        'hmass': abs_state.hmass(),
        'smass': abs_state.smass(),
        'rhomass': abs_state.rhomass(),
        'k_prof': 0.3,  # Profile loading
        'zweifelCoeff': 0.85,
        'num_blades': 20.0,
    }
)

# ================================================
# *** Variable bounds
_bounds_reg = VariableBoundsRegistry()
_bounds_reg.reset()
# _bounds_reg.ignore_defaults = True
_bounds_reg.from_dict(
    {
        'U': (-0.1, 200.0),  # Reduce the search area
        'Vm': (20.0, 150.0),  # Reduce the search area
        # 'num_blades': (1.0, 100.0),
        'delta_smass_.*': (0.0, 20.0),
    }
)
if fluid_settings.model == real_model:
    _bounds_reg.from_dict(
        {
            'p': (abs_state.p_critical() * 0.5, 1.5 * INLET_PRESSURE),
            'T': (abs_state.T_critical() * 0.5, 1.5 * INLET_TEMPERATURE),
            'hmass': (abs_state.hmass() - 200**2, abs_state.hmass() + 200**2),
        }
    )

# ================================================
# *** Shafts
casing = Shaft(0.0, is_constrained=True)
shaft = Shaft(-1, is_constrained=False)


LOSS_MODELS: dict[
    Type[EquationBase],
    int | tuple[int, ...],
] = {
    # *** Blade row losses
    IsentropicProperties: (0, 1),
    SecondaryBSM: (0, 1),
    DentonRectProfile: (0, 1),
    DentonRectLeakage: (0, 1),
    DentonMixingLoss: 1,
    ClearanceByHeight: 1,
    BladeBlockage: 1,
    BoundaryLayerRatios: 1,
    SieverdingBasePressure: (0, 1),
    ModifiedZweifel: (0, 1),
}

# ================================================
# *** Inlet conditions
inlet = Inlet(
    {
        'oth': {
            'cum_massflow': 10,
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'hubtipRatio': 0.82,
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
    in_constraints={
        'geo': {
            'thick_by_pitch': 0.04,
        },
    },
    out_constraints={
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'thick_by_pitch': 0.02,
            'clearance_by_height': 0.01,
            # *** Num blades
            'num_blades': 20,
        },
        'oth': {  # NOTE: These are not used on first pass
            # *** Boundary layer ratios
            'mom_by_bld': 0.075,
            'disp_by_mom': 2,
            'disp_by_hgt': 0.05,  # endwall
            # *** Profile loss coeff
            'Cd_profile': 0.002,
            'xi_by_camb_len_A': 0.375,
            'xi_by_camb_len_B': 0.675,
            # *** Tip leakage discharge coeff
            'dischCoeff': 0.35,
        },
    },
    extra_equations={
        ZeroDeviation(): 0,  # No incidence (design)
        ZeroDeviation(): 1,  # No deviation
        MinimalCamberLine(): (0, 1),
        INITIAL_LOSS: (0, 1),
    },
    constant_variables=['geo_rr_midspan'],
)

# ============ Modify rows
rotor = deepcopy(stator)  # Reuse the stator as template
rotor.shaft = shaft  # Assign the rotating shaft
rotor.name = 'rotor'  # Not strictly required
rotor.row_type = 'rotor'  # Set the type
rotor.add_equation(WorkCoefficient(), (0, 1))
rotor.set_boundary_cond('geo_aspRatio1', 3.0)
stator.set_boundary_cond('geo_flare_angle1', Quantity(30, 'deg'))
# *** Duty coefficients
rotor.bc_from_dict(DUTY_COEFFS)  # Duty coefficients at node 1

# ================================================
# Create network
ntw = ComponentNetwork(
    fluid_settings,
    inlet,
    CasadiSystem(num_span=1),
    components=[stator, rotor],
)

rotor.set_spanwise_constant('geo_chord_ax1')
stator.set_spanwise_constant('geo_chord_ax1', 'geo_hh0', 'kin_Vm0')
rotor.copy_from_previous('geo_hh', 'geo_rr')
rotor.remove_equation(MeridionalVariable, 0)

# Repeated stage definition
ntw.system.add_equation(RepeatedStage(), (0, 1, 2, 3))
ntw.system.add_equation(StaticTotalDegreeOfReaction(), (0, 1, 2, 3))
# Inlet-to-outlet equations
ntw.system.add_equation(FlowCoefficient(), (0, 3))
ntw.system.add_equation(VolumetricFlowRatio(), (0, 3))
ntw.system.add_equation(TotalStaticLoadingCoefficient(), (0, 3))
ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, 3))

# Build
ntw.system.build(SCALED)

# ============ Isentropic Solution ============
rootfinder_is = ntw.system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        'ipopt.hessian_approximation': 'limited-memory',
    },
)

rtfn_kinsol = ntw.system.make_rootfinder('kinsol')

x0_is = ntw.system.get_scaled_guess()
kn_is = ntw.system.get_scaled_constraints()
bnd_is = ntw.system.get_arguments_bounds({'kin_alpha0': (-0.7, 0.7)})
solution = solve_root_problem(
    rootfinder_is,
    x0_is,
    kn_is,
    bnd_is,
    suppress_output=False,
)
solution = solve_root_problem(rtfn_kinsol, solution, kn_is)

stator_is_equations = stator._equations.copy()
rotor_is_equations = rotor._equations.copy()

# Write solution to dict for reading for next solution
sol_dict_is = ntw.system.write_solution_to_nodes(solution)

# ========================== LOSSES
rotor.rm_boundary_cond('geo_num_blades1')
rotor.set_boundary_cond('geo_zweifelCoeff1', 0.85)
stator.rm_boundary_cond('geo_num_blades1')
stator.set_boundary_cond('geo_zweifelCoeff1', 0.85)

# --- Remove zero blockage
rotor.remove_equation(ZeroBlockage, 1)
stator.remove_equation(ZeroBlockage, 1)

for eq, pos in LOSS_MODELS.items():
    stator.add_equation(eq(), pos)
    rotor.add_equation(eq(), pos)

ntw.build()

x0_loss = ntw.system.get_scaled_guess(sol_dict_is)
kn_loss = ntw.system.get_scaled_constraints()
bnd_loss = ntw.system.get_arguments_bounds()

rootfinder_loss = ntw.system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': True,
        'ipopt.hessian_approximation': 'limited-memory',
    },
)
rtfn_kn = ntw.system.make_rootfinder('kinsol')

solution = solve_root_problem(
    rootfinder_loss,
    x0_loss,
    kn_loss,
    suppress_output=False,
    perturbate_guess=False,
)
solution = solve_root_problem(rtfn_kn, solution, kn_loss)

sol_dict_loss = ntw.system.write_solution_to_nodes(solution)

# === MULTI SPAN
ntw.system.num_span = 3
if NUM_SPAN > 1:
    rotor.add_equation(FreeVortexDistribution(), 1)
    stator.add_equation(FreeVortexDistribution(), 1)

ntw.build()

opts = {
    'error_on_fail': False,
    'ipopt.hessian_approximation': 'limited-memory',
}

rtfn_multi_ip = ntw.system.make_rootfinder('ipopt', opts=opts)
rtfn_multi_kn = ntw.system.make_rootfinder('kinsol')

x0 = ntw.system.get_scaled_guess(sol_dict_loss)
kn = ntw.system.get_scaled_constraints()
bnd = ntw.system.get_arguments_bounds()

sol_multi = solve_root_problem(rtfn_multi_ip, x0, kn)
sol_multi = solve_root_problem(rtfn_multi_kn, sol_multi, kn, suppress_output=True)
sol_dict_multi = ntw.system.write_solution_to_nodes(sol_multi)

# Make 3 span
ntw.system.num_span = NUM_SPAN
# --- Remove the first computation loss
rotor.remove_equation(INITIAL_LOSS.__class__, (0, 1))
stator.remove_equation(INITIAL_LOSS.__class__, (0, 1))

# --- Add loss applier function
stator.add_equation(AddAxialLosses(tip_gap=False), (0, 1))
rotor.add_equation(AddAxialLosses(tip_gap=True), (0, 1))
ntw.build()

x0 = ntw.system.get_scaled_guess(sol_dict_multi)
kn = ntw.system.get_scaled_constraints()

rtfn_final = ntw.system.make_rootfinder('ipopt', opts=opts)
sol_multi = solve_root_problem(rtfn_final, x0, kn, suppress_output=True)
sol_dict_multi = ntw.system.write_solution_to_nodes(sol_multi)

keys_loss, solutions_loss = compute_design_map(ntw, sol_multi, N_PTS)

# ========================== DESIGN MAP CONTOUR PLOT
# Extract eta_tt3 from solutions
eta_tt3_idx = ntw.system.free_args.index('oth_eta_tt3')
eta_tt3_position = ntw.system.get_arg_position(eta_tt3_idx)

eta_tt3_values = np.mean(
    solutions_loss[:, eta_tt3_position[0] : eta_tt3_position[1]], axis=1
)

# Extract phi and psi ranges
phi_vals = keys_loss[:, 0]
psi_vals = keys_loss[:, 1]

# Reshape into grids (N_PTS x N_PTS)
eta_tt3_grid = eta_tt3_values.reshape((N_PTS, N_PTS))
phi_grid = phi_vals.reshape((N_PTS, N_PTS))
psi_grid = psi_vals.reshape((N_PTS, N_PTS))

# Create contour plot
fig, ax = plt.subplots(figsize=(10, 8))
LEVELS = 20
cs = ax.contourf(phi_grid, psi_grid, eta_tt3_grid, levels=LEVELS, cmap='viridis')
ax.contour(
    phi_grid,
    psi_grid,
    eta_tt3_grid,
    levels=LEVELS,
    colors='black',
    alpha=0.3,
    linewidths=0.2,
)
cbar = fig.colorbar(cs, ax=ax)
cbar.set_label(r'$\eta_{tt}$ [-]')
ax.set_xlabel(r'Flow Coefficient $\phi$ [-]')
ax.set_ylabel(r'Loading Coefficient $\psi$ [-]')
ax.set_title('Design Map: Total-Total Efficiency')
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()
