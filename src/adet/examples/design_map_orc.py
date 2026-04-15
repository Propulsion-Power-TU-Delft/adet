# === IMPORTS
from copy import deepcopy
import logging
import pathlib
import pickle
import sys
from typing import Type

import CoolProp as cp
import matplotlib.pyplot as plt
import numpy as np
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
    MeridionalGeometry,
    MinimalCamberLine,
    ModifiedZweifel,
    MeridionalRatios,
    MeridionalHack,
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
from adet.losses.leakage import DentonRectLeakage, DentonTrapLeakage
from adet.losses.mixing import DentonMixingLoss, SieverdingBasePressure
from adet.losses.profile import DentonRectProfile, DentonTrapProfile
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
MULTI = False  # Whether to run multi or not
NUM_SPAN = 1
SCALED = True
PLOTS = True
PRINTS = True
INITIAL_LOSS = PercentageEntropyLoss(0.0)


# ================================================
def compute_design_map(
    ntw, first_sol, n_points, starter_keys=None, starter_solutions=None
):
    rtfn_kin = ntw.system.make_rootfinder('kinsol')

    keys = np.zeros((n_points**2, 2))

    solutions = np.zeros((n_points**2, len(first_sol)))
    solutions[0] = first_sol.flatten()

    # Store solution dicts for each point
    solution_dicts = []

    kn = ntw.system.get_scaled_constraints()

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
                    logger.warning('Solution cache miss, going to best next one')
                    x0 = solutions[idx]

            # Overwrite the knowns
            kn[phi_idx] = np.array([phi * ntw.system.constraints_scaling[phi_idx]])
            kn[psi_idx] = np.array([psi * ntw.system.constraints_scaling[psi_idx]])
            try:
                solution = solve_root_problem(rtfn_kin, x0, kn)
            except RuntimeError:
                solution = np.full(solution.shape, np.nan)

            keys[curr_index, :] = curr_key
            solutions[curr_index, :] = solution.flatten()

            # Store the full solution dict
            if not np.isnan(solution).any():
                sol_dict = ntw.system.write_solution_to_nodes(solution)
            else:
                sol_dict = None
            solution_dicts.append(sol_dict)

            curr_index += 1

    return keys, solutions, solution_dicts


# ================================================

# This counts the number of updates in an attribute
abs_state = DebugAbstractState('REFPROP', 'MM')
abs_state.debug_print = False

N_PTS = 40

# Stable solution
DUTY_COEFFS = {
    'oth_flowCoeff1': 0.4,
    'oth_ts_loadCoeff1': 3,
    'oth_volflowRatio1': 3.0,
    'oth_reactDegree_ts1': 0.3,
}

PHI_SPAN = np.linspace(0.4, 1.4, N_PTS)
PSI_SPAN = np.linspace(3.0, 10.0, N_PTS)

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
    DentonTrapLeakage: (0, 1),
    DentonTrapProfile: (0, 1),
    # DentonRectLeakage: (0, 1),
    # DentonRectProfile: (0, 1),
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
            'hubtipRatio': 0.9,
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

# stator.set_boundary_cond('geo_flare_angle1', Quantity(20, 'deg'))
# rotor.set_boundary_cond('geo_flare_angle1', Quantity(20, 'deg'))
# stator.set_boundary_cond('geo_aspRatio1', 2)
# rotor.set_boundary_cond('geo_aspRatio1', 2)

# *** Duty coefficients
rotor.bc_from_dict(DUTY_COEFFS)  # Duty coefficients at node 1

# Testing a hack
stator.remove_equation(MeridionalRatios, (0, 1))
# rotor.remove_equation(MeridionalRatios, (0, 1))

stator.add_equation(MeridionalHack(), (0, 1))

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
rotor.remove_equation(MeridionalGeometry, 0)

# Copy the stator flare to the rotor
ntw.system.add_equalities(
    ('geo_flare_angle1', 'geo_flare_angle3'),
)

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
# --- Remove the first computation loss
rotor.remove_equation(INITIAL_LOSS.__class__, (0, 1))
stator.remove_equation(INITIAL_LOSS.__class__, (0, 1))

# --- Add loss applier function
stator.add_equation(AddAxialLosses(tip_gap=False), (0, 1))
rotor.add_equation(AddAxialLosses(tip_gap=True), (0, 1))

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
if MULTI:
    print('********** RUN WITH 3 SPAN **********')
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

    # sol_multi = solve_root_problem(rtfn_multi_ip, x0, kn, bnd, suppress_output=False)
    sol_multi = solve_root_problem(rtfn_multi_kn, x0, kn, suppress_output=True)
    sol_dict_multi = ntw.system.write_solution_to_nodes(sol_multi)

    ntw.system.num_span = NUM_SPAN

    ntw.build()

    x0 = ntw.system.get_scaled_guess(sol_dict_multi)
    kn = ntw.system.get_scaled_constraints()

    rtfn_final = ntw.system.make_rootfinder('ipopt', opts=opts)
    sol_multi = solve_root_problem(rtfn_final, x0, kn, suppress_output=True)
    sol_dict_multi = ntw.system.write_solution_to_nodes(sol_multi)

    sol_final = sol_multi
else:
    sol_final = solution

keys_loss, solutions_loss, solution_dicts = compute_design_map(ntw, sol_final, N_PTS)

# ========================== SAVE DESIGN MAP DATA
# Extract eta_tt3 from solutions
eta_tt3_idx = ntw.system.free_args.index('oth_eta_tt3')
massflow_idx = ntw.system.free_args.index('oth_massflow3')

eta_tt3_pos = ntw.system.get_arg_position(eta_tt3_idx)
massflow_pos = ntw.system.get_arg_position(massflow_idx)

mf = solutions_loss[:, massflow_pos[0]] * ntw.system.free_args_scaling[massflow_idx]
eta_tt = solutions_loss[:, eta_tt3_pos[0]]
eta_tt3_values = eta_tt

# Extract phi and psi ranges
phi_vals = keys_loss[:, 0]
psi_vals = keys_loss[:, 1]

# Save complete design map data using pickle
data_dir = pathlib.Path(__file__).parent.parent.parent.parent / 'outputs'
data_dir.mkdir(parents=True, exist_ok=True)

# Bundle all data into a single pickle file
design_map_data = {
    'solution_dicts': solution_dicts,
    'keys_loss': keys_loss,
    'phi_vals': phi_vals,
    'psi_vals': psi_vals,
    'eta_tt3_values': eta_tt3_values,
    'massflow': mf,
    'N_PTS': N_PTS,
}

with open(data_dir / 'design_map_orc.pkl', 'wb') as f:
    pickle.dump(design_map_data, f)
logger.info(f'Design map data saved to {data_dir / "design_map_orc.pkl"}')
