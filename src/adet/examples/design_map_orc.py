# === IMPORTS
import logging
import pathlib
import pickle
from copy import deepcopy
from typing import Literal, Type

import CoolProp as cp
import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity

from adet.assembly import CasadiSystem
from adet.components import BladeRow, ComponentNetwork, Inlet, Shaft
from adet.equations.base_equation import EquationBase, LossApplier
from adet.equations.definitions import (
    BoundaryLayerRatios,
    ClearanceByHeight,
    IsentropicProperties,
    RepeatedStage,
)
from adet.equations.geometrical import (
    FlareAngleLimitedAR,
    MeridionalGeometry,
    MeridionalRatios,
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
from adet.losses.leakage import DentonTrapLeakage
from adet.losses.mixing import DentonMixingLoss, SieverdingBasePressure
from adet.losses.profile import DentonTrapProfile
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
NUM_SPAN = 1
MAP_POINTS = 50  # Grid is the square of this

# Volumetric flow ratio
REACT_DEGREE = 0.3
VOL_FLOW = 4.0
FLARE_ANGLE = 30  # deg
ASP_RATIO = 3.0

# Axial chord specification
CHORD_METHOD: Literal['flare_angle', 'aspRatio', 'dynamic']
CHORD_METHOD = 'dynamic'
FLARE_MAX = 30  # deg - ONLY USED IN DYNAMIC !

# Loss used at first pass (isentropic)
INITIAL_LOSS = PercentageEntropyLoss(0.0)


# ================================================
class AddAxialLosses(LossApplier):
    scaling_factor = (0.1, 0.1)

    def __init__(
        self,
        has_tip_gap: bool,
        scaling_factor: list[float] | None = None,
    ):
        super().__init__(scaling_factor)
        self._has_tip_gap = has_tip_gap

    def residual(
        self,
        stc_smass0,
        stc_smass1,
        oth_delta_smass_mixing1,
        oth_delta_smass_profile1,
        oth_delta_smass_secondary1,
        oth_delta_smass_leakage1,
        oth_delta_smass_main1,
    ):
        main_loss = (
            0.0
            + oth_delta_smass_mixing1
            + oth_delta_smass_profile1
            + oth_delta_smass_secondary1
        )

        leak_loss = oth_delta_smass_leakage1

        r1 = oth_delta_smass_main1 - main_loss

        if self._has_tip_gap:
            r3 = stc_smass1 - (stc_smass0 + main_loss + leak_loss)
        else:
            r3 = stc_smass1 - (stc_smass0 + main_loss)

        return r1, r3


def compute_design_map(
    ntw, first_sol, n_points, starter_keys=None, starter_solutions=None
):
    rtfn_kin = ntw.system.make_rootfinder('kinsol')
    rtfn_ip = ntw.system.make_rootfinder(
        'ipopt',
        opts={
            'error_on_fail': True,
            'ipopt.max_wall_time': 1.0,
        },
    )

    keys = np.zeros((n_points**2, 2))

    solutions = np.zeros((n_points**2, len(first_sol)))
    solutions[0] = first_sol.flatten()

    # Store solution dicts for each point
    solution_dicts = []

    kn = ntw.system.get_scaled_constraints()
    bnd = ntw.system.get_arguments_bounds()

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
            solution = x0
            try:
                solution = solve_root_problem(rtfn_kin, x0, kn, suppress_output=True)
            except RuntimeError:
                # Try bounded and unbounded ipopt
                logger.info('KINSOL failed, trying IPOPT...')
                try:
                    solution = solve_root_problem(
                        rtfn_ip, x0, kn, bnd, suppress_output=True
                    )
                except RuntimeError:
                    try:
                        solution = solve_root_problem(
                            rtfn_ip, x0, kn, suppress_output=True
                        )
                    except RuntimeError:
                        logger.info('IPOPT failure, default to closest solution')
                        # Just re-use previous solution
                        # => no cache misses
                        solution = x0

                # solution = np.full(solution.shape, np.nan)

            keys[curr_index, :] = curr_key
            solutions[curr_index, :] = solution.flatten()

            # Store the full solution dict
            if not np.isnan(solution).any():
                sol_dict = ntw.system.write_solution_to_nodes(solution)
                if sol_dict['oth_eta_tt3'] > 1.0 or sol_dict['oth_eta_tt3'] < 0.5:
                    sol_dict = None
                    logger.warning(f'Failed point at phi={phi:.2f}, psi={psi:.2f}')
            else:
                sol_dict = None
                logger.warning(f'Failed point at phi={phi:.2f}, psi={psi:.2f}')

            solution_dicts.append(sol_dict)

            curr_index += 1

    return keys, solutions, solution_dicts


# ================================================

# This counts the number of updates in an attribute
abs_state = DebugAbstractState('REFPROP', 'MM')
abs_state.debug_print = False


# Stable solution
DUTY_COEFFS = {
    'oth_flowCoeff1': 0.4,
    'oth_ts_loadCoeff1': 3.0,
    'oth_volflowRatio1': round(float(VOL_FLOW), 1),
    'oth_reactDegree_ts1': round(float(REACT_DEGREE), 1),
}

PHI_SPAN = np.linspace(0.4, 1.5, MAP_POINTS)
PSI_SPAN = np.linspace(3.0, 10.0, MAP_POINTS)

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
        'num_blades': 100.0,
    }
)

# ================================================
# *** Variable BOUNDS
_bounds_reg = VariableBoundsRegistry()
_bounds_reg.reset()
# _bounds_reg.ignore_defaults = True
_bounds_reg.from_dict(
    {
        'U': (-0.1, 300.0),  # Reduce the search area
        'V': (20.0, 150.0),  # Reduce the search area
        # 'num_blades': (1.0, 100.0),
        'delta_smass_.*': (0.0, 50.0),
        # 'k_prof': (-3.5, 3.5),
    }
)
MAX_V = 250
if fluid_settings.model == real_model:
    _bounds_reg.from_dict(
        {
            'p': (
                abs_state.p_critical() * 0.5,
                1.5 * INLET_PRESSURE,
            ),
            'T': (
                abs_state.T_critical() * 0.5,
                1.5 * INLET_TEMPERATURE,
            ),
            'hmass': (
                abs_state.hmass() - MAX_V**2,
                abs_state.hmass() + MAX_V**2,
            ),
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
    BoundaryLayerRatios: 1,
    SieverdingBasePressure: (0, 1),
    ModifiedZweifel: (0, 1),
}

# ================================================
# *** Inlet conditions
inlet = Inlet(
    {
        'oth': {
            # 'cum_massflow': 100,
        },
        'geo': {
            'rr_midspan': 0.1,
            'meridional_angle': Quantity(0, 'deg'),
            # 'hubtipRatio': 0.9,
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
        # # This was coded in TurboSim
        # 'stc': {
        #     'p': 1.462617e6,
        # },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'thick_by_pitch': 0.02,
            'clearance_by_height': 0.01,
            # *** Num blades
            'num_blades': 30,
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
        # MinimalCamberLine(): (0, 1),
        ParabolicCamberline(): (0, 1),
        INITIAL_LOSS: (0, 1),
    },
    constant_variables=['geo_rr_midspan'],
)

# ============ Modify rotor
rotor = deepcopy(stator)  # Reuse the stator as template
rotor.shaft = shaft  # Assign the rotating shaft
rotor.name = 'rotor'
rotor.row_type = 'rotor'  # Set the type (useless now)
rotor.add_equation(WorkCoefficient(), (0, 1))

if 'stc' in stator._boundary_conditions:
    rotor.rm_boundary_cond('stc_p1')


# *** Duty coefficients
rotor.bc_from_dict({'geo_hubtipRatio1': 0.818})
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
rotor.remove_equation(MeridionalGeometry, 0)

# *** Flare angle hack ***
####
match CHORD_METHOD:
    case 'aspRatio':
        stator.set_boundary_cond('geo_aspRatio1', ASP_RATIO)
        rotor.set_boundary_cond('geo_aspRatio1', ASP_RATIO)
        identifier = f'aspRatio_{ASP_RATIO}'
    case 'flare_angle':
        stator.set_boundary_cond('geo_flare_angle1', Quantity(FLARE_ANGLE, 'deg'))
        rotor.set_boundary_cond('geo_flare_angle1', Quantity(FLARE_ANGLE, 'deg'))
        identifier = f'flare_angle_{FLARE_ANGLE}'
    case 'dynamic':
        flare_max_rad = FLARE_MAX * np.pi / 180
        stator.remove_equation(MeridionalRatios, (0, 1))
        rotor.remove_equation(MeridionalRatios, (0, 1))
        stator.add_equation(
            FlareAngleLimitedAR(ASP_RATIO, flare_max_rad),
            (0, 1),
        )
        rotor.add_equation(
            FlareAngleLimitedAR(ASP_RATIO, flare_max_rad),
            (0, 1),
        )
        identifier = f'dyn_fmax{FLARE_MAX}_ar{ASP_RATIO}'

# OPTIONAL: Force constant flare angle
# ntw.system.add_equalities(('geo_flare_angle1', 'geo_flare_angle3'))

# Repeated stage definition
ntw.system.add_equation(RepeatedStage(), (0, 1, 2, 3))
ntw.system.add_equation(StaticTotalDegreeOfReaction(), (0, 1, 2, 3))
# Inlet-to-outlet equations
ntw.system.add_equation(FlowCoefficient(), (0, 3))
ntw.system.add_equation(VolumetricFlowRatio(), (0, 3))
ntw.system.add_equation(TotalStaticLoadingCoefficient(), (0, 3))
ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, 3))

# Build
ntw.system.build(True)

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

for eq, pos in LOSS_MODELS.items():
    stator.add_equation(eq(), pos)
    rotor.add_equation(eq(), pos)
# --- Remove the first computation loss
rotor.remove_equation(INITIAL_LOSS.__class__, (0, 1))
stator.remove_equation(INITIAL_LOSS.__class__, (0, 1))

# --- Add loss applier function
stator.add_equation(AddAxialLosses(has_tip_gap=False), (0, 1))
rotor.add_equation(AddAxialLosses(has_tip_gap=True), (0, 1))

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

try:
    solution = solve_root_problem(rootfinder_loss, x0_loss, kn_loss)
except RuntimeError:
    solution = solve_root_problem(rootfinder_loss, x0_loss, kn_loss, bnd_loss)

sol_loss = solve_root_problem(rtfn_kn, solution, kn_loss)

sol_dict_loss = ntw.system.write_solution_to_nodes(solution)

keys_loss, solutions_loss, solution_dicts = compute_design_map(
    ntw, sol_loss, MAP_POINTS
)

# ========================== SAVE DESIGN MAP DATA
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
    'N_PTS': MAP_POINTS,
}

vol_flow = DUTY_COEFFS['oth_volflowRatio1']
react = DUTY_COEFFS['oth_reactDegree_ts1']

filename = f'des_map_R{react}_vr{vol_flow}_{identifier}.pkl'
with open(data_dir / filename, 'wb') as f:
    pickle.dump(design_map_data, f)

logger.info(f'Design map data saved to {data_dir / filename}')
