"""
Compute the design map of a repeated stage axial turbine for ORC
applications. Uses physics-based loss modeling.
"""

# === IMPORTS
import logging
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Literal, Type

import CoolProp as cp
import CoolProp

CoolProp.CoolProp.set_config_string(
    CoolProp.ALTERNATIVE_REFPROP_PATH, '/Users/matteomajer/Applications/REFPROP'
)
import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity

from adet.assemblers import CasadiSystem
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
    OptNumBlades,
    ParabolicCamberline,
)
from adet.equations.nondimensional import (
    FlowCoefficient,
    GammaPV,
    StaticTotalDegreeOfReaction,
    TotalStaticLoadingCoefficient,
    TotalTotalExpansionEfficiency,
    VolumetricFlowRatio,
    WorkCoefficient,
)
from adet.fluid.settings import FluidSettings
from adet.losses.basic import IsentropicLink, ZeroDeviation  # noqa: F401
from adet.losses.leakage import DentonTrapLeakage
from adet.losses.mixing import (
    DentonBaumgartnerMixingLoss,
    DentonMixingLoss,
    SieverdingBasePressure,
)
from adet.losses.profile import DentonTrapProfile
from adet.losses.secondary import SecondaryBSM
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.design_map import sweep_design_map
from adet.tools.loggers import setup_logger
from adet.tools.plotting import plot_design_map
from adet.variables import NodeVariables, ThermoVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)
n2 = NodeVariables(2)
n3 = NodeVariables(3)

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
INITIAL_LOSS = IsentropicLink()


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
        s0: n0.stc.Entropy.Hint,
        s1: n1.stc.Entropy.Hint,
        ds_mixing1: n1.loss.Ds_mixing.Hint,
        ds_profile1: n1.loss.Ds_profile.Hint,
        ds_secondary1: n1.loss.Ds_secondary.Hint,
        ds_leakage1: n1.loss.Ds_leakage.Hint,
        ds_main1: n1.loss.Ds_main.Hint,
    ):
        main_loss = ds_mixing1 + ds_profile1 + ds_secondary1
        leak_loss = ds_leakage1

        r1 = ds_main1 - main_loss

        if self._has_tip_gap:
            r2 = s1 - (s0 + main_loss + leak_loss)
        else:
            r2 = s1 - (s0 + main_loss)

        return r1, r2


def compute_design_map(
    ntw: ComponentNetwork[CasadiSystem],
    first_sol,
    n_points,
    starter_keys=None,
    starter_solutions=None,
    custom_bounds={},
):
    """Thin wrapper around the shared ``sweep_design_map`` driver, fixed to
    this file's (phi, psi) = (n3.ndim.FlowCoeff, n3.ndim.TSLoadCoeff) sweep
    keys, PHI_SPAN/PSI_SPAN grid, and EtaTT sanity check."""
    assert n_points == len(PHI_SPAN) == len(PSI_SPAN)
    return sweep_design_map(
        ntw,
        n3.ndim.FlowCoeff,
        n3.ndim.TSLoadCoeff,
        PHI_SPAN,
        PSI_SPAN,
        first_sol,
        starter_keys=starter_keys,
        starter_solutions=starter_solutions,
        custom_bounds=custom_bounds,
        validity_check=lambda sd: 0.5 <= sd[n3.ndim.EtaTT] <= 1.0,
    )


# ================================================

# This counts the number of updates in an attribute
abs_state = DebugAbstractState('REFPROP', 'MM')
abs_state.debug_print = False


# Stable solution
DUTY_COEFFS = {
    n1.ndim.FlowCoeff: 0.4,
    n1.ndim.TSLoadCoeff: 3.0,
    n1.ndim.VolflowRatio: round(float(VOL_FLOW), 1),
    n1.ndim.DegreeOfReactionTS: round(float(REACT_DEGREE), 1),
}

PHI_SPAN = np.linspace(0.4, 1.5, MAP_POINTS)
PSI_SPAN = np.linspace(3.0, 10.0, MAP_POINTS)

INLET_PRESSURE = 1.3 * abs_state.p_critical()
INLET_TEMPERATURE = 1.045 * abs_state.T_critical()
abs_state.update(cp.PT_INPUTS, INLET_PRESSURE, INLET_TEMPERATURE)

thrm = ThermoVariables()
fluid_settings = FluidSettings(
    fluid_state=abs_state,
    update_variables=(thrm.Pressure, thrm.Enthalpy),
    update_length=2,
)


# ================================================
# *** Bounds and Guesses
CUSTOM_BOUNDS = {
    n0.kin.V_mag.Glob: (20.0, 150.0),
    n0.kin.BladeSpeed.Glob: (-0.1, 300.0),
    n0.loss.Ds_mixing.Glob: (0.0, 50.0),
    n0.loss.Ds_profile.Glob: (0.0, 50.0),
    n0.loss.Ds_secondary.Glob: (0.0, 50.0),
    n0.loss.Ds_leakage.Glob: (0.0, 50.0),
    n0.loss.Ds_main.Glob: (0.0, 50.0),
}

# Add Thermodynamic bounds
if fluid_settings.fluid_state == abs_state:
    MAX_V = 250
    CUSTOM_BOUNDS.update(
        {
            n0.stc.Pressure.Glob: (
                abs_state.p_critical() * 0.5,
                1.5 * INLET_PRESSURE,
            ),
            n0.stc.Temperature.Glob: (
                abs_state.T_critical() * 0.5,
                1.5 * INLET_TEMPERATURE,
            ),
            n0.stc.Enthalpy.Glob: (
                abs_state.hmass() - MAX_V**2,
                abs_state.hmass() + MAX_V**2,
            ),
        }
    )

MANUAL_GUESSES = {
    # Geometry
    # Thermodynamic state
    n0.tot.Pressure.Glob: abs_state.p(),
    n0.tot.Temperature.Glob: abs_state.T(),
    n0.stc.Entropy.Glob: abs_state.smass(),
    n0.tot.Enthalpy.Glob: abs_state.hmass(),
    # Reaction degree
    n0.ndim.HdropCoeff: -0.8,
    n0.ndim.WorkCoeff: -0.8,
    n0.ndim.DegreeOfReactionTS.Glob: 0.5,
    n0.oth.ProfileLoading: 0.3,
    n0.geo.ZweifelCoeff.Glob: 0.85,
    n0.geo.NumBlades: 100,
}


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
    # GammaPV: 1,  # needed at node 1 for DentonBaumgartnerMixingLoss
    # DentonBaumgartnerMixingLoss: 1,
    DentonMixingLoss: 1,
    ClearanceByHeight: 1,
    BoundaryLayerRatios: 1,
    SieverdingBasePressure: (0, 1),
    ModifiedZweifel: (0, 1),
    OptNumBlades: 1,
}

# ================================================
# *** Inlet conditions
inlet = Inlet(
    boundary_conditions={
        # n0.oth.CumMassFlow: 100,
        n0.geo.Rmid: 0.1,
        n0.geo.MeridionalAngle: Quantity(0, 'deg'),
        # n0.geo.HubTipRatio: 0.9,
        n0.tot.Pressure: abs_state.p(),
        n0.tot.Enthalpy: abs_state.hmass(),
    }
)


stator = BladeRow(
    name='Stator',
    shaft=casing,
    bound_cond={
        n0.geo.ThickByPitch: 0.04,
        n1.geo.MeridionalAngle: Quantity(0, 'deg'),
        n1.geo.ThickByPitch: 0.02,
        n1.geo.ClearanceByHeight: 0.01,
        n1.geo.NumBlades: 30,
        n1.oth.MomByBld: 0.075,
        n1.oth.DispByMom: 2,
        n1.oth.DispByHgt: 0.05,
        n1.oth.CdProfile: 0.002,
        n1.oth.XiCambLenA: 0.375,
        n1.oth.XiCambLenB: 0.675,
        n1.oth.DischCoeff: 0.35,
        # n1.stc.Pressure: 1.462617e6, # Legacy ?
    },
    extra_equations={
        ZeroDeviation(): 0,  # No incidence (design)
        ZeroDeviation(): 1,  # No deviation
        # MinimalCamberLine(): (0, 1),
        ParabolicCamberline(): (0, 1),
        INITIAL_LOSS: (0, 1),
    },
)
stator.set_constants(n0.geo.Rmid.Glob)

# ============ Modify rotor
rotor = deepcopy(stator)  # Reuse the stator as template
rotor.shaft = shaft  # Assign the rotating shaft
rotor.name = 'rotor'
rotor.add_equation(WorkCoefficient(), (0, 1))

# *** Duty coefficients
rotor.set_boundary_cond(n1.geo.HubTipRatio, 0.818)
rotor.set_bc_from_dict(DUTY_COEFFS)  # Duty coefficients at node 1

# ================================================
# Create network
ntw = ComponentNetwork(
    fluid_settings,
    inlet,
    CasadiSystem(num_span=1),
    components=[stator, rotor],
)

rotor.set_spanwise_constant(n1.geo.MerChord)
stator.set_spanwise_constant(n1.geo.MerChord, n0.geo.HDistr, n0.kin.V_mer)
rotor.copy_from_previous(n0.geo.HDistr, n0.geo.RDistr)
rotor.remove_equation(MeridionalGeometry, 0)

# *** Flare angle hack ***
####
match CHORD_METHOD:
    case 'aspRatio':
        stator.set_boundary_cond(n1.geo.AspectRatio, ASP_RATIO)
        rotor.set_boundary_cond(n1.geo.AspectRatio, ASP_RATIO)
        file_identifier = f'aspRatio_{ASP_RATIO}'
    case 'flare_angle':
        stator.set_boundary_cond(n1.geo.FlareAngle, Quantity(FLARE_ANGLE, 'deg'))
        rotor.set_boundary_cond(n1.geo.FlareAngle, Quantity(FLARE_ANGLE, 'deg'))
        file_identifier = f'flare_angle_{FLARE_ANGLE}'
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
        file_identifier = f'dyn_fmax{FLARE_MAX}_ar{ASP_RATIO}'

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
ntw.build()

# ============ Isentropic Solution ============
rootfinder_is = ntw.system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        'ipopt.hessian_approximation': 'limited-memory',
    },
)

rtfn_kinsol = ntw.system.make_rootfinder('kinsol')

x0_is = ntw.system.get_guess(
    manual_values=MANUAL_GUESSES,
    fallback=0.5,
)
kn_is = ntw.system.get_boundary_conds()
custom_bounds_is = CUSTOM_BOUNDS.copy()
custom_bounds_is[n0.kin.FlowAngleAbs] = (-0.7, 0.7)
bnd_is = ntw.system.get_bounds(custom_bounds=custom_bounds_is)
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
sol_dict_is = ntw.system.sol_to_dict(solution)

# ========================== LOSSES
rotor.rm_boundary_cond(n1.geo.NumBlades)
rotor.set_boundary_cond(n1.geo.ZweifelCoeff, 0.85)
stator.rm_boundary_cond(n1.geo.NumBlades)
stator.set_boundary_cond(n1.geo.ZweifelCoeff, 0.85)

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

print('*** SOLVING WITH LOSSES ***')

x0_loss = ntw.system.get_guess(sol_dict_is, fallback=0.5)
kn_loss = ntw.system.get_boundary_conds()
bnd_loss = ntw.system.get_bounds(custom_bounds=CUSTOM_BOUNDS)

rootfinder_loss = ntw.system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': True,
        'ipopt.hessian_approximation': 'limited-memory',
    },
)
rtfn_kn = ntw.system.make_rootfinder('kinsol')

try:
    solution = solve_root_problem(rootfinder_loss, x0_loss, kn_loss, bnd_loss)
except RuntimeError:
    solution = solve_root_problem(rootfinder_loss, x0_loss, kn_loss)

sol_loss = solve_root_problem(rtfn_kn, solution, kn_loss)

sol_dict_loss = ntw.system.sol_to_dict(solution)

keys_loss, solutions_loss, solution_dicts = compute_design_map(
    ntw, sol_loss, MAP_POINTS, custom_bounds=CUSTOM_BOUNDS
)

# ========================== SAVE DESIGN MAP DATA
# Extract phi and psi ranges
phi_vals = keys_loss[:, 0]
psi_vals = keys_loss[:, 1]


# Bundle all data into a single pickle file
design_map_data = {
    'solution_dicts': solution_dicts,
    'keys_loss': keys_loss,
    'phi_vals': phi_vals,
    'psi_vals': psi_vals,
    'N_PTS': MAP_POINTS,
}

OUTPUT_DIR = Path(__file__).parents[3] / 'outputs' / 'design_maps'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

design_map_path = OUTPUT_DIR / f'design_map_{file_identifier}.pkl'
with open(design_map_path, 'wb') as f:
    pickle.dump(design_map_data, f)
logger.info(f'Saved design map data to {design_map_path}')

# ========================== POSTPROCESSING
with open(design_map_path, 'rb') as f:
    loaded_map = pickle.load(f)

fig, ax, _ = plot_design_map(
    loaded_map['keys_loss'],
    loaded_map['solution_dicts'],
    n3.ndim.EtaTT,
    loaded_map['N_PTS'],
    z_label=r'Total-total efficiency $\eta_{tt}$',
)
ax.set_title('ORC axial turbine design map')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / f'eta_tt_{file_identifier}.png', dpi=150)

plt.show()

vol_flow = DUTY_COEFFS[n1.ndim.VolflowRatio]
react = DUTY_COEFFS[n1.ndim.DegreeOfReactionTS]
