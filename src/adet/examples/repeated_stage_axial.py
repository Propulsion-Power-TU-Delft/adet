"""
Multistage Turbine Example
==========================
This example demonstrates a complete multistage turbomachinery analysis using ADeT.
It shows how to:
- Configure fluid models (real gas via CoolProp)
- Define blade rows (stators and rotors) with geometry and boundary conditions
- Build a ComponentNetwork with repeated stages
- Apply stage-level constraints (degree of reaction, repeated stage)
- Solve the system using CasADi with IPOPT
- Visualize results (velocity triangles and meridional profile)
"""

# === IMPORTS
# Standard library
from copy import deepcopy
import logging
from typing import Literal
from math import nan

# External libraries
import matplotlib.pyplot as plt
import jax
import numpy as np
import casadi as cs
from pint import Quantity

# Network build
from adet.assembly import CasadiSystem
from adet.components import ComponentNetwork, BladeRow, Shaft, Inlet
from adet.equations.definitions import DegreeOfReaction, RepeatedStage
from adet.fluid.settings import FluidSettings, ExternalFluidModel, IdealGasModel

# Losses
from adet.losses.basic import PercentageEntropyLoss
from adet.equations.nondimensional import FlowCoefficient, WorkCoefficient

# Tooling and utils
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.iter import grouper
from adet.tools.loggers import setup_logger
from adet.components.blade_row import plot_from_nodes
from adet.tools.context import suppress_output
from adet.registries import DefaultUnitsRegistry, ScalingRegistry, GuessRegistry


# === LOGGING SETUP
logger = logging.getLogger(__name__)
jax.config.update('jax_enable_x64', True)

setup_logger(
    logger,
    logging.INFO,
    logging.INFO,
    suppress_modules=['matplotlib', 'jax'],
    banned_keywords=['STREAM', 'findfont', 'sBIT'],
)
logging.getLogger('jax').setLevel(logging.WARNING)


# === CONFIGURATION
# Simulation settings
NUM_SPAN = 3  # Number of spanwise stations
NUM_STAGES = 4  # Number of turbine stages (stator-rotor pairs)
SCALED = True  # Use scaled equations for better numerical conditioning
PLOTS = True  # Show plots at end
PRINTS = True  # Print node information

# === FLUID MODEL SETUP
# Real gas model using CoolProp (HEOS = Helmholtz Equation of State)
abs_state = DebugAbstractState('HEOS', 'Air')
abs_state.debug_print = False
real_model = ExternalFluidModel(abs_state)
ideal_model = IdealGasModel()

# Configure fluid settings with update variables
# Update variables are used to solve for thermodynamic state
# (p, T) chosen for stability
settings = FluidSettings(
    model=real_model,
    update_variables=('p', 'T', 'rhomass', 'smass', 'hmass'),
    update_length=2,
)

# === UNIT REGISTRIES
# Register custom units for turbomachinery-specific variables
_dfu_reg = DefaultUnitsRegistry()
_scl_reg = ScalingRegistry()
_gss_reg = GuessRegistry()

_dfu_reg.from_dict(
    {
        'delta_smass_pct': 'J/(kg*K)',
        'deflection': 'rad',
        'percentage_loss': 'dimensionless',
        'workCoeff': 'dimensionless',
        'flowCoeff': 'dimensionless',
        'specificSpeed': 'dimensionless',
        'STratio': 'dimensionless',
        'VmRatio': 'dimensionless',
        'Vtmid': 'm/s',
        'Cd_profile': 'dimensionless',
        'sizeParameter': 'meters',
        'n_blades': 'dimensionless',
        'x_by_camb_len_A': 'meters',
        'x_by_camb_len_B': 'meters',
        'k_prof': '',
    }
)

_scl_reg.set_fallback_value(1.0)
_gss_reg.set_fallback_value(1.0)

# === SHAFT DEFINITIONS
# Static shaft for stators (no rotation)
static_shaft = Shaft(
    0.0,
    is_constrained=True,
)

# Rotating shaft for rotors (angular velocity to be determined)
rotating_shaft = Shaft(
    Quantity(nan, 'rpm'),
    is_constrained=False,
)

# === COMPONENT DEFINITIONS
# Inlet conditions
inlet = Inlet(
    {
        'kin': {
            'Vm': Quantity(75, 'm/s'),  # Meridional velocity
            'alpha': Quantity(0, 'deg'),  # Flow angle
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.5,  # Mid-span radius [m]
            'height': 0.2,  # Blade height [m]
        },
        'tot': {
            'T': 700,  # Total temperature [K]
            'p': 6e5,  # Total pressure [Pa]
        },
        'oth': {},
    }
)

# Stator blade row definition
row0 = BladeRow(
    'Stator0',
    {
        'kin': {
            'alpha': Quantity(70, 'deg'),  # Exit flow angle
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.5,
            'chord': 0.15,  # Blade chord length [m]
            'n_blades': 40,  # Number of blades
        },
        'tot': {},
        'oth': {},
    },
    shaft=static_shaft,
    extra_equations={
        # Simplified loss model (0% entropy rise)
        PercentageEntropyLoss(0.0): (0, 1),
    },
)

# Rotor blade row definition
row1 = BladeRow(
    'Rotor0',
    {
        'kin': {},
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.5,
            'chord': 0.15,
            'n_blades': 40,
        },
        'oth': {},
    },
    rotating_shaft,
    extra_equations={
        # Work and flow coefficients constrain rotor performance
        WorkCoefficient(): (0, 1),
        FlowCoefficient(): 1,
        # Simplified loss model
        PercentageEntropyLoss(0.0): (0, 1),
    },
)

# === NETWORK ASSEMBLY
# Replicate stage (stator-rotor pair) NUM_STAGES times
stage_obj = [row0, row1]
rows = list(map(deepcopy, NUM_STAGES * stage_obj))

# Create component network
ntw = ComponentNetwork(
    settings,  # Fluid settings
    inlet,  # Inlet conditions
    CasadiSystem(spanwise_stations=1),  # Backend (single spanwise station)
    *rows,
)

# === GLOBAL CONSTRAINTS
# Add ideal gas reference and loss model coefficients
ntw.system.add_global_constraints(
    {
        'oth': {
            # Ideal gas properties (reference)
            'cpmassid': 1004.0,  # Specific heat at constant pressure [J/kg/K]
            'cvmassid': 717.0,  # Specific heat at constant volume [J/kg/K]
            'T_ref': 1.0,
            'p_ref': 1.0,
            # Profile loss coefficients
            'Cd_profile': 0.002,
            'x_by_camb_len_A': 0.375,
            'x_by_camb_len_B': 0.675,
        }
    }
)

# === STAGE-LEVEL EQUATIONS
# Apply repeated stage and degree of reaction constraints
# Each stage consists of 4 nodes: stator_in, stator_out, rotor_in, rotor_out
nodes_by_stage = list(grouper(range(2 * ntw.num_components), 4, incomplete='strict'))
for stage in range(ntw.num_components // 2):
    nodes = nodes_by_stage[stage]
    # Repeated stage: inlet conditions same as outlet of previous stage
    # Constant meridional velocity
    ntw.system.add_equation(RepeatedStage(), nodes)
    # Degree of reaction constraint
    ntw.system.add_equation(DegreeOfReaction(), nodes)
    # Set 50% degree of reaction
    ntw.system.add_boundary_conditions({'oth': {'reactDegree': 0.5}}, nodes[-1])


# Build the system (assemble all equations and constraints)
ntw.system.build(SCALED)


# === SOLVER DEFINITION
def solve_casadi_sys(
    system: CasadiSystem,
    method: Literal['newton', 'nlpsol'],
    manual_guess={},
):
    """
    Solve the system of equations using CasADi rootfinder.

    Parameters
    ----------
    system : CasadiSystem
        The assembled system to solve
    method : {'newton', 'nlpsol'}
        Solver method ('newton' for Newton-Raphson, 'nlpsol' for IPOPT)
    manual_guess : dict
        Manual initial guess overrides

    Returns
    -------
    sol : casadi.DM
        Solution vector
    """
    x0 = system.get_initial_guess(manual_guess)
    knowns_stack = system.get_scaled_constraints()
    res_func_casadi = system.make_residual_function()
    free_args_symbols = cs.vertcat(*system.free_args_sym)

    res_expr_partial = res_func_casadi(
        free_args_symbols,  # Unknowns -> Symbols
        knowns_stack.flatten(),  # Knowns -> Numerical values
    )

    rootfind_problem = {
        'x': free_args_symbols,
        'g': res_expr_partial,
    }

    # Newton-Raphson solver -> Fast but unstable w/o good guess
    G_newt = cs.rootfinder(
        'newton_roots',
        'newton',
        rootfind_problem,
        {
            'print_iteration': False,
            'error_on_fail': True,
        },
    )

    # IPOPT solver (more robust)
    G_nlp = cs.rootfinder(
        'nlpsol_roots',
        'nlpsol',
        rootfind_problem,
        {
            'error_on_fail': True,
            'nlpsol': 'ipopt',
            'nlpsol_options': {
                'ipopt.print_level': 1,
                'ipopt.max_iter': 100,
                # Quasi-Newton approximation (required for real gas EOS)
                'ipopt.hessian_approximation': 'limited-memory',
            },
        },
    )

    with suppress_output():
        logger.info('Solving the system...')
        match method:
            case 'newton':
                sol = G_newt(x0.flatten(), 0.0)
            case 'nlpsol':
                sol = G_nlp(x0.flatten(), 0.0)

    return sol


# === SOLVE THE SYSTEM
sol = solve_casadi_sys(ntw.system, 'nlpsol')

# Convert solution to dictionary format
sol_dict = ntw.system.solution_to_dict(sol.toarray())

# Write solution back to FlowNodes
num_args = len(ntw.system.free_args)
ntw.system.write_solution_to_nodes(np.array(sol).reshape(num_args, -1))


# === POST-PROCESSING AND VISUALIZATION
if PLOTS:
    FONTSIZE = 18
    FONTDICT = {'fontsize': FONTSIZE}

    # Plot velocity triangles for each node
    for i, n in enumerate(ntw.system.nodes):
        n.kin.plot(n.geo, FONTSIZE)
        plt.title(f'Node number {i}')

    plt.tick_params(labelsize=FONTSIZE / 1.5 // 1)

    # Plot meridional profile (blade geometry)
    num_nodes = range(len(ntw.system.nodes))

    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.set_ylabel('radius [m]', {'fontsize': 18})
    ax.set_xlabel('axial  [m]', {'fontsize': 18})

    # Calculate y-axis limit based on last node geometry
    max_Y = (
        1.1
        * (
            ntw.system.nodes[-1].geo.get('rmid').magnitude
            + ntw.system.nodes[-1].geo.get('height').magnitude / 2
        )[0]
    )
    ax.set_ylim(-0.01, max_Y)
    ax.tick_params('both', labelsize=18)
    ax.grid()
    ax.set_title('Meridional profile', {'fontsize': 18})

    # Plot blade rows
    offset = 0.0
    for n0_idx, n1_idx in grouper(num_nodes, 2, incomplete='ignore'):
        n0 = ntw.system.nodes[n0_idx]
        n1 = ntw.system.nodes[n1_idx]
        ax_chord = n1.geo.get('chord').to_base_units().magnitude[0]
        lines = plot_from_nodes(
            n0,
            n1,
            ax_chord,
            False,
            offset,
        )
        offset += ax_chord * 1.15

    # Draw centerline
    ax.plot([0.0, offset], [0.0, 0.0], color='r', linestyle='dashdot', linewidth=2.5)


# === PRINT NODE INFORMATION
if PRINTS:
    for i, node in enumerate(ntw.system.nodes):
        # For simpler access set n0, n1, n2, ...
        globals()[f'n{i}'] = node
        to_print = f"""
##################
##### NODE {i} #####
##################
    {node}\n
    """
        print(to_print)

    ntw.print_structure()


# === DISPLAY PLOTS
if PLOTS:
    plt.show()
else:
    plt.close('all')
