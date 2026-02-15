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

import matplotlib.pyplot as plt
from pint import Quantity

from adet.assembly import CasadiSystem
from adet.components import BladeRow, ComponentNetwork, Inlet, Shaft
from adet.components.blade_row import plot_from_nodes
from adet.equations.base_equation import EquationBase
from adet.equations.definitions import RepeatedStage
from adet.equations.fundamental import FreeVortexDistribution
from adet.equations.geometrical import MinimalCamberLine, ParabolicCamberline
from adet.equations.nondimensional import (
    FlowCoefficient,
    TotalStaticDegreeOfReaction,
    WorkCoefficient,
)
from adet.fluid.settings import AnalyticalFluidModel, ExternalFluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.losses.base_loss import LossModel
from adet.losses.basic import PercentageEntropyLoss, ZeroDeviation
from adet.losses.profile import DentonProfileLoss
from adet.registries import (
    DefaultUnitsRegistry,
    GuessRegistry,
    ScalingRegistry,
    VariableBoundsRegistry,
)
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.iter import grouper
from adet.tools.loggers import setup_logger


# === LOGGING SETUP
logger = logging.getLogger(__name__)

setup_logger(
    logger,
    logging.INFO,
    logging.INFO,
    suppress_modules=['matplotlib', 'jax'],
    banned_keywords=['STREAM', 'findfont', 'sBIT'],
)
plt.close('all')
logging.getLogger('jax').setLevel(logging.WARNING)


# === CONFIGURATION
# Simulation settings
NUM_SPAN = 3  # Number of spanwise stations
NUM_STAGES = 2  # Number of turbine stages (stator-rotor pairs)
# Runtime options
RUN_MULTI = False  # Run the multi streamline case
SCALED = True  # Use scaled equations for better numerical conditioning
PLOTS = True  # Show plots at end
PRINTS = False  # Print node information

# === FLUID MODEL SETUP
# Real gas model using CoolProp (HEOS = Helmholtz Equation of State)
abs_state = DebugAbstractState('HEOS', 'Air')
idl_state = IdealGasState(1.4, 287.0, 3e-5)
abs_state.debug_print = False

real_model = ExternalFluidModel(abs_state)
ideal_model = AnalyticalFluidModel(idl_state)

# Configure fluid settings with update variables
# Update variables are used to solve for thermodynamic state
# (p, T) chosen for stability
settings = FluidSettings(
    model=ideal_model,
    update_variables=('p', 'T'),
    update_length=2,
)

# === UNIT REGISTRIES
# Register custom units for turbomachinery-specific variables
_dfu_reg = DefaultUnitsRegistry()
_scl_reg = ScalingRegistry()
_gss_reg = GuessRegistry()

_dfu_reg.from_dict(
    {
        'k_prof': 'dimensionless',
        'xi_by_camb_len_.*': 'dimensionless',
        'Cd_profile': 'dimensionless',
    }
)

_gss_reg.reset()
_gss_reg.from_dict(
    {
        'workCoeff': -0.9,
        'k_prof': 0.2,
        'hdropCoeff': -2.0,
    }
)

_bnd_reg = VariableBoundsRegistry()
_bnd_reg.reset()
_bnd_reg.from_dict(
    {
        'U': (0, 500),
        'hdropCoeff': (-6.0, -0.1),
    }
)

# === SHAFT DEFINITIONS
# Static shaft for stators (no rotation)
casing = Shaft(
    0.0,
    is_constrained=True,
)

# Rotating shaft for rotors (angular velocity to be determined)
rotating_shaft = Shaft(
    Quantity(-1, 'rpm'),
    is_constrained=False,
)

#  =  =  =  =  =  =  =  =  =  =  =  =  =  =  COMPONENT DEFINITIONS
# Inlet conditions
inlet = Inlet(
    {
        'kin': {
            'alpha': Quantity(0, 'deg'),  # Flow angle
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'hubtipRatio': 0.7,
        },
        'tot': {
            'T': 700,  # Total temperature [K]
            'p': 10e5,  # Total pressure [Pa]
        },
    }
)

# Stator blade row definition
stator = BladeRow(
    'Stator',
    shaft=casing,
    row_type='stator',
    in_constraints={
        'geo': {
            'thick_by_pitch': 0.04,
        },
    },
    out_constraints={
        'kin': {
            # 'alpha': Quantity(70, 'deg'),  # Exit flow angle
        },
        'geo': {
            'aspRatio': 2.0,
            'num_blades': 20,
            'thick_by_pitch': 0.02,
            'meridional_angle': Quantity(0, 'deg'),
        },
    },
    extra_equations={
        PercentageEntropyLoss(0.0): (0, 1),
        MinimalCamberLine(): (0, 1),
        ZeroDeviation(): 0,
        ZeroDeviation(): 1,
    },
)

# Rotor blade row definition
rotor = BladeRow(
    'Rotor',
    shaft=rotating_shaft,
    row_type='rotor',
    in_constraints={
        'geo': {
            'thick_by_pitch': 0.04,
        },
    },
    out_constraints={
        'geo': {
            'aspRatio': 2.0,
            'num_blades': 20,
            'thick_by_pitch': 0.02,
            'meridional_angle': Quantity(0, 'deg'),
        },
    },
    extra_equations={
        PercentageEntropyLoss(0.0): (0, 1),
        MinimalCamberLine(): (0, 1),
        ZeroDeviation(): 0,
        ZeroDeviation(): 1,
        # Work and flow coefficients defined only on rotor
        WorkCoefficient(): (0, 1),
        FlowCoefficient(): (0, 1),
    },
)

# === NETWORK ASSEMBLY
# Replicate stage (stator-rotor pair) NUM_STAGES times
stage_obj = [stator, rotor]
rows = list(map(deepcopy, NUM_STAGES * stage_obj))

# Create component network
ntw = ComponentNetwork(
    settings,  # Fluid settings
    inlet,  # Inlet conditions
    CasadiSystem(num_span=1),  # Backend (single spanwise station)
    rows,
)

# === GLOBAL CONSTRAINTS
# Add ideal gas reference and loss model coefficients
ntw.system.add_global_constraints(
    {
        'oth': {
            # Profile loss parameters
            'Cd_profile': 0.002,
            'xi_by_camb_len_A': 0.375,
            'xi_by_camb_len_B': 0.675,
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
    ntw.system.add_equation(TotalStaticDegreeOfReaction(), nodes)
    # Set nondimensional design parameters
    ntw.system.add_boundary_conditions(
        {
            'oth': {
                'reactDegree_ts': 0.5,
                'workCoeff': -1.0,
            },
        },
        nodes[3],
    )

ntw.system.boundary_conditions[3]['oth']['flowCoeff'] = 0.8


# === SOLVE STAGE 1: Meanline isentropic with minimal camberline ===
# This determines the rotational speed at midspan (single spanwise station)
ntw.system.build(SCALED)
rootfinder_mean_is = ntw.system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
    },
)
x0_mean_is = ntw.get_scaled_guess()
kn_mean_is = ntw.get_scaled_constraints()
bnd_mean_is = ntw.get_arguments_bounds()
solution = solve_root_problem(
    rootfinder_mean_is,
    x0_mean_is,
    kn_mean_is,
    bnd_mean_is,
    perturbate_guess=False,
    delta_pert=0.01,
    num_samples=100,
)

sol_mean_is_dict = ntw.system.write_solution_to_nodes(solution)


# # # # # # # # # # # # # #
if RUN_MULTI:
    # Only use profile
    class LossAdder(EquationBase):
        def residual(self, stc_smass0, stc_smass1, oth_delta_smass_profile1):
            return stc_smass1 - (stc_smass0 + oth_delta_smass_profile1)

    # ntw.system.remove_equation_type(LossModel)
    ntw.system.boundary_conditions[0]['geo'].pop('hubtipRatio')
    for nodes in nodes_by_stage:
        # 1. Fix blade heights from meanline solution
        ntw.system.boundary_conditions[nodes[0]]['geo']['height'] = sol_mean_is_dict[
            f'geo_height{nodes[0]}'
        ]
        ntw.system.boundary_conditions[nodes[1]]['geo']['height'] = sol_mean_is_dict[
            f'geo_height{nodes[1]}'
        ]
        ntw.system.boundary_conditions[nodes[3]]['geo']['height'] = sol_mean_is_dict[
            f'geo_height{nodes[3]}'
        ]
        ntw.system.boundary_conditions[nodes[0]]['geo']['rr_midspan'] = (
            sol_mean_is_dict[f'geo_rr_midspan{nodes[0]}']
        )
        ntw.system.boundary_conditions[nodes[1]]['geo']['rr_midspan'] = (
            sol_mean_is_dict[f'geo_rr_midspan{nodes[1]}']
        )
        ntw.system.boundary_conditions[nodes[3]]['geo']['rr_midspan'] = (
            sol_mean_is_dict[f'geo_rr_midspan{nodes[3]}']
        )

        # 2. Fix rotational speed from meanline solution
        ntw.system.boundary_conditions[nodes[3]]['kin']['omega'] = sol_mean_is_dict[
            f'kin_omega{nodes[3]}'
        ]

        # 3. Free vortex distribution
        ntw.system.boundary_conditions[nodes[1]]['kin']['Vt_midspan'] = (
            sol_mean_is_dict[f'kin_Vt{nodes[1]}']
        )
        ntw.system.boundary_conditions[nodes[3]]['kin']['Vt_midspan'] = (
            sol_mean_is_dict[f'kin_Vt{nodes[3]}']
        )

        # 4. Remove fixed deflection/workCoeff, impose free vortex
        # ntw.system.boundary_conditions[nodes[1]]['kin'].pop('alpha', None)
        ntw.system.boundary_conditions[nodes[3]]['oth'].pop('reactDegree_ts', None)
        ntw.system.boundary_conditions[nodes[3]]['oth'].pop('workCoeff', None)
        ntw.system.boundary_conditions[nodes[3]]['oth'].pop('flowCoeff', None)

        ntw.system.add_equation(FreeVortexDistribution(), nodes[1])
        ntw.system.add_equation(FreeVortexDistribution(), nodes[3])

        # 5. Add profile losses
        # ntw.system.add_equation(DentonProfileLoss(), (nodes[0], nodes[1]))
        # ntw.system.add_equation(LossAdder(), (nodes[0], nodes[1]))
        # ntw.system.add_equation(DentonProfileLoss(), (nodes[2], nodes[3]))
        # ntw.system.add_equation(LossAdder(), (nodes[2], nodes[3]))

    ntw.system.remove_equation_type(RepeatedStage)
    ntw.system.num_span = NUM_SPAN
    ntw.system.build()

    x0_span_is = ntw.system.get_scaled_guess(sol_mean_is_dict)
    kn_span_is = ntw.system.get_scaled_constraints()
    bnd_span_is = ntw.system.get_arguments_bounds()
    rootfind_span_is = ntw.system.make_rootfinder(
        'kinsol',
        {'error_on_fail': False},
    )

    sol_span_is = solve_root_problem(
        rootfind_span_is, x0_span_is, kn_span_is, bnd_span_is
    )
    sol_span_is_dict = ntw.system.write_solution_to_nodes(sol_span_is)


# === POST-PROCESSING AND VISUALIZATION
# plt.style.use('dark_background')
if PLOTS:
    FONTSIZE = 18
    FONTDICT = {'fontsize': FONTSIZE}

    # Plot velocity triangles for each node
    for i, n in enumerate(ntw.system.nodes):
        _, ax = plt.subplots()
        ax.set_aspect('equal')
        n.kin.plot(n.geo, FONTSIZE, ax)
        plt.title(f'Node number {i}')

    plt.tick_params(labelsize=FONTSIZE / 1.5 // 1)

    num_nodes = range(len(ntw.system.nodes))

    # Plot meridional profile and camberlines in subplots
    fig, (ax_merid, ax_camber) = plt.subplots(1, 2, figsize=(14, 6))

    # Configure meridional profile subplot
    ax_merid.axis('equal')
    ax_merid.set_ylabel('radius [m]', {'fontsize': 18})
    ax_merid.set_xlabel('axial  [m]', {'fontsize': 18})
    max_Y = (
        1.1
        * (
            ntw.system.nodes[-1].geo.get('rr_midspan').magnitude
            + ntw.system.nodes[-1].geo.get('height').magnitude / 2
        )[0]
    )
    ax_merid.set_ylim(-0.01, max_Y)
    ax_merid.tick_params('both', labelsize=18)
    ax_merid.grid()
    ax_merid.set_title('Meridional profile', {'fontsize': 18})

    # Configure camberline subplot
    ax_camber.set_title('Camberlines at midspan', {'fontsize': 18})
    ax_camber.axis('equal')
    ax_camber.tick_params('both', labelsize=18)
    ax_camber.grid()

    # Merged loop for both plots
    pbl = ParabolicCamberline()
    offset = 0.0
    for n0_idx, n1_idx in grouper(num_nodes, 2, incomplete='ignore'):
        n0 = ntw.system.nodes[n0_idx]
        n1 = ntw.system.nodes[n1_idx]
        ax_chord = n1.geo.get('chord_ax').to_base_units().magnitude[0]

        # Plot meridional profile
        is_stator = (n0_idx // 2) % 2 == 0
        blade_type = 'Stator' if is_stator else 'Rotor'
        stage_num = n0_idx // 4
        color = 'steelblue' if is_stator else 'coral'

        lines = plot_from_nodes(
            n0,
            n1,
            False,
            offset,
            ax=ax_merid,
            color=color,
        )

        # Plot camberlines at midspan (3 blades for all rows)
        midspan_idx = ntw.system.num_span // 2
        inlet_angle = n0.geo.metal_angle[midspan_idx]  # pyright:ignore
        outlet_angle = n1.geo.metal_angle[midspan_idx]  # pyright:ignore
        chord_ax = n1.geo.chord_ax[midspan_idx]  # pyright:ignore
        pitch = n1.geo.pitch[midspan_idx]  # pyright:ignore

        # Draw 3 camberlines for all blade rows
        num_blades = 3

        for blade_num in range(num_blades):
            pbl.plot_camber_line(
                ax_camber,
                inlet_angle,
                outlet_angle,
                chord_ax,
                'w',
                axial_offset=offset,
                tangential_offset=blade_num * pitch,
            )
        # Plot hub and tip
        pbl.plot_camber_line(
            ax_camber,
            n0.geo.metal_angle[0],
            n1.geo.metal_angle[0],
            n1.geo.chord_ax[0],
            'orange',
            axial_offset=offset,
        )
        pbl.plot_camber_line(
            ax_camber,
            n0.geo.metal_angle[-1],
            n1.geo.metal_angle[-1],
            n1.geo.chord_ax[-1],
            'seagreen',
            axial_offset=offset,
        )

        offset += ax_chord * 1.1

    # Add axis reference line for meridional plot
    ax_merid.plot(
        [0.0, offset], [0.0, 0.0], color='r', linestyle='dashdot', linewidth=2.5
    )

    plt.tight_layout()

    blade_rows = list(grouper(num_nodes, 2, incomplete='ignore'))

    fig, ax = plt.subplots(figsize=(5, 5))

    cmap = plt.colormaps.get('OrRd')
    for idx, (n0_idx, n1_idx) in enumerate(blade_rows):
        n0 = ntw.system.nodes[n0_idx]
        n1 = ntw.system.nodes[n1_idx]

        smass_in = n0.stc.smass
        smass_out = n1.stc.smass

        # Extract blade height for normalization
        height_in = n0.geo.height
        # Normalize radial positions by blade height (hub = 0, tip = 1)
        radii = n0.geo.rr
        rr_midspan = n0.geo.rr_midspan
        span_normalized = (radii - (rr_midspan - height_in / 2)) / height_in

        # Determine blade type and color
        is_stator = (n0_idx // 2) % 2 == 0
        blade_type = 'Stator' if is_stator else 'Rotor'
        stage_num = n0_idx // 4
        linestyle = '-' if is_stator else '--'

        color = cmap(idx / (len(ntw.components) - 1))

        # Plot
        ax.plot(
            span_normalized,
            smass_out,  # Convert to percentage
            label=f'Stage {stage_num} {blade_type}',
            color=color,
            linestyle='-',
            linewidth=2,
            marker='o',
        )

    ax.set_xlabel('Normalized Span (hub=0, tip=1)', FONTDICT)
    ax.set_title('Entropy [J / kg / K]', FONTDICT)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=FONTSIZE // 1.5)
    fig.tight_layout()


# === PRINT NODE INFORMATION
for i, node in enumerate(ntw.system.nodes):
    # For simpler access set n0, n1, n2, ...
    globals()[f'n{i}'] = node

    if PRINTS:
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
