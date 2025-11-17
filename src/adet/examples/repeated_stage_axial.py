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

# External libraries
import matplotlib.pyplot as plt
from pint import Quantity

# Network build
from adet.assembly import CasadiSystem, solve_problem
from adet.components import ComponentNetwork, BladeRow, Shaft, Inlet
from adet.equations.definitions import DegreeOfReaction, RepeatedStage
from adet.equations.geometrical import ParabolicCamberline, MinimalCamberLine
from adet.fluid.casadi_eos import CasadiEoS
from adet.fluid.settings import FluidSettings, ExternalFluidModel, IdealGasModel

# Losses
from adet.losses.base_loss import LossModel
from adet.losses.basic import PercentageEntropyLoss, ZeroDeviation
from adet.losses.profile import DentonProfileLoss
from adet.equations.nondimensional import FlowCoefficient, WorkCoefficient

# Tooling and utils
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.iter import grouper
from adet.tools.loggers import setup_logger
from adet.components.blade_row import plot_from_nodes
from adet.registries import DefaultUnitsRegistry, ScalingRegistry, GuessRegistry


# === LOGGING SETUP
logger = logging.getLogger(__name__)

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
NUM_SPAN = 1  # Number of spanwise stations
NUM_STAGES = 5  # Number of turbine stages (stator-rotor pairs)
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
        'xi_by_camb_len_A': 'meters',
        'xi_by_camb_len_B': 'meters',
        'k_prof': '',
    }
)

_scl_reg.set_fallback_value(1.0)
_gss_reg.set_fallback_value(1.0)

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
            'height': 0.15,  # Blade height [m]
        },
        'tot': {
            'T': 1300,  # Total temperature [K]
            'p': 20e5,  # Total pressure [Pa]
        },
    }
)

# Stator blade row definition
stator = BladeRow(
    'Stator',
    {
        'kin': {
            'alpha': Quantity(70, 'deg'),  # Exit flow angle
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.5,
            'chord': 0.15,  # Blade chord length [m]
            'n_blades': 30,  # Number of blades
            'te_by_pitch': 0.02,
        },
    },
    shaft=casing,
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
    {
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rmid': 0.5,
            'chord': 0.15,
            'n_blades': 30,
            'te_by_pitch': 0.02,
        },
    },
    shaft=rotating_shaft,
    extra_equations={
        PercentageEntropyLoss(0.0): (0, 1),
        MinimalCamberLine(): (0, 1),
        ZeroDeviation(): 0,
        ZeroDeviation(): 1,
        # Work and flow coefficients defined only on rotor
        WorkCoefficient(): (0, 1),
        FlowCoefficient(): 1,
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
    ntw.system.add_equation(DegreeOfReaction(), nodes)
    # Set 50% degree of reaction
    ntw.system.add_boundary_conditions({'oth': {'reactDegree': 0.5}}, nodes[-1])


# === SOLVE STAGE 1: Meanline isentropic with minimal camberline ===
# This determines the rotational speed at midspan (single spanwise station)
ntw.system.build(SCALED)
rootfinder_mean_is = ntw.system.make_rootfinder('nlpsol')
x0_mean_is = ntw.system.get_initial_guess()
kn_mean_is = ntw.system.get_scaled_constraints()
sol_mean_is = solve_problem(rootfinder_mean_is, x0_mean_is, kn_mean_is)
sol_mean_is_dict = ntw.system.solution_to_dict(sol_mean_is)

# === SOLVE STAGE 2: Meanline with Denton losses (midspan only) ===
sys_mean_loss = ntw.system.copy()
sys_mean_loss.remove_equation_type(LossModel)
# Keep MinimalCamberLine, add Denton losses
for nodes in nodes_by_stage:
    stator_nodes = (nodes[0], nodes[1])
    rotor_nodes = (nodes[2], nodes[3])
    sys_mean_loss.add_equation(DentonProfileLoss(real_model), stator_nodes)
    sys_mean_loss.add_equation(DentonProfileLoss(real_model), rotor_nodes)

sys_mean_loss.build(SCALED)
rootfinder_mean_loss = sys_mean_loss.make_rootfinder('nlpsol')
x0_mean_loss = sys_mean_loss.get_initial_guess(sol_mean_is_dict)
kn_mean_loss = sys_mean_loss.get_scaled_constraints()
sol_mean_loss = solve_problem(rootfinder_mean_loss, x0_mean_loss, kn_mean_loss)
sol_mean_loss_dict = sys_mean_loss.solution_to_dict(sol_mean_loss)

# === SOLVE STAGE 3: Multi-span with Denton losses and minimal camberline ===
sys_span_loss = sys_mean_loss.copy()
sys_span_loss.spanwise_stations = NUM_SPAN
sys_span_loss.remove_equation_type(RepeatedStage)

for nodes in nodes_by_stage:
    # Fix rotational speed from meanline solution
    sys_span_loss.boundary_conditions[nodes[3]]['kin']['omega'] = sol_mean_loss_dict[
        f'kin_omega{nodes[3]}'
    ]
    # Fix blade heights from meanline solution
    sys_span_loss.boundary_conditions[nodes[1]]['geo']['height'] = sol_mean_loss_dict[
        f'geo_height{nodes[1]}'
    ]
    sys_span_loss.boundary_conditions[nodes[3]]['geo']['height'] = sol_mean_loss_dict[
        f'geo_height{nodes[3]}'
    ]

sys_span_loss.build(SCALED)
rootfinder_span_loss = sys_span_loss.make_rootfinder('nlpsol')
x0_span_loss = sys_span_loss.get_initial_guess(sol_mean_loss_dict)
kn_span_loss = sys_span_loss.get_scaled_constraints()
sol_span_loss = solve_problem(rootfinder_span_loss, x0_span_loss, kn_span_loss)
sol_span_loss_dict = sys_span_loss.solution_to_dict(sol_span_loss)

# === SOLVE STAGE 4: Multi-span with Denton losses and parabolic camberline ===
sys_camber = sys_span_loss.copy()
sys_camber.remove_equation_type(MinimalCamberLine)
for nodes in nodes_by_stage:
    stator_nodes = (nodes[0], nodes[1])
    rotor_nodes = (nodes[2], nodes[3])
    sys_camber.add_equation(ParabolicCamberline(), stator_nodes)
    sys_camber.add_equation(ParabolicCamberline(), rotor_nodes)

sys_camber.build(SCALED)
rootfinder_camber = sys_camber.make_rootfinder('nlpsol')
x0_camber = sys_camber.get_initial_guess(sol_span_loss_dict)
kn_camber = sys_camber.get_scaled_constraints()
sol_camber = solve_problem(rootfinder_camber, x0_camber, kn_camber)
sol_camber_dict = sys_camber.solution_to_dict(sol_camber)

# === SOLVE STAGE 5: High-resolution multi-span ===

sys_highres = sys_camber.copy()
sys_highres.spanwise_stations = NUM_SPAN * 5


sys_highres.build(SCALED)
rootfinder_highres = sys_highres.make_rootfinder('nlpsol')
# Use interpolated solution as initial guess
x0_highres = sys_highres.get_initial_guess(sol_camber_dict)
kn_highres = sys_highres.get_scaled_constraints()
sol_highres = solve_problem(rootfinder_highres, x0_highres, kn_highres)

# Write solution back to FlowNodes
ntw.system = sys_highres
ntw.system.write_solution_to_nodes(sol_highres)


# === POST-PROCESSING AND VISUALIZATION
if PLOTS:
    FONTSIZE = 18
    FONTDICT = {'fontsize': FONTSIZE}
    COLORMAP = plt.get_cmap('viridis')

    # Plot velocity triangles for each node
    for i, n in enumerate(ntw.system.nodes):
        n.kin.plot(n.geo, FONTSIZE)
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
            ntw.system.nodes[-1].geo.get('rmid').magnitude
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
        lines = plot_from_nodes(
            n0,
            n1,
            False,
            offset,
            ax=ax_merid,
            color=COLORMAP((n1_idx - 1) // ntw.num_components),
        )

        # Plot camberlines at midspan (3 blades for all rows)
        midspan_idx = ntw.system.spanwise_stations // 2
        inlet_angle = n0.geo.metal_angle[midspan_idx]  # pyright:ignore
        outlet_angle = n1.geo.metal_angle[midspan_idx]  # pyright:ignore
        chord_ax = n1.geo.chord_ax[midspan_idx]  # pyright:ignore
        pitch = n1.geo.pitch[midspan_idx]  # pyright:ignore

        # Draw 3 camberlines for all blade rows
        n_blades = 3

        for blade_num in range(n_blades):
            pbl.plot_camber_line(
                ax_camber,
                inlet_angle,
                outlet_angle,
                chord_ax,
                'k',
                axial_offset=offset,
                tangential_offset=blade_num * pitch,
            )

        offset += ax_chord * 1.1

    # Add axis reference line for meridional plot
    ax_merid.plot(
        [0.0, offset], [0.0, 0.0], color='r', linestyle='dashdot', linewidth=2.5
    )

    plt.tight_layout()

    blade_rows = list(grouper(num_nodes, 2, incomplete='ignore'))

    # === RELATIVE TOTAL PRESSURE INCREMENT PLOT
    # Plot the relative total pressure change across blade rows
    # as well as entropy change
    from CoolProp import PT_INPUTS

    PT_EOS = CasadiEoS(
        'PT_EoS',
        real_model.eos_object,
        PT_INPUTS,
        ['smass'],
        ntw.system.spanwise_stations,
    )
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    for idx, (n0_idx, n1_idx) in enumerate(blade_rows):
        n0 = ntw.system.nodes[n0_idx]
        n1 = ntw.system.nodes[n1_idx]

        # Extract relative total pressure at inlet and outlet
        p_rlt_in = n0.rlt.get('p').to('Pa').magnitude
        p_rlt_out = n1.rlt.get('p').to('Pa').magnitude

        T_rlt_in = n0.rlt.get('T').to('K').magnitude
        T_rlt_out = n1.rlt.get('T').to('K').magnitude

        smass_in = PT_EOS(p_rlt_in, T_rlt_in)
        smass_out = PT_EOS(p_rlt_out, T_rlt_out)

        # Calculate pressure increment (normalized by inlet pressure)
        delta_p_normalized = (p_rlt_out - p_rlt_in) / p_rlt_in

        # Extract blade height for normalization
        height_in = n0.geo.get('height').to('m').magnitude[0]

        # Normalize radial positions by blade height (hub = 0, tip = 1)
        radii = n0.geo.get('rr').to('m').magnitude
        rmid = n0.geo.get('rmid').to('m').magnitude[0]
        span_normalized = (radii - (rmid - height_in / 2)) / height_in

        # Determine blade type and color
        is_stator = (n0_idx // 2) % 2 == 0
        blade_type = 'Stator' if is_stator else 'Rotor'
        stage_num = n0_idx // 4
        color = 'steelblue' if is_stator else 'coral'
        linestyle = '-' if is_stator else '--'

        # Plot
        ax[0].plot(
            span_normalized,
            delta_p_normalized * 100,  # Convert to percentage
            label=f'Stage {stage_num} {blade_type}',
            color=color,
            linestyle=linestyle,
            linewidth=2,
            marker='o',
        )
        ax[1].plot(
            span_normalized,
            smass_out,  # Convert to percentage
            label=f'Stage {stage_num} {blade_type}',
            color=color,
            linestyle=linestyle,
            linewidth=2,
            marker='o',
        )

    ax[0].set_xlabel('Normalized Span (hub=0, tip=1)', FONTDICT)
    ax[0].set_ylabel('Relative Total Pressure Change [%]', FONTDICT)
    ax[0].set_title('Relative Total Pressure Increment Across Blade Rows', FONTDICT)
    ax[1].set_title('Entropy [J / kg / K]', FONTDICT)
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc='best', fontsize=FONTSIZE // 1.5)
    fig.tight_layout()


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
