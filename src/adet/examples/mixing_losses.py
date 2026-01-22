# === IMPORTS
# Standard library
import logging

from pint import Quantity

# Equations
from adet.components.blade_row import DownstreamMixer
from adet.equations.definitions import BoundaryLayerRatios
from adet.equations.fundamental import ZeroBlockage
from adet.equations.geometrical import (
    MeridionalVariable,
    MinimalCamberLine,
    ParabolicCamberline,
)
from adet.equations.nondimensional import WorkCoefficient
from adet.losses.basic import (
    PercentageEntropyLoss,
    PlaceHolderLoss,
    ZeroDeviation,
)

# Tooling & Components
from adet.tools.coolprop_utils import DebugAbstractState
from adet.registries import DefaultUnitsRegistry, GuessRegistry, VariableBoundsRegistry
from adet.fluid.settings import ExternalFluidModel
from adet.components import BladeRow, Shaft, Inlet

# External libraries
import matplotlib.pyplot as plt
import jax

# Network build
from adet.assembly import CasadiSystem, solve_root_problem
from adet.components import ComponentNetwork
from adet.fluid.settings import FluidSettings

# Objects Configuration => MODIFY CONFIG FILE TO SET BOUNDARY CONDITIONS

# Tooling and utils
from adet.losses.base_loss import LossModel
from adet.losses.profile import DentonProfileLoss
from adet.tools.iter import grouper
from adet.tools.loggers import setup_logger
from adet.components.blade_row import plot_from_nodes

logger = logging.getLogger(__name__)
jax.config.update('jax_enable_x64', True)

setup_logger(
    logger,
    logging.INFO,
    logging.INFO,
    suppress_modules=['matplotlib', 'jax'],
    banned_keywords=['STREAM', 'findfont', 'sBIT'],
)

# Disable verbose jax debug logs that somehow elude
# the logging filter I set up for it

# === SETTINGS
NUM_SPAN = 5
SCALED = True
PLOTS = True
PRINTS = True

# This counts the number of updates in an attribute
abs_state = DebugAbstractState('HEOS', 'Air')
abs_state.debug_print = False

real_model = ExternalFluidModel(abs_state)
settings = FluidSettings(
    model=real_model,
    update_variables=('p', 'T', 'rhomass', 'smass', 'hmass'),
    update_length=2,
)

_defreg = DefaultUnitsRegistry()
_guess_reg = GuessRegistry()
_guess_reg.reset()
_guess_reg.set_fallback_value(1.2)

_limreg = VariableBoundsRegistry()
# _limreg.set('Vm', (0.0, 100.0))

# Set fallback values for scales and guesses to 1.0

# *** Shafts
casing = Shaft(
    Quantity(0.0, 'rpm'),
    is_constrained=True,
)
rotating_shaft = Shaft(
    Quantity(1000.0, 'rpm'),
    is_constrained=True,
)

# COMPONENT STACK
inlet = Inlet(
    {
        'kin': {
            'alpha': Quantity(0, 'deg'),
            'Vm': Quantity(80, 'm/s'),
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rr_midspan': 0.5,
            'height': 0.2,
        },
        'tot': {
            'p': 6e5,
            'T': 700,
        },
    }
)

row0 = BladeRow(
    name='Stator',
    shaft=casing,
    in_constraints={
        'geo': {
            'thick_by_pitch': 0.04,
        },
    },
    out_constraints={
        'kin': {
            'alpha': Quantity(60, 'deg'),
            # 'mach': 0.2,
        },
        'geo': {
            # Meridional
            'meridional_angle': Quantity(0, 'deg'),
            'rr_midspan': 0.5,
            # Blade
            'chord_ax': 0.15,
            'num_blades': 25,
            'thick_by_pitch': 0.02,  # Blade thickness by pitch
            'heightRatio': 1.1,
        },
        'tot': {
            # 'p': 5e5,  # Impose either here or at inlet
        },
        'oth': {
            'mom_by_bld_Ratio': 0.075,
            'disp_by_mom_Ratio': 2,
        },
    },
    extra_equations={
        # Camberline model
        MinimalCamberLine(): (0, 1),
        # TwoSegmentCamberline(): (0, 1),
        # ParabolicCamberline(): (0, 1),
        # |> Losses & Dev
        ZeroDeviation(): 0,
        ZeroDeviation(): 1,
        PercentageEntropyLoss(0.0): (0, 1),
        # |> Boundary layer properties for mixing
        BoundaryLayerRatios(): 1,
    },
)

mixer = DownstreamMixer(
    'Mixer',
    in_constraints={
        'geo': {
            'metal_angle': 0.0,
        },
    },
    out_constraints={
        'geo': {
            # PLOTTING for sanity checks, no physical meaning
            'chord_ax': 0.05,
            'metal_angle': 0.0,
        },
    },
    extra_equations={
        PlaceHolderLoss(): (0, 1),
        MeridionalVariable(): 1,
    },
)


# Create network
ntw = ComponentNetwork(
    settings,  # Fluid settings
    inlet,  # Inlet conditions
    CasadiSystem(num_span=NUM_SPAN),  # Backend
    [
        row0,
        mixer,
    ],
)

ntw.system.build(SCALED)

rootfinder_is = ntw.system.make_rootfinder(
    'ipopt',
    opts={
        'ipopt.tol': 1e-6,
    },
)

x0_is = ntw.system.get_initial_guess()
kn_is = ntw.system.get_scaled_constraints()
bnd_is = ntw.system.get_arguments_bounds()
sol_is = solve_root_problem(rootfinder_is, x0_is, kn_is, bnd_is, suppress_output=False)

# Write solution to network (just for post processing)
ntw.system.write_solution_to_nodes(sol_is)

ntw.print_structure()
nodes = {}
for i, node in enumerate(ntw.system.nodes):
    # For simpler access set n0, n1, n2, ...
    nodes[i] = node
    globals()[f'n{i}'] = node


if PLOTS:
    FONTSIZE = 18
    FONTDICT = {'fontsize': FONTSIZE}

    fig, ax = plt.subplots(len(nodes))
    # Plot velocity triangles
    for i, n in enumerate(ntw.system.nodes):
        n.kin.plot(n.geo, FONTSIZE, ax[i])
        plt.title(f'Node number {i}')

    # Plot entropy rise
    fig, ax = plt.subplots()
    ax.set_title('Entropy rise')
    smass0 = ntw.system.nodes[0].stc.smass
    for i, n in enumerate(ntw.system.nodes):
        # Plot entropy distributions
        ax.plot(n.geo.rr, n.stc.smass - smass0)  # pyright:ignore

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
        nodes[0] = ntw.system.nodes[n0_idx]
        nodes[1] = ntw.system.nodes[n1_idx]
        ax_chord = nodes[1].geo.get('chord_ax').to_base_units().magnitude[0]

        # Plot meridional profile
        lines = plot_from_nodes(
            nodes[0],
            nodes[1],
            False,
            offset,
            ax=ax_merid,
        )

        # Plot camberlines at midspan (3 blades for rotor, 1 for stator)
        midspan_idx = ntw.system.num_span // 2
        inlet_angle = nodes[0].geo.metal_angle[midspan_idx]  # pyright:ignore
        outlet_angle = nodes[1].geo.metal_angle[midspan_idx]  # pyright:ignore
        chord_ax = nodes[1].geo.chord_ax[midspan_idx]  # pyright:ignore
        pitch = nodes[1].geo.pitch[midspan_idx]  # pyright:ignore
        num_blades = 3  # blades to plot

        for blade_num in range(num_blades):
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

print(f'Num updates = {real_model.eos_object.num_updates}')
plt.show()
plt.close('all')
