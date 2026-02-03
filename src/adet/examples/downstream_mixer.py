# === IMPORTS
# Standard library
import logging

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from pint import Quantity

from adet.assembly import CasadiSystem, solve_root_problem
from adet.components import BladeRow, Inlet, Shaft
from adet.components import ComponentNetwork
from adet.components.blade_row import DownstreamMixer
from adet.components.blade_row import plot_from_nodes
from adet.equations.definitions import BoundaryLayerRatios, IsentropicProperties
from adet.equations.fundamental import BladeBlockage
from adet.equations.geometrical import MinimalCamberLine, ParabolicCamberline
from adet.equations.nondimensional import TotalTotalExpansionEfficiency
from adet.fluid.settings import ExternalFluidModel
from adet.fluid.settings import FluidSettings
from adet.losses.basic import PlaceHolderLoss, ZeroDeviation
from adet.losses.mixing import SieverdingBasePressure
from adet.losses.profile import DentonProfileLoss, RectVelocityIncompressible
from adet.registries import DefaultUnitsRegistry, GuessRegistry, VariableBoundsRegistry
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

# === SETTINGS
NUM_SPAN = 5
SCALED = True
PLOTS = True
PRINTS = True
INITIAL_LOSS = RectVelocityIncompressible()

# This counts the number of updates in an attribute
abs_state = DebugAbstractState('REFPROP', 'air')
abs_state.debug_print = False

real_model = ExternalFluidModel(abs_state)
settings = FluidSettings(
    model=real_model,
    update_variables=('p', 'T', 'rhomass', 'smass', 'hmass'),
    update_length=2,
)

_defreg = DefaultUnitsRegistry()
_defreg.from_dict(
    {
        'xi_by_camb_.*': 'dimensionless',
        'Cd_prof': 'dimensionless',
        'k_prof': 'dimensionless',
    }
)
_guess_reg = GuessRegistry()
_guess_reg.reset()
_guess_reg.set_fallback_value(0.5)

_bounds_reg = VariableBoundsRegistry()
_bounds_reg.reset()
_bounds_reg.set('delta_smass_.*', (0.0, 100.0))

# *** Shafts
shaft = Shaft(
    Quantity(0.0, 'rpm'),
    is_constrained=True,
)

# COMPONENT STACK
inlet = Inlet(
    {
        'kin': {
            'beta': Quantity(-30, 'deg'),
            'Vm': Quantity(70, 'm/s'),
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rr_midspan': 0.5,
            'height': 0.2,
        },
        'tot': {
            'p': 5e5,
            'T': 600,
        },
    }
)

row = BladeRow(
    name='Stator',
    shaft=shaft,
    in_constraints={
        'geo': {
            'thick_by_pitch': 0.04,
        },
    },
    out_constraints={
        'kin': {
            'beta': Quantity(-40, 'deg'),
            # 'relmach': 1.0,
            # 'mach': 0.2,
        },
        'geo': {
            # Meridional
            'meridional_angle': Quantity(0, 'deg'),
            'rr_midspan': 0.5,
            # Blade
            'chord_ax': 0.15,
            'num_blades': 30,
            'thick_by_pitch': 0.02,
            'heightRatio': 1.0,
        },
        'oth': {
            'mom_by_bld': 0.075,
            'disp_by_mom': 2,
            'disp_by_hgt': 0.05,
            # Profile losses
            'Cd_profile': 0.002,
            'xi_by_camb_len_A': 0.375,
            'xi_by_camb_len_B': 0.675,
        },
    },
    extra_equations={
        # Camberline model
        # MinimalCamberLine(): (0, 1),
        # TwoSegmentCamberline(): (0, 1),
        ParabolicCamberline(): (0, 1),
        # |> Losses & Dev
        ZeroDeviation(): 0,
        ZeroDeviation(): 1,
        # PercentageEntropyLoss(0.0): (0, 1),
        # |> Boundary layer properties
        BoundaryLayerRatios(): 1,
        BladeBlockage(): 1,
        SieverdingBasePressure(): (0, 1),
        INITIAL_LOSS: (0, 1),
        # Efficiency measures
    },
)

mixer = DownstreamMixer(
    'Mixer',
    out_constraints={
        'geo': {
            # PLOTTING for sanity checks, no physical meaning
            'chord_ax': 0.05,
        },
    },
    extra_equations={
        ZeroDeviation(): 1,  # Create a fake metal angle for plotting
        # PercentageEntropyLoss(0.0): (0, 1),
    },
)


# Create network
ntw = ComponentNetwork(
    settings,  # Fluid settings
    inlet,  # Inlet conditions
    CasadiSystem(num_span=NUM_SPAN),  # Backend
    [
        row,
        mixer,
    ],
)

if ntw.num_components == 2:
    ntw.system.add_equation(IsentropicProperties(), (0, 3))
    ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, 3))

ntw.system.build(SCALED)

rootfinder_is = ntw.system.make_rootfinder('ipopt')

x0_is = ntw.system.get_initial_guess()
kn_is = ntw.system.get_scaled_constraints()
bnd_is = ntw.system.get_arguments_bounds()
solution = solve_root_problem(
    rootfinder_is,
    x0_is,
    kn_is,
    bnd_is,
    suppress_output=True,
    perturbate_guess=True,
    delta_pert=0.01,
)

ntw.system.write_solution_to_nodes(solution)
ntw.print_structure()

nodes = {}
for i, node in enumerate(ntw.system.nodes):
    # For simpler access set n0, n1, n2, ...
    nodes[i] = node
n0 = nodes[0]
n1 = nodes[1]


if PLOTS:
    FONTSIZE = 18
    FONTDICT = {'fontsize': FONTSIZE}

    # Plot velocity triangles
    for i, node in enumerate(ntw.system.nodes):
        _, ax = plt.subplots()
        ax.set_aspect('equal')
        node.kin.plot(node.geo, FONTSIZE, ax)
        plt.title(f'Node number {i}')

    # Plot entropy rise
    fig, ax = plt.subplots()
    ax.set_title('Entropy rise')
    smass0 = ntw.system.nodes[0].stc.smass
    for i, node in enumerate(ntw.system.nodes):
        # Plot entropy distributions
        ax.plot(node.geo.rr, node.stc.smass - smass0, label=f'Node {i}')  # pyright:ignore
        ax.legend()

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
        num_plt_blades = 3  # blades to plot

        for blade_num in range(num_plt_blades):
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
if ntw.num_components > 1:
    n2 = nodes[2]
    n3 = nodes[3]
    print(f'Entropy rise {n3.stc.smass - n2.stc.smass}')
    print(f'Deviation {n3.kin.beta - n2.kin.beta}')

    q = 0.5 * n2.stc.rhomass * n2.kin.W**2
    w = n2.geo.pitch * np.cos(n2.geo.metal_angle)
    cpb = (n2.oth.p_base - n2.stc.p) / q

    zeta_inc = (
        -(cpb * n2.geo.bld_thick) / w
        + 2 * n2.oth.mom_thick / w
        + ((n2.oth.disp_thick + n2.geo.bld_thick) / w) ** 2
    )

    zeta_actual = (n2.rlt.p - n3.rlt.p) / q

    print(f'Incompressible vs actual zeta {zeta_inc}, {zeta_actual}')

plt.show()
# plt.close('all')
