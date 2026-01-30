# === IMPORTS
# Standard library
from copy import deepcopy
import logging

import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity

from adet.assembly import CasadiSystem, solve_root_problem
from adet.components import BladeRow, Inlet, Shaft
from adet.components import ComponentNetwork
from adet.components.blade_row import DownstreamMixer
from adet.components.blade_row import plot_from_nodes
from adet.equations.base_equation import EquationBase
from adet.equations.definitions import (
    BoundaryLayerRatios,
    ClearanceByHeight,
    IsentropicProperties,
    ReducedThermoQuantities,
)
from adet.equations.fundamental import BladeBlockage
from adet.equations.geometrical import ParabolicCamberline
from adet.equations.nondimensional import FlowCoefficient, TotalTotalExpansionEfficiency
from adet.equations.utils import residual_debugger
from adet.fluid.settings import ExternalFluidModel
from adet.fluid.settings import FluidSettings
from adet.losses.basic import PercentageEntropyLoss, PlaceHolderLoss, ZeroDeviation
from adet.losses.leakage import DentonLeakageLoss
from adet.losses.mixing import MixingMomentumBalances, SieverdingBasePressure
from adet.losses.profile import DentonProfileLoss
from adet.losses.secondary import SecondaryBSM
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
plt.close('all')

# === SETTINGS
NUM_SPAN = 1
SCALED = True
PLOTS = True
PRINTS = True
INITIAL_LOSS = PercentageEntropyLoss(0.0)

# This counts the number of updates in an attribute
abs_state = DebugAbstractState('HEOS', 'MM')
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
        'Y_.*': 'dimensionless',
    }
)
_guess_reg = GuessRegistry()
_guess_reg.reset()
_guess_reg.set('rhomass', 400)
_guess_reg.set_fallback_value(0.5)

_bounds_reg = VariableBoundsRegistry()
_bounds_reg.reset()
_bounds_reg.from_dict(
    {
        'delta_smass_.*': (0.0, 100.0),
        'mach': (0.0, 1.2),
    }
)

# *** Shafts
shaft = Shaft(
    Quantity(0.0, 'rpm'),
    is_constrained=True,
)

# COMPONENT STACK
inlet = Inlet(
    {
        'kin': {
            'beta': Quantity(30, 'deg'),
            'mach': 0.4,
        },
        'geo': {
            'meridional_angle': Quantity(0, 'deg'),
            'rr_midspan': 0.1,
            'height': 0.06,
        },
        # 'tot': {
        #     'p': 5e5,
        #     'T': 600,
        # },
        'oth': {
            'tot_p_red': 2.071,
            'tot_T_red': 1.052,
        },
    }
)

# Set the pressure and temperature guesses just above the inlet ones
_guess_reg.from_dict(
    {
        'p': 1.1
        * abs_state.p_critical()
        * inlet.boundary_conditions['oth']['tot_p_red'],
        'T': 1.1
        * abs_state.T_critical()
        * inlet.boundary_conditions['oth']['tot_T_red'],
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
        },
        'geo': {
            # Meridional
            'meridional_angle': Quantity(0, 'deg'),
            # Blade
            'radiusRatio': 1,
            # ***  height <-> chord
            # 'aspRatio': 2,
            'flare_angle': Quantity(5, 'deg'),
            # ***
            # If too low velocity suction explodes!
            'num_blades': 25,
            # 'solidity': 1.3,
            'thick_by_pitch': 0.02,
            'heightRatio': 1.1,
            'clearance_by_height': 0.01,
        },
        'oth': {
            'mom_by_bld': 0.075,
            'disp_by_mom': 2,
            'disp_by_hgt': 0.05,
            # Profile losses
            'Cd_profile': 0.002,
            'xi_by_camb_len_A': 0.375,
            'xi_by_camb_len_B': 0.675,
            # For tip leakage
            'dischCoeff': 0.35,
            # For secondary
        },
    },
    extra_equations={
        # Camberline model
        # MinimalCamberLine(): (0, 1),
        # TwoSegmentCamberline(): (0, 1),
        ParabolicCamberline(): (0, 1),
        ClearanceByHeight(): 1,
        ReducedThermoQuantities(): 0,
        # |> Losses & Dev
        ZeroDeviation(): 0,  # No incidence (design)
        ZeroDeviation(): 1,
        IsentropicProperties(): (0, 1),
        # |> Boundary layer properties
        BoundaryLayerRatios(): 1,
        BladeBlockage(): 1,
        SieverdingBasePressure(): (0, 1),
        INITIAL_LOSS: (0, 1),
        # Nondimensional
        # FlowCoefficient(): (0, 1),
    },
)


class LossMatcher(EquationBase):
    def residual(
        self,
        stc_smass0,
        stc_smass1,
        oth_delta_smass_leakage1,
        oth_delta_smass_profile1,
        oth_delta_smass_secondary1,
    ):
        return stc_smass1 - (
            stc_smass0
            + oth_delta_smass_leakage1
            + oth_delta_smass_profile1
            + oth_delta_smass_secondary1
        )


mixer = DownstreamMixer(
    'Mixer',
    # The axial chord is just for plotting it like a row
    out_constraints={'geo': {'chord_ax': 0.01}},
    # This creates a dummy metal angle for plotting like a blade
    extra_equations={ZeroDeviation(): 1},
)


# Create network
ntw = ComponentNetwork(
    settings,  # Fluid settings
    inlet,  # Inlet conditions
    CasadiSystem(num_span=NUM_SPAN),  # Backend
    components=[
        row,
        mixer,
        deepcopy(row),
        deepcopy(mixer),
    ],
)

if ntw.num_components > 1:
    final_node = ntw.num_components * 2 - 1
    # ntw.system.add_equation(IsentropicProperties(), (0, final_node))
    # ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, final_node))

if ntw.num_components >= 2:
    ntw.system.boundary_conditions[5]['kin'].pop('beta')
    # Set outlet of second row to axial
    ntw.system.boundary_conditions[5]['kin']['alpha'] = 0.0
    ntw.system.boundary_conditions[5]['kin']['omega'] = -300

ntw.system.num_span = NUM_SPAN
ntw.system.build(SCALED)

rootfinder_is = ntw.system.make_rootfinder('ipopt', opts={'error_on_fail': True})

x0_is = ntw.system.get_initial_guess()
kn_is = ntw.system.get_scaled_constraints()
bnd_is = ntw.system.get_arguments_bounds()
solution = solve_root_problem(
    rootfinder_is,
    x0_is,
    kn_is,
    bnd_is,
    suppress_output=True,
    perturbate_guess=False,
    delta_pert=0.02,
    num_samples=1000,
)

user = input('INPUT >>> Isentropic problem solved, continue with losses? [y/n] ')
if user in ('y', 'Y'):
    # Write solution to dict for reading for next solution
    sol_dict_is = ntw.system.solution_to_dict(solution)
    # Set outlet to 0 degrees
    new_span = input(
        'INPUT >>> Select number of spanwise stations for full system [int] '
    )

    ntw.system.num_span = int(new_span)
    # = = = AFTER INITIALIZING = = =
    ntw.system.remove_equation(INITIAL_LOSS.__class__, (0, 1))
    ntw.system.add_equation(DentonProfileLoss(), (0, 1))
    ntw.system.add_equation(DentonLeakageLoss(), (0, 1))
    ntw.system.add_equation(SecondaryBSM(), (0, 1))
    ntw.system.add_equation(LossMatcher(), (0, 1))  # For losses
    # ntw.system.add_equation(PercentageEntropyLoss(0.0), (0, 1))  # Compute w/o adding

    if ntw.num_components >= 2:
        # Add equations to second row
        ntw.system.remove_equation(INITIAL_LOSS.__class__, (4, 5))
        ntw.system.add_equation(DentonProfileLoss(), (4, 5))
        ntw.system.add_equation(DentonLeakageLoss(), (4, 5))
        ntw.system.add_equation(SecondaryBSM(), (4, 5))
        ntw.system.add_equation(LossMatcher(), (4, 5))
        # ntw.system.add_equation(PercentageEntropyLoss(0.0), (4, 5))

    ntw.system.build()

    x0_loss = ntw.system.get_initial_guess(sol_dict_is)
    kn_loss = ntw.system.get_scaled_constraints()
    bnd_loss = ntw.system.get_arguments_bounds()
    err_on_fail = int(
        input('INPUT >>> Fail on rootfinding error? [0/1] '),
    )
    rootfinder_loss = ntw.system.make_rootfinder(
        'ipopt',
        opts={
            'error_on_fail': bool(err_on_fail),
            'ipopt.constr_viol_tol': 1e-7,
        },
    )
    solution = solve_root_problem(
        rootfinder_loss,
        x0_loss,
        kn_loss,
        bnd_loss,
        suppress_output=False,
        perturbate_guess=False,
    )
    # = = = = = = = = = =


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
    smass_inlet = ntw.system.nodes[0].stc.smass
    for i, node in enumerate(ntw.system.nodes):
        # Plot entropy distributions
        if i % 2 == 0:
            continue

        ax.plot(node.geo.rr, node.stc.smass - smass_inlet, label=f'Node {i}')
        plt.legend()

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

        offset += ax_chord * 1.03

    # Add axis reference line for meridional plot
    ax_merid.plot(
        [0.0, offset], [0.0, 0.0], color='r', linestyle='dashdot', linewidth=2.5
    )

    plt.tight_layout()

print(f'Num updates = {real_model.eos_object.num_updates}')
if ntw.num_components > 1:
    n2 = nodes[2]
    n3 = nodes[3]
    print(f'delta_smass_mixing {n3.stc.smass - n2.stc.smass}')
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

print(n1.oth)
plt.show(block=False)
input('Press enter to close ')
plt.close('all')

# DEBUGGING EQUATIONS
globals().update(
    residual_debugger(
        MixingMomentumBalances(),
        [nodes[6], nodes[7]],
    )
)
