# === IMPORTS
# Standard library
import logging

# External libraries
import matplotlib.pyplot as plt
import jax

# Network build
from adet.assembly import CasadiSystem, solve_problem
from adet.components import ComponentNetwork
from adet.equations.geometrical import (
    MinimalCamberLine,
    ParabolicCamberline,
    TwoSegmentCamberline,
)
from adet.fluid.casadi_eos import CasadiEoS
from adet.fluid.settings import FluidSettings

# Objects Configuration => MODIFY CONFIG FILE TO SET BOUNDARY CONDITIONS
from adet.config_main import real_model, inlet, row0, row1

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
logging.getLogger('jax').setLevel(logging.WARNING)


# === SETTINGS
NUM_SPAN = 1
SCALED = True
PLOTS = True
PRINTS = True

# NOTE: I have now forced the system to add all possible update variables to each single
# state (tot, stc, rlt). So this means that the first two update variables will always
# be used. This has a significant influence on the convergence of the system, p and T
# variables seem to provide the most stable couple so far

settings = FluidSettings(
    model=real_model,
    update_variables=('p', 'T', 'rhomass', 'smass', 'hmass'),
    update_length=2,
)

# Create network
ntw = ComponentNetwork(
    settings,  # Fluid settings
    inlet,  # Inlet conditions
    CasadiSystem(spanwise_stations=NUM_SPAN),  # Backend
    *[row0, row1],
)

# Add global constraints for ideal gas and
# loss models
ntw.system.add_global_constraints(
    {
        'oth': {
            # Ideal gas (keep even w/ real model)
            'cpmassid': 1004.0,
            'cvmassid': 717.0,
            'T_ref': 1.0,
            'p_ref': 1.0,
            # Profile losses coefficients
            'Cd_profile': 0.002,
            'xi_by_camb_len_A': 0.375,  # First profile coord
            'xi_by_camb_len_B': 0.675,  # Second profile coord
        }
    }
)


ntw.system.build(SCALED)


rootfinder_is = ntw.system.make_rootfinder('nlpsol')

x0_is = ntw.system.get_initial_guess()
kn_is = ntw.system.get_scaled_constraints()

sol_is = solve_problem(rootfinder_is, x0_is, kn_is)

sol_is_dict = ntw.system.write_solution_to_nodes(sol_is)

# === Add losses and increase spanwise stations
sys_loss = ntw.system.copy()
sys_loss.spanwise_stations = NUM_SPAN * 3
sys_loss.remove_equation_type(LossModel)
sys_loss.add_equation(DentonProfileLoss(real_model), (0, 1))
sys_loss.add_equation(DentonProfileLoss(real_model), (2, 3))
sys_loss.build(SCALED)
rootfinder_loss = sys_loss.make_rootfinder('nlpsol')

x0_loss = sys_loss.get_initial_guess(sol_is_dict)
kn_loss = sys_loss.get_scaled_constraints()
sol_loss = solve_problem(rootfinder_loss, x0_loss, kn_loss)
sol_loss_dict = sys_loss.write_solution_to_nodes(sol_loss)

# === Switch to parabolic camberline
sys_camber = sys_loss.copy()
sys_camber.spanwise_stations = NUM_SPAN * 9
sys_camber.remove_equation_type(TwoSegmentCamberline, MinimalCamberLine)
sys_camber.add_equation(ParabolicCamberline(), (0, 1))
sys_camber.add_equation(ParabolicCamberline(), (2, 3))
sys_camber.build(SCALED)
rootfinder_camber = sys_camber.make_rootfinder('nlpsol')

x0_camber = sys_camber.get_initial_guess(sol_loss_dict)
kn_camber = sys_camber.get_scaled_constraints()
sol_camber = solve_problem(rootfinder_camber, x0_camber, kn_camber)

# Write solution to network (just for post processing)
ntw.system = sys_camber
sol = sol_camber
ntw.system.write_solution_to_nodes(sol)

# === POST-PROCESS ===
nodes = {}
if PRINTS:
    for i, node in enumerate(ntw.system.nodes):
        # For simpler access set n0, n1, n2, ...
        nodes[i] = node
        to_print = f"""
###################
##### NODE {i:<2} #####
###################
{node}\n
    """
        print(to_print)

    ntw.print_structure()


PT_EOS = CasadiEoS(
    'PT_EOS',
    real_model.eos_object,
    9,
    ['rhomass', 'smass'],
    ntw.system.spanwise_stations,
)


if PLOTS:
    FONTSIZE = 18
    FONTDICT = {'fontsize': FONTSIZE}

    # Plot velocity triangles
    for i, n in enumerate(ntw.system.nodes):
        n.kin.plot(n.geo, FONTSIZE)
        plt.title(f'Node number {i}')

    _, smass0 = PT_EOS(nodes[0].tot.p, nodes[0].tot.T)  # pyright:ignore
    # Plot entropy rise
    fig, ax = plt.subplots()
    ax.set_title('Entropy rise')
    for i, n in enumerate(ntw.system.nodes):
        rho, smass = PT_EOS(n.tot.p, n.tot.T)  # pyright:ignore
        # Plot entropy distributions
        ax.plot(n.geo.rr, smass - smass0)  # pyright:ignore

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
        midspan_idx = ntw.system.spanwise_stations // 2
        inlet_angle = nodes[0].geo.metal_angle[midspan_idx]  # pyright:ignore
        outlet_angle = nodes[1].geo.metal_angle[midspan_idx]  # pyright:ignore
        chord_ax = nodes[1].geo.chord_ax[midspan_idx]  # pyright:ignore
        pitch = nodes[1].geo.pitch[midspan_idx]  # pyright:ignore
        n_blades = 3  # blades to plot

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

if PLOTS:
    plt.show()
else:
    plt.close('all')
