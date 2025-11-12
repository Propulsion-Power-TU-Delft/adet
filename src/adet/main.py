# === IMPORTS
# Standard library
import logging

# External libraries
import matplotlib.pyplot as plt
import jax
import numpy as np
import casadi as cs

# Network build
from adet.assembly import CasadiSystem
from adet.components import ComponentNetwork
from adet.equations.fundamental import (
    FreeVortexDistribution,
    NisRe,
    SimpleRadialEquilibrium,
)
from adet.fluid.settings import FluidSettings

# Objects Configuration => MODIFY CONFIG FILE TO SET BOUNDARY CONDITIONS
from adet.config_main import real_model, inlet, row0

# Tooling and utils
from adet.losses.base_loss import LossModel
from adet.losses.basic import DesignAngle
from adet.losses.profile import DentonProfileLoss
from adet.tools.iter import grouper
from adet.tools.loggers import setup_logger
from adet.components.blade_row import plot_from_nodes
from adet.tools.context import suppress_output
from adet.diagnostics import SystemDiagnostics


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
NUM_SPAN = 25
NUM_STAGES = 6
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
    row0,
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


def solve_problem(rootfinder, guess, knowns):
    """Solve rootfinding problem"""
    with suppress_output():
        logger.info('Solving the system...')

        sol = np.array(
            rootfinder(
                guess.flatten(),
                knowns.flatten(),
            )
        )

        return sol


x0_is = ntw.system.get_initial_guess()
kn_is = ntw.system.get_scaled_constraints()
rootfinder_is = ntw.system.make_rootfinder('nlpsol')
sol_is = solve_problem(rootfinder_is, x0_is, kn_is)
ntw.system.write_solution_to_nodes(sol_is)
sol_is_dict = ntw.system.solution_to_dict(sol_is)

# *** OFF DESIGN
sys_is_off = ntw.system.copy()
sys_is_off.spanwise_stations = NUM_SPAN

sys_is_off.remove_equation_type(DesignAngle)
sys_is_off.add_equation(NisRe(), 1)

# Fix the metal angle
sys_is_off.boundary_conditions[0]['geo']['metal_angle'] = ntw.system.nodes[
    0
].geo.metal_angle
sys_is_off.boundary_conditions[1]['geo']['metal_angle'] = ntw.system.nodes[
    1
].geo.metal_angle
# Change the inlet angle

sys_is_off.boundary_conditions[0]['kin']['alpha'] = 0.0
sys_is_off.boundary_conditions[1]['kin'].pop('alpha')

sys_is_off.build(SCALED)
rootfinder_is_off = sys_is_off.make_rootfinder('nlpsol')

x0_is_off = sys_is_off.get_initial_guess(sol_is_dict).flatten()
kn_is_off = sys_is_off.get_scaled_constraints().flatten()
sol_is_off = solve_problem(rootfinder_is_off, x0_is_off, kn_is_off)

# # *** Get residual function and jacobian for debugging
# res_func = sys_is_off.make_residual_function()
# initial_residual = res_func(x0_is_off, kn_is_off)
# jac_func = res_func.jacobian()
# jacobian = np.array(
#     jac_func(x0_is_off, kn_is_off, initial_residual)[0],
# )


# Copy the system
# sys_design_loss = ntw.system.copy()

# # Modify spanwise stations and equations
# sys_design_loss.spanwise_stations = NUM_SPAN
# sys_design_loss.remove_equation_type(LossModel)
# sys_design_loss.add_equation(DentonProfileLoss(real_model), (0, 1))

# # Add the system back to the network (just for access), rebuild it
# ntw.system = sys_design_loss
# ntw.system.build(SCALED)

# rootfinder_design = ntw.system.make_rootfinder('nlpsol')
# # Use primal solution
# x0_design = ntw.system.get_initial_guess(sol_is_dict)
# kn_design = ntw.system.get_scaled_constraints()

# with suppress_output():
#     sol_full = np.array(
#         rootfinder_design(
#             x0_design.flatten(),
#             kn_design.flatten(),
#         )
#     )

# sys_off_design = sys_design_loss.copy()


num_args = len(ntw.system.free_args)

ntw.system = sys_is_off
sol = sol_is_off
ntw.system.write_solution_to_nodes(sol)


if PLOTS:
    FONTSIZE = 18
    FONTDICT = {'fontsize': FONTSIZE}

    for i, n in enumerate(ntw.system.nodes):
        n.kin.plot(n.geo, FONTSIZE)
        plt.title(f'Node number {i}')

    plt.tick_params(labelsize=FONTSIZE / 1.5 // 1)

    num_nodes = range(len(ntw.system.nodes))

    fig, ax = plt.subplots()

    ax.axis('equal')
    # ax.set_ylim(0.0, 1.2)
    ax.set_ylabel('radius [m]', {'fontsize': 18})
    ax.set_xlabel('axial  [m]', {'fontsize': 18})
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

    ax.plot([0.0, offset], [0.0, 0.0], color='r', linestyle='dashdot', linewidth=2.5)


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

if PLOTS:
    plt.show()
else:
    plt.close('all')
