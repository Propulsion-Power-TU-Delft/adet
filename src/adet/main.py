# === IMPORTS
# Standard library
from copy import deepcopy
import logging
from typing import Literal

# External libraries
import matplotlib.pyplot as plt
import jax
import numpy as np
import casadi as cs

# Network build
from adet.assembly import CasadiSystem
from adet.components import ComponentNetwork
from adet.equations.definitions import DegreeOfReaction, RepeatedStage
from adet.equations.fundamental import ForcedVortexDistribution, FreeVortexDistribution
from adet.fluid.settings import FluidSettings
from adet.fluid import CasadiEoS
import CoolProp as cp

# Objects Configuration => MODIFY CONFIG FILE TO SET BOUNDARY CONDITIONS
from adet.config_main import real_model, ideal_model, inlet, row0

# Tooling and utils
from adet.losses.base_loss import LossModel
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
NUM_SPAN = 3
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
    *[row0],
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


# === CASADI VERSION - Function Extraction + Solution
def solve_casadi_sys(
    system: CasadiSystem,
    method: Literal['newton', 'nlpsol'],
    manual_guess={},
    overwrite_known=[],
):
    x0 = system.get_initial_guess(manual_guess)
    knowns_stack = system.get_scaled_constraints()

    if overwrite_known:
        for ow in overwrite_known:
            knowns_stack[ow[0]] = ow[1]

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

    # IPOPT solver
    G_nlp = cs.rootfinder(
        'nlpsol_roots',
        'nlpsol',
        rootfind_problem,
        {
            'error_on_fail': True,
            'nlpsol': 'ipopt',
            'nlpsol_options': {
                'ipopt.print_level': 10,
                'ipopt.max_iter': 100,
                # Need the limited-memory, approx (quasi-newton)
                # the eos does not have an hessian
                'ipopt.hessian_approximation': 'limited-memory',
                # 'ipopt.jacobian_approximation': 'finite-difference-values',
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


sol = solve_casadi_sys(ntw.system, 'nlpsol', {})

plt.show()

num_args = len(ntw.system.free_args)
ntw.system.write_solution_to_nodes(np.array(sol).reshape(num_args, -1))


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
