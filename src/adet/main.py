# === IMPORTS
import logging
from typing import Literal

import matplotlib.pyplot as plt
import optimistix as optx
import jax
import numpy as np
from itertools import accumulate
import casadi as cs

from adet.assembly import CasadiSystem
from adet.components import ComponentNetwork
from adet.diagnostics import SystemDiagnostics
from adet.components.blade_row import plot_from_nodes
from adet.fluid.settings import FluidSettings
from adet.config_main import real_model, ideal_model, inlet, row1, row2, row3, row4

# Tooling
from adet.tools.iter import grouper
from adet.tools.loggers import setup_logger
from adet.tools.context import suppress_output


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
NUM_SPAN = 11

# Thermodynamic model

SCALED = True
SOLVER_LSTSQ = optx.BestSoFarLeastSquares(
    optx.LevenbergMarquardt(1e-2, 1e-3),
)
SOLVER_NEWTON = optx.Newton(1e-8, 1e-10)

# === SYSTEM DEFINITION

settings = FluidSettings(
    model=ideal_model,
    update_variables=('p', 'T', 'hmass', 'smass', 'rhomass'),
    update_length=2,
)


# Create network
ntw = ComponentNetwork(
    settings,
    inlet,
    CasadiSystem(spanwise_stations=1),  # Backend
    *[
        row1,
        # row2,
        # row3,
        # row4,
    ],
)

ntw.system.add_global_constraints(
    {
        'oth': {
            'cpmassid': 1004,
            'cvmassid': 700,
            'T_ref': 1,
            'p_ref': 1,
            'Cd_profile': 0.002,
            'x_by_camb_len_A': 0.375,  # First profile coord
            'x_by_camb_len_B': 0.675,  # First profile coord
        }
    }
)


ntw.build_network()

n0 = ntw.system.nodes[0]
n1 = ntw.system.nodes[1]

x0 = ntw.system.get_initial_guess()


# === CASADI VERSION - Function Extraction + Solution
def solve_casadi_sys(
    system: CasadiSystem,
    method: Literal['newton', 'nlpsol'],
    manual_guess={},
):
    x0 = system.get_initial_guess(manual_guess)
    knowns_stack = system.get_scaled_constraints()
    res_func_casadi = system.make_residual_function()
    free_args_symbols = cs.vertcat(*system.free_args_sym)

    res_expr_partial = res_func_casadi(
        free_args_symbols,  # Unknowns -> Symbols
        knowns_stack.flatten(),  # Knowns -> Numerical values
    )

    # diagn = SystemDiagnostics(system, knowns_stack)

    rootfind_problem = {
        'x': free_args_symbols,
        'g': res_expr_partial,
    }

    # Newton-Raphson solver
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
                'ipopt.print_level': 1,
                'ipopt.max_iter': 100,
                # Need that, the eos does not have an hessian
                #   (jah)
                'ipopt.hessian_approximation': 'limited-memory',
            },
        },
    )

    # with suppress_output():
    logger.info('Solving the system...')
    match method:
        case 'newton':
            sol = G_newt(x0.flatten(), 0.0)
        case 'nlpsol':
            sol = G_nlp(x0.flatten(), 0.0)

    return sol


sol = solve_casadi_sys(ntw.system, 'nlpsol')

sol_dict = ntw.system.solution_to_dict(sol.toarray())

# Use midspan as precursor
sys_multi = ntw.system.copy()
sys_multi.spanwise_stations = NUM_SPAN
sys_multi.build(SCALED)

sol_multi = solve_casadi_sys(sys_multi, 'nlpsol', sol_dict)

# Overwrite
sol = sol_multi
ntw.system = sys_multi


num_args = len(ntw.system.free_args)
ntw.system.write_solution_to_nodes(np.array(sol).reshape(num_args, -1))


PLOTS = True
if PLOTS:
    FONTSIZE = 26
    FONTDICT = {'fontsize': FONTSIZE}

    for idx, n in enumerate(ntw.system.nodes):
        n.kin.plot()
        plt.title(f'Node number {idx}')

    plt.tick_params(labelsize=FONTSIZE / 1.5 // 1)

    num_nodes = range(len(ntw.system.nodes))

    fig, ax = plt.subplots()

    ax.axis('equal')
    # ax.set_ylim(0.0, 1.2)
    ax.set_ylabel('radius [m]', {'fontsize': 18})
    ax.set_xlabel('axial  [m]', {'fontsize': 18})
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
        offset += ax_chord

    plt.show()
else:
    plt.close('all')


print(f'\n\n\n################# NODE 0 #################\n{n0}')
print(f'\n\n\n################# NODE 1 #################\n{n1}')
