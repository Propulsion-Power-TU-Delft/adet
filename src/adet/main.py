# === IMPORTS
import logging
from itertools import accumulate
from typing import Literal

import matplotlib.pyplot as plt
import optimistix as optx
import jax
import numpy as np
from pint import Quantity
import casadi as cs

from adet.assembly import CasadiSystem
from adet.components.network import ComponentNetwork
from adet.diagnostics import SystemDiagnostics
from adet.registries import DefaultUnitsRegistry, GuessRegistry, ScalingRegistry
from adet.tools.context import suppress_output
from adet.tools.coolprop_utils import CountingAbstractState
from adet.tools.iter import grouper

from adet.fluid.settings import AbstractStateModel, FluidSettings, IdealGasModel

from adet.losses.basic import PercentageEntropyLoss

from adet.tools.loggers import setup_logger

from adet.components.connections import Inlet, Shaft
from adet.components.blade_row import BladeRow, plot_from_nodes

from adet.equations.nondimensional import (
    StaticTotalPressRatio,
    WorkCoefficient,
    FlowCoefficient,
    SizeParameter,
    SpecificSpeed,
)
from adet.equations.definitions import AngleDeflection

# Equations
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
NUM_SPAN = 5

# Thermodynamic model
MODEL: Literal['ideal', 'abstate'] = 'abstate'

SCALED = True
SOLVER_LSTSQ = optx.BestSoFarLeastSquares(
    optx.LevenbergMarquardt(1e-2, 1e-3),
)
SOLVER_NEWTON = optx.Newton(1e-8, 1e-10)

# === SYSTEM DEFINITION
match MODEL:
    case 'ideal':
        model = IdealGasModel(287.0, 1.4)
    case 'abstate':
        # This counts the number of updates in an attribute
        abs_state = CountingAbstractState('HEOS', 'Air')
        model = AbstractStateModel(abs_state)

settings = FluidSettings(
    model=model,
    update_variables=('p', 'T', 'hmass', 'smass', 'rhomass'),
    update_length=2,
)


# Set custom units and defaults
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
        'sizeParameter': 'meters',
    }
)

# Set default values for scales and guesses to 1.0
_scl_reg.set_fallback_value(1.0)
_gss_reg.set_fallback_value(1.0)

# *** Shafts
static_shaft = Shaft(0.0)
rotating_shaft = Shaft(Quantity(800, 'rpm'))

# *** Constraints
CONSTR0 = {
    'kin': {
        'meridional_angle': Quantity(0, 'deg'),
        'alpha': Quantity(25, 'deg'),
        'rmid': 0.5,
        'height': 0.2,
    },
    'tot': {
        'p': 3e5,
        'T': 500,
    },
    'oth': {
        'flowCoeff': 1.5,
        # 'cum_massflow': 100,
    },
}

CONSTR1 = {
    'kin': {
        'meridional_angle': 0.0,
        # 'alpha': Quantity(0, 'deg'),
        'rmid': 0.55,
        'height': 0.15,
    },
    'stc': {
        'p': 2e5,
    },
    'oth': {
        # 'STratio': 0.8,
        # 'workCoeff': 1.0,
        # These two don't converge
        # 'specificSpeed': 0.4,
        # 'sizeParameter': 0.1,
    },
}

# *** Inlet
inlet = Inlet(CONSTR0)


# *** Blade rows
# STATOR
row1 = BladeRow(
    CONSTR1,
    rotating_shaft,
    loss_models=[
        PercentageEntropyLoss(0.0),
    ],
    extra_equations={
        FlowCoefficient(): 0,
        WorkCoefficient(): (0, 1),
        SpecificSpeed(): (0, 1),
        SizeParameter(): (0, 1),
        StaticTotalPressRatio(): (0, 1),
    },
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

ntw.build_network()

x0 = ntw.system.get_initial_guess()


# === CASADI VERSION - Function Extraction + Solution
USE_CASADI = True
if USE_CASADI:

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
                'nlpsol': 'ipopt',
                'nlpsol_options': {
                    'ipopt.print_level': 0,
                    'ipopt.hessian_approximation': 'limited-memory',  #! Need this
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


# === JAX VERSION
USE_JAX = not USE_CASADI
if USE_JAX:
    sys_jax = ntw.system.to_jax()
    sys_jax.build(SCALED)

    x0 = ntw.system.get_initial_guess()
    knowns_stack = ntw.system.get_scaled_constraints()
    res_func_jax = sys_jax.make_residual_function()

    @jax.jit
    def partial_res(args, aux):
        return res_func_jax(args, knowns_stack)

    flat_func = jax.jit(sys_jax._make_flat_resfunc(knowns_stack))

    sol_lsq = optx.root_find(partial_res, SOLVER_LSTSQ, x0)
    sol_newt = optx.root_find(partial_res, SOLVER_NEWTON, sol_lsq.value)
    sol = sol_newt.value

ntw.system.write_solution_to_nodes(np.array(sol).reshape(ntw.system.num_args, -1))


PLOTS = True
if PLOTS:
    # plt.rcParams.update(
    #     {
    #         'text.usetex': False,
    #         'font.family': 'serif',
    #     }
    # )
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

    chords = [0.2, 0.4, 0.4, 0.6]
    offset = list(
        accumulate([0.0, 0.4, 0.4, 0.35]),
    )
    for n0_idx, n1_idx in grouper(num_nodes, 2, incomplete='ignore'):
        n0 = ntw.system.nodes[n0_idx]
        n1 = ntw.system.nodes[n1_idx]
        lines = plot_from_nodes(
            n0,
            n1,
            chords[n0_idx // 2],
            False,
            offset[n0_idx // 2],
        )

    plt.show()
else:
    plt.close('all')


#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
#                                                        #
#                   PRISTINE UNUSED ROWS                 #
#                                                        #
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
row2 = BladeRow(
    {
        'kin': {
            'meridional_angle': Quantity(0.0, 'deg'),
            'rmid': 1.0,
            'height': 0.3,
        },
        'oth': {
            'workCoeff': 1.1,
            # 'deflection': Quantity(65, 'deg'),
        },
    },
    rotating_shaft,
    loss_models=[
        PercentageEntropyLoss(0.0),
    ],
    extra_equations={
        WorkCoefficient(): (0, 1),
        # AngleDeflection(): (0, 1),
    },
)

row3 = BladeRow(
    {
        'kin': {
            'meridional_angle': Quantity(-70.0, 'deg'),
            # 'alpha': Quantity(65.0, 'deg'),
            'rmid': 0.8,
            'height': 0.35,
        },
        'oth': {
            # 'workCoeff': 1.0,
            'deflection': Quantity(100.0, 'deg'),
        },
    },
    static_shaft,
    loss_models=[
        PercentageEntropyLoss(0.0),
    ],
    extra_equations={
        # WorkCoefficient(): (0, 1),
        AngleDeflection(): (0, 1),
    },
)

row4 = BladeRow(
    {
        'kin': {
            'meridional_angle': Quantity(0.0, 'deg'),
            'rmid': 0.4,
            'height': 0.5,
        },
        'oth': {
            'workCoeff': 1.0,
        },
    },
    rotating_shaft,
    loss_models=[
        PercentageEntropyLoss(0.0),
    ],
    extra_equations={
        WorkCoefficient(): (0, 1),
    },
)
