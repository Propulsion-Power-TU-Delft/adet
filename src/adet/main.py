# === IMPORTS
from dataclasses import dataclass
from typing import Iterable
import logging

import matplotlib.pyplot as plt
import optimistix as optx
import jax
import numpy as np
import casadi as cs
from pint.facets.plain import PlainQuantity

from adet.assembly import CasadiSystem
from adet.fluid.settings import FluidSettings, IdealGasModel
from adet.registries import DefaultUnitsRegistry
from adet.tools.loggers import setup_logger
from pint import Quantity

# Equations
from adet.equations.fundamental import (
    EulerEquation,
    MassConservation,
    Kinematics,
    MassAreaRelation,
    MeridionalUniform,
    TotalStaticMatching,
)

from adet.equations.linkers import SpeedLinker, ComponentLinker
from adet.equations.simplelosses import PercentageEntropyLoss
from adet.equations.ideal_gas import IdealStcEos, IdealTotEos, IdealRltEos
from adet.tools.iter import grouper

logger = logging.getLogger(__name__)
jax.config.update('jax_enable_x64', True)

setup_logger(
    logger,
    logging.DEBUG,
    logging.DEBUG,
    suppress_modules=['matplotlib', 'jax'],
    banned_keywords=['STREAM', 'findfont', 'sBIT'],
)

# Disable verbose jax debug logs that somehow elude
# the logging filter I set up for it
logging.getLogger('jax').setLevel(logging.WARNING)


@dataclass
class Shaft:
    omega: float | PlainQuantity
    rows: Iterable[int]

    def __post_init__(self):
        if isinstance(self.omega, float):
            self.omega = Quantity(self.omega, 'rad/s')


# === SETTINGS
NUM_SPAN = 1
NUM_ROWS = 4
SHAFTS = [
    Shaft(
        omega=0.0,
        rows=range(0, NUM_ROWS, 2),
    ),
    Shaft(
        omega=200.0,
        rows=range(1, NUM_ROWS, 2),
    ),
]

SCALED = True
SORT_BY_LINEARITY = False

SOLVER_NLP = 'ipopt'
SOLVER_LSTSQ = optx.BestSoFarLeastSquares(
    optx.LevenbergMarquardt(1e-2, 1e-3),
)
SOLVER_NEWTON = optx.Newton(1e-8, 1e-10)

# === SYSTEM DEFINITION
system_cas = CasadiSystem()
model = IdealGasModel(287.0, 1.4)
fluid_settings = FluidSettings(model)
system_cas.settings = fluid_settings

# === Define custom units
units_reg = DefaultUnitsRegistry()
units_reg['cpmassid'] = 'J/(kg*K)'
units_reg['cvmassid'] = 'J/(kg*K)'


# Nomenclature
# ~~~~~~~~~~~~
#        _________________________________
#          |         |        _________
#          |         |       |         |
#   V      |         |       |         |
#  -->   0 |  ROW 0  | 1   2 |  ROW 1  | 3
#          |         |       |         |
#        __|_________|_______|_________|__
#        /////////////////////////////////
#        \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#        /////////////////////////////////
#       _ . _ . _ . _ . _ . _ . _ . _ . _ . _
#
# * row couples : (0, 1), (2, 3), ...
# * link couples : (1, 2), ...
# * shafts: {shaft number: iterator[row(s) connected to shaft]}

row_couples = list(
    grouper(range(2 * NUM_ROWS), 2, incomplete='strict'),
)

link_couples = list(
    grouper(range(1, 2 * NUM_ROWS), 2, incomplete='ignore'),
)

# === EQUATIONS ===

# Fundamental equations
for nodes in row_couples:
    system_cas.add_equation(EulerEquation(), nodes)
    system_cas.add_equation(MassConservation(), nodes)
    system_cas.add_equation(PercentageEntropyLoss(0.0), nodes)

# Compatibility between rows
if link_couples:
    for nodes in link_couples:
        system_cas.add_equation(ComponentLinker(), nodes)

# Single node relations
for node in range(2 * NUM_ROWS):
    system_cas.add_equation(MassAreaRelation(), node)
    system_cas.add_equation(Kinematics(), node)
    system_cas.add_equation(MeridionalUniform(), node)
    system_cas.add_equation(TotalStaticMatching(), node)

    system_cas.add_equation(IdealStcEos(), node)
    system_cas.add_equation(IdealTotEos(), node)
    system_cas.add_equation(IdealRltEos(), node)

# Speed links
for nodes in row_couples:
    system_cas.add_equation(SpeedLinker(), (nodes[0], nodes[0]))
    system_cas.add_equation(SpeedLinker(), (nodes[0], nodes[1]))

# Add rotating speed boundary conditions
for shaft in SHAFTS:
    for row in shaft.rows:
        node_number = 2 * row
        system_cas.boundary_conditions[node_number]['kin']['omega'] = shaft.omega

# === BOUNDARY CONDITIONS ===
# INLET
CONSTR0 = {
    'kin': {
        'meridional_angle': 0.0,
        'V': 30.0,
        'rmid': 0.3,
        'alpha': Quantity(0, 'deg'),
        'height': 0.05,
    },
    'tot': {
        'p': 1.5e5,
        'T': 476,
    },
}

# ROW OUTLETS
OUTLET_CONSTRAINTS = {
    1: {
        'kin': {
            'meridional_angle': 0.0,
            'alpha': Quantity(65, 'deg'),
            'rmid': 0.3,
            'height': 0.05,
        },
    },
    3: {
        'kin': {
            'meridional_angle': 0.0,
            'alpha': Quantity(0, 'deg'),
            'rmid': 0.3,
            'height': 0.05,
        },
    },
    5: {
        'kin': {
            'meridional_angle': 0.0,
            'alpha': Quantity(65, 'deg'),
            'rmid': 0.3,
            'height': 0.05,
        },
    },
    7: {
        'kin': {
            'meridional_angle': 0.0,
            'alpha': Quantity(-10, 'deg'),
            'rmid': 0.3,
            'height': 0.05,
        },
    },
}
system_cas.add_boundary_conditions(CONSTR0, 0)
for node in range(1, 2 * NUM_ROWS, 2):
    system_cas.add_boundary_conditions(OUTLET_CONSTRAINTS[node], node)


# *** Compare speed to casadi formulation
system_cas.build(SCALED)

y0_midspan = system_cas.get_initial_guess()
knowns_stack = system_cas.get_scaled_constraints()

res_func_casadi = system_cas.make_residual_function()

free_args_symbols = system_cas.free_args_sym

res_func_partial = res_func_casadi(
    free_args_symbols,
    np.array(knowns_stack).flatten(),
)

rootfind_problem = {
    'x': free_args_symbols,
    'g': res_func_partial,
}

G_newt = cs.rootfinder(
    'newton_roots',
    'newton',
    rootfind_problem,
    {'print_iteration': True},
)
G_nlp = cs.rootfinder(
    'nlpsol_roots',
    'nlpsol',
    rootfind_problem,
    {'nlpsol': SOLVER_NLP},
)

logger.info('Trying solution with Newton method...')
sol_newt = G_newt(y0_midspan.flatten(), 0.0)

if np.isnan(sol_newt).any():
    logger.info('Newton method failed, trying {SOLVER_NLP}')
    sol_newt = G_newt(G_nlp(y0_midspan.flatten(), 0.0), 0.0)

system_cas.write_solution_to_nodes(np.array(sol_newt).reshape(system_cas.num_args, -1))

system_cas_spanwise = system_cas.copy()


RUN_JAX = False
if RUN_JAX:
    system_jax = system_cas.to_jax()
    system_jax.build(SCALED)

    res_func_midspan = system_jax.make_residual_function()

    def res_partial(x, aux):
        return res_func_midspan(x, knowns_stack)

    logger.info('Solving system in least square sense...')
    sol_lstsq = optx.root_find(
        res_partial,
        SOLVER_LSTSQ,
        y0_midspan,
    )

    logger.info('Solving system with Newton-Raphson...')
    sol_newton = optx.root_find(
        res_partial, SOLVER_NEWTON, sol_lstsq.value, max_steps=20
    )

    system_jax.write_solution_to_nodes(sol_newton.value)


system_cas.to_symbolic()

PLOTS = True
if PLOTS:
    for idx, n in enumerate(system_cas.nodes):
        n.kin.plot()
        plt.title(f'Node number {idx}')
    plt.show()
else:
    plt.close('all')
