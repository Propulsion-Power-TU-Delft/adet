# === IMPORTS
import logging

import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity

from adet.assemblers import CasadiSystem
from adet.components import BladeRow, Inlet
from adet.components.connections import Shaft
from adet.components.network import ComponentNetwork
from adet.equations.base_equation import EquationBase
from adet.equations.fundamental import (  # noqa: F401
    ForcedVortexDistribution,
    FreeVortexDistribution,
)
from adet.equations.utils import get_midspan_idx, safe_abs
from adet.fluid.settings import FluidSettings
from adet.fluid.ideal_eos import IdealGasState
from adet.losses.basic import IsentropicLink, ZeroDeviation
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger
from adet.tools.plotting import plot_velocity_triangles, setup_mpl
from adet.variables import NodeVariables
from adet.varspec import VarSpec

n0 = NodeVariables(0)
n1 = NodeVariables(1)
n2 = NodeVariables(2)
n3 = NodeVariables(3)
logger = logging.getLogger(__name__)
setup_logger(logger)

abs_state = DebugAbstractState('HEOS', 'Air')
abs_state = IdealGasState(1.4, 287, 2e-5)


DeflectionMidspan = VarSpec('defl_mid', 'radians', node=1, scalar=True)


class MidspanDeflection(EquationBase):
    def residual(
        self,
        defl: DeflectionMidspan.Hint,
        b0: n0.kin.FlowAngleRel.Hint,
        b1: n1.kin.FlowAngleRel.Hint,
    ):
        midspan = get_midspan_idx(b0)
        return defl - safe_abs(b1[midspan] - b0[midspan])


INLET_PTOT = 10e5
INLET_TEMPERATURE = 600
MERIDIONAL_VEL = 50

# *** Inlet conditions
inlet = Inlet(
    boundary_conditions={
        # *** Inlet total conditions
        n0.tot.Pressure: INLET_PTOT,
        n0.tot.Temperature: INLET_TEMPERATURE,
        # *** Inlet geometry
        n0.kin.V_mer: MERIDIONAL_VEL,
        n0.kin.FlowAngleAbs: 0.0,
        n0.geo.Rmid: 0.1,
        n0.geo.HubTipRatio: 0.577,
        n0.geo.MeridionalAngle: Quantity(0, 'deg'),
    }
)


casing = Shaft(0, is_constrained=True)
shaft = Shaft(1000, is_constrained=False)


rotor = BladeRow(
    name='stator',
    shaft=shaft,
    bound_cond={
        n1.geo.MeridionalAngle: Quantity(0, 'deg'),
        # Blade thickness
        n0.geo.BldThick: 0.0,
        n1.geo.BldThick: 0.0,
        n1.geo.NumBlades: 10.0,
        # Blade chord
        n1.geo.ChordAx: 0.05,
        DeflectionMidspan: Quantity(60, 'deg'),
        n1.kin.Beta_mid: Quantity(0, 'deg'),
        n1.kin.V_mer: MERIDIONAL_VEL,
    },
    extra_equations={
        ZeroDeviation(): 0,  # No incidence (design)
        ZeroDeviation(): 1,  # Flow angle = Metal angle
        IsentropicLink(): (0, 1),
        MidspanDeflection(): (0, 1),
        # ForcedVortexDistribution(): 1,
        FreeVortexDistribution(): 1,
    },
    constant_variables=[n0.geo.Rmid],  # Constant mean radius
)

stator = BladeRow(
    name='rotor',
    shaft=casing,
    bound_cond={
        n1.geo.MeridionalAngle: Quantity(0, 'deg'),
        # Blade thickness
        n0.geo.BldThick: 0.0,
        n1.geo.BldThick: 0.0,
        n1.geo.NumBlades: 10.0,
        # Blade chord
        n1.geo.ChordAx: 0.05,
        n1.kin.FlowAngleAbs: 0.0,
        n1.kin.V_mer: MERIDIONAL_VEL,
    },
    extra_equations={
        ZeroDeviation(): 0,  # No incidence (design)
        ZeroDeviation(): 1,  # Flow angle = Metal angle
        IsentropicLink(): (0, 1),
    },
    constant_variables=[n0.geo.Rmid],  # Constant mean radius
)


stator.set_spanwise_constant(n0.geo.HDistr)
rotor.set_spanwise_constant(n1.geo.HDistr)

# *** Network buildup
fluid_settings = FluidSettings(
    fluid_state=abs_state,
    update_variables=(n0.stc.Pressure, n0.stc.Temperature),
)

ntw = ComponentNetwork(
    fluid_settings,
    inlet,
    CasadiSystem(num_span=3),
    [
        rotor,
        stator,
    ],
)

ntw.build()
input('Continue?')

# *** Solution
x0 = ntw.system.get_guess(fallback=0.8)
kn = ntw.system.get_boundary_conds()
bnd = ntw.system.get_bounds(
    {
        # n0.geo.Chord.Glob: (0.0, 1e5),
        n0.stc.Pressure.Glob: (10.0, 1.3 * INLET_PTOT),
        n0.stc.Temperature.Glob: (60.0, 1.3 * INLET_TEMPERATURE),
        n0.kin.RelMach.Glob: (0.0, 1.8),
    },
    ignore_defaults=False,
)


# Ipopt rootfinding
rtfn = ntw.system.make_rootfinder(
    'ipopt',
    {
        'error_on_fail': False,
        'ipopt.max_wall_time': 10,
    },
)
sol = solve_root_problem(rtfn, x0, kn, bnd, suppress_output=False)

# kinsol
rtfn = ntw.system.make_rootfinder('kinsol')
sol = solve_root_problem(rtfn, sol, kn, suppress_output=True)

data = ntw.system.sol_to_dict(sol)

#  #  #  #  #  #  #  Plots and prints #  #  #  #  #  #  #
print(f'Effective HTR {data[n1.geo.RDistr][0] / data[n1.geo.RDistr][-1]}')

setup_mpl({'font.family': 'EB Garamond', 'font.size': 20})

fig, axs = plt.subplots(1, 3, figsize=(16, 8))
fig.tight_layout()
axs = axs.flatten()

[a.set_aspect('equal') for a in axs]

plot_velocity_triangles(
    data[n0.kin.V_tan],
    data[n0.kin.V_mer],
    data[n0.kin.BladeSpeed],
    data[n0.geo.RDistr],
    axs[0],
)
plot_velocity_triangles(
    data[n1.kin.V_tan],
    data[n1.kin.V_mer],
    data[n1.kin.BladeSpeed],
    data[n1.geo.RDistr],
    axs[1],
)
plot_velocity_triangles(
    data[n3.kin.V_tan],
    data[n3.kin.V_mer],
    data[n3.kin.BladeSpeed],
    data[n3.geo.RDistr],
    axs[2],
)

plot_velocity_triangles(
    data[n2.kin.V_tan],
    data[n2.kin.V_mer],
    data[n2.kin.BladeSpeed],
    data[n2.geo.RDistr],
    axs[2],
)
react_degree = (data[n1.stc.Enthalpy] - data[n0.stc.Enthalpy]) / (
    data[n3.tot.Enthalpy] - data[n0.tot.Enthalpy]
)
work_coeff = (data[n3.tot.Enthalpy] - data[n0.tot.Enthalpy]) / (
    data[n0.kin.BladeSpeed] ** 2
)
flow_coeff = (data[n3.kin.V_mer]) / (data[n0.kin.BladeSpeed])
deflection = np.degrees(data[n1.kin.FlowAngleRel] - data[n0.kin.FlowAngleRel])

print(f'Reaction degree is {react_degree}')
print(f'Work coefficient is {work_coeff}')
print(f'Flow coefficient is {flow_coeff}')
print(f'Deflections are {deflection}')


fig.show()

#  #  #  #  #  #  #  #  #  #  #  #  #  #
