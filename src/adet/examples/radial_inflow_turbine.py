import logging

import matplotlib.pyplot as plt
from pint import Quantity

from adet.assembly import CasadiSystem
from adet.components.blade_row import BladeRow, RowGeometry
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork
from adet.equations.variables import NodeVariables
from adet.fluid.settings import ExternalFluidModel, FluidSettings
from adet.losses.basic import ZeroDeviation, IsentropicLink
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger

logger = logging.getLogger(__name__)
setup_logger(logger)

n0 = NodeVariables(0)
n1 = NodeVariables(1)
n2 = NodeVariables(2)
n3 = NodeVariables(3)

inl = Inlet(
    {
        # n0.tot.Pressure: 18.1e5, # This is before the stator!
        n0.tot.Temperature: Quantity(300, 'degC'),
        n0.oth.CumMassFlow: 0.132,
        n0.kin.FlowAngleAbs: Quantity(65, 'deg'),
        n0.kin.Mach: 2.2,
    }
)

shaft = Shaft(
    Quantity(-98100, 'rpm'),
    is_constrained=True,
)

rotor = BladeRow(
    'impeller',
    row_type='rotor',
    bound_cond={
        # *** Inlet
        n0.geo.Height: Quantity(2, 'mm'),
        n0.geo.Rmid: Quantity(38, 'mm'),
        n0.geo.MeridionalAngle: Quantity(-90, 'deg'),
        n0.geo.BldThick: 0.0,
        # *** Outlet
        n1.geo.Height: Quantity(12, 'mm'),
        n1.geo.Rmid: Quantity(20, 'mm'),
        n1.geo.MeridionalAngle: Quantity(0, 'deg'),
        n1.kin.FlowAngleAbs: Quantity(0, 'deg'),
        n1.geo.BldThick: 0.0,
        # *** Blades
        n1.geo.NumBlades: 10,
        n1.geo.ChordAx: Quantity(13, 'mm'),
    },
    shaft=shaft,
    extra_equations={
        IsentropicLink(): (0, 1),
        ZeroDeviation(): 0,  # No incidence
    },
)

abs_state = DebugAbstractState('REFPROP', 'MM')
fluid_model = ExternalFluidModel(abs_state)

fluid_settings = FluidSettings(
    fluid_model,
    update_variables=(
        n0.stc.Pressure.Glob,
        n0.stc.Temperature.Glob,
    ),
)

ntw = ComponentNetwork(
    fluid_settings,
    inl,
    CasadiSystem(1),
    [rotor],
)

ntw.build()

rtfn = ntw.system.make_rootfinder('ipopt')
x0 = ntw.system.get_scaled_guess(fallback=0.5)
kn = ntw.system.get_scaled_constraints()

sol = solve_root_problem(rtfn, x0, kn, suppress_output=True)
sol_dct = ntw.system.sol_to_dict(sol)

plain_bcs = ntw.system._constraint_manager.get_plain_bc_dict()

all_data = {**plain_bcs, **sol_dct}


geom = RowGeometry(
    float(all_data[n0.geo.Rmid][0]),
    float(all_data[n1.geo.Rmid][0]),
    float(all_data[n0.geo.Height][0]),
    float(all_data[n1.geo.Height][0]),
    float(all_data[n0.geo.MeridionalAngle][0]),
    float(all_data[n1.geo.MeridionalAngle][0]),
    float(all_data[n1.geo.ChordAx][0]),
)

fig, ax = plt.subplots()
ax.set_aspect('equal')
lines = geom.plot_meridional_profile(color='k', ax=ax)
fig.show()
