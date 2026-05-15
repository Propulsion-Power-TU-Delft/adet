from adet.tools.plotting import plot_velocity_triangles
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
        n0.tot.Pressure: 18.1e5,  # This is before the stator!
        n0.tot.Temperature: Quantity(300, 'degC'),
        n0.oth.CumMassFlow: 0.132,
        n0.kin.FlowAngleAbs: Quantity(60, 'deg'),
        # Geometrical
        n0.geo.Height: Quantity(2, 'mm'),
        n0.geo.Rmid: Quantity(30, 'mm'),
        n0.geo.MeridionalAngle: Quantity(-90, 'deg'),
    }
)

casing = Shaft(
    Quantity(0, 'rpm'),
    is_constrained=True,
)
shaft = Shaft(
    Quantity(-98100, 'rpm'),
    is_constrained=True,
)


stator = BladeRow(
    'nozzle',
    row_type='stator',
    bound_cond={
        n1.geo.Height: Quantity(2, 'mm'),
        n1.geo.Rmid: Quantity(33, 'mm'),
        n1.geo.MeridionalAngle: Quantity(-90, 'deg'),
        n0.geo.BldThick: 0.0,
        n1.geo.BldThick: 0.0,
        n1.geo.MetalAngle: Quantity(70, 'deg'),
        # *** Blades
        n1.geo.NumBlades: 12,
        n1.geo.ChordAx: Quantity(0.01, 'mm'),
    },
    shaft=casing,
    extra_equations={
        IsentropicLink(): (0, 1),
        ZeroDeviation(): 0,  # No incidence
    },
)

rotor = BladeRow(
    'impeller',
    row_type='rotor',
    bound_cond={
        # *** Inlet
        n0.geo.BldThick: 0.0,
        # *** Outlet
        n1.geo.Height: Quantity(12, 'mm'),
        n1.geo.Rmid: Quantity(20, 'mm'),
        n1.geo.MeridionalAngle: Quantity(0, 'deg'),
        n1.kin.FlowAngleAbs: Quantity(0, 'deg'),
        n1.geo.BldThick: 0.0,
        # *** Blades
        n1.geo.NumBlades: 13,
        n1.geo.ChordAx: Quantity(10.0, 'mm'),
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
    [
        # stator,
        rotor,
    ],
)

ntw.build()

rtfn = ntw.system.make_rootfinder('ipopt', opts={'error_on_fail': False})
x0 = ntw.system.get_scaled_guess(fallback=0.1)
kn = ntw.system.get_scaled_constraints()
bnd = ntw.system.get_arguments_bounds(
    # {n1.kin.Mach: (1.1, 10.0)},
)
sol = solve_root_problem(rtfn, x0, kn, suppress_output=False)

sol_dct = ntw.system.sol_to_dict(sol)

plain_bcs = ntw.system._constraint_manager.get_plain_bc_dict()

all_data = {**plain_bcs, **sol_dct}

fig, ax = plt.subplots()

sta_geom = RowGeometry(
    float(all_data[n0.geo.Rmid][0]),
    float(all_data[n1.geo.Rmid][0]),
    float(all_data[n0.geo.Height][0]),
    float(all_data[n1.geo.Height][0]),
    float(all_data[n0.geo.MeridionalAngle][0]),
    float(all_data[n1.geo.MeridionalAngle][0]),
    float(all_data[n1.geo.ChordAx][0]),
)

# rot_geom = RowGeometry(
#     float(all_data[n2.geo.Rmid][0]),
#     float(all_data[n3.geo.Rmid][0]),
#     float(all_data[n2.geo.Height][0]),
#     float(all_data[n3.geo.Height][0]),
#     float(all_data[n2.geo.MeridionalAngle][0]),
#     float(all_data[n3.geo.MeridionalAngle][0]),
#     float(all_data[n3.geo.ChordAx][0]),
# )

ax.set_aspect('equal')
sta_geom.plot_meridional_profile(color='k', ax=ax)
# rot_geom.plot_meridional_profile(color='k', ax=ax)

fig, ax = plt.subplots(1, 2)
plot_velocity_triangles(
    all_data[n0.kin.V_tan],
    all_data[n0.kin.W_mer],
    all_data[n0.kin.W_tan],
    all_data[n0.kin.BladeSpeed],
    all_data[n0.geo.RDistr],
    ax[0],
)
plot_velocity_triangles(
    all_data[n1.kin.V_tan],
    all_data[n1.kin.W_mer],
    all_data[n1.kin.W_tan],
    all_data[n1.kin.BladeSpeed],
    all_data[n1.geo.RDistr],
    ax[1],
)

plt.show()
