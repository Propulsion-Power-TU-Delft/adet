from adet.tools.plotting import plot_velocity_triangles
import CoolProp as cp
import logging

import matplotlib.pyplot as plt
from pint import Quantity

from adet.assembly import CasadiSystem
from adet.components.blade_row import BladeRow, RowGeometry, Interspace
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork
from adet.variables import NodeVariables
from adet.fluid.settings import ExternalFluidModel, FluidSettings
from adet.losses.basic import ZeroDeviation, IsentropicLink
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger

logger = logging.getLogger(__name__)
setup_logger(logger)

# |> Stator 0 - 1
n0 = NodeVariables(0)
n1 = NodeVariables(1)
# |> Inters 2 - 3
n2 = NodeVariables(2)
n3 = NodeVariables(3)
# |> Rotor  4 - 5
n4 = NodeVariables(4)
n5 = NodeVariables(5)

inl = Inlet(
    {
        # *** Total conditions
        n0.tot.Pressure: Quantity(18.1, 'bar'),
        n0.tot.Temperature: Quantity(300, 'degC'),
        # ***
        n0.oth.CumMassFlow: 0.132,
        n0.kin.FlowAngleRel: Quantity(65, 'deg'),
        # *** Geometrical
        n0.geo.Height: Quantity(2, 'mm'),
        n0.geo.MeridionalAngle: Quantity(-90, 'deg'),
    }
)

casing = Shaft(
    Quantity(0, 'rpm'),
    is_constrained=True,
)
shaft = Shaft(
    Quantity(98100, 'rpm'),
    is_constrained=True,
)

stator = BladeRow(
    'nozzle',
    row_type='stator',
    bound_cond={
        n1.geo.HeightRatio: 1.0,
        n1.geo.RadiusRatio: 0.75,
        n1.geo.Rmid: Quantity(25.75, 'mm'),
        n1.geo.MeridionalAngle: Quantity(-90, 'deg'),
        # *** Outlet
        n1.kin.FlowAngleAbs: Quantity(78, 'deg'),
        # n1.kin.Mach: 2.2,
        # *** Blades
        n0.geo.BldThick: 0.0,
        n1.geo.BldThick: 0.0,
        n1.geo.NumBlades: 12,
        n1.geo.ChordAx: Quantity(0.01, 'mm'),
    },
    shaft=casing,
    extra_equations={
        IsentropicLink(): (0, 1),
        ZeroDeviation(): 0,  # No incidence
    },
)

interspace = Interspace(
    'intrspc',
    {
        n1.geo.RadiusRatio: 0.96,
        n1.geo.HeightRatio: 1.0,
    },
    extra_equations={
        IsentropicLink(): (0, 1),
    },
)

rotor = BladeRow(
    'impeller',
    row_type='rotor',
    bound_cond={
        # *** Meridional Geometry
        n1.geo.Rhub: Quantity(8.2, 'mm'),
        n1.geo.Height: Quantity(12.3, 'mm'),
        n1.geo.MeridionalAngle: Quantity(0, 'deg'),
        # *** Blade Geometry
        n0.geo.BldThick: 0.0,
        n1.geo.BldThick: 0.0,
        n1.geo.ChordAx: Quantity(10.2, 'mm'),
        n1.geo.NumBlades: 13,
        # *** Outlet condition
        n1.kin.FlowAngleAbs: Quantity(0, 'deg'),
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
        stator,
        interspace,
        # rotor,
    ],
)

ntw.build()

rtfn = ntw.system.make_rootfinder(
    'ipopt',
    opts={'error_on_fail': False},
)
x0 = ntw.system.get_scaled_guess(fallback=0.1)
kn = ntw.system.get_scaled_constraints()
bnd = ntw.system.get_arguments_bounds(
    {
        # WARN: Force the supersonic solution w/ bounds
        n1.kin.Mach: (1.1, 4.0),
        # NOTE: These stabilize massively the solution
        n0.stc.Pressure.Glob: (0.01, 22e5),
        n0.stc.Temperature.Glob: (450, abs_state.Tmax()),
    }
)

sol = solve_root_problem(rtfn, x0, kn, bnd, suppress_output=False)

# Kinsol pass
# rtfn = ntw.system.make_rootfinder('kinsol')
# sol = solve_root_problem(rtfn, sol, kn)

# Merge boundary conditions and solution
sol_dct = ntw.system.sol_to_dict(sol)
plain_bcs = ntw.system._constraint_manager.get_plain_bc_dict()
sol_data = {**plain_bcs, **ntw.system.sol_to_dict(sol)}

# Plotting
fig, ax_mer = plt.subplots()
ax_mer.set_aspect('equal')

sta_geom = RowGeometry(
    float(sol_data[n0.geo.Rmid][0]),
    float(sol_data[n1.geo.Rmid][0]),
    float(sol_data[n0.geo.Height][0]),
    float(sol_data[n1.geo.Height][0]),
    float(sol_data[n0.geo.MeridionalAngle][0]),
    float(sol_data[n1.geo.MeridionalAngle][0]),
    float(sol_data[n1.geo.ChordAx][0]),
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

sta_geom.plot_meridional_profile(color='k', ax=ax_mer)
# rot_geom.plot_meridional_profile(color='k', ax=ax)

fig, axs_tri = plt.subplots(1, 2)
[ax.set_aspect('equal') for ax in axs_tri]

plot_velocity_triangles(
    sol_data[n0.kin.V_tan],
    sol_data[n0.kin.V_mer],
    sol_data[n0.kin.BladeSpeed],
    sol_data[n0.geo.RDistr],
    axs_tri[0],
)
plot_velocity_triangles(
    sol_data[n1.kin.V_tan],
    sol_data[n1.kin.V_mer],
    sol_data[n1.kin.BladeSpeed],
    sol_data[n1.geo.RDistr],
    axs_tri[1],
)

# TODO: Make this automated
abs_state.update(
    cp.PT_INPUTS,
    sol_data[n0.tot.Pressure],
    sol_data[n0.tot.Temperature],
)
ht0 = abs_state.hmass()

abs_state.update(
    cp.PT_INPUTS,
    sol_data[n1.tot.Pressure],
    sol_data[n1.tot.Temperature],
)
ht1 = abs_state.hmass()

print(f'Turbine power {sol_data[n0.oth.CumMassFlow] * (ht0 - ht1)}')

plt.show()
