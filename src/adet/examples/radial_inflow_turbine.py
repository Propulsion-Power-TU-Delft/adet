from adet.tools.plotting import plot_velocity_triangles
import CoolProp as cp
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
        # EITHER
        # n0.stc.Pressure: Quantity(0.7, 'bar'),
        # n0.tot.Pressure: Quantity(18.1, 'bar'),
        n0.kin.Mach: 2.2,
        # ---
        n0.tot.Temperature: Quantity(300, 'degC'),
        n0.oth.CumMassFlow: 0.132,
        n0.kin.FlowAngleAbs: Quantity(45, 'deg'),
        # --- Geometrical
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
    Quantity(98100, 'rpm'),
    is_constrained=True,
)


stator = BladeRow(
    'nozzle',
    row_type='stator',
    bound_cond={
        n1.geo.Rmid: Quantity(33, 'mm'),
        n1.geo.Height: Quantity(2, 'mm'),
        n1.geo.MetalAngle: Quantity(70, 'deg'),
        n1.geo.MeridionalAngle: Quantity(-90, 'deg'),
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

rotor = BladeRow(
    'impeller',
    row_type='rotor',
    bound_cond={
        # *** Inlet
        n0.geo.BldThick: 0.0,
        # *** Outlet
        n1.geo.HeightRatio: 6.5,
        n1.geo.Rmid: Quantity(15, 'mm'),
        n1.geo.MeridionalAngle: Quantity(0, 'deg'),
        n1.geo.MetalAngle: Quantity(-20, 'deg'),
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
bnd = ntw.system.get_arguments_bounds()
sol = solve_root_problem(rtfn, x0, kn, suppress_output=False)

# Kinsol pass
# rtfn = ntw.system.make_rootfinder('kinsol')
# sol = solve_root_problem(rtfn, sol, kn)

# Merge boundary conditions and solution
sol_dct = ntw.system.sol_to_dict(sol)
plain_bcs = ntw.system._constraint_manager.get_plain_bc_dict()
all_data = {**plain_bcs, **sol_dct}

# Plotting
fig, ax_mer = plt.subplots()
ax_mer.set_aspect('equal')

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

sta_geom.plot_meridional_profile(color='k', ax=ax_mer)
# rot_geom.plot_meridional_profile(color='k', ax=ax)

fig, axs_tri = plt.subplots(1, 2)
[ax.set_aspect('equal') for ax in axs_tri]

plot_velocity_triangles(
    all_data[n0.kin.V_tan],
    all_data[n0.kin.V_mer],
    all_data[n0.kin.BladeSpeed],
    all_data[n0.geo.RDistr],
    axs_tri[0],
)
plot_velocity_triangles(
    all_data[n1.kin.V_tan],
    all_data[n1.kin.V_mer],
    all_data[n1.kin.BladeSpeed],
    all_data[n1.geo.RDistr],
    axs_tri[1],
)

# TODO: Make this automated
abs_state.update(
    cp.PT_INPUTS,
    all_data[n0.tot.Pressure],
    all_data[n0.tot.Temperature],
)
ht0 = abs_state.hmass()

abs_state.update(
    cp.PT_INPUTS,
    all_data[n1.tot.Pressure],
    all_data[n1.tot.Temperature],
)
ht1 = abs_state.hmass()

print(f'Turbine power {all_data[n0.oth.CumMassFlow] * (ht0 - ht1)}')

plt.show()
