import logging

import matplotlib.pyplot as plt
from pint import Quantity

from adet.assemblers import CasadiSystem
from adet.components.blade_row import BladeRow, Interspace, ShockMixer
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork
from adet.equations.definitions import BoundaryLayerRatios, IsentropicProperties
from adet.equations.nondimensional import GammaPV
from adet.equations.utils import residual_debugger
from adet.fluid.settings import FluidSettings
from adet.losses.basic import IsentropicLink, ZeroDeviation
from adet.losses.rit import StatorProfileLoss
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger
from adet.tools.plotting import plot_velocity_triangles
from adet.variables import NodeVariables

logger = logging.getLogger(__name__)
setup_logger(logger)
PLOTS = False

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
    bound_cond={
        n1.geo.HeightRatio: 1.0,
        n1.geo.RadiusRatio: 0.75,
        n1.geo.Rmid: Quantity(25.75, 'mm'),
        n1.geo.MeridionalAngle: Quantity(-90, 'deg'),
        # *** Outlet
        n1.geo.MetalAngle: Quantity(78, 'deg'),
        # n1.kin.FlowAngleRel: Quantity(78, 'deg'),
        # n1.kin.Mach: 2.2,
        # *** Blades
        n0.geo.ThickByPitch: 0.05,
        n1.geo.ThickByPitch: 0.05,
        n1.geo.NumBlades: 12,
        n1.geo.ChordAx: Quantity(0.01, 'mm'),
        # *** Boundary Layer
        n1.oth.MomByBld: 0.075,
        n1.oth.DispByMom: 2.0,
        n1.oth.DispByHgt: 0.09,
        # *** Shock
    },
    shaft=casing,
    extra_equations={
        IsentropicLink(): (0, 1),
        ZeroDeviation(): 0,  # No incidence
        # *** Loss + Dependencies
        # ShockLoss(): 1,
        StatorProfileLoss(): (0, 1),
        IsentropicProperties(): (0, 1),
        BoundaryLayerRatios(): 1,
        GammaPV(): 0,
        GammaPV(): 1,
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

shock_mix = ShockMixer(
    'st_shock',
    bound_cond={
        n1.oth.ShockAngle: Quantity(90, 'deg'),
    },
)

rotor = BladeRow(
    'impeller',
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

fluid_settings = FluidSettings(
    fluid_state=abs_state,
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
        # shock_mix,
        # interspace,
        # rotor,
    ],
)

ntw.build()

rtfn = ntw.system.make_rootfinder(
    'ipopt',
    opts={'error_on_fail': False},
)
x0 = ntw.system.get_scaled_guess(fallback=0.5)
kn = ntw.system.get_scaled_constraints()
bnd = ntw.system.get_arguments_bounds(
    {
        # WARN: Force the supersonic solution w/ bounds
        # n1.kin.Mach: (1.1, 4.0),
        # n1.oth.ShockDeflection: (0.1, 1.5),
        # NOTE: Thermo bounding stabilizes a lot
        n0.stc.Temperature.Glob: (300, 580),
        n0.stc.Pressure.Glob: (1e3, 1e9),
    }
)

sol = solve_root_problem(rtfn, x0, kn, bnd, suppress_output=False)

# Kinsol pass
rtfn = ntw.system.make_rootfinder('kinsol')
sol = solve_root_problem(rtfn, sol, kn)

# Merge boundary conditions and solution
sol_data = ntw.system.sol_to_dict(sol)
# pprint.pprint(sol_data)

if PLOTS:
    # Plotting
    fig, ax_mer = plt.subplots()
    ax_mer.set_aspect('equal')

    # sta_geom = RowGeometry(
    #     float(sol_data[n0.geo.Rmid][0]),
    #     float(sol_data[n1.geo.Rmid][0]),
    #     float(sol_data[n0.geo.Height][0]),
    #     float(sol_data[n1.geo.Height][0]),
    #     float(sol_data[n0.geo.MeridionalAngle][0]),
    #     float(sol_data[n1.geo.MeridionalAngle][0]),
    #     float(sol_data[n1.geo.ChordAx][0]),
    # )
    #
    # rot_geom = RowGeometry(
    #     float(sol_data[n4.geo.Rmid][0]),
    #     float(sol_data[n5.geo.Rmid][0]),
    #     float(sol_data[n4.geo.Height][0]),
    #     float(sol_data[n5.geo.Height][0]),
    #     float(sol_data[n4.geo.MeridionalAngle][0]),
    #     float(sol_data[n5.geo.MeridionalAngle][0]),
    #     float(sol_data[n5.geo.ChordAx][0]),
    # )
    #
    # sta_geom.plot_meridional_profile(color='b', ax=ax_mer)
    # rot_geom.plot_meridional_profile(color='k', ax=ax_mer)

    fig, axs_tri = plt.subplots(2, 2, figsize=(10, 20), dpi=70)
    [ax.set_aspect('equal') for ax in axs_tri.flat]

    plot_velocity_triangles(
        sol_data[n0.kin.V_tan],
        sol_data[n0.kin.V_mer],
        sol_data[n0.kin.BladeSpeed],
        sol_data[n0.geo.RDistr],
        axs_tri[0, 0],
        fontsize=17,
    )
    plot_velocity_triangles(
        sol_data[n1.kin.V_tan],
        sol_data[n1.kin.V_mer],
        sol_data[n1.kin.BladeSpeed],
        sol_data[n1.geo.RDistr],
        axs_tri[0, 1],
        fontsize=17,
    )
    plot_velocity_triangles(
        sol_data[n2.kin.V_tan],
        sol_data[n2.kin.V_mer],
        sol_data[n2.kin.BladeSpeed],
        sol_data[n2.geo.RDistr],
        axs_tri[1, 0],
        fontsize=17,
    )
    plot_velocity_triangles(
        sol_data[n3.kin.V_tan],
        sol_data[n3.kin.V_mer],
        sol_data[n3.kin.BladeSpeed],
        sol_data[n3.geo.RDistr],
        axs_tri[1, 1],
        fontsize=17,
    )
    plt.show()


# Turbine power
# pwr = sol_data[n0.oth.CumMassFlow] * (
#     sol_data[n4.tot.Enthalpy] - sol_data[n5.tot.Enthalpy]
# )
# print(f'Turbine power {pwr}')

# Debug loss
globals().update(residual_debugger(StatorProfileLoss(), [0, 1], sol_data))
