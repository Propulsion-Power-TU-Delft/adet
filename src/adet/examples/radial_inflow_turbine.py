from adet.assembly import CasadiSystem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.fluid.settings import ExternalFluidModel, FluidSettings
from adet.components.network import ComponentNetwork
from adet.losses.basic import PercentageEntropyLoss, ZeroDeviation
from pint import Quantity
from adet.equations.variables import NodeVariables
from adet.components.connections import Inlet, Shaft
from adet.components.blade_row import BladeRow

n0 = NodeVariables(0)
n1 = NodeVariables(1)

inl = Inlet(
    {
        n0.tot.Pressure: 18.1e5,
        n0.tot.Temperature: Quantity(300, 'degC'),
        n0.kin.FlowAngleAbs: Quantity(65, 'deg'),
        n0.kin.FlowAngleAbs: Quantity(65, 'deg'),
    }
)

shaft = Shaft(Quantity(98100, 'rpm'), is_constrained=True)

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
        n1.geo.Height: Quantity(8, 'mm'),
        n1.geo.Rmid: Quantity(38, 'mm'),
        n1.geo.MeridionalAngle: Quantity(-90, 'deg'),
        n1.geo.BldThick: 0.0,
    },
    shaft=shaft,
    extra_equations={
        PercentageEntropyLoss(0.0): (0, 1),
        ZeroDeviation(): 0,
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
