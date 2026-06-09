# === IMPORTS
from adet.solution import solve_root_problem
from adet.equations.geometrical import MinimalCamberLine, MeridionalGeometry
from copy import deepcopy

from pint import Quantity

from adet.assembly import CasadiSystem
from adet.components import BladeRow, Inlet
from adet.components.connections import Shaft
from adet.components.network import ComponentNetwork
from adet.equations.definitions import RepeatedStage
from adet.equations.nondimensional import (
    FlowCoefficient,
    StaticTotalDegreeOfReaction,
    TotalStaticLoadingCoefficient,
    TotalTotalExpansionEfficiency,
    VolumetricFlowRatio,
    WorkCoefficient,
)
from adet.fluid.settings import FluidModel, FluidSettings
from adet.losses.basic import IsentropicLink, ZeroDeviation
from adet.tools.coolprop_utils import DebugAbstractState
from adet.variables import NodeVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)
n2 = NodeVariables(2)
n3 = NodeVariables(3)

abs_state = DebugAbstractState('HEOS', 'Air')

# *** Inlet conditions
inlet = Inlet(
    boundary_conditions={
        # n0.oth.CumMassFlow: 100,
        n0.geo.Rmid: 0.1,
        n0.geo.MeridionalAngle: Quantity(0, 'deg'),
        # n0.geo.HubTipRatio: 0.9,
        n0.tot.Pressure: 10e5,
        n0.tot.Temperature: 500,
    }
)


casing = Shaft(0, is_constrained=True)
shaft = Shaft(0, is_constrained=False)


stator = BladeRow(
    name='Stator',
    shaft=casing,
    row_type='stator',
    bound_cond={
        n0.geo.ThickByPitch: 0.04,
        n1.geo.MeridionalAngle: Quantity(0, 'deg'),
        n1.geo.ThickByPitch: 0.02,
        n1.geo.ClearanceByHeight: 0.01,
        n1.geo.NumBlades: 30,
        #
        n1.oth.MomByBld: 0.075,
        n1.oth.DispByMom: 2,
        n1.oth.DispByHgt: 0.05,
        #
        n1.oth.CdProfile: 0.002,
        n1.oth.XiCambLenA: 0.375,
        n1.oth.XiCambLenB: 0.675,
        n1.oth.DischCoeff: 0.35,
        # n1.stc.Pressure: 1.462617e6, # Legacy ?
    },
    extra_equations={
        ZeroDeviation(): 0,  # No incidence (design)
        ZeroDeviation(): 1,  # No deviation
        # MinimalCamberLine(): (0, 1),
        # ParabolicCamberline(): (0, 1),
        IsentropicLink(): (0, 1),
    },
)
stator.set_component_constants(n0.geo.Rmid.Glob)

# ============ Modify rotor
rotor = deepcopy(stator)  # Reuse the stator as template
rotor.shaft = shaft  # Assign the rotating shaft
rotor.name = 'rotor'
rotor.row_type = 'rotor'  # Set the type (useless now)
rotor.add_equation(WorkCoefficient(), (0, 1))

fluid_model = FluidModel(abs_state)
fluid_settings = FluidSettings(
    fluid_model,
    (n0.stc.Pressure, n0.stc.Temperature),
)

ntw = ComponentNetwork(
    fluid_settings,
    inlet,
    CasadiSystem(1),
    [stator, rotor],
)


ntw.system.add_equation(RepeatedStage(), (0, 1, 2, 3))
ntw.system.add_equation(StaticTotalDegreeOfReaction(), (0, 1, 2, 3))

ntw.system.add_equation(FlowCoefficient(), (0, 3))
ntw.system.add_equation(VolumetricFlowRatio(), (0, 3))
ntw.system.add_equation(TotalStaticLoadingCoefficient(), (0, 3))
ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, 3))

rotor.set_spanwise_constant(n1.geo.ChordAx)
stator.set_spanwise_constant(n1.geo.ChordAx, n0.geo.HDistr, n0.kin.V_mer)
rotor.copy_from_previous(n0.geo.HDistr, n0.geo.RDistr)
rotor.remove_equation(MeridionalGeometry, 0)

stator.set_boundary_cond(n1.geo.FlareAngle, Quantity(0, 'deg'))
rotor.set_boundary_cond(n1.geo.FlareAngle, Quantity(0, 'deg'))

rotor.set_boundary_cond(n1.geo.HubTipRatio, 0.818)
DUTY_COEFFS = {
    n1.ndim.FlowCoeff: 0.4,
    n1.ndim.TSLoadCoeff: 3.0,
    n1.ndim.VolflowRatio: round(4, 1),
    n1.ndim.DegreeOfReactionTS: round(0.5, 1),
}
rotor.bc_from_dict(DUTY_COEFFS)  # Duty coefficients at node 1

ntw.build()

rtfn = ntw.system.make_rootfinder('ipopt')

x0 = ntw.system.get_scaled_guess(fallback=0.8)
kn = ntw.system.get_scaled_constraints()

solve_root_problem(rtfn, x0, kn)
