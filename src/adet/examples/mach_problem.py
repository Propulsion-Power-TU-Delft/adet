from pint import Quantity
from adet.assembly import CasadiSystem
from adet.equations.fundamental import (
    Kinematics,
    MassAreaRelation,
    TotalStaticMatching,
    ZeroBlockage,
)
from adet.equations.geometrical import AnnulusAreas
from adet.equations.special import ThermoVarsAdder
from adet.equations.variables import NodeVariables, ThermoVariables
from adet.fluid.settings import ExternalFluidModel, FluidModel, FluidSettings
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState


EQUATIONS = {
    TotalStaticMatching(): 0,
    AnnulusAreas(): 0,
    MassAreaRelation(): 0,
    ZeroBlockage(): 0,  # Area = Eff area
    Kinematics(): 0,
    ThermoVarsAdder(): 0,
}

system = CasadiSystem()
model = ExternalFluidModel(DebugAbstractState('REFPROP', 'MM'))
thrm = ThermoVariables()
fluid_settings = FluidSettings(model, (thrm.Pressure, thrm.Temperature))
system.fluid_settings = fluid_settings

for eq, pos in EQUATIONS.items():
    system.add_equation(eq, pos)

n0 = NodeVariables(0)
BC = {
    n0.kin.Omega: 0.0,
    n0.kin.FlowAngleAbs: Quantity(0, 'deg'),
    n0.oth.MassFlow: 0.132,
    n0.geo.RDistr: 0.02,
    n0.geo.HDistr: 0.02 / 2,
    n0.tot.Pressure: 18.1e5,
    n0.tot.Temperature: 573.15,
}

system.add_boundary_conditions(BC)
system.build()

rtfn = system.make_rootfinder('ipopt')

x0 = system.get_scaled_guess()
kn = system.get_scaled_constraints()

sol = solve_root_problem(rtfn, x0, kn)

system.write_solution_to_nodes(sol)

n0 = system.nodes[0]
