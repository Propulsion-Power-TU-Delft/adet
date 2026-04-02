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

BC = {
    'kin': {
        'omega': 0.0,
        'alpha': Quantity(65, 'deg'),
    },
    'oth': {
        'massflow': 0.132,
    },
    'geo': {
        'rr': 0.0375,
        'hh': 0.002,
    },
    'tot': {
        'p': 18.1e5,
        'T': 573.15,
    },
}

system = CasadiSystem()
model = ExternalFluidModel(DebugAbstractState('REFPROP', 'MM'))
fluid_settings = FluidSettings(model, ('p', 'T'))
system.fluid_settings = fluid_settings

for eq, pos in EQUATIONS.items():
    system.add_equation(eq, pos)


system.add_boundary_conditions(BC, 0)
system.build()

rtfn = system.make_rootfinder('ipopt')

x0 = system.get_scaled_guess()
kn = system.get_scaled_constraints()

sol = solve_root_problem(rtfn, x0, kn)

system.write_solution_to_nodes(sol)

n0 = system.nodes[0]
