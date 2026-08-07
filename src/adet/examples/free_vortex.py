import logging

import matplotlib.pyplot as plt
from pint import Quantity

from adet.assemblers import CasadiSystem
from adet.equations.fundamental import (
    FreeVortexDistribution,
    Kinematics,
    MassAreaRelation,
    TotalStaticMatching,
    ZeroBlockage,
)
from adet.equations.geometrical import (
    AnnulusAreas,
    EndwallProperties,
    MeridionalGeometry,
)
from adet.fluid.ideal_eos import IdealGasState
from adet.fluid.settings import FluidSettings
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger
from adet.tools.plotting import plot_velocity_triangles
from adet.variables import NodeVariables

logger = logging.getLogger(__name__)
setup_logger(logger)
n0 = NodeVariables(0)

system = CasadiSystem(num_span=3)

EQS = {
    # *** Node 0
    Kinematics(): 0,
    TotalStaticMatching(): 0,
    MeridionalGeometry(): 0,
    EndwallProperties(): 0,
    FreeVortexDistribution(): 0,
    MassAreaRelation(): 0,
    AnnulusAreas(): 0,
    ZeroBlockage(): 0,
}

BCS = {
    n0.tot.Pressure: 1e6,
    n0.tot.Temperature: 500,
    # Kine
    n0.kin.Omega: 500,
    n0.kin.V_mer: 30,
    n0.kin.Beta_mid: Quantity(0, 'deg'),
    # Geometry
    n0.geo.Rmid: 0.1,
    n0.geo.HubTipRatio: 0.7,
    n0.geo.MeridionalAngle: 0.0,
}
idl_state = IdealGasState(1.4, 287, 2e-5)

system.fluid_settings = FluidSettings(
    fluid_state=idl_state,
    update_variables=(n0.stc.Pressure.Glob, n0.stc.Temperature.Glob),
)
system.add_spanwise_constants(n0.oth.StreamMassFlow)


[system.add_equation(eq, pos) for eq, pos in EQS.items()]
system.add_boundary_conditions(BCS)

system.build()

x0 = system.get_guess(fallback=0.01)
kn = system.get_boundary_conds()
bnd = system.get_bounds(
    {
        # Node limiters
        n0.stc.Pressure.Glob: (1, 1e7),
        n0.stc.Temperature.Glob: (110, 1e4),
    },
    ignore_defaults=False,
)

rtfn = system.make_rootfinder('kinsol', {'error_on_fail': False})
sol = solve_root_problem(rtfn, x0, kn, bnd)

data = system.sol_to_dict(sol)

fig, ax = plt.subplots(figsize=(6, 10))
ax.set_aspect('equal')
plot_velocity_triangles(
    data[n0.kin.V_tan],
    data[n0.kin.V_mer],
    data[n0.kin.BladeSpeed],
    data[n0.geo.RDistr],
    ax,
)
fig.tight_layout()
fig.show()
