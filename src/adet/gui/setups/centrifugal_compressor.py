"""Blade row chain exported from the ADeT GUI."""

import pickle
from pathlib import Path

import CoolProp as cp  # noqa: F401
from CoolProp import AbstractState  # noqa: F401

from adet.assemblers import CasadiSystem
from adet.components import BladeRow, Inlet
from adet.components.connections import Shaft
from adet.components.network import ComponentNetwork
from adet.fluid.ideal_eos import IdealGasState  # noqa: F401
from adet.fluid.settings import FluidSettings
from adet.losses.basic import TotalPressureLoss, ZeroDeviation
from adet.solution import solve_root_problem
from adet.variables import NodeVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)

# Inlet total conditions
INLET_CONDITIONS = {
    n0.oth.TotMassFlow: 50.0,
    n0.tot.Pressure: 1000000.0,
    n0.tot.Temperature: 500.0,
}

# Boundary conditions of each row in its own nodes (n0 inlet, n1 outlet)
ROW_PARAMS = [
    {
        n0.geo.Rmid: 0.15058410625743693,
        n0.geo.Height: 0.15374810974571146,
        n0.geo.MeridionalAngle: 0.0,
        n1.geo.MeridionalAngle: 1.5707963267948966,
        n1.geo.Height: 0.022321976778238935,
        n1.geo.Rmid: 0.4191600845342942,
        n1.geo.Chord: 0.2949630454932621,
        n1.geo.NumBlades: 40.0,
        n0.kin.FlowAngleAbs: 0.0,
        n1.kin.FlowAngleRel: -0.5235987755982988,
        n1.kin.Omega: 1150.0,
    },
    {
        n1.geo.MeridionalAngle: 1.5707963267948966,
        n1.geo.Height: 0.022321976778238935,
        n1.geo.Rmid: 0.5180229076977626,
        n1.geo.Chord: 0.08886282316346848,
        n1.geo.NumBlades: 40.0,
        n1.kin.FlowAngleRel: 0.6749056258003521,
        n1.kin.Omega: 0.0,
    },
]

state = IdealGasState(1.4, 287.0, 2e-05)
inlet = Inlet(boundary_conditions=INLET_CONDITIONS)

rows = []
for index, params in enumerate(ROW_PARAMS):
    bound_cond = {s: v for s, v in params.items() if s != n1.kin.Omega}
    row = BladeRow(
        name=f'row{index}',
        shaft=Shaft(params[n1.kin.Omega], is_constrained=True),
        bound_cond=bound_cond,
        extra_equations={
            ZeroDeviation(): 0,
            TotalPressureLoss(0.9): (0, 1),
        },
        spanwise_constants=[n1.geo.ChordAx],
    )
    if index == 0:
        row.set_spanwise_constant(n0.kin.V_mer, n0.geo.HDistr)
    rows.append(row)

ntw = ComponentNetwork(
    fluid_settings=FluidSettings(
        fluid_state=state,
        update_variables=(n0.stc.Pressure, n0.stc.Temperature),
    ),
    inlet=inlet,
    backend=CasadiSystem(num_span=1),
    components=rows,
)
ntw.build()

system = ntw.system
# Converged solution of the GUI, values by variable
with open(Path(__file__).parent / 'centrifugal_compressor_solution.pkl', 'rb') as file:
    guess = pickle.load(file)
x0 = system.get_guess(guess, fallback=0.8)
kn = system.get_boundary_conds()
rootfinder = system.make_rootfinder(
    'ipopt', {'error_on_fail': True, 'ipopt.max_wall_time': 5.0}
)
sol = solve_root_problem(rootfinder, x0, kn, suppress_output=True)
rootfinder = system.make_rootfinder('newton', {'max_iter': 25})
sol = solve_root_problem(rootfinder, sol, kn, suppress_output=True)
sol_dict = system.sol_to_dict(sol)

if __name__ == "__main__":
    for index in range(len(rows)):
        node_in, node_out = NodeVariables(2 * index), NodeVariables(2 * index + 1)
        print(
            f'row {index}: Vm = {sol_dict[node_out.kin.V_mer][0]:.2f} m/s, '
            f'Pt out = {sol_dict[node_out.tot.Pressure][0]:.1f} Pa'
        )
