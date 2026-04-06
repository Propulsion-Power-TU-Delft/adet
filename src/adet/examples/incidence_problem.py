import logging
from pint import Quantity
from adet.assembly import CasadiSystem
from adet.components.blade_row import IncidenceVolume
from adet.components.connections import Inlet
from adet.components.network import ComponentNetwork
from adet.fluid.settings import AnalyticalFluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger

logger = logging.getLogger(__name__)
setup_logger(logger)

INLET_BC = {
    'tot': {
        'p': 1e5,
        'T': 300,
    },
    'kin': {
        'Vm': 80,
        'alpha': 0.0,
    },
}

OUTLET_BC = {
    'geo': {
        'metal_angle': Quantity(30, 'deg'),
        'thick_by_pitch': 0.02,
        'rr_midspan': 0.1,
        'height': 0.1,
        'meridional_angle': 0,
        'num_blades': 15,
        'chord': 0.0,
    },
    'kin': {
        'omega': Quantity(1000, 'rpm'),
    },
    'oth': {
        'disp_thick': 0.0,
    },
}

inlet = Inlet(INLET_BC)
fluid_state = IdealGasState(1.4, 287, 2e-5)
model = AnalyticalFluidModel(fluid_state)
settings = FluidSettings(model, ('p', 'T'))

cv = IncidenceVolume('incVol', outlet_bc=OUTLET_BC)

ntw = ComponentNetwork(settings, inlet, CasadiSystem(1), [cv])

ntw.build()

rtfn_IP = ntw.system.make_rootfinder('ipopt')
rtfn_KN = ntw.system.make_rootfinder('kinsol')

x0 = ntw.system.get_scaled_guess()
kn = ntw.system.get_scaled_constraints()

sol = solve_root_problem(rtfn_IP, x0, kn)
sol = solve_root_problem(rtfn_KN, sol, kn)
sol_dict = ntw.system.write_solution_to_nodes(sol)
n0 = ntw.system.nodes[0]
n1 = ntw.system.nodes[1]
