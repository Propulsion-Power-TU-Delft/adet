"""
Mach sweep of the Denton-Baumgartner trailing-edge mixing loss

Single-station study of ``DentonBaumgartnerMixingLoss``. The fixed
parameters are taken from the stator of ``radial_inflow_turbine.py``
solved without any mixing model. The relative Mach number at the trailing
edge is swept; for each value the system is solved for the mixing entropy
rise, the base pressure (from ``SieverdingBasePressure``) and the
relative-total pressure after mixing.

State:
------
- Two nodes: node 0 is the upstream reference, node 1 is the trailing edge
  where the mixing model acts (as ``DentonBaumgartnerMixingLoss(): 1`` in
  the turbine example).
"""

import logging

import numpy as np
from pint import Quantity
from tabulate import tabulate

from adet.assemblers import CasadiSystem
from adet.equations.definitions import BoundaryLayerRatios
from adet.equations.fundamental import TotalStaticMatching
from adet.equations.nondimensional import GammaPV, RelativeMachNumber
from adet.fluid.settings import FluidSettings
from adet.losses.mixing import DentonBaumgartnerMixingLoss, SieverdingBasePressure
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables

logger = logging.getLogger(__name__)
setup_logger(logger)

# *** Swept parameter: relative Mach number at the trailing edge
MACH_VALUES = np.linspace(1.1, 2.5, 15)

# *** Fixed parameters: stator of radial_inflow_turbine.py, no mixing model
# Node 0: stator inlet (static state, absolute and relative velocity)
P0 = Quantity(18.069, 'bar')
T0 = Quantity(573.1, 'K')
V0 = 8.6  # m/s
W0 = 8.6  # m/s
# Node 1: stator outlet (static state, absolute velocity)
P1 = Quantity(1.613, 'bar')
T1 = Quantity(535.6, 'K')
V1 = 320.2  # m/s
# Geometry at the trailing edge: Rmid = 26.1 mm, 12 blades
R_MID = 26.1e-3
N_BLADES = 12
PITCH1 = 2 * np.pi * R_MID / N_BLADES  # m
THICK_BY_PITCH = 0.05
HEIGHT1 = Quantity(2, 'mm')
METAL_ANGLE1 = Quantity(78, 'deg')
# Boundary layer ratios (as in the stator of the turbine example)
MOM_BY_BLD = 0.075
DISP_BY_MOM = 2.0
DISP_BY_HGT = 0.09

n0 = NodeVariables(0)
n1 = NodeVariables(1)

abs_state = DebugAbstractState('HEOS', 'MM')

system = CasadiSystem()
system.fluid_settings = FluidSettings(
    fluid_state=abs_state,
    update_variables=(
        n0.stc.Pressure.Glob,
        n0.stc.Temperature.Glob,
    ),
)

EQUATIONS = {
    TotalStaticMatching(): 0,
    TotalStaticMatching(): 1,
    RelativeMachNumber(): 1,
    GammaPV(): 1,
    BoundaryLayerRatios(): 1,
    # *** Link: Sieverding base pressure at node 1 from the node 0 total pressure
    SieverdingBasePressure(): (0, 1),
    # *** Mixing model under study, evaluated at the trailing edge
    DentonBaumgartnerMixingLoss(): 1,
}
for eq, pos in EQUATIONS.items():
    system.add_equation(eq, pos)

BC = {
    # *** Node 0
    n0.stc.Pressure: P0,
    n0.stc.Temperature: T0,
    n0.kin.V_mag: V0,
    n0.kin.W_mag: W0,
    # *** Node 1 (trailing edge)
    n1.stc.Pressure: P1,
    n1.stc.Temperature: T1,
    n1.kin.V_mag: V1,
    n1.geo.Pitch: PITCH1,
    n1.geo.BldThick: THICK_BY_PITCH * PITCH1,
    n1.geo.Height: HEIGHT1,
    n1.geo.MetalAngle: METAL_ANGLE1,
    n1.oth.MomByBld: MOM_BY_BLD,
    n1.oth.DispByMom: DISP_BY_MOM,
    n1.oth.DispByHgt: DISP_BY_HGT,
    n1.kin.RelMach: MACH_VALUES[0],  # Swept below
}
system.add_boundary_conditions(BC)
system.build()

rtfn = system.make_rootfinder(
    'ipopt',
    opts={'error_on_fail': False},
)
bnd = system.get_bounds({n1.kin.RelMach: (1.0, 4.0)})
x0 = system.get_guess(fallback=0.5)

rows = []
for mach in MACH_VALUES:
    system.data.boun_cond[n1.kin.RelMach] = mach
    kn = system.get_boundary_conds()
    sol = solve_root_problem(rtfn, x0, kn, bnd, suppress_output=True)
    x0 = sol  # Warm start the next Mach value
    sol_data = system.sol_to_dict(sol)

    def val(spec, _sol=sol_data):
        return float(_sol[spec][0])

    rows.append(
        [
            f'{mach:.3f}',
            f'{val(n1.loss.Ds_mixing):.3f}',
            f'{val(n1.oth.PBase) / 1e5:.4f}',
            f'{val(n0.rlt.Pressure) / 1e5:.4f}',
            f'{val(n1.rlt.Pressure) / 1e5:.4f}',
        ]
    )

print(
    tabulate(
        rows,
        headers=[
            'M_rel (node 1)',
            'Ds_mixing\n[J/kg/K]',
            'p_base\n[bar]',
            'p0_rel inlet\n[bar]',
            'p0_rel trailing edge\n[bar]',
        ],
        tablefmt='github',
    )
)
