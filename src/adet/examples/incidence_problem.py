import logging

import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity

from adet.assembly import CasadiSystem
from adet.components.blade_row import BladeRow, IncidenceVolume
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork
from adet.equations.base_equation import EquationBase
from adet.equations.fundamental import BladeBlockage
from adet.fluid.settings import AnalyticalFluidModel, ExternalFluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.losses.basic import PercentageEntropyLoss
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.interpolation import resample_linear
from adet.tools.loggers import setup_logger

logger = logging.getLogger(__name__)
setup_logger(logger)


# Find the optimal massflow
class UseOptimalIncidence(EquationBase):
    def residual(self, kin_beta0, kin_beta_opt0):
        return kin_beta0 - kin_beta_opt0


NUM_SPAN = 1
inlet = Inlet(
    {
        'tot': {'p': 101352.9, 'T': 288.16},
        'oth': {'cum_massflow': 4.98},
        'kin': {'alpha': Quantity(0, 'deg')},
    }
)
fluid_state = IdealGasState(1.4, 287, 1.8e-5)
real_state = DebugAbstractState('REFPROP', 'Air')
model = ExternalFluidModel(real_state)
model = AnalyticalFluidModel(fluid_state)

settings = FluidSettings(model, ('p', 'T'))

incVol = IncidenceVolume('incVol')
shaft = Shaft(
    Quantity(21789, 'rpm'),
    is_constrained=True,
)

# Metal angle distribution
METAL_ANGLE = np.array([-30, -44, -53])
angle_values = resample_linear(METAL_ANGLE, NUM_SPAN)
if NUM_SPAN == 1:
    angle_values = Quantity(-44, 'deg')
else:
    angle_values = Quantity(angle_values, 'deg')

angle_distribution = Quantity(angle_values, 'deg')
# +++ Components
impeller = BladeRow(
    name='rotor',
    shaft=shaft,
    row_type='rotor',
    in_constraints={
        'geo': {
            # *** Meridional geometry
            'meridional_angle': Quantity(0, 'deg'),
            'rr_midspan': Quantity(0.07416165, 'm'),
            'height': Quantity(0.0670433, 'm'),
            # *** Blades specs
            'metal_angle': angle_values,
            'bld_thick': 0.002,
        },
        'oth': {
            'disp_thick': 0.0,
        },
    },
    out_constraints={
        'geo': {
            # *** Meridional geometry
            'meridional_angle': Quantity(90, 'deg'),
            'rr_midspan': Quantity(0.2159, 'm'),
            'height': Quantity(0.01524, 'm'),
            # *** Blades specs
            'metal_angle': Quantity(-30, 'deg'),
            'thick_by_pitch': 0.02,  # Thickness by pitch ratio
            'chord_ax': Quantity(0.133879895, 'm'),
            'num_blades': 15,
        },
        'oth': {
            # 'eta_tt': 0.821,  # Total total efficiency
        },
    },
    extra_equations={
        BladeBlockage(): 0,  # WARN: This is added manually for now, can cause mass imba
        PercentageEntropyLoss(0.0): (0, 1),
    },
)
ntw = ComponentNetwork(
    settings,
    inlet,
    CasadiSystem(num_span=NUM_SPAN),
    [
        incVol,
        impeller,
    ],
)

# WARN: SET CONSTANT PRESSURE (Vt = 0)
incVol.set_spanwise_constant('kin_Vm0', 'geo_hh0', 'kin_Vm1')
impeller.set_spanwise_constant('stc_p1')

ntw.build()

rtfn_IP = ntw.system.make_rootfinder('ipopt')
rtfn_KN = ntw.system.make_rootfinder('kinsol')

x0 = ntw.system.get_scaled_guess()
kn = ntw.system.get_scaled_constraints()
bnd = ntw.system.get_arguments_bounds()

sol = solve_root_problem(rtfn_IP, x0, kn, bnd)
sol = solve_root_problem(rtfn_KN, sol, kn)
sol_dict = ntw.system.write_solution_to_nodes(sol)

n0 = ntw.system.nodes[0]
n1 = ntw.system.nodes[1]
n2 = ntw.system.nodes[2]
n3 = ntw.system.nodes[3]

fig, ax = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
n0.kin.plot(n0.geo, 15, ax[0])
n1.kin.plot(n1.geo, 15, ax[1])
n2.kin.plot(n2.geo, 15, ax[2])
n3.kin.plot(n3.geo, 15, ax[3])
ax[-1].set_ylim(-250, 440)
ax[-1].set_aspect('equal')
fig.show()
