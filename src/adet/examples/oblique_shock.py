from adet.fluid.symbolic_eos import IdealGasState
import logging

import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity

from adet.assembly import CasadiSystem
from adet.equations.control_volumes import ObliqueShock
from adet.equations.fundamental import (
    Kinematics,
    MassAreaRelation,
    TotalStaticMatching,
    ZeroBlockage,
)
from adet.equations.geometrical import AnnulusAreas
from adet.equations.nondimensional import (
    AbsoluteMachNumber,
    RelativeMachNumber,
)
from adet.fluid.settings import FluidSettings, AnalyticalFluidModel
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables, ThermoVariables

logger = logging.getLogger(__name__)
setup_logger(logger)

EQUATIONS = {
    TotalStaticMatching(): 0,
    AnnulusAreas(): 0,
    MassAreaRelation(): 0,
    AbsoluteMachNumber(): 0,
    RelativeMachNumber(): 0,
    ZeroBlockage(): 0,
    Kinematics(): 0,
    # GammaPV(): 0,
    # *** Out
    TotalStaticMatching(): 1,
    AnnulusAreas(): 1,
    MassAreaRelation(): 1,
    AbsoluteMachNumber(): 1,
    RelativeMachNumber(): 1,
    ZeroBlockage(): 1,
    Kinematics(): 1,
    # GammaPV(): 1,
    # *** Link
    ObliqueShock(): (0, 1),
}

n0 = NodeVariables(0)
n1 = NodeVariables(1)


# Build system once outside the loops
system = CasadiSystem()

# *** Fluid model
model = AnalyticalFluidModel(IdealGasState(1.4, 287, 2e-5))
thrm = ThermoVariables()
fluid_settings = FluidSettings(model, (thrm.Pressure, thrm.Temperature))
system.fluid_settings = fluid_settings

for eq, pos in EQUATIONS.items():
    system.add_equation(eq, pos)

# Initial boundary conditions (shock angle will be updated per iteration)
BC = {
    # *** INLET
    n0.kin.Omega: 0.0,
    n0.kin.FlowAngleAbs: Quantity(0, 'deg'),
    n0.kin.Mach: 2.0,  # Placeholder, will be updated
    n0.geo.RDistr: 0.1,
    n0.geo.HDistr: 0.1,
    n0.tot.Pressure: 30e5,
    n0.tot.Temperature: 573.15,
    # *** OUTLET
    n1.kin.Omega: 0.0,
    n1.geo.RDistr: 0.1,
    n1.geo.HDistr: 0.1,
    n1.oth.ShockAngle: Quantity(10, 'deg'),  # Placeholder, will be updated
}

system.add_boundary_conditions(BC)
system.build()

bnd = system.get_arguments_bounds(
    {
        n0.stc.Temperature.Glob: (100, 1e4),
        n0.stc.Pressure.Glob: (1e5, 1e9),
    },
)

# Parametric sweep ranges
mach_values = np.linspace(2.0, 5.0, 4)
shock_angle_values = np.linspace(40, 90, 20)
results = {}
for mach in mach_values:
    results[mach] = {'shock_angles': [], 'deflections': []}

sol_dict = {}
for mach_idx, mach in enumerate(mach_values):
    print(f'Mach = {mach:.2f}  [{mach_idx + 1}/{len(mach_values)}]')

    for angle in shock_angle_values:
        # Update Mach in system.data
        system.data.boun_cond[n0.kin.Mach] = mach
        # Update shock angle in system.data (convert to radians)
        system.data.boun_cond[n1.oth.ShockAngle] = np.radians(angle)

        x0 = system.get_scaled_guess(sol_dict)
        kn = system.get_scaled_constraints()

        rtfn = system.make_rootfinder(
            'ipopt',
            opts={
                'error_on_fail': False,
                'ipopt.max_wall_time': 5,
                'ipopt.print_level': 0,
            },
        )
        sol = solve_root_problem(rtfn, x0, kn, suppress_output=True)

        rtfn = system.make_rootfinder(
            'kinsol',
            opts={'error_on_fail': True},
        )

        try:
            sol = solve_root_problem(rtfn, sol, kn, suppress_output=True)
            sol_dict = system.sol_to_dict(sol)
            # Extract deflection angle (convert from radians to degrees)
            deflection_val = sol_dict[n1.oth.ShockDeflection]
            deflection_rad = float(np.atleast_1d(deflection_val)[0])

            # Normalize angle to  to handle periodic convergence
            deflection_rad = np.arctan2(np.sin(deflection_rad), np.cos(deflection_rad))
            deflection_deg = deflection_rad * 180 / np.pi

            results[mach]['shock_angles'].append(angle)
            results[mach]['deflections'].append(deflection_deg)
        except (RuntimeError, ValueError) as e:
            err_msg = str(e)[:50]
            print(
                f'  WARNING: Failed to solve at M={mach:.2f}, '
                f'angle={angle:.1f}°: {err_msg}'
            )
            continue

    print(f'  Completed {len(shock_angle_values)} points')

# Plot: deflection on x-axis, shock angle on y-axis, colors for different Mach
fig, ax = plt.subplots(figsize=(10, 8))

cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, len(mach_values)))

for mach, color in zip(mach_values, colors):
    ax.plot(
        results[mach]['deflections'],
        results[mach]['shock_angles'],
        'o-',
        label=f'M = {mach:.1f}',
        color=color,
        linewidth=2,
        markersize=6,
    )

ax.set_xlabel('Deflection Angle (°)', fontsize=12, fontweight='bold')
ax.set_ylabel('Shock Angle (°)', fontsize=12, fontweight='bold')
ax.set_title('Oblique Shock: Deflection vs Shock Angle', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=11)
ax.set_xlim(left=0)
ax.set_ylim(5, 95)

plt.tight_layout()
plt.show()
