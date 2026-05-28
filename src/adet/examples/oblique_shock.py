from adet.tools.plotting import setup_mpl
import matplotlib.font_manager as fm
import logging

import matplotlib as mpl
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
    GammaPV,
    RelativeMachNumber,
)
from adet.equations.utils import residual_debugger
from adet.fluid.settings import AnalyticalFluidModel, ExternalFluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables, ThermoVariables

logger = logging.getLogger(__name__)
setup_logger(logger)

RUN_SWEEP = True

EQUATIONS = {
    TotalStaticMatching(): 0,
    AnnulusAreas(): 0,
    MassAreaRelation(): 0,
    AbsoluteMachNumber(): 0,
    RelativeMachNumber(): 0,
    ZeroBlockage(): 0,
    Kinematics(): 0,
    GammaPV(): 0,
    # *** Out
    TotalStaticMatching(): 1,
    AnnulusAreas(): 1,
    MassAreaRelation(): 1,
    AbsoluteMachNumber(): 1,
    RelativeMachNumber(): 1,
    ZeroBlockage(): 1,
    Kinematics(): 1,
    GammaPV(): 1,
    # *** Link
    ObliqueShock(): (0, 1),
}

n0 = NodeVariables(0)
n1 = NodeVariables(1)


# Build system once outside the loops
system = CasadiSystem()

# *** Fluid model
model = AnalyticalFluidModel(IdealGasState(1.4, 287, 2e-5))
abs_state = DebugAbstractState('REFPROP', 'MM')
model = ExternalFluidModel(abs_state)

thrm = ThermoVariables()
fluid_settings = FluidSettings(model, (thrm.Pressure, thrm.Temperature))
system.fluid_settings = fluid_settings

for eq, pos in EQUATIONS.items():
    system.add_equation(eq, pos)

# Initial boundary conditions (shock angle will be updated per iteration)
BC = {
    # *** INLET
    n0.kin.Omega: 0.0,
    n0.geo.RDistr: 0.1,
    n0.geo.HDistr: 0.1,
    n0.tot.Pressure: 18.1e5,
    n0.tot.Temperature: 573.15,
    n0.kin.FlowAngleAbs: Quantity(0, 'deg'),
    n0.kin.Mach: 2.2,  # Placeholder, will be updated
    # *** OUTLET
    n1.kin.Omega: 0.0,
    n1.geo.RDistr: 0.1,
    n1.geo.HDistr: 0.1,
    n1.oth.ShockAngle: Quantity(70, 'deg'),  # Placeholder, will be updated
}

# NOTE:
# - mach0=2 with shock angle of 60 deg => mach1=1.2

system.add_boundary_conditions(BC)
system.build()

rtfn = system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        # 'ipopt.max_wall_time': 5,
        # 'ipopt.print_level': 0,
    },
)

x0 = system.get_scaled_guess()
kn = system.get_scaled_constraints()
bnd = system.get_arguments_bounds(
    {
        n0.stc.Temperature.Glob: (100, 600),
        n0.stc.Pressure.Glob: (1e2, 1e9),
    },
)

sol = solve_root_problem(rtfn, x0, kn, bnd, suppress_output=False)
rtfn = system.make_rootfinder('kinsol')
sol = solve_root_problem(rtfn, sol, kn, suppress_output=False)

sol_data = system.sol_to_dict(sol)

globals().update(residual_debugger(ObliqueShock(), [0, 1], sol_data))

if RUN_SWEEP:
    # Parametric sweep ranges
    mach_values = np.linspace(1.5, 3.0, 10)
    shock_angle_values = np.linspace(90, 30, 60)
    results = {}
    for mach in mach_values:
        results[mach] = {'shock_angles': [], 'deflections': [], 'outlet_machs': []}

    sol_data = {}
    sol_dict_nrm_shk = {}
    for mach_idx, mach in enumerate(mach_values):
        print(f'Mach = {mach:.2f}  [{mach_idx + 1}/{len(mach_values)}]')

        for idx, angle in enumerate(shock_angle_values):
            # Update Mach in system.data
            system.data.boun_cond[n0.kin.Mach] = mach
            # Update shock angle in system.data (convert to radians)
            system.data.boun_cond[n1.oth.ShockAngle] = np.radians(angle)

            if idx == 0:
                precursor = sol_dict_nrm_shk
            else:
                precursor = sol_data
            x0 = system.get_scaled_guess(sol_dict_nrm_shk)
            kn = system.get_scaled_constraints()

            bnd = system.get_arguments_bounds(
                {
                    n0.stc.Temperature.Glob: (100, 600),
                    n0.stc.Pressure.Glob: (1e2, 1e9),
                    n1.kin.Mach: (0.0, 0.88 * mach),
                }
            )

            rtfn = system.make_rootfinder(
                'ipopt',
                opts={
                    'error_on_fail': False,
                    'ipopt.max_wall_time': 2.5,
                    'ipopt.print_level': 0,
                },
            )
            sol = solve_root_problem(rtfn, x0, kn, bnd, suppress_output=True)

            rtfn = system.make_rootfinder('kinsol')

            try:
                sol = solve_root_problem(rtfn, sol, kn, suppress_output=True)
                sol_data = system.sol_to_dict(sol)
                print(f'Outlet mach is {sol_data[n1.kin.Mach]}')
                if idx == 0:
                    sol_dict_nrm_shk = sol_data
                # Extract deflection angle (convert from radians to degrees)
                deflection_val = sol_data[n1.oth.ShockDeflection]
                deflection_rad = float(np.atleast_1d(deflection_val)[0])

                # Normalize angle to  to handle periodic convergence
                deflection_rad = np.arctan2(
                    np.sin(deflection_rad), np.cos(deflection_rad)
                )
                deflection_deg = deflection_rad * 180 / np.pi

                results[mach]['shock_angles'].append(angle)
                results[mach]['deflections'].append(deflection_deg)
                results[mach]['outlet_machs'].append(sol_data[n1.kin.Mach])
            except (RuntimeError, ValueError) as e:
                err_msg = str(e)[:50]
                print(
                    f'  WARNING: Failed to solve at M={mach:.2f}, '
                    f'angle={angle:.1f}°: {err_msg}'
                )
                continue

        print(f'  Completed {len(shock_angle_values)} points')

    # *** PLOTS
    setup_mpl(
        {
            'font.family': 'EB Garamond',
            'font.size': '20',
        }
    )
    cmap = plt.get_cmap('plasma')

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = cmap(np.linspace(0, 0.85, len(mach_values)))

    for mach, color in zip(mach_values, colors):
        ax.plot(
            results[mach]['deflections'],
            results[mach]['shock_angles'],
            '-',
            label=r'$M_{in}$' f' = {mach:.1f}',
            color=color,
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel(r'Deflection angle $\theta$ [deg]')
    ax.set_ylabel(r'Shock angle $\beta$ [deg]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_xlim(left=0)
    # ax.set_ylim(5, 95)

    plt.tight_layout()
    plt.show()
