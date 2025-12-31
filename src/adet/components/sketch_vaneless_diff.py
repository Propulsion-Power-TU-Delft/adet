"""
This is just a sketch for a vaneless diffuser solution
using a real gas model
"""

import CoolProp as cp
import casadi as cs
import numpy as np
import matplotlib.pyplot as plt

from adet.fluid.casadi_eos import CasadiEos
from adet.tools.coolprop_utils import DebugAbstractState


def solve_diffuser_problem(
    r2, r3, p2, T2, Vm2, Vt2, casadi_eos, entropy_gen_pct, num_points
):
    casadi_eos.update(cp.PT_INPUTS, p2, T2)

    s2 = casadi_eos.smass()
    rho2 = casadi_eos.rhomass()

    get_density_PS = CasadiEos(
        'casadi_eos', casadi_eos, cp.PSmass_INPUTS, ['rhomass'], 1
    )

    # CasADi setup
    r_dist = np.linspace(r2, r3, num_points)

    # State: [Vm, Vt, p, theta]
    r = cs.SX.sym('r')  # pyright:ignore

    Vm = cs.SX.sym('Vm')  # pyright:ignore
    Vt = cs.SX.sym('Vt')  # pyright:ignore

    p = cs.SX.sym('p')  # pyright:ignore
    s = cs.SX.sym('s')  # pyright:ignore
    rho = cs.SX.sym('rho')  # pyright:ignore

    theta = cs.SX.sym('theta')  # pyright:ignore

    ode_variables = cs.vertcat(r, Vm, Vt, p, theta)

    # Differential equations
    dr_dr = 1.0  # Compute radius distribution with the rest
    dVm_dr = -Vm / r
    dVt_dr = -Vt / r
    dp_dr = rho * (Vt**2 / r - Vm * dVm_dr)
    dtheta_dr = Vt / (r * Vm)  # dtheta/dr = (dtheta/dt) / (dr/dt) = (Vt/r) / Vm

    ode = cs.vertcat(dr_dr, dVm_dr, dVt_dr, dp_dr, dtheta_dr)

    # Simple model
    # Linear entropy generation between in and out

    deltaS = entropy_gen_pct * s2

    def entropy_gen(r):
        return s2 + deltaS / (r3 - r2) * (r - r2)

    density_casadi_eos = rho - get_density_PS(p, s)
    entropy_inc = s - entropy_gen(r)

    alg = cs.vertcat(density_casadi_eos, entropy_inc)
    alg_variables = cs.vertcat(rho, s)

    # Integrator
    dae = {
        'x': ode_variables,
        'z': alg_variables,
        'ode': ode,
        'alg': alg,
    }
    integrator = cs.integrator('F', 'idas', dae, r2, r_dist)

    # Initial conditions
    x0 = np.array([r2, Vm2, Vt2, p2, 0.0])  # Initial theta is 0
    z0 = np.array([rho2, s2])

    result = integrator(x0=x0, z0=z0)

    x_solution = result['xf']
    z_solution = result['zf']

    solution = {}

    solution['r'] = x_solution[0, :].toarray().flatten()
    solution['Vm'] = x_solution[1, :].toarray().flatten()
    solution['Vt'] = x_solution[2, :].toarray().flatten()
    solution['p'] = x_solution[3, :].toarray().flatten()
    solution['theta'] = x_solution[4, :].toarray().flatten()
    solution['rho'] = z_solution[0, :].toarray().flatten()
    solution['s'] = z_solution[1, :].toarray().flatten()

    return solution


if __name__ == '__main__':
    # Parameters
    r2 = 0.1  # Impeller exit radius [m]
    r3 = 0.15  # Diffuser exit radius [m]

    # Initial conditions
    Vm2 = 50.0  # Radial velocity [m/s]
    Vt2 = 90.0  # Tangential velocity [m/s]

    p2 = 6e5  # Static pressure [Pa]
    T2 = 500  # Static pressure [Pa]

    casadi_eos = DebugAbstractState('HEOS', 'Water')

    solution = solve_diffuser_problem(r2, r3, p2, T2, Vm2, Vt2, casadi_eos, 0.05, 100)

    # Convert to Cartesian
    x_traj = solution['r'] * np.cos(solution['theta'])
    y_traj = solution['r'] * np.sin(solution['theta'])

    # - - - - - - - - - - - - - - - - - - - - PLOTS
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Trajectory
    axes[0, 0].plot(x_traj * 1000, y_traj * 1000, 'b-', linewidth=2)
    axes[0, 0].plot(
        x_traj[0] * 1000,
        y_traj[0] * 1000,
        'ko',
        markersize=10,
        label='Start',
    )
    axes[0, 0].plot(
        x_traj[-1] * 1000, y_traj[-1] * 1000, 'ro', markersize=10, label='End'
    )
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    axes[0, 0].plot(
        r2 * np.cos(theta_circle) * 1000,
        r2 * np.sin(theta_circle) * 1000,
        'k--',
        alpha=0.5,
    )
    axes[0, 0].plot(
        r3 * np.cos(theta_circle) * 1000,
        r3 * np.sin(theta_circle) * 1000,
        'k--',
        alpha=0.5,
    )
    axes[0, 0].set_xlabel('X [mm]')
    axes[0, 0].set_ylabel('Y [mm]')
    axes[0, 0].set_title('Particle Trajectory')
    axes[0, 0].set_aspect('equal')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Velocities
    axes[0, 1].plot(solution['r'], solution['Vm'], label='$V_m$', linewidth=2)
    axes[0, 1].plot(solution['r'], solution['Vt'], label='$V_t$', linewidth=2)
    axes[0, 1].set_xlabel('Radius [m]')
    axes[0, 1].set_ylabel('Velocity [m/s]')
    axes[0, 1].set_title('Velocity Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(solution['r'], solution['rho'], label='$\\rho$', linewidth=2)
    axes[1, 0].set_ylabel('$\\rho$ [$kg/m^3$]')
    axes[1, 0].set_xlabel('Radius [m]')

    axes[1, 1].plot(solution['r'], solution['p'], label='$p$', linewidth=2)
    axes[1, 1].set_ylabel('Pressure [Pa]')
    axes[1, 1].set_xlabel('Radius [m]')

    plt.tight_layout()
    plt.show()

    # Summary
    print(f'\nPressure rise: {(solution["p"][-1] - p2) / 1000:.2f} kPa')
    print(f'Angular travel: {np.degrees(solution["theta"][-1]):.1f}°')
