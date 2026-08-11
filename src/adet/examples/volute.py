"""
Volute design methods for a supersonic ORC (ORCHID) turbine.
Area distribution is assumed linear
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity

from adet.assemblers import CasadiSystem
from adet.equations import EquationBase
from adet.equations.fundamental import (
    ConstRelEnthalpy,
    Kinematics,
    MassAreaRelation,
    MassConservation,
    TotalStaticMatching,
)
from adet.equations.nondimensional import AbsoluteMachNumber
from adet.equations.utils import safe_abs, safe_if_else
from adet.fluid.settings import FluidSettings
from adet.losses.basic import IsentropicLink
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables, ThermoVariables
from adet.varspec import VarSpec

logger = logging.getLogger(__name__)
setup_logger(logger)

# Commons
n0 = NodeVariables(0)  # Volute inlet
n1 = NodeVariables(1)  # Volute outlet (stator inlet)

# Volute inlet cross-section radius
R_volute = VarSpec('r_vol', 'm', node=0, bounds=(0.0, 100.0), guess=1.0)

# Whitfield volute loss coefficients
F1CoeffLoss = VarSpec('f1_coeff', 'dimensionless', node=1, guess=0.8)
F2CoeffLoss = VarSpec('f2_coeff', 'dimensionless', node=1, guess=0.8)


class VoluteDesignFister(EquationBase):
    def residual(
        self,
        rr1: n1.geo.RDistr.Hint,
        vt1: n1.kin.V_tan.Hint,
        mf1: n1.oth.StreamMassFlow.Hint,
        rho1: n1.stc.Density.Hint,
        rr_vol0: R_volute.Hint,
    ):
        volume_flow_rate = mf1 / rho1
        fister_constant_K = 4 * np.pi**2 * rr1 * vt1 / volume_flow_rate

        radius_fister = 2 * np.pi / fister_constant_K + np.sqrt(
            2 * rr1 * 2 * np.pi / fister_constant_K
        )

        return rr_vol0 - radius_fister


class ConstantTangVelocity(EquationBase):
    def residual(
        self,
        v0: n0.kin.V_mag.Hint,
        vt1: n1.kin.V_tan.Hint,
    ):
        return v0 - vt1


class ConstantAngMomentum(EquationBase):
    def residual(
        self,
        v0: n0.kin.V_mag.Hint,
        rr0: n0.geo.RDistr.Hint,
        vt1: n1.kin.V_tan.Hint,
        rr1: n1.geo.RDistr.Hint,
    ):
        return rr0 * v0 - safe_abs(rr1 * vt1)


class VoluteAreas(EquationBase):
    def residual(
        self,
        a_eff0: n0.geo.EffArea.Hint,
        a_eff1: n1.geo.EffArea.Hint,
        rr0: n0.geo.RDistr.Hint,
        rr1: n1.geo.RDistr.Hint,
        rr_vol0: R_volute.Hint,
        h1: n1.geo.HDistr.Hint,
    ):
        r1 = a_eff0 - np.pi * rr_vol0**2
        r2 = a_eff1 - 2 * np.pi * rr1 * h1
        r3 = rr0 - (rr1 + rr_vol0)

        return r1, r2, r3


class VoluteLossWhitfield(EquationBase):
    def residual(
        self,
        f1: F1CoeffLoss.Hint,
        f2: F2CoeffLoss.Hint,
        a_eff0: n0.geo.EffArea.Hint,
        a_eff1: n1.geo.EffArea.Hint,
        vt1: n1.kin.V_tan.Hint,
        vm1: n1.kin.V_mer.Hint,
        rr0: n0.geo.RDistr.Hint,
        rr1: n1.geo.RDistr.Hint,
        pt0: n0.tot.Pressure.Hint,
        pt1: n1.tot.Pressure.Hint,
        p1: n1.stc.Pressure.Hint,
    ):
        area_ratio = a_eff0 / a_eff1
        swirl = vt1 / vm1

        product = area_ratio * swirl

        k_m = f1 / (1 + swirl**2)
        k_theta = f2 * (rr1 / rr0) ** 2 * (swirl - 1 / area_ratio) ** 2 / (1 + swirl**2)

        k_theta = safe_if_else(product > 1, k_theta, 0.0)

        deltaPt = (pt1 - p1) * (k_m + k_theta)

        return pt1 - (pt0 - deltaPt)


def plot_volute(designs_dict, num_points=1000):
    """Plot all volute designs in a single figure.

    Args:
        designs_dict: Dictionary with design names as keys and (sol_dict, rr1) tuples
        num_points: Number of points for radius distribution
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')

    stator_radius = None
    for design_name, (sol_dict, rr1) in designs_dict.items():
        inlet_radius = sol_dict[R_volute]
        stator_radius = rr1

        theta_distribution = np.linspace(0, 2 * np.pi, num_points)
        inlet_area = np.pi * inlet_radius**2
        area_distribution = np.linspace(0, inlet_area, num_points)
        radius_distribution = np.sqrt(area_distribution / np.pi)
        outer_radius = stator_radius + 2 * radius_distribution

        if design_name == 'stepanoff':
            label = r'$V_{\theta} = \mathrm{const.}$'
        elif design_name == 'fister':
            label = r'$rV_{\theta} = \mathrm{const.}$'
        elif design_name == 'whitfield':
            label = r'Whitfield'
        ax.plot(
            theta_distribution,
            outer_radius,
            linewidth=3.5,
            label=label,
        )

    CURR_RADIUS = 0.02
    final_area = np.pi * CURR_RADIUS**2
    theta_distribution = np.linspace(0, 2 * np.pi, num_points)
    area_distribution = np.linspace(0, final_area, num_points)
    rad_distribution = stator_radius + 2 * np.sqrt(area_distribution / np.pi)

    ax.plot(
        theta_distribution,
        rad_distribution,
        label='Current design',
        linewidth=3.5,
    )

    ax.set_xlabel(r'$\theta$ [rad]', fontsize=30)
    ax.set_ylabel(r'$r$ [m]', fontsize=30)
    ax.tick_params(labelsize=15)

    plt.tight_layout()
    ax.plot(
        theta_distribution,
        stator_radius * np.ones(num_points),
        color='k',
        linewidth=2.0,
        linestyle='--',
        label='Stator Inlet',
    )
    ax.legend(fontsize=15, loc='lower left')
    ax.grid(alpha=0.6)

    fig.show()


def plot_volute_area_trend(designs_dict, num_points=1000):
    """Plot area trend for all volute designs in a linear graph.

    Args:
        designs_dict: Dictionary with design names as keys and (sol_dict, rr1) tuples
        num_points: Number of points for area distribution
    """
    _, ax = plt.subplots(figsize=(12, 8), dpi=150)

    for design_name, (sol_dict, _) in designs_dict.items():
        inlet_radius = sol_dict[R_volute]
        theta_distribution = np.linspace(0, 2 * np.pi, num_points)
        inlet_area = np.pi * inlet_radius**2
        area_distribution = np.linspace(0, inlet_area, num_points)

        if design_name == 'stepanoff':
            label = r'$V_{\theta} = \mathrm{const.}$'
        elif design_name == 'fister':
            label = r'$rV_{\theta} = \mathrm{const.}$'
        elif design_name == 'whitfield':
            label = r'Whitfield'
        else:
            label = design_name

        ax.plot(
            theta_distribution,
            area_distribution * 1e4,  # Convert to cm²
            linewidth=3.5,
            label=label,
        )

    # Current design
    CURR_RADIUS = 0.02
    final_area = np.pi * CURR_RADIUS**2
    theta_distribution = np.linspace(0, 2 * np.pi, num_points)
    area_distribution = np.linspace(0, final_area, num_points)

    ax.plot(
        theta_distribution,
        area_distribution * 1e4,  # Convert to cm²
        linewidth=3.5,
        label='Current design',
    )

    ax.set_xlabel(r'$\theta$ [rad]', fontsize=24)
    ax.set_ylabel(r'Area [cm$^2$]', fontsize=24)
    ax.tick_params(labelsize=17)
    ax.legend(fontsize=19)
    ax.grid(alpha=0.7)
    # fig.savefig(
    #     'C:\\Users\\fvaccari\\OneDrive - Delft University of Technology\\latex'
    #     '\\gpps26_ORCHID\\Images\\volute_area_comparison.pdf'
    # )
    plt.tight_layout()

    plt.show()


# Basic equations
EQUATIONS = {
    Kinematics(): 0,
    Kinematics(): 1,
    VoluteAreas(): (0, 1),
    MassAreaRelation(): 0,
    MassAreaRelation(): 1,
    TotalStaticMatching(): 0,
    TotalStaticMatching(): 1,
    AbsoluteMachNumber(): 0,
    AbsoluteMachNumber(): 1,
    MassConservation(): (0, 1),
    ConstRelEnthalpy(): (0, 1),
}


if __name__ == '__main__':
    design_methods = [
        'whitfield',
        'stepanoff',
        'fister',
    ]
    results = {}

    for DESIGN_METHOD in design_methods:
        print(f'\n--- Solving for {DESIGN_METHOD} design method ---')

        system = CasadiSystem()

        fluid_state = DebugAbstractState('HEOS', 'MM')
        thrm = ThermoVariables()
        fluid_settings = FluidSettings(
            fluid_state=fluid_state,
            update_variables=(thrm.Pressure, thrm.Temperature),
        )
        system.fluid_settings = fluid_settings

        for eq, pos in EQUATIONS.items():
            system.add_equation(eq, pos)

        match DESIGN_METHOD:
            case 'whitfield':
                system.add_equation(VoluteLossWhitfield(), (0, 1))
                system.add_equation(ConstantAngMomentum(), (0, 1))
            case 'fister':
                system.add_equation(VoluteDesignFister(), (0, 1))
                # system.add_equation(ConstantAngMomentum(), (0, 1))
                system.add_equation(IsentropicLink(), (0, 1))
            case 'stepanoff':
                system.add_equation(ConstantTangVelocity(), (0, 1))
                system.add_equation(IsentropicLink(), (0, 1))

        INLET_BC = {
            n0.kin.Omega: 0.0,
            n0.kin.FlowAngleAbs: 0.0,
        }

        OUTLET_BC = {
            n1.tot.Pressure: Quantity(18.1, 'bar'),
            n1.tot.Temperature: Quantity(300, 'degC'),
            n1.kin.FlowAngleAbs: Quantity(65, 'deg'),
            n1.oth.StreamMassFlow: 0.132,
            n1.kin.Omega: 0.0,
            n1.geo.HDistr: 0.002,
            n1.geo.RDistr: Quantity(37.5, 'mm'),
            F1CoeffLoss: 0.8,
            F2CoeffLoss: 0.8,
        }

        system.add_boundary_conditions(INLET_BC)
        system.add_boundary_conditions(OUTLET_BC)
        system.build()

        rootfinder = system.make_rootfinder(
            'ipopt',
            opts={
                'error_on_fail': False,
            },
        )

        x0 = system.get_guess(fallback=0.8)
        kn = system.get_boundary_conds()
        bnd = system.get_bounds(
            {
                # n0.stc.Pressure.Glob: (100, 1e10),
                n0.stc.Temperature.Glob: (300, 580),
                n0.kin.Mach.Glob: (0, 0.8),
            }
        )

        sol = solve_root_problem(
            rootfinder,
            x0,
            kn,
            bnd,
            suppress_output=False,
        )

        sol_dict = system.sol_to_dict(sol)

        rr1 = float(system.data.boun_cond[n1.geo.RDistr])
        results[DESIGN_METHOD] = (sol_dict, rr1)

        print(f'Volute inlet section radius is {sol_dict[R_volute]} mm')
        print(f'Volute outlet radius is {rr1 * 1000:.3f} mm')
        print(f'Volute outlet velocity is {sol_dict[n1.kin.V_mag]} m/s')

    # Plot all designs
    plot_volute(results)
    plot_volute_area_trend(results)
