import logging
import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity

from adet.solution import solve_root_problem
from adet.assembly import CasadiSystem
from adet.equations import EquationBase
from adet.equations.fundamental import (
    ConstRelEnthalpy,
    Kinematics,
    MassAreaRelation,
    MassConservation,
    TotalStaticMatching,
)
from adet.equations.nondimensional import AbsoluteMachNumber
from adet.equations.special import ThermoVarsAdder
from adet.equations.utils import safe_abs, safe_if_else
from adet.fluid.settings import ExternalFluidModel, FluidSettings
from adet.losses.basic import PercentageEntropyLoss
from adet.registries import GuessRegistry, VariableBoundsRegistry
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger

logger = logging.getLogger(__name__)
setup_logger(logger)


class VoluteDesignFister(EquationBase):
    def residual(
        self,
        geo_rr1,
        kin_Vt1,
        oth_massflow1,
        stc_rhomass1,
        geo_rr_vol0,
        # geo_height1,
    ):
        volume_flow_rate = oth_massflow1 / stc_rhomass1
        fister_constant_K = (
            4 * np.pi**2 * geo_rr1 * kin_Vt1 / volume_flow_rate
        )  # Constant

        # Calculate radius_fister at volute inlet
        radius_fister = 2 * np.pi / fister_constant_K + np.sqrt(
            2 * geo_rr1 * 2 * np.pi / fister_constant_K
        )

        # # Corrected radius
        # area_volute = np.pi * radius_fister**2
        # radius_fister_corr = -geo_height1 * radius_fister / (4 * np.pi) + np.sqrt(
        #     (geo_height1 * radius_fister / (4 * np.pi)) ** 2 + area_volute / np.pi
        # )

        return geo_rr_vol0 - radius_fister


class VoluteDesingStepanoff(EquationBase):
    def residual(self, oth_massflow1, stc_rhomass1, kin_Vt1, geo_rr_vol0):
        volume_flow_rate = oth_massflow1 / stc_rhomass1
        # Stepanoff approach
        area_stepanoff = volume_flow_rate / (kin_Vt1)
        return geo_rr_vol0 - np.sqrt(area_stepanoff / np.pi)


class ConstantTangVelocity(EquationBase):
    def residual(self, kin_V0, kin_Vt1):
        return kin_V0 - kin_Vt1


class VoluteAreas(EquationBase):
    def residual(
        self,
        geo_eff_area0,
        geo_eff_area1,
        geo_rr0,
        geo_rr1,
        geo_rr_vol0,
        geo_height1,
        geo_radiusRatio1,
    ):
        r1 = geo_eff_area0 - np.pi * geo_rr_vol0**2
        r2 = geo_eff_area1 - 2 * np.pi * geo_rr1 * geo_height1
        r3 = geo_radiusRatio1 - geo_rr0 / geo_rr1
        r4 = geo_rr0 - (geo_rr1 + geo_rr_vol0)

        return r1, r2, r3, r4


class VoluteLoss(EquationBase):
    def residual(
        self,
        oth_f1Coeff1,
        oth_f2Coeff1,
        geo_eff_area0,
        geo_eff_area1,
        kin_Vt1,
        kin_Vm1,
        geo_rr0,
        geo_rr1,
        tot_p0,
        tot_p1,
        stc_p1,
    ):
        area_ratio = geo_eff_area0 / geo_eff_area1
        swirl = kin_Vt1 / kin_Vm1

        product = area_ratio * swirl

        k_m = oth_f1Coeff1 / (1 + swirl**2)
        k_theta = (
            oth_f2Coeff1
            * (geo_rr1 / geo_rr0) ** 2
            * (swirl - 1 / area_ratio) ** 2
            / (1 + swirl**2)
        )

        k_theta = safe_if_else(product > 1, k_theta, 0.0)

        deltaPt = (tot_p1 - stc_p1) * (k_m + k_theta)

        return tot_p1 - (tot_p0 - deltaPt)


def plot_volute(designs_dict, num_points=1000):
    """Plot all volute designs in a single figure.

    Args:
        designs_dict: Dictionary with design names as keys and (n0, n1) tuples as values
        num_points: Number of points for radius distribution
    """
    fig = plt.figure(figsize=(13, 13), dpi=150)
    ax = fig.add_subplot(111, projection='polar')

    for design_name, (n0, n1) in designs_dict.items():
        inlet_radius = n0.geo.rr_vol
        stator_radius = n1.geo.rr[0]

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
        designs_dict: Dictionary with design names as keys and (n0, n1) tuples as values
        num_points: Number of points for area distribution
    """
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    for design_name, (n0, n1) in designs_dict.items():
        inlet_radius = n0.geo.rr_vol
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

    fig.show()


class ConstantAngMomentum(EquationBase):
    def residual(self, kin_V0, geo_rr0, kin_Vt1, geo_rr1):
        # Use the completely tangential inlet velocity
        return geo_rr0 * kin_V0 - safe_abs(geo_rr1 * kin_Vt1)


# Basic equations
EQUATIONS = {
    Kinematics(): 0,
    Kinematics(): 1,
    VoluteAreas(): (0, 1),
    MassAreaRelation(): 0,
    MassAreaRelation(): 1,
    ThermoVarsAdder(): 0,
    ThermoVarsAdder(): 1,
    TotalStaticMatching(): 0,
    TotalStaticMatching(): 1,
    AbsoluteMachNumber(): 0,
    AbsoluteMachNumber(): 1,
    MassConservation(): (0, 1),
    ConstRelEnthalpy(): (0, 1),
}


if __name__ == '__main__':
    VariableBoundsRegistry().reset()
    VariableBoundsRegistry().set('mach', (0, 1.0))
    GuessRegistry().reset()
    GuessRegistry().set_fallback_value(0.8)

    design_methods = [
        'whitfield',
        'stepanoff',
        'fister',
    ]
    results = {}

    INLET = {
        'kin': {
            'omega': 0.0,
            'alpha': 0.0,
        },
        'geo': {
            # 'rr_vol': 0.02,
        },
    }

    OUTLET = {
        'tot': {
            'p': Quantity(18.1, 'bar'),
            'T': Quantity(300, 'degC'),
        },
        'kin': {
            'alpha': Quantity(65, 'deg'),
            'omega': 0.0,
        },
        'geo': {
            'height': 0.002,
            'rr': Quantity(37.5, 'mm'),  # Stator inlet
            # 'radiusRatio': 1.8,
        },
        'oth': {
            'f1Coeff': 0.8,
            'f2Coeff': 0.8,
            'massflow': 0.132,
        },
    }

    for DESIGN_METHOD in design_methods:
        print(f'\n--- Solving for {DESIGN_METHOD} design method ---')

        system = CasadiSystem()

        fluid_model = ExternalFluidModel(
            DebugAbstractState('HEOS', 'MM'),
        )
        fluid_settings = FluidSettings(fluid_model, ('p', 'T'))
        system.fluid_settings = fluid_settings

        for eq, pos in EQUATIONS.items():
            system.add_equation(eq, pos)

        match DESIGN_METHOD:
            case 'whitfield':
                system.add_equation(VoluteLoss(), (0, 1))  # 1
                system.add_equation(ConstantAngMomentum(), (0, 1))  # 1
            case 'fister':
                system.add_equation(VoluteDesignFister(), (0, 1))  # 1
                # system.add_equation(ConstantAngMomentum(), (0, 1))  # 1
                system.add_equation(PercentageEntropyLoss(), (0, 1))  # 1
            case 'stepanoff':
                system.add_equation(VoluteDesingStepanoff(), (0, 1))  # 1
                # system.add_equation(ConstantTangVelocity(), (0, 1))
                system.add_equation(PercentageEntropyLoss(), (0, 1))  # 1

        system.add_boundary_conditions(INLET, 0)
        system.add_boundary_conditions(OUTLET, 1)
        system.build()

        rootfinder = system.make_rootfinder(
            'ipopt',
            opts={
                'error_on_fail': False,
            },
        )

        x0 = system.get_scaled_guess()
        kn = system.get_scaled_constraints()
        bnd = system.get_arguments_bounds()

        solution = solve_root_problem(
            rootfinder,
            x0,
            kn,
            bnd,
            suppress_output=False,
        )

        system.write_solution_to_nodes(solution)

        n0 = system.nodes[0]
        n1 = system.nodes[1]

        results[DESIGN_METHOD] = (n0, n1)

        print(f'Volute inlet section radius is {n0.geo.rr_vol[0] * 1000:.3f} mm')
        print(
            f'Volute inlet area is '
            f'{n0.geo.get("eff_area").to("cm**2").magnitude[0]:.3f} cm**2'
        )
        print(f'Volute inlet velocity is {n0.kin.V[0]:.3f} m/s')
        print(f'Volute inlet centroid radius is {n0.geo.rr[0]:.3f} m')
        print(f'Volute outlet velocity is {n1.kin.V[0]:.3f} m/s')
        print(f'Radius ratio is {n1.geo.radiusRatio}')

    # Plot all three designs
    plot_volute(results)
    plot_volute_area_trend(results)
