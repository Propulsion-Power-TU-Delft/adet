import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity

from adet.assembly import CasadiSystem, solve_root_problem
from adet.equations import EquationBase
from adet.equations.fundamental import (
    ConstantEnergy,
    Kinematics,
    MassAreaRelation,
    MassConservation,
    TotalStaticMatching,
)
from adet.equations.nondimensional import AbsoluteMachNumber
from adet.equations.special import ThermoVarsAdder
from adet.equations.utils import safe_abs
from adet.fluid.settings import ExternalFluidModel, FluidSettings
from adet.losses.basic import PercentageEntropyLoss
from adet.registries import GuessRegistry, VariableBoundsRegistry
from adet.tools.coolprop_utils import DebugAbstractState


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

        return r1, r2, r3


class VoluteDesignFister(EquationBase):
    # manual_units = ('m',)

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

        # area_volute = np.pi * radius_fister**2
        # radius_fister_corrected = -geo_height1 * radius_fister / (4 * np.pi) + np.sqrt(
        #     (geo_height1 * radius_fister / (4 * np.pi)) ** 2 + area_volute / np.pi
        # )

        return geo_rr_vol0 - radius_fister


class VoluteDesingStepanoff(EquationBase):
    def residual(self, oth_massflow1, stc_rhomass1, kin_Vt1, geo_rr_vol0):
        volume_flow_rate = oth_massflow1 / stc_rhomass1
        # Stepanoff approach
        area_stepanoff = volume_flow_rate / (kin_Vt1)
        return geo_rr_vol0 - np.sqrt(area_stepanoff / np.pi)


class VoluteLoss(EquationBase):
    manual_units = ('Pa',)

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

        k_theta = cs.if_else(product > 1, k_theta, 0.0)

        deltaPt = (tot_p1 - stc_p1) * (k_m + k_theta)

        return tot_p1 - (tot_p0 - deltaPt)


def plot_volute(inlet_radius, stator_radius, num_points=8):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    theta_distribution = np.linspace(0, 2 * np.pi, num_points)
    radius_distribution = np.linspace(0, inlet_radius[0], num_points)
    outer_radius = stator_radius + 2 * radius_distribution

    ax.plot(theta_distribution, outer_radius, 'k')
    ax.plot(theta_distribution, stator_radius * np.ones(num_points), 'k')

    fig.show()


# Basic equations
EQUATIONS = {
    Kinematics(): 0,
    Kinematics(): 1,
    MassAreaRelation(): 0,
    MassAreaRelation(): 1,
    ThermoVarsAdder(): 0,
    ThermoVarsAdder(): 1,
    TotalStaticMatching(): 0,
    TotalStaticMatching(): 1,
    AbsoluteMachNumber(): 0,
    AbsoluteMachNumber(): 1,
    VoluteAreas(): (0, 1),
    MassConservation(): (0, 1),
    ConstantEnergy(): (0, 1),
}


class VoluteFreeVortex(EquationBase):
    def residual(self, kin_V0, geo_rr0, kin_Vt1, geo_rr1):
        # Use the completely tangential inlet velocity
        return geo_rr0 * kin_V0 - safe_abs(geo_rr1 * kin_Vt1)


if __name__ == '__main__':
    VariableBoundsRegistry().set('mach', (0, 1.0))
    GuessRegistry().set_fallback_value(0.8)

    method_map = {1: 'conservation', 2: 'fister', 3: 'stepanoff'}
    DESIGN_METHOD = method_map[int(input(f'Choose design method {method_map} = '))]

    system = CasadiSystem()

    fluid_model = ExternalFluidModel(
        DebugAbstractState('REFPROP', 'MM'),
    )
    fluid_settings = FluidSettings(fluid_model, ('p', 'T'))
    system.fluid_settings = fluid_settings

    for eq, pos in EQUATIONS.items():
        system.add_equation(eq, pos)

    match DESIGN_METHOD:
        case 'conservation':
            system.add_equation(VoluteLoss(), (0, 1))  # 1
            system.add_equation(VoluteFreeVortex(), (0, 1))  # 1
        case 'fister':
            system.add_equation(VoluteDesignFister(), (0, 1))  # 1
            system.add_equation(PercentageEntropyLoss(), (0, 1))  # 1
        case 'stepanoff':
            system.add_equation(VoluteDesingStepanoff(), (0, 1))  # 1
            system.add_equation(PercentageEntropyLoss(), (0, 1))  # 1

    # Losses along the volute

    INLET = {
        'tot': {
            'p': Quantity(18.1, 'bar'),
            'T': Quantity(300, 'degC'),
        },
        'kin': {
            # 'V': 4,  # Inlet velocity
            'omega': 0.0,
            'alpha': 0.0,
        },
    }

    OUTLET = {
        'kin': {
            'alpha': Quantity(65, 'deg'),
            'omega': 0.0,
        },
        'geo': {
            'height': 0.002,
            'rr': Quantity(37.5, 'mm'),
            'radiusRatio': 1.8,  # Used in first method
        },
        'oth': {
            'f1Coeff': 0.8,
            'f2Coeff': 0.8,
            'massflow': 0.132,
        },
    }

    system.add_boundary_conditions(INLET, 0)
    system.add_boundary_conditions(OUTLET, 1)
    system.build()

    rootfinder = system.make_rootfinder(
        'ipopt',
        opts={
            'error_on_fail': False,
            'ipopt.tol': 1e-8,
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

    plot_volute(n0.geo.rr_vol, n1.geo.rr, 200)

    print(f'Volute inlet section radius is {n0.geo.rr_vol[0] * 1000:.3f} mm')
    print(
        f'Volute inlet area is '
        f'{n0.geo.get("eff_area").to("cm**2").magnitude[0]:.3f} cm**2'
    )
    print(f'Volute inlet velocity is {n0.kin.V[0]:.3f} m/s')
    print(f'Volute inlet centroid radius is {n0.geo.rr[0]:.3f} m')
    print(f'Volute outlet velocity is {n1.kin.V[0]:.3f} m/s')
    print(f'Radius ratio is {n1.geo.radiusRatio}')
