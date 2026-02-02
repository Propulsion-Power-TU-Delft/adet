import numpy as np
import casadi as cs

from pint import Quantity
from adet.equations import EquationBase
from adet.assembly import CasadiSystem, solve_root_problem
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

VariableBoundsRegistry().set('mach', (0, 1.0))
GuessRegistry().set_fallback_value(0.8)


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


class VoluteAngularMomentum(EquationBase):
    def residual(self, kin_Vm0, kin_W0, kin_V0, geo_rr0, kin_Vt1, geo_rr1):
        # Minimal velocity triangle
        r1 = kin_Vm0 - kin_V0
        r2 = kin_W0 - kin_V0

        # Use the completely tangential inlet velocity
        r3 = geo_rr0 * kin_V0 - safe_abs(geo_rr1 * kin_Vt1)

        return r1, r2, r3


system = CasadiSystem()

fluid_model = ExternalFluidModel(
    DebugAbstractState('REFPROP', 'MM'),
)
fluid_settings = FluidSettings(fluid_model, ('p', 'T'))
system.fluid_settings = fluid_settings

# Basic equations
system.add_equation(Kinematics(), 1)  # 7
system.add_equation(MassAreaRelation(), 0)  # 1
system.add_equation(MassAreaRelation(), 1)  # 1
system.add_equation(ThermoVarsAdder(), 0)  # 0
system.add_equation(ThermoVarsAdder(), 1)  # 0
system.add_equation(TotalStaticMatching(), 0)  # 4
system.add_equation(TotalStaticMatching(), 1)  # 4
system.add_equation(AbsoluteMachNumber(), 0)
system.add_equation(AbsoluteMachNumber(), 1)

# Conservation equations
system.add_equation(ConstantEnergy(), (0, 1))  # 1
system.add_equation(MassConservation(), (0, 1))  # 1
system.add_equation(VoluteAngularMomentum(), (0, 1))  # 1
system.add_equation(VoluteAreas(), (0, 1))  # 3

# Losses along the volute
# system.add_equation(PercentageEntropyLoss(), (0, 1))  # 1
system.add_equation(VoluteLoss(), (0, 1))  # 1

INLET = {
    'tot': {
        'p': Quantity(18.1, 'bar'),
        'T': Quantity(300, 'degC'),
    },
    'kin': {
        # 'V': 4,  # Inlet velocity
    },
}

OUTLET = {
    'kin': {
        'alpha': Quantity(65, 'deg'),
        'omega': 0.0,
    },
    'geo': {
        'height': 0.002,
        'rr': 1.1 * Quantity(37.5, 'mm'),
        'radiusRatio': 1.8,
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

x0 = system.get_initial_guess()
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

print(f'Volute inlet section radius is {n0.geo.rr_vol[0] * 1000:.3f} mm')
print(
    f'Volute inlet area is {n0.geo.get("eff_area").to("cm**2").magnitude[0]:.3f} cm**2'
)
print(f'Volute inlet velocity is {n0.kin.V[0]:.3f} m/s')
print(f'Volute inlet centroid radius is {n0.geo.rr[0]:.3f} m')
print(f'Volute outlet velocity is {n1.kin.V[0]:.3f} m/s')
print(f'Radius ratio is {n1.geo.radiusRatio}')
