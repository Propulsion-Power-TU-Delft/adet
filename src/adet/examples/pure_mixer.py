import numpy as np
from pint import Quantity
import matplotlib.pyplot as plt


from adet.assembly import CasadiSystem
from adet.equations import EquationBase
from adet.equations.fundamental import Kinematics, TotalStaticMatching
from adet.equations.special import ThermoVarsAdder
from adet.equations.utils import safe_abs
from adet.fluid.settings import AnalyticalFluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.registries import VariableBoundsRegistry
from adet.solution import solve_root_problem


class BalanceEquations(EquationBase):
    def residual(
        self,
        kin_W0,
        stc_rhomass0,
        stc_rhomass1,
        geo_bld_thick0,
        oth_disp_thick0,
        oth_mom_thick0,
        oth_p_base0,
        tot_hmass0,
        tot_hmass1,
        stc_p0,
        stc_speed_sound0,
        stc_p1,
        kin_W1,
        kin_Wm1,
        geo_pitch0,
        kin_beta0,
        kin_beta1,
        kin_dev_angle1,
    ):
        mass_in = (
            stc_rhomass0
            * kin_W0
            * (geo_pitch0 * np.cos(kin_beta0) - geo_bld_thick0 - oth_disp_thick0)
        )

        mass_out = stc_rhomass1 * kin_Wm1 * geo_pitch0

        r_mass = mass_in - mass_out

        # 1 *** X-Momentum
        mom_in_x = (
            mass_in * kin_W0
            - stc_rhomass0 * kin_W0**2 * oth_mom_thick0
            + stc_p0 * (geo_pitch0 * np.cos(kin_beta0) - geo_bld_thick0)
            + oth_p_base0 * geo_bld_thick0
        )
        mom_out_x = mass_out * kin_W1 * np.cos(
            kin_dev_angle1
        ) + stc_p1 * geo_pitch0 * np.cos(kin_beta0)
        r_momx = mom_in_x - mom_out_x

        # 2 *** Y-Momentum
        p_suct = stc_p1
        area_y = safe_abs(geo_pitch0 * np.sin(kin_beta0))
        mom_in_y = p_suct * area_y
        mom_out_y = stc_p1 * area_y + mass_out * kin_W1 * np.sin(kin_dev_angle1)
        r_momy = mom_in_y - mom_out_y

        r_energ = tot_hmass0 - tot_hmass1

        # 3 *** Supersonic vs. subsonic switch
        # r_choke = kin_W0 - stc_speed_sound0
        r_dev = kin_dev_angle1 - (kin_beta0 - kin_beta1)

        return r_mass, r_momx, r_momy, r_dev, r_energ


sys = CasadiSystem(1)
model = AnalyticalFluidModel(IdealGasState(1.4, 287, 1.8e-5))
settings = FluidSettings(model, update_variables=('p', 'hmass'))
sys.fluid_settings = settings

sys.add_equation(BalanceEquations(), (0, 1))

sys.add_equation(Kinematics(), 0)
sys.add_equation(Kinematics(), 1)
sys.add_equation(ThermoVarsAdder(), 0)
sys.add_equation(ThermoVarsAdder(), 1)
sys.add_equation(TotalStaticMatching(), 0)
sys.add_equation(TotalStaticMatching(), 1)

sys.add_equalities(
    ('kin_omega0', 'kin_omega1'),
    ('geo_rr0', 'geo_rr1'),
)

sys.boundary_conditions[0]['tot'] = {'p': 3e5, 'T': 400}

sys.boundary_conditions[0]['geo'] = {
    'pitch': 0.1,
    'bld_thick': 0.002,
    'rr': 0.1,
}
sys.boundary_conditions[0]['oth'] = {
    'mom_thick': 0.002 * 0.075,
    'disp_thick': 0.002 * 0.15,
    'p_base': 0.6 * 3e5,
}

sys.boundary_conditions[0]['kin'] = {
    'omega': 0.0,
    'alpha': Quantity(40.0),
    'V': 30,
}


sys.build()

rtfn = sys.make_rootfinder('ipopt', opts={'error_on_fail': False})
x0 = sys.get_scaled_guess()
kn = sys.get_scaled_constraints()
bnd = sys.get_arguments_bounds()
sol = solve_root_problem(rtfn, x0, kn, bnd)

# rtfn = sys.make_rootfinder('kinsol')
# sol = solve_root_problem(rtfn, x0, kn)


sys.write_solution_to_nodes(sol)
n0 = sys.nodes[0]
n1 = sys.nodes[1]

print(n0.kin)
print(n1.kin)

fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
n0.kin.plot(n0.geo, 10, ax[0])
n1.kin.plot(n1.geo, 10, ax[1])
plt.show()
