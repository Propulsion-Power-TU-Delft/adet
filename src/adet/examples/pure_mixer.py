import numpy as np
from pint import Quantity
import matplotlib.pyplot as plt


from adet.assembly import CasadiSystem
from adet.equations import EquationBase
from adet.equations.fundamental import Kinematics, TotalStaticMatching
from adet.equations.nondimensional import AbsoluteMachNumber, StaticPressRatio
from adet.equations.special import ThermoVarsAdder
from adet.equations.utils import safe_abs
from adet.fluid.settings import AnalyticalFluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.solution import solve_root_problem


class BalanceEquations(EquationBase):
    def residual(
        self,
        kin_W0,
        kin_W1,
        stc_rhomass0,
        stc_rhomass1,
        kin_beta0,
        kin_beta1,
        geo_bld_thick0,
        oth_disp_thick0,
        oth_mom_thick0,
        oth_p_base0,
        rlt_hmass0,
        rlt_hmass1,
        stc_p0,
        stc_p1,
        geo_pitch0,
        kin_dev_angle1,
    ):
        opening = geo_pitch0 * np.cos(kin_beta0)

        mass_in = stc_rhomass0 * kin_W0 * (opening - geo_bld_thick0 - oth_disp_thick0)
        mass_out = (
            stc_rhomass1 * kin_W1 * geo_pitch0 * np.cos(kin_beta0 - kin_dev_angle1)
        )

        r_mass = mass_in - mass_out

        # 1 *** X-Momentum
        mom_in_x = (
            mass_in * kin_W0
            - stc_rhomass0 * oth_mom_thick0 * kin_W0**2
            + stc_p0 * (opening - geo_bld_thick0)
            + oth_p_base0 * geo_bld_thick0
        )
        mom_out_x = mass_in * kin_W1 * np.cos(kin_dev_angle1) + stc_p1 * opening
        r_momx = mom_in_x - mom_out_x

        # 2 *** Y-Momentum
        # p_suct = stc_p1
        # area_y = safe_abs(geo_pitch0 * np.sin(kin_beta0))
        # mom_in_y = p_suct * area_y
        # mom_out_y = stc_p1 * area_y + mass_out * kin_W1 * np.sin(kin_dev_angle1)
        # r_momy = mom_in_y - mom_out_y

        r_energ = rlt_hmass0 - rlt_hmass1

        # 3 *** Supersonic vs. subsonic switch
        # r_choke = kin_W0 - stc_speed_sound0
        r_dev = kin_dev_angle1 - (kin_beta0 - kin_beta1)

        return r_mass, r_momx, r_dev, r_energ


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
sys.add_equation(AbsoluteMachNumber(), 0)
sys.add_equation(AbsoluteMachNumber(), 1)
sys.add_equation(StaticPressRatio(), (0, 1))

sys.add_equalities(
    ('kin_omega0', 'kin_omega1'),
    ('geo_rr0', 'geo_rr1'),
)

sys.boundary_conditions[0]['tot'] = {'p': 3e5, 'T': 400}

sys.boundary_conditions[0]['geo'] = {
    'pitch': 1.0,
    'bld_thick': 0.0,
    'rr': 0.1,
}
sys.boundary_conditions[0]['oth'] = {
    'mom_thick': 0,
    'disp_thick': 0,
    'p_base': 1.0,
}

sys.boundary_conditions[0]['kin'] = {
    'omega': 0.0,
    'alpha': Quantity(45.0, 'deg'),
    'mach': 1.0,
}

# sys.boundary_conditions[1]['kin'] = {'mach': 1.4}
sys.boundary_conditions[1]['oth'] = {'pRatio': 0.6}


sys.build()

rtfn_ip = sys.make_rootfinder('ipopt', opts={'error_on_fail': False})
x0 = sys.get_scaled_guess()
kn = sys.get_scaled_constraints()
bnd = sys.get_arguments_bounds()
sol = solve_root_problem(rtfn_ip, x0, kn, bnd)


sys.write_solution_to_nodes(sol)
n0 = sys.nodes[0]
n1 = sys.nodes[1]

print(n0.kin)
print(n1.kin)

# fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
# n0.kin.plot(n0.geo, 10, ax[0])
# n1.kin.plot(n1.geo, 10, ax[1])
# plt.show()

pRatio_idx = sys.constraints.index('oth_pRatio1')
angle_idx = sys.constraints.index('kin_alpha0')

ANGLES = [
    45,
    60,
    75,
]
alphas = [a * np.pi / 180 for a in ANGLES]
pratios = np.linspace(0.2, 1.0, 100)
deviations = {a: [] for a in alphas}
loss_coeffs = {a: [] for a in alphas}
mer_machs = {a: [] for a in alphas}

rtfn_kn = sys.make_rootfinder('kinsol')

RUN_SWEEP = True
if RUN_SWEEP:
    fig, ax = plt.subplots(3, 1, figsize=(4, 10), sharex=True)
    for alpha in alphas:
        kn[angle_idx] = np.array([alpha])
        # First sweep with ipopt
        sol = solve_root_problem(rtfn_ip, sol, kn, bnd)
        for pr in pratios:
            kn[pRatio_idx] = np.array([pr])
            sol = solve_root_problem(rtfn_kn, sol, kn)
            sys.write_solution_to_nodes(sol)

            deviations[alpha].append(n1.kin.dev_angle)
            loss_coeffs[alpha].append((n0.tot.p - n1.tot.p) / (n0.tot.p - n0.stc.p))
            mer_machs[alpha].append(n1.kin.Vm / n1.stc.speed_sound)

        ax[0].plot(
            pratios,
            np.array(deviations[alpha]) * 180 / np.pi,
            label=f'alpha {alpha * 180 / np.pi:.2f}',
        )
        ax[1].plot(
            pratios,
            np.array(loss_coeffs[alpha]) * 100,
            label=f'alpha {alpha * 180 / np.pi:.2f}',
        )
        ax[2].plot(pratios, mer_machs[alpha], label=f'alpha {alpha * 180 / np.pi:.2f}')

    [ax[i].legend() for i in range(3)]
    ax[0].grid(alpha=0.8)
    ax[1].grid(alpha=0.8)
    ax[2].grid(alpha=0.8)
    ax[0].set_ylim(0, 20)
    ax[1].set_ylim(0, 0.4)
    ax[2].set_ylim(0, 1.1)

    fig.show()
