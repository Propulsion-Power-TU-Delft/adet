import ipdb
import jax
import numpy as np
from pint import Quantity
import matplotlib.pyplot as plt
import logging


from adet.assembly import CasadiSystem
from adet.equations import EquationBase
from adet.equations.fundamental import Kinematics, TotalStaticMatching
from adet.equations.nondimensional import AbsoluteMachNumber, StaticPressRatio
from adet.equations.special import ThermoVarsAdder
from adet.equations.utils import residual_debugger, safe_abs
from adet.fluid.settings import AnalyticalFluidModel, ExternalFluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.registries import VariableBoundsRegistry
from adet.solution import solve_root_problem
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.loggers import setup_logger

logger = logging.getLogger(__name__)
setup_logger(logger, logging.INFO, logging.INFO)


def dev_osnaghi(pr, Ma, alpha_a, gamma):
    """
    pr = p1/pa, can be a vector
    """
    chi = gamma / (gamma - 1)

    numerator_linear = gamma / (1 - gamma) * pr * np.tan(alpha_a)

    discriminant = (1 - pr) * (2 * chi * Ma**2 - 1 - (gamma + 1) / (gamma - 1) * pr) + (
        chi * pr * np.tan(alpha_a)
    ) ** 2

    denominator = 1 + gamma * Ma**2 - pr

    tan_delta_alpha = np.where(
        pr > 1,  # shock: pressure rise
        (numerator_linear - np.sqrt(discriminant)) / denominator,  # expansion
        (numerator_linear + np.sqrt(discriminant)) / denominator,
    )

    return np.arctan(tan_delta_alpha)


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
        mass_out = stc_rhomass1 * kin_W1 * geo_pitch0 * np.cos(kin_beta1)

        r_mass = mass_in - mass_out

        # 1 *** X-Momentum
        mom_in_x = (
            mass_in * kin_W0
            + stc_p0 * (opening - geo_bld_thick0)
            + oth_p_base0 * geo_bld_thick0
            - stc_rhomass0 * oth_mom_thick0 * kin_W0**2
        )
        mom_out_x = mass_out * kin_W1 * np.cos(kin_dev_angle1) + stc_p1 * opening
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
        r_dev = kin_dev_angle1 - np.sign(kin_beta0) * (kin_beta0 - kin_beta1)

        return r_mass, r_momx, r_dev, r_energ


sys = CasadiSystem(1)
model = AnalyticalFluidModel(IdealGasState(1.4, 287, 1.8e-5))
# model = ExternalFluidModel(DebugAbstractState('HEOS', 'air'))
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
    'beta': Quantity(45.0, 'deg'),
    'mach': 1.0,
}

# sys.boundary_conditions[1]['kin'] = {'mach': 1.4}
sys.boundary_conditions[1]['oth'] = {'pRatio': 0.3}


sys.build()
VariableBoundsRegistry().from_dict(
    {
        'dev_angle': (0.0, 0.8),
        'V': (0.0, 1000),
        'W': (0.0, 1000),
    }
)

rtfn_ip = sys.make_rootfinder('ipopt', opts={'error_on_fail': True})
x0 = sys.get_scaled_guess()
kn = sys.get_scaled_constraints()
bnd = sys.get_arguments_bounds()
sol = solve_root_problem(rtfn_ip, x0, kn, bnd)


sys.write_solution_to_nodes(sol)
n0 = sys.nodes[0]
n1 = sys.nodes[1]

if not np.isclose(n0.kin.beta, n0.kin.alpha):
    ipdb.set_trace()

print(n0.kin)
print(n1.kin)

fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
n0.kin.plot(n0.geo, 10, ax[0])
n1.kin.plot(n1.geo, 10, ax[1])
plt.show()

pRatio_idx = sys.constraints.index('oth_pRatio1')
angle_idx = sys.constraints.index('kin_beta0')
out_mach_idx = sys.free_args.index('kin_mach1')
RUN_SWEEP = True
N_PTS = 40
ANGLES = [45, 60, 75]

betas = [a * np.pi / 180 for a in ANGLES]
pratios = np.linspace(0.3, 1.0, N_PTS)
deviations = {a: [] for a in betas}
loss_coeffs = {a: [] for a in betas}
mer_machs = {a: [] for a in betas}
out_machs = {a: [] for a in betas}
out_angles = {a: [] for a in betas}

rtfn_kn = sys.make_rootfinder('kinsol', {'error_on_fail': True})
res_func = sys.make_residual_function()

if RUN_SWEEP:
    fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for beta in betas:
        kn[angle_idx] = np.array([beta]) * sys.constraints_scaling[angle_idx]
        # First sweep with ipopt
        sol = solve_root_problem(rtfn_ip, sol, kn)
        for pr in pratios:
            kn[pRatio_idx] = np.array([pr]) * sys.constraints_scaling[pRatio_idx]
            sol = solve_root_problem(rtfn_kn, sol, kn)
            sys.write_solution_to_nodes(sol)
            n0 = sys.nodes[0]
            n1 = sys.nodes[1]

            deviations[beta].append(np.abs(n0.kin.alpha - n1.kin.alpha))
            loss_coeffs[beta].append((n0.tot.p - n1.tot.p) / (n0.tot.p - n0.stc.p))
            mer_machs[beta].append(n1.kin.Wm / n1.stc.speed_sound)
            out_machs[beta].append(n1.kin.W / n1.stc.speed_sound)
            out_angles[beta].append(n1.kin.beta)

        label = f'alpha {beta * 180 / np.pi:.2f}'

        ax[0].set_title('Deviations')
        ax[0].plot(pratios, np.array(deviations[beta]) * 180 / np.pi, label=label)

        ax[1].set_title('Loss Y')
        ax[1].plot(pratios, np.array(loss_coeffs[beta]), label=label)

        ax[2].set_title('Oulet Machs')
        # ax[2].plot(pratios, mer_machs[alpha], label=label)
        ax[2].plot(pratios, out_machs[beta], label=label)

    jax.tree.map(
        lambda a: a.grid(alpha=0.8),
        ax.tolist(),
    )
    jax.tree.map(
        lambda a: a.legend(),
        ax.tolist(),
    )

    ax[0].set_ylim(0, 20)
    ax[1].set_ylim(0, 0.4)
    ax[2].set_ylim(0, 1.9)

    fig.tight_layout()
    fig.show()

globals().update(residual_debugger(Kinematics(), [n0]))
