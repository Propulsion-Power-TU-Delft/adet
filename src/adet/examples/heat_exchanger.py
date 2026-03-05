import numpy as np

from adet.assembly import CasadiSystem, EquationBase
from adet.diagnostics import SystemDiagnostics
from adet.registries import DefaultUnitsRegistry


class HeatExchangerProblem(EquationBase):
    def residual(
        self, f, Q, kp, ps, pi, po, kh, kv, pd, c, To, Ts, gamma, A, Ta, gammaz, fz, nu
    ):
        r0 = f - kp * (ps - pi) ** 0.5
        r1 = pi - po - kh * f**2
        r2 = f - kv * (po - pd) ** 0.5
        r3 = Q - f * c * (To - Ts)
        r4 = Q - gamma * A * (Ta - (Ts + To) / 2)
        r5 = gamma - gammaz * (f / fz) ** nu

        return r0, r1, r2, r3, r4, r5


system = CasadiSystem(num_span=1)
system.add_equation(HeatExchangerProblem(), 0)

# Set everything to dimensionless
DefaultUnitsRegistry().set_forced_value('dimensionless')

BOUND_COND = {
    'oth': {
        'ps': 2.201,
        'pd': 1.0,
        'kp': 1000**0.5,
        'kh': 0.2,
        'c': 1.0,
        'fz': 1.0,
        'gammaz': 1.0,
        'nu': 0.8,
        'Ts': 0.0,
        'Ta': 6.0,
        'Q': 4.0,
        'A': 1.0,
    }
}

system.add_boundary_conditions(BOUND_COND, 0)
system.build(scaled=False)  # Don't scale, everything is dimensionless

exact_sol = {
    'oth_To0': 4.0,
    'oth_f0': 1.0,
    'oth_gamma0': 1.0,
    'oth_kv0': 1.0,
    'oth_pi0': 2.2,
    'oth_po0': 2.0,
}

# That is found in the paper
original_order = (
    'oth_f0',
    'oth_kv0',
    'oth_To0',
    'oth_gamma0',
    'oth_po0',
    'oth_pi0',
)

index_map = {system.free_args.index(arg): idx for idx, arg in enumerate(original_order)}

exact_x0 = np.array([exact_sol[arg] for arg in system.free_args])

analyzer = SystemDiagnostics(system, np.concatenate(system.constraints_values))

# === The first guesses are rounded in the paper
x0_case1 = np.round(exact_x0 - 1e-5 * exact_x0, 5)
x0_case2 = np.round(exact_x0 - 1e-3 * exact_x0, 3)
x0_case3 = np.round(exact_x0 - 1e-2 * exact_x0, 3)
x0_case4 = np.round(exact_x0 - 1e-1 * exact_x0, 3)
x0_case5 = np.array([3.6, 0.9, 0.9, 0.9, 2.151, 1.8])

analyzer.compute_bounding_coeffs(x0_case3)
analyzer.compute_curvatures(x0_case3)
analyzer.compute_sensitivities(x0_case3)


def sensitivities_printer(sigma):
    for row in sigma:
        print_str = ''
        for entry in row:
            print_str += f'{entry:.3f}  '
        print(print_str)
