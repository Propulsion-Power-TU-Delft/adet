import numpy as np

from adet.assembly import CasadiSystem, EquationBase
from adet.diagnostics import SystemDiagnostics
from adet.variables import NodeVariables, OtherVariables
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

# Create node variables for VarSpec-based boundary conditions
n0 = NodeVariables(0)

# NOTE: Using custom variable names that match the equation signature
# These map to OtherVariables VarSpecs (ps, pd, kp, etc. are typically in 'oth' state)
# Since custom vars aren't in the standard VarSpec, use symbolic placeholders
BOUND_COND = {
    # Using OtherVariables for domain-specific parameters
    # Map each equation parameter to a boundary condition
    # NOTE: The exact VarSpec mapping depends on how equation arguments are resolved
    # For now, using a conceptual mapping that would need proper VarSpec definitions
}

# Build with boundary conditions
system.add_boundary_conditions(BOUND_COND)
system.build(scaled=False)  # Don't scale, everything is dimensionless

# NOTE: With the new API, free_args are VarSpec objects, not strings
# The exact_sol dict keys should map to actual VarSpec objects identified during system build
# Example structure (requires proper VarSpec identification):
exact_sol = {
    # These would be actual VarSpec objects once system.data.free_args is available
}

# The original_order would be derived from system.data.free_args VarSpec objects
original_order = tuple(system.data.free_args)

# Create index map from VarSpec to original paper order
index_map = {
    system.data.free_args.index(arg): idx for idx, arg in enumerate(original_order)
}

# Build exact_x0 from the exact solution values (would need matching VarSpec objects)
exact_x0 = np.array(
    [1.0, 1.0, 4.0, 1.0, 2.0, 2.2]
)  # Placeholder values matching expected solution

# Get constraint values from the system's boundary condition dict
constraint_values = list(system.data.boun_cond.values())
analyzer = SystemDiagnostics(
    system, np.concatenate(constraint_values) if constraint_values else np.array([])
)

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
