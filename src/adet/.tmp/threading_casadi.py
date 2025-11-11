import timeit
import casadi as cs

x = cs.MX.sym('x', 1)
a = cs.MX.sym('a', 1)

num_span = 5000  # Smaller for comparison

a_values = cs.linspace(0, cs.pi / 2, num_span)

# Problem: x**2 - cos(a) = 0
expression = x**2 - cs.cos(a)

print('=' * 70)
print('COMPARISON: rootfinder vs nlpsol')
print('=' * 70)

# ============================================================================
# APPROACH 1: Using rootfinder (the standard way)
# ============================================================================
print('\n1. ROOTFINDER approach:')
print('-' * 70)

rootfind_problem = {'x': x, 'g': expression, 'p': a}

rootfinder = cs.rootfinder(
    'rootfinder',
    'nlpsol',
    rootfind_problem,
    {
        'error_on_fail': False,
        'nlpsol': 'ipopt',
        'nlpsol_options': {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 100,
            'print_time': False,
        },
    },
)

# Serial only (OpenMP crashes with rootfinder)
rootfinder_mapped = rootfinder.map(num_span, 'serial', 4)

start_time = timeit.default_timer()
sol_rootfinder = rootfinder_mapped(num_span * [1.0], a_values)
rootfinder_time = timeit.default_timer() - start_time

print(f'Time: {rootfinder_time:.4f}s')
print(f'First 5 solutions: {sol_rootfinder[:5]}')

# ============================================================================
# APPROACH 2: Using nlpsol directly
# ============================================================================
print('\n2. NLPSOL approach:')
print('-' * 70)

# Reformulate as NLP with equality constraint
nlp = {
    'x': x,
    'f': 0,  # No objective (or could use g^2 as objective)
    'g': expression,  # Constraint: g(x) = 0
    'p': a,
}

solver = cs.nlpsol(
    'solver',
    'ipopt',
    nlp,
    {
        'ipopt.print_level': 0,
        'ipopt.max_iter': 100,
        'print_time': False,
    },
)

# Test both serial and OpenMP
solver_serial = solver.map(num_span, 'serial', 4)
start_time = timeit.default_timer()
sol_dict_serial = solver_serial(
    x0=num_span * [1.0],
    p=a_values,
    lbg=num_span * [0.0],
    ubg=num_span * [0.0],
)
nlpsol_serial_time = timeit.default_timer() - start_time

solver_openmp = solver.map(num_span, 'openmp', 4)
start_time = timeit.default_timer()
sol_dict_openmp = solver_openmp(
    x0=num_span * [1.0],
    p=a_values,
    lbg=num_span * [0.0],
    ubg=num_span * [0.0],
)
nlpsol_openmp_time = timeit.default_timer() - start_time

print(f'Serial time: {nlpsol_serial_time:.4f}s')
print(f'OpenMP time: {nlpsol_openmp_time:.4f}s')
print(f'Speedup: {nlpsol_serial_time / nlpsol_openmp_time:.2f}x')
print(f'First 5 solutions: {sol_dict_openmp["x"][:5]}')
print(f"Returns: Dict with 'x', 'f', 'g', 'lam_x', 'lam_g', etc.")
print(f'OpenMP support: ✓ WORKS (no crash)')
