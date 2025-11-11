import timeit
import casadi as cs

x = cs.MX.sym('x', 1)  # pyright: ignore
a = cs.MX.sym('a', 1)  # pyright: ignore

num_span = 10000

a_values = cs.linspace(0, cs.pi / 2, num_span)


# x**2 - cos(a) = 0
def residual(x, a):
    return x**2 - cs.cos(a)


expression = residual(x, a)

# Using nlpsol directly instead of rootfinder wrapper
# We need to reformulate the rootfinding problem as an NLP
# Rootfinding: find x such that g(x) = 0
# NLP formulation: minimize g(x)^2 (or just set as constraint)
nlp = {
    'x': x,
    'g': expression,  # Constraint: residual = 0
    'p': a,  # Parameter
    'f': 0,  # No objective, we just want feasibility
}

solver = cs.nlpsol(
    'solver',
    'ipopt',
    nlp,
    {
        'ipopt.print_level': 0,
        'ipopt.max_iter': 100,
        'print_time': False,
        # Need the limited-memory, approx (quasi-newton)
        # the eos does not have an hessian
        # 'ipopt.hessian_approximation': 'limited-memory',
        # 'ipopt.jacobian_approximation': 'finite-difference-values',
    },
)

solver_mapped = solver.map(num_span, 'openmp')


start_time = timeit.default_timer()
# nlpsol returns a dict with 'x', 'f', 'g', etc.
# We need to specify lbg=0, ubg=0 to enforce g(x) = 0
sol_dict = solver_mapped(
    x0=num_span * [1.0],
    p=a_values,
    lbg=num_span * [0.0],  # Lower bound: g = 0
    ubg=num_span * [0.0],  # Upper bound: g = 0
)
sol = sol_dict['x']  # Extract just the solution
end_time = timeit.default_timer()
print(f'Solving time was {end_time - start_time}')
print(f'First 5 solutions: {sol[:5]}')
print(f'Last 5 solutions: {sol[-5:]}')
