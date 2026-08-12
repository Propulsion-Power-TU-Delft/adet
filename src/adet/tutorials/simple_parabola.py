from pprint import pprint
from adet.assemblers import CasadiSystem
from adet.equations.base_equation import EquationBase
from adet.solution import solve_root_problem
from adet.varspec import VarSpec
from numpy import sin

# Define variables
x_spec = VarSpec('x', 'dimensionless', guess=1.0, bounds=(-1000, 1000))
p_spec = VarSpec('p', 'dimensionless', node=0)

my_x0 = x_spec.at_node(0)
my_x1 = x_spec.at_node(1)
my_p = p_spec


# Define an equation
class MyParabola(EquationBase):
    def residual(
        self,
        x0: my_x0.Hint,
        x1: my_x1.Hint,
        p: my_p.Hint,
    ):
        residual_1 = 3 * x0**2 - 2 * x0 - p
        residual_2 = sin(x0) + x1
        return residual_1, residual_2


# Set up and solve
system = CasadiSystem(num_span=1)
system.add_equation(MyParabola(), (0, 1))
system.add_boundary_conditions({my_p: 5})
system.build()

rootfinder = system.make_rootfinder('kinsol')
guess = system.get_guess()
knowns = system.get_boundary_conds()
solution = solve_root_problem(rootfinder, guess, knowns)
sol_dict = system.sol_to_dict(solution)

print('The solution is:\n')
pprint(sol_dict)
