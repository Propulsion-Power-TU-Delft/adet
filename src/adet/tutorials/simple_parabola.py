from numpy import sin

from adet.assemblers import CasadiSystem
from adet.equations.base_equation import EquationBase
from adet.solution import solve_root_problem
from adet.varspec import VarSpec

# Define custom variables
my_x = VarSpec('x', 'dimensionless', guess=1.0, bounds=(-1000, 1000), node=0)
my_y = VarSpec('y', 'dimensionless', guess=1.0, bounds=(-1000, 1000), node=0)
my_p = VarSpec('p', 'dimensionless', node=0)


# Define custom equation
class MyParabola(EquationBase):
    def residual(
        self,
        x: my_x.Hint,
        y: my_y.Hint,
        p: my_p.Hint,
    ):
        residual_1 = 3 * x**2 - 2 * x - p
        residual_2 = sin(x) + y

        return residual_1, residual_2


system = CasadiSystem(num_span=1)

system.add_equation(MyParabola(), 0)
system.add_boundary_conditions({my_p: 5})

system.build()

rootfinder = system.make_rootfinder('kinsol')
guess = system.get_guess()
knowns = system.get_boundary_conds()

solution = solve_root_problem(rootfinder, guess, knowns)
