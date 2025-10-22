import casadi as cs
from CoolProp import AbstractState
from adet.assembly import CasadiSystem

from adet.equations.base_equation import EquationBase
from adet.equations.fundamental import Kinematics, TotalStaticMatching

from adet.equations.ideal_gas import IdealRltEos, IdealStcEos, IdealTotEos
from adet.equations.linkers import SpeedLinker
from adet.fluid.settings import AbstractStateModel, FluidSettings, IdealGasModel
from adet.registries import DefaultUnitsRegistry, GuessRegistry


# Example problem:
# Given the total state (pt,Tt) and Mach number
# get the static state and velocity magnitude
class MachProblem(EquationBase):
    def residual(self, oth_Ma, kin_V, stc_speed_sound):
        return stc_speed_sound * oth_Ma - kin_V


ss = CasadiSystem(3, scale_suffix='<|')

# Set up the fluid model
fluid_model = AbstractStateModel(
    AbstractState('HEOS', 'Air'),
)

fluid_model = IdealGasModel(287.0, 1.4)

ss.settings = FluidSettings(
    fluid_model,
    ('p', 'T', 'hmass', 'smass'),
    2,
)

DefaultUnitsRegistry()['Ma'] = 'dimensionless'

# Main problem
ss.add_equation(MachProblem(), 0)

# Match between static, total and relative total conditions
ss.add_equation(TotalStaticMatching(), 0)
# Kinematics
ss.add_equation(Kinematics(), 0)
ss.add_equation(SpeedLinker(), (0, 0))

if isinstance(fluid_model, IdealGasModel):
    ss.add_equation(IdealStcEos(), 0)
    ss.add_equation(IdealTotEos(), 0)
    ss.add_equation(IdealRltEos(), 0)


ss.add_boundary_conditions(
    {
        'tot': {'p': 5e5, 'T': 1300},
        'oth': {'Ma': 2.2},
        'kin': {'U': 0, 'rr': 100.0, 'alpha': 0},
    },
    0,
)

ss.build(scaled=False)

args = cs.vertcat(*ss.free_args_sym)
cons = ss.constraints_values

expression = ss.make_residual_function()(args, cons.flatten())

root_problem = {'x': args, 'g': expression}


inlet_conditions = {}

# Tweak initial guess
for vals in ss.boundary_conditions[0].values():
    _g_reg = GuessRegistry()
    _g_reg.from_dict(vals)  # pyright:ignore

    p_guess = _g_reg.get('p')
    T_guess = _g_reg.get('T')


x0 = ss.get_initial_guess().flatten()

G_nlp = cs.rootfinder(
    'nlpsol_roots',
    'nlpsol',
    root_problem,
    {
        'nlpsol': 'ipopt',
        'nlpsol_options': {
            'ipopt.hessian_approximation': 'limited-memory',  # => Quasi-newton
            # 'ipopt.jacobian_approximation': 'finite-difference-values',
        },
    },
)

sol = G_nlp(x0, 0.0)
