"""
Sketch file on how to overload derivatives calulations within
functions using CoolProp and CasADi
"""

import casadi as cs
import CoolProp as cp


# These are to avoid annoying typing warnings
# the subs are wrong in casadi python API
class MX(cs.MX):
    @staticmethod
    def sym(*args) -> cs.MX:
        return cs.MX.sym(*args)


class Sparsity(cs.Sparsity):
    @staticmethod
    def dense(*args):
        return cs.Sparsity.dense(*args)


def pt_abstate_update(eos: cp.AbstractState, var0, var1):
    eos.update(cp.PT_INPUTS, var0, var1)
    return eos.hmass()


class CoolPropEoS(cs.Callback):
    def __init__(self, name, backend, fluid, opts={}):
        cs.Callback.__init__(self)
        self._eos = cp.AbstractState(backend, fluid)
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        return Sparsity.dense(2, 1)

    def get_sparsity_out(self, arg):
        return Sparsity.dense(1, 1)

    # Evaluate numerically
    def eval(self, arg):
        p, T = cs.vertsplit(arg[0])
        ret = cs.DM(1, 1)
        ret[0, 0] = pt_abstate_update(self._eos, p, T)

        return [ret]

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        class JacFun(cs.Callback):
            def __init__(self, eos: cp.AbstractState, opts={}):
                cs.Callback.__init__(self)
                self.construct(name, opts)
                self._eos = eos

            def get_n_in(self):
                return 2

            def get_n_out(self):
                return 1

            def get_sparsity_in(self, i):
                if i == 0:
                    return Sparsity.dense(2, 1)
                elif i == 1:
                    return Sparsity(1, 1)

            def get_sparsity_out(self, arg):
                return Sparsity.dense(1, 2)

            def eval(self, arg):
                p, T = cs.vertsplit(arg[0])
                pt_abstate_update(self._eos, p, T)
                ret = cs.DM(1, 2)

                ret[0, 0] = self._eos.first_partial_deriv(cp.iHmass, cp.iP, cp.iT)
                ret[0, 1] = self._eos.first_partial_deriv(cp.iHmass, cp.iT, cp.iP)

                return [ret]

        self.jac_callback = JacFun(self._eos)
        return self.jac_callback


def dummy_eos(X):
    p, T = cs.vertsplit(X)
    R = 8.314
    return R * T / p


x = MX.sym('x', 2)
eos_manual = cs.Function('test', [x], [dummy_eos(x)])

print(f'Original func struct:\n {eos_manual}\n')
# Look at the shape and implement the same in the callback!
print(f'Required Jacobian struct:\n {eos_manual.jacobian()}\n')

eos_func = CoolPropEoS('eos_func', 'HEOS', 'Air')
jac_func = cs.Function(
    'jac_func',
    [x],
    [cs.jacobian(eos_func(x), x)],
)

# p and T numerical values, list for convenience
p_val = 1e5
T_val = 300

f_val = eos_func([p_val, T_val])
j_val = jac_func([p_val, T_val])

print(f'Hmass eos call: h = h(p={p_val},T={T_val}) = {f_val}')
print(f'Jacobian eos call:\n[[∂h/∂p, ∂h/∂T]]\n{j_val}\n\n')

# Create symbolic variables
p = MX.sym('p', 1)
T = MX.sym('T', 1)
pt = MX.sym('pt')
Tt = MX.sym('Tt')
V = MX.sym('V')


# List for convenience
all_arguments = [p, T, pt, Tt, V]


# Define a composite function
# Custom callback + symbolic expressionsj
def enthalpy_conversion(p, T, pt, Tt, V):
    h = eos_func(cs.vertcat(p, T))
    ht = eos_func(cs.vertcat(pt, Tt))
    return ht - h - V**2 / 2  # pyright:ignore


# Make residual close to 0
# => Tt - T = V**2 / 2 / cpmass = 100**2 / 2 / 1004 = 4.98
arg_values = [
    p_val,  # Static pressure
    T_val,  # Static temperature
    p_val,  # Total pressure
    T_val + 4.98,  # Total temperature
    100.0,
]

# Convert python function to MX symbolic
# by giving it MX.sym inputs
composed_expr = enthalpy_conversion(p, T, pt, Tt, V)

# Create the function associated with the expr
comp_func = cs.Function('comp', all_arguments, [composed_expr])


# Evaluate the function
res = comp_func(*arg_values)
print(
    'Residual function:\n  f(p, T, pt, Tt, V) = ht - h - V**2 / 2 = 0\n '
    ' where h = h(p,T), ht = ht(p,T)\n'
)

print(f'Residual function {res}')

# Compute the jacobian expression w.r.t. the arguments
jac_expr = cs.jacobian(composed_expr, cs.vertcat(*all_arguments))

# Create the associated function
Jf = cs.Function('jac_comp', all_arguments, [jac_expr])

# Evaluate and print
J_val = Jf(*arg_values)
jac_string = 'J = [[∂f/∂p, ∂f/∂T, ∂f/∂pt, ∂f/∂Tt, ∂f/∂V]] =\n'
jac_string += '  = [[-∂h/∂p, -∂h/∂T, ∂h/∂pt, ∂h/∂Tt, - V]] ='

print(f'Composed jacobian:\n{jac_string}\n  = {J_val}')
