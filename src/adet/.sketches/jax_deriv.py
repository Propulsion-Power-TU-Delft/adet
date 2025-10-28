import jax
import jax.numpy as jnp
import CoolProp.CoolProp as cp

# jax.config.update('jax_enable_x64', True)

AS = cp.AbstractState('HEOS', 'Air')

p_val = 1e5
T_val = 300.0


AS.update(cp.PT_INPUTS, p_val, T_val)
h0 = AS.hmass()


dhdp_exact = AS.first_partial_deriv(cp.iHmass, cp.iP, cp.iT)
dhdT_exact = AS.first_partial_deriv(cp.iHmass, cp.iT, cp.iP)

eps = 1e-5

deltaP = p_val * eps
deltaT = T_val * eps


AS.update(cp.PT_INPUTS, p_val + deltaP, T_val)
dhdp_FD = (AS.hmass() - h0) / deltaP

AS.update(cp.PT_INPUTS, p_val, T_val + deltaT)
dhdT_FD = (AS.hmass() - h0) / deltaT


# Create a function h = h(p, T)
@jax.custom_jvp
def hmass(p, T):
    AS.update(cp.PT_INPUTS, p, T)
    return AS.hmass()


@hmass.defjvp
def eos_jvp(primals, tangents):
    p, T = primals
    p_dot, T_dot = tangents

    # Call CoolProp on real numbers
    val = hmass.__wrapped__(p.astype(float), T.astype(float))

    dhdp = AS.first_partial_deriv(cp.iHmass, cp.iP, cp.iT)
    dhdT = AS.first_partial_deriv(cp.iHmass, cp.iT, cp.iP)

    # Replace float0 tangents
    if p_dot.dtype is jax.float0:
        p_dot = jnp.zeros_like(p)
    if T_dot.dtype is jax.float0:
        T_dot = jnp.zeros_like(T)

    tangent_out = dhdp * p_dot + dhdT * T_dot
    return val, tangent_out


# Merge the arguments for a clean jacobian
def wrapped_func(X):
    p, T = X
    return hmass(p, T)


X0 = jnp.array([p_val, T_val])
jac = jax.jacrev(wrapped_func)

print(f'Hmass eos call: h = h(p={p_val},T={T_val}) = {hmass(p_val, T_val)}')
print(f'Jacobian eos call:\n[[∂h/∂p, ∂h/∂T]]\n{jac(X0)}\n\n')


# === Let's try to use these in the total to static definintion
def total_to_static(args):
    # Only the true arguments will stay here
    p, T, pt, Tt, V = args

    # Apply equations of state
    h = hmass(p, T)
    ht = hmass(pt, Tt)

    return ht - h - V**2 / 2


jac_tts = jax.jacrev(total_to_static)

print(
    'Residual function:\n  f(p, T, pt, Tt, V) = ht - h - V**2 / 2 = 0\n '
    ' where h = h(p,T), ht = ht(p,T)\n'
)

args = jnp.array([p_val, T_val, p_val, T_val + 4.98, 100.0])
print(f'Residual function {total_to_static(args)}')

J_val = jac_tts(args)


jac_string = 'J = [[∂f/∂p, ∂f/∂T, ∂f/∂pt, ∂f/∂Tt, ∂f/∂V]] =\n'
jac_string += '  = [[-∂h/∂p, -∂h/∂T, ∂h/∂pt, ∂h/∂Tt, - V]] ='

print(f'Composed jacobian:\n{jac_string}\n  = {J_val}')
