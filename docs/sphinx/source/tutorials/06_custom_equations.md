# Writing Custom Equations

While ADeT provides many built-in equations for common physics, you often need to implement custom equations for specialized applications. This tutorial covers how to write and integrate custom equations into your systems.

## Anatomy of an Equation

Every equation in ADeT is a subclass of `EquationBase` with a `residual()` method:

```python
from adet.equations.base_equation import EquationBase
from adet.variables import NodeVariables

n0 = NodeVariables(0)

class SimpleEquation(EquationBase):
    """An equation represents a residual r(x) = 0."""

    def residual(self, var1: n0.stc.Pressure.Hint, var2: n0.kin.V_abs.Hint):
        # Compute residual from symbolic variables
        r = var1 * var2 - 1000  # Example: p·V = 1000
        return r
```

The key elements are:

1. **Subclass EquationBase** — Provides infrastructure for symbolic computation
2. **Define residual()** — Compute the residual from variables
3. **Use type hints with `.Hint`** — Let ADeT match parameters to variables
4. **Return residual(s)** — Can be scalar or tuple of residuals

## Parameter Matching via Hints

ADeT matches function parameters to system variables using hints. A hint is created by accessing a variable's `.Hint` attribute:

```python
from adet.variables import NodeVariables, ThermoVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)
thrm = ThermoVariables()

class EnergyBalance(EquationBase):
    def residual(
        self,
        # These hints tell ADeT which variable each parameter represents
        h0: n0.tot.Enthalpy.Hint,      # Total enthalpy at node 0
        V0: n0.kin.V_abs.Hint,         # Velocity at node 0
        h1: n1.tot.Enthalpy.Hint,      # Total enthalpy at node 1
        V1: n1.kin.V_abs.Hint,         # Velocity at node 1
    ):
        # Total enthalpy is conserved (ideal case)
        h0_total = h0 + V0**2 / 2
        h1_total = h1 + V1**2 / 2
        return h0_total - h1_total
```

**Important:** Parameter names don't have to match variable names. The `.Hint` attribute is what matters.

If you forget the hints, ADeT will raise an error explaining which hints are missing.

## Equations with Single and Multiple Residuals

An equation can return:

**Single residual:**

```python
class PressureDrop(EquationBase):
    def residual(self, p_in, p_out, dp_friction):
        r = (p_in - p_out) - dp_friction
        return r  # Single scalar
```

**Multiple residuals (tuple):**

```python
class MomentumBalance(EquationBase):
    def residual(self, rho, A, V, p, F):
        # Mass continuity
        r1 = rho * A * V - 0.1  # 0.1 kg/s

        # Momentum balance: ∑F = d(ṁV)/dt
        r2 = p * A - F - rho * A * V**2

        return r1, r2  # Tuple of residuals
```

Returning a tuple creates **two equations** in the system. This is useful for systems where one variable association naturally produces multiple constraints.

## Accessing the Equation of State (EOS)

Equations can use the system's fluid model via `self.eos()`:

```python
from adet.equations.base_equation import EquationBase, EquationConfig
import CoolProp as cp

class ThermoRelation(EquationBase):
    # Specify which inputs the EOS will use and what it outputs
    config = EquationConfig(
        input_pair=cp.HmassSmass_INPUTS,              # Inputs: h, s
        out_properties=(thrm.Pressure, thrm.Density), # Outputs: p, ρ
    )

    def residual(self, h, s, p_actual, rho_actual):
        # Compute properties from enthalpy and entropy
        p_eos, rho_eos = self.eos(h, s)

        # Residuals: actual values should match EOS values
        r1 = p_actual - p_eos
        r2 = rho_actual - rho_eos
        return r1, r2
```

The `EquationConfig` tells ADeT:

- **input_pair** — Which CoolProp input pair to use (e.g., `HmassSmass_INPUTS` means enthalpy + entropy)
- **out_properties** — Which properties to compute (as `VarSpec` objects)

**Common input pairs:**

- `cp.PT_INPUTS` — Pressure, temperature
- `cp.HmassSmass_INPUTS` — Enthalpy, entropy
- `cp.DmassT_INPUTS` — Density, temperature
- `cp.PSmass_INPUTS` — Pressure, entropy

Once configured, call `self.eos(input1, input2)` and it returns a tuple of the `out_properties` in order.

## NumPy Operations in Equations

You can use NumPy functions in equations. They work with symbolic variables just as they do with floats:

```python
import numpy as np
from adet.equations.base_equation import EquationBase

class MachNumber(EquationBase):
    def residual(self, V, a, M_actual):
        # M = V / a
        M_computed = V / a

        return M_actual - M_computed

class TrigonometricRelation(EquationBase):
    def residual(self, theta_rad, theta_deg):
        # Convert radians to degrees
        theta_computed = np.degrees(theta_rad)

        return theta_deg - theta_computed

class RootFinding(EquationBase):
    def residual(self, x, root):
        # Symbolic sqrt works fine
        r = np.sqrt(x) - root
        return r
```

NumPy's mathematical functions (`sin`, `cos`, `sqrt`, `exp`, `log`, etc.) all work with CasADi symbolic variables.

**Avoid:**
- `if` statements (use `cs.if_else()` for branching)
- `for` loops (use NumPy vectorized operations)
- Non-mathematical operations (string manipulation, etc.)

## Example: Shock Equations

Let's implement a simple oblique shock equation:

```python
import numpy as np
from adet.equations.base_equation import EquationBase
from adet.variables import NodeVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)

class ObliqueShock(EquationBase):
    """Oblique shock normal momentum and continuity."""

    def residual(
        self,
        # Upstream
        rho0: n0.stc.Density.Hint,
        p0: n0.stc.Pressure.Hint,
        V0: n0.kin.V_abs.Hint,
        beta0: n0.kin.FlowAngleRel.Hint,  # Shock angle relative to flow
        # Downstream
        rho1: n1.stc.Density.Hint,
        p1: n1.stc.Pressure.Hint,
        V1: n1.kin.V_abs.Hint,
        beta1: n1.kin.FlowAngleRel.Hint,
        # Deflection
        delta: n0.oth.FlowDeflection.Hint,  # Shock deflection angle
    ):
        # Decompose velocity into normal and tangential components
        u0 = V0 * np.sin(beta0)  # Normal component
        w0 = V0 * np.cos(beta0)  # Tangential component

        u1 = V1 * np.sin(beta1 - delta)
        w1 = V1 * np.cos(beta1 - delta)

        # Normal momentum conservation: $p_0 + \rho_0 u_0^2 = p_1 + \rho_1 u_1^2$
        r1 = (p0 + rho0 * u0**2) - (p1 + rho1 * u1**2)

        # Mass continuity (normal): $\rho_0 u_0 = \rho_1 u_1$
        r2 = (rho0 * u0) - (rho1 * u1)

        # Tangential momentum conservation: $\rho_0 u_0 w_0 = \rho_1 u_1 w_1$
        r3 = (rho0 * u0 * w0) - (rho1 * u1 * w1)

        return r1, r2, r3
```

## Using Custom Equations

Once you've written an equation, use it like any other:

```python
from adet.assembly import CasadiSystem

system = CasadiSystem()

# Configure fluid model, add boundary conditions, etc.
# ...

# Add your custom equation
system.add_equation(ObliqueShock(), 0)

system.build()
rtfn = system.make_rootfinder('kinsol')
x0 = system.get_scaled_guess()
knowns = system.get_scaled_constraints()

sol = solve_root_problem(rtfn, x0, knowns)
```

## Debugging Custom Equations

If your equation isn't being recognized or produces unexpected results:

**Missing hint error:**

If you get an error like "Cannot find hint for parameter `x`", you forgot a `.Hint` on a parameter:

```python
# Wrong:
def residual(self, p, T):  # No hints!
    return p / (R * T) - rho

# Right:
def residual(self, p: n0.stc.Pressure.Hint, T: n0.stc.Temperature.Hint):
    return p / (R * T) - rho
```

**Variable not found:**

If you reference a variable that doesn't exist in the system, ADeT will error during `build()`. Check that all variables in your hints are actually in a `NodeVariables` instance:

```python
# Check: does this variable exist?
n = NodeVariables(0)
print(n.stc.MyVar)  # Error if MyVar doesn't exist in stc
```

**Residual has wrong shape:**

If your residual produces the wrong shape (e.g., a matrix instead of a scalar), the solver will fail:

```python
# Wrong: returns array
def residual(self, x):
    return np.array([x - 1, x + 1])  # Error!

# Right: returns tuple
def residual(self, x):
    return (x - 1, x + 1)  # Correct
```

## Advanced: Conditional Logic

For equations with branching logic (e.g., "if supersonic then use shock relations, else use isentropic"), use `casadi.if_else()`:

```python
import casadi as cs

class ConditionalEquation(EquationBase):
    def residual(self, M, p_shock, p_isentropic, p_actual):
        # Use shock relations if M > 1, isentropic if M < 1
        p_computed = cs.if_else(
            M > 1.0,
            p_shock,   # If M > 1
            p_isentropic,  # Else
        )

        return p_actual - p_computed
```

**Note:** `if_else` returns a single symbolic value suitable for residuals. Never use Python's `if` statement in residuals, as the branching wouldn't be symbolic.

## Understanding Your Equation in the System

When your equation is added to the system, it becomes **part of the big CasADi residual function**. This is important to understand how your equation interacts with the rest of the system.

**Single Residual:**

```python
class SimpleEquation(EquationBase):
    def residual(self, x, y):
        return x + y - 1.0  # One residual
```

This contributes **1 equation** to the system. When the system builds, this becomes one entry in the big residual vector.

**Multiple Residuals:**

```python
class DoubleEquation(EquationBase):
    def residual(self, x, y, z):
        r1 = x + y - 1.0
        r2 = x * y - z
        return r1, r2  # Two residuals
```

This contributes **2 equations** to the system. Both residuals go into the big vector.

The solver then finds **x** such that the entire big residual vector equals zero.

**Important:** The number of residuals you return must be chosen so that the total number of residuals equals the number of unknowns. This is why the system consistency check is critical.

## Debugging Your Equation in the Solution

After solving, inspect how your equation behaved:

```python
from adet.equations.utils import residual_debugger
from adet.variables import NodeVariables

sol = solve_root_problem(rtfn, x0, knowns)
sol_dict = system.sol_to_dict(sol)

# Debug your custom equation
my_eq = MyCustomEquation()
n0 = NodeVariables(0)

debug_vars = residual_debugger(my_eq, [0], sol_dict)
globals().update(debug_vars)

# Now inspect:
print(f"Variable X: {X}")
print(f"Variable Y: {Y}")

# Evaluate your residual with solution values
residual_value = self.residual(X, Y)
print(f"Residual at solution: {residual_value}")  # Should be ~0
```

If the residual at the solution isn't close to zero, something is wrong with your equation definition.

## Complete Example: Custom Turbomachinery Loss

Here's a complete example implementing a custom loss model:

```python
import numpy as np
from adet.equations.base_equation import EquationBase
from adet.variables import NodeVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)

class CustomLossModel(EquationBase):
    """Custom loss model: $\Delta p = \zeta \cdot \frac{\rho V^2}{2}$"""

    def __init__(self, loss_coefficient=0.05):
        self.zeta = loss_coefficient

    def residual(
        self,
        rho0: n0.stc.Density.Hint,
        V0: n0.kin.V_abs.Hint,
        p0: n0.stc.Pressure.Hint,
        p1: n1.stc.Pressure.Hint,
    ):
        # Dynamic pressure
        q = 0.5 * rho0 * V0**2

        # Loss is proportional to dynamic pressure
        dp_loss = self.zeta * q

        # Pressure drop due to loss
        r = (p0 - p1) - dp_loss

        return r

# Use it:
loss_eq = CustomLossModel(loss_coefficient=0.08)
system.add_equation(loss_eq, 0)
```

## Key Takeaways

- **Write equations as residuals**: $r(x) = 0$
- **Use `.Hint` attributes**: So ADeT can match parameters to variables
- **Leverage `self.eos()`**: For thermodynamic properties
- **Use NumPy**: All standard math functions work symbolically
- **Test incrementally**: Add one equation at a time, verify it solves
- **Check dimensions**: Residuals must be scalars or tuples of scalars

## Next Steps

- Back to [Quickstart](01_quickstart.md) — Apply custom equations to a complete system
- [Solving Systems](03_solving_systems.md) — Debugging strategies for complex systems
- Check `src/adet/equations/` in the repository for more examples
