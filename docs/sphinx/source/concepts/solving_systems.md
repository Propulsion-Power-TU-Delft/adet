# Solving Systems

Once you've built a system of equations, the next step is to solve it. This tutorial covers solving strategies, debugging, and how to work with solutions.

## From Equations to Solver

The process of converting your equation definitions to a solvable form involves several steps.

### Building the System

After adding all equations and boundary conditions, you must call `build()`:

```python
system.add_equation(Eq1(), 0)
system.add_equation(Eq2(), 0)
system.add_boundary_conditions(bc_dict)

system.build()  # Compiles equations into residual vector
```

The `build()` call:

1. Collects all equations and variables
2. Counts equations and unknowns (variables not in boundary conditions)
3. **Verifies consistency**: number of equations must equal number of unknowns
4. Creates a symbolic residual vector
5. Sets up scaling for numerical stability
6. Initializes the EOS interface for the configured fluid model

**Critical: System Consistency**

For a well-posed problem, you need exactly as many equations as unknowns:

```python
# Example counts:
system.build()
# Output: "System info: 8 total equations, 8 total variables"  ✓ OK
```

If counts don't match, `build()` will ask for confirmation before continuing:

```python
system.build()
# Output: "*** WARNING: Mismatch in number of equations 7 and variables 8"
# Prompted: "continue anyway? [y/n]"
```

**You are responsible for ensuring the system is well-posed.** A mismatch means:

- **More equations than unknowns**: Over-constrained (typically impossible to solve)
- **More unknowns than equations**: Under-determined (infinite solutions)
- **Equal counts**: Well-posed (typically has a unique or discrete set of solutions)

To diagnose mismatches:

1. Count your equations (how many residuals does each equation produce?)
2. Count your unknowns (total variables minus boundary conditions)
3. Add equations or boundary conditions until they balance

### What Gets Compiled: The Big Residual Function

Behind the scenes, ADeT performs a critical transformation:

1. **Collects variables** from all equations and boundary conditions
2. **Organizes unknowns** (variables not in boundary conditions) into a vector **x**
3. **Organizes constraints** (boundary condition values) into a vector **p**
4. **Builds symbolic residual expressions** from each equation
5. **Concatenates all residuals** into a single big residual vector **r(x, p)**
6. **Computes the Jacobian** `∂r/∂x` automatically using CasADi's symbolic differentiation

The result is a **single large CasADi function**:

```
r(x, p) = [r_eq1(x, p), r_eq2(x, p), r_eq3(x, p), ...]ᵀ
```

This is a **nonlinear root-finding problem**: find **x** such that `r(x, p) = 0`.

**Why this matters:** Because the system is one big symbolic function, you have powerful capabilities:

- **Automatic differentiation** — CasADi computes Jacobians exactly (not numerically)
- **Symbolic manipulation** — Can inspect, optimize, or transform the entire system
- **Efficient solving** — Solvers use the analytical Jacobian for Newton-Raphson iterations
- **But:** You must access residuals through the function, not individually

### Accessing the Residual Function

To evaluate residuals at a specific point, use `make_residual_function()`:

```python
system.build()

# Get the compiled residual function
res_func = system.make_residual_function()

# Evaluate at guess point
x0 = system.get_scaled_guess()
knowns = system.get_scaled_constraints()

residuals = res_func(x0, knowns)

print(f"Residual norm: {np.linalg.norm(residuals):.3e}")
print(f"Max residual: {np.max(np.abs(residuals)):.3e}")
```

The residual function takes two arguments:

- **x** (or x0) — The unknown variables (scaled)
- **p** (or knowns) — The constraint values (scaled)

And returns the residual vector evaluated at those points.

**This is why boundary conditions are "constraints":** They're not additional equations; they're the **p** vector passed to the residual function.

### Creating a Root Finder

ADeT supports multiple solver backends via CasADi:

```python
# Newton-Krylov solver (KINSOL from SUNDIALS)
rootfinder = system.make_rootfinder('kinsol')

# Interior point optimizer (IPOPT)
rootfinder = system.make_rootfinder('ipopt')
```

The choice depends on your problem:

- **KINSOL** is faster for well-scaled problems with good initial guesses
- **IPOPT** is more robust but slower; useful when initial guesses are poor

### Setting Up the Solve

Before solving, you need to prepare the guess and constraints:

```python
# Get scaled initial guess for unknowns
x0 = system.get_scaled_guess()

# Get scaled constraint values (boundary conditions)
knowns = system.get_scaled_constraints()
```

Scaling is critical for numerical stability. ADeT scales variables by their typical magnitudes (e.g., pressure is scaled by ~1e5 Pa).

You can also specify bounds on variables:

```python
arg_bounds = system.get_bounds()  # (lower_bounds, upper_bounds)
```

### Solving with `solve_root_problem()`

The `solve_root_problem()` utility handles the actual solve:

```python
from adet.solution import solve_root_problem
import numpy as np

sol = solve_root_problem(
    rootfinder,
    guess=x0,
    knowns=knowns,
    arg_bounds=arg_bounds,
)
```

The solver iterates until:

- **Convergence**: Residuals are below tolerance (~1e-6)
- **Max iterations**: Default limit reached
- **Failure**: Numerical issues (NaN, Inf) encountered

### Accessing Solutions

The returned solution is a scaled vector. Convert it back to a dictionary of variables:

```python
sol_dict = system.sol_to_dict(sol)

# Iterate over all variables
for var_spec, value in sol_dict.items():
    print(f"{var_spec}: {value:.6e}")
```

The `sol_dict` keys are `VarSpec` objects, which contain unit information:

```python
# Access specific values
pressure = sol_dict[n0.stc.Pressure]
temperature = sol_dict[n0.stc.Temperature]
mach = sol_dict[n0.oth.Mach]
```

## Debugging Failed Solves

When a solve fails, the first step is to understand why. ADeT provides tools to diagnose issues.

### Inspecting and Debugging Residuals

**Step 1: Evaluate the Big Residual Function**

Before solving, evaluate residuals at the initial guess:

```python
system.build()

x0 = system.get_scaled_guess()
knowns = system.get_scaled_constraints()

res_func = system.make_residual_function()
residuals = res_func(x0, knowns)

print("Initial residual norm:", np.linalg.norm(residuals))
print("Max residual:", np.max(np.abs(residuals)))
```

If the initial residual is very large (> 1.0 in scaled units), your boundary conditions or initial guess is far from a solution.

**Step 2: Debug Individual Equations with the Residual Debugger**

When a specific equation is suspect, use the `residual_debugger` to inspect its behavior:

```python
from adet.equations.utils import residual_debugger
from adet.equations.fundamental import Kinematics
from adet.variables import NodeVariables
import numpy as np

# After solving, use the solution to debug an equation
sol_dict = system.sol_to_dict(sol)

# Create a residual debugger for the Kinematics equation
kin_eq = Kinematics()
n0 = NodeVariables(0)

# Map equation's nodes (0, 1, ...) to global nodes (0, 1, ...)
debug_vars = residual_debugger(kin_eq, [0], sol_dict)

# Now inspect in interactive mode:
# - Access equation: debug_vars['self']
# - Access solution data: debug_vars['V0'], debug_vars['p0'], etc.
# - Compute residual manually: debug_vars['self'].residual(...)
```

The `residual_debugger` takes:

1. **equation** — The equation instance to debug
2. **glob_nodes** — Mapping of equation's local nodes to global node indices
3. **data** — The solution dictionary (from `sol_to_dict`)

It returns a dictionary where you can:

- Access the equation instance: `debug_vars['self']`
- Access all variables the equation uses: `debug_vars['V0']`, `debug_vars['p0']`, etc.
- Call the equation's residual method directly: `debug_vars['self'].residual(...)`

**Step 3: Interactive Debugging in IPython**

This is powerful in an interactive environment:

```python
# In IPython or Jupyter:
sol = solve_root_problem(rtfn, x0, knowns)
sol_dict = system.sol_to_dict(sol)

# Debug a specific equation
from adet.equations.fundamental import Kinematics
kin_eq = Kinematics()

debug_vars = residual_debugger(kin_eq, [0], sol_dict)
globals().update(debug_vars)  # Import into namespace

# Now you can directly access and inspect variables
print(f"V0 = {V0}")
print(f"theta0 = {theta0}")

# Manually evaluate the residual
r = self.residual(V0, theta0, V_abs, theta_abs)
print(f"Residual: {r}")
```

This lets you **step through individual equation logic** with the solution values, making it easy to spot errors in your equation definitions.

### Improve Initial Guesses

If the default guess is poor, you can:

1. **Manually set guess values** in variable specifications:

```python
from adet.varspec import VarSpec

my_var = VarSpec('Pressure', 'Pa', guess=5e5)  # Better guess
```

2. **Use physical reasoning** to set boundary conditions that help the solver:

```python
BC = {
    n0.tot.Pressure: 101325,           # Near atmospheric
    n0.tot.Temperature: 288,           # Standard conditions
    n0.geo.Radius: 0.05,               # Conservative size
}
```

3. **Reduce the system size** by fixing more variables initially, then solving incrementally:

```python
# Step 1: Solve with fewer unknowns
BC_initial = {...}
system1.add_boundary_conditions(BC_initial)
system1.build()
sol1 = solve_root_problem(...)

# Step 2: Use solution as guess for larger system
BC_extended = {**BC_initial, **new_bc}
system2.add_boundary_conditions(BC_extended)
x0_better = system2.get_scaled_guess()  # Uses sol1 values
sol2 = solve_root_problem(..., guess=x0_better)
```

### Check for Physical Inconsistencies

Solver failure often indicates physically impossible boundary conditions:

```python
# These boundary conditions are inconsistent:
BC = {
    n0.tot.Pressure: 1e5,       # Low pressure
    n0.tot.Temperature: 3000,   # Very high temperature
    n0.kin.MachNumber: 5.0,     # Supersonic
    # Mach > 1 at high T and low P is unlikely in most contexts
}
```

Verify your input assumptions against:

- Physical law (conservation of energy, mass)
- Material properties (e.g., real gases at extreme conditions)
- Component limits (e.g., blade materials can't handle arbitrarily high temperatures)

### Enable Solver Logging

By default, `solve_root_problem()` suppresses solver output. Enable logging to see iteration details:

```python
sol = solve_root_problem(
    rootfinder,
    guess=x0,
    knowns=knowns,
    suppress_output=False,  # Show solver iterations
)
```

Look for:

- **Residual norm decreasing**: Solver is converging
- **Residual norm increasing or stalling**: Poor initial guess or ill-conditioned system
- **NaN or Inf**: Numerical issue (e.g., EOS failure at unphysical state)

## Understanding Residual Structure

The residual function is the key to understanding your system's behavior. Since all equations are concatenated into one big vector, you need to understand which residuals correspond to which equations.

### Equation Order in the Residual Vector

When you add equations to the system, they're compiled in order:

```python
system.add_equation(Eq1(), 0)   # Contributes residuals [0, ...]
system.add_equation(Eq2(), 0)   # Contributes residuals [..., N]
system.add_equation(Eq3(), 0)   # Contributes residuals [..., M]
```

If Eq1 returns 2 residuals, Eq2 returns 1, and Eq3 returns 3, then the big residual vector has 6 entries arranged as:

```
r = [r_Eq1_0, r_Eq1_1, r_Eq2_0, r_Eq3_0, r_Eq3_1, r_Eq3_2]ᵀ
```

Use the system's logging to see the arrangement:

```python
import logging
logging.basicConfig(level=logging.INFO)

system.build()  # Will print equation details including ordering
```

### Identifying Problem Equations

If solving fails, which equation is causing trouble?

```python
res_func = system.make_residual_function()
residuals = res_func(x0, knowns)

# Find equations with large residuals
problem_indices = np.where(np.abs(residuals) > 0.1)[0]

print(f"Problem residuals at indices: {problem_indices}")
```

Then use the residual debugger on the equations at those indices to understand why they're not satisfied.

## Advanced Solving Strategies

### Perturbation-Based Search

For stubborn systems, try perturbed initial guesses:

```python
sol = solve_root_problem(
    rootfinder,
    guess=x0,
    knowns=knowns,
    perturbate_guess=True,
    num_samples=100,  # Try 100 Latin hypercube samples
    delta_pert=0.05,  # ±5% perturbation
)
```

ADeT will try multiple perturbed guesses and return the first successful solution.

### Switching Between Solvers

If KINSOL fails, try IPOPT:

```python
# First try: fast KINSOL
rtfn_kinsol = system.make_rootfinder('kinsol')
try:
    sol = solve_root_problem(rtfn_kinsol, x0, knowns)
except RuntimeError:
    print("KINSOL failed, trying IPOPT...")
    rtfn_ipopt = system.make_rootfinder('ipopt')
    sol = solve_root_problem(rtfn_ipopt, x0, knowns)
```

### Solver Options

Customize solver behavior by passing options to `make_rootfinder()`:

```python
rtfn = system.make_rootfinder('kinsol', opts={
    'max_iter': 500,              # Increase max iterations
    'abs_tol': 1e-8,              # Tighter tolerance
})
```

For IPOPT:

```python
rtfn = system.make_rootfinder('ipopt', opts={
    'ipopt.max_iter': 1000,
    'ipopt.constr_viol_tol': 1e-7,
})
```

## Working with Solutions

### Post-Processing Results

Once you have a solution, extract and process the results:

```python
sol_dict = system.sol_to_dict(sol)

# Extract specific variables
p0 = sol_dict[n0.stc.Pressure]
T0 = sol_dict[n0.stc.Temperature]
rho0 = sol_dict[n0.stc.Density]
V0 = sol_dict[n0.kin.V_abs]

print(f"Static pressure: {p0 / 1e5:.2f} bar")
print(f"Static temperature: {T0:.1f} K")
print(f"Density: {rho0:.3f} kg/m³")
print(f"Velocity: {V0:.1f} m/s")
```

### Validating Solutions

Always check that your solution is physically reasonable:

```python
# Residual check
residuals = residual_fn(sol, knowns)
res_norm = np.linalg.norm(residuals)
print(f"Final residual norm: {res_norm:.3e}")

# Physical validity
if p0 < 0 or T0 < 0:
    print("Warning: Non-physical solution!")

# Consistency checks
h_tot = sol_dict[n0.tot.Enthalpy]
h_stat = sol_dict[n0.stc.Enthalpy]
V = sol_dict[n0.kin.V_abs]

h_check = h_stat + V**2 / 2
if abs(h_tot - h_check) > 1:
    print(f"Warning: Energy not conserved! {h_tot} vs {h_check}")
```

### Saving and Loading Solutions

Convert solutions to NumPy arrays for post-processing or visualization:

```python
import numpy as np

# Extract values only (lose variable names)
values = np.array([v for v in sol_dict.values()])

# Or save with labels
data = {
    'pressure': sol_dict[n0.stc.Pressure],
    'temperature': sol_dict[n0.stc.Temperature],
    'velocity': sol_dict[n0.kin.V_abs],
}

np.savez('solution.npz', **data)
```

## Complete Solve Example

Here's a complete example with error handling:

```python
from adet.assembly import CasadiSystem
from adet.equations.fundamental import (
    Kinematics, MassAreaRelation, TotalStaticMatching, ZeroBlockage
)
from adet.equations.geometrical import AnnulusAreas
from adet.equations.nondimensional import AbsoluteMachNumber, GammaIdeal
from adet.equations.special import ThermoVarsAdder
from adet.variables import NodeVariables, ThermoVariables
from adet.fluid.settings import AnalyticalFluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.solution import solve_root_problem
from pint import Quantity
import numpy as np

system = CasadiSystem()

model = AnalyticalFluidModel(IdealGasState(1.4, 287, 2e-5))
thrm = ThermoVariables()
fluid_settings = FluidSettings(model, (thrm.Pressure, thrm.Temperature))
system.fluid_settings = fluid_settings

n0 = NodeVariables(0)

EQUATIONS = {
    TotalStaticMatching(): 0,
    AnnulusAreas(): 0,
    MassAreaRelation(): 0,
    AbsoluteMachNumber(): 0,
    ZeroBlockage(): 0,
    Kinematics(): 0,
    ThermoVarsAdder(): 0,
    GammaIdeal(): 0,
}

for eq, pos in EQUATIONS.items():
    system.add_equation(eq, pos)

BC = {
    n0.kin.Omega: 0.0,
    n0.kin.FlowAngleAbs: Quantity(0, 'deg'),
    n0.oth.MassFlow: 0.132,
    n0.geo.RDistr: 0.038,
    n0.geo.HDistr: 0.002,
    n0.tot.Pressure: 18.1e5,
    n0.tot.Temperature: 573.15,
}

system.add_boundary_conditions(BC)
system.build()

rtfn = system.make_rootfinder('kinsol')
x0 = system.get_scaled_guess()
knowns = system.get_scaled_constraints()

try:
    sol = solve_root_problem(rtfn, x0, knowns, suppress_output=False)
    sol_dict = system.sol_to_dict(sol)

    print("Solution found!")
    print(f"Static pressure: {sol_dict[n0.stc.Pressure] / 1e5:.2f} bar")
    print(f"Mach number: {sol_dict[n0.oth.Mach]:.3f}")

except RuntimeError as e:
    print(f"Solve failed: {e}")
    print("Trying IPOPT...")

    rtfn_ipopt = system.make_rootfinder('ipopt')
    sol = solve_root_problem(rtfn_ipopt, x0, knowns)
    sol_dict = system.sol_to_dict(sol)

    print("IPOPT succeeded!")
```


