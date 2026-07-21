# Quickstart: Setting Up Your First System

In this tutorial, we'll create a minimal but complete example of using ADeT's equation-based modeling system. We'll define a simple thermodynamic system, add equations, specify boundary conditions, and solve for the unknowns.

## What You'll Learn

- How to create a `CasadiSystem` instance
- How to access and define variables using `NodeVariables` and `ThermoVariables`
- How to add equations to the system
- How to set boundary conditions
- How to build and solve the system

## Prerequisites

You should have ADeT installed. If not, run:

```bash
uv sync
```

## Creating a Simple Mach Number System

Let's create a system to compute properties of a flow at a specified Mach number. We'll need:

1. A flow state with known pressure, temperature, and Mach number
2. Equations to relate the properties (kinematics, total-static relations, ideal gas relations)

### Step 1: Import and Setup

```python
from adet.assemblers import CasadiSystem
from adet.equations.fundamental import (
    Kinematics,
    MassAreaRelation,
    TotalStaticMatching,
    ZeroBlockage,
)
from adet.equations.geometrical import AnnulusAreas
from adet.equations.nondimensional import AbsoluteMachNumber
from adet.fluid.ideal_eos import IdealGasState
from adet.fluid.settings import FluidSettings
from adet.solution import solve_root_problem
from adet.variables import NodeVariables, ThermoVariables
from pint import Quantity

# Create the system
system = CasadiSystem()
```

### Step 2: Configure the Fluid Model

Specify which thermodynamic model and update variables the system should use:

```python
# Use ideal gas with gamma=1.4, R=287 J/(kg·K)
ideal_state = IdealGasState(1.4, 287, 2e-5)

# Define which variables to compute from the EOS
thrm = ThermoVariables()
fluid_settings = FluidSettings(
    fluid_state=ideal_state,
    update_variables=(thrm.Pressure, thrm.Temperature),
)

system.fluid_settings = fluid_settings
```

The `FluidSettings` tells ADeT which thermodynamic variables should be used for state updates as . In this case, we're using pressure and temperature to update the thermodynamic state at each iteration; the rest of thermodynamic variables are then extracted to be plugged into the residual.

### Step 3: Define the Equations

Add the equations that govern your system. Each equation is represented as a residual that should equal zero:

```python
EQUATIONS = {
    AnnulusAreas(): 0,             # A = 2 pi r H
    MassAreaRelation(): 0,         # m_dot = rho V A
    AbsoluteMachNumber(): 0,       # Define Mach number
    TotalStaticMatching(): 0,      # Matches total and static state
    ZeroBlockage(): 0,             # No blockage in passage
    Kinematics(): 0,               # Defines angles velocity
}

for eq, pos in EQUATIONS.items():
    system.add_equation(eq, pos)
```

The second parameter (`pos`) specifies the position in the equation ordering. A value of `0` means it can be placed anywhere.

### Step 4: Set Boundary Conditions

Specify the known values at your nodes:

```python
n0 = NodeVariables(0)

BC = {
    n0.kin.Omega: 0.0,                          # No rotation
    n0.kin.FlowAngleAbs: Quantity(0, 'deg'),    # Axial flow
    n0.oth.MassFlow: 0.132,                     # kg/s
    n0.geo.RDistr: 0.038,                       # Radius distribution
    n0.geo.HDistr: 0.002,                       # Height distribution
    n0.tot.Pressure: 18.1e5,                    # Total pressure, Pa
    n0.tot.Temperature: 573.15,                 # Total temperature, K
}

system.add_boundary_conditions(BC)
```

Each boundary condition fixes a variable value, reducing the problem size.

### Step 5: Build and Prepare to Solve

Compile the system:

```python
system.build()
```

This performs a crucial transformation: **all your equations are compiled into one big CasADi residual function**.

Behind the scenes, the system:

1. Collects all equation residuals: `r_eq1(x, p), r_eq2(x, p), ..., r_eqN(x, p)`
2. Concatenates them into a single vector: `r(x, p) = [r_eq1, r_eq2, ..., r_eqN]ᵀ`
3. Computes analytical Jacobians using automatic differentiation
4. Sets up numerical scaling for solver stability

This **big residual function** is what solvers use: they find **x** such that `r(x, p) = 0`.

Understanding this architecture is important for debugging (see [Solving Systems](03_solving_systems.md) for details on the residual debugger).

### Step 6: Create a Root Finder

ADeT supports multiple solvers. The `'kinsol'` solver is a Newton-Krylov method:

```python
rtfn = system.make_rootfinder('kinsol')

# Get a guess for unknowns and boundary condition values
x0 = system.get_guess()
kn = system.get_boundary_conds()
```

### Step 7: Solve the System

Use the root solving utility to find the solution:

```python
sol = solve_root_problem(rtfn, x0, kn)
```

The solver iterates until the residuals converge to zero (or reaches max iterations).

### Step 8: Extract Results

Convert the solution back to a dictionary of variables:

```python
sol_dict = system.sol_to_dict(sol)

# Access results
for var_spec, value in sol_dict.items():
    print(f"{var_spec}: {value}")
```

The `sol_dict` contains all variables (both solved and computed from the EOS), indexed by their `VarSpec` objects.

Single variables can also be added using the node object

```python
>>> sol_dict[node_0.kin.V_mag] # velocity magnitude
array([25.15642531])
```

## Full Example

Here's the complete working example, (also found in `examples/mach_problem.py`):

```python
from adet.assemblers import CasadiSystem
from adet.equations.fundamental import (
    Kinematics,
    MassAreaRelation,
    TotalStaticMatching,
    ZeroBlockage,
)
from adet.equations.geometrical import AnnulusAreas
from adet.equations.nondimensional import AbsoluteMachNumber
from adet.fluid.ideal_eos import IdealGasState
from adet.fluid.settings import FluidSettings
from adet.solution import solve_root_problem
from adet.variables import NodeVariables, ThermoVariables
from pint import Quantity

system = CasadiSystem()

ideal_state = IdealGasState(1.4, 287, 2e-5)
thrm = ThermoVariables()
fluid_settings = FluidSettings(
    fluid_state=ideal_state,
    update_variables=(thrm.Pressure, thrm.Temperature),
)
system.fluid_settings = fluid_settings

EQUATIONS = {
    AnnulusAreas(): 0,
    MassAreaRelation(): 0,
    AbsoluteMachNumber(): 0,
    TotalStaticMatching(): 0,
    ZeroBlockage(): 0,
    Kinematics(): 0,
}

for eq, pos in EQUATIONS.items():
    system.add_equation(eq, pos)

n0 = NodeVariables(0)
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
x0 = system.get_guess()
kn = system.get_boundary_conds()

sol = solve_root_problem(rtfn, x0, kn)
sol_dict = system.sol_to_dict(sol)

for var_spec, value in sol_dict.items():
    print(f"{var_spec}: {value}")
```

## What's Next?

- [Equations and Variables](02_equations_and_variables.md) — Learn about how equations are structured and how to work with variables
- [Solving Systems](03_solving_systems.md) — Explore different solving strategies and debugging techniques
- [Fluid Models](04_fluid_models.md) — Understand thermodynamic models and equation of state selection
