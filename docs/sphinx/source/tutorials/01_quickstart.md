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
from adet.assembly import CasadiSystem
from adet.equations.fundamental import (
    Kinematics,
    MassAreaRelation,
    TotalStaticMatching,
    ZeroBlockage,
)
from adet.equations.geometrical import AnnulusAreas
from adet.equations.nondimensional import AbsoluteMachNumber, GammaIdeal
from adet.equations.special import ThermoVarsAdder
from adet.variables import NodeVariables, ThermoVariables
from adet.fluid.settings import AnalyticalFluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.solution import solve_root_problem
from pint import Quantity

# Create the system
system = CasadiSystem()
```

### Step 2: Configure the Fluid Model

Specify which thermodynamic model and update variables the system should use:

```python
# Use ideal gas with gamma=1.4, R=287 J/(kg·K)
model = AnalyticalFluidModel(IdealGasState(1.4, 287, 2e-5))

# Define which variables to compute from the EOS
thrm = ThermoVariables()
fluid_settings = FluidSettings(model, (thrm.Pressure, thrm.Temperature))

system.fluid_settings = fluid_settings
```

The `FluidSettings` tells ADeT which variables should be computed from the equation of state. In this case, we're computing pressure and temperature from enthalpy and entropy.

### Step 3: Define the Equations

Add the equations that govern your system. Each equation is represented as a residual that should equal zero:

```python
EQUATIONS = {
    TotalStaticMatching(): 0,      # Relate total and static properties
    AnnulusAreas(): 0,             # Compute annulus areas from radii
    MassAreaRelation(): 0,         # Mass flow continuity
    AbsoluteMachNumber(): 0,       # Compute Mach number
    ZeroBlockage(): 0,             # Area = Effective area
    Kinematics(): 0,               # Velocity components
    ThermoVarsAdder(): 0,          # Add secondary thermo properties
    GammaIdeal(): 0,               # Heat capacity ratio
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

# Get a scaled guess for unknowns and known constraint values
x0 = system.get_scaled_guess()
kn = system.get_scaled_constraints()
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

## Full Example

Here's the complete working example:

```python
from adet.assembly import CasadiSystem
from adet.equations.fundamental import (
    Kinematics,
    MassAreaRelation,
    TotalStaticMatching,
    ZeroBlockage,
)
from adet.equations.geometrical import AnnulusAreas
from adet.equations.nondimensional import AbsoluteMachNumber, GammaIdeal
from adet.equations.special import ThermoVarsAdder
from adet.variables import NodeVariables, ThermoVariables
from adet.fluid.settings import AnalyticalFluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.solution import solve_root_problem
from pint import Quantity

system = CasadiSystem()

model = AnalyticalFluidModel(IdealGasState(1.4, 287, 2e-5))
thrm = ThermoVariables()
fluid_settings = FluidSettings(model, (thrm.Pressure, thrm.Temperature))
system.fluid_settings = fluid_settings

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
x0 = system.get_scaled_guess()
kn = system.get_scaled_constraints()

sol = solve_root_problem(rtfn, x0, kn)
sol_dict = system.sol_to_dict(sol)

for var_spec, value in sol_dict.items():
    print(f"{var_spec}: {value}")
```

## What's Next?

- [Equations and Variables](02_equations_and_variables.md) — Learn about how equations are structured and how to work with variables
- [Solving Systems](03_solving_systems.md) — Explore different solving strategies and debugging techniques
- [Fluid Models](04_fluid_models.md) — Understand thermodynamic models and equation of state selection
