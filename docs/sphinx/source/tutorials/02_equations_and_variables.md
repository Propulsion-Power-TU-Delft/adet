# Equations and Variables

This tutorial covers the core abstractions in ADeT: how equations are defined, how variables are specified with units and bounds, and how the system manages variable access.

## Variables: The Foundation

Everything in ADeT is built around variables. Variables represent physical quantities (pressure, temperature, velocity) and are accessed through a structured hierarchy.

### Variable Specifications (VarSpec)

A `VarSpec` is the atomic unit describing a variable: its name, units, and physical properties.

```python
from adet.varspec import VarSpec
from adet.variables import ThermoVariables

# Create a custom variable specification
my_var = VarSpec('MyVar', 'Pa', guess=1e5, bounds=(1, 150e5))
```

The parameters are:

- **name** (str): Symbolic name for the variable (e.g., `'Pressure'`)
- **units** (str): Pint-compatible unit string (e.g., `'Pa'`, `'J / kg / K'`)
- **guess** (float, optional): Initial guess for solving
- **bounds** (tuple, optional): Physical bounds `(min, max)` for the variable

### Predefined Thermo Variables

ADeT provides a `ThermoVariables` class with common thermodynamic properties already defined:

```python
from adet.variables import ThermoVariables

thrm = ThermoVariables()

# Access properties with units and metadata
print(thrm.Pressure)        # VarSpec for pressure
print(thrm.Temperature)     # VarSpec for temperature
print(thrm.Enthalpy)        # VarSpec for enthalpy
print(thrm.Entropy)         # VarSpec for entropy
print(thrm.Density)         # VarSpec for density
```

Each property includes:

- Unit specification
- A default guess value
- Physical bounds (where applicable)
- Scaling hints for numerical solvers

### Node Variables

`NodeVariables` groups variables by their location in the flow (node) and state type:

```python
from adet.variables import NodeVariables

n0 = NodeVariables(0)  # Node 0
n1 = NodeVariables(1)  # Node 1

# Access state containers
n0.stc   # Static thermodynamic state
n0.tot   # Total thermodynamic state
n0.rlt   # Relative total thermodynamic state
n0.kin   # Kinematics (velocities, angles)
n0.geo   # Geometry (radii, heights, areas)
n0.oth   # Other properties (entropy, losses, flow conditions)
```

Each container provides property access:

```python
n0.tot.Pressure           # Total pressure at node 0
n0.stc.Temperature        # Static temperature at node 0
n0.kin.FlowAngleAbs       # Absolute flow angle at node 0
n0.oth.MassFlow           # Mass flow at node 0
```

### Working with Units

All variables carry unit information. When accessing a variable, you can work with quantities:

```python
from pint import Quantity

# Boundary condition with explicit units
BC = {
    n0.kin.FlowAngleAbs: Quantity(0, 'deg'),    # Axial flow
    n0.tot.Pressure: Quantity(18.1, 'bar'),
}
```

## Equations: Defining Residuals

An equation in ADeT is any residual that should equal zero. Equations are defined by subclassing `EquationBase`.

### Basic Equation Structure

```python
from adet.equations.base_equation import EquationBase, EquationConfig
import numpy as np

class SimpleEquation(EquationBase):
    """Define an equation by implementing the residual method."""

    def residual(self, var1, var2, var3):
        # Parameters must match variable names in your nodes
        r = var1 + var2 - var3  # This residual should equal zero
        return r
```

- **Method signature**: The parameter names must match variable names in your system (e.g., `stc_p0` for static pressure at node 0)
- **Return value**: Can return a scalar residual or tuple of residuals
- **Symbolic computation**: Parameters are symbolic variables from CasADi, supporting automatic differentiation

### Equation Hints and Type Hints

To ensure correct variable matching, use *hints* to specify which variable each parameter represents:

```python
from adet.variables import NodeVariables
n0 = NodeVariables(0)
n1 = NodeVariables(1)

class MassConservation(EquationBase):
    def residual(
        self,
        rho0: n0.stc.Density.Hint,      # Density at node 0
        A0: n0.geo.Area.Hint,            # Area at node 0
        V0: n0.kin.V_abs.Hint,           # Velocity at node 0
        rho1: n1.stc.Density.Hint,
        A1: n1.geo.Area.Hint,
        V1: n1.kin.V_abs.Hint,
    ):
        # Mass balance: ρ₀·A₀·V₀ = ρ₁·A₁·V₁
        r = (rho0 * A0 * V0) - (rho1 * A1 * V1)
        return r
```

The `.Hint` attribute tells ADeT which variable each parameter represents. This is required for ADeT to match function parameters to system variables.

### Equation Configuration

Some equations need special configuration, particularly when they interact with the equation of state (EOS):

```python
from adet.equations.base_equation import EquationConfig
import CoolProp as cp

class ThermoRelation(EquationBase):
    config = EquationConfig(
        input_pair=cp.HmassSmass_INPUTS,              # EOS inputs: h, s
        out_properties=(thrm.Pressure, thrm.Density), # EOS outputs: p, ρ
    )

    def residual(self, h, s, p, rho):
        # Use self.eos() to compute properties from the configured inputs
        p_eos, rho_eos = self.eos(h, s)

        r1 = p - p_eos
        r2 = rho - rho_eos
        return r1, r2
```

The `EquationConfig` specifies:

- **input_pair**: Which CoolProp inputs the EOS should use (e.g., `HmassSmass_INPUTS` means enthalpy and entropy)
- **out_properties**: Which properties to compute from the EOS

When you call `self.eos(h, s)`, ADeT uses the configured EOS and fluid model to compute the outputs.

### Built-in Equations

ADeT provides many pre-built equations covering common physics:

**Fundamental equations** (`adet.equations.fundamental`):
- `TotalStaticMatching` — Relate total and static properties
- `Kinematics` — Velocity components from magnitude and angles
- `MassAreaRelation` — Mass flow continuity
- `ZeroBlockage` — No blockage (effective area = geometric area)

**Thermodynamic relations** (`adet.equations.nondimensional`):
- `GammaIdeal` — Heat capacity ratio for ideal gas
- `AbsoluteMachNumber` — Mach number from velocity and speed of sound

**Geometric equations** (`adet.equations.geometrical`):
- `AnnulusAreas` — Compute annulus areas from hub and tip radii

You can use these directly without defining your own:

```python
from adet.equations.fundamental import Kinematics, MassAreaRelation
from adet.equations.nondimensional import GammaIdeal

system.add_equation(Kinematics(), 0)
system.add_equation(MassAreaRelation(), 0)
system.add_equation(GammaIdeal(), 0)
```

## How Equations Are Assembled into a System

This section explains the architecture that makes ADeT powerful: how individual equations are combined into a single large system.

### From Individual Equations to a Residual Function

When you write an equation like:

```python
class MyEquation(EquationBase):
    def residual(self, p, T, rho):
        return p / (R * T) - rho  # One scalar residual
```

You're writing one constraint: `p / (R·T) - ρ = 0`.

When you add multiple equations to the system:

```python
system.add_equation(MyEquation(), 0)
system.add_equation(EnergyBalance(), 0)
system.add_equation(MassBalance(), 0)
system.build()
```

ADeT:

1. **Extracts residual expressions** from each equation
2. **Concatenates them** into a single big vector: `r(x, p) = [r₁, r₂, r₃, ...]ᵀ`
3. **Compiles a CasADi function** that evaluates all residuals at once
4. **Computes analytical Jacobians** automatically

The result is a **nonlinear root-finding problem**:

```
Find x such that r(x, p) = 0
```

where:

- **x** is the vector of all unknown variables
- **p** is the vector of known constraint values (boundary conditions)
- **r** is the concatenated residual from all equations

This is **why you need equation-variable consistency**: The solver needs exactly as many equations (residuals) as unknowns. If you have 10 unknowns and 8 equations, the system is under-determined. If you have 10 unknowns and 12 equations, the system is over-determined.

### Accessing the Compiled Residual Function

After `build()`, you can access and evaluate the big residual function:

```python
system.build()

# Get the compiled residual function
res_func = system.make_residual_function()

# Evaluate at any point
x = system.get_scaled_guess()
p = system.get_scaled_constraints()

residuals = res_func(x, p)

# Check which equations are satisfied
print(f"Residual norms: {np.abs(residuals)}")
print(f"System residual norm: {np.linalg.norm(residuals)}")
```

This is useful for:

- **Debugging** — Which equations are problematic?
- **Monitoring** — How close are we to a solution?
- **Analysis** — Understanding equation interactions

The residual function is **the core** of ADeT's power. Understanding it is key to effective use of the system.

## System States and Variables

Understanding how ADeT organizes variables within nodes is essential for defining equations correctly.

### Node State Containers

Each `NodeVariables` instance contains six state containers:

```python
n = NodeVariables(0)

n.stc   # Static properties (pressure, temperature, density at static conditions)
n.tot   # Total properties (measured in a frame where flow is brought to rest)
n.rlt   # Relative total properties (in the rotor frame if rotating)
n.kin   # Kinematics (velocities, flow angles, rotation speed)
n.geo   # Geometry (radii, heights, areas)
n.oth   # Other derived properties (entropy, shock parameters, loss coefficients)
```

The **state structure** allows equations to reference the correct variables:

- Use `stc` when you need static pressure/temperature
- Use `tot` for total pressure/temperature (useful for isentropic relations)
- Use `rlt` in turbomachinery components with blade rows
- Use `kin` for velocity-related equations
- Use `geo` for geometric and area-related equations

## Building a Complete Example

Let's build a small system with custom equations:

```python
from adet.assembly import CasadiSystem
from adet.equations.base_equation import EquationBase, EquationConfig
from adet.variables import NodeVariables, ThermoVariables
from adet.fluid.settings import AnalyticalFluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.solution import solve_root_problem
from pint import Quantity
import CoolProp as cp

n0 = NodeVariables(0)
thrm = ThermoVariables()

# Define a custom energy equation
class EnergyBalance(EquationBase):
    """Total enthalpy is constant."""
    def residual(
        self,
        h_tot0: n0.tot.Enthalpy.Hint,
        h_tot1: n0.tot.Enthalpy.Hint,
    ):
        return h_tot0 - h_tot1  # Energy conserved

# Set up system
system = CasadiSystem()
model = AnalyticalFluidModel(IdealGasState(1.4, 287, 2e-5))
fluid_settings = FluidSettings(model, (thrm.Pressure, thrm.Temperature))
system.fluid_settings = fluid_settings

system.add_equation(EnergyBalance(), 0)

BC = {
    n0.tot.Pressure: Quantity(101325, 'Pa'),
    n0.tot.Temperature: Quantity(288, 'K'),
}
system.add_boundary_conditions(BC)
system.build()

rtfn = system.make_rootfinder('kinsol')
x0 = system.get_scaled_guess()
kn = system.get_scaled_constraints()

sol = solve_root_problem(rtfn, x0, kn)
sol_dict = system.sol_to_dict(sol)
```

## Next Steps

- [Solving Systems](03_solving_systems.md) — Learn strategies for solving complex systems and debugging failures
- [Fluid Models](04_fluid_models.md) — Understand how to work with different fluid models and EOS backends
