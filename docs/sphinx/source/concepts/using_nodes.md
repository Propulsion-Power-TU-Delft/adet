# An introduction to nodes

## `NodeVariables`
Convenience

## Variables: The Foundation

Everything in ADeT is built around variables. Variables represent physical quantities (pressure, temperature, velocity) and are accessed through a structured hierarchy.

### Variable Specifications (VarSpec)

A `VarSpec` is the atomic unit describing a variable: its name, units, guess and bounds.

```python
from adet.varspec import VarSpec
from adet.variables import ThermoVariables

# Create a custom variable specification
my_var = VarSpec('MyVar', 'dimensionless', guess=10, bounds=(-1000, 1000))
```

The parameters are:

- **name** (str): Symbolic name for the variable (e.g., `'Pressure'`)
- **units** (str): Pint-compatible unit string (e.g., `'Pa'`, `'J / kg / K'`)
- **guess** (float, optional): Initial guess for solving
- **bounds** (tuple, optional): Physical bounds `(min, max)` for the variable
- **node** (int, optional): 
- **scalar** (bool, optional): Whether this quantity should be always treated as a scalar (spanwise uniform) quantity

### Variable Enums

ADeT provides a series of enums already contaning a large number of pre-defined variables relevant to preliminary problems in the field of turbomachinery. They are filed under the following categories 

- Kinematic variables `KinematicVariables`
- Geometric variables `GeometricVariables`
- Nondimensional groups/ratios `Nondimensional`
- Entropy generation related quantities `Losses`
- Variables that do not strictly belong to any of the above `OtherVariables`

A special container for thermodynamic variables, that can made to access either the static, total or relative total state.
- Thermodynamic variables `ThermoVariables`

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

### Node Variables and State Containers

`NodeVariables` organizes variables by their location in the flow (node) and state type. Create a `NodeVariables` instance for each node in your system:

```python
from adet.variables import NodeVariables

n0 = NodeVariables(0)  # Node 0
n1 = NodeVariables(1)  # Node 1
n2 = NodeVariables(2)  # Node 2
```

Each `NodeVariables` instance contains six state containers:

```python
n0.stc   # Static thermodynamic state
n0.tot   # Total thermodynamic state (isentropic/reversible)
n0.rlt   # Relative total thermodynamic state (rotor reference frame)
n0.kin   # Kinematics (velocities, flow angles, rotation speed)
n0.geo   # Geometry (radii, heights, areas)
n0.oth   # Other properties (entropy, shock parameters, loss coefficients)
```

Each state container provides access to `VarSpec` objects for specific properties:

```python
n0.tot.Pressure           # Total pressure at node 0
n0.stc.Temperature        # Static temperature at node 0
n0.kin.FlowAngleAbs       # Absolute flow angle at node 0
n0.oth.MassFlow           # Mass flow at node 0
```

You can then use the `.Hint` attribute in equation residuals, just like with custom `VarSpec` variables:

```python
class EnergyBalance(EquationBase):
    def residual(
        self,
        h0: n0.tot.Enthalpy.Hint,
        h1: n1.tot.Enthalpy.Hint,
    ):
        return h0 - h1  # Total enthalpy conserved between nodes
```

### Using ThermoVariables with Node States

For thermodynamic properties, you can use `ThermoVariables` to access pre-defined property specifications, and then apply them to specific nodes using `NodeVariables`. This is useful when you want to specify a node and state at definition time:

```python
from adet.variables import NodeVariables, ThermoVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)
thrm = ThermoVariables()

# Access thermodynamic variables with node state
p_tot_0 = n0.tot.Pressure        # Total pressure at node 0
T_stc_1 = n1.stc.Temperature     # Static temperature at node 1
rho_tot_0 = n0.tot.Density       # Total density at node 0
```

Alternatively, you can create `ThermoVariables()` and then access properties through `NodeVariables`:

```python
thrm = ThermoVariables()

# thrm.Pressure is a generic thermodynamic variable
# but you specify the node state when writing equations
class MyEquation(EquationBase):
    def residual(
        self,
        p: n0.tot.Pressure.Hint,      # Use it at node 0, total state
        T: n0.tot.Temperature.Hint,    # Use it at node 0, total state
    ):
        # Your equation using p and T
        pass
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

An equation in ADeT is any residual that should equal zero. Equations are defined by subclassing `EquationBase` and implementing a `residual()` method.

### Basic Equation Structure

The simplest way to write an equation is with the `.Hint` attribute:

```python
from adet.equations.base_equation import EquationBase

class SimpleEquation(EquationBase):
    def residual(
        self,
        var1: my_x.Hint,
        var2: my_y.Hint,
        var3: my_z.Hint,
    ):
        r = var1 + var2 - var3  # This residual should equal zero
        return r
```

- **Subclass `EquationBase`**: All equations inherit from this abstract class
- **Implement `residual()`**: Method that defines your constraint(s)
- **Use `.Hint` for variable linking**: Each parameter needs a type hint with `.Hint` so ADeT knows which variable to pass
- **Return value**: Can be a scalar or tuple of residuals

The `.Hint` attribute tells ADeT which variable each parameter represents, ensuring automatic variable matching.

### Example: Mass Conservation

Here's a practical example with multiple nodes:

```python
from adet.variables import NodeVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)

class MassConservation(EquationBase):
    def residual(
        self,
        rho0: n0.stc.Density.Hint,      # Density at node 0, static state
        A0: n0.geo.Area.Hint,            # Area at node 0, geometry
        V0: n0.kin.V_abs.Hint,           # Velocity at node 0, kinematics
        rho1: n1.stc.Density.Hint,      # Density at node 1, static state
        A1: n1.geo.Area.Hint,            # Area at node 1, geometry
        V1: n1.kin.V_abs.Hint,           # Velocity at node 1, kinematics
    ):
        # Mass balance: ρ₀·A₀·V₀ = ρ₁·A₁·V₁
        r = (rho0 * A0 * V0) - (rho1 * A1 * V1)
        return r
```

This equation:
- References **two different nodes** (node 0 and node 1)
- Uses **different state containers** (`stc` for density, `geo` for area, `kin` for velocity)
- Returns a **single scalar residual** that should equal zero at the solution

### Returning Multiple Residuals

Some equations define multiple constraints. Return them as a tuple:

```python
class TwoConstraints(EquationBase):
    def residual(
        self,
        p: n0.tot.Pressure.Hint,
        T: n0.tot.Temperature.Hint,
        rho: n0.stc.Density.Hint,
    ):
        # Constraint 1: Ideal gas law
        R = 287  # Gas constant for air
        r1 = p - (rho * R * T)
        
        # Constraint 2: Some other constraint
        r2 = T - 300
        
        return r1, r2  # Return multiple residuals
```

When you return a tuple, ADeT adds each residual as a separate equation to the system.

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

This section explains how individual equations are combined into a single system that ADeT can solve.

### Building a System Step-by-Step

When you add multiple equations to a system:

```python
system = CasadiSystem()
system.add_equation(MyParabola(), 0)
system.add_equation(EnergyBalance(), 0)
system.add_equation(MassBalance(), 0)
system.build()
```

Behind the scenes, ADeT:

1. **Parses each equation's `residual()` method** using the `.Hint` attributes to identify which variables are needed
2. **Extracts residual expressions** from each equation, creating a symbolic representation
3. **Registers all variables** with their units, scaling, and bounds
4. **Counts equations vs. unknowns** to verify the system is well-posed
5. **Concatenates all residuals** into a single vector: `r(x, p) = [r₁, r₂, r₃, ...]ᵀ`
6. **Compiles with CasADi** to create an efficient function that evaluates all residuals
7. **Computes analytical Jacobians** automatically for efficient numerical solving

### The Nonlinear Root-Finding Problem

After assembly, you have a nonlinear root-finding problem:

```
Find x such that r(x, p) = 0
```

Where:
- **x**: Vector of all unknown variables (the ones not fixed by boundary conditions)
- **p**: Vector of known values from boundary conditions
- **r**: Vector of concatenated residuals from all equations

### Equation-Variable Consistency

For a solvable system, you need:

**Number of equations (residuals) = Number of unknowns**

If you have 10 unknowns but only 8 equations, the system is **under-determined** (infinite solutions). If you have 10 unknowns and 12 equations, the system is **over-determined** (likely no solution). ADeT will check this and warn you during `build()`.

### Inspecting the Residual Function

After `build()`, you can access the compiled residual function for debugging:

```python
system.build()

# Get the compiled residual function
res_func = system.make_residual_function()

# Evaluate at any point
x = system.get_scaled_guess()
p = system.get_scaled_constraints()

residuals = res_func(x, p)

# Check which equations are satisfied
print(f"Residual values: {residuals}")
print(f"System residual norm: {np.linalg.norm(residuals)}")
```

This is useful for:
- **Debugging** — Which equations are problematic?
- **Monitoring** — How close are we to a solution?
- **Analysis** — Understanding equation interactions

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

## Building a Complete Example with NodeVariables

Let's build a more complex system using `NodeVariables` and thermodynamic equations. We'll build it step by step.

### Step 1: Set Up Nodes and Variables

First, create `NodeVariables` instances for each node in your system:

```python
from adet.variables import NodeVariables, ThermoVariables

n0 = NodeVariables(0)  # First node (inlet)
n1 = NodeVariables(1)  # Second node (outlet)
thrm = ThermoVariables()  # Access pre-defined thermodynamic properties
```

Now you can access variables at any node with any state:

```python
n0.tot.Pressure       # Total pressure at node 0
n0.stc.Temperature    # Static temperature at node 0
n1.tot.Enthalpy       # Total enthalpy at node 1
n1.kin.V_abs          # Absolute velocity at node 1
```

### Step 2: Define Equations Using NodeVariables

Write equations that reference variables across nodes:

```python
from adet.equations.base_equation import EquationBase

class EnergyConservation(EquationBase):
    """Total enthalpy is constant between nodes."""
    def residual(
        self,
        h0: n0.tot.Enthalpy.Hint,
        h1: n1.tot.Enthalpy.Hint,
    ):
        return h0 - h1

class MassFlow(EquationBase):
    """Mass flow is conserved."""
    def residual(
        self,
        rho0: n0.stc.Density.Hint,
        A0: n0.geo.Area.Hint,
        V0: n0.kin.V_abs.Hint,
        rho1: n1.stc.Density.Hint,
        A1: n1.geo.Area.Hint,
        V1: n1.kin.V_abs.Hint,
    ):
        return (rho0 * A0 * V0) - (rho1 * A1 * V1)
```

### Step 3: Set Up Fluid Model and System

Configure the thermodynamic model and create the system:

```python
from adet.assemblers import CasadiSystem
from adet.fluid.settings import AnalyticalFluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState

# Create an ideal gas model (gamma=1.4, R=287 J/kg/K, viscosity=2e-5 Pa·s)
model = AnalyticalFluidModel(IdealGasState(1.4, 287, 2e-5))

# Configure fluid settings: which variables update the thermodynamic state
# (Here: Pressure and Temperature determine all other properties)
fluid_settings = FluidSettings(model, (thrm.Pressure, thrm.Temperature))

system = CasadiSystem()
system.fluid_settings = fluid_settings
```

### Step 4: Add Equations and Boundary Conditions

Add your equations and specify which variables are known:

```python
from pint import Quantity

system.add_equation(EnergyConservation(), 0)
system.add_equation(MassFlow(), 0)

# Boundary conditions: fix these variables
BC = {
    n0.tot.Pressure: Quantity(101325, 'Pa'),
    n0.tot.Temperature: Quantity(288, 'K'),
    n0.geo.Area: Quantity(0.1, 'm**2'),
    n1.geo.Area: Quantity(0.12, 'm**2'),
}
system.add_boundary_conditions(BC)
```

### Step 5: Build and Solve

Compile the system and solve:

```python
from adet.solution import solve_root_problem

system.build()

# Create a root finder (Newton-Raphson via KINSOL)
rootfinder = system.make_rootfinder('kinsol')

# Get initial guess and constraints
x0 = system.get_scaled_guess()
knowns = system.get_scaled_constraints()

# Solve
solution = solve_root_problem(rootfinder, x0, knowns)

# Convert solution back to physical variables
sol_dict = system.sol_to_dict(solution)
print(sol_dict)
```

### Full Example

Here's the complete script:

```python
from adet.assemblers import CasadiSystem
from adet.equations.base_equation import EquationBase
from adet.variables import NodeVariables, ThermoVariables
from adet.fluid.settings import AnalyticalFluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.solution import solve_root_problem
from pint import Quantity

# Set up nodes
n0 = NodeVariables(0)
n1 = NodeVariables(1)
thrm = ThermoVariables()

# Define equations
class EnergyConservation(EquationBase):
    def residual(
        self,
        h0: n0.tot.Enthalpy.Hint,
        h1: n1.tot.Enthalpy.Hint,
    ):
        return h0 - h1

class MassFlow(EquationBase):
    def residual(
        self,
        rho0: n0.stc.Density.Hint,
        A0: n0.geo.Area.Hint,
        V0: n0.kin.V_abs.Hint,
        rho1: n1.stc.Density.Hint,
        A1: n1.geo.Area.Hint,
        V1: n1.kin.V_abs.Hint,
    ):
        return (rho0 * A0 * V0) - (rho1 * A1 * V1)

# Set up system
model = AnalyticalFluidModel(IdealGasState(1.4, 287, 2e-5))
fluid_settings = FluidSettings(model, (thrm.Pressure, thrm.Temperature))

system = CasadiSystem()
system.fluid_settings = fluid_settings
system.add_equation(EnergyConservation(), 0)
system.add_equation(MassFlow(), 0)

BC = {
    n0.tot.Pressure: Quantity(101325, 'Pa'),
    n0.tot.Temperature: Quantity(288, 'K'),
    n0.geo.Area: Quantity(0.1, 'm**2'),
    n1.geo.Area: Quantity(0.12, 'm**2'),
}
system.add_boundary_conditions(BC)
system.build()

rootfinder = system.make_rootfinder('kinsol')
x0 = system.get_scaled_guess()
knowns = system.get_scaled_constraints()

solution = solve_root_problem(rootfinder, x0, knowns)
sol_dict = system.sol_to_dict(solution)
```

## Next Steps

- [Solving Systems](03_solving_systems.md) — Learn strategies for solving complex systems and debugging failures
- [Fluid Models](04_fluid_models.md) — Understand how to work with different fluid models and EOS backends
