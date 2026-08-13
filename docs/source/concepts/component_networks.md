# Component Networks: High-Level API

While the low-level equation API gives you full control over every variable and residual, ADeT's component network API provides a higher-level abstraction for building turbomachinery systems from pre-defined components. This tutorial covers the network-based workflow.

## When to Use the Network API

Use the **network API** when:

- Building multi-stage turbomachinery systems (compressors, turbines)
- Components are connected in series (inlet → blade row → diffuser → outlet)
- You want automatic connection management (flow continuity, angle transfers, etc.)
- You're working with complex components that have many internal equations

Use the **low-level equation API** when:

- You need fine-grained control over every equation
- Building custom, non-standard systems
- Experimenting with new physics or correlations
- Your system doesn't fit the component paradigm

## Components and Connections

The network API revolves around **components** (inlet, blade row, diffuser) and **connections** between them.

### Available Components

ADeT provides several pre-built components:

- **Inlet** — Flow entry point, sets stagnation conditions
- **BladeRow** — Compressor or turbine stage with losses
- **SketchVanelessDiff** — Vaneless diffuser with geometric constraints

Each component handles:

- Thermodynamic relations (total-static, energy balance)
- Geometric constraints (annulus areas)
- Mass continuity
- Loss models (if applicable)

### The ComponentNetwork

The `ComponentNetwork` orchestrates components and defines how they connect:

```python
from adet.components.network import ComponentNetwork
from adet.components.inlet import Inlet
from adet.components.blade_row import BladeRow

# Create component instances
inlet = Inlet(node_idx=0)
blade_row = BladeRow(node_inlet=1, node_outlet=2)

# Create network
network = ComponentNetwork()
network.add_component(inlet)
network.add_component(blade_row)

# Define connections (handled by network)
network.connect(inlet, blade_row)  # Auto-connect outlet to inlet

# Build system with network
system = CasadiSystem()
system.from_network(network)
```

## High-Level Workflow

Here's the typical workflow using the network API:

**Step 1: Create Components**

```python
from adet.components.inlet import Inlet
from adet.components.blade_row import BladeRow
from adet.components.network import ComponentNetwork

inlet = Inlet(node_idx=0, name='inlet')
stage = BladeRow(
    node_inlet=1,
    node_outlet=2,
    name='compressor_stage_1',
    is_turbine=False,  # Compressor
)
```

**Step 2: Create and Configure Network**

```python
network = ComponentNetwork()

# Add components
network.add_component(inlet)
network.add_component(stage)

# Define connections (flow path)
network.connect(inlet, stage)
```

**Step 3: Set Boundary Conditions**

```python
from adet.variables import NodeVariables
from pint import Quantity

n0 = NodeVariables(0)

BC = {
    n0.tot.Pressure: Quantity(101325, 'Pa'),
    n0.tot.Temperature: Quantity(288, 'K'),
    n0.kin.FlowAngleAbs: Quantity(0, 'deg'),
    n0.oth.MassFlow: 1.0,  # kg/s
}
```

**Step 4: Build and Solve**

```python
system = CasadiSystem()
system.from_network(network)
system.fluid_settings = fluid_settings
system.add_boundary_conditions(BC)
system.build()

rtfn = system.make_rootfinder('kinsol')
x0 = system.get_scaled_guess()
knowns = system.get_scaled_constraints()

sol = solve_root_problem(rtfn, x0, knowns)
sol_dict = system.sol_to_dict(sol)
```

**Step 5: Extract Results**

```python
n1 = NodeVariables(1)  # Blade row outlet
p_out = sol_dict[n1.stc.Pressure]
T_out = sol_dict[n1.stc.Temperature]
mach = sol_dict[n1.oth.Mach]

print(f"Outlet pressure: {p_out / 1e5:.2f} bar")
print(f"Outlet Mach: {mach:.3f}")
```

## Building Multi-Stage Systems

For systems with multiple stages, add each component and define the connection sequence:

```python
network = ComponentNetwork()

# Create components
inlet = Inlet(node_idx=0)
stage1 = BladeRow(node_inlet=1, node_outlet=2, is_turbine=False)
diffuser = SketchVanelessDiff(node_inlet=3, node_outlet=4)
stage2 = BladeRow(node_inlet=5, node_outlet=6, is_turbine=False)

# Add to network
network.add_component(inlet)
network.add_component(stage1)
network.add_component(diffuser)
network.add_component(stage2)

# Define connections (flow path)
network.connect(inlet, stage1)      # Inlet outlet → stage1 inlet
network.connect(stage1, diffuser)   # Stage1 outlet → diffuser inlet
network.connect(diffuser, stage2)   # Diffuser outlet → stage2 inlet
```

This automatically handles:

- Flow continuity between components
- State variable transfers (pressure, temperature, angles)
- Area and geometry continuity

## Component Configuration

Each component has configuration options. Use them to specify physics and geometry.

### BladeRow Configuration

```python
from adet.components.blade_row import BladeRow

stage = BladeRow(
    node_inlet=1,
    node_outlet=2,
    name='hpc_stage_2',
    is_turbine=False,              # Compressor (True for turbine)
    loss_model='basic',            # Loss correlation to use
    include_shock=True,            # Include shock losses
    blade_count=50,                # Number of blades
    chord_distribution='linear',   # Blade geometry type
)
```

**Common parameters:**

- **is_turbine** — Set to `True` for turbine, `False` for compressor
- **loss_model** — Which loss model to use ('basic', 'profile', etc.)
- **include_shock** — Add oblique shock equations for high-speed flows
- **blade_count** — Number of blades (affects blockage)

## Accessing Component Results

After solving, extract results by component:

```python
# Get inlet state
n0 = NodeVariables(0)
p_inlet = sol_dict[n0.stc.Pressure]
T_inlet = sol_dict[n0.stc.Temperature]

# Get stage outlet
n2 = NodeVariables(2)
p_out = sol_dict[n2.stc.Pressure]
T_out = sol_dict[n2.stc.Temperature]

# Compute stage pressure ratio
pr = p_out / p_inlet
print(f"Stage pressure ratio: {pr:.2f}")
```

**Node numbering:** Each component specifies which nodes it occupies. Check the component documentation to find the correct node indices for your system.

## Debugging Network Systems

Network systems are more complex than direct equation systems. Use these strategies if solving fails:

**Visualize the network:**

```python
# Print component connections
for comp in network.components:
    print(f"{comp.name}: nodes {comp.nodes}")
```

**Check generated equations:**

After `build()`, the system logs which equations it added:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

system.build()  # Will print equation details
```

**Reduce complexity:**

Start with simpler components or fewer stages:

```python
# First: Just inlet + one stage
network_simple = ComponentNetwork()
network_simple.add_component(inlet)
network_simple.add_component(stage1)

# Solve and verify
# Then: Add diffuser
# Then: Add more stages
```

**Use intermediate solutions:**

Solve a simpler system, then use its solution as a guess for the full system:

```python
# Solve simple system
system_simple = CasadiSystem()
system_simple.from_network(network_simple)
# ... boundary conditions, solve ...
sol_simple = solve_root_problem(...)

# Use as guess for full system
# (Requires careful variable mapping)
```

## Switching Between APIs

You can mix the APIs for flexibility:

```python
# Start with network
network = ComponentNetwork()
network.add_component(inlet)
network.add_component(stage)

# Build with network
system = CasadiSystem()
system.from_network(network)

# Add custom equations directly
system.add_equation(MyCustomEquation(), 0)

# Continue as normal
system.build()
```

This lets you use components for standard physics and add custom equations for specialized behavior.

## Complete Multi-Stage Example

Here's a complete example of a simple two-stage compressor:

```python
from adet.assembly import CasadiSystem
from adet.components.inlet import Inlet
from adet.components.blade_row import BladeRow
from adet.components.network import ComponentNetwork
from adet.variables import NodeVariables, ThermoVariables
from adet.fluid.settings import AnalyticalFluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.solution import solve_root_problem
from pint import Quantity
import numpy as np

# Setup fluid
model = AnalyticalFluidModel(IdealGasState(1.4, 287))
thrm = ThermoVariables()
fluid_settings = FluidSettings(model, (thrm.Pressure, thrm.Temperature))

# Create components
inlet = Inlet(node_idx=0, name='inlet')
stage1 = BladeRow(
    node_inlet=1,
    node_outlet=2,
    name='stage_1',
    is_turbine=False,
)
stage2 = BladeRow(
    node_inlet=3,
    node_outlet=4,
    name='stage_2',
    is_turbine=False,
)

# Build network
network = ComponentNetwork()
network.add_component(inlet)
network.add_component(stage1)
network.add_component(stage2)

network.connect(inlet, stage1)
network.connect(stage1, stage2)

# Build system
system = CasadiSystem()
system.from_network(network)
system.fluid_settings = fluid_settings

# Boundary conditions
n0 = NodeVariables(0)
BC = {
    n0.tot.Pressure: Quantity(101325, 'Pa'),
    n0.tot.Temperature: Quantity(288, 'K'),
    n0.kin.FlowAngleAbs: Quantity(0, 'deg'),
    n0.oth.MassFlow: 1.5,
    n0.geo.RDistr: 0.05,
    n0.geo.HDistr: 0.01,
}

system.add_boundary_conditions(BC)
system.build()

# Solve
rtfn = system.make_rootfinder('kinsol')
x0 = system.get_scaled_guess()
knowns = system.get_scaled_constraints()

sol = solve_root_problem(rtfn, x0, knowns)
sol_dict = system.sol_to_dict(sol)

# Results
n4 = NodeVariables(4)  # Stage 2 outlet
p_out = sol_dict[n4.stc.Pressure]
T_out = sol_dict[n4.stc.Temperature]
pr_total = p_out / sol_dict[n0.tot.Pressure]

print(f"Overall pressure ratio: {pr_total:.2f}")
print(f"Final temperature: {T_out:.1f} K")
```

## Next Steps

