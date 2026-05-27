# Fluid Models and Equations of State

ADeT supports multiple thermodynamic models ranging from ideal gas to accurate real-gas equations of state via REFPROP. This tutorial covers how to select and configure fluid models for your applications.

## Overview of Thermodynamic Models

ADeT provides three levels of thermodynamic accuracy:

1. **Ideal Gas** — Fast, analytical derivatives, suitable for compressible flow at moderate conditions
2. **Real Gas (Analytical)** — Empirical correlations for common gases without external dependencies
3. **Real Gas (REFPROP)** — Highly accurate, covers all thermodynamic regions, requires external REFPROP installation

### Choosing a Model

Use **Ideal Gas** when:
- Speed is critical (interactive design loops)
- Gases are at moderate to high temperatures (~300 K+)
- Accuracies within 5-10% are acceptable
- You need fully analytical derivatives

Use **Real Gas (Analytical)** when:
- Better accuracy is needed (~2-5% error)
- You're working with air or simple diatomic gases
- External dependencies must be minimized

Use **Real Gas (REFPROP)** when:
- High accuracy is required (~0.1% error)
- Working near phase boundaries or critical points
- Dealing with mixtures or exotic fluids
- You can afford the computational cost

## Ideal Gas Model

The ideal gas model is the fastest and simplest:

```python
from adet.fluid.settings import AnalyticalFluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.variables import ThermoVariables

# Create ideal gas EOS
# Parameters: gamma (heat capacity ratio), R (gas constant), viscosity_ref
eos = IdealGasState(gamma=1.4, R=287, mu_ref=2e-5)

model = AnalyticalFluidModel(eos)
thrm = ThermoVariables()

fluid_settings = FluidSettings(
    model=model,
    update_from=(thrm.Pressure, thrm.Temperature),
)

system.fluid_settings = fluid_settings
```

**Parameters:**

- **gamma** ($\gamma$) — Heat capacity ratio ($C_p/C_v$), typically 1.4 for diatomic gases (air, N₂, O₂)
- **R** — Specific gas constant in J/(kg·K), typically 287 for air
- **mu_ref** — Reference viscosity at reference temperature (optional), used for Sutherland correlations

**Properties computed:**

- Density from ideal gas law: $\rho = \frac{p}{R \cdot T}$
- Enthalpy: $h = C_p \cdot T$ where $C_p = \frac{\gamma \cdot R}{\gamma - 1}$
- Entropy: $s = C_p \ln(T) - R \ln(p) + s_0$
- Speed of sound: $a = \sqrt{\gamma \cdot R \cdot T}$

## Real Gas Models

For better accuracy, use real gas models. ADeT interfaces with CoolProp, which can use multiple backends.

### AnalyticalFluidModel (Fast EOS)

The analytical fluid model uses fast correlations (e.g., NIST correlations for air):

```python
from adet.fluid.settings import AnalyticalFluidModel, FluidSettings
from adet.fluid.symbolic_eos import NistAirState  # Fast air correlations

eos = NistAirState()
model = AnalyticalFluidModel(eos)

fluid_settings = FluidSettings(model, (thrm.Pressure, thrm.Temperature))
system.fluid_settings = fluid_settings
```

### ExternalFluidModel (REFPROP via CoolProp)

For highest accuracy, use REFPROP through CoolProp:

```python
from adet.fluid.settings import ExternalFluidModel, FluidSettings
from adet.tools.coolprop_utils import DebugAbstractState
from adet.variables import ThermoVariables

# REFPROP backend for high accuracy
abstract_state = DebugAbstractState('REFPROP', 'MM')  # Monomethylamine

model = ExternalFluidModel(abstract_state)
thrm = ThermoVariables()

fluid_settings = FluidSettings(model, (thrm.Pressure, thrm.Temperature))
system.fluid_settings = fluid_settings
```

**Available fluids via REFPROP:**

- `'Air'` — Natural air composition
- `'N2'`, `'O2'` — Pure gases
- `'CO2'`, `'H2O'` — Other substances
- And many more (hundreds of fluids supported)

**Note:** REFPROP requires a separate installation. See the README for setup instructions.

### Alternative: CoolProp HEOS Backend

If REFPROP is unavailable, CoolProp can use faster empirical fits (HEOS):

```python
abstract_state = DebugAbstractState('HEOS', 'Air')
```

This is faster than REFPROP but less accurate (~1-3% error).

## Configuring Update Variables

The `FluidSettings` specifies which variables the EOS computes from other variables. This is crucial for equation formulation.

### Common Configurations

**From pressure and temperature:**

```python
thrm = ThermoVariables()
fluid_settings = FluidSettings(
    model,
    update_from=(thrm.Pressure, thrm.Temperature),
)
```

This tells ADeT: "Given pressure and temperature, compute other properties."

**From enthalpy and entropy:**

```python
fluid_settings = FluidSettings(
    model,
    update_from=(thrm.Enthalpy, thrm.Entropy),
)
```

This is useful for isentropic or shock relations where you know $h$ and $s$ but not $p$ and $T$.

**From density and temperature:**

```python
fluid_settings = FluidSettings(
    model,
    update_from=(thrm.Density, thrm.Temperature),
)
```

### What Gets Computed?

Once you specify `update_from`, the EOS automatically provides:

- Thermodynamic properties (pressure, temperature, density, enthalpy, entropy)
- Transport properties (viscosity, thermal conductivity)
- Derived properties (speed of sound, specific heats)

Your equations can request any of these from the EOS:

```python
from adet.equations.base_equation import EquationBase, EquationConfig
import CoolProp as cp

class MyEquation(EquationBase):
    config = EquationConfig(
        input_pair=cp.HmassSmass_INPUTS,        # Compute from h, s
        out_properties=(thrm.Pressure, thrm.Density),  # Outputs
    )

    def residual(self, h, s, p, rho):
        p_eos, rho_eos = self.eos(h, s)  # Call the EOS
        return p - p_eos, rho - rho_eos
```

## Working with Multiple Fluids

ADeT can handle systems with different fluids at different locations (e.g., different stages in a multi-stage compressor).

### Single Fluid (Recommended for Beginners)

Most systems use one fluid throughout:

```python
model = AnalyticalFluidModel(IdealGasState(1.4, 287))
fluid_settings = FluidSettings(model, (thrm.Pressure, thrm.Temperature))
system.fluid_settings = fluid_settings  # Applied to entire system
```

## Example: Complete System with Real Gas

Here's a complete example using a real gas model:

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

# Use ideal gas for speed, but could use ExternalFluidModel for accuracy
model = AnalyticalFluidModel(IdealGasState(1.4, 287, 2e-5))
thrm = ThermoVariables()
fluid_settings = FluidSettings(model, (thrm.Pressure, thrm.Temperature))
system.fluid_settings = fluid_settings

n0 = NodeVariables(0)

# Add equations
equations = [
    TotalStaticMatching(),
    AnnulusAreas(),
    MassAreaRelation(),
    AbsoluteMachNumber(),
    ZeroBlockage(),
    Kinematics(),
    ThermoVarsAdder(),
    GammaIdeal(),
]

for eq in equations:
    system.add_equation(eq, 0)

# Set boundary conditions
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

# Solve
rtfn = system.make_rootfinder('kinsol')
x0 = system.get_scaled_guess()
knowns = system.get_scaled_constraints()

sol = solve_root_problem(rtfn, x0, knowns)
sol_dict = system.sol_to_dict(sol)

# Extract results
p_stat = sol_dict[n0.stc.Pressure]
T_stat = sol_dict[n0.stc.Temperature]
rho_stat = sol_dict[n0.stc.Density]
V = sol_dict[n0.kin.V_abs]

print(f"Static pressure: {p_stat / 1e5:.2f} bar")
print(f"Static temperature: {T_stat:.1f} K")
print(f"Density: {rho_stat:.3f} kg/m³")
print(f"Velocity: {V:.1f} m/s")
```

## Debugging Fluid Model Issues

If your solver fails and you suspect the fluid model, try these steps:

1. **Switch to ideal gas** to isolate whether the issue is the EOS or the equations:

```python
model = AnalyticalFluidModel(IdealGasState(1.4, 287))
```

2. **Check if properties are computed**:

```python
sol_dict = system.sol_to_dict(sol)

# Verify secondary properties are present
if n0.stc.Density in sol_dict:
    print("Density computed OK")
else:
    print("Density NOT in solution!")
```

3. **Validate boundary conditions** are physically realizable:

```python
# For air at 18.1 bar, 573 K
# Check if this is reasonable with your chosen EOS
import CoolProp as cp
AS = cp.AbstractState('HEOS', 'Air')
AS.update(cp.PT_INPUTS, 18.1e5, 573.15)
print(f"CoolProp density: {AS.rhomass():.3f} kg/m³")
```

## Comparing Models

To compare accuracy across models for your application:

```python
models_to_test = [
    ('Ideal Gas', AnalyticalFluidModel(IdealGasState(1.4, 287))),
    # ('HEOS', ExternalFluidModel(DebugAbstractState('HEOS', 'Air'))),
    # ('REFPROP', ExternalFluidModel(DebugAbstractState('REFPROP', 'Air'))),
]

for name, model in models_to_test:
    print(f"\nTesting {name}...")
    system = CasadiSystem()
    # ... configure system with this model ...
    # ... solve and extract results ...
```

## Next Steps

- [Custom Equations](05_custom_equations.md) — Extend ADeT with custom equations for specialized physics
- Back to [Solving Systems](03_solving_systems.md) — Strategies for challenging problems
