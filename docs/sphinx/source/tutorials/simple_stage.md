# Design a Repeated Turbine Stage

In this tutorial, we'll design and analyze a two-stage turbine (stator-rotor pair) using ADeT's high-level component-based API. We'll create blade rows, set design specifications, and solve the complete system.

## What You'll Learn

- How to create turbomachinery components (inlet, blade rows)
- How to build a `ComponentNetwork` with multiple stages
- How to set up system equations and design constraints
- How to specify spanwise distributions
- How to solve and visualize triangles and preliminary blade geometry

## Designing a Simple Turbine Stage

We'll solve a nondimensional design problem by specifying the flow coefficient, work coefficient, degree of reaction.

### Step 1: Imports and Setup

Start by importing the necessary components and utilities:

```python
import logging
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from CoolProp import AbstractState
from pint import Quantity

from adet.assemblers import CasadiSystem
from adet.components import BladeRow, Inlet
from adet.components.blade_row import RowGeometry
from adet.components.connections import Shaft
from adet.components.network import ComponentNetwork
from adet.equations.definitions import RepeatedStage
from adet.equations.fundamental import FreeVortexDistribution
from adet.equations.geometrical import ModifiedZweifel
from adet.equations.nondimensional import (
    FlowCoefficientMid,
    StaticTotalDegreeOfReaction,
    TotalTotalExpansionEfficiency,
    WorkCoefficientMid,
)
from adet.fluid.settings import FluidSettings
from adet.losses.basic import (
    TotalPressureLoss,
    ZeroDeviation,
)
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger
from adet.tools.plotting import plot_camberline, plot_velocity_triangles, setup_mpl
from adet.variables import NodeVariables
```

Create node variables for each station in the flow. 
```{figure} ../../images/stage_layout.svg
:align: center
:width: 400px
Schematic of the node layout for the stage. 0s, 1s, 0r, 1r, are the relative node indices of stator and rotor respectively.
```

We define the following `NodeVariables` objects, they will allow us to define boundary conditions at the desired position.

```python
n0 = NodeVariables(0)  # Inlet
n1 = NodeVariables(1)  # Stator outlet 
n2 = NodeVariables(2)  # Rotor inlet
n3 = NodeVariables(3)  # Rotor outlet
```

```{Note}
Nodes 1 and 2 are conceptually the *same physical point* in the flow. This respects the rule that each component uniquely owns its node pair. Within the network, information is passed from `n1` to `n2` through automatically defined equations, to ensure consistency.
```


### Step 2: Initialize Fluid State

Set up the thermodynamic model. 

```python
abs_state = AbstractState('HEOS', 'Air')

fluid_settings = FluidSettings(
    fluid_state=abs_state,
    update_variables=(n0.stc.Pressure, n0.stc.Temperature),
)
```

### Step 3: Create the Inlet

Define boundary conditions at the inlet (total pressure, temperature, geometry):

```python
inlet = Inlet(
    boundary_conditions={
        n0.oth.TotMassFlow: 10.0, # Total massflow
        n0.geo.Rmid: 0.1, # Midspan radius
        n0.geo.HubTipRatio: 0.65,
        n0.geo.MeridionalAngle: Quantity(0, 'deg'),
        n0.tot.Pressure: 10e5, # Total pressure
        n0.tot.Temperature: 500, # Total temperature
    }
)
```

The `MeridionalAngle` $\varphi$ is defined in the figure below, and it is null across the stage (purely axial).


```{figure} ../../images/meridional_angle.svg
:align: center
:width: 300px
Definition of the meridional angle $\varphi$ and channel height $H$.
```

```{Hint}
All properties on node 0 can be defined either on the inlet or on the first component.
The `Inlet` object is useful for separating properties that we do not want to copy between `BladeRow` objects.
```

### Step 4: Create Shafts and Blade Rows

Create a stationary casing shaft and a rotating shaft. The rotational speed of the rotating shaft is a free design parameter. This is specified by using the `is_constrained` argument.

```python
casing = Shaft(0, is_constrained=True) # Fixed (null) rotational speed
shaft = Shaft(-1, is_constrained=False) # Free rotational speed
```

The (relative) total pressure loss coefficient $Y$, defined below, is set to 0.9.

$$Y = \frac{p_{t,in}^{r} - p_{t,out}^{r}}{p_{t,in}^{r} - p_{in}}$$

$p_{t}^{r}$ is relative total pressure and $p$ the static pressure. 

Define the stator blade row:

```python
stator = BladeRow(
    name='stator',
    shaft=casing,
    bound_cond={
        n1.geo.MeridionalAngle: Quantity(0, 'deg'),
        n1.geo.AspectRatio: 3.0, # Height / axial chord
        n1.geo.NumBlades: 40,  # Actual number of blades
        n1.geo.ZweifelCoeff: 0.9,  # Theoretical optimal num blades
    },
    extra_equations={
        ZeroDeviation(): 0,  # Geometric angle = Flow angle (No incidence)
        ZeroDeviation(): 1,  # Geometric angle = Flow angle (No deviation)
        TotalPressureLoss(0.9): (0, 1),  # Loss coefficient Y = 0.9
        ModifiedZweifel(): (0, 1),  # Zweifel criterion for optimal blades
    },
    constant_variables=[n0.geo.Rmid], # Constant midspan radius
    spanwise_constants=[n1.geo.ChordAx], # Constant axial chord (no taper)
)
```

`constant_variables` specified which variables are constant bewteen inlet and outlet. In this case we can use both `n0` and `n1` to specify the condition $r_{mid,0} = r_{mid,1}$.

`spanwise_constants` specifies which variables are to be kept constant along the span. In this case the node index is important, as the axial chord is stored by convention on the outlet node. 

```{Hint}
If you are wondering whether a variable is assigned by convention to the inlet or outlet node, you can use your LSP *Go to References* function on the VarSpec of that variable and see its uses across the codebase.
```

Create the rotor by copying the stator. The copy operation carries over all the boundary condtions, constant variables, spanwise constants and extra equeations. All of these conditions can be manipulated using the `BladeRow` API. In this case we just want to assign the rotor to the rotating shaft instead of the casing.

```python
rotor = deepcopy(stator)
rotor.shaft = shaft
rotor.name = 'rotor'
```

### Step 5: Set Spanwise Distributions

Specify spanwise-constant variables at the inlet of the stage:

```python
stator.set_spanwise_constant(
    n0.kin.V_mer,
    n0.geo.HDistr,
)
```

The $\Delta H$ (`HDistr`) condition specifies the inlet as a uniform distribution of equal streamtubes. The condition on $V_m$ (`V_mer`) implies an inlet with uniform meridional velocity.

```{Warning}
Specifying the spanwise constants before creating the rotor will overconstrain the problem. When copying the stator to create the rotor, these conditions would also transfer to the rotor inlet.
```

### Step 6: Build the Component Network

Assemble all components into a network of components. A `ComponentNetwork` is a convenience class that assembles your components into a system of equations (specified by `backend`). To learn how to interact with the system directly check out [](annulus_system.md)

```python
ntw = ComponentNetwork(
    fluid_settings=fluid_settings,
    inlet=inlet,
    backend=CasadiSystem(num_span=1), 
    components=[stator, rotor],
)
```

### Step 7: Cross-component Equations

We also want to add cross-component equations that define nondimensional coefficients and repeated stage conditions. When multiple spanwise stations are used, we specify the coefficients at the midspan location:

$$
\begin{gather}
    \phi_{mid,3} = V_{m, 0} / U_{m,3} \\
    \psi_{mid,3} = (h_{t,3} - h_{t,0}) / U_{m,3}^2 \\
    R_3 = | \Delta h_{ROT} / \Delta h_t | \\
    V_m = \mathrm{const}  \qquad  \alpha_0 = \alpha_3 \\
    \eta_{tt,3} = \frac{h_{t,0} - h_{t,3}}{h_{t,0} - h_{t,3ss}}
\end{gather}
$$

These are added accessing directly the system's API:

```python
ntw.system.add_equation(FlowCoefficientMid(), (0, 3))
ntw.system.add_equation(WorkCoefficientMid(), (0, 3))
ntw.system.add_equation(RepeatedStage(), (0, 1, 2, 3))
ntw.system.add_equation(StaticTotalDegreeOfReaction(), (0, 1, 2, 3))
ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, 3)) # compute eta_tt
```

```{Hint}
By convention the value of coefficients or properties which do not strictly belong to a position in the flow are assigned to the **largest node**.
```

### Step 8: Set Design Constraints

Now that their equations have been added, we can specify the nondimensional design parameters as boundary conditions. 

There are two equivalent ways to specify these constraints, which showcase the mapping between absolute and relative indices. The outlet relative index of the rotor is 1, but the absolute index we are acting on is index 3. 

::::{tab-set}
:::{tab-item} Component API
:sync: component-api

Use the `BladeRow` API to set boundary conditions on the component directly. This uses relative node indices:

```python
rotor.set_bc_from_dict(
    {
        n1.ndim.FlowCoeffMid: 0.4,
        n1.ndim.WorkCoeffMid: -1.1,  # Negative = Work extraction
        n1.ndim.DegreeOfReactionTS: 0.6,
    }
)
```
:::
:::{tab-item} System API
:sync: system-api

Alternatively, use the system's API directly with absolute node indices:

```python
ntw.system.add_boundary_conditions(
    {
        n3.ndim.FlowCoeffMid: 0.4,
        n3.ndim.WorkCoeffMid: -1.1,  # Negative = Work extraction
        n3.ndim.DegreeOfReactionTS: 0.6,
    },
)
```
:::
::::


### Step 9: Build and Solve

Compile the system. Get the initial guess, using 0.8 as a fallback values for quantities with no guess. We fetch variable bounds, using custom values to avoid negative chords and enforcing reasonable thermodynamic and velocity magnitude bounds.

```{Hint}
If the bound is specified with the `Glob` attribute it is applied to all variable of that type (of all thermodynamic states). Otherwise it is applied only on the specified node.
```

```python
ntw.build()

x0 = ntw.system.get_guess(fallback=0.8)
kn = ntw.system.get_boundary_conds()
bnd = ntw.system.get_bounds(
    {
        n0.geo.Chord.Glob: (0.0, 1e5),
        n0.kin.V_mag.Glob: (0.0, 500.0),
        n0.stc.Pressure.Glob: (10.0, 13e5),
        n0.stc.Temperature.Glob: (60.0, 500),
    },
    ignore_defaults=False,
)
```

`ipopt` is first used for an unbounded solve attempt then bounds are enforced if the first approach fails. 

`kinsol` is then used to refine the root or escape unfeasible solutions which `ipopt` might have converged to. If all these approaches fail either the problem admits no solution, or a better initial guess and/or bounds are needed.


```python 
try:
    # Unbounded solve with IPOPT
    rtfn = ntw.system.make_rootfinder('ipopt', {'error_on_fail': True, 'ipopt.max_wall_time': 5})
    sol = solve_root_problem(rtfn, x0, kn, suppress_output=True)
except RuntimeError:
    # Bounded solve if unbounded fails
    rtfn = ntw.system.make_rootfinder('ipopt', {'error_on_fail': False})
    sol = solve_root_problem(rtfn, x0, kn, bnd, suppress_output=False)

# Refine with Kinsol
rtfn = ntw.system.make_rootfinder('kinsol')
sol = solve_root_problem(rtfn, sol, kn)

data = ntw.system.sol_to_dict(sol)
```

```{Note}
:class: dropdown
`kinsol` also fails upon finding rank-deficient Jacobians, which `ipopt` still solves without warning. A system where the number of variables and equations is equal can still have a rank deficient Jacobian. 

This is usually a symptom of an ill-defined problem (e.g., redundant equation definitions with missing boundary conditions). 
```

### Step 10: Visualize Results

Plot the velocity triangles.

```python
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for ax, node in zip(axs.flatten(), [n0, n1, n2, n3]):
    ax.set_aspect('equal')
    plot_velocity_triangles(
        data[node.kin.V_tan],
        data[node.kin.V_mer],
        data[node.kin.BladeSpeed],
        data[node.geo.RDistr],
        ax,
    )

plt.tight_layout()
plt.show()
```
For more complete plotting examples including camber lines and additional visualizations, see `src/adet/tutorials/axial_turbine.py`.

## Full Example

Here's the complete working example (also found in `src/adet/tutorials/axial_turbine.py`):

```python
from copy import deepcopy

import matplotlib.pyplot as plt
from CoolProp import AbstractState
from pint import Quantity

from adet.assemblers import CasadiSystem
from adet.components import BladeRow, Inlet
from adet.components.connections import Shaft
from adet.components.network import ComponentNetwork
from adet.equations.definitions import RepeatedStage
from adet.equations.geometrical import ModifiedZweifel
from adet.equations.fundamental import FreeVortexDistribution
from adet.equations.nondimensional import (
    FlowCoefficientMid,
    StaticTotalDegreeOfReaction,
    WorkCoefficientMid,
    TotalTotalExpansionEfficiency,
)
from adet.fluid.settings import FluidSettings
from adet.losses.basic import TotalPressureLoss, ZeroDeviation
from adet.solution import solve_root_problem
from adet.tools.plotting import plot_velocity_triangles
from adet.variables import NodeVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)
n2 = NodeVariables(2)
n3 = NodeVariables(3)

abs_state = AbstractState('HEOS', 'Air')
fluid_settings = FluidSettings(
    fluid_state=abs_state,
    update_variables=(n0.stc.Pressure, n0.stc.Temperature),
)

inlet = Inlet(
    boundary_conditions={
        n0.oth.TotMassFlow: 10.0,
        n0.geo.Rmid: 0.1,
        n0.geo.HubTipRatio: 0.65,
        n0.geo.MeridionalAngle: Quantity(0, 'deg'),
        n0.tot.Pressure: 10e5,
        n0.tot.Temperature: 500,
    }
)

casing = Shaft(0, is_constrained=True)
shaft = Shaft(-1, is_constrained=False)

stator = BladeRow(
    name='stator',
    shaft=casing,
    bound_cond={
        n1.geo.MeridionalAngle: Quantity(0, 'deg'),
        n1.geo.AspectRatio: 3.0,
        n1.geo.NumBlades: 40,
        n1.geo.ZweifelCoeff: 0.9,
    },
    extra_equations={
        ZeroDeviation(): 0,
        ZeroDeviation(): 1,
        TotalPressureLoss(0.9): (0, 1),
        ModifiedZweifel(): (0, 1),
    },
    constant_variables=[n0.geo.Rmid],
    spanwise_constants=[n1.geo.ChordAx],
)

rotor = deepcopy(stator)
rotor.shaft = shaft
rotor.name = 'rotor'

ntw = ComponentNetwork(
    fluid_settings=fluid_settings,
    inlet=inlet,
    backend=CasadiSystem(num_span=1),
    components=[stator, rotor],
)

stator.set_spanwise_constant(
    n0.kin.V_mer,
    n0.geo.HDistr,
)

ntw.system.add_equation(FlowCoefficientMid(), (0, 3))
ntw.system.add_equation(WorkCoefficientMid(), (0, 3))
ntw.system.add_equation(RepeatedStage(), (0, 1, 2, 3))
ntw.system.add_equation(StaticTotalDegreeOfReaction(), (0, 1, 2, 3))
ntw.system.add_equation(TotalTotalExpansionEfficiency(), (0, 3))


rotor.set_bc_from_dict(
    {
        n1.ndim.FlowCoeffMid: 0.4,
        n1.ndim.WorkCoeffMid: -1.1,
        n1.ndim.DegreeOfReactionTS: 0.6,
    }
)

ntw.build()

x0 = ntw.system.get_guess(fallback=0.8)
kn = ntw.system.get_boundary_conds()
bnd = ntw.system.get_bounds(
    {
        n0.geo.Chord.Glob: (0.0, 1e5),
        n0.kin.V_mag.Glob: (0.0, 500.0),
        n0.stc.Pressure.Glob: (10.0, 13e5),
        n0.stc.Temperature.Glob: (60.0, 500),
    },
    ignore_defaults=False,
)

try:
    rtfn = ntw.system.make_rootfinder('ipopt', {'error_on_fail': True, 'ipopt.max_wall_time': 5})
    sol = solve_root_problem(rtfn, x0, kn, suppress_output=True)
except RuntimeError:
    rtfn = ntw.system.make_rootfinder('ipopt', {'error_on_fail': False})
    sol = solve_root_problem(rtfn, x0, kn, bnd, suppress_output=False)

rtfn = ntw.system.make_rootfinder('kinsol')
sol = solve_root_problem(rtfn, sol, kn)

data = ntw.system.sol_to_dict(sol)
```

## Multiple Spanwise Stations

```{figure} ../../images/flow_station.svg
:align: center
:width: 300px
Definition of the meridional geometry on a node. $r_i$ (`RDistr`) and $b_i$ (`HDistr`) are streamtube radii and heights distributions respectively. $\varphi$ is the meridional angle.
```

When a single spanwise station is used, a single average streamtube covers the whole channel. Therefore $b_0 = H$ and $r_0 = r_{mid}$.

To expand the design to multiple spanwise stations (it must be an **odd** number), you can modify the backend initialization:

```python
ntw = ComponentNetwork(
    fluid_settings=fluid_settings,
    inlet=inlet,
    backend=CasadiSystem(num_span=3),  # Use 3 spanwise stations
    components=[stator, rotor],
)
```

When `num_span > 1`, the system solves for radial equilibrium conditions at the stator and rotor outlets. To design the blade angle along the span, use the `FreeVortexDistribution` equation on the outlet of both blade rows.

```python
if ntw.system.num_span > 1:
    stator.add_equation(FreeVortexDistribution(), 1)
    rotor.add_equation(FreeVortexDistribution(), 1)
```

The `FreeVortexDistribution` equation enforces $rV_{\theta}(r)= \mathrm{const}$ across the span. By respecting radiual equilibrium this also results in constant spanwise meridional velocity $V_m(r)$.

This example is also extended into a modular repeated-stage setup in `src/adet/examples/modular_turbine.py`
