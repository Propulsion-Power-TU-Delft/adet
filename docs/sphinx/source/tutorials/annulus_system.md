# Solving an Annulus Flow

In this tutorial, we'll create a minimal but complete example of using ADeT's equation-based modeling system. We'll define a simple thermodynamic system, add equations, specify boundary conditions, and solve for the unknowns.

## What You'll Learn

- How to create a `CasadiSystem` instance with multiple spanwise stations
- How to access and define variables using `NodeVariables`
- How to configure the fluid model and update variables
- How to add equations to the system
- How to set boundary conditions and spanwise constants
- How to build and solve the system


## Computing the Flow Across an Annulus

Let's create a system that represents a fundamental set of equations for turbomachinery modeling: the average flow on a **circular** annulus. We will refer to the position of this annulus section as *node 0*. 

- We will use both a stationary and rotating frame with rotational speed $\Omega$. 
- The total quantities at this station $p_t$ and $T_t$ are chosen by the user
- The annulus is defined by its height $H$, its midpoint radius $r_{mid}$ and its angle w.r.t. the vertical $\varphi$ (meridional angle)


```{figure} ../../images/basic_annulus.svg
:align: center
:width: 400px
Annulus problem definition. $\mathbf{V}$ is the velocity in the absolute frame of reference, $\mathbf{W}$ in the relative frame of reference, $\mathbf{U}$ is the peripheral velocity. 
```

```{figure} ../../images/mer_angle_annulus.svg
:width: 300px
:align: center
Definition of the meridional angle $\varphi$ of a generic annulus on the meridional plane $z-r$.
```


The full system of equations that defines this problem is reported below, $h, h_t, h_{t,r}$ are the static, total and relative total enthalpies.

$$
\begin{gather}
    \begin{cases}
        V_{\theta} - (W_{\theta} + U) = 0 \\
        W_m - V_m = 0 \\
        \alpha - \arctan(V_{\theta} / V_m) = 0 \\
        \beta - \arctan(W_{\theta} / W_m) = 0 \\
        U - \Omega r_{mid} = 0 \\ \\
        2 \pi r_{mid} H - A_{geo} = 0 \\
        \rho V_m A_{eff} - \dot{m} = 0 \\
        A_{eff} - A_{geo} = 0 \ \text{(No blockage)} \\ \\
        h_{t,r} - (h + W^2/2) = 0 \\
        h_t - (h + V^2/2) = 0  \\
    \end{cases}
\end{gather}
$$

```{Note}
:class: dropdown
$\varphi$ has no influence on the system of equations, and is only needed to uniquely define the annulus' position in the meridional $z-r$ plane. This is important when discretizing the annulus into multiple *sub-annuli*, or annular streamtubes, and is handled by the `MeridionalGeometry` equation.
```

### Step 1: Import and Setup

```python
import logging

from pint import Quantity

from adet.assemblers import CasadiSystem
from adet.equations.fundamental import (
    Kinematics,
    MassAreaRelation,
    TotalMassFlow,
    TotalStaticMatching,
    ZeroBlockage,
)
from adet.equations.geometrical import AnnulusAreas, MeridionalGeometry
from adet.equations.nondimensional import AbsoluteMachNumber
from adet.fluid.ideal_eos import IdealGasState
from adet.fluid.settings import FluidSettings
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables

logger = logging.getLogger(__name__)
setup_logger(logger)

# Create the system
system = CasadiSystem(num_span=1)
```

### Step 2: Configure the Fluid Model

Specify which thermodynamic model and update variables the system should use:

```python
# Fundamental entities
node0 = NodeVariables(0)

# *** Fluid model and settings
ideal_state = IdealGasState(1.4, 287, 2e-5)

fluid_settings = FluidSettings(
    fluid_state=ideal_state,
    update_variables=(node0.stc.Pressure, node0.stc.Temperature),
)

system.fluid_settings = fluid_settings
```

`NodeVariables` are convenience enums for accessing pre-defined variables (`VarSpec`). For example:

```python
>>> node4 = NodeVariables(4)
>>> node4.tot.Pressure # Total pressure
VarSpec: p, node=4, state=tot
>>> node4.kin.V_tan
VarSpec: Vt, node=4 # Tangential velocity
```

```{Tip}
The node index and state of the variables declared for the updates is irrelevant. The same variables are used for updates across all states (total, static, relative total) and nodes.
```

`FluidSettings` tells ADeT which thermodynamic variables should be used for thermodynamic state updates at each iteration. In this case, we're using pressure and temperature, while the rest of thermodynamic variables are extracted to be plugged into the residual.


### Step 3: Define the Equations

Add the equations that govern your system, these are already implemented in ADeT.
Equation blocks are already provided. To learn how to define equations see [](equations_and_variables.md).

```python
EQUATIONS = {
    AnnulusAreas(): 0,             # A_geo = 2 pi r H
    ZeroBlockage(): 0,             # No blockage (A_eff = A_geo)
    MassAreaRelation(): 0,         # m_dot = rho V A_eff
    AbsoluteMachNumber(): 0,       # Defines Mach number
    TotalStaticMatching(): 0,      # Matches total and static state
    Kinematics(): 0,               # Defines velocity triangles
    TotalMassFlow(): 0,            # Total massflow across all streamtubes
    MeridionalGeometry(): 0,       # Geometry of the annulus
}

for eq, pos in EQUATIONS.items():
    system.add_equation(eq, pos)
```


```{Warning}
The second parameter (`pos`) specifies the node position of the equation. **It does not represent the right hand side of the equation**. In this case all equations are referred to the same average flow in position 0.
```


### Step 4: Set Boundary Conditions

Specify the known values at your nodes. Add geometry and flow constraints:

```python
BC = {
    node0.kin.Omega: 1000.0,                       # Rotational speed, rad/s
    node0.kin.FlowAngleAbs: Quantity(30, 'deg'),   # Inlet absolute flow angle
    node0.oth.TotMassFlow: 100.0,                  # Total massflow across the annulus
    node0.geo.Rmid: 0.1,                           # Midpoint annulus radius
    node0.geo.Height: 0.1,                         # Annulus height
    node0.geo.MeridionalAngle: 0.0,                # Meridional angle
    node0.tot.Pressure: 18.1e5,                    # Total pressure, Pa
    node0.tot.Temperature: 573.15,                 # Total temperature, K
}

system.add_boundary_conditions(BC)
system.add_spanwise_constants(node0.kin.V_mer, node0.geo.HDistr)
```

Each boundary condition fixes a variable value, reducing the problem size. 

The spanwise constants define variables that remain constant across the spanwise direction. These will become relevant for multiple discreitazions of the annulus (`num_span > 1`), otherwise they do not affect the system.

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

### Step 6: Create a Root Finder

ADeT accesses optimizers and rootfinders through `CasADi`. The `'kinsol'` solver is a Newton-Krylov method:

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

The solver iterates until the residuals converge to approximaetly zero (or reaches other stopping conditions).

### Step 8: Extract Results

Convert the solution back to a dictionary of variables:

```python
sol_dict = system.sol_to_dict(sol)

# Access results
for var_spec, value in sol_dict.items():
    print(f"{var_spec}: {value}")
```

The `sol_dict` contains all variables (both solved and computed from the EOS), indexed by their `VarSpec` objects. Single variables can be accessed using the `NodeVariables` object:

```python
>>> sol_dict[node0.kin.V_mag] # velocity magnitude
array([25.15642531])
```

## Solving for multiple streamtubes

Let's now explain the the role of `add_spanwise_constants` and `MeridionalGeometry` that appeared in the example.

### Geometric model

If we were to divide the annulus into a series of annular streamtubes, by specifying for example
```python
system = CasadiSystem(num_span=5) # Must be an odd number!
```
We would obtain a geometric structure as the one shown below.

```{figure} ../../images/flow_station.svg
:align: center
:width: 300px
Definition of the meridional geometry on a node. $r_i$ (`RDistr`) and $b_i$ (`HDistr`) are streamtube midpoint radii and heights respectively. $\varphi$ is the meridional angle.
```
When a single spanwise station is used, a single average streamtube covers the whole channel. For multiple streamtubes the logic needs to match the inner and outer radius of each. 


This is handled by the `MeridionalGeometry` equations, that compute the vectors $\mathbf{r} = [r_0, \ r_1, \ \cdots \ , \ r_N]$, $\mathbf{b} = [b_0, \ b_1, \ \cdots \ , \ b_N]$, i.e., `RDistr` and `HDistr`.

::::{tab-set}

:::{tab-item} Single streamtube
$$
\begin{gather}
    \mathbf{b} = [ b_0 ] = [ H ] \\
    \mathbf{r} = [ r_0 ] = [ r_{mid} ]
\end{gather}
$$

:::

:::{tab-item} $N$ streamtubes
$$
\begin{gather}
    r_0 = r_{mid} - \frac H 2 \cos \varphi + b_0 / 2 \\
    r_i = \left( r_{i-1} + \frac{b_{i-1} + b_i}{2} \right) 
         \quad i = 1,...,N-1 \\
    \sum_{i=0}^{N-1} b_i = H 
\end{gather}
$$

:::

::::

The area distribution is then given by $A_{geo,i} = 2 \pi r_i b_i$.

This means that the geometry is now defined by $2N$ variables (excluding $\varphi$), but `MeridionalGeometry` only adds $N+1$ equations. To close the system we need to impose another $N-1$ conditions. For simplicity we will assume to use a uniform height distribution $b_i = \mathrm{const}$.

### Distributing the total massflow

Another problem is that the condition on the total massflow only imposes 

```{math}
    \sum_{i=0}^{N-1} \dot{m}_i = \dot{m}_{tot}
```
which does not give information on how to distribute the massflow across the different streamtubes. In practice this gives only 1 equation for $N$ streamtubes, which is well-posed only for the $N=1$ case. 

To solve this, we also need to add $N-1$ conditions, in this case we choose to set $V_{m,i} = \mathrm{const}$


:::{note}
:class: dropdown
If we had imposed a certain meridional velocity instead of a total massflow

```python
system.data.boun_cond.pop(node0.oth.TotMassFlow)
system.add_boundary_conditions({node0.kin.V_mer: 50})
system.add_spanwise_constants(node0.geo.HDistr)
```

The velocity would be set to the specified value on all streamtubes ($N$ conditions), removing the need of the condition on $V_m$
:::


## Full Example

Here's the complete working example, (also found in `src/adet/tutorials/annulus_flow.py`):

```python
import logging

from pint import Quantity

from adet.assemblers import CasadiSystem
from adet.equations.fundamental import (
    Kinematics,
    MassAreaRelation,
    TotalMassFlow,
    TotalStaticMatching,
    ZeroBlockage,
)
from adet.equations.geometrical import AnnulusAreas, MeridionalGeometry
from adet.equations.nondimensional import AbsoluteMachNumber
from adet.fluid.ideal_eos import IdealGasState
from adet.fluid.settings import FluidSettings
from adet.solution import solve_root_problem
from adet.tools.loggers import setup_logger
from adet.variables import NodeVariables

logger = logging.getLogger(__name__)
setup_logger(logger)

EQUATIONS = {
    AnnulusAreas(): 0,
    ZeroBlockage(): 0,
    MassAreaRelation(): 0,
    AbsoluteMachNumber(): 0,
    TotalStaticMatching(): 0,
    Kinematics(): 0,
    TotalMassFlow(): 0,
    MeridionalGeometry(): 0,
}

system = CasadiSystem(num_span=1)
node0 = NodeVariables(0)

ideal_state = IdealGasState(1.4, 287, 2e-5)
fluid_settings = FluidSettings(
    fluid_state=ideal_state,
    update_variables=(node0.stc.Pressure, node0.stc.Temperature),
)
system.fluid_settings = fluid_settings

for eq, pos in EQUATIONS.items():
    system.add_equation(eq, pos)

BC = {
    node0.kin.Omega: 1000.0,
    node0.kin.FlowAngleAbs: Quantity(30, 'deg'),
    node0.oth.TotMassFlow: 100.0,
    node0.geo.Rmid: 0.1,
    node0.geo.Height: 0.1,
    node0.geo.MeridionalAngle: 0.0,
    node0.tot.Pressure: 18.1e5,
    node0.tot.Temperature: 573.15,
}

system.add_boundary_conditions(BC)
system.add_spanwise_constants(node0.kin.V_mer, node0.geo.HDistr)

system.build()

rtfn = system.make_rootfinder('kinsol')

x0 = system.get_guess()
kn = system.get_boundary_conds()

sol = solve_root_problem(rtfn, x0, kn)

sol_dict = system.sol_to_dict(sol)
```

