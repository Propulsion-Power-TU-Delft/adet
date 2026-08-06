# Custom Equations and Variables

This tutorial covers the core abstractions in ADeT: how to define equations, specify variables with units and bounds, and assemble them into a solvable system. We'll learn by breaking down a concrete example.

## A Simple Example: Solving Equations

Let's walk through a concrete problem: solving two equations with three unknowns and one boundary condition. We'll break it down step by step.

### Step 1: Imports

Start by importing the core tools you'll need:

```python
from adet.assemblers import CasadiSystem
from adet.equations.base_equation import EquationBase
from adet.solution import solve_root_problem
from adet.varspec import VarSpec
from numpy import sin
```

### Step 2: Define Variables with VarSpec

Variables in ADeT are defined using `VarSpec`, which specifies:
- **Name**: Symbolic identifier for the variable
- **Units**: Pint-compatible unit string
- **Guess**: Initial guess for the solver
- **Bounds**: Physical bounds
- **Node**: Which node (index) this variable belongs to

```python
my_x = VarSpec('x', 'dimensionless', guess=1.0, bounds=(-1000, 1000), node=0)
my_y = VarSpec('y', 'dimensionless', guess=1.0, bounds=(-1000, 1000), node=0)
my_p = VarSpec('p', 'dimensionless', node=0)
```

The `node=0` parameter tells ADeT that all three variables belong to *node* 0. In practice, nodes are used as markers of different positions in the flow where we are computing those lumped parameters. Each node can contain just one instance of a certain a variable spec.

### Step 3: Define Two Equations
Let's solve this simple system of nonlinear equation by writing it in `ADeT`

$$
\begin{cases}
    3 x^2 - 2 x - p = 0 \\
    \sin(x) + y = 0
\end{cases}
$$

Equations are defined by subclassing `EquationBase` and implementing the `residual()` method. The method parameters use the `.Hint` attribute from each variable to link parameters to the variables:

```python
class MyParabola(EquationBase):
    def residual(
        self,
        x: my_x.Hint,
        y: my_y.Hint,
        p: my_p.Hint,
    ):
        residual_1 = 3 * x**2 - 2 * x - p # = 0
        residual_2 = sin(x) + y # = 0
        return residual_1, residual_2
```

The `.Hint` attribute tells ADeT to pass the value of that variable into this parameter. The residuals should evaluate to zero at the solution.

Within an equation the variables nodes should always be numbered increasing from 0, and they represent the **relative** position of the variables involved in the equation.

The `residual` method can either return a single residual or a `tuple` of residuals.

### Step 4: Create the System

Instantiate a `CasadiSystem` and add your equation:

```python
system = CasadiSystem()
system.add_equation(MyParabola(), 0)
```

The second argument `0` specifies the **absolute** node position in which we are adding the equation. 


### Step 5: Add Boundary Conditions

Specify which variables are known (boundary conditions):

```python
system.add_boundary_conditions({my_p: 5})
```

This tells the system that `my_p` ($p$) is fixed at 5. The remaining unknowns (`my_x`, `my_y`) will be solved for.


:::{important}
In this case absolute and relative position match, but if we were to instead add the equation in position 1, the *absolute* arguments of the equations would all be shifted to node 1. 
In that case you would need to specify the boundary condition on `my_p` on the appropriate node. You can use the `at_node` method to quickly shift the node at which you are specifying it.
```python
system.add_equation(MyParabola(), 1)
system.add_boundary_conditions({my_p.at_node(1): 5})
```
:::


### Step 6: Build and Solve

Compile the system and solve:

```python
system.build()

rootfinder = system.make_rootfinder('kinsol')
guess = system.get_guess()
knowns = system.get_boundary_conds()
solution = solve_root_problem(rootfinder, guess, knowns)
```

### Full Example

Here's the complete script (also found in `src/adet/tutorials/simple_parabola.py`):

```python
from adet.assemblers import CasadiSystem
from adet.equations.base_equation import EquationBase
from adet.solution import solve_root_problem
from adet.varspec import VarSpec
from numpy import sin

# Define variables
my_x = VarSpec('x', 'dimensionless', guess=1.0, bounds=(-1000, 1000), node=0)
my_y = VarSpec('y', 'dimensionless', guess=1.0, bounds=(-1000, 1000), node=0)
my_p = VarSpec('p', 'dimensionless', node=0)

# Define an equation
class MyParabola(EquationBase):
    def residual(
        self,
        x: my_x.Hint,
        y: my_y.Hint,
        p: my_p.Hint,
    ):
        residual_1 = 3 * x**2 - 2 * x - p
        residual_2 = sin(x) + y
        return residual_1, residual_2

# Set up and solve
system = CasadiSystem()
system.add_equation(MyParabola(), 0)
system.add_boundary_conditions({my_p: 5})
system.build()

rootfinder = system.make_rootfinder('kinsol')
guess = system.get_guess()
knowns = system.get_boundary_conds()
solution = solve_root_problem(rootfinder, guess, knowns)
```

