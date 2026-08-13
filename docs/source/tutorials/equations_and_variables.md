# Variables and Equations

This tutorial covers the core abstractions in ADeT: how to define equations, specify variables and assemble them into a solvable system. 

## A Simple Example: Solving Equations

Let's walk through a concrete problem: solving two equations with three variables: two unknowns and one boundary condition. We'll break it down step by step.

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
- **Guess**: Initial guess for the solver (optional)
- **Bounds**: Physical bounds (optional)
- **Node**: Which node (index) this variable belongs to (optional, defaults to -1)
- Whether the variable is always treated as scalar (in multi-dimensional problems)

```python
x_spec = VarSpec('x', 'dimensionless', guess=1.0, bounds=(-1000, 1000))
p_spec = VarSpec('p', 'dimensionless', node=0)
```
Let's create two versions of the `x_spec` at two different positions, or *nodes*.

```python
my_x0 = x_spec.at_node(0)
my_x1 = x_spec.at_node(1)
```
This method simply returns the same `VarSpec` with a different `node` attribute. The `node` attribute of the spec denotes the position at which we are assigning that variable. This will become useful when dealing with the same type of variable (e.g., pressure, velocity) on different positions of the flow.

### Step 3: Define Two Equations
Let's solve this simple system of nonlinear equations:

$$
\begin{cases}
    3 x_0^2 - 2 x_0 - p = 0 \\
    \sin(x_0) + x_1 = 0
\end{cases}
$$

Equations are defined by subclassing `EquationBase` and implementing the `residual()` method. The method parameters use the `.Hint` attribute from each `VarSpec` to link parameters to the variables:

```python
class MyParabola(EquationBase):
    def residual(
        self,
        x0: my_x0.Hint,
        x1: my_x1.Hint,
        p: my_p.Hint,
    ):
        residual_1 = 3 * x0**2 - 2 * x0 - p # = 0
        residual_2 = sin(x0) + x1 # = 0
        return residual_1, residual_2
```

Within an equation the variables' nodes should always be numbered increasing from 0, and they represent the **relative** position of the variables involved in the equation. 

```{Hint}
- The `residual` method can either return a single residual or a `tuple` of residuals.
- The names of the parameters (`x0`,`x1`,`p`) are irrelevant and can be chosen for readability
```

### Step 4: Create the System

Instantiate a `CasadiSystem` and add your equation:

```python
system = CasadiSystem(num_span=1)
system.add_equation(MyParabola(), (0, 1))
```

The $N$ = `num_span` argument indicates the dimension of all variables that are not explicitly set to `scalar=True` in their `VarSpec`. 

In this example, we could increase arbitrarily `num_span` and still have a well-posed problem. The solution would be made up of uniform vectors $x_0, x_1 \in \mathbb{R}^N$. 

The second argument of `add_equation` specifies the **absolute** position in which we are adding the equation. Since the equation involves two relative nodes ($0_r$, $1_r$), we are prescribing a relative to absolute argument mapping: 
```{math}
0_r \rightarrow 0_a \\
1_r \rightarrow 1_a
```

### Step 5: Add Boundary Conditions

Specify which variables are known (boundary conditions):

```python
system.add_boundary_conditions({my_p: 5})
```

This tells the system that `my_p` ($p$) is fixed at 5. The remaining unknowns (`my_x0` $x_0$, `my_x1` $x_1$) will be solved for.


:::{important}
In this example absolute and relative positions match. If we were instead to add the equation inverting the absolute and relative indices:

```python
system.add_equation(MyParabola(), (1, 0))
```
the argument mapping would become: 
```{math}
0_r \rightarrow 1_a \\
1_r \rightarrow 0_a
```
In that case you would need to specify the boundary condition on `my_p` on node 1 (absolute). You can use the `at_node` method to quickly shift the node at which you are specifying it.
```python
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
The solution is made up of $x_0$ and $x_1$ concatenated in a $\mathbb{R}^{2N}$ vector.

```python
>>> solution
array([[-0.99540786],
       [ 1.66666659]])
```

For readability we recommend converting the solution to a dictionary:

```python
sol_dict = system.sol_to_dict(solution)
```

Which returns the variables and boundary conditions gathered in a single dictionary:

```python
>>> sol_dict
{VarSpec: p, node=0: array([5]),
 VarSpec: x, node=0: array([1.66666659]),
 VarSpec: x, node=1: array([-0.99540786])}
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
x_spec = VarSpec('x', 'dimensionless', guess=1.0, bounds=(-1000, 1000))
p_spec = VarSpec('p', 'dimensionless', node=0)

my_x0 = x_spec.at_node(0)
my_x1 = x_spec.at_node(1)
my_p = p_spec

# Define an equation
class MyParabola(EquationBase):
    def residual(
        self,
        x0: my_x0.Hint,
        x1: my_x1.Hint,
        p: my_p.Hint,
    ):
        residual_1 = 3 * x0**2 - 2 * x0 - p
        residual_2 = sin(x0) + x1
        return residual_1, residual_2

# Set up and solve
system = CasadiSystem(num_span=1)
system.add_equation(MyParabola(), (0, 1))
system.add_boundary_conditions({my_p: 5})
system.build()

rootfinder = system.make_rootfinder('kinsol')
guess = system.get_guess()
knowns = system.get_boundary_conds()
solution = solve_root_problem(rootfinder, guess, knowns)
sol_dict = system.sol_to_dict(solution)
```

