# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ADeT** (Autodiff DEsigner for Turbomachinery) is a Python engineering library for design and analysis of turbomachinery components. It uses automatic differentiation (CasADi, JAX) for equation-based modeling combined with real gas thermodynamics (CoolProp) and modern optimization techniques.

- **Language**: Python 3.11+
- **Package Manager**: UV (Astral's modern Python packaging tool)
- **Entry Point**: `adet:main` (CLI command `adet`)

## Essential Commands

### Environment Setup
```bash
uv sync                    # Install base dependencies
uv sync --all-groups       # Install dev and docs dependencies
```

### Running Code
```bash
uv run src/adet/main.py    # Run the main solver demonstration
uv run <script_name>.py    # Run any Python script with UV
```

### Development Tools

**Linting & Formatting** (Ruff):
```bash
ruff check src/            # Check for linting issues
ruff format src/           # Auto-format code
```
Ruff can also be integrated into your IDE:
- **VS Code**: Install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
- **PyCharm**: Configure as an external tool or use the Ruff plugin
- **Other IDEs**: Most support Ruff via plugins or external tool configuration

**Type Checking** (Basedpyright/Pyright):
```bash
pyright                    # Run type checker in basic mode
```
Pyright/Basedpyright can be integrated into your IDE:
- **VS Code**: Install [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) (includes Pyright) or Basedpyright extension
- **PyCharm**: Enable Pyright via plugin
- **Other IDEs**: Most modern Python IDEs support Pyright integration

**Interactive Development**:
```bash
uv run ipython             # Launch IPython for interactive work
```

## Architecture Overview

### Core System Architecture

ADeT uses an equation-based modeling approach centered around three key concepts:

1. **FlowNode** (`src/adet/node.py`):
   - Represents a point in the flow with complete thermokinematic state
   - Maintains static, total, and relative total thermodynamic properties
   - Handles unit conversions via Pint
   - Supports multiple spanwise stations

2. **SystemAssembler** (`src/adet/assembly.py`):
   - Builds and compiles systems of equations
   - Manages variable registries and unit scaling
   - Bridges component networks to backend solvers (CasADi, JAX)
   - Handles residual computation and solution tracking

3. **ComponentNetwork** (`src/adet/components/network.py`):
   - Orchestrates turbomachinery components (inlets, blade rows, diffusers, etc.)
   - Manages inter-component connections (mass flow, enthalpy, angles)
   - Builds complete system topology from component definitions

### Equation and Component Structure

**Equation System** (`src/adet/equations/`):
- **EquationBase** (`base_equation.py`): Abstract base class enabling symbolic equation definitions with automatic EOS integration
- **Fundamental equations** (`fundamental.py`): Mass balance, kinematics, energy, total-static relations
- **Gas relations** (`nondimensional.py`, `special.py`): Ideal gas, real gas (CoolProp), nondimensional forms (Mach, gamma)
- **Shock models** (`control_volumes.py`): Oblique shock relations, outlet shock equations for blade row analysis
- **Component linkers**: Equations connecting outputs of one component to inputs of another
- **Loss models**: Integrated into blade row equations via `losses/` module

**Components** (`src/adet/components/`):
- **BaseComponent**: Abstract interface for all turbomachinery components
- **BladeRow**: Blade row (compressor/turbine stage) with loss integration
- **SketchVanelessDiff**: Vaneless diffuser implementation
- **Connections**: Define component inlets/outlets and state variables

### Fluid and Thermodynamic Models

**Fluid Module** (`src/adet/fluid/`):
- **eos.py**: Equation of state interface using CoolProp
  - Supports multiple thermodynamic backends (analytical, CoolProp real gas via REFPROP)
  - Automatic derivative frameworks (CasADi symbolic, JAX autodiff)
- **settings.py**: Configuration for fluid model and update variables
- **properties.py**: Gas property calculations (density, viscosity, heat capacities, etc.)
- **derivatives_sketches/**: Experimental derivative implementations (CasADi, JAX)

**Loss Models** (`src/adet/losses/`):
- **profile.py**: Detailed profile loss calculations (largest loss module)
- **basic.py**: Simplified loss correlations
- Extensible loss base class for custom models

### Data Flow and Typical Workflows

**Low-level API** (equation-by-equation, as in `src/adet/examples/mach_problem.py`):
1. Create `CasadiSystem()` instance
2. Define fluid settings (`FluidSettings` with `FluidModel` and update variables)
3. Add equations explicitly with `system.add_equation(equation_instance, position)`
4. Set boundary conditions with `system.add_boundary_conditions(bc_dict)`
5. Call `system.build()` to compile equations
6. Create rootfinder with `system.make_rootfinder('kinsol')` or `'ipopt'`
7. Get scaled guess and constraints, solve with `solve_root_problem()`
8. Convert solution back with `system.sol_to_dict(sol)`

**High-level API** (component-network based, as in `src/adet/main.py`):
1. Configure fluid settings (thermodynamic model, real gas, update variables)
2. Create component instances (inlet, blade rows, diffusers)
3. Define component connections (geometry parameters, flow angles)
4. Build SystemAssembler with ComponentNetwork
5. Select backend solver (Newton-Raphson, IPOPT via CasADi)
6. Solve system to convergence
7. Extract and visualize results (kinematics, losses, flow properties)

## Naming Conventions

### File Names
- **Python modules**: `snake_case` (e.g., `base_equation.py`, `blade_row.py`, `sketch_vaneless_diff.py`)
- **Configuration**: `pyproject.toml`, `README.md`, `CLAUDE.md`

### Class Names
- **PascalCase** for all classes:
  - Core: `FlowNode`, `EquationBase`, `BaseComponent`, `SystemAssembler`, `ComponentNetwork`
  - Containers: `VariableContainer`, `KinematicContainer`, `ThermostateContainer`
  - Registries: `DefaultUnitsRegistry`, `GuessRegistry`, `ScalingRegistry`
  - Models: `FluidSettings`, `FluidModel`, `LossModel`
  - Specialized: `UniqueEquation`, `DeviationModel`, `IncidenceModel`, `CamberLineGeom`

### Variable Names and Specifications

**Variables Module** (`src/adet/variables.py`):
- **NodeVariables**: Per-node, per-state variable access (e.g., `n0 = NodeVariables(0)`)
- **ThermoVariables**: Thermodynamic properties with unit specs, scaling hints, and bounds
- **VarSpec**: Variable specification with name, units, guess value, and physical bounds
- Access pattern: `node.container.PropertyName._at_node()` for unit-aware variable definitions

**FlowNode state containers**:
- `stc` - Static thermodynamic state
- `tot` - Total thermodynamic state
- `rlt` - Relative total thermodynamic state
- `kin` - Kinematics (velocity components)
- `geo` - Geometry
- `oth` - Other variables (entropy, angles, losses, etc.)

**Equation arguments**: Format `<state>_<var_type><index>` (e.g., `stc_p0`, `tot_T1`, `kin_V0`)
- State identifiers: `stc`, `tot`, `rlt`, `kin`, `geo`, `oth`
- Trailing digit indicates node index

**Common variables**: `snake_case` (e.g., `num_span`, `scaling_factor`, `node_name`)

### Function/Method Names
- **snake_case** for all functions and methods (e.g., `read_from_node`, `fetch_state`, `to_symbolic`)
- **Private methods**: Leading underscore (e.g., `_validate_argument`, `_count_equations_ast`)

## Code Style and Conventions

- **Line Length**: 88 characters (Ruff enforcement)
- **Quotes**: Single quotes for strings
- **Type Hints**: Basic type checking enabled; add hints for modules you work on (except functions with intentional polymorphism like residuals)
- **Linting Rules**: E (pycodestyle errors), W (warnings), F (pyflakes), ARG (unused arguments), C4 (comprehensions)
- **Unused Imports**: F401 ignored (handled by pyright)
- **Per-file Exceptions**: `__init__.py`, `main.py`, and files in `tests/`, `docs/`, `tools/` ignore E402 (import order)

### Ruff Configuration
```toml
line-length = 88
select = ["E", "W", "F", "ARG", "C4"]
ignore = ["F401"]
quote-style = "single"
docstring-code-format = true
```

## Key Directories and Modules

```
src/adet/
├── components/        # Turbomachinery component definitions
├── equations/         # Equation definitions and residuals
│   ├── base_equation.py    # EquationBase abstract class
│   ├── fundamental.py      # Core equations (mass, energy, kinematics)
│   ├── nondimensional.py   # Mach, gamma, and dimensionless relations
│   ├── control_volumes.py  # Shock models, outlet conditions
│   ├── special.py          # Thermo variable adders, specialized equations
│   └── geometrical.py      # Annulus areas, geometric relations
├── losses/           # Loss model implementations
├── fluid/            # Thermodynamic models and EOS
├── tools/            # Utilities (plotting, numerics, strings, etc.)
├── examples/         # Example scripts demonstrating usage
├── assembly.py       # System assembly and solving (CasadiSystem, SystemAssembler)
├── variables.py      # Variable enums (NodeVariables, ThermoVariables)
├── varspec.py        # VarSpec for unit-aware variable definitions
├── node.py           # FlowNode for thermokinematic state
├── solution.py       # Root problem solving utilities
├── main.py           # Main demonstration/entry point
└── config_main.py    # Configuration constants
```

## Important Notes

- **Symbolic Computing**: Equations are defined symbolically using CasADi, enabling automatic differentiation and efficient Jacobian computation
- **Real Gas Support**: CoolProp integration for accurate thermodynamic properties; REFPROP access requires separate installation (see README.md)
- **Unit Handling**: Pint library manages unit conversions throughout; always specify units on physical quantities
- **Extensibility**: Add new components by subclassing BaseComponent and defining residual equations; add new equations by subclassing EquationBase
- **Testing**: Use Pytest; tests can be discovered and run with `pytest` command
