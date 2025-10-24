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

**Type Checking** (Basedpyright):
```bash
pyright                    # Run type checker in basic mode
```

**Testing** (Pytest):
```bash
pytest                     # Run all tests
pytest tests/<test_file>   # Run specific test file
pytest -v                  # Verbose output
pytest -x                  # Stop on first failure
```

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
- **EquationBase** (`base_equation.py`): Abstract base class enabling symbolic equation definitions
- Fundamental equations: mass balance, kinematics, energy
- Gas relations: ideal gas, real gas (CoolProp), nondimensional forms
- Component linkers: equations connecting outputs of one component to inputs of another
- Loss models: integrated into blade row equations

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

### Data Flow and Typical Workflow

As demonstrated in `src/adet/main.py`:
1. Configure fluid settings (thermodynamic model, real gas, update variables)
2. Create component instances (inlet, blade rows)
3. Define component connections (geometry parameters, flow angles)
4. Build SystemAssembler with ComponentNetwork
5. Select backend solver (Newton-Raphson, IPOPT via CasADi)
6. Solve system to convergence
7. Extract and visualize results (kinematics, losses, flow properties)

## Code Style and Conventions

- **Line Length**: 88 characters (Ruff enforcement)
- **Quotes**: Single quotes for strings
- **Type Hints**: Basic type checking enabled; add hints for modules you work on (except functions with intentional polymorphism like residuals)
- **Linting Rules**: E (pycodestyle errors), W (warnings), F (pyflakes)
- **Per-file Exceptions**: `__init__.py` and `main.py` ignore E402 (import order)

### Ruff Configuration
```toml
line-length = 88
select = ["E", "W", "F"]
quote-style = "single"
docstring-code-format = true
```

## Key Directories

```
src/adet/
├── components/        # Turbomachinery component definitions
├── equations/         # Equation definitions and residuals
├── losses/           # Loss model implementations
├── fluid/            # Thermodynamic models and EOS
├── tools/            # Utilities (plotting, numerics, strings, etc.)
├── examples/         # Example scripts demonstrating usage
├── assembly.py       # System assembly and solving
├── node.py          # FlowNode for thermokinematic state
├── main.py          # Main demonstration/entry point
└── config_main.py   # Configuration constants
```

## Important Notes

- **Symbolic Computing**: Equations are defined symbolically using CasADi, enabling automatic differentiation and efficient Jacobian computation
- **Real Gas Support**: CoolProp integration for accurate thermodynamic properties; REFPROP access requires separate installation (see README.md)
- **Unit Handling**: Pint library manages unit conversions throughout; always specify units on physical quantities
- **Extensibility**: Add new components by subclassing BaseComponent and defining residual equations; add new equations by subclassing EquationBase
- **Testing**: Use Pytest; tests can be discovered and run with `pytest` command
