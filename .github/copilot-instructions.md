# Copilot Instructions

## Project overview

ADeT (Autodiff DEsigner for Turbomachinery) is a Python engineering library for design and analysis of turbomachinery components. It uses automatic differentiation (CasADi, JAX) for equation-based modeling combined with real gas thermodynamics (CoolProp) and modern optimization techniques.

- Language: Python 3.11+
- Package manager: UV (Astral's modern Python packaging tool)

## Environment setup

```bash
uv sync                    # Install base dependencies
uv sync --all-groups       # Install dev and docs dependencies
```

## Running the project

```bash
uv run src/adet/main.py    # Run the main solver demonstration
uv run <script_name>.py    # Run any Python script with UV
```

## Development workflow

After modifying each file, do the following:

1. Format the file:

```bash
ruff format <file_name.py>
```

2. Run a linting check and fix any issues you find:

```bash
ty check <file_name.py>
```

## Naming conventions

- UPPERCASE for constants
- snake_case for functions and variables
- CamelCase for classes
