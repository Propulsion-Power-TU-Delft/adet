# ADeT
**A**utodiff **De**signer for **T**urbomachinery

A Python library for equation-oriented design and analysis of turbomachinery components using automatic differentiation (CasADi, JAX) and real gas thermodynamics (CoolProp).

## Features

- **Equation-oriented modeling**: Define physics symbolically, solve systems automatically
- **Multiple solver backends**: CasADi (symbolic, C-code generation) and JAX (autodiff)
- **Real gas thermodynamics**: CoolProp integration with REFPROP support
- **Automatic differentiation**: Jacobians computed automatically
- **Unit-aware**: Pint integration for automatic unit checking and conversion

## Getting Started

This project uses [uv](https://docs.astral.sh/uv) for packaging.

### Installation

1. Clone this repository
2. [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)
3. Navigate to the project root directory
4. Install dependencies:
   ```bash
   uv sync                    # Install base dependencies
   uv sync --all-groups       # Install dev and docs dependencies
   ```

### Quick Start

Run the one of the examples:
```bash
uv run src/adet/examples/air_supply_compressor_design.py
```

> **Note**: To access [REFPROP](https://www.nist.gov/srd/refprop) through CoolProp, see the [integration guide](https://coolprop.org/coolprop/REFPROP.html).

## Development

### Code Quality Tools

- **[Ruff](https://docs.astral.sh/ruff)**: Linting and formatting
- **[Basedpyright](https://github.com/DetachHead/basedpyright)** / **[Pyright](https://github.com/microsoft/pyright)**: Static type checking

These tools can be used via CLI or integrated directly into your IDE:
- **VS Code**: Install the [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) and [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) (includes Pyright) extensions
- **PyCharm**: Configure Ruff as an external tool and enable Pyright via plugin
- **Other IDEs**: Most modern Python IDEs support these tools via plugins or external tool configuration

### Development Commands

```bash
# Linting and formatting (CLI)
ruff check src/            # Check for linting issues
ruff format src/           # Auto-format code

# Type checking (CLI)
pyright                    # Run type checker
```

### Code Style

- **Line length**: 88 characters
- **Quotes**: Single quotes for strings
- **Type hints**: Basic type checking enabled; add hints for new code (except for polymorphic functions)
- **Linting rules**: E (pycodestyle errors), W (warnings), F (pyflakes), ARG (unused arguments), C4 (comprehensions)

### Contributing

Ensure your IDE is configured to use Ruff and Basedpyright for the best development experience. Type hints are recommended for new code, but optional for functions with intentional polymorphism (e.g., equation residuals).
