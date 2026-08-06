# ADeT
**A**utodiff **De**signer for **T**urbomachinery

A Python library for equation-oriented modeling, with a focus on turbomachinery problems.
Backend for automatic differentiation provided by CasaDi and real gas thermodynamics with CoolProp.

## Features

- **Equation-oriented modeling**: Define physics symbolically as residual equations
- **Real gas thermodynamics**: CoolProp integration with REFPROP support
- **Automatic differentiation**: Jacobians computed automatically
- **Unit-aware**: Pint integration for automatic unit checking and conversion

## Getting Started

Check out the **[DOCUMENTATION](https://fvaccari1.bitbucket.io/index.html)**

This project uses [uv](https://docs.astral.sh/uv) for packaging and dependency management.

### Installation

1. Clone this repository
2. [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)
3. Navigate to the project root directory
4. Install dependencies:
   ```bash
   uv sync                    # Install base dependencies
   uv sync --all-groups       # Install dev and docs dependencies
   ```
> **Note**: To access [REFPROP](https://www.nist.gov/srd/refprop) through CoolProp, see the [integration guide](https://coolprop.org/coolprop/REFPROP.html).

## Development

### Code Quality Tools

- **[Ruff](https://docs.astral.sh/ruff)**: Linting and formatting
- **[ty](https://github.com/bnemetis/ty)**: Static type checking

Rules are included in `pyproject.toml` and should be read automatically. 
These tools can be used via CLI or integrated directly into your IDE.
