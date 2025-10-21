# ADeT
**A**utodiff **De**signer for **T**urbomachinery

This is a tool for design and analysis of turbomachinery

## Getting started
This project uses [https://docs.astral.sh/uv](uv) for packaging.

1. Clone this repository
2. [https://docs.astral.sh/uv/getting-started/installation/](Install `uv`)
3. Run `uv sync` from within this folder 
    - (Optionally use `--all-groups` flag for dev and docs dependency groups)
4. Run scripts using `uv run <script_name>.py`
    - `uv` creates a `.venv` virtual environment folder within the root directory, which can be activated as normal, or linked to VSCode or PyCharm

### Note
For accessing thermodynamic libraries through [https://www.nist.gov/srd/refprop](REFPROP) you must link your [https://coolprop.org/](CoolProp) installation to the former's dynamic library, check out the [https://coolprop.org/coolprop/REFPROP.html](guide)

## Contributing
This project uses [https://docs.astral.sh/ruff](ruff) for linting and formatting, and [](pyright) (or equivalently [](basedpyright)) for static type checking. Please ensure your IDE supports and has these two tools installed to ensure compliance with coding style rules. 

We only apply basic type checking, therefore missing type and return hints will not be reported aggressively by `pyright`, nontheless it is recommended that you add hints for modules you work on, except where it is not suitable (e.g. functions with intentional polymorphism, such as residuals).

