# ADeT
**A**utodiff **De**signer for **T**urbomachinery

This is a tool for design and analysis of turbomachinery

## Getting started
This project uses [uv](https://docs.astral.sh/uv) for packaging.

1. Clone this repository
1. [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)
1. Move into the project root directory
1. Run `uv sync`  
    - Optionally use `--all-groups` flag for dev and docs dependency groups
1. Run scripts using `uv run <script_name>.py`
    - `uv` creates a `.venv` virtual environment folder within the root directory, which can be activated as normal, or linked to VSCode or PyCharm

> ### Note
> To access [REFPROP](https://www.nist.gov/srd/refprop) through CoolProp, check out the [guide](https://coolprop.org/coolprop/REFPROP.html).

## Contributing
This project uses :

- [ruff](https://docs.astral.sh/ruff) for linting and formatting, 
- [basedpyright](https://github.com/DetachHead/basedpyright) for static type checking.

Please ensure your IDE supports and has these tools installed to easily comply with style guidelines. 

We only apply basic type checking, therefore missing type and return hints will not be reported aggressively, nontheless it is recommended that you add hints for modules you work on, except where it is not suitable (e.g. functions with intentional polymorphism, such as residuals).
