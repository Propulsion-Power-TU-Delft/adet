# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information
project = 'ADeT'
copyright = '2026, Francesco Vaccari'
author = 'Francesco Vaccari'
release = '0.0.1'

import inspect
from typing import Annotated, get_origin

from adet.varspec import VarSpec

# -- General configuration
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_togglebutton',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = []

# MyST configuration
myst_enable_extensions = [
    'colon_fence',
    'dollarmath',
]
# Anchors for headings up to level 3, so tables can link to individual models
myst_heading_anchors = 3

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# -- HTML output
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_logo = '../images/adet_logo.png'
html_theme_options = {
    'logo': {
        'image_light': '../images/adet_logo_light.png',
        'image_dark': '../images/adet_logo.png',
    }
}


# -- Autodoc signatures of variable hints
# Each residual argument is annotated with `VarSpec.Hint`, i.e.
# `Annotated[MX | Quantity, VarSpec]`. Sphinx would expand the VarSpec dataclass
# fields, so the signature is rebuilt listing each argument by its full symbol,
# e.g. `residual(rho0: stc_rhomass0, ...)`.


def _hint_spec(annotation) -> VarSpec | None:
    if get_origin(annotation) is Annotated and isinstance(
        annotation.__metadata__[0], VarSpec
    ):
        return annotation.__metadata__[0]
    return None


def _hint_signature(app, what, name, obj, options, signature, return_annotation):  # noqa: ARG001 (fixed signature required by Sphinx)
    if signature is None or 'Annotated[' not in signature:
        return None

    params = [p for p in inspect.signature(obj).parameters.values() if p.name != 'self']
    specs = [_hint_spec(p.annotation) for p in params]
    if not params or any(spec is None for spec in specs):
        return None

    args = ', '.join(
        f'{p.name}: {spec.full_symbol(index=True)}' for p, spec in zip(params, specs)
    )
    return f'({args})', return_annotation


def setup(app):
    app.connect('autodoc-process-signature', _hint_signature)
