# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information
project = 'ADeT'
copyright = '2026, Francesco Vaccari'
author = 'Francesco Vaccari'
release = '0.0.1'

# -- General configuration
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
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
