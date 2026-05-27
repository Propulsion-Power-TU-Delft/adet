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
    'sphinx_inline_tabs',
]

templates_path = ['_templates']
exclude_patterns = []

# MyST configuration
myst_enable_extensions = [
    'colon_fence',
    'dollarmath',
]

# -- HTML output
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {}
