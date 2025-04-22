import os
import sys
sys.path.insert(0, os.path.abspath('/home/loqman/Downloads/projs/DiffusionModels'))  # Points to DiffusionModels directory

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TorchDiff'
copyright = '2025, Loghman Samani'
author = 'Loghman Samani'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',   # For automatic docstring extraction
    'sphinx.ext.napoleon',  # For Google/NumPy docstring support
    'sphinx.ext.viewcode',  # To include links to source code
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
