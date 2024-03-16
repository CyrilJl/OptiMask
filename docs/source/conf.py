# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

# Obtenir le chemin absolu du dossier parent du dossier parent du fichier actuel
path = str(Path(__file__).resolve().parent.parent.parent)
print(path)
sys.path.insert(0, path)

project = 'OptiMask'
copyright = '2024, Cyril Joly'
author = 'Cyril Joly'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.napoleon', 'nbsphinx', 'nbsphinx_link', 'sphinx_copybutton']

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_sidebars = {
    "**": ["globaltoc.html"]
}

autodoc_default_options = {
    'members': True,
    'undoc-members': True
}
