# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "OptiMask"
author = "Cyril Joly"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon", "nbsphinx", "nbsphinx_link", "sphinx_copybutton", "sphinx_favicon"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_sidebars = {"**": []}
html_logo = "_static/icon.svg"
html_show_sphinx = False
html_theme_options = {"footer_items": []}

favicons = ["favicon-16x16.png", "favicon-32x32.png", "favicon-128x128.png"]

autodoc_default_options = {"members": True, "undoc-members": True}
