# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
import sphinx_gallery
from sphinx_gallery.sorting import ExampleTitleSortKey, ExplicitOrder

import sphinx_bootstrap_theme

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

curdir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath('../../lameg'))

project = 'laMEG'
copyright = '2024, DANC lab'
author = 'DANC lab'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    #'sphinx_gallery.gen_gallery',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'numpydoc',
    'sphinx_copybutton',
    'sphinx.ext.napoleon'
]

# generate autosummary even if no references
autosummary_generate = True
autodoc_default_options = {'inherited-members': None}
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
default_role = 'autolink'  # XXX silently allows bad syntax, someone should fix

# Sphinx-Copybutton configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = '.rst'

pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'bootstrap'
html_static_path = ['_static']

html_theme_options = {
    'navbar_sidebarrel': False,
    'navbar_links': [
        ("API", "modules"),
        #("Glossary", "glossary"),
        #("What's new", "whats_new"),
        ("GitHub", "https://github.com/danclab/laMEG", True)
    ],
    'bootswatch_theme': "yeti"
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()