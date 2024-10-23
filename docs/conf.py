# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# import matplotlib as mpl

# mpl.use("agg")

curdir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curdir, "..")))
sys.path.append(os.path.abspath(os.path.join(curdir, "..", "lameg")))

import lameg # noqa

# -- Project information -----------------------------------------------------

project = 'laMEG'
copyright = '2024, DANC lab'
author = 'DANC lab'
release = "0.0.5"

# release = lameg.__release__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "numpydoc",
    "jupyter_sphinx",
    "sphinx_gallery.gen_gallery",
    "sphinxemoji.sphinxemoji",
    'sphinx_sitemap'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "config.py"]

# generate autosummary even if no references
# autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "special-members": "__init__",
    "undoc-members": True,
    "show-inheritance": True,
    "exclude-members": "__weakref__"
}
numpydoc_show_class_members = False

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages
html_theme = "pydata_sphinx_theme"

# SEO stuff
html_baseurl = 'https://danclab.github.io/laMEG/'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
   "logo": {
      "image_light": "_static/logo.png",
      "image_dark": "_static/logo.png",
      "text": "lameg",
   },
    "show_toc_level": 1,
    "external_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/danclab/laMEG",
        }
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/danclab/laMEG",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/danc_labo",
            "icon": "fa-brands fa-twitter",
        },
    ],
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "danclab",
    "github_repo": "laMEG",
    "github_version": "master",
    "doc_path": "docs",
    "metatags": """
       <meta name="description" content="laMEG: A Python package for laminar analysis of MEG data">
       <meta name="keywords" content="MEG, laminar analysis, neuroscience, Python">
       """
}

# SEO stuff
html_extra_path = ['robots.txt', 'google0bc7cbc5c0e0b225.html']

# -- Options for Sphinx-gallery HTML ------------------------------------------
from sphinx_gallery.sorting import FileNameSortKey

sphinx_gallery_conf = {
    "doc_module": ("laMEG",),
    'examples_dirs': ['../tutorials'],   # Path to your tutorials
    'gallery_dirs': ['auto_tutorials'],  # Path to where you want to store the generated output
    'filename_pattern': 'tutorial_',
    'within_subsection_order': FileNameSortKey,
    'min_reported_time': 0,
    'remove_config_comments': True,
    'first_notebook_cell': "%matplotlib inline",
}

suppress_warnings = ["config.cache"]
