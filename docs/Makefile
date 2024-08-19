# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    = .
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = lameg
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@echo "Please use \`make <target>' where <target> is one of:"
	@echo "  html			to make standalone HTML files"
	@echo "  linkcheck		to check all external links for integrity"
	@echo "  doctest		to run all doctests embedded in the documentation (if enabled)"

.PHONY: help Makefile clean html linkcheck doctest

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# %: Makefile
# 	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	-rm -rf _build/*
	-rm -rf _source/*
	-rm -rf _autosummary/*
	-rm -rf auto_examples/*
	-rm -rf auto_tutorials/*
	-rm -rf generated

apidoc:
	sphinx-apidoc -f -o _source/ ../lameg/ ../tests*
	@echo
	@echo "Sources built. The sources are in _sources/"

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

doc: clean html

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(SOURCEDIR) _build/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in _build/linkcheck/output.txt."

doctest:
	$(SPHINXBUILD) -b doctest .$(SOURCEDIR) _build/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in _build/doctest/output.txt."

view:
	@python -c "import webbrowser; webbrowser.open_new_tab('file://$(PWD)/_build/html/index.html')"

show: view

