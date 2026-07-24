# Minimal makefile for Sphinx documentation
#
# Run `make html` from the docs/ directory to build the site locally. The
# `copy-notebooks` prerequisite stages the tutorial notebooks into the source
# tree first (see copy_notebooks.py).

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
PYTHON        ?= python3

.PHONY: help html copy-notebooks clean Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

copy-notebooks:
	$(PYTHON) copy_notebooks.py

html: copy-notebooks
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)" tutorials/bouligand tutorials/tanaka data

# Catch-all target: route unknown targets to sphinx-build after staging notebooks.
%: copy-notebooks Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
