% The contributing guide is maintained as CONTRIBUTING.md at the repository root
% (GitHub surfaces it on the issue/PR pages) and included here verbatim.

```{include} ../CONTRIBUTING.md
```

## Building the documentation

The documentation is built with [Sphinx](https://www.sphinx-doc.org/) using the
[Furo](https://pradyunsg.me/furo/) theme, with [MyST-NB](https://myst-nb.readthedocs.io/)
rendering both the Markdown pages and the tutorial notebooks. To build it
locally:

```bash
python3 -m pip install -e ".[docs,examples,mapping]"
python3 docs/copy_notebooks.py            # stage notebooks into docs/tutorials/
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. The self-contained tutorials (Ex1–Ex4 for
both methods) execute at build time; the two `Ex5` notebooks each need ~600 MB of
external data and render as code only.

## Releasing (maintainers)

Continuous integration runs on GitHub Actions:

- **`tests.yml`** — runs the test suite on every push and pull request across
  Python 3.9–3.13.
- **`docs.yml`** — builds these docs on every push and pull request, and deploys
  them to GitHub Pages from `master`.
- **`publish.yml`** — builds the sdist and wheel and publishes them to PyPI when
  a GitHub release is published.

A release is cut by publishing a GitHub release for the tagged version; the
`publish.yml` workflow then uploads to PyPI automatically. Two settings are
configured once on the repository:

- **GitHub Pages** — *Settings → Pages → Source* set to **GitHub Actions**, so
  `docs.yml` can deploy to <https://brmather.github.io/pycurious/>.
- **PyPI Trusted Publishing** — on the `pycurious` project at PyPI, add a trusted
  publisher pointing at owner `brmather`, repository `pycurious`, workflow
  `publish.yml`, environment `pypi`. This lets Actions publish over OIDC with no
  stored password or API token.
