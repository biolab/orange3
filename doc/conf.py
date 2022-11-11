"""
This configuration use sphinx-multiproject which builds multiple
Sphinx projects for the Read the Docs. We publish each project at read-the-docs
as orange3 RTD project's subproject. This config file is only required for the
Read the Docs build. Each documentation project can still be built separately
with sphinx-build (make html).

To select a documentation project that the RTD will build, set the PROJECT
environment variable in RTD subprojects to the documentation project name
(e.g. PROJECT=data-mining-library)

To test the documentation build locally run (from doc directory):
```
PROJECT="<project name>" sphinx-build . ./_build
```
More about shpinx-multiproject:
https://sphinx-multiproject.readthedocs.io/en/latest/index.html
"""

# pylint: disable=duplicate-code
extensions = [
    "multiproject",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "recommonmark",
]

# Define the projects that will share this configuration file.
multiproject_projects = {
    "data-mining-library": {
        "path": "data-mining-library/source/"
    },
    "development": {
        "path": "development/source/"
    },
    "visual-programming": {
        "path": "visual-programming/source/"
    },
}
