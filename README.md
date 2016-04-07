Orange
======

[![build: passing](https://img.shields.io/travis/biolab/orange3.svg)](https://travis-ci.org/biolab/orange3)
[![codecov.io](https://codecov.io/github/biolab/orange3/coverage.svg?branch=master)](https://codecov.io/github/biolab/orange3?branch=master)

[Orange] is a component-based data mining software. It includes a range of data
visualization, exploration, preprocessing and modeling techniques. It can be
used through a nice and intuitive user interface or, for more advanced users,
as a module for the Python programming language.

This is a development version of Orange 3. The stable version 2.7 is still
available ([binaries] and [sources]).

[Orange]: http://orange.biolab.si/
[binaries]: http://orange.biolab.si/orange2/
[sources]: https://github.com/biolab/orange


Installing
----------
This version of Orange requires Python 3.4 or newer. To build it and install
it in a development environment, run:

    # Create a separate Python environment for Orange and its dependencies,
    # and make it the active one
    virtualenv --python=python3 --system-site-packages orange3venv
    source orange3venv/bin/activate

    # Install the minimum required dependencies first
    pip install numpy
    pip install scipy
    pip install -r requirements-core.txt  # For Orange Python library
    pip install -r requirements-gui.txt   # For Orange GUI

    pip install -r requirements-sql.txt   # To use SQL support
    pip install -r requirements-opt.txt   # Optional dependencies, may fail

    # Finally install Orange in editable/development mode.
    pip install -e .

Installation of SciPy and qt-graph-helpers is sometimes challenging because of
their non-python dependencies that have to be installed manually. More
detailed, if mostly obsolete, guides for some platforms can be found in
the [wiki].

[wiki]: https://github.com/biolab/orange3/wiki


Starting Orange GUI
-------------------

Orange GUI requires PyQt, which is not pip-installable in Python 3. You
have to download and install it system-wide. Make sure that the virtual
environment for orange is created with `--system-site-packages`, so it will
have access to the installed PyQt4.

To start Orange GUI from the command line, assuming it was successfully
installed, run:

    orange-canvas
    # or
    python3 -m Orange.canvas

Append `--help` for a list of program options.


Windows dev setup
-----------------

Windows + GCC:

    python setup.py build_ext --inplace --compile=mingw32
