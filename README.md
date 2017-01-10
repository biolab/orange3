Orange
======

[![build: passing](https://img.shields.io/travis/biolab/orange3.svg)](https://travis-ci.org/biolab/orange3)
[![codecov](https://codecov.io/gh/biolab/orange3/branch/master/graph/badge.svg)](https://codecov.io/gh/biolab/orange3)

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

    # Install some build requirements via your system's package manager
    sudo apt-get install virtualenv git build-essential

    # Also install Qt dependencies for the GUI
    sudo apt-get install python3-pyqt4
    # or if python version is >= 3.5
    #  pip install pyqt5 

    # Create a separate Python environment for Orange and its dependencies ...
    virtualenv --python=python3 --system-site-packages orange3venv
    # ... and make it the active one
    source orange3venv/bin/activate

    # Clone the repository and move into it
    git clone https://github.com/biolab/orange3.git
    cd orange3

    # Install the minimum required dependencies first
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

Anaconda Installation
---------------------

First, install [Anaconda] for your OS (Python version 3.5+). Create virtual environment for Orange:

    conda create python=3 --name orange3 

In your Anaconda Prompt add conda-forge to your channels:

    conda config --add channels conda-forge

This will enable access to the latest Orange release. Then install Orange3:

    conda install orange3

[Anaconda]: https://www.continuum.io/downloads

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

If you're running Orange with PyQt5 or if you have multiple PyQt versions
available, set the environmental variable `QT_API` to the PyQt version to use,
e.g.:

    export QT_API=pyqt5
    orange-canvas


Compiling on Windows
--------------------

Get appropriate wheels for missing libraries. You will need [numpy+mkl] and [scipy].

[numpy+mkl]: http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
[scipy]: http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy

Install them with

    pip install some-wheel.whl

Install [Visual Studio compiler]. Then go to Orange3 folder and run:

[Visual Studio compiler]: http://landinghub.visualstudio.com/visual-cpp-build-tools

    python setup.py build_ext -i --compiler=msvc install
