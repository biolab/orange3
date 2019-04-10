Orange
======

[![Join the chat at https://gitter.im/biolab/orange3](https://badges.gitter.im/biolab/orange3.svg)](https://gitter.im/biolab/orange3?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

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

Orange requires Python 3.6 or newer.

First, install [Anaconda] for your OS. Create virtual environment for Orange:

    conda create python=3 --name orange3

In your Anaconda Prompt add conda-forge to your channels:

    conda config --add channels conda-forge

This will enable access to the latest Orange release. Then install Orange3:

    conda install orange3

[Anaconda]: https://www.continuum.io/downloads


Installing with pip
-------------------

To install Orange with pip, run the following.

    # Install some build requirements via your system's package manager
    sudo apt install virtualenv git build-essential python3-dev

    # Create a separate Python environment for Orange and its dependencies ...
    virtualenv --python=python3 --system-site-packages orange3venv
    # ... and make it the active one
    source orange3venv/bin/activate

    # Clone the repository and move into it
    git clone https://github.com/biolab/orange3.git
    cd orange3

    # Install Qt dependencies for the GUI
    pip install PyQt5

    # Install other minimum required dependencies
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

### Missing WebKit/WebEngine

Some distributions of PyQt5 come without WebKit or WebEngine, required by some
add-ons and for reporting. Running `pip install PyQtWebEngine` may solve this issue.

Starting Orange GUI
-------------------

To start Orange GUI from the command line, run:

    orange-canvas
    # or
    python3 -m Orange.canvas

Append `--help` for a list of program options.


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
