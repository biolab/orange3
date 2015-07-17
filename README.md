Orange
======

[![build: passing](https://img.shields.io/travis/biolab/orange3.svg)](https://travis-ci.org/biolab/orange3)
[![coverage: poor](https://img.shields.io/coveralls/biolab/orange3.svg)](https://coveralls.io/r/biolab/orange3?branch=master)
[![code quality: worse](https://img.shields.io/scrutinizer/g/biolab/orange3.svg)](https://scrutinizer-ci.com/g/biolab/orange3/)

Orange is a component-based data mining software. It includes a range of data
visualization, exploration, preprocessing and modeling techniques. It can be
used through a nice and intuitive user interface or, for more advanced users,
as a module for the Python programming language.

This is an early development version of Orange 3. The current stable version
2.7 is available ([binaries] and [sources]).

[binaries]: http://orange.biolab.si
[sources]: https://github.com/biolab/orange


Installing
----------

This version of Orange requires Python 3.2 or newer. To build it, run::

    pip install numpy
    pip install -r requirements.txt
    python setup.py develop

inside a virtual environment that uses Python 3.2.

Installation of SciPy and qt-graph-helpers is sometimes challenging because of
their non-python dependencies that have to be installed manually. Detailed
guides for some platforms can be found in the [wiki].

[wiki]: https://github.com/biolab/orange3/wiki


Starting Orange Canvas
----------------------

Orange Canvas requires PyQt, which is not pip-installable in Python 3. You
have to download and install it system-wide. Make sure that the virtual
environment for orange is created with `--system-site-packages`, so it will
have access to the installed PyQt4.

To start Orange Canvas from the command line, run:

    python3 -m Orange.canvas


Windows dev setup
-----------------

Windows + GCC:

    python setup.py build_ext -i --compile=mingw32
