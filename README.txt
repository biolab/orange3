Orange
======

Orange is a component-based data mining software. It includes a range of data
visualization, exploration, preprocessing and modeling techniques. It can be
used through a nice and intuitive user interface or, for more advanced users,
as a module for Python programming language.

This is an early development version of Orange 3.0. The current stable version
2.7 is available on http://orange.biolab.si (binaries) and
https://bitbucket.org/biolab/orange (sources).

Installing
----------

This version of Orange requires Python 3.2 or newer. To build it, run::

    pip install numpy
    pip install -r requirements.txt
    python setup.py develop

inside a virtual environment that uses Python 3.2.

Installation of Scipy and qt-graph-helpers is sometimes challenging because of
their non-python dependencies that have to be installed manually. Detailed
guides for some platforms can be found on wiki
(https://github.com/biolab/orange3/wiki).

Starting Orange Canvas
----------------------

Orange Canvas requires PyQt, which is not pip-installable in Python 3. You
have to download and install it system-wide. Make sure that the virtual
environment for orange is created with --system-site-packages, so it will have
access to installed PyQt4.

To start orange canvas from the command line, run::

     python3 -m Orange.canvas

Windows dev setup
-----------------

Windows + gcc:
	python setup.py build_ext -i --compile=mingw32
