Orange
======

Orange is a component-based data mining software. It includes a range of data
visualization, exploration, preprocessing and modeling techniques. It can be
used through a nice and intuitive user interface or, for more advanced users,
as a module for Python programming language.

Installing
----------

To build and install Orange run::

     python setup.py build
     python setup.py install

from the command line. You can customize the build process by
editing the setup-site.cfg file in this directory (see the comments
in that file for instructions on how to do that).

Starting Orange Canvas
----------------------

Start orange canvas from the command line with::

     orange-canvas

Installation for Developers
---------------------------

To install in `development mode`_ run::

    python setup.py develop
   
.. _development mode: http://packages.python.org/distribute/setuptools.html#development-mode
