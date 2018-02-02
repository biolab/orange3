import sys
import os
import unittest
from doctest import DocTestSuite, ELLIPSIS, NORMALIZE_WHITESPACE
from distutils.version import LooseVersion

import numpy

SKIP_DIRS = (
    # Skip modules which import and initialize stuff that require QApplication
    'Orange/widgets',
    'Orange/canvas',
    # Skip because we don't want Orange.datasets as a module (yet)
    'Orange/datasets/'
)

if sys.platform == "win32":
    # convert to platform native path component separators
    SKIP_DIRS = tuple(os.path.normpath(p) for p in SKIP_DIRS)


def find_modules(package):
    """Return a recursive list of submodules for a given package"""
    from os import path, walk
    module = path.dirname(getattr(package, '__file__', package))
    parent = path.dirname(module)
    files = (path.join(dir, file)[len(parent) + 1:-3]
             for dir, dirs, files in walk(module)
             for file in files
             if file.endswith('.py'))
    files = (f for f in files if not f.startswith(SKIP_DIRS))
    files = (f.replace(path.sep, '.') for f in files)
    return files


class Context(dict):
    """
    Execution context that retains the changes the tests make. Preferably
    use one per module to obtain nice "literate" modules that "follow along".

    In other words, directly the opposite of:
    https://docs.python.org/3/library/doctest.html#what-s-the-execution-context

    By popular demand:
    http://stackoverflow.com/questions/13106118/object-reuse-in-python-doctest/13106793#13106793
    http://stackoverflow.com/questions/3286658/embedding-test-code-or-data-within-doctest-strings
    """
    def copy(self):
        return self

    def clear(self):
        pass


def suite(package):
    """Assemble test suite for doctests in path (recursively)"""
    from importlib import import_module
    # numpy 1.14 changed array str/repr (NORMALIZE_WHITESPACE does not
    # handle this). When 1.15 is released update all docstrings and skip the
    # tests for < 1.14.
    npversion = LooseVersion(numpy.__version__)
    if npversion >= LooseVersion("1.14"):
        def setUp(test):
            raise unittest.SkipTest("Skip doctest on numpy >= 1.14.0")
    else:
        def setUp(test):
            pass

    for module in find_modules(package.__file__):
        try:
            module = import_module(module)
            yield DocTestSuite(module,
                               globs=Context(module.__dict__.copy()),
                               optionflags=ELLIPSIS | NORMALIZE_WHITESPACE,
                               setUp=setUp)
        except ValueError:
            pass  # No doctests in module
        except ImportError:
            import warnings
            warnings.warn('Unimportable module: {}'.format(module))


def load_tests(loader, tests, ignore):
    # This follows the load_tests protocol
    # https://docs.python.org/3/library/unittest.html#load-tests-protocol
    import Orange
    tests.addTests(suite(Orange))
    return tests
