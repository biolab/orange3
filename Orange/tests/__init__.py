import os
import unittest
import tempfile
from contextlib import contextmanager

import numpy as np
import Orange


@contextmanager
def named_file(content, encoding=None, suffix=''):
    file = tempfile.NamedTemporaryFile("wt", delete=False,
                                       encoding=encoding, suffix=suffix)
    file.write(content)
    name = file.name
    file.close()
    try:
        yield name
    finally:
        os.remove(name)


@np.vectorize
def naneq(a, b):
    try:
        return (np.isnan(a) and np.isnan(b)) or a == b
    except TypeError:
        return a == b


def assert_array_nanequal(a, b, *args, **kwargs):
    """
    Similar as np.testing.assert_array_equal but with better handling of
    object arrays.

    Note
    ----
    Is not fast!

    Parameters
    ----------
    a : array-like
    b : array-like
    """
    return np.testing.utils.assert_array_compare(naneq, a, b, *args, **kwargs)


def test_dirname():
    """
    Return the absolute path to the Orange.tests package.

    Returns
    -------
    path : str
    """
    return os.path.dirname(__file__)


def test_filename(path):
    """
    Return an absolute path to a resource within Orange.tests package.

    Parameters
    ----------
    path : str
        Path relative to `test_dirname()`
    Returns
    -------
    abspath : str
        Absolute path
    """
    return os.path.join(test_dirname(), path)


def suite(loader=None, pattern='test*.py'):
    test_dir = os.path.dirname(__file__)
    if loader is None:
        loader = unittest.TestLoader()
    if pattern is None:
        pattern = 'test*.py'
    orange_dir = os.path.dirname(Orange.__file__)
    top_level_dir = os.path.dirname(orange_dir)
    all_tests = [loader.discover(test_dir, pattern, top_level_dir)]
    if not suite.in_tests:  # prevent recursion
        suite.in_tests = True
        all_tests += (loader.discover(dir, pattern, dir)
                      for dir in (os.path.join(orange_dir, fn, "tests")
                                  for fn in os.listdir(orange_dir)
                                  if fn != "widgets")
                      if os.path.exists(dir))
    return unittest.TestSuite(all_tests)

suite.in_tests = False

def load_tests(loader, tests, pattern):
    return suite(loader, pattern)


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
