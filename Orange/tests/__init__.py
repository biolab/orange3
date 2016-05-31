import os
import unittest
import tempfile
from contextlib import contextmanager

import Orange


@contextmanager
def named_file(content, encoding=None):
    file = tempfile.NamedTemporaryFile("wt", delete=False, encoding=encoding)
    file.write(content)
    name = file.name
    file.close()
    try:
        yield name
    finally:
        os.remove(name)


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
    top_level_dir = os.path.dirname(os.path.dirname(Orange.__file__))
    all_tests = [
        loader.discover(test_dir, pattern, top_level_dir),
    ]

    return unittest.TestSuite(all_tests)


def load_tests(loader, tests, pattern):
    return suite(loader, pattern)


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
