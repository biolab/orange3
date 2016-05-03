import os
import unittest
from contextlib import contextmanager

from Orange import widgets


@contextmanager
def named_file(content, encoding=None):
    import tempfile
    import os
    file = tempfile.NamedTemporaryFile("wt", delete=False, encoding=encoding)
    file.write(content)
    name = file.name
    file.close()
    try:
        yield name
    finally:
        os.remove(name)


def suite(loader=None, pattern='test*.py'):
    test_dir = os.path.dirname(__file__)
    widgets_dir = os.path.dirname(widgets.__file__)
    if loader is None:
        loader = unittest.TestLoader()
    if pattern is None:
        pattern = 'test*.py'
    all_tests = [
        loader.discover(test_dir, pattern),
        loader.discover(widgets_dir, pattern, widgets_dir)
    ]

    return unittest.TestSuite(all_tests)


test_suite = suite()


def load_tests(loader, tests, pattern):
    return suite(loader, pattern)


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
