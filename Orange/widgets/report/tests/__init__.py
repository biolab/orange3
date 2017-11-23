from os.path import dirname
import unittest

import Orange


def suite(loader=None, pattern='test*.py'):
    return unittest.TestSuite(loader.discover(dirname(__file__),
                                              pattern or 'test*.py',
                                              dirname(Orange.__file__)))


def load_tests(loader, tests, pattern):
    return suite(loader, pattern)


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
