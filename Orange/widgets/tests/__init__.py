import os
import unittest
import Orange.widgets


def load_tests(loader, tests, pattern):
    # Need to guard against inf. recursion. This package will be found again
    # within the discovery process.
    if getattr(load_tests, "_in_load_tests", False):
        return unittest.TestSuite([])

    widgets_dir = os.path.dirname(Orange.widgets.__file__)
    top_level_dir = os.path.dirname(os.path.dirname(Orange.__file__))

    if loader is None:
        loader = unittest.TestLoader()
    if pattern is None:
        pattern = 'test*.py'

    load_tests._in_load_tests = True
    try:
        all_tests = [
            loader.discover(widgets_dir, pattern, top_level_dir)
        ]
    finally:
        load_tests._in_load_tests = False

    return unittest.TestSuite(all_tests)
