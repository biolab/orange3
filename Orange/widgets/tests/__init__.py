import os
import unittest

import Orange
import Orange.widgets

if Orange.data.Table.LOCKING is None:
    Orange.data.Table.LOCKING = True


def load_tests(loader, tests, pattern):
    # Need to guard against inf. recursion. This package will be found again
    # within the discovery process.
    if getattr(load_tests, "_in_load_tests", False):
        return unittest.TestSuite([])

    widgets_dir = os.path.dirname(Orange.widgets.__file__)
    widget_tests_dir = os.path.dirname(__file__)
    top_level_dir = os.path.dirname(os.path.dirname(Orange.__file__))

    if loader is None:
        loader = unittest.TestLoader()
    if pattern is None:
        pattern = 'test*.py'

    load_tests._in_load_tests = True
    try:
        all_tests = [
            # Widgets in this package are discovered separately to avoid
            # infinite recursion (see the guard above)
            loader.discover(widget_tests_dir, pattern, widget_tests_dir),
            loader.discover(widgets_dir, pattern, top_level_dir)
        ]
    finally:
        load_tests._in_load_tests = False

    return unittest.TestSuite(all_tests)
