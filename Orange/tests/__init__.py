import os
import unittest

try:
  from Orange.widgets.tests import test_settings, test_setting_provider
  from Orange.widgets.data.tests import test_owselectcolumns
  run_widget_tests = True
except ImportError:
  run_widget_tests = False

def suite():
    test_dir = os.path.dirname(__file__)
    all_tests = [
        unittest.TestLoader().discover(test_dir),
    ]
    if run_widget_tests:
      all_tests.extend([
        unittest.TestLoader().loadTestsFromModule(test_settings),
        unittest.TestLoader().loadTestsFromModule(test_setting_provider),
        unittest.TestLoader().loadTestsFromModule(test_owselectcolumns),
      ])
    return unittest.TestSuite(all_tests)


test_suite = suite()


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
