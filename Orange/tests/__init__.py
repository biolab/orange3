import os
import unittest

from Orange.widgets.tests import test_setting_provider, \
    test_settings_handler, test_context_handler, \
    test_class_values_context_handler, test_domain_context_handler
from Orange.widgets.data.tests import test_owselectcolumns

try:
    from Orange.widgets.tests import test_widget

    run_widget_tests = True
except ImportError:
    run_widget_tests = False


def suite():
    test_dir = os.path.dirname(__file__)
    all_tests = [
        unittest.TestLoader().discover(test_dir),
    ]
    load = unittest.TestLoader().loadTestsFromModule
    all_tests.extend([
        load(test_setting_provider),
        load(test_settings_handler),
        load(test_context_handler),

        load(test_class_values_context_handler),
        load(test_domain_context_handler),
        load(test_owselectcolumns)
    ])
    if run_widget_tests:
        all_tests.extend([
            #load(test_widget), # does not run on travis
        ])
    return unittest.TestSuite(all_tests)


test_suite = suite()

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
