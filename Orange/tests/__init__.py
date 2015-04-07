import os
import unittest

try:
    from Orange.widgets.tests import test_setting_provider, \
        test_settings_handler, test_context_handler, \
        test_class_values_context_handler, test_domain_context_handler, \
        test_widget
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
        load = unittest.TestLoader().loadTestsFromModule
        all_tests.extend([
            load(test_setting_provider),
            load(test_settings_handler),
            load(test_context_handler),

            load(test_class_values_context_handler),
            load(test_domain_context_handler),
            load(test_owselectcolumns)
        ])
    return unittest.TestSuite(all_tests)


test_suite = suite()

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
