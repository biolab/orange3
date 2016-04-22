import os
import unittest
from contextlib import contextmanager

try:
    from Orange.widgets.tests import test_setting_provider, \
        test_settings_handler, test_context_handler, \
        test_class_values_context_handler, test_domain_context_handler, \
        test_perfect_domain_context_handler, \
        test_owselectcolumns, test_scatterplot_density, test_widgets_outputs
    run_widget_tests = True
except ImportError:
    run_widget_tests = False


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
    if loader is None:
        loader = unittest.TestLoader()
    if pattern is None:
        pattern = 'test*.py'
    all_tests = [
        loader.discover(test_dir, pattern),
    ]
    load = loader.loadTestsFromModule

    if run_widget_tests:
        all_tests.extend([
            load(test_setting_provider),
            load(test_settings_handler),
            load(test_context_handler),
            load(test_class_values_context_handler),
            load(test_domain_context_handler),
            load(test_perfect_domain_context_handler),

            load(test_owselectcolumns),
            load(test_scatterplot_density),
            load(test_widgets_outputs),
        ])
    return unittest.TestSuite(all_tests)


test_suite = suite()


def load_tests(loader, tests, pattern):
    return suite(loader, pattern)


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
