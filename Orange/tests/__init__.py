import os
import unittest

from Orange.widgets.tests import test_settings, test_setting_provider


def suite():
    test_dir = os.path.dirname(__file__)
    return unittest.TestSuite([
        unittest.TestLoader().discover(test_dir),
        unittest.TestLoader().loadTestsFromModule(test_settings),
        unittest.TestLoader().loadTestsFromModule(test_setting_provider),
    ])


test_suite = suite()


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
