# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import unittest

from Orange import options
from PyQt4 import QtGui


class BaseTestCase(unittest.TestCase):
    @property
    def options(self):
        return (
            (options.ObjectOption('obj'), 1),
            (options.StringOption('string'), 'text'),
            (options.BoolOption('boolean', default=True), False),
            (options.IntegerOption('integer', default=0), -1),
            (options.FloatOption('fl', default=.0), -1.),
        )


class TestOptionsGui(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        cls.qApp = QtGui.QApplication([], True)

    @classmethod
    def tearDownClass(cls):
        cls.qApp.quit()

    def test_layout(self):
        layout = QtGui.QGridLayout()
        for option, new_value in self.options:
            row = layout.rowCount()
            value = option(option.default)
            value.add_to_layout(layout)
            self.assertGreater(layout.rowCount(), row)


class TestOptions(BaseTestCase):

    def test_validate(self):
        def length_validator(line):
            if len(line) > 4:
                raise options.ValidationError()

        opt = options.StringOption(validator=length_validator)
        value = opt()
        value.value = 'long string'
        self.assertRaises(options.ValidationError, value.validate)

    def test_verbose_name(self):
        opt = options.StringOption('name')
        self.assertIn('name', opt.verbose_name.lower())
        opt.verbose_name = 'title'
        self.assertNotIn('name', opt.verbose_name.lower())

    def test_callback(self):

        def callback():
            callback.c += 1
        callback.c = 0

        for option, new_value in self.options:
            value = option(option.default)
            value.add_callback(callback)
            prev_value = callback.c
            value.value = new_value
            self.assertGreater(callback.c, prev_value)
