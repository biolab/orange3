# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

from PyQt4 import QtGui

from Orange import options
from Orange.wrappers import BaseWrapper, WrappersMix


class DumpObj:
    def __init__(self, *args, **kwargs):
        pass


class TestMetaclass(unittest.TestCase):

    def test_options_are_properties(self):
        opts = (
            options.StringOption(name='string'),
            options.BoolOption(name='boolean'),
            options.IntegerOption(name='integer'),
            options.FloatOption(name='fl'),
        )

        class Wrapper(BaseWrapper):
            options = opts
            __wraps__ = DumpObj

        for opt in opts:
            self.assertIsInstance(getattr(Wrapper, opt.name), property)

    def test_set_option(self):
        class Wrapper(BaseWrapper):
            __wraps__ = DumpObj
            number = options.IntegerOption()

            options = (
                options.StringOption(name='test'),
            )

        w = Wrapper()
        w.test = 'test'
        w.number = 2
        self.assertEqual(w.test, 'test')
        self.assertEqual(w._values['test'].value, 'test')

        self.assertEqual(w.number, 2)
        self.assertEqual(w._values['number'].value, 2)

    def test_bad_options(self):
        with self.assertRaises(TypeError):
            class BadWrapper(BaseWrapper):
                options = (
                    1, 1
                )


class TestWrapper(unittest.TestCase):

    def test_init(self):
        class Wrapper(BaseWrapper):
            options = (
                options.IntegerOption(name='int_value'),
            )
            __wraps__ = DumpObj

        w = Wrapper(int_value=5)
        self.assertEqual(w.int_value, 5)

        self.assertRaises(TypeError, Wrapper, unknown_value=0)

        w2 = Wrapper(int_value=3)
        self.assertNotEqual(w.int_value, w2.int_value)

    def test_validation(self):
        def positive_validator(value):
            if value < 0:
                raise options.ValidationError

        class Wrapper(BaseWrapper):
            options = (
                options.FloatOption(name='fl', validator=positive_validator),
            )
            __wraps__ = DumpObj

        w = Wrapper()
        w.fl = -1
        with self.assertRaises(options.ValidationError):
            w.apply_changes()

    def test_share_options(self):
        class Wrapper1(BaseWrapper):
            options = (options.FloatOption(name='shared'), )
            __wraps__ = DumpObj

        class Wrapper2(Wrapper1):
            options = Wrapper1.options

        w1 = Wrapper1(shared=1.)
        w2 = Wrapper2(shared=-1.)
        self.assertNotEqual(w1.shared, w2.shared)

        w2.share_values(w1)
        self.assertEqual(w1.shared, w2.shared)

        w2.shared = 4.
        self.assertEqual(w1.shared, w2.shared)

    def test_repr(self):
        class Wrapper(BaseWrapper):
            options = (
                options.StringOption('arg'),
            )
            __wraps__ = DumpObj
            name = 'test'
        w = Wrapper(arg='text')
        self.assertEqual(repr(w), "DumpObj(arg='text')")
        self.assertIn('test', str(w).lower())

    def test_callback(self):
        class Wrapper(BaseWrapper):
            options = (
                options.StringOption('opt'),
            )
            __wraps__ = DumpObj

        def callback():
            raise AssertionError()

        w = Wrapper()
        w.callback = callback
        with self.assertRaises(AssertionError):
            w.opt = '1'


class TestWrappersGui(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.qApp = QtGui.QApplication([])

    @classmethod
    def tearDownClass(cls):
        cls.qApp.quit()

    def test_layout(self):
        class Wrapper(BaseWrapper):
            options = (
                options.StringOption('opt'),
            )
            __wraps__ = DumpObj

        w = Wrapper()
        self.assertIsInstance(w.options_layout(), QtGui.QLayout)

        class Wrapper2(BaseWrapper):
            options = (
                options.FloatOption('fl'),
                options.StringOption('text')
            )

        mixin = WrappersMix([Wrapper(), Wrapper2()])
        self.assertIsInstance(mixin.options_layout(), QtGui.QLayout)
