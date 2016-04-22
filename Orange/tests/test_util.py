import unittest

import numpy as np

from Orange.util import scale, export_globals, flatten, abstract, deprecated


SOMETHING = 0xf00babe


class UtilTest(unittest.TestCase):
    def test_scale(self):
        self.assertTrue(np.all(scale([0, 1, 2], -1, 1) == [-1, 0, 1]))
        self.assertTrue(np.all(scale([3, 3, 3]) == [1, 1, 1]))
        self.assertTrue(not np.all(np.isnan(scale([.5, np.nan]))))

    def test_abstract(self):
        @abstract
        class AbstractClass:
            pass

        @abstract
        class AbstractWithParams:
            def __init__(self, x):
                self.x = x

        @abstract
        class AbstractWithNew:
            def __new__(cls, *args, **kwargs):
                return super().__new__(cls)

        class DerivedClass(AbstractWithParams):
            def __init__(self, x, y):
                super().__init__(x)
                self.y = y

        class Class:
            @abstract
            def method(self): pass

            @staticmethod
            @abstract
            def staticmethod_(arg): pass

            @classmethod
            @abstract
            def classmethod_(cls): pass

            @property
            @abstract
            def property_(self): pass

        def invalid_order():
            class Invalid:
                @abstract      # This way reads nicer,
                @staticmethod  # but it doesn't work
                def non_method_descriptor(arg): pass

        for AbsClass in (AbstractClass, AbstractWithNew, AbstractWithParams):
            with self.assertRaises(NotImplementedError) as cm:
                AbsClass()
            self.assertRegex(cm.exception.args[0], AbsClass.__name__)

        self.assertIsInstance(DerivedClass(1, 2), DerivedClass)

        for attr in ('method',
                     'staticmethod_',
                     'classmethod_',
                     'property_'):
            with self.assertRaises(NotImplementedError) as cm:
                getattr(Class(), attr)()
            self.assertRegex(cm.exception.args[0], 'Class')
            self.assertRegex(cm.exception.args[0], attr)

        with self.assertRaises(TypeError) as cm:
            invalid_order()
        self.assertRegex(cm.exception.args[0], '@abstract')


    def test_export_globals(self):
        self.assertEqual(sorted(export_globals(globals(), __name__)),
                         ['SOMETHING', 'UtilTest'])

    def test_flatten(self):
        self.assertEqual(list(flatten([[1, 2], [3]])), [1, 2, 3])

    def test_deprecated(self):
        @deprecated
        def identity(x): return x

        with self.assertWarns(DeprecationWarning) as cm:
            x = identity(10)
        self.assertEqual(x, 10)
        self.assertTrue('deprecated' in cm.warning.args[0])
        self.assertTrue('identity' in cm.warning.args[0])
