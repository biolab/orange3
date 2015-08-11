import unittest

import numpy as np

from Orange.util import *


SOMETHING = 0xf00babe


class UtilTest(unittest.TestCase):
    def test_scale(self):
        self.assertTrue(np.all(scale([0, 1, 2], -1, 1) == [-1, 0, 1]))
        self.assertTrue(np.all(scale([3, 3, 3]) == [1, 1, 1]))
        self.assertTrue(not np.all(np.isnan(scale([.5, np.nan]))))

    def test_abstract(self):
        @abstract
        class AbstractClass: pass

        class ClassWithAbstractMethod:
            @abstract
            def method(self): pass

        with self.assertRaises(NotImplementedError) as cm: AbstractClass()
        self.assertRegex(cm.exception.args[0], 'AbstractClass')
        with self.assertRaises(NotImplementedError) as cm: ClassWithAbstractMethod().method()
        self.assertRegex(cm.exception.args[0], 'ClassWithAbstractMethod')
        self.assertRegex(cm.exception.args[0], 'method')

    def test_export_globals(self):
        self.assertEqual(sorted(export_globals(globals(), __name__)),
                         ['SOMETHING', 'UtilTest'])

    def test_flatten(self):
        self.assertEqual(list(flatten([[1,2],[3]])), [1,2,3])

    def test_deprecated(self):
        @deprecated
        def identity(x): return x

        with self.assertLogs() as cm: x = identity(10)
        self.assertTrue(x == 10)
        self.assertTrue('deprecated' in cm.output[0])
        self.assertTrue('identity' in cm.output[0])
