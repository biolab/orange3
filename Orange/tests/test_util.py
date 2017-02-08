import os
import unittest
import warnings

import numpy as np

from Orange.util import export_globals, flatten, deprecated, try_, deepgetattr, \
    OrangeDeprecationWarning

SOMETHING = 0xf00babe


class TestUtil(unittest.TestCase):
    def test_export_globals(self):
        self.assertEqual(sorted(export_globals(globals(), __name__)),
                         ['SOMETHING', 'TestUtil'])

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

    def test_try_(self):
        self.assertTrue(try_(lambda: np.ones(3).any()))
        self.assertFalse(try_(lambda: np.whatever()))
        self.assertEqual(try_(len, default=SOMETHING), SOMETHING)

    def test_reprable(self):
        from Orange.data import ContinuousVariable
        from Orange.preprocess.impute import ReplaceUnknownsRandom
        from Orange.statistics.distribution import Continuous

        var = ContinuousVariable('x')
        transform = ReplaceUnknownsRandom(var, Continuous(1, var))

        self.assertEqual(repr(transform).replace('\n       ', ' '),
                         "ReplaceUnknownsRandom("
                         "variable=ContinuousVariable(name='x', number_of_decimals=3), "
                         "distribution=Continuous([[ 0.], [ 0.]]))")

    def test_deepgetattr(self):
        class a:
            l = []
        self.assertTrue(deepgetattr(a, 'l.__len__.__call__'), a.l.__len__.__call__)
        self.assertTrue(deepgetattr(a, 'l.__nx__.__x__', 42), 42)
        self.assertRaises(AttributeError, lambda: deepgetattr(a, 'l.__nx__.__x__'))

    @unittest.skipUnless(os.environ.get('ORANGE_DEPRECATIONS_ERROR'),
                         'ORANGE_DEPRECATIONS_ERROR not set')
    def test_raise_deprecations(self):
        with self.assertRaises(OrangeDeprecationWarning):
            warnings.warn('foo', OrangeDeprecationWarning)
