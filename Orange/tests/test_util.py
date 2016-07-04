import unittest

import numpy as np

from Orange.util import export_globals, flatten, deprecated, try_


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
