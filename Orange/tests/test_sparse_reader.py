# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import os
import tempfile
import unittest

import numpy as np

from Orange.data import _io


simple_file = """\
abc, def, g=1, h ,  ij k  =5,   t # ignore this, foo=42

def  , g   , h,ij,kl=4,m,,,
# nothing here
\t\t\tdef
"""

complex_file = """\
abc, g=1, h ,  ij | k  =5,   t # ignore this, foo=42

, g   , h,ij|,kl=4, k ;m,,,
# nothing here
\t\t\t;def
"""

class TestTabReader(unittest.TestCase):
    def test_read_simple(self):
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(simple_file.encode("ascii"))
        f.close()
        try:
            X, Y, metas, attr_indices, class_indices, meta_indices = \
                _io.sparse_read_float(f.name.encode("ascii"))

            self.assertEqual(
                attr_indices,
                {b"abc": 0, b"def": 1, b"g": 2, b"h": 3, b"ij k": 4, b"t": 5,
                 b"ij": 6, b"kl": 7, b"m": 8})
            np.testing.assert_almost_equal(X.data, [1, 1, 1, 1, 5, 1,
                                                    1, 1, 1, 1, 4, 1,
                                                    1])
            np.testing.assert_equal(X.indices, [0, 1, 2, 3, 4, 5,
                                                1, 2, 3, 6, 7, 8,
                                                1])
            np.testing.assert_equal(X.indptr, [0, 6, 12, 13])

            self.assertEqual(class_indices, {})
            self.assertIsNone(Y)

            self.assertEqual(meta_indices, {})
            self.assertIsNone(metas)
        finally:
            os.remove(f.name)


    # for readability, pylint: disable=bad-whitespace
    def test_read_complex(self):
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(complex_file.encode("ascii"))
        f.close()
        try:
            X, Y, metas, attr_indices, class_indices, meta_indices = \
                _io.sparse_read_float(f.name.encode("ascii"))

            self.assertEqual(
                attr_indices,
                {b"abc": 0, b"g": 1, b"h": 2, b"ij": 3})
            np.testing.assert_equal(X.data,    [1, 1, 1, 1, 1, 1, 1])
            np.testing.assert_equal(X.indices, [0, 1, 2, 3, 1, 2, 3])
            np.testing.assert_equal(X.indptr,  [0,          4,       7, 7])


            self.assertEqual(class_indices, {b"k": 0, b"t": 1, b"kl": 2})
            np.testing.assert_equal(Y.data,    [5, 1, 1, 4])
            np.testing.assert_equal(Y.indices, [0, 1, 0, 2])
            np.testing.assert_equal(Y.indptr,  [0,    2,   4, 4])

            self.assertEqual(meta_indices, {b"m": 0, b"def": 1})
            np.testing.assert_equal(metas.data,    [   1, 1])
            np.testing.assert_equal(metas.indices, [   0, 1])
            np.testing.assert_equal(metas.indptr,  [0, 0, 1, 2])
        finally:
            os.remove(f.name)


    # TODO checks for quotes, escapes, error checking

if __name__ == "__main__":
    unittest.main()
