import os
import tempfile
import unittest
import Orange.data._io as _io

import numpy as np

from Orange.data import ContinuousVariable, DiscreteVariable
from Orange.data.io import TabDelimReader


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
    def test_scan_simple(self):
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(simple_file.encode("ascii"))
        f.close()
        fname = f.name
        try:
            (attributes, classes, metas, n_attributes, n_classes, n_metas, lines
            ) = _io.sparse_prescan(fname.encode("utf-8"))
            self.assertEqual(attributes,
                {x.encode("ascii") for x in ("abc", "g", "ij k", "t", "def",
                                             "g", "h", "ij", "kl", "m")})
            self.assertEqual(classes, set())
            self.assertEqual(metas, set())
            self.assertEqual(n_attributes, 13)
            self.assertEqual(n_classes, 0)
            self.assertEqual(n_metas, 0)
            self.assertEqual(lines, 3)
        finally:
            os.remove(fname)

    def test_scan_complex(self):
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(complex_file.encode("ascii"))
        f.close()
        fname = f.name
        try:
            (attributes, classes, metas, n_attributes, n_classes, n_metas, lines
            ) = _io.sparse_prescan(fname.encode("utf-8"))
            self.assertEqual(attributes,
                {x.encode("ascii") for x in ("abc", "g", "h", "ij")})
            self.assertEqual(classes, {b"k", b"t", b"kl", b"k"})
            self.assertEqual(metas, {b"m", b"def"})
            self.assertEqual(n_attributes, 7)
            self.assertEqual(n_classes, 4)
            self.assertEqual(n_metas, 2)
            self.assertEqual(lines, 3)
        finally:
            os.remove(fname)


if __name__ == "__main__":
    unittest.main()
