# coding=utf-8
import io
import functools
import os, tempfile
import unittest

import numpy as np

from Orange.data.io import BasketFormat


def with_file(s):
    def fle_decorator(f, s=s):
        @functools.wraps(f)
        def decorated(self, s=s):
            fle = tempfile.NamedTemporaryFile(delete=False)
            fle.write(s.encode("utf-8"))
            fle.close()
            fname = fle.name
            try:
                return f(self, fname)
            finally:
                os.remove(fname)
        return decorated
    return fle_decorator



class TestBasketReader(unittest.TestCase):
    @with_file("""a=1,b=2,c=3""")
    def test_read_variable_is_value_syntax(self, fname):
        table = BasketFormat().read_file(fname)
        self.assertEqual(len(table.domain.variables), 3)
        self.assertEqual(["a", "b", "c"],
                         list(map(lambda x: x.name, table.domain.variables)))
        np.testing.assert_almost_equal(table.X.todense(),
                                       np.array([[1, 2, 3]]))

    @with_file("""a,b,c,d,e""")
    def test_read_variable_only_syntax(self, fname):
        table = BasketFormat().read_file(fname)
        self.assertEqual(len(table.domain.variables), 5)
        np.testing.assert_almost_equal(table.X.todense(),
                                       np.array([[1, 1, 1, 1, 1]]))

    @with_file("""a=1, b=2, c=3""")
    def test_handles_spaces_between_variables(self, fname):
        table = BasketFormat().read_file(fname)
        self.assertEqual(len(table.domain.variables), 3)
        self.assertEqual(set(x for x in table[0]), {1, 2, 3})

    @with_file("""a=1, b=2, a=3""")
    def test_handles_duplicate_variables(self, fname):
        self.assertRaises(ValueError, BasketFormat().read_file, fname)

    @with_file("""a, b, b, a, a, c, c, d, e""")
    def test_handles_duplicate_variables2(self, fname):
        self.assertRaises(ValueError, BasketFormat().read_file, fname)

    @with_file("""a=1, b=2\na=1, b=4""")
    def test_variables_can_be_listed_in_any_order(self, fname):
        table = BasketFormat().read_file(fname)
        self.assertEqual(len(table.domain.variables), 2)
        np.testing.assert_almost_equal(table.X.todense(),
                                       np.array([[1, 2], [1, 4]]))


    @with_file("""a,b\nc,b,a""")
    def test_variables_can_be_listed_in_any_order(self, fname):
        table = BasketFormat().read_file(fname)
        self.assertEqual(len(table.domain.variables), 3)
        np.testing.assert_almost_equal(table.X.todense(),
                                       np.array([[1, 1, 0], [1, 1, 1]]))

    @with_file("""č,š,ž""")
    def test_handles_unicode(self, fname):
        table = BasketFormat().read_file(fname)
        self.assertEqual(len(table.domain.variables), 3)
        np.testing.assert_almost_equal(table.X.todense(),
                                       np.array([[1, 1, 1]]))

    @with_file("""a=4,"x"=1.0,"y"=2.0,b=5\n"x"=1.0""")
    def test_handles_quote(self, fname):
        table = BasketFormat().read_file(fname)
        self.assertEqual(len(table.domain.variables), 4)


if __name__ == "__main__":
    unittest.main()
