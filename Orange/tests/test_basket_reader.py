# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import io
import functools
import os, tempfile
import unittest

import numpy as np

from Orange.data.io import BasketReader


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


def read_basket(filename):
    return BasketReader(filename).read()


class TestBasketReader(unittest.TestCase):
    @with_file("""a=1,b=2,c=3""")
    def test_read_variable_is_value_syntax(self, fname):
        table = read_basket(fname)
        self.assertEqual(len(table.domain.variables), 3)
        self.assertEqual(["a", "b", "c"],
                         list(map(lambda x: x.name, table.domain.variables)))
        np.testing.assert_almost_equal(table.X.todense(),
                                       np.array([[1, 2, 3]]))

    @with_file("""a,b,c,d,e""")
    def test_read_variable_only_syntax(self, fname):
        table = read_basket(fname)
        self.assertEqual(len(table.domain.variables), 5)
        np.testing.assert_almost_equal(table.X.todense(),
                                       np.array([[1, 1, 1, 1, 1]]))

    @with_file("""a=1, b=2, c=3""")
    def test_handles_spaces_between_variables(self, fname):
        table = read_basket(fname)
        self.assertEqual(len(table.domain.variables), 3)
        self.assertEqual(set(x for x in table[0]), {1, 2, 3})

    @with_file("""a=1, b=2, a=3""")
    def test_handles_duplicate_variables(self, fname):
        self.assertRaises(ValueError, read_basket, fname)

    @with_file("""a, b, b, a, a, c, c, d, e""")
    def test_handles_duplicate_variables2(self, fname):
        self.assertRaises(ValueError, read_basket, fname)

    @with_file("""a=1, b=2\na=1, b=4""")
    def test_variables_can_be_listed_in_any_order(self, fname):
        table = read_basket(fname)
        self.assertEqual(len(table.domain.variables), 2)
        np.testing.assert_almost_equal(table.X.todense(),
                                       np.array([[1, 2], [1, 4]]))


    @with_file("""a,b\nc,b,a""")
    def test_variables_can_be_listed_in_any_order(self, fname):
        table = read_basket(fname)
        self.assertEqual(len(table.domain.variables), 3)
        np.testing.assert_almost_equal(table.X.todense(),
                                       np.array([[1, 1, 0], [1, 1, 1]]))

    @with_file("""č,š,ž""")
    def test_handles_unicode(self, fname):
        table = read_basket(fname)
        self.assertEqual(len(table.domain.variables), 3)
        np.testing.assert_almost_equal(table.X.todense(),
                                       np.array([[1, 1, 1]]))

    @with_file("""a=4,"x"=1.0,"y"=2.0,b=5\n"x"=1.0""")
    def test_handles_quote(self, fname):
        table = read_basket(fname)
        self.assertEqual(len(table.domain.variables), 4)

    def test_data_name(self):
        filename = os.path.join(os.path.dirname(__file__), 'iris_basket.basket')
        self.assertEqual(read_basket(filename).name, 'iris_basket')


if __name__ == "__main__":
    unittest.main()
