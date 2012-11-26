import io
import unittest

import numpy as np

from Orange.data.io import BasketReader


class TestBasketReader(unittest.TestCase):
    def test_read_variable_is_value_syntax(self):
        file = io.StringIO("""a=1,b=2,c=3""")
        table = BasketReader()._read_file(file)

        self.assertEqual(len(table.domain.variables), 3)
        self.assertEqual(["a", "b", "c"], list(map(lambda x: x.name, table.domain.variables)))
        np.testing.assert_almost_equal(table.X.todense(), np.array([[1, 2, 3]]))

    def test_read_variable_only_syntax(self):
        file = io.StringIO("""a,b,c,d,e""")
        table = BasketReader()._read_file(file)
        self.assertEqual(len(table.domain.variables), 5)
        np.testing.assert_almost_equal(table.X.todense(), np.array([[1, 1, 1, 1, 1]]))

    def test_handles_spaces_between_variables(self):
        file = io.StringIO("""a=1, b=2, c=3""")
        table = BasketReader()._read_file(file)

        self.assertEqual(len(table.domain.variables), 3)
        np.testing.assert_almost_equal(table.X.todense(), np.array([[1, 2, 3]]))

    def test_handles_duplicate_variables(self):
        file = io.StringIO("""a=1, b=2, a=3""")
        with self.assertWarns(UserWarning):
            table = BasketReader()._read_file(file)

        self.assertEqual(len(table.domain.variables), 2)
        np.testing.assert_almost_equal(table.X.todense(), np.array([[3, 2]]))

    def test_handles_duplicate_variables2(self):
        file = io.StringIO("""a, b, b, a, a, c, c, d, e""")
        with self.assertWarns(UserWarning):
            table = BasketReader()._read_file(file)

        self.assertEqual(len(table.domain.variables), 5)
        np.testing.assert_almost_equal(table.X.todense(), np.array([[1, 1, 1, 1, 1]]))

    def test_variables_can_be_listed_in_any_order(self):
        file = io.StringIO("""a=1, b=2\na=1, b=4""")
        table = BasketReader()._read_file(file)

        self.assertEqual(len(table.domain.variables), 2)
        np.testing.assert_almost_equal(table.X.todense(), np.array([[1, 2], [1, 4]]))

    def test_variables_can_be_listed_in_any_order(self):
        file = io.StringIO("""a=1, b=2\na=1, b=4""")
        table = BasketReader()._read_file(file)

        self.assertEqual(len(table.domain.variables), 2)
        np.testing.assert_almost_equal(table.X.todense(), np.array([[1, 2], [1, 4]]))

    def test_variables_can_be_listed_in_any_order(self):
        file = io.StringIO("""a,b\nc,b,a""")
        table = BasketReader()._read_file(file)

        self.assertEqual(len(table.domain.variables), 3)
        np.testing.assert_almost_equal(table.X.todense(), np.array([[1, 1, 0], [1, 1, 1]]))

    def test_handles_unicode(self):
        file = io.StringIO("""č,š,ž""")
        table = BasketReader()._read_file(file)

        self.assertEqual(len(table.domain.variables), 3)
        np.testing.assert_almost_equal(table.X.todense(), np.array([[1, 1, 1]]))


if __name__ == "__main__":
    unittest.main()
