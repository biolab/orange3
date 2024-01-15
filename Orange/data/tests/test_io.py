import os
import unittest
from tempfile import NamedTemporaryFile

import numpy as np

from Orange.data import ContinuousVariable, DiscreteVariable, StringVariable, \
    TimeVariable, Domain, Table
from Orange.data.io import TabReader, ExcelReader
from Orange.data.io_util import guess_data_type
from Orange.misc.collections import natural_sorted


class TestTableFilters(unittest.TestCase):
    def test_guess_data_type_continuous(self):
        # should be ContinuousVariable
        valuemap, values, coltype = guess_data_type(list(range(1, 100)))
        self.assertEqual(ContinuousVariable, coltype)
        self.assertIsNone(valuemap)
        np.testing.assert_array_equal(np.array(list(range(1, 100))), values)

        valuemap, values, coltype = guess_data_type([1, 2, 3, 1, 2, 3])
        self.assertEqual(ContinuousVariable, coltype)
        self.assertIsNone(valuemap)
        np.testing.assert_array_equal([1, 2, 3, 1, 2, 3], values)

        valuemap, values, coltype = guess_data_type(
            ["1", "2", "3", "1", "2", "3"])
        self.assertEqual(ContinuousVariable, coltype)
        self.assertIsNone(valuemap)
        np.testing.assert_array_equal([1, 2, 3, 1, 2, 3], values)

    def test_guess_data_type_discrete(self):
        # should be DiscreteVariable
        valuemap, values, coltype = guess_data_type([1, 2, 1, 2])
        self.assertEqual(DiscreteVariable, coltype)
        self.assertEqual([1, 2], valuemap)
        np.testing.assert_array_equal([1, 2, 1, 2], values)

        valuemap, values, coltype = guess_data_type(["1", "2", "1", "2", "a"])
        self.assertEqual(DiscreteVariable, coltype)
        self.assertEqual(["1", "2", "a"], valuemap)
        np.testing.assert_array_equal(['1', '2', '1', '2', 'a'], values)

        # just below the threshold for string variable
        in_values = list(map(lambda x: str(x) + "a", range(24))) + ["a"] * 76
        valuemap, values, coltype = guess_data_type(in_values)
        self.assertEqual(DiscreteVariable, coltype)
        self.assertEqual(natural_sorted(set(in_values)), valuemap)
        np.testing.assert_array_equal(in_values, values)

    def test_guess_data_type_string(self):
        # should be StringVariable
        # too many different values for discrete
        in_values = list(map(lambda x: str(x) + "a", range(90)))
        valuemap, values, coltype = guess_data_type(in_values)
        self.assertEqual(StringVariable, coltype)
        self.assertIsNone(valuemap)
        np.testing.assert_array_equal(in_values, values)

        # more than len(values)**0.7
        in_values = list(map(lambda x: str(x) + "a", range(25))) + ["a"] * 75
        valuemap, values, coltype = guess_data_type(in_values)
        self.assertEqual(StringVariable, coltype)
        self.assertIsNone(valuemap)
        np.testing.assert_array_equal(in_values, values)

        # more than 100 different values - exactly 101
        # this is the case when len(values)**0.7 rule would vote for the
        # DiscreteVariable
        in_values = list(map(lambda x: str(x) + "a", range(100))) + ["a"] * 999
        valuemap, values, coltype = guess_data_type(in_values)
        self.assertEqual(StringVariable, coltype)
        self.assertIsNone(valuemap)
        np.testing.assert_array_equal(in_values, values)

    def test_guess_data_type_time(self):
        in_values = ["2019-10-10", "2019-10-10", "2019-10-10", "2019-10-01"]
        valuemap, _, coltype = guess_data_type(in_values)
        self.assertEqual(TimeVariable, coltype)
        self.assertIsNone(valuemap)

        in_values = ["2019-10-10T12:08:51", "2019-10-10T12:08:51",
                     "2019-10-10T12:08:51", "2019-10-01T12:08:51"]
        valuemap, _, coltype = guess_data_type(in_values)
        self.assertEqual(TimeVariable, coltype)
        self.assertIsNone(valuemap)

        in_values = ["2019-10-10 12:08:51", "2019-10-10 12:08:51",
                     "2019-10-10 12:08:51", "2019-10-01 12:08:51"]
        valuemap, _, coltype = guess_data_type(in_values)
        self.assertEqual(TimeVariable, coltype)
        self.assertIsNone(valuemap)

        in_values = ["2019-10-10 12:08", "2019-10-10 12:08",
                     "2019-10-10 12:08", "2019-10-01 12:08"]
        valuemap, _, coltype = guess_data_type(in_values)
        self.assertEqual(TimeVariable, coltype)
        self.assertIsNone(valuemap)

    def test_guess_data_type_values_order(self):
        """
        Test if values are ordered naturally
        """
        in_values = [
            "something1", "something12", "something2", "something1",
            "something20", "something1", "something2", "something12",
            "something1", "something12"
        ]
        res = ["something1", "something2", "something12", "something20"]
        valuemap, _, coltype = guess_data_type(in_values)
        self.assertEqual(DiscreteVariable, coltype)
        self.assertListEqual(res, valuemap)

class TestWriters(unittest.TestCase):
    def setUp(self):
        self.domain = Domain([DiscreteVariable("a", values=tuple("xyz")),
                         ContinuousVariable("b", number_of_decimals=3)],
                        ContinuousVariable("c", number_of_decimals=0),
                        [StringVariable("d")])
        self.data = Table.from_numpy(
            self.domain,
            np.array([[1, 0.5], [2, np.nan], [np.nan, 1.0625]]),
            np.array([3, 1, 7]),
            np.array(["foo bar baz".split()]).T
        )

    def test_write_tab(self):
        with NamedTemporaryFile(suffix=".tab", delete=False) as f:
            fname = f.name
        try:
            TabReader.write(fname, self.data)
            with open(fname, encoding="utf-8") as f:
                self.assertEqual(f.read().strip(), """
c\td\ta\tb
continuous\tstring\tx y z\tcontinuous
class\tmeta\t\t
3\tfoo\ty\t0.500
1\tbar\tz\t
7\tbaz\t\t1.06250""".strip())
        finally:
            os.remove(fname)

    def test_roundtrip_xlsx(self):
        with NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            fname = f.name
        try:
            ExcelReader.write(fname, self.data)
            data = ExcelReader(fname).read()
            np.testing.assert_equal(data.X, self.data.X)
            np.testing.assert_equal(data.Y, self.data.Y)
            np.testing.assert_equal(data.metas, self.data.metas)
            np.testing.assert_equal(data.domain, self.data.domain)
        finally:
            os.remove(fname)


if __name__ == "__main__":
    unittest.main()
