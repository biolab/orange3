# pylint: disable=protected-access
import unittest
from unittest.mock import Mock

import numpy as np

from Orange.data import ContinuousVariable, TimeVariable, \
    DiscreteVariable, StringVariable, Table
from Orange.data.io_base import _TableHeader, _TableBuilder, DataTableMixin


class InitTestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header0 = [["0.1", "0.5", "0.1", "21.0"],
                       ["0.2", "0.1", "2.5", "123.0"],
                       ["0.0", "0.0", "0.0", "0.0"]]
        cls.header1 = [["a", "b", "c", "d"],
                       ["red", "2019-10-10", "0.0", "21.0"],
                       ["red", "2019-10-12", "0.0", "123.0"],
                       ["green", "2019-10-11", "", "0.0"]]
        cls.header1_flags = [["m#a", "cC#b", "m#c", "d", "i#e", "f"],
                             ["red", "0.5", "0.0", "0.0", "aa", "a"],
                             ["red", "0.1", "1.0", "1.0", "b", "b"],
                             ["green", "0.0", "2.0", "2.0", "c", "c"]]
        cls.header3 = [["a", "b", "c", "d", "w", "e", "f", "g"],
                       ["d", "c", "c", "c", "c", "d", "s", "yes no"],
                       ["meta", "class", "meta", "", "weight", "i", "", ""],
                       ["red", "0.5", "0.0", "0.0", "0.5", "aa", "a", "no"]]


class TestTableHeader(InitTestData):
    def test_rename_variables(self):
        th = _TableHeader([["a", "", "b", "", "a"]])
        self.assertListEqual(th.names, ["a (1)", "", "b", "", "a (2)"])

    def test_get_header_data_0(self):
        names, types, flags = _TableHeader.create_header_data([])
        self.assertListEqual(names, [])
        self.assertListEqual(types, [])
        self.assertListEqual(flags, [])

    def test_get_header_data_1(self):
        names, types, flags = _TableHeader.create_header_data(self.header1[:1])
        self.assertListEqual(names, ["a", "b", "c", "d"])
        self.assertListEqual(types, ["", "", "", ""])
        self.assertListEqual(flags, ["", "", "", ""])

    def test_get_header_data_1_flags(self):
        names, types, flags = _TableHeader.create_header_data(
            self.header1_flags[:1])
        self.assertListEqual(names, ["a", "b", "c", "d", "e", "f"])
        self.assertListEqual(types, ["", "c", "", "", "", ""])
        self.assertListEqual(flags, ["m", "c", "m", "", "i", ""])

    def test_get_header_data_3(self):
        names, types, flags = _TableHeader.create_header_data(self.header3[:3])
        self.assertListEqual(names, ["a", "b", "c", "d", "w", "e", "f", "g"])
        self.assertListEqual(
            types, ["d", "c", "c", "c", "c", "d", "s", "yes no"])
        self.assertListEqual(
            flags, ["meta", "class", "meta", "", "weight", "i", "", ""])


class TestTableBuilder(InitTestData):
    def test_string_column(self):
        data = np.array(self.header1[1:])
        creator = _TableBuilder._get_column_creator("s")
        column = creator(data, 0)

        self.assertIsNone(column.valuemap)
        np.testing.assert_array_equal(column.values, ["red", "red", "green"])
        np.testing.assert_array_equal(column.orig_values,
                                      ["red", "red", "green"])
        self.assertEqual(column.coltype, StringVariable)
        self.assertDictEqual(column.coltype_kwargs, {})

    def test_continuous_column(self):
        data = np.array(self.header0)
        creator = _TableBuilder._get_column_creator("c")
        column = creator(data, 0)

        self.assertIsNone(column.valuemap)
        np.testing.assert_array_equal(column.values, [0.1, 0.2, 0])
        np.testing.assert_array_equal(column.orig_values,
                                      ["0.1", "0.2", "0.0"])
        self.assertEqual(column.coltype, ContinuousVariable)
        self.assertDictEqual(column.coltype_kwargs, {})

    def test_continuous_column_raises(self):
        data = np.array([["a", "2"], ["3", "4"]])
        creator = _TableBuilder._get_column_creator("c")
        self.assertRaises(ValueError, creator, data, 0)

    def test_time_column(self):
        data = np.array(self.header1[1:])
        creator = _TableBuilder._get_column_creator("t")
        column = creator(data, 1)

        self.assertIsNone(column.valuemap)
        np.testing.assert_array_equal(
            column.values, ['2019-10-10', '2019-10-12', '2019-10-11'])
        np.testing.assert_array_equal(
            column.orig_values, ['2019-10-10', '2019-10-12', '2019-10-11'])
        self.assertEqual(column.coltype, TimeVariable)
        self.assertDictEqual(column.coltype_kwargs, {})

    def test_discrete_column(self):
        data = np.array(self.header1[1:])
        creator = _TableBuilder._get_column_creator("d")
        columns = creator(data, 0)

        self.assertListEqual(columns.valuemap, ['green', 'red'])
        np.testing.assert_array_equal(columns.values, ["red", "red", "green"])
        np.testing.assert_array_equal(columns.orig_values,
                                      ["red", "red", "green"])
        self.assertEqual(columns.coltype, DiscreteVariable)
        self.assertDictEqual(columns.coltype_kwargs, {})

    def test_column_parts_discrete_values(self):
        data = np.array(self.header1[1:])
        vals = "green red"
        creator = _TableBuilder._get_column_creator(vals)
        column = creator(data, 0, vals)

        self.assertListEqual(column.valuemap, ['green', 'red'])
        np.testing.assert_array_equal(column.values, ["red", "red", "green"])
        np.testing.assert_array_equal(column.orig_values,
                                      ["red", "red", "green"])
        self.assertEqual(column.coltype, DiscreteVariable)
        self.assertDictEqual(column.coltype_kwargs, {})

    def test_unknown_type_column(self):
        data = np.array(self.header0)
        creator = _TableBuilder._get_column_creator("")
        column = creator(data, 0)

        self.assertIsNone(column.valuemap)
        np.testing.assert_array_equal(column.values, [0.1, 0.2, 0])
        np.testing.assert_array_equal(column.orig_values, ["0.1", "0.2", "0.0"])
        self.assertEqual(column.coltype, ContinuousVariable)
        self.assertDictEqual(column.coltype_kwargs, {})


class TestDataTableMixin(InitTestData):
    def test_data_table_empty(self):
        self.assertIsInstance(DataTableMixin.data_table([]), Table)

    def test_data_table_0(self):
        self.assertIsInstance(DataTableMixin.data_table(self.header0), Table)

    def test_data_table_1(self):
        self.assertIsInstance(DataTableMixin.data_table(self.header1), Table)

    def test_data_table_1_flags(self):
        self.assertIsInstance(DataTableMixin.data_table(
            self.header1_flags), Table)

    def test_data_table_3(self):
        self.assertIsInstance(DataTableMixin.data_table(self.header3), Table)

    def test_parse_headers_empty(self):
        headers, data = DataTableMixin.parse_headers([])
        self.assertListEqual(headers, [])
        self.assertListEqual(list(data), [])

    def test_parse_headers_0(self):
        hdata = self.header0
        headers, data = DataTableMixin.parse_headers(hdata)
        self.assertListEqual(headers, [])
        self.assertListEqual(list(data), hdata)

    def test_parse_headers_1(self):
        hdata = self.header1
        headers, data = DataTableMixin.parse_headers(hdata)
        self.assertListEqual(headers, [["a", "b", "c", "d"]])
        self.assertListEqual(list(data), hdata[1:])

    def test_parse_headers_1_flags(self):
        hdata = self.header1_flags
        headers, data = DataTableMixin.parse_headers(hdata)
        self.assertListEqual(
            headers, [["m#a", "cC#b", "m#c", "d", "i#e", "f"]])
        self.assertListEqual(list(data), hdata[1:])

    def test_parse_headers_3(self):
        hdata = self.header3
        headers, data = DataTableMixin.parse_headers(hdata)
        self.assertListEqual(
            headers, [["a", "b", "c", "d", "w", "e", "f", "g"],
                      ["d", "c", "c", "c", "c", "d", "s", "yes no"],
                      ["meta", "class", "meta", "", "weight", "i", "", ""]])
        self.assertListEqual(list(data), hdata[3:])

    def test_adjust_data_width_lengthen(self):
        names, types, flags = ["a", "b", "c", "d", "e"], ["", "c"], ["m", "c"]
        header = Mock()
        header.names = names
        header.types = types
        header.flags = flags
        adjusted, n = DataTableMixin.adjust_data_width(self.header0, header)
        _data = np.hstack((np.array(self.header0, dtype=object),
                           np.array([[""]] * 3, dtype=object)))
        np.testing.assert_array_equal(_data, adjusted)
        self.assertEqual(adjusted.shape, (len(self.header0), 5))
        self.assertEqual(n, 5)
        self.assertListEqual(names, ["a", "b", "c", "d", "e"])
        self.assertListEqual(types, ["", "c", "", "", ""])
        self.assertListEqual(flags, ["m", "c", "", "", ""])

    def test_adjust_data_width_shorten(self):
        names, types, flags = ["a", "b", "c"], ["", "c"], ["m", "c"]
        header = Mock()
        header.names = names
        header.types = types
        header.flags = flags
        with self.assertWarns(UserWarning):
            adjusted, n = DataTableMixin.adjust_data_width(self.header0, header)
        np.testing.assert_array_equal(
            adjusted, np.array(self.header0, dtype=object)[:, :3])
        self.assertEqual(adjusted.shape, (len(self.header0), 3))
        self.assertEqual(n, 3)
        self.assertListEqual(names, ["a", "b", "c"])
        self.assertListEqual(types, ["", "c", ""])
        self.assertListEqual(flags, ["m", "c", ""])

    def test_adjust_data_width_empty(self):
        names, types, flags = ["a", "b"], [], []
        header = Mock()
        header.names = names
        header.types = types
        header.flags = flags
        data = [["", ""], ["", ""]]
        adjusted, n = DataTableMixin.adjust_data_width(data, header)
        np.testing.assert_array_equal(adjusted, [])
        self.assertEqual(n, 2)
        self.assertListEqual(names, ["a", "b"])
        self.assertListEqual(types, [])
        self.assertListEqual(flags, [])


if __name__ == '__main__':
    unittest.main()
