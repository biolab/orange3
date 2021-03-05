# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

from Orange.data.sql.table import SqlTable
from Orange.data import filter, domain, Instance
from Orange.tests.sql.base import DataBaseTest as dbt


class TestIsDefinedSql(unittest.TestCase, dbt):
    def setUpDB(self):
        self.data = [
            [1, 2, 3, None, 'm'],
            [2, 3, 1, 4, 'f'],
            [None, None, None, None, None],
            [7, None, 3, None, 'f'],
        ]
        conn, self.table_name = self.create_sql_table(self.data)
        self.table = SqlTable(conn, self.table_name, inspect_values=True)

    def tearDownDB(self):
        self.drop_sql_table(self.table_name)

    @dbt.run_on(["postgres", "mssql"])
    def test_on_all_columns(self):
        filtered_data = filter.IsDefined()(self.table)
        correct_data = [row for row in self.data if all(row)]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres", "mssql"])
    def test_selected_columns(self):
        filtered_data = filter.IsDefined(columns=[0])(self.table)
        correct_data = [row for row in self.data if row[0]]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_all_columns_negated(self):
        filtered_data = filter.IsDefined(negate=True)(self.table)
        correct_data = [row for row in self.data if not all(row)]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres", "mssql"])
    def test_selected_columns_negated(self):
        filtered_data = \
            filter.IsDefined(negate=True, columns=[4])(self.table)
        correct_data = [row for row in self.data if not row[4]]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_can_inherit_is_defined_filter(self):
        filtered_data = filter.IsDefined(columns=[1])(self.table)
        filtered_data = filtered_data[:, 4]
        correct_data = [[row[4]]for row in self.data if row[1]]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)


class TestHasClass(unittest.TestCase, dbt):
    def setUpDB(self):
        self.data = [
            [1, 2, 3, None, 'm'],
            [2, 3, 1, 4, 'f'],
            [None, None, None, None, None],
            [7, None, 3, None, 'f'],
        ]
        self.conn, self.table_name = self.create_sql_table(self.data)
        table = SqlTable(self.conn, self.table_name, inspect_values=True)
        variables = table.domain.variables
        new_table = table.copy()
        new_table.domain = domain.Domain(variables[:-1], variables[-1:])
        self.table = new_table

    def tearDownDB(self):
        self.drop_sql_table(self.table_name)

    @dbt.run_on(["postgres", "mssql"])
    def test_has_class(self):
        filtered_data = filter.HasClass()(self.table)
        correct_data = [row for row in self.data if row[-1]]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres", "mssql"])
    def test_negated(self):
        filtered_data = filter.HasClass(negate=True)(self.table)
        correct_data = [row for row in self.data if not row[-1]]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)


class TestSameValueSql(unittest.TestCase, dbt):
    def setUpDB(self):
        self.data = [
            [1, 2, 3, 'a', 'm'],
            [2, None, 1, 'a', 'f'],
            [None, 3, 1, 'b', None],
            [2, 2, 3, 'b', 'f'],
        ]
        self.conn, self.table_name = self.create_sql_table(self.data)
        self.table = SqlTable(self.conn, self.table_name, inspect_values=True)

    def tearDownDB(self):
        self.drop_sql_table(self.table_name)

    @dbt.run_on(["postgres", "mssql"])
    def test_on_continuous_attribute(self):
        filtered_data = filter.SameValue(0, 1)(self.table)
        correct_data = [row for row in self.data if row[0] == 1]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres", "mssql"])
    def test_on_continuous_attribute_with_unknowns(self):
        filtered_data = filter.SameValue(1, 2)(self.table)
        correct_data = [row for row in self.data if row[1] == 2]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres", "mssql"])
    def test_on_continuous_attribute_with_unknown_value(self):
        filtered_data = filter.SameValue(1, None)(self.table)
        correct_data = [row for row in self.data if row[1] is None]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_on_continuous_attribute_negated(self):
        filtered_data = filter.SameValue(0, 1, negate=True)(self.table)
        correct_data = [row for row in self.data if not row[0] == 1]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres", "mssql"])
    def test_on_discrete_attribute(self):
        filtered_data = filter.SameValue(3, 'a')(self.table)
        correct_data = [row for row in self.data if row[3] == 'a']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres", "mssql"])
    def test_on_discrete_attribute_with_unknown_value(self):
        filtered_data = filter.SameValue(4, None)(self.table)
        correct_data = [row for row in self.data if row[4] is None]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres", "mssql"])
    def test_on_discrete_attribute_with_unknowns(self):
        filtered_data = filter.SameValue(4, 'm')(self.table)
        correct_data = [row for row in self.data if row[4] == 'm']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres", "mssql"])
    def test_on_discrete_attribute_negated(self):
        filtered_data = filter.SameValue(3, 'a', negate=True)(self.table)
        correct_data = [row for row in self.data if not row[3] == 'a']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres", "mssql"])
    def test_on_discrete_attribute_value_passed_as_int(self):
        values = self.table.domain[3].values
        filtered_data = filter.SameValue(3, 0, negate=True)(self.table)
        correct_data = [row for row in self.data if not row[3] == values[0]]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres", "mssql"])
    def test_on_discrete_attribute_value_passed_as_float(self):
        values = self.table.domain[3].values
        filtered_data = filter.SameValue(3, 0., negate=True)(self.table)
        correct_data = [row for row in self.data if not row[3] == values[0]]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)


class TestValuesSql(unittest.TestCase, dbt):
    def setUpDB(self):
        self.data = [
            [1, 2, 3, 'a', 'm'],
            [2, None, 1, 'a', 'f'],
            [None, 3, 1, 'b', None],
            [2, 2, 3, 'b', 'f'],
        ]
        conn, self.table_name = self.create_sql_table(self.data)
        self.table = SqlTable(conn, self.table_name, inspect_values=True)

    def tearDownDB(self):
        self.drop_sql_table(self.table_name)

    @dbt.run_on(["postgres", "mssql"])
    def test_values_filter_with_no_conditions(self):
        with self.assertRaises(ValueError):
            filter.Values([])(self.table)

    @dbt.run_on(["postgres", "mssql"])
    def test_discrete_value_filter(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterDiscrete(3, ['a'])
        ])(self.table)
        correct_data = [row for row in self.data if row[3] in ['a']]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_discrete_value_filter_with_multiple_values(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterDiscrete(3, ['a', 'b'])
        ])(self.table)
        correct_data = [row for row in self.data if row[3] in ['a', 'b']]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_discrete_value_filter_with_None(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterDiscrete(3, None)
        ])(self.table)
        correct_data = [row for row in self.data if row[3] is not None]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres", "mssql"])
    def test_continuous_value_filter_equal(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterContinuous(0, filter.FilterContinuous.Equal, 1)
        ])(self.table)
        correct_data = [row for row in self.data if row[0] == 1]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_continuous_value_filter_not_equal(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterContinuous(0, filter.FilterContinuous.NotEqual, 1)
        ])(self.table)
        correct_data = [row for row in self.data if row[0] != 1]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres", "mssql"])
    def test_continuous_value_filter_less(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterContinuous(0, filter.FilterContinuous.Less, 2)
        ])(self.table)
        correct_data = [row for row in self.data
                        if row[0] is not None and row[0] < 2]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_continuous_value_filter_less_equal(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterContinuous(0, filter.FilterContinuous.LessEqual, 2)
        ])(self.table)
        correct_data = [row for row in self.data
                        if row[0] is not None and row[0] <= 2]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_continuous_value_filter_greater(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterContinuous(0, filter.FilterContinuous.Greater, 1)
        ])(self.table)
        correct_data = [row for row in self.data
                        if row[0] is not None and row[0] > 1]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_continuous_value_filter_greater_equal(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterContinuous(0, filter.FilterContinuous.GreaterEqual, 1)
        ])(self.table)
        correct_data = [row for row in self.data
                        if row[0] is not None and row[0] >= 1]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_continuous_value_filter_between(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterContinuous(0, filter.FilterContinuous.Between, 1, 2)
        ])(self.table)
        correct_data = [row for row in self.data
                        if row[0] is not None and 1 <= row[0] <= 2]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres", "mssql"])
    def test_continuous_value_filter_outside(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterContinuous(0, filter.FilterContinuous.Outside, 2, 3)
        ])(self.table)
        correct_data = [row for row in self.data
                        if row[0] is not None and not 2 <= row[0] <= 3]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_continuous_value_filter_isdefined(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterContinuous(1, filter.FilterContinuous.IsDefined)
        ])(self.table)
        correct_data = [row for row in self.data if row[1] is not None]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)


class TestFilterStringSql(unittest.TestCase, dbt):
    def setUpDB(self):
        self.data = [
            [w] for w in "Lorem ipsum dolor sit amet, consectetur adipiscing"
            "elit. Vestibulum vel dolor nulla. Etiam elit lectus, mollis nec"
            "mattis sed, pellentesque in turpis. Vivamus non nisi dolor. Etiam"
            "lacinia dictum purus, in ullamcorper ante vulputate sed. Nullam"
            "congue blandit elementum. Donec blandit laoreet posuere. Proin"
            "quis augue eget tortor posuere mollis. Fusce vestibulum bibendum"
            "neque at convallis. Donec iaculis risus volutpat malesuada"
            "vehicula. Ut cursus tempor massa vulputate lacinia. Pellentesque"
            "eu tortor sed diam placerat porttitor et volutpat risus. In"
            "vulputate rutrum lacus ac sagittis. Suspendisse interdum luctus"
            "sem auctor commodo.".split(' ')] + [[None], [None]]
        self.conn, self.table_name = self.create_sql_table(self.data)
        self.table = SqlTable(self.conn, self.table_name)

    def tearDownDB(self):
        self.drop_sql_table(self.table_name)

    @dbt.run_on(["postgres"])
    def test_filter_string_is_defined(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.IsDefined)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data if row[0] is not None]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres", "mssql"])
    def test_filter_string_equal(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.Equal, 'in')
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data if row[0] == 'in']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_equal_case_insensitive_value(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.Equal, 'In',
                                case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data if row[0] == 'in']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_equal_case_insensitive_data(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.Equal, 'donec',
                                case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data if row[0] == 'Donec']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_not_equal(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.NotEqual, 'in')
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data if row[0] != 'in']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_not_equal_case_insensitive_value(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.NotEqual, 'In',
                                case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data if row[0] != 'in']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_not_equal_case_insensitive_data(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.NotEqual, 'donec',
                                case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data if row[0] != 'Donec']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres", "mssql"])
    def test_filter_string_less(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.Less, 'A')
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and row[0] < 'A']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_less_case_insensitive_value(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.Less, 'In',
                                case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and row[0].lower() < 'in']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_less_case_insensitive_data(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.Less, 'donec',
                                case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and row[0].lower() < 'donec']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres", "mssql"])
    def test_filter_string_less_equal(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.LessEqual, 'A')
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and row[0] <= 'A']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_less_equal_case_insensitive_value(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.LessEqual, 'In',
                                case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and row[0].lower() <= 'in']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_less_equal_case_insensitive_data(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.LessEqual, 'donec',
                                case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and row[0].lower() <= 'donec']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres", "mssql"])
    def test_filter_string_greater(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.Greater, 'volutpat')
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and row[0] > 'volutpat']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_greater_case_insensitive_value(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.Greater, 'In',
                                case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and row[0].lower() > 'in']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_greater_case_insensitive_data(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.Greater, 'donec',
                                case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and row[0].lower() > 'donec']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_greater_equal(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.GreaterEqual, 'volutpat')
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and row[0] >= 'volutpat']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_greater_equal_case_insensitive_value(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.GreaterEqual, 'In',
                                case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and row[0].lower() >= 'in']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_greater_equal_case_insensitive_data(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.GreaterEqual, 'donec',
                                case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and row[0].lower() >= 'donec']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_between(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.Between, 'a', 'c')
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and 'a' <= row[0] <= 'c']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_between_case_insensitive_value(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.Between, 'I', 'O',
                                case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and 'i' < row[0].lower() <= 'o']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_between_case_insensitive_data(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.Between, 'i', 'O',
                                case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and 'i' <= row[0].lower() <= 'o']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_contains(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.Contains, 'et')
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and 'et' in row[0]]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_contains_case_insensitive_value(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.Contains, 'eT',
                                case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and 'et' in row[0].lower()]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_contains_case_insensitive_data(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.Contains, 'do',
                                case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and 'do' in row[0].lower()]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_outside(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.Outside, 'am', 'di')
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and not 'am' < row[0] < 'di']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_outside_case_insensitive(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.Outside, 'd', 'k',
                                case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and not 'd' < row[0].lower() < 'k']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_starts_with(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.StartsWith, 'D')
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and row[0].startswith('D')]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_starts_with_case_insensitive(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.StartsWith, 'D',
                                case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None
                        and row[0].lower().startswith('d')]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_ends_with(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.EndsWith, 's')
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None and row[0].endswith('s')]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_ends_with_case_insensitive(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterString(-1, filter.FilterString.EndsWith, 'S',
                                case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data
                        if row[0] is not None
                        and row[0].lower().endswith('s')]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_list(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterStringList(-1, ['et', 'in'])
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data if row[0] in ['et', 'in']]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres"])
    def test_filter_string_list_case_insensitive_value(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterStringList(-1, ['Et', 'In'], case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data if row[0] in ['et', 'in']]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    @dbt.run_on(["postgres", "mssql"])
    def test_filter_string_list_case_insensitive_data(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterStringList(-1, ['donec'], case_sensitive=False)
        ])(self.table)
        correct_data = [Instance(filtered_data.domain, row)
                        for row in self.data if row[0] in ['Donec']]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)


if __name__ == '__main__':
    unittest.main()
