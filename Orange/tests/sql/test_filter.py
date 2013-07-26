from Orange.data.sql.table import SqlTable
from Orange.data import filter, domain

from Orange.tests.sql.base import PostgresTest


class IsDefinedFilterTests(PostgresTest):
    def setUp(self):
        self.data = [
            [1, 2, 3, None, 'm'],
            [2, 3, 1, 4, 'f'],
            [None, None, None, None, None],
            [7, None, 3, None, 'f'],
        ]
        self.table_uri = self.create_sql_table(self.data)
        self.table = SqlTable(self.table_uri)

    def tearDown(self):
        self.table.backend.connection.close()

    def test_on_all_columns(self):
        filtered_data = filter.IsDefined()(self.table)
        correct_data = [row for row in self.data if all(row)]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_selected_columns(self):
        filtered_data = filter.IsDefined(columns=[0])(self.table)
        correct_data = [row for row in self.data if row[0]]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_all_columns_negated(self):
        filtered_data = filter.IsDefined(negate=True)(self.table)
        correct_data = [row for row in self.data if not all(row)]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_selected_columns_negated(self):
        filtered_data = \
            filter.IsDefined(negate=True, columns=[4])(self.table)
        correct_data = [row for row in self.data if not row[4]]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_can_inherit_is_defined_filter(self):
        filtered_data = filter.IsDefined(columns=[1])(self.table)
        filtered_data = filtered_data[:, 4]
        correct_data = [[row[4]]for row in self.data if row[1]]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)


class HasClassFilterTests(PostgresTest):
    def setUp(self):
        self.data = [
            [1, 2, 3, None, 'm'],
            [2, 3, 1, 4, 'f'],
            [None, None, None, None, None],
            [7, None, 3, None, 'f'],
        ]
        self.table_uri = self.create_sql_table(self.data)
        table = SqlTable(self.table_uri)
        variables = table.domain.variables
        new_table = table.copy()
        new_table.domain = domain.Domain(variables[:-1], variables[-1:])
        self.table = new_table

    def tearDown(self):
        self.table.backend.connection.close()

    def test_has_class(self):
        filtered_data = filter.HasClass()(self.table)
        correct_data = [row for row in self.data if row[-1]]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_negated(self):
        filtered_data = filter.HasClass(negate=True)(self.table)
        correct_data = [row for row in self.data if not row[-1]]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)


class SameValueFilterTests(PostgresTest):
    def setUp(self):
        self.data = [
            [1, 2, 3, 'a', 'm'],
            [2, None, 1, 'a', 'f'],
            [1, 3, 1, 'b', None],
            [2, 2, 3, 'b', 'f'],
        ]
        self.table_uri = self.create_sql_table(self.data)
        self.table = SqlTable(self.table_uri)

    def tearDown(self):
        self.table.backend.connection.close()

    def test_on_continuous_attribute(self):
        filtered_data = filter.SameValue(0, 1)(self.table)
        correct_data = [row for row in self.data if row[0] == 1]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_on_continuous_attribute_with_unknowns(self):
        filtered_data = filter.SameValue(1, 2)(self.table)
        correct_data = [row for row in self.data if row[1] == 2]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_on_continuous_attribute_with_unknown_value(self):
        filtered_data = filter.SameValue(1, None)(self.table)
        correct_data = [row for row in self.data if row[1] is None]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_on_continuous_attribute_negated(self):
        filtered_data = filter.SameValue(0, 1, negate=True)(self.table)
        correct_data = [row for row in self.data if not row[0] == 1]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_on_discrete_attribute(self):
        filtered_data = filter.SameValue(3, 'a')(self.table)
        correct_data = [row for row in self.data if row[3] == 'a']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_on_discrete_attribute_with_unknown_value(self):
        filtered_data = filter.SameValue(4, None)(self.table)
        correct_data = [row for row in self.data if row[4] is None]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_on_discrete_attribute_with_unknowns(self):
        filtered_data = filter.SameValue(4, 'm')(self.table)
        correct_data = [row for row in self.data if row[4] == 'm']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_on_discrete_attribute_negated(self):
        filtered_data = filter.SameValue(3, 'a', negate=True)(self.table)
        correct_data = [row for row in self.data if not row[3] == 'a']

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_on_discrete_attribute_value_passed_as_int(self):
        values = self.table.domain[3].values
        filtered_data = filter.SameValue(3, 0, negate=True)(self.table)
        correct_data = [row for row in self.data if not row[3] == values[0]]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_on_discrete_attribute_value_passed_as_float(self):
        values = self.table.domain[3].values
        filtered_data = filter.SameValue(3, 0., negate=True)(self.table)
        correct_data = [row for row in self.data if not row[3] == values[0]]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)


class ValuesFilterTests(PostgresTest):
    def setUp(self):
        self.data = [
            [1, 2, 3, 'a', 'm'],
            [2, None, 1, 'a', 'f'],
            [1, 3, 1, 'b', None],
            [2, 2, 3, 'b', 'f'],
        ]
        self.table_uri = self.create_sql_table(self.data)
        self.table = SqlTable(self.table_uri)

    def tearDown(self):
        self.table.backend.connection.close()

    def test_values_filter_with_no_conditions(self):
        filtered_data = filter.Values()(self.table)
        correct_data = self.data

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_discrete_value_filter(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterDiscrete(3, ['a'])
        ])(self.table)
        correct_data = [row for row in self.data if row[3] in ['a']]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_discrete_value_filter_with_multiple_values(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterDiscrete(3, ['a', 'b'])
        ])(self.table)
        correct_data = [row for row in self.data if row[3] in ['a', 'b']]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_discrete_value_filter_with_None(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterDiscrete(3, None)
        ])(self.table)
        correct_data = [row for row in self.data if row[3] is not None]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_continuous_value_filter_equal(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterContinuous(0, filter.FilterContinuous.Equal, 1)
        ])(self.table)
        correct_data = [row for row in self.data if row[0] == 1]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_continuous_value_filter_not_equal(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterContinuous(0, filter.FilterContinuous.NotEqual, 1)
        ])(self.table)
        correct_data = [row for row in self.data if row[0] != 1]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_continuous_value_filter_less(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterContinuous(0, filter.FilterContinuous.Less, 2)
        ])(self.table)
        correct_data = [row for row in self.data if row[0] < 2]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_continuous_value_filter_less_equal(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterContinuous(0, filter.FilterContinuous.LessEqual, 2)
        ])(self.table)
        correct_data = [row for row in self.data if row[0] <= 2]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_continuous_value_filter_greater(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterContinuous(0, filter.FilterContinuous.Greater, 1)
        ])(self.table)
        correct_data = [row for row in self.data if row[0] > 1]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_continuous_value_filter_greater_equal(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterContinuous(0, filter.FilterContinuous.GreaterEqual, 1)
        ])(self.table)
        correct_data = [row for row in self.data if row[0] >= 1]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_continuous_value_filter_between(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterContinuous(0, filter.FilterContinuous.Between, 1, 2)
        ])(self.table)
        correct_data = [row for row in self.data if 1 <= row[0] <= 2]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_continuous_value_filter_outside(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterContinuous(0, filter.FilterContinuous.Outside, 2, 3)
        ])(self.table)
        correct_data = [row for row in self.data if not 2 <= row[0] <= 3]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)

    def test_continuous_value_filter_isdefined(self):
        filtered_data = filter.Values(conditions=[
            filter.FilterContinuous(1, filter.FilterContinuous.IsDefined)
        ])(self.table)
        correct_data = [row for row in self.data if row[1] is not None]

        self.assertEqual(len(filtered_data), len(correct_data))
        self.assertSequenceEqual(filtered_data, correct_data)
