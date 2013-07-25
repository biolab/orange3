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


