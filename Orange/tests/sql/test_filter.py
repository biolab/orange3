from Orange.data.sql.table import SqlTable
from Orange.data.sql import filter

from Orange.tests.sql.base import PostgresTest


class IsDefinedFilterTests(PostgresTest):
    def setUp(self):
        self.table = self.create_sql_table([
            [1, 2, 3, None, 'm'],
            [2, 3, 1, 4, 'f'],
            [None, None, None, None, None],
            [7, None, 3, None, 'f'],
        ])

    def test_on_all_columns(self):
        t = SqlTable(self.table)
        t2 = filter.IsDefinedSql()(t)

        self.assertEqual(len(t2), 1)
        self.assertEqual(list(t2[0]), [2., 3., 1., 4., 0])

