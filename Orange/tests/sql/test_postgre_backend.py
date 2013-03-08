import unittest

from Orange.data.sql import postgre_backend


class PostgreBckendTests(unittest.TestCase):
    def setUp(self):
        self.backend = postgre_backend.PostgreBackend()
        self.table_name = 'iris'

    def test_can_connect_to_database(self):
        self.backend.connect(
            hostname='localhost',
            database='test',
            table=self.table_name,
        )

        self.assertEqual(self.backend.table_name, 'iris')
        self.assertEqual(
            self.backend.table_info.field_names,
            ('sepal_length', 'sepal_width', 'petal_length', 'petal_width',
             'iris')
        )
        self.assertSequenceEqual(self.backend.table_info.values['iris'],
                                 ['Iris-setosa', 'Iris-versicolor',
                                  'Iris-virginica'])
        self.assertEqual(self.backend.table_info.nrows, 150)

