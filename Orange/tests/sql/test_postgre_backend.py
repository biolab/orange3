import unittest

from Orange.data.sql import postgre_backend


class PostgreBckendTests(unittest.TestCase):
    def setUp(self):
        self.backend = postgre_backend.PostgreBackend()
        self.table_name = 'iris'
        self.backend.connect(
            hostname='localhost',
            database='test',
            table=self.table_name,
        )

    def test_can_connect_to_database(self):
        self.assertEqual(self.backend.table_name, 'iris')
        self.assertEqual(
            self.backend.table_info.field_names,
            ('sepal length', 'sepal width', 'petal length', 'petal width',
             'iris')
        )
        self.assertSequenceEqual(self.backend.table_info.values['iris'],
                                 ['Iris-setosa', 'Iris-versicolor',
                                  'Iris-virginica'])
        self.assertEqual(self.backend.table_info.nrows, 150)

    def test_query_all(self):
        results = list(self.backend.query())

        self.assertEqual(len(results), 150)

    def test_query_subset_of_attributes(self):
        attributes = [
            self._mock_attribute("sepal length"),
            self._mock_attribute("sepal width"),
            self._mock_attribute("double width", '2 * "sepal width"')
        ]
        results = list(self.backend.query(
            attributes
        ))

        self.assertSequenceEqual(
            results[:5],
            [(5.1, 3.5, 7.0),
             (4.9, 3.0, 6.0),
             (4.7, 3.2, 6.4),
             (4.6, 3.1, 6.2),
             (5.0, 3.6, 7.2)]
        )

    def test_query_subset_of_rows(self):
        all_results = list(self.backend.query())

        results = list(self.backend.query(rows=range(10)))
        self.assertEqual(len(results), 10)
        self.assertSequenceEqual(results, all_results[:10])

        results = list(self.backend.query(rows=range(10)))
        self.assertEqual(len(results), 10)
        self.assertSequenceEqual(results, all_results[:10])

        results = list(self.backend.query(rows=slice(None, 10)))
        self.assertEqual(len(results), 10)
        self.assertSequenceEqual(results, all_results[:10])

        results = list(self.backend.query(rows=slice(10, None)))
        self.assertEqual(len(results), 140)
        self.assertSequenceEqual(results, all_results[10:])

    def _mock_attribute(self, attr_name, formula=None):
        if formula is None:
            formula = '"%s"' % attr_name
        class attr:
            name = attr_name

            @staticmethod
            def get_value_from(xxx):
                return formula
        return attr

