# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import unittest
import numpy as np

from Orange.data import Table, Domain
from Orange.projection import RadViz


class TestRadViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table("iris")
        with cls.iris.unlocked():
            cls.iris[3, 3] = np.nan
        cls.titanic = Table("titanic")

    def test_radviz(self):
        table = self.iris
        table = table.transform(Domain(table.domain.attributes[2:]))
        projector = RadViz()
        projection = projector(table)
        embedding = projection(table)
        self.assertEqual(len(embedding), len(table))
        self.assertTrue(np.isnan(embedding.X).any())
        np.testing.assert_array_equal(embedding[:100], projection(table[:100]))

    def test_discrete_features(self):
        table = self.titanic[::10]
        projector = RadViz()
        self.assertRaises(ValueError, projector, table)

        table = table.transform(Domain(table.domain.attributes[1:]))
        projector = RadViz()
        projection = projector(table)
        embedding = projection(table[::10])
        self.assertEqual(np.sum(embedding), -17)
