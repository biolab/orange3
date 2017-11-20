# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import numpy as np

from Orange.data import Table
from Orange.preprocess import Continuize
from Orange.projection import FreeViz


class TestFreeviz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table("iris")
        cls.housing = Table("housing")
        cls.zoo = Table("zoo")

    def test_basic(self):
        table = Table("iris")
        table[3, 3] = np.nan
        freeviz = FreeViz()
        model = freeviz(table)
        model(table)

    def test_regression(self):
        table = Table("housing")[::10]
        freeviz = FreeViz()
        freeviz(table)

        freeviz = FreeViz(p=2)
        freeviz(table)

    def test_prepare_freeviz_data(self):
        table = Table("iris")
        FreeViz.prepare_freeviz_data(table)

        table.X = table.X * np.nan
        self.assertEqual(FreeViz.prepare_freeviz_data(table), (None, None, None))

        table.X = None
        FreeViz.prepare_freeviz_data(table)

    @unittest.skip("Test weights is too slow.")
    def test_weights(self):
        table = Table("iris")
        weights = np.random.rand(150, 1).flatten()
        freeviz = FreeViz(weights=weights, p=3, scale=False, center=False)
        freeviz(table)

        scale = np.array([0.5, 0.4, 0.6, 0.8])
        freeviz = FreeViz(scale=scale, center=[0.2, 0.6, 0.4, 0.2])
        freeviz(table)

    def test_raising_errors(self):
        table = Table("iris")
        freeviz = FreeViz(initial=(2, 4))
        self.assertRaises(ValueError, freeviz, table)

        scale = np.array([0.5, 0.4, 0.6])
        freeviz = FreeViz(scale=scale)
        self.assertRaises(ValueError, freeviz, table)

        freeviz = FreeViz(center=[0.6, 0.4, 0.2])
        self.assertRaises(ValueError, freeviz, table)

        weights = np.random.rand(100, 1).flatten()
        freeviz = FreeViz(weights=weights)
        self.assertRaises(ValueError, freeviz, table)

    def test_initial(self):
        FreeViz.init_radial(1)
        FreeViz.init_radial(2)
        FreeViz.init_radial(3)
        FreeViz.init_random(2, 4, 5)

    def test_transform_changed_domain(self):
        """
        1. Open data, apply some preprocessor, splits the data into two parts,
        use FreeViz on the first part, and then transform the second part.

        2. Open data, split into two parts, apply the same preprocessor and
        FreeViz only on the first part, and then transform the second part.

        The transformed second part in (1) and (2) has to be the same.
        """
        data = Table("titanic")[::10]
        normalize = Continuize()
        freeviz = FreeViz(maxiter=40)

        # normalize all
        ndata = normalize(data)
        model = freeviz(ndata[:100])
        result_1 = model(ndata[100:])

        # normalize only the "training" part
        ndata = normalize(data[:100])
        model = freeviz(ndata)
        result_2 = model(data[100:])

        np.testing.assert_almost_equal(result_1.X, result_2.X)
