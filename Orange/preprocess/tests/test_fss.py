import unittest
from unittest.mock import Mock

import numpy as np

from Orange.data import Domain, Table, DiscreteVariable, ContinuousVariable
from Orange.preprocess import fss


class SelectBestFeaturesTest(unittest.TestCase):
    def test_no_nice_features(self):
        method = Mock()
        method.feature_type = DiscreteVariable
        selector = fss.SelectBestFeatures(method, 5)

        domain = Domain([])
        data = Table.from_numpy(domain, np.zeros((100, 0)))
        selection = selector.score_only_nice_features(data, method)
        self.assertEqual(selection.size, 0)
        method.assert_not_called()

        domain = Domain([ContinuousVariable("x")])
        data = Table.from_numpy(domain, np.zeros((100, 1)))
        selector.decreasing = True
        selection = selector.score_only_nice_features(data, method)
        np.testing.assert_equal(selection, [float('-inf')])
        method.assert_not_called()

        selector.decreasing = False
        selection = selector.score_only_nice_features(data, method)
        np.testing.assert_equal(selection, [float('inf')])
        method.assert_not_called()


if __name__ == "__main__":
    unittest.main()
