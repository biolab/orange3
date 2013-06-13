import unittest
import numpy as np

from Orange import data
from Orange.classification import majority
from Orange.evaluation import testing


class CrossValidationTestCase(unittest.TestCase):
    def setUp(self):
        self.data = data.Table('iris')[30:130]
        self.fitters = [majority.MajorityLearner(), majority.MajorityLearner()]

    def test_10_fold_cross(self):
        results = testing.CrossValidation(k=10)(self.data, self.fitters)

        self.assertEqual(results.predicted.shape, (2, len(self.data)))
        np.testing.assert_equal(results.predicted, np.ones((2, 100)))
        probs = results.probabilities
        self.assertTrue((probs[:, :, 0] < probs[:, :, 2]).all())
        self.assertTrue((probs[:, :, 2] < probs[:, :, 1]).all())
