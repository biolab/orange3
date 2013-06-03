import unittest
import numpy as np
from Orange.data import Table
from Orange.feature import scoring


class FeatureScoringTest(unittest.TestCase):

    def setUp(self):
        self.zoo = Table("zoo")

    def test_info_gain(self):
        scorer = scoring.InfoGain()
        correct = [0.79067, 0.71795, 0.83014, 0.97432, 0.46970]
        np.testing.assert_almost_equal([scorer(a, self.zoo) for a in range(5)], correct, decimal=5)

    def test_gain_ratio(self):
        scorer = scoring.GainRatio()
        correct = [0.80351, 1.00000, 0.84754, 1.00000, 0.59376]
        np.testing.assert_almost_equal([scorer(a, self.zoo) for a in range(5)], correct, decimal=5)

    def test_gini(self):
        scorer = scoring.Gini()
        correct = [0.11893, 0.10427, 0.13117, 0.14650, 0.05973]
        np.testing.assert_almost_equal([scorer(a, self.zoo) for a in range(5)], correct, decimal=5)
