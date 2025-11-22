import unittest

import numpy as np

from Orange.evaluation import MAPE
from Orange.evaluation.scoring import SMAPE


class TestScoring(unittest.TestCase):
    def test_mape(self):
        f = MAPE.__wraps__
        exp = np.array([100, -200, 300, 60])
        pred = np.array([110, -180, 340, 60])
        self.assertEqual(f(exp, pred), (10 / 100 + 20 / 200 + 40 / 300) / 4 * 100)

        exp = np.array([0, 200, 300])
        self.assertEqual(f(exp, pred), np.inf)

    def test_smape(self):
        f = SMAPE.__wraps__
        exp = np.array([100, -200, 300, 60, 80])
        pred = np.array([110, -180, -340, 60, 50])
        self.assertEqual(
            f(exp, pred),
            2 * (10 / 210 + 20 / 380 + 640 / 640 + 0 / 120 + 30 / 130) / 5 * 100)

        exp = np.array([0, -200, 300, 60, 80])
        self.assertEqual(
            f(exp, pred),
            2 * (110 / 110 + 20 / 380 + 640 / 640 + 0 / 120 + 30 / 130) / 5 * 100)


if __name__ == '__main__':
    unittest.main()
