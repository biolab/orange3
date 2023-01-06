# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import numpy as np
from Orange.data import Table
from Orange.evaluation import CrossValidation, CA, RMSE
from Orange.classification import RandomForestLearner
from Orange.regression import RandomForestRegressionLearner
from Orange.tests import test_filename


class RandomForestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.housing = Table('housing')

    def test_RandomForest(self):
        forest = RandomForestLearner()
        cv = CrossValidation(k=10)
        results = cv(self.iris, [forest])
        ca = CA(results)
        self.assertGreater(ca, 0.9)
        self.assertLess(ca, 0.99)

    def test_predict_single_instance(self):
        forest = RandomForestLearner()
        c = forest(self.iris)
        for ins in self.iris:
            c(ins)
            val, prob = c(ins, c.ValueProbs)

    def test_predict_table(self):
        forest = RandomForestLearner()
        c = forest(self.iris)
        c(self.iris)
        vals, probs = c(self.iris, c.ValueProbs)

    def test_predict_numpy(self):
        forest = RandomForestLearner()
        c = forest(self.iris)
        c(self.iris.X)
        vals, probs = c(self.iris.X, c.ValueProbs)

    def test_RandomForestRegression(self):
        forest = RandomForestRegressionLearner()
        cv = CrossValidation(k=10)
        results = cv(self.housing, [forest])
        _ = RMSE(results)

    def test_predict_single_instance_reg(self):
        forest = RandomForestRegressionLearner()
        model = forest(self.housing)
        for ins in self.housing:
            pred = model(ins)
            self.assertGreater(pred, 0)

    def test_predict_table_reg(self):
        forest = RandomForestRegressionLearner()
        model = forest(self.housing)
        pred = model(self.housing)
        self.assertEqual(len(self.housing), len(pred))
        self.assertGreater(all(pred), 0)

    def test_predict_numpy_reg(self):
        forest = RandomForestRegressionLearner()
        model = forest(self.housing)
        pred = model(self.housing.X)
        self.assertEqual(len(self.housing), len(pred))
        self.assertGreater(all(pred), 0)

    def test_classification_scorer(self):
        learner = RandomForestLearner()
        scores = learner.score_data(self.iris)
        self.assertEqual(scores.shape[1], len(self.iris.domain.attributes))
        self.assertNotEqual(sum(scores[0]), 0)
        self.assertEqual(['petal length', 'petal width'],
                         sorted([self.iris.domain.attributes[i].name
                                 for i in np.argsort(scores[0])[-2:]]))

    def test_regression_scorer(self):
        learner = RandomForestRegressionLearner()
        scores = learner.score_data(self.housing)
        self.assertEqual(['LSTAT', 'RM'],
                         sorted([self.housing.domain.attributes[i].name
                                 for i in np.argsort(scores[0])[-2:]]))

    def test_scorer_feature(self):
        np.random.seed(42)
        data = Table(test_filename('datasets/test4.tab'))
        learner = RandomForestLearner()
        scores = learner.score_data(data)
        for i, attr in enumerate(data.domain.attributes):
            np.random.seed(42)
            score = learner.score_data(data, attr)
            np.testing.assert_array_almost_equal(score, scores[:, i])

    def test_get_classification_trees(self):
        n = 5
        forest = RandomForestLearner(n_estimators=n)
        model = forest(self.iris)
        self.assertEqual(len(model.trees), n)
        tree = model.trees[0]
        self.assertEqual(tree(self.iris[0]), 0)

    def test_get_regression_trees(self):
        n = 5
        forest = RandomForestRegressionLearner(n_estimators=n)
        model = forest(self.housing)
        self.assertEqual(len(model.trees), n)
        tree = model.trees[0]
        tree(self.housing[0])

    def test_max_features_cls(self):
        data = Table("heart_disease")
        forest_1 = RandomForestLearner(random_state=0, max_features=1)
        model_1 = forest_1(data[1:])

        forest_2 = RandomForestLearner(random_state=0, max_features=1.)
        model_2 = forest_2(data[1:])
        diff = np.sum(np.abs(model_1(data[:1], ret=model_2.Probs) -
                             model_2(data[:1], ret=model_2.Probs)))
        self.assertGreaterEqual(diff, 0.2)

    def test_max_features_reg(self):
        data = self.housing
        forest_1 = RandomForestRegressionLearner(random_state=0, max_features=1)
        model_1 = forest_1(data[1:])

        forest_2 = RandomForestRegressionLearner(random_state=0, max_features=1.)
        model_2 = forest_2(data[1:])
        diff = np.sum(np.abs(model_1(data[:1]) - model_2(data[:1])))
        self.assertGreater(diff, 1.2)


if __name__ == "__main__":
    unittest.main()
