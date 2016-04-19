# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import numpy as np

from Orange.data import Table, ContinuousVariable, Domain
from Orange.classification import LogisticRegressionLearner, Model
from Orange.evaluation import CrossValidation, CA


class LogisticRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.voting = Table('voting')
        cls.zoo = Table('zoo')

    def test_LogisticRegression(self):
        learn = LogisticRegressionLearner()
        results = CrossValidation(self.voting, [learn], k=2)
        ca = CA(results)
        self.assertGreater(ca, 0.8)
        self.assertLess(ca, 1.0)

    @unittest.skip("Re-enable when Logistic regression supports normalization.")
    def test_LogisticRegressionNormalization(self):
        np.random.seed(42)
        new_attrs = (ContinuousVariable('c0'),) + self.iris.domain.attributes
        new_domain = Domain(new_attrs,
                            self.iris.domain.class_vars,
                            self.iris.domain.metas)
        new_table = np.hstack((
            1000000 * np.random.random((self.iris.X.shape[0], 1)),
            self.iris))
        table = self.iris.from_numpy(new_domain, new_table)
        lr = LogisticRegressionLearner(normalize=False)
        lr_norm = LogisticRegressionLearner(normalize=True)

        # check that normalization produces better results
        results = CrossValidation(table, [lr_norm, lr], k=3)
        ca = CA(results)
        self.assertGreater(ca[0], ca[1])

        # check that coefficients are properly scaled back to unnormalized data
        model = lr_norm(table)
        y = np.argmax(np.dot(table.X, model.coefficients.T) + model.intercept,
                      axis=1)
        np.testing.assert_array_equal(model(table), y)

    def test_LogisticRegressionNormalization_todo(self):
        with self.assertRaises(TypeError):
            lr = LogisticRegressionLearner(normalize=True)
            # Do not skip the above test when this is implemented

    def test_probability(self):
        learn = LogisticRegressionLearner(penalty='l1')
        clf = learn(self.iris[:100])
        p = clf(self.iris[100:], ret=Model.Probs)
        self.assertLess(abs(p.sum(axis=1) - 1).all(), 1e-6)

    def test_learner_scorer(self):
        learner = LogisticRegressionLearner()
        scores = learner.score_data(self.voting)
        self.assertEqual('physician-fee-freeze',
                         self.voting.domain.attributes[np.argmax(scores)].name)
        self.assertEqual(len(scores), len(self.voting.domain.attributes))

    def test_coefficients(self):
        learn = LogisticRegressionLearner()
        model = learn(self.voting)
        coef = model.coefficients
        self.assertEqual(len(coef[0]), len(model.domain.attributes))

    def test_predict_on_instance(self):
        lr = LogisticRegressionLearner()
        m = lr(self.zoo)
        probs = m(self.zoo[50], m.Probs)
        probs2 = m(self.zoo[50, :], m.Probs)
        np.testing.assert_almost_equal(probs, probs2)
