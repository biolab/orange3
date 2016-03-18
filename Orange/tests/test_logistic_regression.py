import unittest

from Orange.data import Table, ContinuousVariable, Domain
from Orange.classification import LogisticRegressionLearner, Model
from Orange.evaluation import CrossValidation, CA
import numpy as np


class LogisticRegressionTest(unittest.TestCase):
    def test_LogisticRegression(self):
        table = Table('iris')
        learn = LogisticRegressionLearner()
        results = CrossValidation(table, [learn], k=2)
        ca = CA(results)
        self.assertTrue(0.8 < ca < 1.0)

    @unittest.skip("Re-enable when Logistic regression supports normalization.")
    def test_LogisticRegressionNormalization(self):
        np.random.seed(42)
        table = Table('iris')
        new_attrs = (ContinuousVariable('c0'),) + table.domain.attributes
        new_domain = Domain(new_attrs,
                            table.domain.class_vars,
                            table.domain.metas)
        new_table = np.hstack((
            1000000 * np.random.random((table.X.shape[0], 1)),
            table))
        table = table.from_numpy(new_domain, new_table)
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
        table = Table('iris')
        learn = LogisticRegressionLearner(penalty='l1')
        clf = learn(table[:100])
        p = clf(table[100:], ret=Model.Probs)
        self.assertTrue(all(abs(p.sum(axis=1) - 1) < 1e-6))

    def test_learner_scorer(self):
        data = Table('voting')
        learner = LogisticRegressionLearner()
        scores = learner.score_data(data)
        self.assertEqual('physician-fee-freeze',
                         data.domain.attributes[np.argmax(scores)].name)
        self.assertEqual(len(scores), len(data.domain.attributes))

    def test_coefficients(self):
        data = Table("voting")
        learn = LogisticRegressionLearner()
        model = learn(data)
        coef = model.coefficients
        self.assertEqual(len(coef[0]), len(model.domain.attributes))
