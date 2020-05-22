# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np
import sklearn

from Orange.data import Table, ContinuousVariable, Domain
from Orange.classification import LogisticRegressionLearner, Model
from Orange.evaluation import CrossValidation, CA


class TestLogisticRegressionLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.heart_disease = Table('heart_disease.tab')
        cls.zoo = Table('zoo')

    def test_LogisticRegression(self):
        learn = LogisticRegressionLearner()
        cv = CrossValidation(k=2)
        results = cv(self.heart_disease, [learn])
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
        cv = CrossValidation(k=3)
        results = cv(table, [lr_norm, lr])
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
        scores = learner.score_data(self.heart_disease)
        self.assertEqual('chest pain',
                         self.heart_disease.domain.attributes[np.argmax(scores)].name)
        self.assertEqual(scores.shape, (1, len(self.heart_disease.domain.attributes)))

    def test_learner_scorer_feature(self):
        learner = LogisticRegressionLearner()
        scores = learner.score_data(self.heart_disease)
        for i, attr in enumerate(self.heart_disease.domain.attributes):
            score = learner.score_data(self.heart_disease, attr)
            np.testing.assert_array_almost_equal(score, scores[:, i])

    def test_learner_scorer_previous_transformation(self):
        learner = LogisticRegressionLearner()
        from Orange.preprocess import Discretize
        data = Discretize()(self.iris)
        scores = learner.score_data(data)
        # scores should be defined and positive
        self.assertTrue(np.all(scores > 0))

    def test_learner_scorer_multiclass(self):
        attr = self.zoo.domain.attributes
        learner = LogisticRegressionLearner()
        scores = learner.score_data(self.zoo)
        self.assertEqual('legs', attr[np.argmax(scores[0])].name)  # amphibian
        self.assertEqual('feathers', attr[np.argmax(scores[1])].name)  # bird
        self.assertEqual('fins', attr[np.argmax(scores[2])].name)  # fish
        self.assertEqual('legs', attr[np.argmax(scores[3])].name)  # insect
        self.assertEqual('backbone', attr[np.argmax(scores[4])].name)  # invertebrate
        self.assertEqual('milk', attr[np.argmax(scores[5])].name)  # mammal
        self.assertEqual('aquatic', attr[np.argmax(scores[6])].name)  # reptile
        self.assertEqual(scores.shape,
                         (len(self.zoo.domain.class_var.values), len(attr)))

    def test_learner_scorer_multiclass_feature(self):
        learner = LogisticRegressionLearner()
        scores = learner.score_data(self.zoo)
        for i, attr in enumerate(self.zoo.domain.attributes):
            score = learner.score_data(self.zoo, attr)
            np.testing.assert_array_almost_equal(score, scores[:, i])

    def test_coefficients(self):
        learn = LogisticRegressionLearner()
        model = learn(self.heart_disease)
        coef = model.coefficients
        self.assertEqual(len(coef[0]), len(model.domain.attributes))

    def test_predict_on_instance(self):
        lr = LogisticRegressionLearner()
        m = lr(self.zoo)
        probs = m(self.zoo[50], m.Probs)
        probs2 = m(self.zoo[50, :], m.Probs)
        np.testing.assert_almost_equal(probs, probs2[0])

    def test_single_class(self):
        t = self.iris[60:90]
        self.assertEqual(len(np.unique(t.Y)), 1)
        learn = LogisticRegressionLearner()
        with self.assertRaises(ValueError):
            learn(t)

    def test_sklearn_single_class(self):
        t = self.iris[60:90]
        self.assertEqual(len(np.unique(t.Y)), 1)
        lr = sklearn.linear_model.LogisticRegression()
        self.assertRaises(ValueError, lr.fit, t.X, t.Y)

    def test_auto_solver(self):
        # These defaults are valid as of sklearn v0.23.0
        # lbfgs is default for l2 penalty
        lr = LogisticRegressionLearner(penalty="l2", solver="auto")
        skl_clf = lr._initialize_wrapped()
        self.assertEqual(skl_clf.solver, "lbfgs")
        self.assertEqual(skl_clf.penalty, "l2")

        # lbfgs is default for no penalty
        lr = LogisticRegressionLearner(penalty=None, solver="auto")
        skl_clf = lr._initialize_wrapped()
        self.assertEqual(skl_clf.solver, "lbfgs")
        self.assertEqual(skl_clf.penalty, None)

        # liblinear is default for l2 penalty
        lr = LogisticRegressionLearner(penalty="l1", solver="auto")
        skl_clf = lr._initialize_wrapped()
        self.assertEqual(skl_clf.solver, "liblinear")
        self.assertEqual(skl_clf.penalty, "l1")
