import inspect
import os
import pkgutil
import unittest

import numpy as np

import Orange.classification
from Orange.classification import (
    Learner, Model, NaiveBayesLearner, LogisticRegressionLearner)
from Orange.data import DiscreteVariable, Domain, Table
from Orange.data.io import BasketFormat
from Orange.evaluation import CrossValidation
from Orange.tests.dummy_learners import DummyLearner, DummyMulticlassLearner


class MultiClassTest(unittest.TestCase):
    def test_unsupported(self):
        nrows = 20
        ncols = 10
        x = np.random.random_integers(1, 3, (nrows, ncols))

        # multiple class variables
        y = np.random.random_integers(10, 11, (nrows, 2))
        t = Table(x, y)
        learn = DummyLearner()
        with self.assertRaises(TypeError):
            clf = learn(t)

        # single class variable
        y = np.random.random_integers(10, 11, (nrows, 1))
        t = Table(x, y)
        learn = DummyLearner()
        clf = learn(t)
        z = clf(x)
        self.assertEqual(z.ndim, 1)

    def test_supported(self):
        nrows = 20
        ncols = 10
        x = np.random.random_integers(1, 3, (nrows, ncols))
        y = np.random.random_integers(10, 11, (nrows, 2))
        t = Table(x, y)
        learn = DummyMulticlassLearner()
        clf = learn(t)
        z = clf(x)
        self.assertEqual(z.shape, y.shape)


class ModelTest(unittest.TestCase):
    def test_predict_single_instance(self):
        table = Table("titanic")
        learn = NaiveBayesLearner()
        clf = learn(table)
        pred = []
        for row in table:
            pred.append(clf(row))

    def test_value_from_probs(self):
        nrows = 100
        ncols = 5
        x = np.random.random_integers(0, 1, (nrows, ncols))

        # single class variable
        y = np.random.random_integers(1, 3, (nrows, 1)) // 2    # majority = 1
        t = Table(x, y)
        learn = DummyLearner()
        clf = learn(t)
        clf.ret = Model.Probs
        y2 = clf(x, ret=Model.Value)
        self.assertTrue(y2.shape == (nrows,))
        y2, probs = clf(x, ret=Model.ValueProbs)
        self.assertTrue(y2.shape == (nrows, ))
        self.assertTrue(probs.shape == (nrows, 2))

        # multitarget
        y = np.random.random_integers(1, 5, (nrows, 2))
        y[:, 0] = y[:, 0] // 3          # majority = 1
        y[:, 1] = (y[:, 1] + 4) // 3    # majority = 2
        t = Table(x, y)
        learn = DummyMulticlassLearner()
        clf = learn(t)
        clf.ret = Model.Probs
        y2 = clf(x, ret=Model.Value)
        self.assertEqual(y2.shape, y.shape)
        y2, probs = clf(x, ret=Model.ValueProbs)
        self.assertEqual(y2.shape, y.shape)
        self.assertEqual(probs.shape, (nrows, 2, 4))

    def test_probs_from_value(self):
        nrows = 100
        ncols = 5
        x = np.random.random_integers(0, 1, (nrows, ncols))

        # single class variable
        y = np.random.random_integers(1, 2, (nrows, 1))
        t = Table(x, y)
        learn = DummyLearner()
        clf = learn(t)
        clf.ret = Model.Value
        y2 = clf(x, ret=Model.Probs)
        self.assertTrue(y2.shape == (nrows, 3))
        y2, probs = clf(x, ret=Model.ValueProbs)
        self.assertTrue(y2.shape == (nrows, ))
        self.assertTrue(probs.shape == (nrows, 3))

        # multitarget
        y = np.random.random_integers(1, 5, (nrows, 2))
        y[:, 0] = y[:, 0] // 3          # majority = 1
        y[:, 1] = (y[:, 1] + 4) // 3    # majority = 2
        t = Table(x, y)
        learn = DummyMulticlassLearner()
        clf = learn(t)
        clf.ret = Model.Value
        probs = clf(x, ret=Model.Probs)
        self.assertEqual(probs.shape, (nrows, 2, 4))
        y2, probs = clf(x, ret=Model.ValueProbs)
        self.assertEqual(y2.shape, y.shape)
        self.assertEqual(probs.shape, (nrows, 2, 4))


class ExpandProbabilitiesTest(unittest.TestCase):
    def prepareTable(self, rows, attr, vars, class_var_domain):
        attributes = ["Feature %i" % i for i in range(attr)]
        classes = ["Class %i" % i for i in range(vars)]
        attr_vars = [DiscreteVariable(name=a) for a in attributes]
        class_vars = [DiscreteVariable(name=c,
                                            values=range(class_var_domain))
                      for c in classes]
        meta_vars = []
        self.domain = Domain(attr_vars, class_vars, meta_vars)
        self.x = np.random.random_integers(0, 1, (rows, attr))

    def test_single_class(self):
        rows = 10
        attr = 3
        vars = 1
        class_var_domain = 20
        self.prepareTable(rows, attr, vars, class_var_domain)
        y = np.random.random_integers(2, 5, (rows, vars)) * 2
        t = Table(self.domain, self.x, y)
        learn = DummyLearner()
        clf = learn(t)
        z, p = clf(self.x, ret=Model.ValueProbs)
        self.assertEqual(p.shape, (rows, class_var_domain))
        self.assertTrue(np.all(z == np.argmax(p, axis=-1)))

    def test_multi_class(self):
        rows = 10
        attr = 3
        vars = 5
        class_var_domain = 20
        self.prepareTable(rows, attr, vars, class_var_domain)
        y = np.random.random_integers(2, 5, (rows, vars)) * 2
        t = Table(self.domain, self.x, y)
        learn = DummyMulticlassLearner()
        clf = learn(t)
        z, p = clf(self.x, ret=Model.ValueProbs)
        self.assertEqual(p.shape, (rows, vars, class_var_domain))
        self.assertTrue(np.all(z == np.argmax(p, axis=-1)))


class SklTest(unittest.TestCase):
    def test_multinomial(self):
        table = Table("titanic")
        lr = LogisticRegressionLearner()
        assert isinstance(lr, Orange.classification.SklLearner)
        res = CrossValidation(table, [lr], k=2)
        self.assertTrue(0.7 < Orange.evaluation.AUC(res)[0] < 0.9)

    def test_nan_columns(self):
        data = Orange.data.Table("iris")
        data.X[:, (1, 3)] = np.NaN
        lr = LogisticRegressionLearner()
        res = CrossValidation(data, [lr], k=2, store_models=True)
        self.assertEqual(len(res.models[0][0].domain.attributes), 2)
        self.assertGreater(Orange.evaluation.CA(res)[0], 0.8)


class LearnerAccessibility(unittest.TestCase):
    def test_all_learners_accessible_in_Orange_classification_namespace(self):
        classification_modules = pkgutil.walk_packages(
            path=Orange.classification.__path__,
            prefix="Orange.classification.",
            onerror=lambda x: None)
        for importer, modname, ispkg in classification_modules:
            try:
                module = pkgutil.importlib.import_module(modname)
            except ImportError:
                continue

            for name, class_ in inspect.getmembers(module, inspect.isclass):
                if issubclass(class_, Learner):
                    if not hasattr(Orange.classification, class_.__name__):
                        self.fail("%s is not visible in Orange.classification"
                                  " namespace" % class_.__name__)
