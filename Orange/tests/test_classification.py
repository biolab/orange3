import inspect
import os
import pickle
import pkgutil
import unittest

import numpy as np
import traceback
from Orange.base import SklLearner

import Orange.classification
from Orange.classification import (
    Learner, Model, NaiveBayesLearner, LogisticRegressionLearner)
from Orange.data import DiscreteVariable, Domain, Table, Variable
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

    def test_learner_adequacy(self):
        table = Table("housing")
        learner = NaiveBayesLearner()
        self.assertRaises(ValueError, learner, table)

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

    def test_params(self):
        learner = SklLearner()
        self.assertIsInstance(learner.params, dict)


class ClassfierListInputTest(unittest.TestCase):
    def test_discrete(self):
        table = Table("titanic")
        tree = Orange.classification.TreeLearner()(table)
        strlist = [ [ "crew", "adult", "male" ],
                    [ "crew", "adult", None ] ]
        for se in strlist: #individual examples
            assert(all(tree(se) == tree(Orange.data.Table(table.domain, [se]))))
        assert(all(tree(strlist) == tree(Orange.data.Table(table.domain, strlist))))

    def test_continuous(lf):
        table = Table("iris")
        tree = Orange.classification.TreeLearner()(table)
        strlist = [ [ 2, 3, 4, 5 ],
                    [ 1, 2, 3, 5 ] ]
        for se in strlist: #individual examples
            assert(all(tree(se) == tree(Orange.data.Table(table.domain, [se]))))
        assert(all(tree(strlist) == tree(Orange.data.Table(table.domain, strlist))))


class LearnerAccessibility(unittest.TestCase):
    def all_learners(self):
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
                if issubclass(class_, Learner) and 'base' not in class_.__module__:
                    yield class_

    def test_all_learners_accessible_in_Orange_classification_namespace(self):
        for learner in self.all_learners():
            if not hasattr(Orange.classification, learner.__name__):
                self.fail("%s is not visible in Orange.classification"
                          " namespace" % learner.__name__)

    def test_all_models_work_after_unpickling(self):
        Variable._clear_all_caches()
        datasets = [Table('iris'), Table('titanic')]
        for learner in list(self.all_learners()):
            try:
                learner = learner()
            except Exception as err:
                print('%s cannot be used with default parameters' % learner.__name__)
                traceback.print_exc()
                continue

            for ds in datasets:
                model = learner(ds)
                s = pickle.dumps(model, 0)
                model2 = pickle.loads(s)

                np.testing.assert_almost_equal(Table(model.domain, ds).X, Table(model2.domain, ds).X)
                np.testing.assert_almost_equal(model(ds), model2(ds),
                                               err_msg='%s does not return same values when unpickled %s' % (learner.__class__.__name__, ds.name))
                #print('%s on %s works' % (learner, ds.name))

    def test_adequacy_all_learners(self):
        for learner in self.all_learners():
            try:
                learner = learner()
                table = Table("housing")
                self.assertRaises(ValueError, learner, table)
            except TypeError as err:
                traceback.print_exc()
                continue
