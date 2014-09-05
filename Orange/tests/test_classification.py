import os
import unittest
import numpy as np

from Orange import data
import Orange.classification
from Orange.tests.dummy_learners import DummyLearner, DummyMulticlassLearner
import Orange.classification.naive_bayes as nb
from Orange.data.io import BasketReader


class MultiClassTest(unittest.TestCase):
    def test_unsupported(self):
        nrows = 20
        ncols = 10
        x = np.random.random_integers(1, 3, (nrows, ncols))

        # multiple class variables
        y = np.random.random_integers(10, 11, (nrows, 2))
        t = data.Table(x, y)
        learn = DummyLearner()
        with self.assertRaises(TypeError):
            clf = learn(t)

        # single class variable
        y = np.random.random_integers(10, 11, (nrows, 1))
        t = data.Table(x, y)
        learn = DummyLearner()
        clf = learn(t)
        z = clf(x)
        self.assertEqual(z.ndim, 1)

    def test_supported(self):
        nrows = 20
        ncols = 10
        x = np.random.random_integers(1, 3, (nrows, ncols))
        y = np.random.random_integers(10, 11, (nrows, 2))
        t = data.Table(x, y)
        learn = DummyMulticlassLearner()
        clf = learn(t)
        z = clf(x)
        self.assertEqual(z.shape, y.shape)


class ModelTest(unittest.TestCase):

    def test_predict_single_instance(self):
        table = data.Table("titanic")
        learn = nb.BayesLearner()
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
        t = data.Table(x, y)
        learn = DummyLearner()
        clf = learn(t)
        clf.ret = Orange.classification.Model.Probs
        y2 = clf(x, ret=Orange.classification.Model.Value)
        self.assertTrue(y2.shape == (nrows,))
        y2, probs = clf(x, ret=Orange.classification.Model.ValueProbs)
        self.assertTrue(y2.shape == (nrows, ))
        self.assertTrue(probs.shape == (nrows, 2))

        # multitarget
        y = np.random.random_integers(1, 5, (nrows, 2))
        y[:, 0] = y[:, 0] // 3          # majority = 1
        y[:, 1] = (y[:, 1] + 4) // 3    # majority = 2
        t = data.Table(x, y)
        learn = DummyMulticlassLearner()
        clf = learn(t)
        clf.ret = Orange.classification.Model.Probs
        y2 = clf(x, ret=Orange.classification.Model.Value)
        self.assertEqual(y2.shape, y.shape)
        y2, probs = clf(x, ret=Orange.classification.Model.ValueProbs)
        self.assertEqual(y2.shape, y.shape)
        self.assertEqual(probs.shape, (nrows, 2, 4))

    def test_probs_from_value(self):
        nrows = 100
        ncols = 5
        x = np.random.random_integers(0, 1, (nrows, ncols))

        # single class variable
        y = np.random.random_integers(1, 2, (nrows, 1))
        t = data.Table(x, y)
        learn = DummyLearner()
        clf = learn(t)
        clf.ret = Orange.classification.Model.Value
        y2 = clf(x, ret=Orange.classification.Model.Probs)
        self.assertTrue(y2.shape == (nrows, 3))
        y2, probs = clf(x, ret=Orange.classification.Model.ValueProbs)
        self.assertTrue(y2.shape == (nrows, ))
        self.assertTrue(probs.shape == (nrows, 3))

        # multitarget
        y = np.random.random_integers(1, 5, (nrows, 2))
        y[:, 0] = y[:, 0] // 3          # majority = 1
        y[:, 1] = (y[:, 1] + 4) // 3    # majority = 2
        t = data.Table(x, y)
        learn = DummyMulticlassLearner()
        clf = learn(t)
        clf.ret = Orange.classification.Model.Value
        probs = clf(x, ret=Orange.classification.Model.Probs)
        self.assertEqual(probs.shape, (nrows, 2, 4))
        y2, probs = clf(x, ret=Orange.classification.Model.ValueProbs)
        self.assertEqual(y2.shape, y.shape)
        self.assertEqual(probs.shape, (nrows, 2, 4))


class ExpandProbabilitiesTest(unittest.TestCase):
    def prepareTable(self, rows, attr, vars, class_var_domain):
        attributes = ["Feature %i" % i for i in range(attr)]
        classes = ["Class %i" % i for i in range(vars)]
        attr_vars = [data.DiscreteVariable(name=a) for a in attributes]
        class_vars = [data.DiscreteVariable(name=c,
                                            values=range(class_var_domain))
                      for c in classes]
        meta_vars = []
        self.domain = data.Domain(attr_vars, class_vars, meta_vars)
        self.x = np.random.random_integers(0, 1, (rows, attr))

    def test_single_class(self):
        rows = 10
        attr = 3
        vars = 1
        class_var_domain = 20
        self.prepareTable(rows, attr, vars, class_var_domain)
        y = np.random.random_integers(2, 5, (rows, vars)) * 2
        t = data.Table(self.domain, self.x, y)
        learn = DummyLearner()
        clf = learn(t)
        z, p = clf(self.x, ret=Orange.classification.Model.ValueProbs)
        self.assertEqual(p.shape, (rows, class_var_domain))
        self.assertTrue(np.all(z == np.argmax(p, axis=-1)))

    def test_multi_class(self):
        rows = 10
        attr = 3
        vars = 5
        class_var_domain = 20
        self.prepareTable(rows, attr, vars, class_var_domain)
        y = np.random.random_integers(2, 5, (rows, vars)) * 2
        t = data.Table(self.domain, self.x, y)
        learn = DummyMulticlassLearner()
        clf = learn(t)
        z, p = clf(self.x, ret=Orange.classification.Model.ValueProbs)
        self.assertEqual(p.shape, (rows, vars, class_var_domain))
        self.assertTrue(np.all(z == np.argmax(p, axis=-1)))


class SparseTest(unittest.TestCase):
    @unittest.skip("TODO: learner for sparse multiclass data.")
    def test_sparse_basket(self):
        current_dir = os.path.dirname(__file__)
        dataset = os.path.join(current_dir, "iris_basket.basket")
        table = BasketReader().read_file(dataset)
        test = Orange.data.Table.from_table_rows(table,
                                                 range(0, len(table), 2))
        train = Orange.data.Table.from_table_rows(table,
                                                  range(1, len(table), 2))
        learn = DummyMulticlassLearner()
        clf = learn(train)
        p = clf(test)
        self.assertEqual(p.shape, test.Y.shape)
        p = clf(test.X)
        self.assertEqual(p.shape, test.Y.shape)
