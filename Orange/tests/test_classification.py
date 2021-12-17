# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import inspect
import pickle
import pkgutil
import unittest

import traceback
import warnings

import numpy as np
from scipy import sparse as sp
from sklearn.exceptions import ConvergenceWarning

from Orange.base import SklLearner

import Orange.classification
from Orange.classification import (
    Learner, Model,
    NaiveBayesLearner, LogisticRegressionLearner, NuSVMLearner,
    MajorityLearner,
    RandomForestLearner, SimpleTreeLearner, SoftmaxRegressionLearner,
    SVMLearner, LinearSVMLearner, OneClassSVMLearner, TreeLearner, KNNLearner,
    SimpleRandomForestLearner, EllipticEnvelopeLearner, ThresholdLearner,
    CalibratedLearner)
from Orange.classification.rules import _RuleLearner
from Orange.data import (ContinuousVariable, DiscreteVariable,
                         Domain, Table)
from Orange.data.table import DomainTransformationError
from Orange.evaluation import CrossValidation
from Orange.tests.dummy_learners import DummyLearner, DummyMulticlassLearner
from Orange.tests import test_filename


def all_learners():
    classification_modules = pkgutil.walk_packages(
        path=Orange.classification.__path__,
        prefix="Orange.classification.",
        onerror=lambda x: None)
    for _, modname, _ in classification_modules:
        try:
            module = pkgutil.importlib.import_module(modname)
        except ImportError:
            continue

        for name, class_ in inspect.getmembers(module, inspect.isclass):
            if (issubclass(class_, Learner) and
                    not name.startswith('_') and
                    'base' not in class_.__module__):
                yield class_


class MultiClassTest(unittest.TestCase):
    def test_unsupported(self):
        nrows = 20
        ncols = 10
        x = np.random.randint(1, 4, (nrows, ncols))

        # multiple class variables
        y = np.random.randint(0, 2, (nrows, 2))
        t = Table.from_numpy(None, x, y)
        learn = DummyLearner()
        # TODO: Errors raised from various data checks should be made consistent
        with self.assertRaises((ValueError, TypeError)):
            clf = learn(t)

        # single class variable
        y = np.random.randint(0, 2, (nrows, 1))
        t = Table.from_numpy(None, x, y)
        learn = DummyLearner()
        clf = learn(t)
        z = clf(x)
        self.assertEqual(z.ndim, 1)

    def test_supported(self):
        nrows = 20
        ncols = 10
        x = np.random.randint(1, 4, (nrows, ncols))
        y = np.random.randint(0, 2, (nrows, 2))
        t = Table.from_numpy(None, x, y)
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

    def test_prediction_dimensions(self):
        class MockModel(Model):
            def predict(self, data):
                return np.zeros((data.shape[0], len(domain.class_var.values)))

        x = np.zeros((42, 5))
        y = np.zeros(42)
        domain = Domain([ContinuousVariable(n) for n in "abcde"],
                        DiscreteVariable("y", values=("a", "b")))
        data = Table.from_numpy(domain, x, y)
        a_list = [[0] * 5] * 42
        a_tuple = ((0, ) * 5,) * 42
        m = MockModel(domain)

        for inp in (data, x, sp.csr_matrix(x), a_list, a_tuple):
            msg = f"in test for type '{type(inp)}'"
            # two-dimensional
            self.assertEqual(m(inp, ret=m.Value).shape, (42, ), msg)
            self.assertEqual(m(inp, ret=m.Probs).shape, (42, 2), msg)
            values, probs = m(inp, ret=m.ValueProbs)
            self.assertEqual(values.shape, (42, ), msg)
            self.assertEqual(probs.shape, (42, 2), msg)

            # one-dimensional
            if not isinstance(inp, sp.csr_matrix):
                self.assertEqual(m(inp[0], ret=m.Value).shape, (), msg)
                self.assertEqual(m(inp[0], ret=m.Probs).shape, (2, ), msg)
                values, probs = m(inp[0], ret=m.ValueProbs)
                self.assertEqual(values.shape, (), msg)
                self.assertEqual(probs.shape, (2, ), msg)

    def test_learner_adequacy(self):
        table = Table("housing")
        learner = NaiveBayesLearner()
        self.assertRaises(ValueError, learner, table)

    def test_value_from_probs(self):
        nrows = 100
        ncols = 5
        x = np.random.randint(0, 2, (nrows, ncols))

        # single class variable
        y = np.random.randint(1, 4, (nrows, 1)) // 2    # majority = 1
        t = Table.from_numpy(None, x, y)
        learn = DummyLearner()
        clf = learn(t)
        clf.ret = Model.Probs
        y2 = clf(x, ret=Model.Value)
        self.assertEqual(y2.shape, (nrows,))
        y2, probs = clf(x, ret=Model.ValueProbs)
        self.assertEqual(y2.shape, (nrows,))
        self.assertEqual(probs.shape, (nrows, 2))

        # multitarget
        y = np.random.randint(1, 6, (nrows, 2))
        y[:, 0] = y[:, 0] // 3          # majority = 1
        y[:, 1] = (y[:, 1] + 4) // 3    # majority = 2
        domain = Domain([ContinuousVariable('i' + str(i)) for i in range(ncols)],
                        [DiscreteVariable('c' + str(i), values="0123")
                         for i in range(y.shape[1])])
        t = Table(domain, x, y)
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
        x = np.random.randint(0, 2, (nrows, ncols))

        # single class variable
        y = np.random.randint(0, 2, (nrows, 1))
        d = Domain([DiscreteVariable('v' + str(i),
                                     values=[str(v)
                                             for v in np.unique(x[:, i])])
                    for i in range(ncols)],
                   DiscreteVariable('c', values="12"))
        t = Table(d, x, y)
        learn = DummyLearner()
        clf = learn(t)
        clf.ret = Model.Value
        y2 = clf(x, ret=Model.Probs)
        self.assertEqual(y2.shape, (nrows, 2))
        y2, probs = clf(x, ret=Model.ValueProbs)
        self.assertEqual(y2.shape, (nrows, ))
        self.assertEqual(probs.shape, (nrows, 2))

        # multitarget
        y = np.random.randint(1, 6, (nrows, 2))
        y[:, 0] = y[:, 0] // 3             # majority = 1
        y[:, 1] = (y[:, 1] + 4) // 3 - 1   # majority = 1
        domain = Domain([ContinuousVariable('i' + str(i)) for i in range(ncols)],
                        [DiscreteVariable('c' + str(i), values="0123")
                         for i in range(y.shape[1])])
        t = Table(domain, x, y)
        learn = DummyMulticlassLearner()
        clf = learn(t)
        clf.ret = Model.Value
        probs = clf(x, ret=Model.Probs)
        self.assertEqual(probs.shape, (nrows, 2, 4))
        y2, probs = clf(x, ret=Model.ValueProbs)
        self.assertEqual(y2.shape, y.shape)
        self.assertEqual(probs.shape, (nrows, 2, 4))

    def test_incompatible_domain(self):
        iris = Table("iris")
        titanic = Table("titanic")
        clf = DummyLearner()(iris)
        with self.assertRaises(DomainTransformationError):
            clf(titanic)

    def test_result_shape(self):
        """
        Test if the results shapes are correct
        """
        iris = Table('iris')
        for learner in all_learners():
            # calibration, threshold learners' __init__ requires arguments
            if learner in (ThresholdLearner, CalibratedLearner):
                continue

            with self.subTest(learner.__name__):
                # model trained on only one value (but three in the domain)
                model = learner()(iris[0:100])

                res = model(iris[0:50])
                self.assertTupleEqual((50,), res.shape)

                # probabilities must still be for three classes
                res = model(iris[0:50], model.Probs)
                self.assertTupleEqual((50, 3), res.shape)

                # model trained on all classes and predicting with one class
                try:
                    model = learner()(iris[0:100])
                except TypeError:
                    # calibration, threshold learners are skipped
                    # they have some specifics regarding data
                    continue
                res = model(iris[0:50], model.Probs)
                self.assertTupleEqual((50, 3), res.shape)

    def test_result_shape_numpy(self):
        """
        Test whether results shapes are correct when testing on numpy data
        """
        iris = Table('iris')
        iris_bin = Table(
            Domain(
                iris.domain.attributes,
                DiscreteVariable("iris", values=["a", "b"])
            ),
            iris.X[:100], iris.Y[:100]
        )
        for learner in all_learners():
            with self.subTest(learner.__name__):
                args = []
                if learner in (ThresholdLearner, CalibratedLearner):
                    args = [LogisticRegressionLearner()]
                data = iris_bin if learner is ThresholdLearner else iris
                model = learner(*args)(data)
                transformed_iris = model.data_to_model_domain(data)

                res = model(transformed_iris.X[0:5])
                self.assertTupleEqual((5,), res.shape)

                res = model(transformed_iris.X[0:1], model.Probs)
                self.assertTupleEqual(
                    (1, len(data.domain.class_var.values)), res.shape
                )


class ExpandProbabilitiesTest(unittest.TestCase):
    def prepareTable(self, rows, attr, vars, class_var_domain):
        attributes = ["Feature %i" % i for i in range(attr)]
        classes = ["Class %i" % i for i in range(vars)]
        attr_vars = [DiscreteVariable(name=a, values="01") for a in attributes]
        class_vars = [
            DiscreteVariable(name=c,
                             values=[str(v) for v in range(class_var_domain)])
            for c in classes]
        meta_vars = []
        self.domain = Domain(attr_vars, class_vars, meta_vars)
        self.x = np.random.randint(0, 2, (rows, attr))

    def test_single_class(self):
        rows = 10
        attr = 3
        vars = 1
        class_var_domain = 20
        self.prepareTable(rows, attr, vars, class_var_domain)
        y = np.random.randint(2, 6, (rows, vars)) * 2
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
        y = np.random.randint(2, 6, (rows, vars)) * 2
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
        cv = CrossValidation(k=2)
        res = cv(table, [lr])
        self.assertGreater(Orange.evaluation.AUC(res)[0], 0.7)
        self.assertLess(Orange.evaluation.AUC(res)[0], 0.9)

    def test_nan_columns(self):
        data = Orange.data.Table("iris")
        with data.unlocked():
            data.X[:, (1, 3)] = np.NaN
        lr = LogisticRegressionLearner()
        cv = CrossValidation(k=2, store_models=True)
        res = cv(data, [lr])
        self.assertEqual(len(res.models[0][0].domain.attributes), 2)
        self.assertGreater(Orange.evaluation.CA(res)[0], 0.8)

    def test_params(self):
        learner = SklLearner()
        self.assertIsInstance(learner.params, dict)


class ClassfierListInputTest(unittest.TestCase):
    def test_discrete(self):
        table = Table("titanic")
        tree = Orange.classification.SklTreeLearner()(table)
        strlist = [["crew", "adult", "male"],
                   ["crew", "adult", None]]
        for se in strlist: #individual examples
            assert(all(tree(se) ==
                       tree(Orange.data.Table.from_list(table.domain, [se]))))
        assert(all(tree(strlist) ==
                   tree(Orange.data.Table.from_list(table.domain, strlist))))

    def test_continuous(self):
        table = Table("iris")
        tree = Orange.classification.SklTreeLearner()(table)
        strlist = [[2, 3, 4, 5],
                   [1, 2, 3, 5]]
        for se in strlist: #individual examples
            assert(all(tree(se) ==
                       tree(Orange.data.Table.from_list(table.domain, [se]))))
        assert(all(tree(strlist) ==
                   tree(Orange.data.Table.from_list(table.domain, strlist))))


class UnknownValuesInPrediction(unittest.TestCase):
    def test_unknown(self):
        table = Table("iris")
        tree = LogisticRegressionLearner()(table)
        tree([1, 2, None, 4])

    def test_missing_class(self):
        table = Table(test_filename("datasets/adult_sample_missing"))
        for learner in all_learners():
            # calibration, threshold learners' __init__ require arguments
            if learner in (ThresholdLearner, CalibratedLearner):
                continue
            # Skip slow tests
            if isinstance(learner, _RuleLearner):
                continue
            with self.subTest(learner.__name__):
                learner = learner()
                if isinstance(learner, NuSVMLearner):
                    learner.params["nu"] = 0.01
                model = learner(table)
                model(table)


class LearnerAccessibility(unittest.TestCase):

    def setUp(self):
        # Convergence warnings are irrelevant for these tests
        warnings.filterwarnings("ignore", ".*", ConvergenceWarning)

    def test_all_learners_accessible_in_Orange_classification_namespace(self):
        for learner in all_learners():
            if not hasattr(Orange.classification, learner.__name__):
                self.fail("%s is not visible in Orange.classification"
                          " namespace" % learner.__name__)

    def test_all_models_work_after_unpickling(self):
        datasets = [Table('iris'), Table('titanic')]
        for learner in list(all_learners()):
            # calibration, threshold learners' __init__ require arguments
            if learner in (ThresholdLearner, CalibratedLearner):
                continue
            # Skip slow tests
            if isinstance(learner, _RuleLearner):
                continue
            with self.subTest(learner.__name__):
                if "RandomForest" not in learner.__name__:
                    continue
                learner = learner()
                for ds in datasets:
                    model = learner(ds)
                    s = pickle.dumps(model, 0)
                    model2 = pickle.loads(s)

                    np.testing.assert_almost_equal(
                        Table.from_table(model.domain, ds).X,
                        Table.from_table(model2.domain, ds).X)
                    np.testing.assert_almost_equal(
                        model(ds), model2(ds),
                        err_msg='%s does not return same values when unpickled %s'
                        % (learner.__class__.__name__, ds.name))

    def test_adequacy_all_learners(self):
        for learner in all_learners():
            # calibration, threshold learners' __init__ requires arguments
            if learner in (ThresholdLearner, CalibratedLearner):
                continue
            with self.subTest(learner.__name__):
                learner = learner()
                table = Table("housing")
                self.assertRaises(ValueError, learner, table)

    def test_adequacy_all_learners_multiclass(self):
        for learner in all_learners():
            # calibration, threshold learners' __init__ require arguments
            if learner in (ThresholdLearner, CalibratedLearner):
                continue
            with self.subTest(learner.__name__):
                learner = learner()
                table = Table(test_filename("datasets/test8.tab"))
                self.assertRaises(ValueError, learner, table)


class LearnerReprs(unittest.TestCase):
    def test_reprs(self):
        lr = LogisticRegressionLearner(tol=0.0002)
        m = MajorityLearner()
        nb = NaiveBayesLearner()
        rf = RandomForestLearner(bootstrap=False, n_jobs=3)
        st = SimpleTreeLearner(seed=1, bootstrap=True)
        sm = SoftmaxRegressionLearner()
        svm = SVMLearner(shrinking=False)
        lsvm = LinearSVMLearner(tol=0.022, dual=False)
        nsvm = NuSVMLearner(tol=0.003, cache_size=190)
        osvm = OneClassSVMLearner(degree=2)
        tl = TreeLearner(max_depth=3, min_samples_split=1)
        knn = KNNLearner(n_neighbors=4)
        el = EllipticEnvelopeLearner(store_precision=False)
        srf = SimpleRandomForestLearner(n_estimators=20)

        learners = [lr, m, nb, rf, st, sm, svm,
                    lsvm, nsvm, osvm, tl, knn, el, srf]

        for l in learners:
            repr_str = repr(l)
            new_l = eval(repr_str)
            self.assertEqual(repr(new_l), repr_str)


if __name__ == "__main__":
    unittest.main()
