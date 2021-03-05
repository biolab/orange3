import unittest

import numpy as np

from Orange.data import Table, DiscreteVariable, Domain
from Orange.classification import LogisticRegressionLearner, TreeLearner


class TestModelMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = iris = Table("iris")

        tables = []
        ix = iris.X
        y = np.hstack((np.zeros(50), np.ones(50)))
        attrs = cls.iris.domain.attributes
        classes = cls.iris.domain.class_var.values
        for i, x in enumerate([ix[50:],
                               np.vstack((ix[:50], ix[100:])),
                               ix[:100]]):
            class_var = DiscreteVariable(
                "iris",
                values=tuple(n for j, n in enumerate(classes) if j != i))
            domain = Domain(attrs, class_var)
            tables.append(Table.from_numpy(domain, x, y))
        # pylint: disable=unbalanced-tuple-unpacking
        cls.iris0, cls.iris1, cls.iris2 = tables

    def test_larger_model(self):
        # Train on all data, test on subset of values
        # Probabilities should stay the same, but normalized
        # Class predictions for existing classes should stay the same
        def normalized(a):
            n = np.sum(a, axis=1)
            n[n == 0] = 1.0 / a.shape[1]
            a[n == 0] = 1
            return a / n[:, None]

        for lrn in [TreeLearner, LogisticRegressionLearner]:  # skl and non-skl
            model = lrn()(self.iris)
            val, prob = model(self.iris, model.ValueProbs)

            val0, prob0 = model(self.iris0, model.ValueProbs)
            vale = val[50:]
            probe = normalized(prob[50:, 1:])
            # No effect on class predictions
            np.testing.assert_array_equal(
                val0[vale != 0],
                vale[vale != 0] - 1)
            # Classes that to not exist are replaced with the most probable;
            # don't use argmax because of possible ties
            np.testing.assert_array_equal(
                probe[vale == 0, val0[vale == 0].astype(int)],
                np.max(probe[vale == 0], axis=1))
            # Probabilities are not affected (but normalized)
            np.testing.assert_almost_equal(prob0, probe)

            # Same as above for other two classes ...
            val1, prob1 = model(self.iris1, model.ValueProbs)
            vale = np.hstack((val[:50], val[100:])).astype(float)
            no1 = vale != 1
            vale[vale == 2] -= 1
            probe = np.vstack((prob[:50], prob[100:]))
            probe = normalized(np.hstack((probe[:, :1], probe[:, 2:])))
            np.testing.assert_array_equal(
                val1[no1],
                vale[no1])
            np.testing.assert_array_equal(
                probe[~no1, val1[~no1].astype(int)],
                np.max(probe[~no1], axis=1))
            np.testing.assert_almost_equal(prob1, probe)

            val2, prob2 = model(self.iris2, model.ValueProbs)
            vale = val[:100]
            probe = normalized(prob[:100, :2])
            np.testing.assert_array_equal(
                val2[vale != 2],
                vale[vale != 2])
            np.testing.assert_array_equal(
                probe[vale != 2, val2[vale != 2].astype(int)],
                np.max(probe[vale != 2], axis=1))
            np.testing.assert_almost_equal(prob2, probe)

    def test_smaller_model(self):
        for lrn in [LogisticRegressionLearner, TreeLearner]:  # skl and non-skl
            model = lrn()(self.iris0)
            val0, prob0 = model(self.iris0, model.ValueProbs)
            val, prob = model(self.iris, model.ValueProbs)
            # Model can't predict class 0 in whole data
            np.testing.assert_array_equal(val0, val[50:] - 1)
            np.testing.assert_almost_equal(prob0, prob[50:, 1:])
            # First 50 instances in whole data can be assigned anything 1 or 2
            # and should not be nan
            self.assertTrue(np.all((val[:50] == 1) + (val[:50] == 2)))
            np.testing.assert_almost_equal(np.sum(prob, axis=1), 1)
            np.testing.assert_almost_equal(prob[:, 0], 0)

            model = lrn()(self.iris1)
            val, prob = model(self.iris, model.ValueProbs)
            val1, prob1 = model(self.iris1, model.ValueProbs)
            np.testing.assert_array_equal(val1[:50], val[:50])
            np.testing.assert_array_equal(val1[50:], val[100:] - 1)
            np.testing.assert_almost_equal(prob1[:50, 0], prob[:50, 0])
            np.testing.assert_almost_equal(prob1[:50, 1], prob[:50, 2])
            np.testing.assert_almost_equal(prob1[50:, 0], prob[100:, 0])
            np.testing.assert_almost_equal(prob1[50:, 1], prob[100:, 2])
            self.assertTrue(np.all((val[50:100] == 0) + (val[50:100] == 2)))
            np.testing.assert_almost_equal(np.sum(prob, axis=1), 1)
            np.testing.assert_almost_equal(prob[:, 1], 0)

            model = lrn()(self.iris2)
            val, prob = model(self.iris, model.ValueProbs)
            val2, prob2 = model(self.iris2, model.ValueProbs)
            np.testing.assert_array_equal(val2, val[:100])
            np.testing.assert_almost_equal(prob2, prob[:100, :2])
            self.assertTrue(np.all((val[100:] == 0) + (val[100:] == 1)))
            self.assertTrue(np.all(val[:50] < 2))  # also tests it's not nan
            np.testing.assert_almost_equal(np.sum(prob, axis=1), 1)
            np.testing.assert_almost_equal(prob[:, 2], 0)

    def test_model_different(self):
        def test_val_prob(val, prob):
            np.testing.assert_almost_equal(np.sum(prob, axis=1), 1)
            np.testing.assert_array_equal(
                np.choose(val.astype(int), (prob[:, 0], prob[:, 1])),
                np.max(prob, axis=1))

        for lrn in [LogisticRegressionLearner, TreeLearner]:  # skl and non-skl
            model0 = lrn()(self.iris0)
            valp0 = model0(self.iris0)
            model1 = lrn()(self.iris1)
            valp1 = model1(self.iris1)
            model2 = lrn()(self.iris2)
            valp2 = model2(self.iris2)

            val1, prob1 = model0(self.iris1, model0.ValueProbs)
            np.testing.assert_array_equal(val1[valp0 == 1], 1)
            np.testing.assert_array_equal(prob1[valp0 == 1, 0], 0)
            np.testing.assert_array_equal(prob1[valp0 == 1, 1], 1)
            test_val_prob(val1, prob1)

            val2, prob2 = model0(self.iris2, model0.ValueProbs)
            np.testing.assert_array_equal(val2[valp0 == 1], 1)
            np.testing.assert_array_equal(prob2[valp0 == 1, 0], 0)
            np.testing.assert_array_equal(prob2[valp0 == 1, 1], 1)
            np.testing.assert_almost_equal(np.sum(prob2, axis=1), 1)
            test_val_prob(val2, prob2)

            val0, prob0 = model1(self.iris0, model1.ValueProbs)
            np.testing.assert_array_equal(val0[valp1 == 1], 1)
            np.testing.assert_array_equal(prob0[valp1 == 1, 0], 0)
            np.testing.assert_array_equal(prob0[valp1 == 1, 1], 1)
            np.testing.assert_almost_equal(np.sum(prob0, axis=1), 1)
            test_val_prob(val0, prob0)

            val2, prob2 = model1(self.iris2, model1.ValueProbs)
            np.testing.assert_array_equal(val2[valp1 == 0], 0)
            np.testing.assert_array_equal(prob2[valp1 == 0, 0], 1)
            np.testing.assert_array_equal(prob2[valp1 == 0, 1], 0)
            np.testing.assert_almost_equal(np.sum(prob2, axis=1), 1)
            test_val_prob(val2, prob2)

            val0, prob0 = model2(self.iris0, model2.ValueProbs)
            np.testing.assert_array_equal(val0[valp2 == 1], 0)
            np.testing.assert_array_equal(prob0[valp2 == 1, 0], 1)
            np.testing.assert_array_equal(prob0[valp2 == 1, 1], 0)
            np.testing.assert_almost_equal(np.sum(prob0, axis=1), 1)
            test_val_prob(val0, prob0)

            val1, prob1 = model2(self.iris1, model2.ValueProbs)
            np.testing.assert_array_equal(val1[valp2 == 0], 0)
            np.testing.assert_array_equal(prob1[valp2 == 0, 0], 1)
            np.testing.assert_array_equal(prob1[valp2 == 0, 1], 0)
            np.testing.assert_almost_equal(np.sum(prob1, axis=1), 1)
            test_val_prob(val1, prob1)

    def test_no_common_values(self):
        abc = DiscreteVariable("iris", values=tuple("abc"))
        iris_abc = Table.from_numpy(
            Domain(self.iris.domain.attributes, abc),
            self.iris.X, self.iris.Y)
        for lrn in [LogisticRegressionLearner,
                    TreeLearner]:  # skl and non-skl
            model = lrn()(self.iris)
            val, prob = model(iris_abc, model.ValueProbs)
            self.assertTrue(np.all(val >= 0))
            self.assertTrue(np.all(val <= 2))
            np.testing.assert_array_equal(prob, 1 / 3)

    def test_sparse_matrix(self):
        iris_sparse = self.iris.to_sparse()
        for lrn in [LogisticRegressionLearner, TreeLearner]:  # skl and non-skl
            model = lrn()(iris_sparse)
            pred = model(iris_sparse.X.tocsc())
            self.assertTupleEqual((len(self.iris),), pred.shape)
            pred = model(iris_sparse.X.tolil())
            self.assertTupleEqual((len(self.iris),), pred.shape)
            pred = model(iris_sparse.X.tocoo())
            self.assertTupleEqual((len(self.iris),), pred.shape)


if __name__ == '__main__':
    unittest.main()
