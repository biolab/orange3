import unittest
from unittest.mock import patch

import numpy as np

from Orange.modelling import ColumnLearner, ColumnModel
from Orange.data import DiscreteVariable, ContinuousVariable, Domain, Table


class ColumnTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.domain = Domain([DiscreteVariable("d1", values=["a", "b"]),
                             DiscreteVariable("d2", values=["c", "d"]),
                             DiscreteVariable("d3", values=["d", "c"]),
                             ContinuousVariable("c1"),
                             ContinuousVariable("c2")
                            ],
                            DiscreteVariable("cls", values=["c", "d"]),
                            [DiscreteVariable("m1", values=["a", "b"]),
                             DiscreteVariable("m2", values=["d"]),
                             ContinuousVariable("c3")]
                            )
        cls.data = Table.from_numpy(
            cls.domain,
            np.array([[0, 0, 0, 1, 0.5],
                      [0, 1, 1, 0.25, -3],
                      [1, 0, np.nan, np.nan, np.nan]]),
            np.array([0, 1, 1]),
            np.array([[0, 0, 2],
                      [1, 0, 8],
                      [np.nan, np.nan, 5]])
        )

    @patch("Orange.classification.column.ColumnModel")
    def test_fit_storage(self, clsfr):
        learner = ColumnLearner(self.domain.class_var, self.domain["d2"])
        self.assertEqual(learner.name, "column 'd2'")
        learner.fit_storage(self.data)
        clsfr.assert_called_with(self.domain.class_var, self.domain["d2"], None, None)

        learner = ColumnLearner(self.domain.class_var, self.domain["c3"])
        learner.fit_storage(self.data)
        clsfr.assert_called_with(self.domain.class_var, self.domain["c3"], None, None)

        learner = ColumnLearner(self.domain.class_var, self.domain["c3"], 42, 3.5)
        self.assertEqual(learner.name, "column 'c3'")
        learner.fit_storage(self.data)
        clsfr.assert_called_with(self.domain.class_var, self.domain["c3"], 42, 3.5)


    def test_predict_discrete(self):
        # Just copy
        model = ColumnModel(self.domain.class_var, self.domain["d2"])
        self.assertEqual(model.name, "column 'd2'")
        classes, probs = model(self.data, model.ValueProbs)
        np.testing.assert_equal(classes, [0, 1, 0])
        np.testing.assert_equal(probs, [[1, 0], [0, 1], [1, 0]])

        # Values are not in the same order -> map
        model = ColumnModel(self.domain.class_var, self.domain["d3"])
        classes, probs = model(self.data, model.ValueProbs)
        np.testing.assert_equal(classes, [1, 0, np.nan])
        np.testing.assert_equal(probs, [[0, 1], [1, 0], [0.5, 0.5]])

        # Not in the same order, and one is missing -> map
        model = ColumnModel(self.domain.class_var, self.domain["m2"])
        classes, probs = model(self.data, model.ValueProbs)
        np.testing.assert_equal(classes, [1, 1, np.nan])
        np.testing.assert_equal(probs, [[0, 1], [0, 1], [0.5, 0.5]])

        # Non-binary class
        domain = Domain(
            self.domain.attributes,
            DiscreteVariable("cls", values=["a", "c", "b", "d", "e"]))
        data = Table.from_numpy(domain, self.data.X, self.data.Y)
        model = ColumnModel(domain.class_var, domain["d3"])
        classes, probs = model(data, model.ValueProbs)
        np.testing.assert_equal(classes, [3, 1, np.nan])
        np.testing.assert_almost_equal(
            probs,
            np.array([[0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0],
                      [0.2, 0.2, 0.2, 0.2, 0.2]]))

    def test_predict_as_direct_probs(self):
        model = ColumnModel(self.domain.class_var, self.domain["c1"])
        self.assertEqual(model.name, "column 'c1'")
        classes, probs = model(self.data, model.ValueProbs)
        np.testing.assert_equal(classes, [1, 0, np.nan])
        np.testing.assert_equal(probs, [[0, 1], [0.75, 0.25], [0.5, 0.5]])

        model = ColumnModel(self.domain.class_var, self.domain["c2"])
        self.assertRaises(ValueError, model, self.data)

        model = ColumnModel(self.domain.class_var, self.domain["c3"])
        self.assertRaises(ValueError, model, self.data)

    def test_predict_with_logistic(self):
        model = ColumnModel(
            self.domain.class_var, self.domain["c1"], 0.5, 3)
        classes, probs = model(self.data, model.ValueProbs)
        np.testing.assert_equal(classes, [1, 0, np.nan])
        np.testing.assert_almost_equal(
            probs[:, 1], [1 / (1 + np.exp(-3 * (1 - 0.5))),
                          1 / (1 + np.exp(-3 * (0.25 - 0.5))),
                          0.5])
        np.testing.assert_equal(probs[:, 0], 1 - probs[:, 1])

        model = ColumnModel(
            self.domain.class_var, self.domain["c2"], 0.5, 3)
        classes, probs = model(self.data, model.ValueProbs)
        np.testing.assert_equal(classes, [0, 0, np.nan])
        np.testing.assert_almost_equal(
            probs[:, 1], [1 / (1 + np.exp(-3 * (0.5 - 0.5))),
                          1 / (1 + np.exp(-3 * (-3 - 0.5))),
                          0.5])
        np.testing.assert_equal(probs[:, 0], 1 - probs[:, 1])


if __name__ == "__main__":
    unittest.main()
