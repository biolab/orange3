# pylint: disable=missing-docstring

import numpy as np
from scipy import sparse

from Orange.classification import KNNLearner as KNNClassificationLearner
from Orange.data import (
    ContinuousVariable,
    DiscreteVariable,
    Domain,
    Table,
)
from Orange.modelling import KNNLearner
from Orange.preprocess import Randomize
from Orange.widgets.model.owknn import (
    L2KNNClassificationLearner,
    L2KNNLearner,
    NormalizeInstancesL2,
    OWKNNLearner,
)
from Orange.widgets.tests.base import (
    ParameterMapping,
    WidgetLearnerTestMixin,
    WidgetTest,
)

class TestOWKNNLearner(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(
            OWKNNLearner, stored_settings={"auto_apply": False})
        self.init()
        self.parameters = [
            ParameterMapping('metric', self.widget.metrics_combo,
                             self.widget.metrics),
            ParameterMapping('weights', self.widget.weights_combo,
                             self.widget.weights),
            ParameterMapping('n_neighbors', self.widget.n_neighbors_spin),
        ]

    def test_l2_learner_used_only_for_euclidean(self):
        self.assertEqual(
            self.widget.metrics[self.widget.metric_index],
            "euclidean",
        )
        self.assertTrue(self.widget.normalize_l2_checkbox.isEnabled())
        self.assertFalse(self.widget.normalize_instances_l2)

        learner = self.widget.create_learner()
        self.assertIsInstance(learner, KNNLearner)
        self.assertNotIsInstance(learner, L2KNNLearner)

        self.widget.normalize_l2_checkbox.click()
        self.assertTrue(self.widget.normalize_instances_l2)

        learner = self.widget.create_learner()
        self.assertIsInstance(learner, L2KNNLearner)

        self.widget.metric_index = 1
        self.widget._on_metric_changed()

        self.assertFalse(self.widget.normalize_l2_checkbox.isEnabled())
        self.assertFalse(self.widget.normalize_instances_l2)

        learner = self.widget.create_learner()
        self.assertIsInstance(learner, KNNLearner)
        self.assertNotIsInstance(learner, L2KNNLearner)

    def test_l2_checkbox_reenabled_when_returning_to_euclidean(self):
        self.widget.metric_index = 1
        self.widget._on_metric_changed()
        self.assertFalse(self.widget.normalize_l2_checkbox.isEnabled())

        self.widget.metric_index = 0
        self.widget._on_metric_changed()
        self.assertTrue(self.widget.normalize_l2_checkbox.isEnabled())
        self.assertFalse(self.widget.normalize_instances_l2)

    def test_l2_normalizes_dense_instances(self):
        data = Table.from_numpy(
            None,
            np.array([
                [3.0, 4.0],
                [0.0, 0.0],
                [1.0, 0.0],
            ])
        )

        normalized = NormalizeInstancesL2()(data)

        np.testing.assert_allclose(
            normalized.X,
            np.array([
                [0.6, 0.8],
                [0.0, 0.0],
                [1.0, 0.0],
            ])
        )

    def test_l2_normalizes_sparse_instances(self):
        data = Table.from_numpy(
            None,
            sparse.csr_matrix([
                [3.0, 4.0],
                [0.0, 0.0],
            ])
        )

        normalized = NormalizeInstancesL2()(data)

        self.assertTrue(sparse.issparse(normalized.X))
        np.testing.assert_allclose(
            normalized.X.toarray(),
            np.array([
                [0.6, 0.8],
                [0.0, 0.0],
            ])
        )

    def test_l2_does_not_modify_input_data(self):
        data = Table.from_numpy(
            None,
            np.array([[3.0, 4.0]])
        )
        original = data.X.copy()

        NormalizeInstancesL2()(data)

        np.testing.assert_array_equal(data.X, original)

    def test_l2_applied_after_default_preprocessing(self):
        domain = Domain(
            [
                ContinuousVariable("continuous"),
                DiscreteVariable("categorical", values=("a", "b")),
            ],
            DiscreteVariable("target", values=("no", "yes")),
        )
        data = Table.from_numpy(
            domain,
            np.array([
                [3.0, 0.0],
                [np.nan, 1.0],
                [4.0, 0.0],
            ]),
            np.array([0.0, 1.0, 0.0]),
        )

        base_learner = KNNClassificationLearner()
        expected = NormalizeInstancesL2()(
            base_learner.preprocess(data)
        )

        l2_learner = L2KNNClassificationLearner()
        result = l2_learner.preprocess(data)

        for variable in result.domain.attributes:
            self.assertTrue(variable.is_continuous)

        self.assertFalse(np.isnan(result.X).any())
        np.testing.assert_allclose(result.X, expected.X)

    def test_l2_preserves_input_preprocessor(self):
        randomize = Randomize()
        self.widget.preprocessors = randomize
        self.widget.normalize_instances_l2 = True

        learner = self.widget.create_learner()

        self.assertIsInstance(learner, L2KNNLearner)
        self.assertEqual(len(learner.preprocessors), 1)
        self.assertIs(learner.preprocessors[0], randomize)
        self.assertFalse(
            any(
                isinstance(preprocessor, NormalizeInstancesL2)
                for preprocessor in learner.preprocessors
            )
        )

