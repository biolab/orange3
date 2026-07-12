# pylint: disable=missing-docstring

import numpy as np
from scipy import sparse

from Orange.data import Table
from Orange.widgets.model.owknn import OWKNNLearner, NormalizeInstancesL2
from Orange.widgets.tests.base import (
    WidgetTest,
    WidgetLearnerTestMixin,
    ParameterMapping,
)
from Orange.preprocess import Randomize, PreprocessorList


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

    def test_l2_preprocessor_added_only_for_euclidean(self):
        self.assertEqual(self.widget.metrics[self.widget.metric_index], "euclidean")
        self.assertTrue(self.widget.normalize_l2_checkbox.isEnabled())
        self.assertFalse(self.widget.normalize_instances_l2)

        learner = self.widget.create_learner()
        self.assertFalse(any(isinstance(pp, NormalizeInstancesL2)
                             for pp in learner.preprocessors))

        self.widget.normalize_l2_checkbox.click()
        self.assertTrue(self.widget.normalize_instances_l2)
        learner = self.widget.create_learner()
        self.assertTrue(any(isinstance(pp, NormalizeInstancesL2)
                            for pp in learner.preprocessors))

        self.widget.metric_index = 1
        self.widget._on_metric_changed()
        self.assertFalse(self.widget.normalize_l2_checkbox.isEnabled())
        self.assertFalse(self.widget.normalize_instances_l2)
        learner = self.widget.create_learner()
        self.assertFalse(any(isinstance(pp, NormalizeInstancesL2)
                             for pp in learner.preprocessors))

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

    def test_l2_is_appended_to_input_preprocessor(self):
        randomize = Randomize()
        self.widget.preprocessors = randomize
        self.widget.normalize_instances_l2 = True

        learner = self.widget.create_learner()

        self.assertEqual(len(learner.preprocessors), 1)
        preprocessor_list = learner.preprocessors[0]

        self.assertIsInstance(preprocessor_list, PreprocessorList)
        self.assertEqual(preprocessor_list.preprocessors[0], randomize)
        self.assertIsInstance(
            preprocessor_list.preprocessors[1],
            NormalizeInstancesL2,
        )

    def test_l2_is_appended_to_preprocessor_list(self):
        randomize = Randomize()
        preprocessors = PreprocessorList([randomize])
        self.widget.preprocessors = preprocessors
        self.widget.normalize_instances_l2 = True

        learner = self.widget.create_learner()

        self.assertEqual(len(learner.preprocessors), 1)
        preprocessor_list = learner.preprocessors[0]

        self.assertIsInstance(preprocessor_list, PreprocessorList)
        self.assertEqual(preprocessor_list.preprocessors[0], randomize)
        self.assertIsInstance(
            preprocessor_list.preprocessors[1],
            NormalizeInstancesL2,
        )
