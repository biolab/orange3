import os

import unittest
from unittest.mock import patch, Mock, call

import numpy as np
import scipy.sparse as sp

import openTSNE.affinity
from Orange.data import DiscreteVariable, ContinuousVariable, Domain, Table
from Orange.distance import Euclidean
from Orange.misc import DistMatrix
from Orange.preprocess import Normalize
from Orange.projection import manifold, TSNE
from Orange.projection.manifold import TSNEModel
from Orange.widgets.tests.base import (
    WidgetTest, WidgetOutputsTestMixin, ProjectionWidgetTestMixin
)
from Orange.widgets.tests.utils import simulate
from Orange.widgets.unsupervised.owtsne import OWtSNE, TSNERunner, Task, prepare_tsne_obj


class DummyTSNE(manifold.TSNE):
    def compute_affinities(self, X):

        class DummyAffinities(openTSNE.affinity.Affinities):
            def __init__(self, data=None, *args, **kwargs):
                n_samples = data.shape[0]
                self.P = sp.random(n_samples, n_samples, density=0.1)
                self.P /= self.P.sum()

            def to_new(self, data, return_distances=False):
                ones = np.ones((len(data), 2), float)
                if return_distances:
                    return ones, ones
                return ones

        return DummyAffinities(X)

    def compute_initialization(self, X):
        return np.ones((X.shape[0], 2), float)

    def fit(self, X, Y=None):
        return np.ones((X.shape[0], 2), float)


class DummyTSNEModel(manifold.TSNEModel):
    def transform(self, X, **kwargs):
        return np.ones((X.shape[0], 2), float)

    def optimize(self, n_iter, **kwargs):
        return self


class TestOWtSNE(WidgetTest, ProjectionWidgetTestMixin, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)
        cls.same_input_output_domain = False

        cls.signal_name = OWtSNE.Inputs.data
        cls.signal_data = cls.data

        cls.iris = Table("iris")
        cls.iris_distances = Euclidean(cls.iris)
        cls.housing = Table("housing")[:200]
        cls.housing_distances = Euclidean(cls.housing)

        # Load distance-only DistMatrix, without accompanying `.row_items`
        my_dir = os.path.dirname(__file__)
        datasets_dir = os.path.join(my_dir, '..', '..', '..', 'datasets')
        cls.datasets_dir = os.path.realpath(datasets_dir)
        cls.towns = DistMatrix.from_file(
            os.path.join(cls.datasets_dir, "slovenian-towns.dst")
        )

    def setUp(self):
        # For almost all the tests, we won't need to verify t-SNE validity and
        # the tests will run much faster if we dummy them out
        self.tsne = patch("Orange.projection.manifold.TSNE", new=DummyTSNE)
        self.tsne_model = patch("Orange.projection.manifold.TSNEModel", new=DummyTSNEModel)
        self.tsne.start()
        self.tsne_model.start()

        self.widget = self.create_widget(OWtSNE, stored_settings={"multiscale": False})

        self.class_var = DiscreteVariable("Stage name", values=("STG1", "STG2"))
        self.attributes = [ContinuousVariable("GeneName" + str(i)) for i in range(5)]
        self.domain = Domain(self.attributes, class_vars=self.class_var)
        self.empty_domain = Domain([], class_vars=self.class_var)

    def tearDown(self):
        # Some tests may not wait for the widget to finish, and the patched
        # methods might be unpatched before the widget finishes, resulting in
        # a very confusing crash.
        self.widget.onDeleteWidget()
        try:
            self.restore_mocked_functions()
        # If `restore_mocked_functions` was called in the test itself, stopping
        # the patchers here will raise a RuntimeError
        except RuntimeError as e:
            if str(e) != "stop called on unstarted patcher":
                raise e

    def restore_mocked_functions(self):
        self.tsne.stop()
        self.tsne_model.stop()

    def test_wrong_input(self):
        # no data
        data = None
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_stop_blocking()
        self.assertIsNone(self.widget.data)

        # <2 rows
        data = Table.from_list(self.domain, [[1, 2, 3, 4, 5, 'STG1']])
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_stop_blocking()
        self.assertIsNone(self.widget.data)
        self.assertTrue(self.widget.Error.not_enough_rows.is_shown())

        # no attributes
        data = Table.from_list(self.empty_domain, [['STG1']] * 2)
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_stop_blocking()
        self.assertIsNone(self.widget.data)
        self.assertTrue(self.widget.Error.not_enough_cols.is_shown())

        # one attributes
        data = Table.from_list(self.empty_domain, [[1, 'STG1'],
                                                   [2, 'STG1']])
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_stop_blocking()
        self.assertIsNone(self.widget.data)
        self.assertTrue(self.widget.Error.not_enough_cols.is_shown())

        # constant data
        data = Table.from_list(self.domain, [[1, 2, 3, 4, 5, 'STG1']] * 2)
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_stop_blocking()
        self.assertIsNone(self.widget.data)
        self.assertTrue(self.widget.Error.constant_data.is_shown())

        # correct input
        data = Table.from_list(self.domain, [[1, 2, 3, 4, 5, 'STG1'],
                                             [5, 4, 3, 2, 1, 'STG1']])
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_stop_blocking()
        self.assertIsNotNone(self.widget.data)
        self.assertFalse(self.widget.Error.not_enough_rows.is_shown())
        self.assertFalse(self.widget.Error.not_enough_cols.is_shown())
        self.assertFalse(self.widget.Error.constant_data.is_shown())

    def test_input(self):
        data = Table.from_list(self.domain, [[1, 1, 1, 1, 1, 'STG1'],
                                             [2, 2, 2, 2, 2, 'STG1'],
                                             [4, 4, 4, 4, 4, 'STG2'],
                                             [5, 5, 5, 5, 5, 'STG2']])

        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_stop_blocking()

    def test_attr_models(self):
        """Check possible values for 'Color', 'Shape', 'Size' and 'Label'"""
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_stop_blocking()
        controls = self.widget.controls
        for var in self.data.domain.class_vars + self.data.domain.metas:
            self.assertIn(var, controls.attr_color.model())
            self.assertIn(var, controls.attr_label.model())
            if var.is_continuous:
                self.assertIn(var, controls.attr_size.model())
                self.assertNotIn(var, controls.attr_shape.model())
            if var.is_discrete:
                self.assertNotIn(var, controls.attr_size.model())
                self.assertIn(var, controls.attr_shape.model())

    def test_multiscale_changed_updates_ui(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertFalse(self.widget.controls.multiscale.isChecked())
        self.assertTrue(self.widget.perplexity_spin.isEnabled())
        self.widget.controls.multiscale.setChecked(True)
        self.assertFalse(self.widget.perplexity_spin.isEnabled())

        settings = self.widget.settingsHandler.pack_data(self.widget)
        w = self.create_widget(OWtSNE, stored_settings=settings)
        self.send_signal(w.Inputs.data, self.data, widget=w)
        self.assertTrue(w.controls.multiscale.isChecked())
        self.assertFalse(w.perplexity_spin.isEnabled())
        w.onDeleteWidget()

    def test_normalize_data(self):
        # Normalization should be checked by default
        self.assertTrue(self.widget.controls.normalize.isChecked())
        with patch("Orange.preprocess.preprocess.Normalize", wraps=Normalize) as normalize:
            self.send_signal(self.widget.Inputs.data, self.data)
            self.assertTrue(self.widget.controls.normalize.isEnabled())
            self.wait_until_finished()
            normalize.assert_called_once()

        # Disable checkbox
        self.widget.controls.normalize.setChecked(False)
        self.assertFalse(self.widget.controls.normalize.isChecked())
        with patch("Orange.preprocess.preprocess.Normalize", wraps=Normalize) as normalize:
            self.send_signal(self.widget.Inputs.data, self.data)
            self.assertTrue(self.widget.controls.normalize.isEnabled())
            self.wait_until_finished()
            normalize.assert_not_called()

        # Normalization shouldn't work on sparse data
        self.widget.controls.normalize.setChecked(True)
        self.assertTrue(self.widget.controls.normalize.isChecked())

        sparse_data = self.data.to_sparse()
        with patch("Orange.preprocess.preprocess.Normalize", wraps=Normalize) as normalize:
            self.send_signal(self.widget.Inputs.data, sparse_data)
            self.assertFalse(self.widget.controls.normalize.isEnabled())
            self.wait_until_finished()
            normalize.assert_not_called()

    @patch("Orange.projection.manifold.TSNEModel.optimize")
    def test_exaggeration_is_passed_through_properly(self, optimize):
        def _check_exaggeration(call, exaggeration):
            # Check the last call to `optimize`, so we catch one during the
            # regular regime
            _, _, kwargs = call.mock_calls[-1]
            self.assertIn("exaggeration", kwargs)
            self.assertEqual(kwargs["exaggeration"], exaggeration)

        # Since optimize needs to return a valid TSNEModel instance and it is
        # impossible to return `self` in a mock, we'll prepare this one ahead
        # of time and use this
        optimize.return_value = DummyTSNE()(self.data)

        # Set value to 1
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.run_button.clicked.emit()  # stop initial run
        self.wait_until_stop_blocking()
        self.widget.controls.exaggeration.setValue(1)
        self.widget.run_button.clicked.emit()  # run with exaggeration 1
        self.wait_until_finished()
        _check_exaggeration(optimize, 1)

        # Reset and clear state
        self.send_signal(self.widget.Inputs.data, None)
        optimize.reset_mock()

        # Change to 3
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.run_button.clicked.emit()  # stop initial run
        self.wait_until_stop_blocking()
        self.widget.controls.exaggeration.setValue(3)
        self.widget.run_button.clicked.emit()  # run with exaggeration 1
        self.wait_until_finished()
        _check_exaggeration(optimize, 3)

    def test_plot_once(self):
        """Test if data is plotted only once but committed on every input change"""
        self.widget.setup_plot = Mock()
        self.widget.commit.deferred = self.widget.commit.now = Mock()

        self.send_signal(self.widget.Inputs.data, self.data)
        # TODO: The base widget immediately calls `setup_plot` and `commit`
        # even though there's nothing to show yet. Unfortunately, fixing this
        # would require changing `OWDataProjectionWidget` in some strange way,
        # so as a temporary fix, we reset the mocks, so they reflect the calls
        # when the result was available.
        self.widget.setup_plot.reset_mock()
        self.widget.commit.deferred.reset_mock()
        self.wait_until_finished()

        self.widget.setup_plot.assert_called_once()
        self.widget.commit.deferred.assert_called_once()

        self.widget.commit.deferred.reset_mock()
        self.send_signal(self.widget.Inputs.data_subset, self.data[::10])
        self.wait_until_stop_blocking()

        self.widget.setup_plot.assert_called_once()
        self.widget.commit.deferred.assert_called_once()

    def test_modified_info_message_behaviour(self):
        """Information messages should be cleared if the data changes or if
        the data is set to None."""
        self.assertFalse(
            self.widget.Information.modified.is_shown(),
            "The modified info message should be hidden by default"
        )

        self.widget.controls.multiscale.setChecked(False)
        self.assertFalse(
            self.widget.Information.modified.is_shown(),
            "The modified info message should be hidden even after toggling "
            "options if no data is on input"
        )

        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_stop_blocking()
        self.assertFalse(
            self.widget.Information.modified.is_shown(),
            "The modified info message should be hidden after the widget "
            "computes the embedding"
        )

        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_stop_blocking()
        self.assertFalse(
            self.widget.Information.modified.is_shown(),
            "The modified info message should be hidden when reloading the "
            "same data set and no previous messages were shown"
        )

        self.widget.controls.multiscale.setChecked(True)
        self.assertTrue(
            self.widget.Information.modified.is_shown(),
            "The modified info message should be shown when a setting is "
            "changed, but the embedding is not recomputed"
        )

        self.send_signal(self.widget.Inputs.data, Table("housing"))
        self.wait_until_stop_blocking()
        self.assertFalse(
            self.widget.Information.modified.is_shown(),
            "The information message was not cleared on new data"
        )
        # Flip; the info message should now be shown again.
        self.widget.controls.multiscale.setChecked(False)
        assert self.widget.Information.modified.is_shown()  # sanity check

        self.send_signal(self.widget.Inputs.data, None)
        self.wait_until_stop_blocking()
        self.assertFalse(
            self.widget.Information.modified.is_shown(),
            "The information message was not cleared on no data"
        )

        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_stop_blocking()
        self.assertFalse(
            self.widget.Information.modified.is_shown(),
            "The modified info message should be hidden after the widget "
            "computes the embedding"
        )

    def test_invalidation_flow(self):
        # pylint: disable=protected-access
        w = self.widget
        # Setup widget: send data to input with global structure "off", then
        # set global structure "on" (after the embedding is computed)
        w.controls.multiscale.setChecked(False)
        self.send_signal(w.Inputs.data, self.data)

        # By default, t-SNE is smart and disables PCA preprocessing if the
        # number of features is too low. Since we are testing with the iris
        # data set, we want to force t-SNE to use PCA preprocessing.
        w.controls.use_pca_preprocessing.setChecked(True)
        self.widget.run_button.click()

        self.wait_until_finished()
        self.assertFalse(self.widget.Information.modified.is_shown())
        # All the embedding components should be computed
        self.assertIsNotNone(w.preprocessed_data)
        self.assertIsNotNone(w.normalized_data)
        self.assertIsNotNone(w.pca_projection)
        self.assertIsNotNone(w.affinities)
        self.assertIsNotNone(w.tsne_embedding)
        # All the invalidation flags should be set to false
        self.assertFalse(w._invalidated.preprocessed_data)
        self.assertFalse(w._invalidated.normalized_data)
        self.assertFalse(w._invalidated.pca_projection)
        self.assertFalse(w._invalidated.affinities)
        self.assertFalse(w._invalidated.tsne_embedding)

        # Trigger invalidation
        w.controls.multiscale.setChecked(True)
        self.assertTrue(self.widget.Information.modified.is_shown())
        # Setting `multiscale` to true should set the invalidate flags for
        # the affinities and embedding, but not the pca_projection
        self.assertFalse(w._invalidated.preprocessed_data)
        self.assertFalse(w._invalidated.normalized_data)
        self.assertFalse(w._invalidated.pca_projection)
        self.assertTrue(w._invalidated.affinities)
        self.assertTrue(w._invalidated.tsne_embedding)

        # The flags should now be set, but the embedding should still be
        # available when selecting a subset of data and such
        self.assertIsNotNone(w.preprocessed_data)
        self.assertIsNotNone(w.normalized_data)
        self.assertIsNotNone(w.pca_projection)
        self.assertIsNotNone(w.affinities)
        self.assertIsNotNone(w.tsne_embedding)

        # We should still be able to send a data subset to the input and have
        # the points be highlighted
        self.send_signal(w.Inputs.data_subset, self.data[:10])
        self.wait_until_finished()
        subset = [brush.color().name() == "#46befa" for brush in
                  w.graph.scatterplot_item.data["brush"][:10]]
        other = [brush.color().name() == "#000000" for brush in
                 w.graph.scatterplot_item.data["brush"][10:]]
        self.assertTrue(all(subset))
        self.assertTrue(all(other))

        # Clear the data subset
        self.send_signal(w.Inputs.data_subset, None)

        # Run the optimization
        self.widget.run_button.click()
        self.wait_until_finished()
        # All of the inavalidation flags should have been cleared
        self.assertFalse(w._invalidated)

    def test_pca_preprocessing_warning_with_large_number_of_features(self):
        self.assertFalse(
            self.widget.Warning.consider_using_pca_preprocessing.is_shown(),
            "The PCA warning should be hidden by default"
        )

        self.widget.controls.use_pca_preprocessing.setChecked(False)
        self.assertFalse(
            self.widget.Warning.consider_using_pca_preprocessing.is_shown(),
            "The PCA warning should be hidden even after toggling options if "
            "no data is on input"
        )

        # Setup data classes
        x_small = np.random.normal(0, 1, size=(50, 4))
        data_small = Table.from_numpy(Domain.from_numpy(x_small), x_small)
        x_large = np.random.normal(0, 1, size=(50, 250))
        data_large = Table.from_numpy(Domain.from_numpy(x_large), x_large)

        # SMALL data with PCA preprocessing ENABLED
        self.send_signal(self.widget.Inputs.data, data_small)
        self.widget.controls.use_pca_preprocessing.setChecked(True)
        self.widget.run_button.click(), self.wait_until_stop_blocking()
        self.assertFalse(
            self.widget.Warning.consider_using_pca_preprocessing.is_shown(),
            "The PCA warning should be hidden when PCA preprocessing enabled "
            "when the data has <50 features"
        )

        # SMALL data with PCA preprocessing DISABLED
        self.send_signal(self.widget.Inputs.data, data_small)
        self.widget.controls.use_pca_preprocessing.setChecked(False)
        self.widget.run_button.click(), self.wait_until_stop_blocking()
        self.assertFalse(
            self.widget.Warning.consider_using_pca_preprocessing.is_shown(),
            "The PCA warning should be hidden with disabled PCA preprocessing "
            "when the data has <50 features"
        )

        # LARGE data with PCA preprocessing ENABLED
        self.send_signal(self.widget.Inputs.data, data_large)
        self.widget.controls.use_pca_preprocessing.setChecked(True)
        self.widget.run_button.click(), self.wait_until_stop_blocking()
        self.assertFalse(
            self.widget.Warning.consider_using_pca_preprocessing.is_shown(),
            "The PCA warning should be hidden when PCA preprocessing enabled "
            "when has >50 features"
        )

        # LARGE data with PCA preprocessing DISABLED
        self.send_signal(self.widget.Inputs.data, data_large)
        self.widget.controls.use_pca_preprocessing.setChecked(False)
        self.widget.run_button.click(), self.wait_until_stop_blocking()
        self.assertTrue(
            self.widget.Warning.consider_using_pca_preprocessing.is_shown(),
            "The PCA warning should be shown when PCA preprocessing disabled "
            "when the data has >50 features"
        )

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(
            self.widget.Information.modified.is_shown(),
            "The PCA warning should be cleared when problematic data is removed"
        )

    def test_distance_matrix_not_symmetric(self):
        w = self.widget
        self.assertFalse(w.Error.distance_matrix_not_symmetric.is_shown())

        self.send_signal(w.Inputs.distances, DistMatrix([[1, 2, 3], [4, 5, 6]]))
        self.assertTrue(w.Error.distance_matrix_not_symmetric.is_shown())

        self.send_signal(w.Inputs.distances, DistMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        self.assertTrue(w.Error.distance_matrix_not_symmetric.is_shown())

        self.send_signal(w.Inputs.distances, None)
        self.assertFalse(w.Error.distance_matrix_not_symmetric.is_shown())

    def test_matrix_too_small(self):
        w = self.widget
        self.assertFalse(w.Error.distance_matrix_too_small.is_shown())

        self.send_signal(w.Inputs.distances, DistMatrix([[1]]))
        self.assertTrue(w.Error.distance_matrix_too_small.is_shown())

        self.send_signal(w.Inputs.distances, None)
        self.assertFalse(w.Error.distance_matrix_too_small.is_shown())

    def test_mismatching_distances_and_data_size(self):
        w = self.widget
        self.assertFalse(w.Error.dimension_mismatch.is_shown())

        # Send incompatible combination
        self.send_signal(w.Inputs.distances, self.iris_distances)
        self.send_signal(w.Inputs.data, self.housing)
        self.wait_until_finished()
        self.assertTrue(w.Error.dimension_mismatch.is_shown())

        # Remove offending data
        self.send_signal(w.Inputs.data, None)
        self.wait_until_finished()
        self.assertFalse(w.Error.dimension_mismatch.is_shown())

        # Send incompatible combination
        self.send_signal(w.Inputs.distances, None)
        self.send_signal(w.Inputs.distances, self.iris_distances)
        self.send_signal(w.Inputs.data, self.housing)
        self.wait_until_finished()

        # Remove offending distance matrix
        self.send_signal(w.Inputs.distances, None)
        self.wait_until_finished()
        self.assertFalse(w.Error.dimension_mismatch.is_shown())

        # Clear any data
        self.send_signal(w.Inputs.distances, None)
        self.assertFalse(w.Error.dimension_mismatch.is_shown())

    def test_invalid_distance_matrix_with_valid_data_signal_1(self):
        """Provide valid data table and an invalid distance matrix at the
        same time."""
        w = self.widget
        self.assertFalse(w.Error.distance_matrix_too_small.is_shown())

        self.send_signal(w.Inputs.distances, DistMatrix([[1]]))
        self.send_signal(w.Inputs.data, self.iris)
        self.wait_until_finished()

        self.assertIsNone(w.graph.scatterplot_item)
        self.assertIsNone(self.get_output(w.Outputs.annotated_data))

    def test_invalid_distance_matrix_with_valid_data_signal_2(self):
        """Provide an invalid distance matrix, wait to finish, then provide a
        valid data table."""
        w = self.widget
        self.assertFalse(w.Error.distance_matrix_too_small.is_shown())

        self.send_signal(w.Inputs.distances, DistMatrix([[1]]))
        self.wait_until_finished()
        self.send_signal(w.Inputs.data, self.iris)
        self.wait_until_finished()

        self.assertIsNone(w.graph.scatterplot_item)
        self.assertIsNone(self.get_output(w.Outputs.annotated_data))

    def test_invalid_distance_matrix_with_valid_data_signal_3(self):
        """Provide a valid data table, wait to finish, then provide an
        invalid distance matrix."""
        w = self.widget
        self.assertFalse(w.Error.distance_matrix_too_small.is_shown())

        self.send_signal(w.Inputs.data, self.iris)
        self.wait_until_finished()
        self.send_signal(w.Inputs.distances, DistMatrix([[1]]))
        self.wait_until_finished()

        self.assertIsNone(w.graph.scatterplot_item)
        self.assertIsNone(self.get_output(w.Outputs.annotated_data))

    def test_valid_distance_matrix_with_mismatching_data_signal_1(self):
        """Provide valid distance matrix and a mismatching data table at the
        same time."""
        w = self.widget
        self.assertFalse(w.Error.distance_matrix_too_small.is_shown())

        self.send_signal(w.Inputs.distances, self.iris_distances)
        self.send_signal(w.Inputs.data, self.iris[:5])
        self.wait_until_finished()

        self.assertIsNone(w.graph.scatterplot_item)
        self.assertIsNone(self.get_output(w.Outputs.annotated_data))

    def test_valid_distance_matrix_with_mismatching_data_signal_2(self):
        """Provide valid distance matrix, wait to finish, then provide a
        mismatching data table."""
        w = self.widget
        self.assertFalse(w.Error.distance_matrix_too_small.is_shown())

        self.send_signal(w.Inputs.distances, self.iris_distances)
        self.wait_until_finished()
        self.send_signal(w.Inputs.data, self.iris[:5])
        self.wait_until_finished()

        self.assertIsNone(w.graph.scatterplot_item)
        self.assertIsNone(self.get_output(w.Outputs.annotated_data))

    def test_invalid_combination_followed_by_valid_combination(self):
        """Provide valid distance matrix and a mismatching data table at the
        same time."""
        w = self.widget
        self.assertFalse(w.Error.distance_matrix_too_small.is_shown())

        self.send_signal(w.Inputs.distances, self.iris_distances)
        self.send_signal(w.Inputs.data, self.iris[:5])
        self.wait_until_finished()

        self.assertIsNone(w.graph.scatterplot_item)
        self.assertIsNone(self.get_output(w.Outputs.annotated_data))

        self.send_signal(w.Inputs.data, self.iris)
        self.wait_until_finished()
        self.assertIsNotNone(w.graph.scatterplot_item)
        self.assertIsNotNone(self.get_output(w.Outputs.annotated_data))

    def test_invalid_combination_followed_by_valid_distance_only(self):
        """Provide valid distance matrix and a mismatching data table at the
        same time."""
        w = self.widget
        self.assertFalse(w.Error.distance_matrix_too_small.is_shown())

        self.send_signal(w.Inputs.distances, self.iris_distances)
        self.send_signal(w.Inputs.data, self.iris[:5])
        self.wait_until_finished()

        self.assertIsNone(w.graph.scatterplot_item)
        self.assertIsNone(self.get_output(w.Outputs.annotated_data))

        self.send_signal(w.Inputs.data, None)
        self.wait_until_finished()
        self.assertIsNotNone(w.graph.scatterplot_item)
        self.assertIsNotNone(self.get_output(w.Outputs.annotated_data))

    def test_invalid_combination_followed_by_valid_data_only(self):
        """Provide valid distance matrix and a mismatching data table at the
        same time."""
        w = self.widget
        self.assertFalse(w.Error.distance_matrix_too_small.is_shown())

        self.send_signal(w.Inputs.distances, DistMatrix([[1]]))
        self.send_signal(w.Inputs.data, self.iris)
        self.wait_until_finished()

        self.assertIsNone(w.graph.scatterplot_item)
        self.assertIsNone(self.get_output(w.Outputs.annotated_data))

        self.send_signal(w.Inputs.distances, None)
        self.wait_until_finished()
        self.assertIsNotNone(w.graph.scatterplot_item)
        self.assertIsNotNone(self.get_output(w.Outputs.annotated_data))

    def test_adding_data_table_to_distance_matrix_doesnt_trigger_rerun(self):
        """If the embedding is already constructed, and we just update the data
        signal, then we don't need to recompute the embedding."""
        w = self.widget

        with patch("Orange.widgets.unsupervised.owtsne.TSNERunner.run", wraps=TSNERunner.run) as runner:
            self.send_signal(w.Inputs.distances, Euclidean(self.housing[:150]))
            self.wait_until_finished()

            housing_colors = [
                brush.color().name() for brush in
                w.graph.scatterplot_item.data["brush"]
            ]

            self.send_signal(w.Inputs.data, self.iris)
            self.wait_until_finished()
            iris_colors = [
                brush.color().name() for brush in
                w.graph.scatterplot_item.data["brush"]
            ]

            # Ensure the colors have changed
            self.assertTrue(all(c1 != c2 for c1, c2 in zip(housing_colors, iris_colors)))
            # And that the embedding has not been recomputed
            runner.assert_called_once()

    def test_data_change_doesnt_crash(self):
        w = self.widget

        self.send_signal(w.Inputs.data, self.iris)
        self.wait_until_finished()

        self.send_signal(w.Inputs.data, self.housing)
        self.wait_until_finished()

        self.send_signal(w.Inputs.data, self.iris)
        self.wait_until_finished()

    def test_distance_change_doesnt_crash(self):
        w = self.widget

        self.send_signal(w.Inputs.distances, self.iris_distances)
        self.wait_until_finished()

        self.send_signal(w.Inputs.distances, self.housing_distances)
        self.wait_until_finished()

        self.send_signal(w.Inputs.distances, self.iris_distances)
        self.wait_until_finished()

    def test_distances_without_data_axis_0(self):
        w = self.widget
        signal_data = Euclidean(self.data, axis=0)
        signal_data.row_items = None
        self.send_signal(w.Inputs.distances, signal_data)

    def test_distances_without_data_axis_1(self):
        signal_data = Euclidean(self.data, axis=1)
        signal_data.row_items = None
        self.send_signal("Distances", signal_data)

    def test_data_table_with_no_attributes_with_distance_matrix_works(self):
        w = self.widget
        self.send_signal(w.Inputs.distances, self.towns)
        self.wait_until_finished()
        # No errors should be shown
        self.assertEqual(len(w.Error.active), 0)

    def test_controls_are_properly_disabled_with_distance_matrix_1(self):
        """Send both signals first, then disconnect distances."""
        w = self.widget

        disabled_fields = [
            "normalize", "use_pca_preprocessing", "pca_components",
            "initialization_method_idx", "distance_metric_idx",
        ]

        self.send_signal(w.Inputs.distances, self.iris_distances)
        self.send_signal(w.Inputs.data, self.iris)
        self.wait_until_finished()

        for field in disabled_fields:
            self.assertFalse(getattr(w.controls, field).isEnabled())

        # Remove distance matrix, can use data, so fields should be enabled
        self.send_signal(w.Inputs.distances, None)
        # Ensure PCA checkbox is ticked, to enable slider
        w.controls.use_pca_preprocessing.setChecked(True)
        self.wait_until_finished()

        for field in disabled_fields:
            self.assertTrue(getattr(w.controls, field).isEnabled())

    def test_controls_are_properly_disabled_with_distance_matrix_2(self):
        """Send distances first, disconnect distances, then send data."""
        w = self.widget

        disabled_fields = [
            "normalize", "use_pca_preprocessing", "pca_components",
            "initialization_method_idx", "distance_metric_idx",
        ]

        self.send_signal(w.Inputs.distances, self.iris_distances)
        self.wait_until_finished()

        for field in disabled_fields:
            self.assertFalse(getattr(w.controls, field).isEnabled())

        # Remove distance matrix
        self.send_signal(w.Inputs.distances, None)
        self.wait_until_finished()
        # Send data
        self.send_signal(w.Inputs.data, self.iris)
        # Ensure PCA checkbox is ticked, to enable slider
        w.controls.use_pca_preprocessing.setChecked(True)
        self.wait_until_finished()

        # Should now be enabled
        for field in disabled_fields:
            self.assertTrue(getattr(w.controls, field).isEnabled())

    def test_controls_ignored_by_distance_matrix_retain_values_on_table_signal(self):
        """The controls for `normalize`, `pca_preprocessing`, `metric`, and
        `initialization` are overridden/ignored when using a distance matrix
        signal. However, we want to remember their values when using Data
        table signals."""
        w = self.widget

        # SEND IRIS DATA
        # Set some parameters
        self.send_signal(w.Inputs.data, self.iris)
        w.normalize_cbx.setChecked(False)
        w.pca_preprocessing_cbx.setChecked(True)
        w.pca_component_slider.setValue(3)
        simulate.combobox_activate_index(w.initialization_combo, 0)
        simulate.combobox_activate_index(w.distance_metric_combo, 2)
        w.perplexity_spin.setValue(42)

        # Disconnect data
        self.send_signal(w.Inputs.data, None)

        # SEND IRIS DISTANCES
        self.send_signal(w.Inputs.distances, self.iris_distances)
        # Check that distance-related controls are disabled
        self.assertFalse(w.normalize_cbx.isEnabled())

        self.assertFalse(w.pca_preprocessing_cbx.isEnabled())
        self.assertFalse(w.pca_component_slider.isEnabled())

        self.assertFalse(w.initialization_combo.isEnabled())
        # Only spectral layout is supported when we have distances
        self.assertEqual(w.initialization_combo.currentText(), "Spectral")

        self.assertFalse(w.distance_metric_combo.isEnabled())
        self.assertEqual(w.distance_metric_combo.currentText(), "")

        self.assertTrue(w.perplexity_spin.isEnabled())
        self.assertEqual(w.perplexity_spin.value(), 42)

        # Disconnect distances
        self.send_signal(w.Inputs.distances, None)

        # SEND IRIS DATA
        # The distance-related settings should be restored from when we sent in
        # the data, and not overridden by the settings automatically set by the
        # widget when we passed in the distances signal
        self.send_signal(w.Inputs.data, self.iris)

        # Check that the parameters are restored
        self.assertTrue(w.normalize_cbx.isEnabled())
        self.assertFalse(w.normalize_cbx.isChecked())

        self.assertTrue(w.pca_preprocessing_cbx.isEnabled())
        self.assertTrue(w.pca_preprocessing_cbx.isChecked())
        self.assertTrue(w.pca_component_slider.isEnabled())

        self.assertTrue(w.initialization_combo.isEnabled())
        self.assertTrue(w.initialization_combo.currentText(), "PCA")

        self.assertTrue(w.distance_metric_combo.isEnabled())
        self.assertEqual(w.distance_metric_combo.currentIndex(), 2)

        self.assertTrue(w.perplexity_spin.isEnabled())
        self.assertEqual(w.perplexity_spin.value(), 42)

    def test_controls_are_properly_disabled_with_sparse_matrix(self):
        w = self.widget

        # Normalizing sparse matrix is disabled, since this would require
        # centering
        disabled_fields = ["normalize"]
        # PCA preprocessing and supported distance metrics are enable for sparse
        # matrices
        enabled_fields = [
            "use_pca_preprocessing", "distance_metric_idx", "initialization_method_idx"
        ]

        self.send_signal(w.Inputs.data, self.iris.to_sparse())
        self.wait_until_finished()

        for field in disabled_fields:
            self.assertFalse(getattr(w.controls, field).isEnabled())
        for field in enabled_fields:
            self.assertTrue(getattr(w.controls, field).isEnabled())

        # Send dense table, shoule enable disabled fields
        self.send_signal(w.Inputs.data, self.iris)
        self.wait_until_finished()

        for field in disabled_fields:
            self.assertTrue(getattr(w.controls, field).isEnabled())
        for field in enabled_fields:
            self.assertTrue(getattr(w.controls, field).isEnabled())

    def test_data_containing_nans(self):
        x = np.random.normal(0, 1, size=(150, 50))
        # Randomly sprinkle a few NaNs into the matrix
        num_nans = 20
        x[np.random.randint(0, 150, num_nans), np.random.randint(0, 50, num_nans)] = np.nan

        nan_data = Table.from_numpy(Domain.from_numpy(x), x)

        w = self.widget

        self.send_signal(w.Inputs.data, nan_data)
        self.assertTrue(w.controls.normalize.isChecked())
        self.assertTrue(w.controls.use_pca_preprocessing.isChecked())
        self.widget.run_button.click(), self.wait_until_finished()

        # Disable only normalization
        w.controls.normalize.setChecked(False)
        self.widget.run_button.click(), self.wait_until_finished()

        # Disable only PCA preprocessing
        w.controls.normalize.setChecked(True)
        w.controls.use_pca_preprocessing.setChecked(False)
        self.widget.run_button.click(), self.wait_until_finished()

        # Disable both normalization and PCA preprocessing
        w.controls.normalize.setChecked(False)
        w.controls.use_pca_preprocessing.setChecked(False)
        self.widget.run_button.click(), self.wait_until_finished()


class TestTSNERunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Table("iris")
        cls.distances = Euclidean(cls.data)

    def test_run_with_normalization_and_pca_preprocessing(self):
        state = Mock()
        state.is_interruption_requested = Mock(return_value=False)

        task = Task(
            normalize=True,
            use_pca_preprocessing=True,
            data=self.data,
            perplexity=30,
            initialization_method="pca",
            distance_metric="l2",
        )
        task = TSNERunner.run(task, state)

        self.assertEqual(len(state.set_status.mock_calls), 6)
        state.set_status.assert_has_calls([
            call("Preprocessing data..."),
            call("Normalizing data..."),
            call("Computing PCA..."),
            call("Finding nearest neighbors..."),
            call("Preparing initialization..."),
            call("Running optimization..."),
        ])

        self.assertIsInstance(task.normalized_data, Table)
        self.assertIsInstance(task.pca_projection, Table)
        self.assertIsInstance(task.tsne, TSNE)
        self.assertIsInstance(task.tsne_embedding, TSNEModel)

    def test_run_with_normalization(self):
        state = Mock()
        state.is_interruption_requested = Mock(return_value=False)

        task = Task(
            normalize=True,
            use_pca_preprocessing=False,
            data=self.data,
            initialization_method="pca",
            distance_metric="l2",
            perplexity=30,
        )
        task = TSNERunner.run(task, state)

        self.assertEqual(len(state.set_status.mock_calls), 5)
        state.set_status.assert_has_calls([
            call("Preprocessing data..."),
            call("Normalizing data..."),
            call("Finding nearest neighbors..."),
            call("Preparing initialization..."),
            call("Running optimization..."),
        ])

        self.assertIsNone(task.pca_projection, Table)

        self.assertIsInstance(task.normalized_data, Table)
        self.assertIsInstance(task.tsne, TSNE)
        self.assertIsInstance(task.tsne_embedding, TSNEModel)

    def test_run_with_pca_preprocessing(self):
        state = Mock()
        state.is_interruption_requested = Mock(return_value=False)

        task = Task(
            normalize=False,
            use_pca_preprocessing=True,
            data=self.data,
            initialization_method="pca",
            distance_metric="l2",
            perplexity=30,
        )
        task = TSNERunner.run(task, state)

        self.assertEqual(len(state.set_status.mock_calls), 5)
        state.set_status.assert_has_calls([
            call("Preprocessing data..."),
            call("Computing PCA..."),
            call("Finding nearest neighbors..."),
            call("Preparing initialization..."),
            call("Running optimization..."),
        ])
        state.set_status.assert_has_calls

        self.assertIsNone(task.normalized_data, Table)

        self.assertIsInstance(task.pca_projection, Table)
        self.assertIsInstance(task.tsne, TSNE)
        self.assertIsInstance(task.tsne_embedding, TSNEModel)

    def test_run_do_not_modify_model_inplace(self):
        state = Mock()
        state.is_interruption_requested.return_value = True

        task = Task(
            data=self.data,
            initialization_method="pca",
            distance_metric="l2",
            perplexity=30,
            multiscale=False,
            exaggeration=1,
        )
        # Run through all the steps to prepare the t-SNE object
        task.tsne = prepare_tsne_obj(
            task.data.X.shape[0],
            task.initialization_method,
            task.distance_metric,
            task.perplexity,
            task.multiscale,
            task.exaggeration,
        )
        TSNERunner.compute_normalization(task, state)
        TSNERunner.compute_pca(task, state)
        TSNERunner.compute_initialization(task, state)
        TSNERunner.compute_affinities(task, state)
        # Run the t-SNE iteration once to create the object
        TSNERunner.compute_tsne(task, state)

        # Make sure that running t-SNE for another iteration returns a new object
        tsne_obj_before = task.tsne_embedding
        state.reset_mock()
        TSNERunner.compute_tsne(task, state)
        tsne_obj_after = task.tsne_embedding

        state.set_partial_result.assert_called_once()
        self.assertIsNot(tsne_obj_before, tsne_obj_after)

    def test_run_with_distance_matrix(self):
        state = Mock()
        state.is_interruption_requested = Mock(return_value=False)

        task = Task(
            normalize=False,
            use_pca_preprocessing=False,
            distance_matrix=self.distances,
            perplexity=30,
            initialization_method="spectral",
            distance_metric="precomputed",
        )
        task = TSNERunner.run(task, state)

        self.assertEqual(len(state.set_status.mock_calls), 3)
        state.set_status.assert_has_calls([
            call("Finding nearest neighbors..."),
            call("Preparing initialization..."),
            call("Running optimization..."),
        ])

        self.assertIsNone(task.normalized_data)
        self.assertIsNone(task.pca_projection)
        self.assertIsInstance(task.initialization, np.ndarray)
        self.assertIsInstance(task.tsne, TSNE)
        self.assertIsInstance(task.tsne_embedding, TSNEModel)

    def test_task_validation(self):
        # distance matrix with no data table
        Task(
            normalize=True,
            use_pca_preprocessing=True,
            distance_matrix=self.distances,
            perplexity=30,
            initialization_method="spectral",
            distance_metric="precomputed",
        ).validate()

        # both distance matrix and data table are provided
        Task(
            normalize=True,
            use_pca_preprocessing=True,
            data=self.data,
            distance_matrix=self.distances,
            perplexity=30,
            initialization_method="spectral",
            distance_metric="precomputed",
        ).validate()

        # data table with no distance matrix
        Task(
            normalize=True,
            use_pca_preprocessing=True,
            data=self.data,
            perplexity=30,
            initialization_method="pca",
            distance_metric="cosine",
        ).validate()

        # distance_metric="precomputed" with no distance matrix
        with self.assertRaises(Task.ValidationError):
            Task(
                normalize=True,
                use_pca_preprocessing=True,
                data=self.data,
                perplexity=30,
                initialization_method="spectral",
                distance_metric="precomputed",
            ).validate()

        # initialization_method="pca" with distance matrix
        with self.assertRaises(Task.ValidationError):
            Task(
                normalize=True,
                use_pca_preprocessing=True,
                data=self.data,
                distance_matrix=self.distances,
                perplexity=30,
                initialization_method="pca",
                distance_metric="precomputed",
            ).validate()

        # distance_metric="l2" with distance matrix
        with self.assertRaises(Task.ValidationError):
            Task(
                normalize=True,
                use_pca_preprocessing=True,
                data=self.data,
                distance_matrix=self.distances,
                perplexity=30,
                initialization_method="spectral",
                distance_metric="l2",
            ).validate()

    def test_run_with_distance_matrix_ignores_preprocessing(self):
        state = Mock()
        state.is_interruption_requested = Mock(return_value=False)

        task = Task(
            normalize=True,
            use_pca_preprocessing=True,
            distance_matrix=self.distances,
            perplexity=30,
            initialization_method="spectral",
            distance_metric="precomputed",
        )
        task = TSNERunner.run(task, state)

        self.assertEqual(len(state.set_status.mock_calls), 3)
        state.set_status.assert_has_calls([
            call("Finding nearest neighbors..."),
            call("Preparing initialization..."),
            call("Running optimization..."),
        ])

        self.assertIsNone(task.normalized_data)
        self.assertIsNone(task.pca_projection)
        self.assertIsInstance(task.initialization, np.ndarray)
        self.assertIsInstance(task.tsne, TSNE)
        self.assertIsInstance(task.tsne_embedding, TSNEModel)

    def test_run_with_sparse_matrix_ignores_normalization(self):
        state = Mock()
        state.is_interruption_requested = Mock(return_value=False)

        task = Task(
            normalize=False,
            use_pca_preprocessing=True,
            data=self.data.to_sparse(),
            perplexity=30,
            initialization_method="spectral",
            distance_metric="cosine",
        )
        task = TSNERunner.run(task, state)
        self.assertEqual(len(state.set_status.mock_calls), 5)
        state.set_status.assert_has_calls([
            call("Preprocessing data..."),
            call("Computing PCA..."),
            call("Finding nearest neighbors..."),
            call("Preparing initialization..."),
            call("Running optimization..."),
        ])

        self.assertIsNone(task.normalized_data)
        self.assertIsInstance(task.pca_projection, Table)
        self.assertIsInstance(task.initialization, np.ndarray)
        self.assertIsInstance(task.tsne, TSNE)
        self.assertIsInstance(task.tsne_embedding, TSNEModel)


if __name__ == "__main__":
    unittest.main()
