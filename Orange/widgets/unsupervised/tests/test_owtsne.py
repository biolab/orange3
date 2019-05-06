import unittest
from unittest.mock import patch, Mock, call

import numpy as np

from Orange.data import DiscreteVariable, ContinuousVariable, Domain, Table
from Orange.preprocess import Normalize
from Orange.projection import manifold, TSNE
from Orange.projection.manifold import TSNEModel
from Orange.widgets.tests.base import (
    WidgetTest, WidgetOutputsTestMixin, ProjectionWidgetTestMixin
)
from Orange.widgets.unsupervised.owtsne import OWtSNE, TSNERunner, Task, prepare_tsne_obj


class DummyTSNE(manifold.TSNE):
    def fit(self, X, Y=None):
        return np.ones((len(X), 2), float)


class DummyTSNEModel(manifold.TSNEModel):
    def transform(self, X, **kwargs):
        return np.ones((len(X), 2), float)

    def optimize(self, n_iter, **kwargs):
        return self


class TestOWtSNE(WidgetTest, ProjectionWidgetTestMixin, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)
        cls.same_input_output_domain = False

        cls.signal_name = "Data"
        cls.signal_data = cls.data

    def setUp(self):
        # For almost all the tests, we won't need to verify t-SNE validity and
        # the tests will run much faster if we dummy them out
        self.tsne = patch("Orange.projection.manifold.TSNE", new=DummyTSNE)
        self.tsne_model = patch("Orange.projection.manifold.TSNEModel", new=DummyTSNEModel)
        self.tsne.start()
        self.tsne_model.start()

        self.widget = self.create_widget(OWtSNE, stored_settings={"multiscale": False})

        self.class_var = DiscreteVariable("Stage name", values=["STG1", "STG2"])
        self.attributes = [ContinuousVariable("GeneName" + str(i)) for i in range(5)]
        self.domain = Domain(self.attributes, class_vars=self.class_var)
        self.empty_domain = Domain([], class_vars=self.class_var)

    def tearDown(self):
        # Some tests may not wait for the widget to finish, and the patched
        # methods might be unpatched before the widget finishes, resulting in
        # a very confusing crash.
        self.widget.cancel()
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
        self.data = None
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_stop_blocking()
        self.assertIsNone(self.widget.data)

        # <2 rows
        self.data = Table(self.domain, [[1, 2, 3, 4, 5, 'STG1']])
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_stop_blocking()
        self.assertIsNone(self.widget.data)
        self.assertTrue(self.widget.Error.not_enough_rows.is_shown())

        # no attributes
        self.data = Table(self.empty_domain, [['STG1']] * 2)
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_stop_blocking()
        self.assertIsNone(self.widget.data)
        self.assertTrue(self.widget.Error.no_attributes.is_shown())

        # constant data
        self.data = Table(self.domain, [[1, 2, 3, 4, 5, 'STG1']] * 2)
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_stop_blocking()
        self.assertIsNone(self.widget.data)
        self.assertTrue(self.widget.Error.constant_data.is_shown())

        # correct input
        self.data = Table(self.domain, [[1, 2, 3, 4, 5, 'STG1'],
                                        [5, 4, 3, 2, 1, 'STG1']])
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_stop_blocking()
        self.assertIsNotNone(self.widget.data)
        self.assertFalse(self.widget.Error.not_enough_rows.is_shown())
        self.assertFalse(self.widget.Error.no_attributes.is_shown())
        self.assertFalse(self.widget.Error.constant_data.is_shown())

    def test_input(self):
        self.data = Table(self.domain, [[1, 1, 1, 1, 1, 'STG1'],
                                        [2, 2, 2, 2, 2, 'STG1'],
                                        [4, 4, 4, 4, 4, 'STG2'],
                                        [5, 5, 5, 5, 5, 'STG2']])

        self.send_signal(self.widget.Inputs.data, self.data)
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

    def test_normalize_data(self):
        # Normalization should be checked by default
        self.assertTrue(self.widget.controls.normalize.isChecked())
        with patch("Orange.preprocess.preprocess.Normalize", wraps=Normalize) as normalize:
            self.send_signal(self.widget.Inputs.data, self.data)
            self.assertTrue(self.widget.controls.normalize.isEnabled())
            self.wait_until_stop_blocking()
            normalize.assert_called_once()

        # Disable checkbox
        self.widget.controls.normalize.setChecked(False)
        self.assertFalse(self.widget.controls.normalize.isChecked())
        with patch("Orange.preprocess.preprocess.Normalize", wraps=Normalize) as normalize:
            self.send_signal(self.widget.Inputs.data, self.data)
            self.assertTrue(self.widget.controls.normalize.isEnabled())
            self.wait_until_stop_blocking()
            normalize.assert_not_called()

        # Normalization shouldn't work on sparse data
        self.widget.controls.normalize.setChecked(True)
        self.assertTrue(self.widget.controls.normalize.isChecked())

        sparse_data = self.data.to_sparse()
        with patch("Orange.preprocess.preprocess.Normalize", wraps=Normalize) as normalize:
            self.send_signal(self.widget.Inputs.data, sparse_data)
            self.assertFalse(self.widget.controls.normalize.isEnabled())
            self.wait_until_stop_blocking()
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
        self.wait_until_stop_blocking()
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
        self.wait_until_stop_blocking()
        _check_exaggeration(optimize, 3)

    def test_plot_once(self):
        """Test if data is plotted only once but committed on every input change"""
        self.widget.setup_plot = Mock()
        self.widget.commit = Mock()

        self.send_signal(self.widget.Inputs.data, self.data)
        # TODO: The base widget immediately calls `setup_plot` and `commit`
        # even though there's nothing to show yet. Unfortunately, fixing this
        # would require changing `OWDataProjectionWidget` in some strange way,
        # so as a temporary fix, we reset the mocks, so they reflect the calls
        # when the result was available.
        self.widget.setup_plot.reset_mock()
        self.widget.commit.reset_mock()
        self.wait_until_stop_blocking()

        self.widget.setup_plot.assert_called_once()
        self.widget.commit.assert_called_once()

        self.widget.commit.reset_mock()
        self.send_signal(self.widget.Inputs.data_subset, self.data[::10])
        self.wait_until_stop_blocking()

        self.widget.setup_plot.assert_called_once()
        self.widget.commit.assert_called_once()

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

    def test_invalidation_flow(self):
        # pylint: disable=protected-access
        w = self.widget
        # Setup widget: send data to input with global structure "off", then
        # set global structure "on" (after the embedding is computed)
        w.controls.multiscale.setChecked(False)
        self.send_signal(w.Inputs.data, self.data)
        self.wait_until_stop_blocking()
        self.assertFalse(self.widget.Information.modified.is_shown())
        # All the embedding components should computed
        self.assertIsNotNone(w.pca_projection)
        self.assertIsNotNone(w.affinities)
        self.assertIsNotNone(w.tsne_embedding)
        # All the invalidation flags should be set to false
        self.assertFalse(w._invalidated.pca_projection)
        self.assertFalse(w._invalidated.affinities)
        self.assertFalse(w._invalidated.tsne_embedding)

        # Trigger invalidation
        w.controls.multiscale.setChecked(True)
        self.assertTrue(self.widget.Information.modified.is_shown())
        # Setting `multiscale` to true should set the invalidate flags for
        # the affinities and embedding, but not the pca_projection
        self.assertFalse(w._invalidated.pca_projection)
        self.assertTrue(w._invalidated.affinities)
        self.assertTrue(w._invalidated.tsne_embedding)

        # The flags should now be set, but the embedding should still be
        # available when selecting a subset of data and such
        self.assertIsNotNone(w.pca_projection)
        self.assertIsNotNone(w.affinities)
        self.assertIsNotNone(w.tsne_embedding)

        # We should still be able to send a data subset to the input and have
        # the points be highlighted
        self.send_signal(w.Inputs.data_subset, self.data[:10])
        self.wait_until_stop_blocking()
        subset = [brush.color().name() == "#46befa" for brush in
                  w.graph.scatterplot_item.data["brush"][:10]]
        other = [brush.color().name() == "#000000" for brush in
                 w.graph.scatterplot_item.data["brush"][10:]]
        self.assertTrue(all(subset))
        self.assertTrue(all(other))

        # Clear the data subset
        self.send_signal(w.Inputs.data_subset, None)

        # Run the optimization
        self.widget.run_button.clicked.emit()
        self.wait_until_stop_blocking()
        # All of the inavalidation flags should have been cleared
        self.assertFalse(w._invalidated)


class TestTSNERunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Table("iris")

    def test_run(self):
        state = Mock()
        state.is_interruption_requested = Mock(return_value=False)

        task = TSNERunner.run(Task(data=self.data, perplexity=30), state)

        self.assertEqual(len(state.set_status.mock_calls), 4)
        state.set_status.assert_has_calls([
            call("Computing PCA..."),
            call("Preparing initialization..."),
            call("Finding nearest neighbors..."),
            call("Running optimization..."),
        ])

        self.assertIsInstance(task.pca_projection, Table)
        self.assertIsInstance(task.tsne, TSNE)
        self.assertIsInstance(task.tsne_embedding, TSNEModel)

    def test_run_do_not_modify_model_inplace(self):
        state = Mock()
        state.is_interruption_requested.return_value = True

        task = Task(data=self.data, perplexity=30, multiscale=False, exaggeration=1)
        # Run through all the steps to prepare the t-SNE object
        task.tsne = prepare_tsne_obj(
            task.data, task.perplexity, task.multiscale, task.exaggeration
        )
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


if __name__ == "__main__":
    unittest.main()
