# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access
from unittest import skip
from unittest.mock import patch, Mock

import numpy as np
from scipy import sparse

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate, possible_duplicate_table
from Orange.widgets.unsupervised.owmanifoldlearning import OWManifoldLearning


class TestOWManifoldLearning(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")

    def setUp(self):
        self.widget = self.create_widget(
            OWManifoldLearning, stored_settings={"auto_apply": False})  # type: OWManifoldLearning

    def test_input_data(self):
        """Check widget's data"""
        self.assertEqual(self.widget.data, None)
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget.data, self.iris)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.data, None)

    def test_output_data(self):
        """Check if data is on output after apply"""
        self.assertIsNone(self.get_output(self.widget.Outputs.transformed_data))
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.apply_button.button.click()
        self.assertIsInstance(self.get_output(self.widget.Outputs.transformed_data), Table)
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.apply_button.button.click()
        self.assertIsNone(self.get_output(self.widget.Outputs.transformed_data))

    def test_n_components(self):
        """Check the output for various numbers of components"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        for i in range(self.widget.n_components_spin.minimum(),
                       self.widget.n_components_spin.maximum()):
            self.assertEqual(self.widget.data, self.iris)
            self.widget.n_components_spin.setValue(i)
            self.widget.n_components_spin.onEnter()
            self.widget.apply_button.button.click()
            self._compare_tables(self.get_output(self.widget.Outputs.transformed_data), i)

    def test_manifold_methods(self):
        """Check output for various manifold methods"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        n_comp = self.widget.n_components
        for i in range(len(self.widget.MANIFOLD_METHODS)):
            self.assertEqual(self.widget.data, self.iris)
            self.widget.manifold_methods_combo.activated.emit(i)
            self.widget.apply_button.button.click()
            self._compare_tables(self.get_output(self.widget.Outputs.transformed_data), n_comp)

    def _compare_tables(self, _output, n_components):
        """Helper function for table comparison"""
        self.assertEqual((len(self.iris), n_components), _output.X.shape)
        np.testing.assert_array_equal(self.iris.Y, _output.Y)
        np.testing.assert_array_equal(self.iris.metas, _output.metas)

    def test_sparse_data(self):
        data = Table("iris").to_sparse()
        self.assertTrue(sparse.issparse(data.X))

        def __callback():
            # Send sparse data to input
            self.send_signal(self.widget.Inputs.data, data)
            self.widget.apply_button.button.click()
            self.assertTrue(self.widget.Error.sparse_not_supported.is_shown())
            # Clear input
            self.send_signal(self.widget.Inputs.data, None)
            self.widget.apply_button.button.click()
            self.assertFalse(self.widget.Error.sparse_not_supported.is_shown())

        simulate.combobox_run_through_all(
            self.widget.manifold_methods_combo, callback=__callback,
        )

    def test_metrics(self):
        # Select t-SNE method, which is the only method that supports metrics
        simulate.combobox_activate_item(self.widget.manifold_methods_combo, "t-SNE")

        def __callback():
            # Send data to input
            self.send_signal(self.widget.Inputs.data, self.iris)
            self.widget.apply_button.button.click()
            self.assertFalse(self.widget.Error.manifold_error.is_shown())

            # Clear input
            self.send_signal(self.widget.Inputs.data, None)
            self.widget.apply_button.button.click()
            self.assertFalse(self.widget.Error.manifold_error.is_shown())

        simulate.combobox_run_through_all(
            self.widget.tsne_editor.controls.metric_index, callback=__callback,
        )

    def test_unique_domain(self):
        simulate.combobox_activate_item(self.widget.manifold_methods_combo, "MDS")
        data = possible_duplicate_table('C0', class_var=True)
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.apply_button.button.click()
        out = self.get_output(self.widget.Outputs.transformed_data)
        self.assertTrue(out.domain.attributes[0], 'C0 (1)')

    @skip
    def test_singular_matrices(self):
        """
        Handle singular matrices.
        GH-2228

        TODO: This test makes sense with the ``Mahalanobis`` distance metric
        which is currently not supported by tSNE. In case it is ever
        re-introduced, this test is very much required.

        """
        table = Table(
            Domain(
                [ContinuousVariable("a"), ContinuousVariable("b")],
                class_vars=DiscreteVariable("c", values=("0", "1"))),
            list(zip(
                [1, 1, 1],
                [0, 1, 2],
                [0, 1, 1]))
        )
        self.send_signal(self.widget.Inputs.data, table)
        self.widget.manifold_methods_combo.activated.emit(0)  # t-SNE
        self.widget.tsne_editor.metric_combo.activated.emit(4)  # Mahalanobis
        self.assertFalse(self.widget.Error.manifold_error.is_shown())
        self.widget.apply_button.button.click()
        self.assertTrue(self.widget.Error.manifold_error.is_shown())

    def test_out_of_memory(self):
        """
        Show error message when out of memory.
        GH-2441
        """
        table = Table("iris")
        with patch("Orange.projection.manifold.MDS.__call__", Mock()) as mock:
            mock.side_effect = MemoryError
            self.send_signal("Data", table)
            self.widget.manifold_methods_combo.activated.emit(1)
            self.widget.apply_button.button.click()
            self.assertTrue(self.widget.Error.out_of_memory.is_shown())

    def test_unconditional_commit_on_new_signal(self):
        with patch.object(self.widget.commit, 'now') as apply:
            self.widget.auto_apply = False
            apply.reset_mock()
            self.send_signal(self.widget.Inputs.data, self.iris)
            apply.assert_called()
