# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access
import unittest
from unittest.mock import patch, Mock

import numpy as np

from Orange.data import Table, Domain, ContinuousVariable, TimeVariable
from Orange.preprocess import preprocess
from Orange.preprocess.preprocess import Normalize
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import table_dense_sparse, possible_duplicate_table
from Orange.widgets.unsupervised.owpca import OWPCA
from Orange.tests import test_filename
from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip


class TestOWPCA(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWPCA)  # type: OWPCA
        self.iris = Table("iris")  # type: Table

    def test_set_variance100(self):
        self.widget.set_data(self.iris)
        self.widget.variance_covered = 100
        self.widget._update_selection_variance_spin()

    def test_constant_data(self):
        data = self.iris[::5].copy()
        with data.unlocked():
            data.X[:, :] = 1.0
        # Ignore the warning: the test checks whether the widget shows
        # Warning.trivial_components when this happens
        with np.errstate(invalid="ignore"):
            self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Warning.trivial_components.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.transformed_data))
        self.assertIsNone(self.get_output(self.widget.Outputs.components))

    def test_empty_data(self):
        """ Check widget for dataset with no rows and for dataset with no attributes """
        self.send_signal(self.widget.Inputs.data, self.iris[:0])
        self.assertTrue(self.widget.Error.no_instances.is_shown())

        domain = Domain([], None, self.iris.domain.variables)
        new_data = Table.from_table(domain, self.iris)
        self.send_signal(self.widget.Inputs.data, new_data)
        self.assertTrue(self.widget.Error.no_features.is_shown())
        self.assertFalse(self.widget.Error.no_instances.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_features.is_shown())

    def test_limit_components(self):
        X = np.random.RandomState(0).rand(101, 101)
        data = Table.from_numpy(None, X)
        self.widget.ncomponents = 100
        self.send_signal(self.widget.Inputs.data, data)
        tran = self.get_output(self.widget.Outputs.transformed_data)
        self.assertEqual(len(tran.domain.attributes), 100)
        self.widget.ncomponents = 101  # should not be accesible
        with self.assertRaises(IndexError):
            self.send_signal(self.widget.Inputs.data, data)

    def test_migrate_settings_limits_components(self):
        settings = dict(ncomponents=10)
        OWPCA.migrate_settings(settings, 0)
        self.assertEqual(settings['ncomponents'], 10)
        settings = dict(ncomponents=101)
        OWPCA.migrate_settings(settings, 0)
        self.assertEqual(settings['ncomponents'], 100)

    def test_migrate_settings_changes_variance_covered_to_int(self):
        settings = dict(variance_covered=17.5)
        OWPCA.migrate_settings(settings, 0)
        self.assertEqual(settings["variance_covered"], 17)

        settings = dict(variance_covered=float('nan'))
        OWPCA.migrate_settings(settings, 0)
        self.assertEqual(settings["variance_covered"], 100)

    def test_variance_shown(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.maxp = 2
        self.widget._setup_plot()
        var2 = self.widget.variance_covered
        self.widget.ncomponents = 3
        self.widget._update_selection_component_spin()
        var3 = self.widget.variance_covered
        self.assertGreater(var3, var2)

    def test_unique_domain_components(self):
        table = possible_duplicate_table('components')
        self.send_signal(self.widget.Inputs.data, table)
        out = self.get_output(self.widget.Outputs.components)
        self.assertEqual(out.domain.metas[0].name, 'components (1)')

    def test_variance_attr(self):
        self.widget.ncomponents = 2
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_stop_blocking()
        self.widget._variance_ratio = np.array([0.5, 0.25, 0.2, 0.05])
        self.widget.commit.now()

        result = self.get_output(self.widget.Outputs.transformed_data)
        pc1, pc2 = result.domain.attributes
        self.assertEqual(pc1.attributes["variance"], 0.5)
        self.assertEqual(pc2.attributes["variance"], 0.25)

        result = self.get_output(self.widget.Outputs.data)
        pc1, pc2 = result.domain.metas
        self.assertEqual(pc1.attributes["variance"], 0.5)
        self.assertEqual(pc2.attributes["variance"], 0.25)

        result = self.get_output(self.widget.Outputs.components)
        np.testing.assert_almost_equal(result.get_column_view("variance")[0].T,
                                       [0.5, 0.25])

    def test_sparse_data(self):
        """Check that PCA returns the same results for both dense and sparse data."""
        dense_data, sparse_data = self.iris, self.iris.to_sparse()

        def _compute_projection(data):
            self.send_signal(self.widget.Inputs.data, data)
            self.wait_until_stop_blocking()
            result = self.get_output(self.widget.Outputs.transformed_data)
            self.send_signal(self.widget.Inputs.data, None)
            return result

        # Disable normalization
        self.widget.controls.normalize.setChecked(False)
        dense_pca = _compute_projection(dense_data)
        sparse_pca = _compute_projection(sparse_data)
        np.testing.assert_almost_equal(dense_pca.X, sparse_pca.X)

        # Enable normalization
        self.widget.controls.normalize.setChecked(True)
        dense_pca = _compute_projection(dense_data)
        sparse_pca = _compute_projection(sparse_data)
        np.testing.assert_almost_equal(dense_pca.X, sparse_pca.X)

    def test_all_components_continuous(self):
        data = Table(test_filename("datasets/cyber-security-breaches.tab"))
        # GH-2329 only occurred on TimeVariables when normalize=False
        self.assertTrue(any(isinstance(a, TimeVariable)
                            for a in data.domain.attributes))

        self.widget.normalize = False
        self.widget._update_normalize()     # pylint: disable=protected-access
        self.widget.set_data(data)

        components = self.get_output(self.widget.Outputs.components)
        self.assertTrue(all(type(a) is ContinuousVariable   # pylint: disable=unidiomatic-typecheck
                            for a in components.domain.attributes),
                        "Some variables aren't of type ContinuousVariable")

    @table_dense_sparse
    def test_normalize_data(self, prepare_table):
        """Check that normalization is called at the proper times."""
        data = prepare_table(self.iris)

        # Enable checkbox
        self.widget.controls.normalize.setChecked(True)
        self.assertTrue(self.widget.controls.normalize.isChecked())
        with patch.object(preprocess, "Normalize", wraps=Normalize) as normalize:
            self.send_signal(self.widget.Inputs.data, data)
            self.wait_until_stop_blocking()
            self.assertTrue(self.widget.controls.normalize.isEnabled())
            normalize.assert_called_once()

        # Disable checkbox
        self.widget.controls.normalize.setChecked(False)
        self.assertFalse(self.widget.controls.normalize.isChecked())
        with patch.object(preprocess, "Normalize", wraps=Normalize) as normalize:
            self.send_signal(self.widget.Inputs.data, data)
            self.wait_until_stop_blocking()
            self.assertTrue(self.widget.controls.normalize.isEnabled())
            normalize.assert_not_called()

    @table_dense_sparse
    def test_normalization_variance(self, prepare_table):
        data = prepare_table(self.iris)
        self.widget.ncomponents = 2

        # Enable normalization
        self.widget.controls.normalize.setChecked(True)
        self.assertTrue(self.widget.normalize)
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_stop_blocking()
        variance_normalized = self.widget.variance_covered

        # Disable normalization
        self.widget.controls.normalize.setChecked(False)
        self.assertFalse(self.widget.normalize)
        self.wait_until_stop_blocking()
        variance_unnormalized = self.widget.variance_covered

        # normalized data will have lower covered variance
        self.assertLess(variance_normalized, variance_unnormalized)

    @table_dense_sparse
    def test_normalized_gives_correct_result(self, prepare_table):
        """Make sure that normalization through widget gives correct result."""
        # Randomly set some values to zero
        random_state = check_random_state(42)
        mask = random_state.beta(1, 2, size=self.iris.X.shape) > 0.5
        with self.iris.unlocked():
            self.iris.X[mask] = 0

        data = prepare_table(self.iris)

        # Enable normalization and run data through widget
        self.widget.controls.normalize.setChecked(True)
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_stop_blocking()
        widget_result = self.get_output(self.widget.Outputs.transformed_data)

        # Compute the correct embedding
        x = self.iris.X
        x = (x - x.mean(0)) / x.std(0)
        U, S, Va = np.linalg.svd(x)
        U, S, Va = U[:, :2], S[:2], Va[:2]
        U, Va = svd_flip(U, Va)
        pca_embedding = U * S

        np.testing.assert_almost_equal(widget_result.X, pca_embedding)

    def test_do_not_mask_features(self):
        # the widget used to replace cached variables when creating the
        # components output (until 20170726)
        data = Table("iris.tab")
        self.widget.set_data(data)
        ndata = Table("iris.tab")
        self.assertEqual(data.domain[0], ndata.domain[0])

    def test_on_cut_changed(self):
        widget = self.widget
        widget.ncomponents = 2
        invalidate = widget._invalidate_selection = Mock()
        widget._on_cut_changed(2)
        invalidate.assert_not_called()
        widget._on_cut_changed(3)
        invalidate.assert_called()

        widget.ncomponents = 0  # Take all components
        invalidate.reset_mock()
        widget._on_cut_changed(1)
        invalidate.assert_not_called()
        self.assertEqual(widget.ncomponents, 0)

    def test_output_data(self):
        widget = self.widget
        widget.ncomponents = 2
        domain = Domain(self.iris.domain.attributes[:3],
                        self.iris.domain.class_var,
                        self.iris.domain.attributes[3:])
        iris = self.iris.transform(domain)
        self.send_signal(widget.Inputs.data, iris)
        output = self.get_output(widget.Outputs.data)
        outdom = output.domain
        self.assertEqual(domain.attributes, outdom.attributes)
        self.assertEqual(domain.class_var, outdom.class_var)
        self.assertEqual(domain.metas, outdom.metas[:1])
        self.assertEqual(len(outdom.metas), 3)
        np.testing.assert_equal(iris.X, output.X)
        np.testing.assert_equal(iris.Y, output.Y)
        np.testing.assert_equal(iris.metas[:, 0], output.metas[:, 0])

        trans = self.get_output(widget.Outputs.transformed_data)
        self.assertEqual(trans.domain.attributes, outdom.metas[1:])
        np.testing.assert_equal(trans.X, output.metas[:, 1:])

        self.send_signal(widget.Inputs.data, None)
        output = self.get_output(widget.Outputs.data)
        self.assertIsNone(output)


if __name__ == "__main__":
    unittest.main()
