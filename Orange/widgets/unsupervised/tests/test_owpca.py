# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import numpy as np
import scipy.sparse as sp

from Orange.data import Table, Domain, ContinuousVariable, TimeVariable
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owpca import OWPCA, DECOMPOSITIONS


class TestOWPCA(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWPCA)  # type: OWPCA

    def test_set_variance100(self):
        iris = Table("iris")[:5]
        self.widget.set_data(iris)
        self.widget.variance_covered = 100
        self.widget._update_selection_variance_spin()

    def test_constant_data(self):
        data = Table("iris")[::5]
        data.X[:, :] = 1.0
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Warning.trivial_components.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.transformed_data))
        self.assertIsNone(self.get_output(self.widget.Outputs.components))

    def test_empty_data(self):
        """ Check widget for dataset with no rows and for dataset with no attributes """
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data[:0])
        self.assertTrue(self.widget.Error.no_instances.is_shown())

        domain = Domain([], None, data.domain.variables)
        new_data = Table.from_table(domain, data)
        self.send_signal(self.widget.Inputs.data, new_data)
        self.assertTrue(self.widget.Error.no_features.is_shown())
        self.assertFalse(self.widget.Error.no_instances.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_features.is_shown())

    def test_limit_components(self):
        X = np.random.RandomState(0).rand(101, 101)
        data = Table(X)
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
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.maxp = 2
        self.widget._setup_plot()
        var2 = self.widget.variance_covered
        self.widget.ncomponents = 3
        self.widget._update_selection_component_spin()
        var3 = self.widget.variance_covered
        self.assertGreater(var3, var2)

    def test_sparse_data(self):
        data = Table("iris")
        data.X = sp.csr_matrix(data.X)
        self.widget.set_data(data)
        decomposition = DECOMPOSITIONS[self.widget.decomposition_idx]
        self.assertTrue(decomposition.supports_sparse)
        self.assertFalse(self.widget.normalize_box.isEnabled())

        buttons = self.widget.decomposition_box.group.box.buttons
        for i, decomposition in enumerate(DECOMPOSITIONS):
            if not decomposition.supports_sparse:
                self.assertFalse(buttons[i].isEnabled())

        data = Table("iris")
        self.widget.set_data(data)
        self.assertTrue(all([b.isEnabled() for b in buttons]))
        self.assertTrue(self.widget.normalize_box.isEnabled())

    def test_all_components_continuous(self):
        data = Table("banking-crises.tab")
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

    def test_normalization(self):
        data = Table("iris.tab")
        self.widget.ncomponents = 2
        self.assertTrue(self.widget.normalize)
        self.widget.set_data(data)
        varnorm = self.widget.variance_covered
        self.widget.controls.normalize.toggle()
        varnonnorm = self.widget.variance_covered
        # normalized data will have lower covered variance
        self.assertLess(varnorm, varnonnorm)

    def test_do_not_mask_features(self):
        # the widget used to replace cached variables when creating the
        # components output (until 20170726)
        data = Table("iris.tab")
        self.widget.set_data(data)
        ndata = Table("iris.tab")
        self.assertEqual(data.domain[0], ndata.domain[0])
