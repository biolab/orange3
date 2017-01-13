# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table, Domain
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owpca import OWPCA


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
        self.send_signal("Data", data)
        self.assertTrue(self.widget.Warning.trivial_components.is_shown())
        self.assertIsNone(self.get_output("Transformed Data"))
        self.assertIsNone(self.get_output("Components"))

    def test_empty_data(self):
        """ Check widget for dataset with no rows and for dataset with no attributes """
        data = Table("iris")
        self.send_signal("Data", data[:0])
        self.assertTrue(self.widget.Error.no_instances.is_shown())

        domain = Domain([], None, data.domain.variables)
        new_data = Table.from_table(domain, data)
        self.send_signal("Data", new_data)
        self.assertTrue(self.widget.Error.no_features.is_shown())
        self.assertFalse(self.widget.Error.no_instances.is_shown())

        self.send_signal("Data", None)
        self.assertFalse(self.widget.Error.no_features.is_shown())
