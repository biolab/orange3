from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owpca import OWPCA


class TestOWDistanceMatrix(WidgetTest):

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
