from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owpca import OWPCA


class TestOWDistanceMatrix(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWPCA)

    def test_set_variance100(self):
        iris = Table("iris")[:5]
        self.widget.set_data(iris)
        self.widget.variance_covered = 100
        self.widget._update_selection_variance_spin()
