from Orange.data import Table
from Orange.distance import Euclidean
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owdistancematrix import OWDistanceMatrix

class TestOWDistanceMatrix(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWDistanceMatrix)

    def test_set_distances(self):
        assert isinstance(self.widget, OWDistanceMatrix)

        iris = Table("iris")[:5]
        distances = Euclidean(iris)

        # Distances with row data
        self.widget.set_distances(distances)
        self.assertIn(iris.domain[0], self.widget.annot_combo.model())

        # Distances without row data
        distances.row_items = None
        self.widget.set_distances(distances)
        self.assertNotIn(iris.domain[0], self.widget.annot_combo.model())

    def test_context_attribute(self):
        iris = Table("iris")
        distances = Euclidean(iris, axis=0)
        annotations = ["None", "Enumerate"]
        self.widget.set_distances(distances)
        self.widget.openContext(distances, annotations)
