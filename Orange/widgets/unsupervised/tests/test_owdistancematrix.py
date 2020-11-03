from unittest.mock import patch

from Orange.data import Table
from Orange.distance import Euclidean
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owdistancematrix import OWDistanceMatrix


class TestOWDistanceMatrix(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWDistanceMatrix)
        self.iris = Table("iris")[:5]
        self.distances = Euclidean(self.iris)

    def test_set_distances(self):
        assert isinstance(self.widget, OWDistanceMatrix)

        # Distances with row data
        self.widget.set_distances(self.distances)
        self.assertIn(self.iris.domain[0], self.widget.annot_combo.model())

        # Distances without row data
        self.distances.row_items = None
        self.widget.set_distances(self.distances)
        self.assertNotIn(self.iris.domain[0], self.widget.annot_combo.model())

    def test_context_attribute(self):
        distances = Euclidean(self.iris, axis=0)
        annotations = ["None", "Enumerate"]
        self.widget.set_distances(distances)
        self.widget.openContext(distances, annotations)

    def test_unconditional_commit_on_new_signal(self):
        with patch.object(self.widget, 'unconditional_commit') as commit:
            self.widget.auto_commit = False
            commit.reset_mock()
            self.send_signal(self.widget.Inputs.distances, self.distances)
            commit.assert_called()

    def test_labels(self):
        grades = Table.from_url("https://datasets.biolab.si/core/grades-two.tab")
        distances = Euclidean(grades)
        self.widget.set_distances(distances)
        ac = self.widget.annot_combo
        idx = ac.model().indexOf(grades.domain.metas[0])
        ac.setCurrentIndex(idx)
        ac.activated.emit(idx)
        self.assertIsNone(self.widget.tablemodel.label_colors)
