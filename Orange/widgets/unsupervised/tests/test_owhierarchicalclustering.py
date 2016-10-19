# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.distance import Euclidean
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.unsupervised.owhierarchicalclustering import \
    OWHierarchicalClustering
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME


class TestOWHierarchicalClustering(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.distances = Euclidean(cls.data)
        cls.signal_name = "Distances"
        cls.signal_data = cls.distances
        cls.same_input_output_domain = False

    def setUp(self):
        self.widget = self.create_widget(OWHierarchicalClustering)

    def _select_data(self):
        items = self.widget.dendrogram._items
        cluster = items[sorted(list(items.keys()))[4]]
        self.widget.dendrogram.set_selected_items([cluster])
        return [14, 15, 32, 33]

    def _compare_selected_annotated_domains(self, selected, annotated):
        self.assertTrue(all((var in annotated.domain.variables
                             for var in selected.domain.variables)))
        self.assertNotIn("Other", selected.domain.metas[0].values)
        self.assertIn("Other", annotated.domain.metas[0].values)
        self.assertTrue(
            all((var in [var.name for var in annotated.domain.metas]
                 for var in [var.name for var in selected.domain.metas])))

    def test_selection_box_output(self):
        """Check output if Selection method changes"""
        self.send_signal("Distances", self.distances)
        self.assertIsNone(self.get_output("Selected Data"))
        self.assertIsNotNone(self.get_output(ANNOTATED_DATA_SIGNAL_NAME))

        # change selection to 'Height ratio'
        self.widget.selection_box.buttons[1].click()
        self.assertIsNotNone(self.get_output("Selected Data"))
        self.assertIsNone(self.get_output(ANNOTATED_DATA_SIGNAL_NAME))

        # change selection to 'Top N'
        self.widget.selection_box.buttons[2].click()
        self.assertIsNotNone(self.get_output("Selected Data"))
        self.assertIsNone(self.get_output(ANNOTATED_DATA_SIGNAL_NAME))
