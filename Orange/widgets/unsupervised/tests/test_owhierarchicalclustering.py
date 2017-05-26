# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import numpy
import Orange.misc
from Orange.distance import Euclidean
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.unsupervised.owhierarchicalclustering import \
    OWHierarchicalClustering


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
        self.assertEqual(annotated.domain.variables,
                         selected.domain.variables)
        self.assertNotIn("Other", selected.domain.metas[0].values)
        self.assertIn("Other", annotated.domain.metas[0].values)
        self.assertLess(set(var.name for var in selected.domain.metas),
                        set(var.name for var in annotated.domain.metas))

    def test_selection_box_output(self):
        """Check output if Selection method changes"""
        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.annotated_data))

        # change selection to 'Height ratio'
        self.widget.selection_box.buttons[1].click()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNone(self.get_output(self.widget.Outputs.annotated_data))

        # change selection to 'Top N'
        self.widget.selection_box.buttons[2].click()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNone(self.get_output(self.widget.Outputs.annotated_data))

    def test_all_zero_inputs(self):
        d = Orange.misc.DistMatrix(numpy.zeros((10, 10)))
        self.widget.set_distances(d)

    def test_annotation_settings_retrieval(self):
        """Check whether widget retrieves correct settings for annotation"""
        widget = self.widget

        dist_names = Orange.misc.DistMatrix(
            numpy.zeros((4, 4)), self.data, axis=0)
        dist_no_names = Orange.misc.DistMatrix(numpy.zeros((10, 10)), axis=1)

        self.send_signal(self.widget.Inputs.distances, self.distances)
        # Check that default is set (class variable)
        self.assertEqual(widget.annotation, self.data.domain.class_var)

        var2 = self.data.domain[2]
        widget.annotation = var2
        # Iris now has var2 as annotation

        self.send_signal(self.widget.Inputs.distances, dist_no_names)
        self.assertEqual(widget.annotation, "Enumeration")  # Check default
        widget.annotation = "None"
        # Pure matrix with axis=1 now has None as annotation

        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertIs(widget.annotation, var2)
        self.send_signal(self.widget.Inputs.distances, dist_no_names)
        self.assertEqual(widget.annotation, "None")

        self.send_signal(self.widget.Inputs.distances, dist_names)
        self.assertEqual(widget.annotation, "Name")  # Check default
        widget.annotation = "Enumeration"
        # Pure matrix with axis=1 has Enumerate as annotation

        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertIs(widget.annotation, var2)
        self.send_signal(self.widget.Inputs.distances, dist_no_names)
        self.assertEqual(widget.annotation, "None")
        self.send_signal(self.widget.Inputs.distances, dist_names)
        self.assertEqual(widget.annotation, "Enumeration")
        self.send_signal(self.widget.Inputs.distances, dist_no_names)
        self.assertEqual(widget.annotation, "None")

    def test_domain_loses_class(self):
        widget = self.widget
        self.send_signal(self.widget.Inputs.distances, self.distances)
        data = self.data[:, :4]
        distances = Euclidean(data)
        self.send_signal(self.widget.Inputs.distances, distances)
