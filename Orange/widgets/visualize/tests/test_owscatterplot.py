# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from unittest.mock import MagicMock
import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import QRectF, Qt

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin, datasets
from Orange.widgets.visualize.owscatterplot import \
    OWScatterPlot, ScatterPlotVizRank
from Orange.widgets.tests.utils import simulate


class TestOWScatterPlot(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data

    def setUp(self):
        self.widget = self.create_widget(OWScatterPlot)

    def test_set_data(self):
        # Connect iris to scatter plot
        self.send_signal(self.widget.Inputs.data, self.data)

        # First two attribute should be selected as x an y
        self.assertEqual(self.widget.attr_x, self.data.domain[0])
        self.assertEqual(self.widget.attr_y, self.data.domain[1])

        # Class var should be selected as color
        self.assertIs(self.widget.graph.attr_color, self.data.domain.class_var)

        # Change which attributes are displayed
        self.widget.attr_x = self.data.domain[2]
        self.widget.attr_y = self.data.domain[3]

        # Disconnect the data
        self.send_signal(self.widget.Inputs.data, None)

        # removing data should have cleared attributes
        self.assertEqual(self.widget.attr_x, None)
        self.assertEqual(self.widget.attr_y, None)
        self.assertEqual(self.widget.graph.attr_color, None)

        # and remove the legend
        self.assertEqual(self.widget.graph.legend, None)

        # Connect iris again
        # same attributes that were used last time should be selected
        self.send_signal(self.widget.Inputs.data, self.data)

        self.assertIs(self.widget.attr_x, self.data.domain[2])
        self.assertIs(self.widget.attr_y, self.data.domain[3])

    def test_score_heuristics(self):
        domain = Domain([ContinuousVariable(c) for c in "abcd"],
                        DiscreteVariable("c", values="ab"))
        a = np.arange(10).reshape((10, 1))
        data = Table(domain, np.hstack([a, a, a, a]), a >= 5)
        self.send_signal(self.widget.Inputs.data, data)
        vizrank = ScatterPlotVizRank(self.widget)
        self.assertEqual([x.name for x in vizrank.score_heuristic()],
                         list("abcd"))

    def test_optional_combos(self):
        domain = self.data.domain
        d1 = Domain(domain.attributes[:2], domain.class_var,
                    [domain.attributes[2]])
        t1 = Table(d1, self.data)
        self.send_signal(self.widget.Inputs.data, t1)
        self.widget.graph.attr_size = domain.attributes[2]

        d2 = Domain(domain.attributes[:2], domain.class_var,
                    [domain.attributes[3]])
        t2 = Table(d2, self.data)
        self.send_signal(self.widget.Inputs.data, t2)

    def _select_data(self):
        self.widget.graph.select_by_rectangle(QRectF(4, 3, 3, 1))
        return self.widget.graph.get_selection()

    def test_error_message(self):
        """Check if error message appears and then disappears when
        data is removed from input"""
        data = self.data.copy()
        data.X[:, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Warning.missing_coords.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.missing_coords.is_shown())

    def test_report_on_empty(self):
        self.widget.report_plot = MagicMock()
        self.widget.report_caption = MagicMock()
        self.widget.report_items = MagicMock()
        self.widget.send_report()  # Essentially, don't crash
        self.widget.report_plot.assert_not_called()
        self.widget.report_caption.assert_not_called()
        self.widget.report_items.assert_not_called()

    def test_data_column_nans(self):
        """
        ValueError cannot convert float NaN to integer.
        In case when all column values are NaN then it throws that error.
        GH-2061
        """
        table = datasets.data_one_column_nans()
        self.send_signal(self.widget.Inputs.data, table)
        cb_attr_color = self.widget.controls.graph.attr_color
        simulate.combobox_activate_item(cb_attr_color, "b")
        simulate.combobox_activate_item(self.widget.cb_attr_x, "a")
        simulate.combobox_activate_item(self.widget.cb_attr_y, "a")

        self.widget.update_graph()

    def test_regression_line(self):
        """It is possible to draw the line only for pair of continuous attrs"""
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertTrue(self.widget.cb_reg_line.isEnabled())
        self.assertIsNone(self.widget.graph.reg_line_item)
        self.widget.cb_reg_line.setChecked(True)
        self.assertIsNotNone(self.widget.graph.reg_line_item)
        self.widget.cb_attr_y.activated.emit(4)
        self.widget.cb_attr_y.setCurrentIndex(4)
        self.assertFalse(self.widget.cb_reg_line.isEnabled())
        self.assertIsNone(self.widget.graph.reg_line_item)

    def test_points_combo_boxes(self):
        """Check Point box combo models and values"""
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertEqual(len(self.widget.controls.graph.attr_color.model()), 8)
        self.assertEqual(len(self.widget.controls.graph.attr_shape.model()), 3)
        self.assertEqual(len(self.widget.controls.graph.attr_size.model()), 6)
        self.assertEqual(len(self.widget.controls.graph.attr_label.model()), 8)
        other_widget = self.create_widget(OWScatterPlot)
        self.send_signal(self.widget.Inputs.data, self.data, widget=other_widget)
        self.assertEqual(self.widget.graph.controls.attr_color.currentText(),
                         self.data.domain.class_var.name)

    def test_group_selections(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        graph = self.widget.graph
        points = graph.scatterplot_item.points()
        sel_column = np.zeros((len(self.data), 1))

        x = self.data.X

        def selectedx():
            return self.get_output(self.widget.Outputs.selected_data).X

        def annotated():
            return self.get_output(self.widget.Outputs.annotated_data).metas

        def annotations():
            return self.get_output(self.widget.Outputs.annotated_data).domain.metas[0].values

        # Select 0:5
        graph.select(points[:5])
        np.testing.assert_equal(selectedx(), x[:5])
        sel_column[:5] = 1
        np.testing.assert_equal(annotated(), sel_column)
        self.assertEqual(annotations(), ["No", "Yes"])

        # Shift-select 5:10; now we have groups 0:5 and 5:10
        with self.modifiers(Qt.ShiftModifier):
            graph.select(points[5:10])
        np.testing.assert_equal(selectedx(), x[:10])
        sel_column[5:10] = 2
        np.testing.assert_equal(annotated(), sel_column)
        self.assertEqual(len(annotations()), 3)

        # Select: 15:20; we have 15:20
        graph.select(points[15:20])
        sel_column = np.zeros((len(self.data), 1))
        sel_column[15:20] = 1
        np.testing.assert_equal(selectedx(), x[15:20])
        self.assertEqual(annotations(), ["No", "Yes"])

        # Alt-select (remove) 10:17; we have 17:20
        with self.modifiers(Qt.AltModifier):
            graph.select(points[10:17])
        np.testing.assert_equal(selectedx(), x[17:20])
        sel_column[15:17] = 0
        np.testing.assert_equal(annotated(), sel_column)
        self.assertEqual(annotations(), ["No", "Yes"])

        # Ctrl-Shift-select (add-to-last) 10:17; we have 17:25
        with self.modifiers(Qt.ShiftModifier | Qt.ControlModifier):
            graph.select(points[20:25])
        np.testing.assert_equal(selectedx(), x[17:25])
        sel_column[20:25] = 1
        np.testing.assert_equal(annotated(), sel_column)
        self.assertEqual(annotations(), ["No", "Yes"])

        # Shift-select (add) 30:35; we have 17:25, 30:35
        with self.modifiers(Qt.ShiftModifier):
            graph.select(points[30:35])
        # ... then Ctrl-Shift-select (add-to-last) 10:17; we have 17:25, 30:40
        with self.modifiers(Qt.ShiftModifier | Qt.ControlModifier):
            graph.select(points[35:40])
        sel_column[30:40] = 2
        np.testing.assert_equal(annotated(), sel_column)
        self.assertEqual(len(annotations()), 3)

    def test_none_data(self):
        """
        Prevent crash due to missing data.
        GH-2122
        """
        table = Table(
            Domain(
                [ContinuousVariable("a"),
                 DiscreteVariable("b", values=["y", "n"])]
            ),
            list(zip(
                [],
                ""))
        )
        self.send_signal(self.widget.Inputs.data, table)
        self.widget.reset_graph_data()

    def test_points_selection(self):
        # Opening widget with saved selection should restore it
        self.widget.selection = list(range(50))
        self.send_signal(self.widget.Inputs.data, self.data)  # iris
        selected_data = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(len(selected_data), 50)

        # Changing the dataset should clear selection
        titanic = Table("titanic")
        self.send_signal(self.widget.Inputs.data, titanic)
        selected_data = self.get_output(self.widget.Outputs.selected_data)
        self.assertIsNone(selected_data)

    def test_invalid_points_selection(self):
        # if selection contains rows that are not present in the current
        # dataset, widget should select what can be selected.
        self.widget.selection = list(range(50))
        self.send_signal(self.widget.Inputs.data, self.data[:10])
        selected_data = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(len(selected_data), 10)

    def test_set_strings_settings(self):
        """
        Test if settings can be loaded as strings and successfully put
        in new owplotgui combos.
        GH-2240
        """
        self.send_signal(self.widget.Inputs.data, self.data)
        settings = self.widget.settingsHandler.pack_data(self.widget)
        graph_settings = settings["context_settings"][0].values["graph"]
        graph_settings["attr_label"] = ("sepal length", -2)
        graph_settings["attr_color"] = ("sepal width", -2)
        graph_settings["attr_shape"] = ("iris", -2)
        graph_settings["attr_size"] = ("petal width", -2)
        w = self.create_widget(OWScatterPlot, stored_settings=settings)
        self.send_signal(self.widget.Inputs.data, self.data, widget=w)
        self.assertEqual(w.graph.attr_label.name, "sepal length")
        self.assertEqual(w.graph.attr_color.name, "sepal width")
        self.assertEqual(w.graph.attr_shape.name, "iris")
        self.assertEqual(w.graph.attr_size.name, "petal width")

    def test_sparse(self):
        """
        Test sparse data.
        GH-2152
        GH-2157
        """
        table = Table("iris")
        table.X = sp.csr_matrix(table.X)
        self.assertTrue(sp.issparse(table.X))
        table.Y = sp.csr_matrix(table._Y)  # pylint: disable=protected-access
        self.assertTrue(sp.issparse(table.Y))
        self.send_signal(self.widget.Inputs.data, table)
        self.widget.set_subset_data(table[:30])
        data = self.get_output("Data")

        self.assertTrue(data.is_sparse())
        self.assertEqual(len(data.domain), 5)

    def test_features_and_no_data(self):
        """
        Prevent crashing when features are sent but no data.
        GH-2384
        """
        domain = Table("iris").domain
        self.send_signal(self.widget.Inputs.features, domain)


if __name__ == "__main__":
    import unittest
    unittest.main()
