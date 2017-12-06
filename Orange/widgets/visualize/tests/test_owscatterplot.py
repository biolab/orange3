# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from unittest.mock import MagicMock, patch
import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import QRectF, Qt
from AnyQt.QtWidgets import QToolTip

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.visualize.owscatterplotgraph import MAX
from Orange.widgets.widget import AttributeList
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin, datasets
from Orange.widgets.visualize.owscatterplot import \
    OWScatterPlot, ScatterPlotVizRank
from Orange.widgets.tests.utils import simulate


class TestOWScatterPlot(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)
        cls.same_input_output_domain = False

        cls.signal_name = "Data"
        cls.signal_data = cls.data

    def setUp(self):
        self.widget = self.create_widget(OWScatterPlot)

    def _compare_selected_annotated_domains(self, selected, annotated):
        # Base class tests that selected.domain is a subset of annotated.domain
        # In scatter plot, the two domains are unrelated, so we disable the test
        pass

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

    def test_data_column_infs(self):
        """
        Scatter Plot should not crash on data with infinity values
        GH-2707
        GH-2684
        """
        table = datasets.data_one_column_infs()
        self.send_signal(self.widget.Inputs.data, table)
        attr_x = self.widget.controls.attr_x
        simulate.combobox_activate_item(attr_x, "b")

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

        def selected_groups():
            return self.get_output(self.widget.Outputs.selected_data).metas[:, 0]

        def annotated():
            return self.get_output(self.widget.Outputs.annotated_data).metas

        def annotations():
            return self.get_output(self.widget.Outputs.annotated_data).domain.metas[0].values

        # Select 0:5
        graph.select(points[:5])
        np.testing.assert_equal(selectedx(), x[:5])
        np.testing.assert_equal(selected_groups(), np.zeros(5))
        sel_column[:5] = 1
        np.testing.assert_equal(annotated(), sel_column)
        self.assertEqual(annotations(), ["No", "Yes"])

        # Shift-select 5:10; now we have groups 0:5 and 5:10
        with self.modifiers(Qt.ShiftModifier):
            graph.select(points[5:10])
        np.testing.assert_equal(selectedx(), x[:10])
        np.testing.assert_equal(selected_groups(), np.array([0] * 5 + [1] * 5))
        sel_column[5:10] = 2
        np.testing.assert_equal(annotated(), sel_column)
        self.assertEqual(len(annotations()), 3)

        # Select: 15:20; we have 15:20
        graph.select(points[15:20])
        sel_column = np.zeros((len(self.data), 1))
        sel_column[15:20] = 1
        np.testing.assert_equal(selectedx(), x[15:20])
        np.testing.assert_equal(selected_groups(), np.zeros(5))
        self.assertEqual(annotations(), ["No", "Yes"])

        # Alt-select (remove) 10:17; we have 17:20
        with self.modifiers(Qt.AltModifier):
            graph.select(points[10:17])
        np.testing.assert_equal(selectedx(), x[17:20])
        np.testing.assert_equal(selected_groups(), np.zeros(3))
        sel_column[15:17] = 0
        np.testing.assert_equal(annotated(), sel_column)
        self.assertEqual(annotations(), ["No", "Yes"])

        # Ctrl-Shift-select (add-to-last) 10:17; we have 17:25
        with self.modifiers(Qt.ShiftModifier | Qt.ControlModifier):
            graph.select(points[20:25])
        np.testing.assert_equal(selectedx(), x[17:25])
        np.testing.assert_equal(selected_groups(), np.zeros(8))
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
        np.testing.assert_equal(selected_groups(), np.array([0] * 8 + [1] * 10))
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

    def test_saving_selection(self):
        self.send_signal(self.widget.Inputs.data, self.data)  # iris
        self.widget.graph.select_by_rectangle(QRectF(4, 3, 3, 1))
        selected_inds = np.flatnonzero(self.widget.graph.selection)
        settings = self.widget.settingsHandler.pack_data(self.widget)
        np.testing.assert_equal(selected_inds, [i for i, g in settings["selection_group"]])

    def test_points_selection(self):
        # Opening widget with saved selection should restore it
        self.widget = self.create_widget(
            OWScatterPlot, stored_settings={
                "selection_group": [(i, 1) for i in range(50)]}
        )
        self.send_signal(self.widget.Inputs.data, self.data)  # iris
        selected_data = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(len(selected_data), 50)

        # Changing the dataset should clear selection
        titanic = Table("titanic")
        self.send_signal(self.widget.Inputs.data, titanic)
        selected_data = self.get_output(self.widget.Outputs.selected_data)
        self.assertIsNone(selected_data)

    def test_migrate_selection(self):
        settings = dict(selection=list(range(2)))
        OWScatterPlot.migrate_settings(settings, 0)
        self.assertEqual(settings["selection_group"], [(0, 1), (1, 1)])

    def test_invalid_points_selection(self):
        # if selection contains rows that are not present in the current
        # dataset, widget should select what can be selected.
        self.widget = self.create_widget(
            OWScatterPlot, stored_settings={
                "selection_group": [(i, 1) for i in range(50)]}
        )
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
        table = Table("iris").to_sparse(sparse_attributes=True,
                                        sparse_class=True)
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
        self.send_signal(self.widget.Inputs.features,
                         AttributeList(domain.variables))
        self.send_signal(self.widget.Inputs.features, None)

    def test_features_and_data(self):
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.features,
                         AttributeList(data.domain[2:]))
        self.assertIs(self.widget.attr_x, data.domain[2])
        self.assertIs(self.widget.attr_y, data.domain[3])

    def test_output_features(self):
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)

        # This doesn't work because combo's callbacks are connected to signal
        # `activated`, which is only triggered by user interaction, and not to
        # `currentIndexChanged`
        # combo_y = self.widget.controls.attr_y
        # combo_y.setCurrentIndex(combo_y.model().indexOf(data.domain[3]))
        # This is a workaround
        self.widget.attr_y = data.domain[3]
        self.widget.update_attr()

        features = self.get_output(self.widget.Outputs.features)
        self.assertEqual(features, [data.domain[0], data.domain[3]])

    def test_send_report(self):
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.report_button.click()

    def test_update_density(self):
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.cb_class_density.click()

    def test_vizrank(self):
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)
        vizrank = ScatterPlotVizRank(self.widget)
        n_states = len(data.domain.attributes)
        n_states = n_states * (n_states - 1) / 2
        states = [state for state in vizrank.iterate_states(None)]
        self.assertEqual(len(states), n_states)
        self.assertEqual(len(set(states)), n_states)
        self.assertIsNotNone(vizrank.compute_score(states[0]))
        self.send_signal(self.widget.Inputs.data, data[:9])
        self.assertIsNone(vizrank.compute_score(states[0]))

        data = Table("housing")[::10]
        self.send_signal(self.widget.Inputs.data, data)
        vizrank = ScatterPlotVizRank(self.widget)
        states = [state for state in vizrank.iterate_states(None)]
        self.assertIsNotNone(vizrank.compute_score(states[0]))

    def test_vizrank_class_nan(self):
        """
        When class values are nan, vizrank should be disabled. It should behave like
        the class column is missing.
        GH-2757
        """
        def assert_vizrank_enabled(data, is_enabled):
            self.send_signal(self.widget.Inputs.data, data)
            self.assertEqual(is_enabled, self.widget.vizrank_button.isEnabled())

        data1 = Table("iris")[::30]
        data2 = Table("iris")[::30]
        data2.Y[:] = np.nan
        domain = Domain(
            attributes=data2.domain.attributes[:4], class_vars=DiscreteVariable("iris", values=[]))
        data2 = Table(domain, data2.X, Y=data2.Y)
        data3 = Table("iris")[::30]
        data3.Y[:] = np.nan

        for data, is_enabled in zip([data1, data2, data1, data3, data1],
                                    [True, False, True, False, True]):
            assert_vizrank_enabled(data, is_enabled)

    def test_auto_send_selection(self):
        """
        Scatter Plot automatically sends selection only when the checkbox Send automatically
        is checked.
        GH-2649
        GH-2646
        """
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.controls.auto_send_selection.setChecked(False)
        self.assertEqual(False, self.widget.controls.auto_send_selection.isChecked())
        self._select_data()
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        self.widget.controls.auto_send_selection.setChecked(True)
        self.assertIsInstance(self.get_output(self.widget.Outputs.selected_data), Table)

    def test_color_is_optional(self):
        zoo = Table("zoo")
        backbone, breathes, airborne, type = \
            [zoo.domain[x] for x in ["backbone", "breathes", "airborne", "type"]]
        default_x, default_y, default_color = \
            zoo.domain[0], zoo.domain[1], zoo.domain.class_var
        attr_x = self.widget.controls.attr_x
        attr_y = self.widget.controls.attr_y
        attr_color = self.widget.controls.graph.attr_color

        # Send dataset, ensure defaults are what we expect them to be
        self.send_signal(self.widget.Inputs.data, zoo)
        self.assertEqual(attr_x.currentText(), default_x.name)
        self.assertEqual(attr_y.currentText(), default_y.name)
        self.assertEqual(attr_color.currentText(), default_color.name)
        # Select different values
        simulate.combobox_activate_item(attr_x, backbone.name)
        simulate.combobox_activate_item(attr_y, breathes.name)
        simulate.combobox_activate_item(attr_color, airborne.name)

        # Send compatible dataset, values should not change
        zoo2 = zoo[:, (backbone, breathes, airborne, type)]
        self.send_signal(self.widget.Inputs.data, zoo2)
        self.assertEqual(attr_x.currentText(), backbone.name)
        self.assertEqual(attr_y.currentText(), breathes.name)
        self.assertEqual(attr_color.currentText(), airborne.name)

        # Send dataset without color variable
        # x and y should remain, color reset to default
        zoo3 = zoo[:, (backbone, breathes, type)]
        self.send_signal(self.widget.Inputs.data, zoo3)
        self.assertEqual(attr_x.currentText(), backbone.name)
        self.assertEqual(attr_y.currentText(), breathes.name)
        self.assertEqual(attr_color.currentText(), default_color.name)

        # Send dataset without x
        # y and color should be the same as with zoo
        zoo4 = zoo[:, (default_x, default_y, breathes, airborne, type)]
        self.send_signal(self.widget.Inputs.data, zoo4)
        self.assertEqual(attr_x.currentText(), default_x.name)
        self.assertEqual(attr_y.currentText(), default_y.name)
        self.assertEqual(attr_color.currentText(), default_color.name)

        # Send dataset compatible with zoo2 and zoo3
        # Color should reset to one in zoo3, as it was used more
        # recently
        zoo5 = zoo[:, (default_x, backbone, breathes, airborne, type)]
        self.send_signal(self.widget.Inputs.data, zoo5)
        self.assertEqual(attr_x.currentText(), backbone.name)
        self.assertEqual(attr_y.currentText(), breathes.name)
        self.assertEqual(attr_color.currentText(), type.name)

    def test_handle_metas(self):
        """
        Scatter Plot Graph can handle metas
        GH-2699
        """
        w = self.widget
        data = Table("iris")
        domain = Domain(
            attributes=data.domain.attributes[:2],
            class_vars=data.domain.class_vars,
            metas=data.domain.attributes[2:]
        )
        data = data.transform(domain)
        # Sometimes floats in metas are saved as objects
        data.metas = data.metas.astype(object)
        self.send_signal(w.Inputs.data, data)
        simulate.combobox_activate_item(w.cb_attr_x, data.domain.metas[1].name)
        simulate.combobox_activate_item(w.controls.graph.attr_color, data.domain.metas[0].name)
        w.update_graph()

    def test_subset_data(self):
        """
        Scatter Plot subset data is sent to Scatter Plot Graph
        GH-2773
        """
        data = Table("iris")
        w = self.widget
        self.send_signal(w.Inputs.data, data)
        self.send_signal(w.Inputs.data_subset, data[::30])
        self.assertEqual(len(w.graph.subset_indices), 5)

    def test_sparse_subset_data(self):
        """
        Scatter Plot can handle sparse subset data.
        GH-2773
        """
        data = Table("iris")
        w = self.widget
        data.X = sp.csr_matrix(data.X)
        self.send_signal(w.Inputs.data, data)
        self.send_signal(w.Inputs.data_subset, data[::30])
        self.assertEqual(len(w.graph.subset_indices), 5)

    def test_metas_zero_column(self):
        """
        Prevent crash when metas column is zero.
        GH-2775
        """
        data = Table("iris")
        domain = data.domain
        domain = Domain(domain.attributes[:3], domain.class_vars, domain.attributes[3:])
        data = data.transform(domain)
        data.metas[:, 0] = 0
        w = self.widget
        self.send_signal(w.Inputs.data, data)
        simulate.combobox_activate_item(w.controls.attr_x, domain.metas[0].name)

    def test_tooltip(self):
        # The test tests presence of some data,
        # but avoids checking the exact format
        data = Table("heart_disease")
        self.send_signal(self.widget.Inputs.data, data)
        widget = self.widget
        graph = widget.graph
        scatterplot_item = graph.scatterplot_item

        widget.controls.attr_x = data.domain["chest pain"]
        widget.controls.attr_y = data.domain["cholesterol"]
        all_points = scatterplot_item.points()

        event = MagicMock()
        with patch.object(scatterplot_item, "mapFromScene"), \
                patch.object(QToolTip, "showText") as show_text:

            # Single point hovered
            with patch.object(scatterplot_item, "pointsAt",
                              return_value=[all_points[42]]):
                # Show just x and y attribute
                graph.tooltip_shows_all = False
                self.assertTrue(graph.help_event(event))
                (_, text), _ = show_text.call_args
                self.assertIn("age = {}".format(data[42, "age"]), text)
                self.assertIn("gender = {}".format(data[42, "gender"]), text)
                self.assertNotIn("max HR = {}".format(data[42, "max HR"]), text)
                self.assertNotIn("others", text)

                # Show all attributes
                graph.tooltip_shows_all = True
                self.assertTrue(graph.help_event(event))
                (_, text), _ = show_text.call_args
                self.assertIn("age = {}".format(data[42, "age"]), text)
                self.assertIn("gender = {}".format(data[42, "gender"]), text)
                self.assertIn("max HR = {}".format(data[42, "max HR"]), text)
                self.assertIn("... and 4 others", text)

            # Two points hovered
            with patch.object(scatterplot_item, "pointsAt",
                              return_value=[all_points[42], all_points[100]]):
                self.assertTrue(graph.help_event(event))
                (_, text), _ = show_text.call_args
                self.assertIn("age = {}".format(data[42, "age"]), text)
                self.assertIn("gender = {}".format(data[42, "gender"]), text)
                self.assertIn("age = {}".format(data[100, "age"]), text)
                self.assertIn("gender = {}".format(data[100, "gender"]), text)

            # No points hovered
            with patch.object(scatterplot_item, "pointsAt",
                              return_value=[]):
                show_text.reset_mock()
                self.assertFalse(graph.help_event(event))
                self.assertEqual(show_text.call_count, 0)

    def test_many_discrete_values(self):
        """
        Do not show all discrete values if there are too many.
        Also test for values with a nan.
        GH-2804
        """
        def prepare_data():
            data = Table("iris")
            values = list(range(15))
            class_var = DiscreteVariable("iris5", values=[str(v) for v in values])
            data = data.transform(Domain(attributes=data.domain.attributes, class_vars=[class_var]))
            data.Y = np.array(values * 10, dtype=float)
            return data

        def assert_equal(data, max):
            self.send_signal(self.widget.Inputs.data, data)
            pen_data, brush_data = self.widget.graph.compute_colors()
            self.assertEqual(max, len(np.unique([id(p) for p in pen_data])), )

        assert_equal(prepare_data(), MAX)
        # data with nan value
        data = prepare_data()
        data.Y[42] = np.nan
        assert_equal(data, MAX + 1)


if __name__ == "__main__":
    import unittest
    unittest.main()
