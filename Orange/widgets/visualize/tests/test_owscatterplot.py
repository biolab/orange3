# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,too-many-public-methods,protected-access
# pylint: disable=too-many-lines
from unittest.mock import MagicMock, patch, Mock
import numpy as np

from AnyQt.QtCore import QRectF, Qt
from AnyQt.QtWidgets import QToolTip
from AnyQt.QtGui import QColor, QFont

from Orange.data import (
    Table, Domain, ContinuousVariable, DiscreteVariable, TimeVariable
)
from Orange.widgets.tests.base import (
    WidgetTest, WidgetOutputsTestMixin, datasets, ProjectionWidgetTestMixin
)
from Orange.widgets.tests.utils import simulate
from Orange.widgets.utils.colorpalettes import DefaultRGBColors
from Orange.widgets.visualize.owscatterplot import (
    OWScatterPlot, ScatterPlotVizRank, OWScatterPlotGraph)
from Orange.widgets.visualize.utils.widget import MAX_COLORS
from Orange.widgets.widget import AttributeList


class TestOWScatterPlot(WidgetTest, ProjectionWidgetTestMixin,
                        WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)
        cls.same_input_output_domain = False

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
        self.assertIs(self.widget.attr_color, self.data.domain.class_var)

        # Change which attributes are displayed
        self.widget.attr_x = self.data.domain[2]
        self.widget.attr_y = self.data.domain[3]

        # Disconnect the data
        self.send_signal(self.widget.Inputs.data, None)

        # removing data should have cleared attributes
        self.assertIsNone(self.widget.attr_x)
        self.assertIsNone(self.widget.attr_y)
        self.assertIsNone(self.widget.attr_color)

        # and remove the legend
        self.assertEqual(len(self.widget.graph.color_legend.items), 0)

        # Connect iris again
        # same attributes that were used last time should be selected
        self.send_signal(self.widget.Inputs.data, self.data)

        self.assertIs(self.widget.attr_x, self.data.domain[2])
        self.assertIs(self.widget.attr_y, self.data.domain[3])

    def test_score_heuristics(self):
        domain = Domain([ContinuousVariable(c) for c in "abcd"],
                        DiscreteVariable("e", values="ab"))
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
        t1 = self.data.transform(d1)
        self.send_signal(self.widget.Inputs.data, t1)
        self.widget.graph.attr_size = domain.attributes[2]

        d2 = Domain(domain.attributes[:2], domain.class_var,
                    [domain.attributes[3]])
        t2 = self.data.transform(d2)
        self.send_signal(self.widget.Inputs.data, t2)

    def test_error_message(self):
        """Check if error message appears and then disappears when
        data is removed from input"""
        data = self.data.copy()
        with data.unlocked():
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
        cb_attr_color = self.widget.controls.attr_color
        simulate.combobox_activate_item(cb_attr_color, "b")
        simulate.combobox_activate_item(self.widget.cb_attr_x, "a")
        simulate.combobox_activate_item(self.widget.cb_attr_y, "a")

        #self.widget.update_graph()
        self.widget.graph.reset_graph()

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
        controls = self.widget.controls

        # color and label should contain all variables
        # size should contain only continuous variables
        # shape should contain only discrete variables
        for var in self.data.domain.variables + self.data.domain.metas:
            self.assertIn(var, controls.attr_color.model())
            self.assertIn(var, controls.attr_label.model())
            if var.is_continuous:
                self.assertIn(var, controls.attr_size.model())
                self.assertNotIn(var, controls.attr_shape.model())
            if var.is_discrete:
                self.assertNotIn(var, controls.attr_size.model())
                self.assertIn(var, controls.attr_shape.model())

        widget = self.create_widget(OWScatterPlot)
        self.send_signal(self.widget.Inputs.data, self.data, widget=widget)
        self.assertEqual(controls.attr_color.currentText(),
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
        self.assertEqual(annotations(), ("No", "Yes", ))

        # Shift-select 5:10; now we have groups 0:5 and 5:10
        with self.modifiers(Qt.ShiftModifier):
            graph.select(points[5:10])
        np.testing.assert_equal(selectedx(), x[:10])
        np.testing.assert_equal(selected_groups(), np.array([0] * 5 + [1] * 5))
        sel_column[:5] = 0
        sel_column[5:10] = 1
        sel_column[10:] = 2
        np.testing.assert_equal(annotated(), sel_column)
        self.assertEqual(len(annotations()), 3)

        # Select: 15:20; we have 15:20
        graph.select(points[15:20])
        sel_column = np.zeros((len(self.data), 1))
        sel_column[15:20] = 1
        np.testing.assert_equal(selectedx(), x[15:20])
        np.testing.assert_equal(selected_groups(), np.zeros(5))
        self.assertEqual(annotations(), ("No", "Yes"))

        # Alt-select (remove) 10:17; we have 17:20
        with self.modifiers(Qt.AltModifier):
            graph.select(points[10:17])
        np.testing.assert_equal(selectedx(), x[17:20])
        np.testing.assert_equal(selected_groups(), np.zeros(3))
        sel_column[15:17] = 0
        np.testing.assert_equal(annotated(), sel_column)
        self.assertEqual(annotations(), ("No", "Yes"))

        # Ctrl-Shift-select (add-to-last) 10:17; we have 17:25
        with self.modifiers(Qt.ShiftModifier | Qt.ControlModifier):
            graph.select(points[20:25])
        np.testing.assert_equal(selectedx(), x[17:25])
        np.testing.assert_equal(selected_groups(), np.zeros(8))
        sel_column[20:25] = 1
        np.testing.assert_equal(annotated(), sel_column)
        self.assertEqual(annotations(), ("No", "Yes"))

        # Shift-select (add) 30:35; we have 17:25, 30:35
        with self.modifiers(Qt.ShiftModifier):
            graph.select(points[30:35])
        # ... then Ctrl-Shift-select (add-to-last) 10:17; we have 17:25, 30:40
        with self.modifiers(Qt.ShiftModifier | Qt.ControlModifier):
            graph.select(points[35:40])
        sel_column[:] = 2
        sel_column[17:25] = 0
        sel_column[30:40] = 1
        np.testing.assert_equal(selected_groups(), np.array([0] * 8 + [1] * 10))
        np.testing.assert_equal(annotated(), sel_column)
        self.assertEqual(len(annotations()), 3)

    def test_saving_selection(self):
        self.send_signal(self.widget.Inputs.data, self.data)  # iris
        self.widget.graph.select_by_rectangle(QRectF(4, 3, 3, 1))
        selected_inds = np.flatnonzero(self.widget.graph.selection)
        settings = self.widget.settingsHandler.pack_data(self.widget)
        np.testing.assert_equal(selected_inds,
                                [i for i, g in settings["selection"]])

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
        data = self.data[:11].copy()
        with data.unlocked():
            data[0, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

    def test_set_strings_settings(self):
        """
        Test if settings can be loaded as strings and successfully put
        in new owplotgui combos.
        """
        self.send_signal(self.widget.Inputs.data, self.data)
        settings = self.widget.settingsHandler.pack_data(self.widget)
        plot_settings = settings["context_settings"][0].values
        plot_settings["attr_label"] = ("sepal length", -2)
        plot_settings["attr_color"] = ("sepal width", -2)
        plot_settings["attr_shape"] = ("iris", -2)
        plot_settings["attr_size"] = ("petal width", -2)
        w = self.create_widget(OWScatterPlot, stored_settings=settings)
        self.send_signal(self.widget.Inputs.data, self.data, widget=w)
        self.assertEqual(w.attr_label.name, "sepal length")
        self.assertEqual(w.attr_color.name, "sepal width")
        self.assertEqual(w.attr_shape.name, "iris")
        self.assertEqual(w.attr_size.name, "petal width")

    def test_features_and_no_data(self):
        """
        Prevent crashing when features are sent but no data.
        """
        domain = Table("iris").domain
        self.send_signal(self.widget.Inputs.features,
                         AttributeList(domain.variables))
        self.send_signal(self.widget.Inputs.features, None)

    def test_features_and_data(self):
        self.assertTrue(self.widget.attr_box.isEnabled())
        self.send_signal(self.widget.Inputs.data, self.data)
        x, y = self.widget.graph.scatterplot_item.getData()
        np.testing.assert_array_equal(x, self.data.X[:, 0])
        np.testing.assert_array_equal(y, self.data.X[:, 1])
        self.send_signal(self.widget.Inputs.features,
                         AttributeList(self.data.domain[2:]))
        self.assertIs(self.widget.attr_x, self.data.domain[2])
        self.assertIs(self.widget.attr_y, self.data.domain[3])
        self.assertFalse(self.widget.attr_box.isEnabled())
        self.assertFalse(self.widget.vizrank.isEnabled())
        x, y = self.widget.graph.scatterplot_item.getData()
        np.testing.assert_array_equal(x, self.data.X[:, 2])
        np.testing.assert_array_equal(y, self.data.X[:, 3])

        self.send_signal(self.widget.Inputs.data, None)
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertIs(self.widget.attr_x, self.data.domain[2])
        self.assertIs(self.widget.attr_y, self.data.domain[3])
        self.assertFalse(self.widget.attr_box.isEnabled())
        self.assertFalse(self.widget.vizrank.isEnabled())

        self.send_signal(self.widget.Inputs.features, None)
        self.assertTrue(self.widget.attr_box.isEnabled())
        self.assertTrue(self.widget.vizrank.isEnabled())

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
        self.widget.attr_changed()

        features = self.get_output(self.widget.Outputs.features)
        self.assertEqual(features, [data.domain[0], data.domain[3]])

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
        data2 = Table("iris")[::30].copy()
        with data2.unlocked():
            data2.Y[:] = np.nan
        domain = Domain(
            attributes=data2.domain.attributes[:4], class_vars=DiscreteVariable("iris", values=()))
        data2 = Table(domain, data2.X, Y=data2.Y)
        data3 = Table("iris")[::30].copy()
        with data3.unlocked():
            data3.Y[:] = np.nan

        for data, is_enabled in zip([data1, data2, data1, data3, data1],
                                    [True, False, True, False, True]):
            assert_vizrank_enabled(data, is_enabled)

    def test_vizrank_nonprimitives(self):
        """VizRank does not try to include non primitive attributes"""
        data = Table("zoo")
        self.send_signal(self.widget.Inputs.data, data)
        with patch("Orange.widgets.visualize.owscatterplot.ReliefF",
                   new=lambda *_1, **_2: lambda data: np.arange(len(data))):
            self.widget.vizrank.score_heuristic()

    def test_vizrank_enabled(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertTrue(self.widget.vizrank_button.isEnabled())
        self.assertEqual(self.widget.vizrank_button.toolTip(), "")
        self.assertTrue(self.widget.vizrank.button.isEnabled())
        self.widget.vizrank.button.click()

    def test_vizrank_enabled_no_data(self):
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.vizrank_button.isEnabled())
        self.assertEqual(self.widget.vizrank_button.toolTip(), "No data on input")

    def test_vizrank_enabled_sparse_data(self):
        self.send_signal(self.widget.Inputs.data, self.data.to_sparse())
        self.assertFalse(self.widget.vizrank_button.isEnabled())
        self.assertEqual(self.widget.vizrank_button.toolTip(), "Data is sparse")

    def test_vizrank_enabled_constant_data(self):
        domain = Domain([ContinuousVariable("c1"),
                         ContinuousVariable("c2"),
                         ContinuousVariable("c3"),
                         ContinuousVariable("c4")],
                        DiscreteVariable("cls", values=("a", "b")))
        X = np.zeros((10, 4))
        table = Table(domain, X, np.random.randint(2, size=10))
        self.send_signal(self.widget.Inputs.data, table)
        self.assertEqual(self.widget.vizrank_button.toolTip(), "")
        self.assertTrue(self.widget.vizrank_button.isEnabled())
        self.assertTrue(self.widget.vizrank.button.isEnabled())
        self.widget.vizrank.button.click()

    def test_vizrank_enabled_two_features(self):
        self.send_signal(self.widget.Inputs.data, self.data[:, :2])
        self.assertFalse(self.widget.vizrank_button.isEnabled())
        self.assertEqual(self.widget.vizrank_button.toolTip(),
                         "Not enough features for ranking")

    def test_vizrank_enabled_no_color_var(self):
        self.send_signal(self.widget.Inputs.data, self.data[:, :3])
        self.assertFalse(self.widget.vizrank_button.isEnabled())
        self.assertEqual(self.widget.vizrank_button.toolTip(),
                         "Color variable is not selected")

    def test_vizrank_enabled_color_var_nans(self):
        domain = Domain([ContinuousVariable("c1"),
                         ContinuousVariable("c2"),
                         ContinuousVariable("c3"),
                         ContinuousVariable("c4")],
                        DiscreteVariable("cls", values=("a", "b")))
        table = Table(domain, np.random.random((10, 4)), np.full(10, np.nan))
        self.send_signal(self.widget.Inputs.data, table)
        self.assertFalse(self.widget.vizrank_button.isEnabled())
        self.assertEqual(self.widget.vizrank_button.toolTip(),
                         "Color variable has no values")

    def test_auto_send_selection(self):
        """
        Scatter Plot automatically sends selection only when the checkbox Send automatically
        is checked.
        GH-2649
        GH-2646
        """
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.controls.auto_commit.setChecked(False)
        self.assertFalse(self.widget.controls.auto_commit.isChecked())
        self._select_data()
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        self.widget.controls.auto_commit.setChecked(True)
        output = self.get_output(self.widget.Outputs.selected_data)
        self.assertIsInstance(output, Table)

    def test_color_is_optional(self):
        zoo = Table("zoo")
        backbone, breathes, airborne, type = \
            [zoo.domain[x] for x in ["backbone", "breathes", "airborne", "type"]]
        default_x, default_y, default_color = \
            zoo.domain[0], zoo.domain[1], zoo.domain.class_var
        attr_x = self.widget.controls.attr_x
        attr_y = self.widget.controls.attr_y
        attr_color = self.widget.controls.attr_color

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
        data = data.transform(domain).copy()
        # Sometimes floats in metas are saved as objects
        with data.unlocked():
            data.metas = data.metas.astype(object)
        self.send_signal(w.Inputs.data, data)
        simulate.combobox_activate_item(w.cb_attr_x, data.domain.metas[1].name)
        simulate.combobox_activate_item(w.controls.attr_color, data.domain.metas[0].name)
        w.graph.reset_graph()

    def test_subset_data(self):
        """
        Scatter Plot subset data is sent to Scatter Plot Graph
        GH-2773
        """
        data = Table("iris")
        w = self.widget
        self.send_signal(w.Inputs.data, data)
        self.send_signal(w.Inputs.data_subset, data[::30])
        self.assertEqual(len(w.subset_indices), 5)

    def test_opacity_warning(self):
        data = Table("iris")
        w = self.widget
        self.send_signal(w.Inputs.data, data)
        w.graph.controls.alpha_value.setSliderPosition(10)
        self.assertFalse(w.Warning.transparent_subset.is_shown())
        self.send_signal(w.Inputs.data_subset, data[::30])
        self.assertTrue(w.Warning.transparent_subset.is_shown())
        w.graph.controls.alpha_value.setSliderPosition(200)
        self.assertFalse(w.Warning.transparent_subset.is_shown())
        w.graph.controls.alpha_value.setSliderPosition(10)
        self.assertTrue(w.Warning.transparent_subset.is_shown())
        self.send_signal(w.Inputs.data_subset, None)
        self.assertFalse(w.Warning.transparent_subset.is_shown())

    def test_jittering(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.graph.controls.jitter_continuous.setChecked(True)
        self.widget.graph.controls.jitter_size.setValue(1)

    def test_metas_zero_column(self):
        """
        Prevent crash when metas column is zero.
        GH-2775
        """
        data = Table("iris")
        domain = data.domain
        domain = Domain(domain.attributes[:3], domain.class_vars, domain.attributes[3:])
        data = data.transform(domain).copy()
        with data.unlocked():
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
                widget.tooltip_shows_all = False
                self.assertTrue(graph.help_event(event))
                (_, text), _ = show_text.call_args
                self.assertIn("age = {}".format(data[42, "age"]), text)
                self.assertIn("gender = {}".format(data[42, "gender"]), text)
                self.assertNotIn("max HR = {}".format(data[42, "max HR"]), text)
                self.assertNotIn("others", text)

                # Show all attributes
                widget.tooltip_shows_all = True
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
            data = data.transform(
                Domain(attributes=data.domain.attributes,
                       class_vars=[class_var])).copy()
            with data.unlocked():
                data.Y = np.array(values * 10, dtype=float)
            return data

        def assert_equal(data, max):
            self.send_signal(self.widget.Inputs.data, data)
            pen_data, brush_data = self.widget.graph.get_colors()
            self.assertEqual(max, len(np.unique([id(p) for p in pen_data])), )

        assert_equal(prepare_data(), MAX_COLORS)
        # data with nan value
        data = prepare_data()
        with data.unlocked():
            data.Y[42] = np.nan
        assert_equal(data, MAX_COLORS + 1)

    def test_invalidated_same_features(self):
        self.widget.setup_plot = Mock()
        # send data and set default features
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.setup_plot.assert_called_once()
        self.assertListEqual(self.widget.effective_variables,
                             list(self.data.domain.attributes[:2]))

        # send the same features as already set
        self.widget.setup_plot.reset_mock()
        self.send_signal(self.widget.Inputs.features,
                         AttributeList(self.data.domain.attributes[:2]))
        self.widget.setup_plot.assert_not_called()
        self.assertListEqual(self.widget.effective_variables,
                             list(self.data.domain.attributes[:2]))

    def test_invalidated_same_time(self):
        self.widget.setup_plot = Mock()
        # send data and features at the same time (data first)
        features = self.data.domain.attributes[:2]
        signals = [(self.widget.Inputs.data, self.data),
                   (self.widget.Inputs.features, AttributeList(features))]
        self.send_signals(signals)
        self.widget.setup_plot.assert_called_once()
        self.assertListEqual(self.widget.effective_variables, list(features))

    def test_invalidated_features_first(self):
        self.widget.setup_plot = Mock()
        # send features (same as default ones)
        self.send_signal(self.widget.Inputs.features,
                         AttributeList(self.data.domain.attributes[:2]))
        self.assertListEqual(self.widget.effective_variables, [])
        self.widget.setup_plot.assert_called_once()

        # send data
        self.widget.setup_plot.reset_mock()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.setup_plot.assert_called()
        self.assertListEqual(self.widget.effective_variables,
                             list(self.data.domain.attributes[:2]))

    def test_invalidated_same_time_features_first(self):
        self.widget.setup_plot = Mock()
        # send features and data at the same time (features first)
        features = self.data.domain.attributes[:2]
        signals = [(self.widget.Inputs.features, AttributeList(features)),
                   (self.widget.Inputs.data, self.data)]
        self.send_signals(signals)
        self.widget.setup_plot.assert_called_once()
        self.assertListEqual(self.widget.effective_variables, list(features))

    def test_invalidated_diff_features(self):
        self.widget.setup_plot = Mock()
        # send data and set default features
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.setup_plot.assert_called_once()
        self.assertListEqual(self.widget.effective_variables,
                             list(self.data.domain.attributes[:2]))

        # send different features
        self.widget.setup_plot.reset_mock()
        self.send_signal(self.widget.Inputs.features,
                         AttributeList(self.data.domain.attributes[2:4]))
        self.widget.setup_plot.assert_called_once()
        self.assertListEqual(self.widget.effective_variables,
                             list(self.data.domain.attributes[2:4]))

    def test_invalidated_diff_features_same_time(self):
        self.widget.setup_plot = Mock()
        # send data and different features at the same time (data first)
        features = self.data.domain.attributes[2:4]
        signals = [(self.widget.Inputs.data, self.data),
                   (self.widget.Inputs.features, AttributeList(features))]
        self.send_signals(signals)
        self.widget.setup_plot.assert_called_once()
        self.assertListEqual(self.widget.effective_variables, list(features))

    def test_invalidated_diff_features_features_first(self):
        self.widget.setup_plot = Mock()
        # send features (not the same as defaults)
        self.send_signal(self.widget.Inputs.features,
                         AttributeList(self.data.domain.attributes[2:4]))
        self.assertListEqual(self.widget.effective_variables, [])
        self.widget.setup_plot.assert_called_once()

        # send data
        self.widget.setup_plot.reset_mock()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.setup_plot.assert_called_once()
        self.assertListEqual(self.widget.effective_variables,
                             list(self.data.domain.attributes[2:4]))

    def test_invalidated_diff_features_same_time_features_first(self):
        self.widget.setup_plot = Mock()
        # send data and different features at the same time (features first)
        features = self.data.domain.attributes[2:4]
        signals = [(self.widget.Inputs.features, AttributeList(features)),
                   (self.widget.Inputs.data, self.data)]
        self.send_signals(signals)
        self.widget.setup_plot.assert_called_once()
        self.assertListEqual(self.widget.effective_variables, list(features))

    @patch('Orange.widgets.visualize.owscatterplot.ScatterPlotVizRank.'
           'on_manual_change')
    def test_vizrank_receives_manual_change(self, on_manual_change):
        # Recreate the widget so the patch kicks in
        self.widget = self.create_widget(OWScatterPlot)
        data = Table("iris.tab")
        self.send_signal(self.widget.Inputs.data, data)
        model = self.widget.controls.attr_x.model()
        self.widget.attr_x = model[0]
        self.widget.attr_y = model[1]
        simulate.combobox_activate_index(self.widget.controls.attr_x, 2)
        self.assertIs(self.widget.attr_x, model[2])
        on_manual_change.assert_called_with(model[2], model[1])

    def test_on_manual_change(self):
        data = Table("iris.tab")
        self.send_signal(self.widget.Inputs.data, data)
        vizrank = self.widget.vizrank
        vizrank.toggle()
        self.process_events(until=lambda: not vizrank.keep_running)

        model = vizrank.rank_model
        attrs = model.data(model.index(3, 0), vizrank._AttrRole)
        vizrank.on_manual_change(*attrs)
        selection = vizrank.rank_table.selectedIndexes()
        self.assertEqual(len(selection), 1)
        self.assertEqual(selection[0].row(), 3)

        vizrank.on_manual_change(*attrs[::-1])
        selection = vizrank.rank_table.selectedIndexes()
        self.assertEqual(len(selection), 0)

    def test_regression_lines_appear(self):
        self.widget.graph.controls.show_reg_line.setChecked(True)
        self.assertEqual(len(self.widget.graph.reg_line_items), 0)
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertEqual(len(self.widget.graph.reg_line_items), 4)
        simulate.combobox_activate_index(self.widget.controls.attr_color, 0)
        self.assertEqual(len(self.widget.graph.reg_line_items), 1)
        data = self.data.copy()
        with data.unlocked():
            data[:, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        self.assertEqual(len(self.widget.graph.reg_line_items), 0)

    def test_regression_line_coeffs(self):
        widget = self.widget
        graph = widget.graph
        xy = np.array([[0, 0], [1, 0], [1, 2], [2, 2],
                       [0, 1], [1, 3], [2, 5]], dtype=np.float)
        colors = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.float)
        widget.get_coordinates_data = lambda: xy.T
        widget.can_draw_regresssion_line = lambda: True
        widget.get_color_data = lambda: colors
        widget.is_continuous_color = lambda: False
        graph.palette = DefaultRGBColors
        graph.controls.show_reg_line.setChecked(True)

        graph.update_regression_line()

        line1 = graph.reg_line_items[1]
        self.assertEqual(line1.pos().x(), 0)
        self.assertEqual(line1.pos().y(), 0)
        self.assertEqual(line1.angle, 45)
        self.assertEqual(line1.pen.color().hue(), graph.palette[0].hue())

        line2 = graph.reg_line_items[2]
        self.assertEqual(line2.pos().x(), 0)
        self.assertEqual(line2.pos().y(), 1)
        self.assertAlmostEqual(line2.angle, np.degrees(np.arctan2(2, 1)))
        self.assertEqual(line2.pen.color().hue(), graph.palette[1].hue())

        graph.orthonormal_regression = True
        graph.update_regression_line()

        line1 = graph.reg_line_items[1]
        self.assertEqual(line1.pos().x(), 0)
        self.assertAlmostEqual(line1.pos().y(), -0.6180339887498949)
        self.assertAlmostEqual(line1.angle, 58.28252558853899)
        self.assertEqual(line1.pen.color().hue(), graph.palette[0].hue())

        line2 = graph.reg_line_items[2]
        self.assertEqual(line2.pos().x(), 0)
        self.assertEqual(line2.pos().y(), 1)
        self.assertAlmostEqual(line2.angle, np.degrees(np.arctan2(2, 1)))
        self.assertEqual(line2.pen.color().hue(), graph.palette[1].hue())

    def test_orthonormal_line(self):
        color = QColor(1, 2, 3)
        width = 42
        # Normal line
        line = OWScatterPlotGraph._orthonormal_line(
            np.array([0, 1, 1, 2]), np.array([0, 0, 2, 2]), color, width)
        self.assertEqual(line.pos().x(), 0)
        self.assertAlmostEqual(line.pos().y(), -0.6180339887498949)
        self.assertAlmostEqual(line.angle, 58.28252558853899)
        self.assertEqual(line.pen.color(), color)
        self.assertEqual(line.pen.width(), width)

        # Normal line, negative slope
        line = OWScatterPlotGraph._orthonormal_line(
            np.array([1, 2, 3]), np.array([3, 2, 1]), color, width)
        self.assertEqual(line.pos().x(), 1)
        self.assertEqual(line.pos().y(), 3)
        self.assertEqual(line.angle % 360, 315)

        # Horizontal line
        line = OWScatterPlotGraph._orthonormal_line(
            np.array([10, 11, 12]), np.array([42, 42, 42]), color, width)
        self.assertEqual(line.pos().x(), 10)
        self.assertEqual(line.pos().y(), 42)
        self.assertEqual(line.angle, 0)

        # Vertical line
        line = OWScatterPlotGraph._orthonormal_line(
            np.array([42, 42, 42]), np.array([10, 11, 12]), color, width)
        self.assertEqual(line.pos().x(), 42)
        self.assertEqual(line.pos().y(), 10)
        self.assertEqual(line.angle, 90)

        # No line because all points coincide
        line = OWScatterPlotGraph._orthonormal_line(
            np.array([1, 1, 1]), np.array([42, 42, 42]), color, width)
        self.assertIsNone(line)

        # No line because the group is symmetric
        line = OWScatterPlotGraph._orthonormal_line(
            np.array([1, 1, 2, 2]), np.array([42, 5, 5, 42]), color, width)
        self.assertIsNone(line)

    def test_regression_line(self):
        color = QColor(1, 2, 3)
        width = 42
        # Normal line
        line = OWScatterPlotGraph._regression_line(
            np.array([0, 1, 1, 2]), np.array([0, 0, 2, 2]), color, width)
        self.assertEqual(line.pos().x(), 0)
        self.assertAlmostEqual(line.pos().y(), 0)
        self.assertEqual(line.angle, 45)
        self.assertEqual(line.pen.color(), color)
        self.assertEqual(line.pen.width(), width)

        # Normal line, negative slope
        line = OWScatterPlotGraph._regression_line(
            np.array([1, 2, 3]), np.array([3, 2, 1]), color, width)
        self.assertEqual(line.pos().x(), 1)
        self.assertEqual(line.pos().y(), 3)
        self.assertEqual(line.angle % 360, 315)

        # Horizontal line
        line = OWScatterPlotGraph._regression_line(
            np.array([10, 11, 12]), np.array([42, 42, 42]), color, width)
        self.assertEqual(line.pos().x(), 10)
        self.assertEqual(line.pos().y(), 42)
        self.assertEqual(line.angle, 0)

        # Vertical line
        line = OWScatterPlotGraph._regression_line(
            np.array([42, 42, 42]), np.array([10, 11, 12]), color, width)
        self.assertIsNone(line)

        # No line because all points coincide
        line = OWScatterPlotGraph._regression_line(
            np.array([1, 1, 1]), np.array([42, 42, 42]), color, width)
        self.assertIsNone(line)

    def test_add_line_calls_proper_regressor(self):
        graph = self.widget.graph
        graph._orthonormal_line = Mock(return_value=None)
        graph._regression_line = Mock(return_value=None)
        x, y, c = Mock(), Mock(), Mock()

        graph.orthonormal_regression = True
        graph._add_line(x, y, c)
        graph._orthonormal_line.assert_called_once_with(x, y, c, 3, 1)
        graph._regression_line.assert_not_called()
        graph._orthonormal_line.reset_mock()

        graph.orthonormal_regression = False
        graph._add_line(x, y, c)
        graph._regression_line.assert_called_with(x, y, c, 3, 1)
        graph._orthonormal_line.assert_not_called()

    def test_no_regression_line(self):
        graph = self.widget.graph
        graph._orthonormal_line = lambda *_: None
        graph.orthonormal_regression = True

        graph.plot_widget.addItem = Mock()

        x, y, c = Mock(), Mock(), Mock()
        graph._add_line(x, y, c)
        graph.plot_widget.addItem.assert_not_called()
        self.assertEqual(graph.reg_line_items, [])

    def test_update_regression_line_calls_add_line(self):
        widget = self.widget
        graph = widget.graph
        x, y = np.array([[0, 0], [1, 0], [1, 2], [2, 2],
                         [0, 1], [1, 3], [2, 5]], dtype=np.float).T
        colors = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.float)
        widget.get_coordinates_data = lambda: (x, y)
        widget.can_draw_regresssion_line = lambda: True
        widget.get_color_data = lambda: colors
        widget.is_continuous_color = lambda: False
        graph.palette = DefaultRGBColors
        graph.controls.show_reg_line.setChecked(True)

        graph._add_line = Mock()

        graph.update_regression_line()
        (args1, _), (args2, _), (args3, _) = graph._add_line.call_args_list
        np.testing.assert_equal(args1[0], x)
        np.testing.assert_equal(args1[1], y)
        self.assertEqual(args1[2], QColor("#505050"))

        np.testing.assert_equal(args2[0], x[:4])
        np.testing.assert_equal(args2[1], y[:4])
        self.assertEqual(args2[2].hue(), graph.palette[0].hue())

        np.testing.assert_equal(args3[0], x[4:])
        np.testing.assert_equal(args3[1], y[4:])
        self.assertEqual(args3[2].hue(), graph.palette[1].hue())
        graph._add_line.reset_mock()

        # Continuous color - just a single line
        widget.is_continuous_color = lambda: True
        graph.update_regression_line()
        graph._add_line.assert_called_once()
        args1, _ = graph._add_line.call_args_list[0]
        np.testing.assert_equal(args1[0], x)
        np.testing.assert_equal(args1[1], y)
        self.assertEqual(args1[2].hue(), QColor("#505050").hue())
        graph._add_line.reset_mock()
        widget.is_continuous_color = lambda: False

        # No palette - just a single line
        graph.palette = None
        graph.update_regression_line()
        graph._add_line.assert_called_once()
        graph._add_line.reset_mock()
        graph.palette = DefaultRGBColors

        # Regression line is disabled
        graph.show_reg_line = False
        graph.update_regression_line()
        graph._add_line.assert_not_called()
        graph.show_reg_line = True

        # No colors - just one line
        widget.get_color_data = lambda: None
        graph.update_regression_line()
        graph._add_line.assert_called_once()
        graph._add_line.reset_mock()

        # No data
        widget.get_coordinates_data = lambda: (None, None)
        graph.update_regression_line()
        graph._add_line.assert_not_called()
        graph.show_reg_line = True
        widget.get_coordinates_data = lambda: (x, y)

        # One color group contains just one point - skip that line
        widget.get_color_data = lambda: np.array([0] + [1] * (len(x) - 1))

        graph.update_regression_line()
        (args1, _), (args2, _) = graph._add_line.call_args_list
        np.testing.assert_equal(args1[0], x)
        np.testing.assert_equal(args1[1], y)
        self.assertEqual(args1[2].hue(), QColor("#505050").hue())

        np.testing.assert_equal(args2[0], x[1:])
        np.testing.assert_equal(args2[1], y[1:])
        self.assertEqual(args2[2].hue(), graph.palette[1].hue())

    def test_update_regression_line_is_called(self):
        widget = self.widget
        graph = widget.graph
        urline = graph.update_regression_line = Mock()

        self.send_signal(widget.Inputs.data, self.data)
        urline.assert_called_once()
        urline.reset_mock()

        self.send_signal(widget.Inputs.data, None)
        urline.assert_called_once()
        urline.reset_mock()

        self.send_signal(widget.Inputs.data, self.data)
        urline.assert_called_once()
        urline.reset_mock()

        simulate.combobox_activate_index(self.widget.controls.attr_color, 0)
        urline.assert_called_once()
        urline.reset_mock()

        simulate.combobox_activate_index(self.widget.controls.attr_color, 2)
        urline.assert_called_once()
        urline.reset_mock()

        simulate.combobox_activate_index(self.widget.controls.attr_x, 3)
        urline.assert_called_once()
        urline.reset_mock()

    def test_time_axis(self):
        a = np.array([[1581953776, 1], [1581963776, 2], [1582953776, 3]])
        d1 = Domain([ContinuousVariable("time"), ContinuousVariable("value")])
        data = Table.from_numpy(d1, a)
        d2 = Domain([TimeVariable("time"), ContinuousVariable("value")])
        data_time = Table.from_numpy(d2, a)

        x_axis = self.widget.graph.plot_widget.plotItem.getAxis("bottom")

        self.send_signal(self.widget.Inputs.data, data)
        self.assertFalse(x_axis._use_time)
        _ticks = x_axis.tickValues(1581953776, 1582953776, 1000)
        ticks = x_axis.tickStrings(_ticks[0][1], 1, _ticks[0][0])
        try:
            float(ticks[0])
        except ValueError:
            self.fail("axis should display floats")

        self.send_signal(self.widget.Inputs.data, data_time)
        self.assertTrue(x_axis._use_time)
        _ticks = x_axis.tickValues(1581953776, 1582953776, 1000)
        ticks = x_axis.tickStrings(_ticks[0][1], 1, _ticks[0][0])
        with self.assertRaises(ValueError):
            float(ticks[0])

    def test_visual_settings(self):
        super().test_visual_settings()

        graph = self.widget.graph
        font = QFont()
        font.setItalic(True)
        font.setFamily("Helvetica")

        key, value = ('Fonts', 'Axis title', 'Font size'), 16
        self.widget.set_visual_settings(key, value)
        key, value = ('Fonts', 'Axis title', 'Italic'), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(16)
        for item in graph.parameter_setter.axis_items:
            self.assertFontEqual(item.label.font(), font)

        key, value = ('Fonts', 'Axis ticks', 'Font size'), 15
        self.widget.set_visual_settings(key, value)
        key, value = ('Fonts', 'Axis ticks', 'Italic'), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(15)
        for item in graph.parameter_setter.axis_items:
            self.assertFontEqual(item.style["tickFont"], font)

        self.widget.graph.controls.show_reg_line.setChecked(True)
        self.assertGreater(len(graph.parameter_setter.reg_line_label_items), 0)

        key, value = ('Fonts', 'Line label', 'Font size'), 16
        self.widget.set_visual_settings(key, value)
        key, value = ('Fonts', 'Line label', 'Italic'), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(16)
        for label in graph.parameter_setter.reg_line_label_items:
            self.assertFontEqual(label.textItem.font(), font)

        key, value = ('Figure', 'Lines', 'Width'), 10
        self.widget.set_visual_settings(key, value)
        for item in graph.reg_line_items:
            self.assertEqual(item.pen.width(), 10)


if __name__ == "__main__":
    import unittest
    unittest.main()
