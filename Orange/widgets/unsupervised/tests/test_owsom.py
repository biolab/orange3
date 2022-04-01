# pylint: disable=protected-access

import unittest
from unittest.mock import Mock, patch

import numpy as np
import scipy.sparse as sp

from Orange.data import Table, Domain
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_FEATURE_NAME
from Orange.widgets.unsupervised.owsom import OWSOM, SomView, SOM


def _patch_recompute_som(meth):
    def winners_from_weights(cont_x, *_1, **_2):
        n = cont_x.shape[0]
        w = np.zeros((n, 2), dtype=int)
        w[n // 5:] = [0, 1]
        w[n // 3:] = [1, 2]
        return w

    def recompute(self):
        if not self.data:
            return
        self._assign_instances(None, None)
        self._redraw()
        self.update_output()

    def patched(*args, **kwargs):
        with patch.object(SOM, "winner_from_weights", new=winners_from_weights), \
                patch.object(OWSOM, "_recompute_som", new=recompute):
            meth(*args, **kwargs)

    return patched


class TestOWSOM(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSOM)  #: OWSOM
        self.iris = Table("iris")

    def tearDown(self):
        self.widget.onDeleteWidget()
        super().tearDown()

    def test_requires_continuous(self):
        widget = self.widget
        heart = Table("heart_disease")

        self.send_signal(widget.Inputs.data, Table("zoo"))
        self.assertTrue(widget.Error.no_numeric_variables.is_shown())
        self.assertFalse(widget.Warning.ignoring_disc_variables.is_shown())
        self.assertIsNone(widget.data)
        self.assertIsNone(widget.cont_x)

        self.send_signal(widget.Inputs.data, heart)
        self.assertFalse(widget.Error.no_numeric_variables.is_shown())
        self.assertTrue(widget.Warning.ignoring_disc_variables.is_shown())
        self.assertEqual(
            widget.cont_x.shape[1],
            sum(attr.is_continuous for attr in heart.domain.attributes))

        self.send_signal(widget.Inputs.data, self.iris)
        self.assertFalse(widget.Error.no_numeric_variables.is_shown())
        self.assertFalse(widget.Warning.ignoring_disc_variables.is_shown())
        self.assertEqual(widget.cont_x.shape, (150, 4))

    def test_single_attribute(self):
        widget = self.widget
        self.send_signal(widget.Inputs.data, self.iris)
        self.assertFalse(widget.Warning.single_attribute.is_shown())
        iris = self.iris[:, 0]
        self.send_signal(widget.Inputs.data, iris)
        self.assertTrue(widget.Warning.single_attribute.is_shown())

    def test_missing_all_data(self):
        widget = self.widget

        self.send_signal(widget.Inputs.data, Table("heart_disease"))
        self.assertTrue(widget.Warning.ignoring_disc_variables.is_shown())

        with self.iris.unlocked():
            for i in range(150):
                self.iris.X[i, i % 4] = np.nan

        self.send_signal(widget.Inputs.data, self.iris)
        self.assertTrue(widget.Error.no_defined_rows.is_shown())
        self.assertFalse(widget.Warning.ignoring_disc_variables.is_shown())
        self.assertIsNone(widget.data)
        self.assertIsNone(widget.cont_x)

        self.send_signal(widget.Inputs.data, None)
        self.assertFalse(widget.Warning.ignoring_disc_variables.is_shown())

    def test_missing_some_data(self):
        widget = self.widget
        with self.iris.unlocked():
            self.iris.X[:50, 0] = np.nan

        self.send_signal(widget.Inputs.data, self.iris)
        self.assertFalse(widget.Error.no_defined_rows.is_shown())
        self.assertTrue(widget.Warning.missing_values.is_shown())
        np.testing.assert_almost_equal(
            widget.data.Y.flatten(), [1] * 50 + [2] * 50)
        self.assertEqual(widget.cont_x.shape, (100, 4))

        self.send_signal(widget.Inputs.data, None)
        self.assertFalse(widget.Warning.missing_values.is_shown())

    def test_missing_one_row_data(self):
        widget = self.widget
        with self.iris.unlocked():
            self.iris.X[5, 0] = np.nan

        self.send_signal(widget.Inputs.data, self.iris)
        self.assertFalse(widget.Error.no_defined_rows.is_shown())
        self.assertTrue(widget.Warning.missing_values.is_shown())

        self.send_signal(widget.Inputs.data, None)
        self.assertFalse(widget.Warning.missing_values.is_shown())

    @_patch_recompute_som
    def test_sparse_data(self):
        widget = self.widget
        with self.iris.unlocked():
            self.iris.X = sp.csc_matrix(self.iris.X)

        # Table.from_table can decide to return dense data
        with patch.object(Table, "from_table", lambda _, x: x):
            # update_output would fail because of above patch
            widget.update_output = Mock()
            self.send_signal(widget.Inputs.data, self.iris)
        self.assertIs(widget.data, self.iris)
        self.assertTrue(sp.isspmatrix_csr(widget.cont_x))
        self.assertEqual(widget.cont_x.shape, (150, 4))

    def test_auto_compute_dimensions(self):
        widget = self.widget
        self.send_signal(widget.Inputs.data, self.iris)
        spin_x = widget.opt_controls.spin_x
        spin_y = widget.opt_controls.spin_y

        spin_x.setValue(100)
        spin_y.setValue(100)
        widget.auto_dimension = True
        widget.controls.auto_dimension.toggled.emit(True)
        self.assertLess(spin_x.value(), 100)
        self.assertLess(spin_y.value(), 100)

        spin_x.setValue(9)
        spin_y.setValue(7)
        widget.auto_dimension = False
        widget.controls.auto_dimension.toggled.emit(False)
        self.assertEqual(spin_x.value(), 10)
        self.assertEqual(spin_y.value(), 5)

    @_patch_recompute_som
    def test_redraw_grid(self):
        widget = self.widget

        widget.auto_dimension = False

        widget.opt_controls.spin_x.setValue(8)
        widget.opt_controls.spin_y.setValue(5)
        widget.opt_controls.shape.setCurrentIndex(1)
        self.send_signal(widget.Inputs.data, self.iris)
        self.assertEqual(widget.grid_cells.shape, (5, 8))
        self.assertEqual(
            sum(c is not None for c in widget.grid_cells.ravel()), 40)

        widget.opt_controls.shape.setCurrentIndex(0)
        widget.opt_controls.start.clicked.emit()
        self.assertEqual(widget.grid_cells.shape, (5, 8))
        self.assertEqual(
            sum(c is not None for c in widget.grid_cells.ravel()), 38)
        self.assertIsNone(widget.grid_cells[1, 7])
        self.assertIsNone(widget.grid_cells[3, 7])

        widget.opt_controls.shape.setCurrentIndex(1)
        widget.opt_controls.start.clicked.emit()
        self.assertEqual(widget.grid_cells.shape, (5, 8))
        self.assertEqual(
            sum(c is not None for c in widget.grid_cells.ravel()), 40)

    def test_attr_color_change(self):
        widget = self.widget
        widget._redraw = Mock()

        heart = Table("heart_disease")
        self.send_signal(widget.Inputs.data, heart)

        widget._redraw.reset_mock()
        combo = widget.controls.attr_color
        combo.setCurrentIndex(0)
        combo.activated[int].emit(0)
        self.assertFalse(widget.controls.pie_charts.isEnabled())
        self.assertIsNone(widget.colors)
        self.assertIsNone(widget.thresholds)
        widget._redraw.assert_called()

        widget._redraw.reset_mock()
        combo = widget.controls.attr_color
        gender = heart.domain["gender"]
        ind_gen = combo.model().indexOf(gender)
        combo.setCurrentIndex(ind_gen)
        combo.activated[int].emit(ind_gen)
        self.assertTrue(widget.controls.pie_charts.isEnabled())
        np.testing.assert_equal(widget.colors.palette, gender.colors)
        self.assertIsNone(widget.thresholds)
        widget._redraw.assert_called()

        widget._redraw.reset_mock()
        combo = widget.controls.attr_color
        age = heart.domain["age"]
        ind_age = combo.model().indexOf(age)
        combo.setCurrentIndex(ind_age)
        combo.activated[int].emit(ind_age)
        self.assertTrue(widget.controls.pie_charts.isEnabled())
        self.assertIsNotNone(widget.thresholds)
        widget._redraw.assert_called()

    def test_colored_circles_with_constant(self):
        domain = self.iris.domain
        self.widget.pie_charts = False

        with self.iris.unlocked():
            self.iris.X[:, 0] = 1
        self.send_signal(self.widget.Inputs.data, self.iris)
        attr0 = domain.attributes[0]

        combo = self.widget.controls.attr_color
        simulate.combobox_activate_index(combo, combo.model().indexOf(attr0))
        self.assertIsNotNone(self.widget.colors)
        self.assertFalse(self.widget.Warning.no_defined_colors.is_shown())

        dom1 = Domain(domain.attributes[1:], domain.class_var,
                      domain.attributes[:1])
        iris = self.iris.transform(dom1).copy()
        with iris.unlocked(iris.metas):
            iris.metas[::2, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, iris)
        simulate.combobox_activate_index(combo, combo.model().indexOf(attr0))
        self.assertIsNotNone(self.widget.colors)
        self.assertFalse(self.widget.Warning.no_defined_colors.is_shown())

        iris = self.iris.transform(dom1).copy()
        with iris.unlocked(iris.metas):
            iris.metas[:, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, iris)
        simulate.combobox_activate_index(combo, combo.model().indexOf(attr0))
        self.assertIsNone(self.widget.colors)
        self.assertTrue(self.widget.Warning.no_defined_colors.is_shown())

        simulate.combobox_activate_index(combo, 0)
        self.assertIsNone(self.widget.colors)
        self.assertFalse(self.widget.Warning.no_defined_colors.is_shown())

    @_patch_recompute_som
    def test_cell_sizes(self):
        widget = self.widget
        widget._draw_same_color = widget._draw_pie_charts = \
            widget._draw_colored_circles = draw = Mock()

        widget.size_by_instances = False
        self.send_signal(widget.Inputs.data, self.iris)

        widget.size_by_instances = True
        widget.pie_charts = False
        widget.controls.size_by_instances.toggled.emit(True)
        sizes = draw.call_args[0][0]  # pylint: disable=unsubscriptable-object
        self.assertAlmostEqual(sizes[0, 0], 0.8 * (30 / 100))
        self.assertAlmostEqual(sizes[0, 1], 0.8 * (20 / 100))
        self.assertAlmostEqual(sizes[1, 2], 0.8)
        sizes[0, 0] = sizes[0, 1] = sizes[1, 2] = 0
        self.assertTrue(np.all(sizes == 0))

        widget.size_by_instances = False
        widget.controls.size_by_instances.toggled.emit(False)
        sizes = draw.call_args[0][0]  # pylint: disable=unsubscriptable-object
        self.assertAlmostEqual(sizes[0, 0], 0.8)
        self.assertAlmostEqual(sizes[0, 1], 0.8)
        self.assertAlmostEqual(sizes[0, 1], 0.8)
        sizes[0, 0] = sizes[0, 1] = sizes[1, 2] = 0
        self.assertTrue(np.all(sizes == 0))

    @_patch_recompute_som
    def test_same_color_same_size(self):
        widget = self.widget

        widget.size_by_instances = False

        self.send_signal(widget.Inputs.data, self.iris)
        widget.attr_color = None
        widget.controls.attr_color.activated[int].emit(0)
        a = widget.elements.childItems()[0]
        for e, t in zip(widget.elements.childItems(),
                        [(0, 0, 0.4),
                         (0.5, np.sqrt(3) / 2, 0.4),
                         (1, np.sqrt(3), 0.4)]):
            np.testing.assert_almost_equal((e.x(), e.y(), e.r), t)
            self.assertEqual(e.color.getRgb(), a.color.getRgb())

        widget.opt_controls.shape.setCurrentIndex(1)
        widget.opt_controls.start.clicked.emit()
        a = widget.elements.childItems()[0]
        for e, t in zip(widget.elements.childItems(),
                        [(0, 0, 0.4), (0, 1, 0.4), (1, 2, 0.4)]):
            np.testing.assert_almost_equal((e.x(), e.y(), e.r), t)
            self.assertEqual(e.color.getRgb(), a.color.getRgb())

    @_patch_recompute_som
    def test_diff_color_same_size(self):
        widget = self.widget

        widget.size_by_instances = False

        widget.opt_controls.shape.setCurrentIndex(0)
        self.send_signal(widget.Inputs.data, self.iris)
        a, b, c = widget.elements.childItems()
        for e, t in zip(widget.elements.childItems(),
                        [(0, 0, 0.4),
                         (0.5, np.sqrt(3) / 2, 0.4),
                         (1, np.sqrt(3), 0.4)]):
            np.testing.assert_almost_equal((e.x(), e.y(), e.r), t)
        self.assertEqual(a.color.getRgb(), b.color.getRgb())
        self.assertNotEqual(a.color.getRgb(), c.color.getRgb())

        widget.opt_controls.shape.setCurrentIndex(1)
        widget.opt_controls.start.clicked.emit()
        a, b, c = widget.elements.childItems()
        for e, t in zip(widget.elements.childItems(),
                        [(0, 0, 0.4), (0, 1, 0.4), (1, 2, 0.4)]):
            np.testing.assert_almost_equal((e.x(), e.y(), e.r), t)
        self.assertEqual(a.color.getRgb(), b.color.getRgb())
        self.assertNotEqual(a.color.getRgb(), c.color.getRgb())

    @_patch_recompute_som
    def test_same_color_diff_size(self):
        widget = self.widget

        widget.size_by_instances = True
        widget.opt_controls.shape.setCurrentIndex(0)

        self.send_signal(widget.Inputs.data, self.iris)
        widget.attr_color = None
        widget.controls.attr_color.activated[int].emit(0)
        a = widget.elements.childItems()[0]
        for e, t in zip(widget.elements.childItems(),
                        [(0, 0, 0.4 * (30 / 100)),
                         (0.5, np.sqrt(3) / 2, 0.4 * (20 / 100)),
                         (1, np.sqrt(3), 0.4)]):
            np.testing.assert_almost_equal((e.x(), e.y(), e.r), t)
            self.assertEqual(e.color.getRgb(), a.color.getRgb())

        widget.opt_controls.shape.setCurrentIndex(1)
        widget.opt_controls.start.clicked.emit()
        a = widget.elements.childItems()[0]
        for e, t in zip(widget.elements.childItems(),
                        [(0, 0, 0.4 * 30 / 100),
                         (0, 1, 0.4 * 20 / 100),
                         (1, 2, 0.4)]):
            np.testing.assert_almost_equal((e.x(), e.y(), e.r), t)
            self.assertEqual(e.color.getRgb(), a.color.getRgb())

    @_patch_recompute_som
    def test_diff_color_diff_size(self):
        widget = self.widget

        widget.size_by_instances = True

        widget.opt_controls.shape.setCurrentIndex(0)
        self.send_signal(widget.Inputs.data, self.iris)
        a, b, c = widget.elements.childItems()
        for e, t in zip(widget.elements.childItems(),
                        [(0, 0, 0.4 * 30 / 100),
                         (0.5, np.sqrt(3) / 2, 0.4 * 20 / 100),
                         (1, np.sqrt(3), 0.4)]):
            np.testing.assert_almost_equal((e.x(), e.y(), e.r), t)
        self.assertEqual(a.color.getRgb(), b.color.getRgb())
        self.assertNotEqual(a.color.getRgb(), c.color.getRgb())

        widget.opt_controls.shape.setCurrentIndex(1)
        widget.opt_controls.start.clicked.emit()
        a, b, c = widget.elements.childItems()
        for e, t in zip(widget.elements.childItems(),
                        [(0, 0, 0.4 * 30 / 100),
                         (0, 1, 0.4 * 20 / 100),
                         (1, 2, 0.4)]):
            np.testing.assert_almost_equal((e.x(), e.y(), e.r), t)
        self.assertEqual(a.color.getRgb(), b.color.getRgb())
        self.assertNotEqual(a.color.getRgb(), c.color.getRgb())

    @_patch_recompute_som
    def test_pie_charts(self):
        widget = self.widget

        widget.size_by_instances = True

        widget.opt_controls.shape.setCurrentIndex(0)
        self.send_signal(widget.Inputs.data, self.iris)

        widget.pie_charts = True
        widget.controls.pie_charts.toggled.emit(True)

        a, b, c = widget.elements.childItems()
        for e, (x, y, r) in zip(widget.elements.childItems(),
                                [(0, 0, 0.8 * 30 / 100),
                                 (0.5, np.sqrt(3) / 2, 0.8 * 20 / 100),
                                 (1, np.sqrt(3), 0.8)]):
            self.assertEqual(e.x(), x)
            self.assertEqual(e.y(), y)
            self.assertEqual(e.r, r / 2)
            self.assertEqual(len(e.colors), len(widget.colors) + 1)
        np.testing.assert_equal(a.dist, [1, 0, 0, 0])
        np.testing.assert_equal(b.dist, [1, 0, 0, 0])
        np.testing.assert_equal(c.dist, [0, 0.5, 0.5, 0])

        widget.opt_controls.shape.setCurrentIndex(1)
        widget.opt_controls.start.clicked.emit()
        for e, (x, y, r) in zip(widget.elements.childItems(),
                                [(0, 0, 0.8 * 30 / 100),
                                 (0, 1, 0.8 * 20 / 100),
                                 (1, 2, 0.8)]):
            self.assertEqual(e.x(), x)
            self.assertEqual(e.y(), y)
            self.assertEqual(e.r, r / 2)

        with self.iris.unlocked():
            self.iris.Y[:15] = np.nan
        self.send_signal(widget.Inputs.data, self.iris)
        a = widget.elements.childItems()[0]
        np.testing.assert_equal(a.dist, [0.5, 0, 0, 0.5])

    def test_get_color_column(self):
        widget = self.widget

        table = Table("heart_disease")
        domain = table.domain
        new_domain = Domain(
            domain.attributes[3:], domain.class_var, domain.attributes[:3])
        new_table = table.transform(new_domain).copy()
        with new_table.unlocked(new_table.metas):
            new_table.metas = new_table.metas.astype(object)
        self.send_signal(widget.Inputs.data, new_table)

        # discrete attribute
        widget.attr_color = domain["rest ECG"]
        np.testing.assert_equal(
            widget._get_color_column(),
            widget.data.get_column_view("rest ECG")[0].astype(int))

        # discrete meta
        widget.attr_color = domain["gender"]
        np.testing.assert_equal(
            widget._get_color_column(),
            widget.data.get_column_view("gender")[0].astype(int))


        # numeric attribute
        widget.thresholds = np.array([120, 150])
        widget.attr_color = domain["max HR"]
        for c, d in zip(widget._get_color_column(),
                        widget.data.get_column_view("max HR")[0]):
            if d < 120:
                self.assertEqual(c, 0)
            if 120 <= d < 150:
                self.assertEqual(c, 1)
            if d >= 150:
                self.assertEqual(c, 2)

        # numeric meta
        widget.thresholds = np.array([50, 60])
        widget.attr_color = domain["age"]
        for c, d in zip(widget._get_color_column(),
                        widget.data.get_column_view("age")[0]):
            if d < 50:
                self.assertEqual(c, 0)
            if 50 <= d < 60:
                self.assertEqual(c, 1)
            if d >= 60:
                self.assertEqual(c, 2)

        # discrete meta with missing values
        widget.attr_color = domain["gender"]
        with widget.data.unlocked():
            col = widget.data.get_column_view("gender")[0]
            col[:5] = np.nan
        col = col.copy()
        col[:5] = 2
        np.testing.assert_equal(widget._get_color_column(), col)

    @_patch_recompute_som
    def test_colored_circles_with_missing_values(self):
        with self.iris.unlocked():
            self.iris.get_column_view("iris")[0][:5] = np.nan
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertTrue(self.widget.Warning.missing_colors.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.missing_colors.is_shown())

    def test_send_report(self):
        self.widget.send_report()

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.send_report()

    @_patch_recompute_som
    def test_on_selection_change(self):
        def selm(*cells):
            m = np.zeros((widget.size_x, widget.size_y), dtype=bool)
            for x, y in cells:
                m[x, y] = True
            return m

        widget = self.widget
        self.send_signal(self.widget.Inputs.data, self.iris)

        widget.redraw_selection = Mock()
        widget.update_output = Mock()

        m = selm((0, 0)).astype(int)
        widget.on_selection_change(selm((0, 0)))
        np.testing.assert_equal(widget.selection, m)
        widget.redraw_selection.assert_called_once()
        widget.update_output.assert_called_once()

        m = selm((0, 1)).astype(int)
        widget.on_selection_change(selm((0, 1)))
        np.testing.assert_equal(widget.selection, m)

        m[0, 0] = 1
        widget.on_selection_change(selm((0, 0)), SomView.SelectionAddToGroup)
        np.testing.assert_equal(widget.selection, m)

        m[0, 0] = 0
        widget.on_selection_change(selm((0, 0)), SomView.SelectionRemove)
        np.testing.assert_equal(widget.selection, m)

        m[0, 0] = 2
        widget.on_selection_change(selm((0, 0)), SomView.SelectionNewGroup)
        np.testing.assert_equal(widget.selection, m)

    @_patch_recompute_som
    def test_on_selection_change_on_empty(self):
        """Test clicks on empty scene, when no data"""
        widget = self.widget
        widget.on_selection_change([])

    @_patch_recompute_som
    def test_output(self):
        widget = self.widget
        self.send_signal(self.widget.Inputs.data, self.iris)

        self.assertIsNone(self.get_output(widget.Outputs.selected_data))
        out = self.get_output(widget.Outputs.annotated_data)
        self.assertEqual(len(out), 150)
        self.assertTrue(
            np.all(out.get_column_view(ANNOTATED_DATA_FEATURE_NAME)[0] == 0))

        m = np.zeros((widget.size_x, widget.size_y), dtype=bool)
        m[0, 0] = True
        widget.on_selection_change(m)
        out = self.get_output(widget.Outputs.selected_data)
        np.testing.assert_equal(out.ids, self.iris.ids[:30])

        out = self.get_output(widget.Outputs.annotated_data)
        np.testing.assert_equal(
            out.get_column_view(ANNOTATED_DATA_FEATURE_NAME)[0],
            [1] * 30 + [0] * 120)

        m[0, 0] = False
        m[0, 1] = True
        widget.on_selection_change(m, SomView.SelectionNewGroup)
        out = self.get_output(widget.Outputs.selected_data)
        np.testing.assert_equal(out.ids, self.iris.ids[:50])

        out = self.get_output(widget.Outputs.annotated_data)
        np.testing.assert_equal(
            out.get_column_view(ANNOTATED_DATA_FEATURE_NAME)[0],
            [0] * 30 + [1] * 20 + [2] * 100)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(widget.Outputs.selected_data))
        self.assertIsNone(self.get_output(widget.Outputs.annotated_data))

    def test_invalidated(self):
        heart = Table("heart_disease")
        self.widget._recompute_som = Mock()

        # New data - replot
        self.send_signal(self.widget.Inputs.data, heart)
        self.widget._recompute_som.assert_called_once()

        # Same data - no replot
        self.widget._recompute_som.reset_mock()
        self.send_signal(self.widget.Inputs.data, heart)
        self.widget._recompute_som.assert_not_called()

        # Same data.X - no replot
        domain = heart.domain
        domain = Domain(domain.attributes, metas=domain.class_vars)
        heart_with_metas = self.iris.transform(domain)
        self.widget._recompute_som.reset_mock()
        self.send_signal(self.widget.Inputs.data, heart_with_metas)
        self.widget._recompute_som.assert_not_called()

        # Different data, same set of cont. vars - no replot
        attrs = [a for a in heart.domain.attributes if a.is_continuous]
        domain = Domain(attrs)
        heart_with_cont_features = self.iris.transform(domain)
        self.widget._recompute_som.reset_mock()
        self.send_signal(self.widget.Inputs.data, heart_with_cont_features)
        self.widget._recompute_som.assert_not_called()

        # Different data.X - replot
        domain = Domain(heart.domain.attributes[:5])
        heart_with_less_features = heart.transform(domain)
        self.widget._recompute_som.reset_mock()
        self.send_signal(self.widget.Inputs.data, heart_with_less_features)
        self.widget._recompute_som.assert_called_once()


if __name__ == "__main__":
    unittest.main()
