# pylint: disable=protected-access

import unittest
from unittest.mock import Mock, patch

import numpy as np
import scipy.sparse as sp

from Orange.data import Table, Domain
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_FEATURE_NAME
from Orange.widgets.unsupervised.owsom import OWSOM, SomView


class TestOWSOM(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSOM)  #: OWSOM
        self.iris = Table("iris")

    def tearDown(self):
        self.widget.onDeleteWidget()

    def _patch_recompute_som(self):
        class MockSom:
            def __init__(self, n):
                self.n = n

            def winners(self, _):
                w = np.zeros((self.n, 2), dtype=int)
                w[self.n // 5:] = [0, 1]
                w[self.n // 3:] = [1, 2]
                return w

        def recompute():
            if not self.widget.data:
                return
            self.widget._assign_instances(MockSom(len(self.widget.data)))
            self.widget._redraw()

        self.widget._recompute_som = recompute

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

    def test_missing_all_data(self):
        widget = self.widget

        self.send_signal(widget.Inputs.data, Table("heart_disease"))
        self.assertTrue(widget.Warning.ignoring_disc_variables.is_shown())

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
        self.iris.X[:50, 0] = np.nan

        self.send_signal(widget.Inputs.data, self.iris)
        self.assertFalse(widget.Error.no_defined_rows.is_shown())
        np.testing.assert_almost_equal(
            widget.data.Y.flatten(), [1] * 50 + [2] * 50)
        self.assertEqual(widget.cont_x.shape, (100, 4))

    def test_sparse_data(self):
        widget = self.widget
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
        self._patch_recompute_som()
        self.send_signal(widget.Inputs.data, self.iris)

        widget.size_x = widget.size_y = 100
        widget.manual_dimension = False
        widget.controls.manual_dimension.toggled.emit(False)
        self.assertLess(widget.size_x, 100)
        self.assertLess(widget.size_y, 100)

        widget.size_x = 9
        widget.size_y = 7
        widget.manual_dimension = True
        widget.controls.manual_dimension.toggled.emit(True)
        self.assertEqual(widget.size_x, 10)
        self.assertEqual(widget.size_y, 5)

    def test_redraw_grid(self):
        widget = self.widget
        widget.size_x = 8
        widget.size_y = 5

        widget.hexagonal = False
        widget.controls.hexagonal.activated[int].emit(0)
        self.assertEqual(widget.grid_cells.shape, (5, 8))
        self.assertEqual(
            sum(c is not None for c in widget.grid_cells.ravel()), 40)

        widget.hexagonal = True
        widget.controls.hexagonal.activated[int].emit(1)
        self.assertEqual(widget.grid_cells.shape, (5, 8))
        self.assertEqual(
            sum(c is not None for c in widget.grid_cells.ravel()), 38)
        self.assertIsNone(widget.grid_cells[1, 7])
        self.assertIsNone(widget.grid_cells[3, 7])

        widget.hexagonal = False
        widget.controls.hexagonal.activated[int].emit(0)
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
        self.assertEqual(
            [(c.red(), c.green(), c.blue()) for c in widget.colors],
            [tuple(c) for c in gender.colors])
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
        self.assertEqual(len(widget.colors), len(widget.thresholds) + 1)
        widget._redraw.assert_called()

    def test_cell_sizes(self):
        widget = self.widget
        self._patch_recompute_som()
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

    def test_same_color_same_size(self):
        widget = self.widget
        self._patch_recompute_som()

        widget.size_by_instances = False
        widget.hexagonal = True

        self.send_signal(widget.Inputs.data, self.iris)
        widget.attr_color = None
        widget.controls.attr_color.activated[int].emit(0)
        a = widget.elements.childItems()[0]
        for e, t in zip(widget.elements.childItems(),
                        [(0, 0, 0.8),
                         (0.5, np.sqrt(3) / 2, 0.8),
                         (1, np.sqrt(3), 0.8)]):
            r = e.rect()
            w = r.width()
            np.testing.assert_almost_equal((r.x() + w / 2, r.y() + w / 2, w), t)
            self.assertEqual(e.brush().color().getRgb(),
                             a.brush().color().getRgb())
            self.assertEqual(e.pen().color().getRgb(),
                             a.pen().color().getRgb())

        widget.hexagonal = False
        widget.controls.hexagonal.activated[int].emit(0)
        a = widget.elements.childItems()[0]
        for e, t in zip(widget.elements.childItems(),
                        [(0, 0, 0.8), (0, 1, 0.8), (1, 2, 0.8)]):
            r = e.rect()
            w = r.width()
            np.testing.assert_almost_equal((r.x() + w / 2, r.y() + w / 2, w), t)
            self.assertEqual(e.brush().color().getRgb(),
                             a.brush().color().getRgb())
            self.assertEqual(e.pen().color().getRgb(),
                             a.pen().color().getRgb())


    def test_diff_color_same_size(self):
        widget = self.widget
        self._patch_recompute_som()

        widget.size_by_instances = False
        widget.hexagonal = True

        self.send_signal(widget.Inputs.data, self.iris)
        a, b, c = widget.elements.childItems()
        for e, t in zip(widget.elements.childItems(),
                        [(0, 0, 0.8),
                         (0.5, np.sqrt(3) / 2, 0.8),
                         (1, np.sqrt(3), 0.8)]):
            r = e.rect()
            w = r.width()
            np.testing.assert_almost_equal((r.x() + w / 2, r.y() + w / 2, w), t)
        self.assertEqual(a.brush().color().getRgb(),
                         b.brush().color().getRgb())
        self.assertEqual(a.pen().color().getRgb(),
                         b.pen().color().getRgb())
        self.assertNotEqual(a.brush().color().getRgb(),
                            c.brush().color().getRgb())
        self.assertNotEqual(a.pen().color().getRgb(),
                            c.pen().color().getRgb())

        widget.hexagonal = False
        widget.controls.hexagonal.activated[int].emit(0)
        a, b, c = widget.elements.childItems()
        for e, t in zip(widget.elements.childItems(),
                        [(0, 0, 0.8), (0, 1, 0.8), (1, 2, 0.8)]):
            r = e.rect()
            w = r.width()
            np.testing.assert_almost_equal((r.x() + w / 2, r.y() + w / 2, w), t)
        self.assertEqual(a.brush().color().getRgb(),
                         b.brush().color().getRgb())
        self.assertEqual(a.pen().color().getRgb(),
                         b.pen().color().getRgb())
        self.assertNotEqual(a.brush().color().getRgb(),
                            c.brush().color().getRgb())
        self.assertNotEqual(a.pen().color().getRgb(),
                            c.pen().color().getRgb())


    def test_same_color_diff_size(self):
        widget = self.widget
        self._patch_recompute_som()

        widget.size_by_instances = True
        widget.hexagonal = True

        self.send_signal(widget.Inputs.data, self.iris)
        widget.attr_color = None
        widget.controls.attr_color.activated[int].emit(0)
        a = widget.elements.childItems()[0]
        for e, t in zip(widget.elements.childItems(),
                        [(0, 0, 0.8 * (30 / 100)),
                         (0.5, np.sqrt(3) / 2, 0.8 * (20 / 100)),
                         (1, np.sqrt(3), 0.8)]):
            r = e.rect()
            w = r.width()
            np.testing.assert_almost_equal((r.x() + w / 2, r.y() + w / 2, w), t)
            self.assertEqual(e.brush().color().getRgb(),
                             a.brush().color().getRgb())
            self.assertEqual(e.pen().color().getRgb(),
                             a.pen().color().getRgb())

        widget.hexagonal = False
        widget.controls.hexagonal.activated[int].emit(0)
        a = widget.elements.childItems()[0]
        for e, t in zip(widget.elements.childItems(),
                        [(0, 0, 0.8 * 30 / 100),
                         (0, 1, 0.8 * 20 / 100),
                         (1, 2, 0.8)]):
            r = e.rect()
            w = r.width()
            np.testing.assert_almost_equal((r.x() + w / 2, r.y() + w / 2, w), t)
            self.assertEqual(e.brush().color().getRgb(),
                             a.brush().color().getRgb())
            self.assertEqual(e.pen().color().getRgb(),
                             a.pen().color().getRgb())


    def test_diff_color_diff_size(self):
        widget = self.widget
        self._patch_recompute_som()

        widget.size_by_instances = True
        widget.hexagonal = True

        self.send_signal(widget.Inputs.data, self.iris)
        a, b, c = widget.elements.childItems()
        for e, t in zip(widget.elements.childItems(),
                        [(0, 0, 0.8 * 30 / 100),
                         (0.5, np.sqrt(3) / 2, 0.8 * 20 / 100),
                         (1, np.sqrt(3), 0.8)]):
            r = e.rect()
            w = r.width()
            np.testing.assert_almost_equal((r.x() + w / 2, r.y() + w / 2, w), t)
        self.assertEqual(a.brush().color().getRgb(),
                         b.brush().color().getRgb())
        self.assertEqual(a.pen().color().getRgb(),
                         b.pen().color().getRgb())
        self.assertNotEqual(a.brush().color().getRgb(),
                            c.brush().color().getRgb())
        self.assertNotEqual(a.pen().color().getRgb(),
                            c.pen().color().getRgb())

        widget.hexagonal = False
        widget.controls.hexagonal.activated[int].emit(0)
        a, b, c = widget.elements.childItems()
        for e, t in zip(widget.elements.childItems(),
                        [(0, 0, 0.8 * 30 / 100),
                         (0, 1, 0.8 * 20 / 100),
                         (1, 2, 0.8)]):
            r = e.rect()
            w = r.width()
            np.testing.assert_almost_equal((r.x() + w / 2, r.y() + w / 2, w), t)
        self.assertEqual(a.brush().color().getRgb(),
                         b.brush().color().getRgb())
        self.assertEqual(a.pen().color().getRgb(),
                         b.pen().color().getRgb())
        self.assertNotEqual(a.brush().color().getRgb(),
                            c.brush().color().getRgb())
        self.assertNotEqual(a.pen().color().getRgb(),
                            c.pen().color().getRgb())

    def test_pie_charts(self):
        widget = self.widget
        self._patch_recompute_som()

        widget.size_by_instances = True
        widget.hexagonal = True
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

        widget.hexagonal = False
        widget.controls.hexagonal.activated[int].emit(0)
        for e, (x, y, r) in zip(widget.elements.childItems(),
                                [(0, 0, 0.8 * 30 / 100),
                                 (0, 1, 0.8 * 20 / 100),
                                 (1, 2, 0.8)]):
            self.assertEqual(e.x(), x)
            self.assertEqual(e.y(), y)
            self.assertEqual(e.r, r / 2)

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
        new_table = table.transform(new_domain)
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
        col = widget.data.get_column_view("gender")[0]
        col[:5] = np.nan
        col = col.copy()
        col[:5] = 2
        np.testing.assert_equal(widget._get_color_column(), col)

    def test_colored_circles_with_missing_values(self):
        self._patch_recompute_som()
        self.iris.get_column_view("iris")[0][:5] = np.nan
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertTrue(self.widget.Warning.missing_colors.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.missing_colors.is_shown())

    def test_send_report(self):
        self.widget.send_report()

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.send_report()

    def test_on_selection_change(self):
        widget = self.widget

        self._patch_recompute_som()
        self.send_signal(self.widget.Inputs.data, self.iris)

        widget.redraw_selection = Mock()
        widget.update_output = Mock()

        widget.on_selection_change({(0, 0)})
        self.assertEqual(widget.selection, {(0, 0)})
        widget.redraw_selection.assert_called_once()
        widget.update_output.assert_called_once()

        widget.on_selection_change({(0, 1)})
        self.assertEqual(widget.selection, {(0, 1)})

        widget.on_selection_change({(0, 0)}, SomView.SelectionAdd)
        self.assertEqual(widget.selection, {(0, 1), (0, 0)})

        widget.on_selection_change({(0, 0)}, SomView.SelectionRemove)
        self.assertEqual(widget.selection, {(0, 1)})

        widget.on_selection_change({(0, 0)}, SomView.SelectionToggle)
        self.assertEqual(widget.selection, {(0, 1), (0, 0)})

        widget.on_selection_change({(0, 0), (2, 2)}, SomView.SelectionToggle)
        self.assertEqual(widget.selection, {(0, 1), (2, 2)})

    def test_output(self):
        widget = self.widget

        self._patch_recompute_som()
        self.send_signal(self.widget.Inputs.data, self.iris)

        self.assertIsNone(self.get_output(widget.Outputs.selected_data))
        out = self.get_output(widget.Outputs.annotated_data)
        self.assertEqual(len(out), 150)
        self.assertTrue(
            np.all(out.get_column_view(ANNOTATED_DATA_FEATURE_NAME)[0] == 0))

        widget.on_selection_change({(0, 0)})
        out = self.get_output(widget.Outputs.selected_data)
        np.testing.assert_equal(out.ids, self.iris.ids[:30])

        out = self.get_output(widget.Outputs.annotated_data)
        np.testing.assert_equal(
            out.get_column_view(ANNOTATED_DATA_FEATURE_NAME)[0],
            [1] * 30 + [0] * 120)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(widget.Outputs.selected_data))
        self.assertIsNone(self.get_output(widget.Outputs.annotated_data))

if __name__ == "__main__":
    unittest.main()
