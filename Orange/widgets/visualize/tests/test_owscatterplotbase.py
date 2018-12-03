# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from unittest.mock import patch, Mock
import numpy as np

from AnyQt.QtCore import QRectF, Qt
from AnyQt.QtGui import QColor

from pyqtgraph import mkPen

from Orange.widgets.settings import SettingProvider
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.colorpalette import ColorPaletteGenerator, \
    ContinuousPaletteGenerator, NAN_GREY
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase, \
    ScatterPlotItem, SELECTION_WIDTH
from Orange.widgets.widget import OWWidget


class MockWidget(OWWidget):
    name = "Mock"

    get_coordinates_data = Mock(return_value=(None, None))
    get_size_data = Mock(return_value=None)
    get_shape_data = Mock(return_value=None)
    get_color_data = Mock(return_value=None)
    get_label_data = Mock(return_value=None)
    get_color_labels = Mock(return_value=None)
    get_shape_labels = Mock(return_value=None)
    get_subset_mask = Mock(return_value=None)
    get_tooltip = Mock(return_value="")

    is_continuous_color = Mock(return_value=False)
    can_draw_density = Mock(return_value=True)
    combined_legend = Mock(return_value=False)
    selection_changed = Mock(return_value=None)

    GRAPH_CLASS = OWScatterPlotBase
    graph = SettingProvider(OWScatterPlotBase)

    def get_palette(self):
        if self.is_continuous_color():
            return ContinuousPaletteGenerator(Qt.white, Qt.black, False)
        else:
            return ColorPaletteGenerator(12)


class TestOWScatterPlotBase(WidgetTest):
    def setUp(self):
        self.master = MockWidget()
        self.graph = OWScatterPlotBase(self.master)

        self.xy = (np.arange(10, dtype=float), np.arange(10, dtype=float))
        self.master.get_coordinates_data = lambda: self.xy

    # pylint: disable=keyword-arg-before-vararg
    def setRange(self, rect=None, *_, **__):
        if isinstance(rect, QRectF):
            self.last_setRange = [[rect.left(), rect.right()],
                                  [rect.top(), rect.bottom()]]

    def test_update_coordinates_no_data(self):
        self.xy = None, None
        self.graph.reset_graph()
        self.assertIsNone(self.graph.scatterplot_item)
        self.assertIsNone(self.graph.scatterplot_item_sel)

        self.xy = [], []
        self.graph.reset_graph()
        self.assertIsNone(self.graph.scatterplot_item)
        self.assertIsNone(self.graph.scatterplot_item_sel)

    def test_update_coordinates(self):
        graph = self.graph
        xy = self.xy = (np.array([1, 2]), np.array([3, 4]))
        graph.reset_graph()

        scatterplot_item = graph.scatterplot_item
        scatterplot_item_sel = graph.scatterplot_item_sel
        data = scatterplot_item.data

        np.testing.assert_almost_equal(scatterplot_item.getData(), xy)
        np.testing.assert_almost_equal(scatterplot_item_sel.getData(), xy)
        scatterplot_item.setSize([5, 6])
        scatterplot_item.setSymbol([7, 8])
        scatterplot_item.setPen([mkPen(9), mkPen(10)])
        scatterplot_item.setBrush([11, 12])
        data["data"] = np.array([13, 14])

        xy[0][0] = 0
        graph.update_coordinates()
        np.testing.assert_almost_equal(graph.scatterplot_item.getData(), xy)
        np.testing.assert_almost_equal(graph.scatterplot_item_sel.getData(), xy)

        # Graph updates coordinates instead of creating new items
        self.assertIs(scatterplot_item, graph.scatterplot_item)
        self.assertIs(scatterplot_item_sel, graph.scatterplot_item_sel)
        np.testing.assert_almost_equal(data["size"], [5, 6])
        np.testing.assert_almost_equal(data["symbol"], [7, 8])
        self.assertEqual(data["pen"][0], mkPen(9))
        self.assertEqual(data["pen"][1], mkPen(10))
        np.testing.assert_almost_equal(data["brush"], [11, 12])
        np.testing.assert_almost_equal(data["data"], [13, 14])

    def test_update_coordinates_and_labels(self):
        graph = self.graph
        xy = self.xy = (np.array([1, 2]), np.array([3, 4]))
        self.master.get_label_data = lambda: ["a", "b"]
        graph.reset_graph()
        self.assertEqual(graph.labels[0].pos().x(), 1)
        xy[0][0] = 0
        graph.update_coordinates()
        self.assertEqual(graph.labels[0].pos().x(), 0)

    def test_update_coordinates_and_density(self):
        graph = self.graph
        xy = self.xy = (np.array([1, 2]), np.array([3, 4]))
        self.master.get_label_data = lambda: ["a", "b"]
        graph.reset_graph()
        self.assertEqual(graph.labels[0].pos().x(), 1)
        xy[0][0] = 0
        graph.update_density = Mock()
        graph.update_coordinates()
        graph.update_density.assert_called_with()

    def test_update_coordinates_reset_view(self):
        graph = self.graph
        graph.view_box.setRange = self.setRange
        xy = self.xy = (np.array([2, 1]), np.array([3, 10]))
        self.master.get_label_data = lambda: ["a", "b"]
        graph.reset_graph()
        self.assertEqual(self.last_setRange, [[1, 2], [3, 10]])

        xy[0][1] = 0
        graph.update_coordinates()
        self.assertEqual(self.last_setRange, [[0, 2], [3, 10]])

    def test_reset_graph_no_data(self):
        self.xy = (None, None)
        self.graph.scatterplot_item = ScatterPlotItem([1, 2], [3, 4])
        self.graph.reset_graph()
        self.assertIsNone(self.graph.scatterplot_item)
        self.assertIsNone(self.graph.scatterplot_item_sel)

    def test_update_coordinates_indices(self):
        graph = self.graph
        self.xy = (np.array([2, 1]), np.array([3, 10]))
        graph.reset_graph()
        np.testing.assert_almost_equal(
            graph.scatterplot_item.data["data"], [0, 1])

    def test_sampling(self):
        graph = self.graph
        master = self.master

        # Enable sampling before getting the data
        graph.set_sample_size(3)
        xy = self.xy = (np.arange(10, dtype=float),
                        np.arange(0, 30, 3, dtype=float))
        d = np.arange(10, dtype=float)
        master.get_size_data = lambda: d
        master.get_shape_data = lambda: d
        master.get_color_data = lambda: d
        master.get_label_data = lambda: \
            np.array([str(x) for x in d], dtype=object)
        graph.reset_graph()
        self.process_events(until=lambda: not (
            self.graph.timer is not None and self.graph.timer.isActive()))

        # Check proper sampling
        scatterplot_item = graph.scatterplot_item
        x, y = scatterplot_item.getData()
        self.assertEqual(len(x), 3)
        self.assertNotEqual(x[0], x[1])
        self.assertNotEqual(x[0], x[2])
        self.assertNotEqual(x[1], x[2])
        np.testing.assert_almost_equal(3 * x, y)

        data = scatterplot_item.data
        s0, s1, s2 = data["size"] - graph.MinShapeSize
        np.testing.assert_almost_equal(
            (s2 - s1) / (s1 - s0),
            (x[2] - x[1]) / (x[1] - x[0]))
        self.assertEqual(
            list(data["symbol"]),
            [graph.CurveSymbols[int(xi)] for xi in x])
        self.assertEqual(
            [pen.color().hue() for pen in data["pen"]],
            [graph.palette[xi].hue() for xi in x])
        self.assertEqual(
            [label.textItem.toPlainText() for label in graph.labels],
            [str(xi) for xi in x])

        # Check that sample is extended when sample size is changed
        graph.set_sample_size(4)
        self.process_events(until=lambda: not (
            self.graph.timer is not None and self.graph.timer.isActive()))
        scatterplot_item = graph.scatterplot_item
        x, y = scatterplot_item.getData()
        data = scatterplot_item.data
        s0, s1, s2, s3 = data["size"] - graph.MinShapeSize
        np.testing.assert_almost_equal(
            (s2 - s1) / (s1 - s0),
            (x[2] - x[1]) / (x[1] - x[0]))
        np.testing.assert_almost_equal(
            (s2 - s1) / (s1 - s3),
            (x[2] - x[1]) / (x[1] - x[3]))
        self.assertEqual(
            list(data["symbol"]),
            [graph.CurveSymbols[int(xi)] for xi in x])
        self.assertEqual(
            [pen.color().hue() for pen in data["pen"]],
            [graph.palette[xi].hue() for xi in x])
        self.assertEqual(
            [label.textItem.toPlainText() for label in graph.labels],
            [str(xi) for xi in x])

        # Disable sampling
        graph.set_sample_size(None)
        scatterplot_item = graph.scatterplot_item
        x, y = scatterplot_item.getData()
        data = scatterplot_item.data
        np.testing.assert_almost_equal(x, xy[0])
        np.testing.assert_almost_equal(y, xy[1])
        self.assertEqual(
            list(data["symbol"]),
            [graph.CurveSymbols[int(xi)] for xi in d])
        self.assertEqual(
            [pen.color().hue() for pen in data["pen"]],
            [graph.palette[xi].hue() for xi in d])
        self.assertEqual(
            [label.textItem.toPlainText() for label in graph.labels],
            [str(xi) for xi in d])

        # Enable sampling when data is already present and not sampled
        graph.set_sample_size(3)
        self.process_events(until=lambda: not (
            self.graph.timer is not None and self.graph.timer.isActive()))
        scatterplot_item = graph.scatterplot_item
        x, y = scatterplot_item.getData()
        data = scatterplot_item.data
        s0, s1, s2 = data["size"] - graph.MinShapeSize
        np.testing.assert_almost_equal(
            (s2 - s1) / (s1 - s0),
            (x[2] - x[1]) / (x[1] - x[0]))
        self.assertEqual(
            list(data["symbol"]),
            [graph.CurveSymbols[int(xi)] for xi in x])
        self.assertEqual(
            [pen.color().hue() for pen in data["pen"]],
            [graph.palette[xi].hue() for xi in x])
        self.assertEqual(
            [label.textItem.toPlainText() for label in graph.labels],
            [str(xi) for xi in x])

        # Update data when data is present and sampling is enabled
        xy[0][:] = np.arange(9, -1, -1, dtype=float)
        d = xy[0]
        graph.update_coordinates()
        x1, _ = scatterplot_item.getData()
        np.testing.assert_almost_equal(9 - x, x1)
        graph.update_sizes()
        data = scatterplot_item.data
        s0, s1, s2 = data["size"] - graph.MinShapeSize
        np.testing.assert_almost_equal(
            (s2 - s1) / (s1 - s0),
            (x[2] - x[1]) / (x[1] - x[0]))

        # Reset graph when data is present and sampling is enabled
        self.xy = (np.arange(100, 105, dtype=float),
                   np.arange(100, 105, dtype=float))
        d = self.xy[0] - 100
        graph.reset_graph()
        self.process_events(until=lambda: not (
            self.graph.timer is not None and self.graph.timer.isActive()))
        scatterplot_item = graph.scatterplot_item
        x, y = scatterplot_item.getData()
        self.assertEqual(len(x), 3)
        self.assertTrue(np.all(x > 99))
        data = scatterplot_item.data
        s0, s1, s2 = data["size"] - graph.MinShapeSize
        np.testing.assert_almost_equal(
            (s2 - s1) / (s1 - s0),
            (x[2] - x[1]) / (x[1] - x[0]))

        # Don't sample when unnecessary
        self.xy = (np.arange(100, dtype=float), ) * 2
        d = None
        delattr(master, "get_label_data")
        graph.reset_graph()
        graph.set_sample_size(120)
        scatterplot_item = graph.scatterplot_item
        x, y = scatterplot_item.getData()
        np.testing.assert_almost_equal(x, np.arange(100))

    def test_sampling_keeps_selection(self):
        graph = self.graph

        self.xy = (np.arange(100, dtype=float),
                   np.arange(100, dtype=float))
        graph.reset_graph()
        graph.select_by_indices(np.arange(1, 100, 2))
        graph.set_sample_size(30)
        np.testing.assert_almost_equal(graph.selection, np.arange(100) % 2)
        graph.set_sample_size(None)
        np.testing.assert_almost_equal(graph.selection, np.arange(100) % 2)

    base = "Orange.widgets.visualize.owscatterplotgraph.OWScatterPlotBase."

    @patch(base + "update_sizes")
    @patch(base + "update_colors")
    @patch(base + "update_selection_colors")
    @patch(base + "update_shapes")
    @patch(base + "update_labels")
    def test_reset_calls_all_updates_and_update_doesnt(self, *mocks):
        master = MockWidget()
        graph = OWScatterPlotBase(master)
        for mock in mocks:
            mock.assert_not_called()

        graph.reset_graph()
        for mock in mocks:
            mock.assert_called_with()
            mock.reset_mock()

        graph.update_coordinates()
        for mock in mocks:
            mock.assert_not_called()

    def test_jittering(self):
        graph = self.graph
        graph.jitter_size = 10
        graph.reset_graph()
        scatterplot_item = graph.scatterplot_item
        x, y = scatterplot_item.getData()
        a10 = np.arange(10)
        self.assertTrue(np.any(np.nonzero(a10 - x)))
        self.assertTrue(np.any(np.nonzero(a10 - y)))
        np.testing.assert_array_less(a10 - x, 1)
        np.testing.assert_array_less(a10 - y, 1)

        graph.jitter_size = 0
        graph.update_coordinates()
        scatterplot_item = graph.scatterplot_item
        x, y = scatterplot_item.getData()
        np.testing.assert_equal(a10, x)

    def test_size_normalization(self):
        graph = self.graph

        self.master.get_size_data = lambda: d
        d = np.arange(10, dtype=float)

        graph.reset_graph()
        scatterplot_item = graph.scatterplot_item
        size = scatterplot_item.data["size"]
        diffs = [round(y - x, 2) for x, y in zip(size, size[1:])]
        self.assertEqual(len(set(diffs)), 1)
        self.assertGreater(diffs[0], 0)

        d = np.arange(10, 20, dtype=float)
        graph.update_sizes()
        self.assertIs(scatterplot_item, graph.scatterplot_item)
        size = scatterplot_item.data["size"]
        diffs2 = [round(y - x, 2) for x, y in zip(size, size[1:])]
        self.assertEqual(diffs, diffs2)

    def test_size_with_nans(self):
        graph = self.graph

        self.master.get_size_data = lambda: d
        d = np.arange(10, dtype=float)

        graph.reset_graph()
        scatterplot_item = graph.scatterplot_item
        sizes = scatterplot_item.data["size"]

        d[4] = np.nan
        graph.update_sizes()
        self.process_events(until=lambda: not (
            self.graph.timer is not None and self.graph.timer.isActive()))
        sizes2 = scatterplot_item.data["size"]

        self.assertEqual(sizes[1] - sizes[0], sizes2[1] - sizes2[0])
        self.assertLess(sizes2[4], self.graph.MinShapeSize)

        d[:] = np.nan
        graph.update_sizes()
        sizes3 = scatterplot_item.data["size"]
        np.testing.assert_almost_equal(sizes, sizes3)

    def test_sizes_all_same_or_nan(self):
        graph = self.graph

        self.master.get_size_data = lambda: d
        d = np.full((10, ), 3.0)

        graph.reset_graph()
        scatterplot_item = graph.scatterplot_item
        sizes = scatterplot_item.data["size"]
        self.assertEqual(len(set(sizes)), 1)
        self.assertGreater(sizes[0], self.graph.MinShapeSize)

        d = None
        graph.update_sizes()
        scatterplot_item = graph.scatterplot_item
        sizes2 = scatterplot_item.data["size"]
        np.testing.assert_almost_equal(sizes, sizes2)

    def test_sizes_point_width_is_linear(self):
        graph = self.graph

        self.master.get_size_data = lambda: d
        d = np.arange(10, dtype=float)

        graph.point_width = 1
        graph.reset_graph()
        sizes1 = graph.scatterplot_item.data["size"]

        graph.point_width = 2
        graph.update_sizes()
        sizes2 = graph.scatterplot_item.data["size"]

        graph.point_width = 3
        graph.update_sizes()
        sizes3 = graph.scatterplot_item.data["size"]

        np.testing.assert_almost_equal(2 * (sizes2 - sizes1), sizes3 - sizes1)

    def test_sizes_custom_imputation(self):

        def impute_max(size_data):
            size_data[np.isnan(size_data)] = np.nanmax(size_data)

        graph = self.graph

        self.master.get_size_data = lambda: d
        self.master.impute_sizes = impute_max
        d = np.arange(10, dtype=float)
        d[4] = np.nan
        graph.reset_graph()
        sizes = graph.scatterplot_item.data["size"]
        self.assertAlmostEqual(sizes[4], sizes[9])

    def test_sizes_selection(self):
        graph = self.graph
        graph.get_size = lambda: np.arange(10, dtype=float)
        graph.reset_graph()
        np.testing.assert_almost_equal(
            graph.scatterplot_item_sel.data["size"]
            - graph.scatterplot_item.data["size"],
            SELECTION_WIDTH)

    def test_colors_discrete(self):
        self.master.is_continuous_color = lambda: False
        palette = self.master.get_palette()
        graph = self.graph

        self.master.get_color_data = lambda: d
        d = np.arange(10, dtype=float) % 2

        graph.reset_graph()
        self.assertTrue(
            all(pen.color().hue() is palette[i % 2].hue()
                for i, pen in enumerate(graph.scatterplot_item.data["pen"])))
        self.assertTrue(
            all(pen.color().hue() is palette[i % 2].hue()
                for i, pen in enumerate(graph.scatterplot_item.data["brush"])))

    def test_colors_discrete_nan(self):
        self.master.is_continuous_color = lambda: False
        palette = self.master.get_palette()
        graph = self.graph

        d = np.arange(10, dtype=float) % 2
        d[4] = np.nan
        self.master.get_color_data = lambda: d
        graph.reset_graph()
        pens = graph.scatterplot_item.data["pen"]
        brushes = graph.scatterplot_item.data["brush"]
        self.assertEqual(pens[0].color().hue(), palette[0].hue())
        self.assertEqual(pens[1].color().hue(), palette[1].hue())
        self.assertEqual(brushes[0].color().hue(), palette[0].hue())
        self.assertEqual(brushes[1].color().hue(), palette[1].hue())
        self.assertEqual(pens[4].color().hue(), QColor(128, 128, 128).hue())
        self.assertEqual(brushes[4].color().hue(), QColor(128, 128, 128).hue())

    def test_colors_continuous(self):
        self.master.is_continuous_color = lambda: True
        graph = self.graph

        d = np.arange(10, dtype=float)
        self.master.get_color_data = lambda: d
        graph.reset_graph()  # I don't have a good test ... just don't crash

        d[4] = np.nan
        graph.update_colors()  # Ditto

    def test_colors_continuous_nan(self):
        self.master.is_continuous_color = lambda: True
        graph = self.graph

        d = np.arange(10, dtype=float) % 2
        d[4] = np.nan
        self.master.get_color_data = lambda: d
        graph.reset_graph()
        pens = graph.scatterplot_item.data["pen"]
        brushes = graph.scatterplot_item.data["brush"]
        nan_color = QColor(*NAN_GREY)
        self.assertEqual(pens[4].color().hue(), nan_color.hue())
        self.assertEqual(brushes[4].color().hue(), nan_color.hue())

    def test_colors_subset(self):
        def run_tests():
            self.master.get_subset_mask = lambda: None

            graph.alpha_value = 42
            graph.reset_graph()
            brushes = graph.scatterplot_item.data["brush"]
            self.assertEqual(brushes[0].color().alpha(), 42)
            self.assertEqual(brushes[1].color().alpha(), 42)
            self.assertEqual(brushes[4].color().alpha(), 42)

            graph.alpha_value = 123
            graph.update_colors()
            brushes = graph.scatterplot_item.data["brush"]
            self.assertEqual(brushes[0].color().alpha(), 123)
            self.assertEqual(brushes[1].color().alpha(), 123)
            self.assertEqual(brushes[4].color().alpha(), 123)

            self.master.get_subset_mask = lambda: np.arange(10) >= 5
            graph.update_colors()
            brushes = graph.scatterplot_item.data["brush"]
            self.assertEqual(brushes[0].color().alpha(), 0)
            self.assertEqual(brushes[1].color().alpha(), 0)
            self.assertEqual(brushes[4].color().alpha(), 0)
            self.assertEqual(brushes[5].color().alpha(), 255)
            self.assertEqual(brushes[6].color().alpha(), 255)
            self.assertEqual(brushes[7].color().alpha(), 255)

        graph = self.graph

        self.master.get_color_data = lambda: None
        self.master.is_continuous_color = lambda: True
        graph.reset_graph()
        run_tests()

        self.master.is_continuous_color = lambda: False
        graph.reset_graph()
        run_tests()

        d = np.arange(10, dtype=float) % 2
        d[4:6] = np.nan
        self.master.get_color_data = lambda: d

        self.master.is_continuous_color = lambda: True
        graph.reset_graph()
        run_tests()

        self.master.is_continuous_color = lambda: False
        graph.reset_graph()
        run_tests()

    def test_colors_none(self):
        graph = self.graph
        graph.reset_graph()
        hue = QColor(128, 128, 128).hue()

        data = graph.scatterplot_item.data
        self.assertTrue(all(pen.color().hue() == hue for pen in data["pen"]))
        self.assertTrue(all(pen.color().hue() == hue for pen in data["brush"]))

        self.master.get_subset_mask = lambda: np.arange(10) < 5
        graph.update_colors()
        data = graph.scatterplot_item.data
        self.assertTrue(all(pen.color().hue() == hue for pen in data["pen"]))
        self.assertTrue(all(pen.color().hue() == hue for pen in data["brush"]))

    def test_colors_update_legend_and_density(self):
        graph = self.graph
        graph.update_legends = Mock()
        graph.update_density = Mock()
        graph.reset_graph()
        graph.update_legends.assert_called_with()
        graph.update_density.assert_called_with()

        graph.update_legends.reset_mock()
        graph.update_density.reset_mock()

        graph.update_coordinates()
        graph.update_legends.assert_not_called()

        graph.update_colors()
        graph.update_legends.assert_called_with()
        graph.update_density.assert_called_with()

    def test_selection_colors(self):
        graph = self.graph
        graph.reset_graph()
        data = graph.scatterplot_item_sel.data

        # One group
        graph.select_by_indices(np.array([0, 1, 2, 3]))
        graph.update_selection_colors()
        pens = data["pen"]
        for i in range(4):
            self.assertNotEqual(pens[i].style(), Qt.NoPen)
        for i in range(4, 10):
            self.assertEqual(pens[i].style(), Qt.NoPen)

        # Two groups
        with patch("AnyQt.QtWidgets.QApplication.keyboardModifiers",
                   lambda: Qt.ShiftModifier):
            graph.select_by_indices(np.array([4, 5, 6]))

        graph.update_selection_colors()
        pens = data["pen"]
        for i in range(7):
            self.assertNotEqual(pens[i].style(), Qt.NoPen)
        for i in range(7, 10):
            self.assertEqual(pens[i].style(), Qt.NoPen)
        self.assertEqual(len({pen.color().hue() for pen in pens[:4]}), 1)
        self.assertEqual(len({pen.color().hue() for pen in pens[4:7]}), 1)
        color1 = pens[3].color().hue()
        color2 = pens[4].color().hue()
        self.assertNotEqual(color1, color2)

        # Two groups + sampling
        graph.set_sample_size(7)
        x = graph.scatterplot_item.getData()[0]
        pens = graph.scatterplot_item_sel.data["pen"]
        for xi, pen in zip(x, pens):
            if xi < 4:
                self.assertEqual(pen.color().hue(), color1)
            elif xi < 7:
                self.assertEqual(pen.color().hue(), color2)
            else:
                self.assertEqual(pen.style(), Qt.NoPen)

    def test_density(self):
        graph = self.graph
        density = object()
        with patch("Orange.widgets.utils.classdensity.class_density_image",
                   return_value=density):
            graph.reset_graph()
            self.assertIsNone(graph.density_img)

            graph.plot_widget.addItem = Mock()
            graph.plot_widget.removeItem = Mock()

            graph.class_density = True
            graph.update_colors()
            self.assertIsNone(graph.density_img)

            d = np.ones((10, ), dtype=float)
            self.master.get_color_data = lambda: d
            graph.update_colors()
            self.assertIsNone(graph.density_img)

            d = np.arange(10) % 2
            graph.update_colors()
            self.assertIs(graph.density_img, density)
            self.assertIs(graph.plot_widget.addItem.call_args[0][0], density)

            graph.class_density = False
            graph.update_colors()
            self.assertIsNone(graph.density_img)
            self.assertIs(graph.plot_widget.removeItem.call_args[0][0], density)

            graph.class_density = True
            graph.update_colors()
            self.assertIs(graph.density_img, density)
            self.assertIs(graph.plot_widget.addItem.call_args[0][0], density)

            graph.update_coordinates = lambda: (None, None)
            graph.reset_graph()
            self.assertIsNone(graph.density_img)
            self.assertIs(graph.plot_widget.removeItem.call_args[0][0], density)

    def test_labels(self):
        graph = self.graph
        graph.reset_graph()

        self.assertEqual(graph.labels, [])

        self.master.get_label_data = lambda: \
            np.array([str(x) for x in range(10)], dtype=object)
        graph.update_labels()
        self.assertEqual(
            [label.textItem.toPlainText() for label in graph.labels],
            [str(i) for i in range(10)])

        # Label only selected
        selected = [1, 3, 5]
        graph.select_by_indices(selected)
        self.graph.label_only_selected = True
        graph.update_labels()
        self.assertEqual(
            [label.textItem.toPlainText() for label in graph.labels],
            [str(x) for x in selected])
        x, y = graph.scatterplot_item.getData()
        for i, index in enumerate(selected):
            self.assertEqual(x[index], graph.labels[i].x())
            self.assertEqual(y[index], graph.labels[i].y())

        # Disable label only selected
        self.graph.label_only_selected = False
        graph.update_labels()
        self.assertEqual(
            [label.textItem.toPlainText() for label in graph.labels],
            [str(i) for i in range(10)])
        x, y = graph.scatterplot_item.getData()
        for xi, yi, label in zip(x, y, graph.labels):
            self.assertEqual(xi, label.x())
            self.assertEqual(yi, label.y())

        # Label only selected + sampling
        selected = [1, 3, 4, 5, 6, 7, 9]
        graph.select_by_indices(selected)
        self.graph.label_only_selected = True
        graph.update_labels()
        graph.set_sample_size(5)
        for label in graph.labels:
            ind = int(label.textItem.toPlainText())
            self.assertIn(ind, selected)
            self.assertEqual(label.x(), x[ind])
            self.assertEqual(label.y(), y[ind])

    def test_labels_update_coordinates(self):
        graph = self.graph
        self.master.get_label_data = lambda: \
            np.array([str(x) for x in range(10)], dtype=object)

        graph.reset_graph()
        graph.set_sample_size(7)
        x, y = graph.scatterplot_item.getData()
        for xi, yi, label in zip(x, y, graph.labels):
            self.assertEqual(xi, label.x())
            self.assertEqual(yi, label.y())

        self.master.get_coordinates_data = \
            lambda: (np.arange(10, 20), np.arange(50, 60))
        graph.update_coordinates()
        x, y = graph.scatterplot_item.getData()
        for xi, yi, label in zip(x, y, graph.labels):
            self.assertEqual(xi, label.x())
            self.assertEqual(yi, label.y())

    def test_shapes(self):
        graph = self.graph

        self.master.get_shape_data = lambda: d
        d = np.arange(10, dtype=float) % 3

        graph.reset_graph()
        scatterplot_item = graph.scatterplot_item
        symbols = scatterplot_item.data["symbol"]
        self.assertTrue(all(symbol == graph.CurveSymbols[i % 3]
                            for i, symbol in enumerate(symbols)))

        d = np.arange(10, dtype=float) % 2
        graph.update_shapes()
        symbols = scatterplot_item.data["symbol"]
        self.assertTrue(all(symbol == graph.CurveSymbols[i % 2]
                            for i, symbol in enumerate(symbols)))

        d = None
        graph.update_shapes()
        symbols = scatterplot_item.data["symbol"]
        self.assertEqual(len(set(symbols)), 1)

    def test_shapes_nan(self):
        graph = self.graph

        self.master.get_shape_data = lambda: d
        d = np.arange(10, dtype=float) % 3
        d[2] = np.nan

        graph.reset_graph()
        self.assertEqual(graph.scatterplot_item.data["symbol"][2], '?')

        d[:] = np.nan
        graph.update_shapes()
        self.assertTrue(
            all(symbol == '?'
                for symbol in graph.scatterplot_item.data["symbol"]))

        def impute0(data, _):
            data[np.isnan(data)] = 0

        self.master.impute_shapes = impute0
        d = np.arange(10, dtype=float) % 3
        d[2] = np.nan
        graph.update_shapes()
        self.assertEqual(graph.scatterplot_item.data["symbol"][2],
                         graph.CurveSymbols[0])

    def test_show_grid(self):
        graph = self.graph
        show_grid = self.graph.plot_widget.showGrid = Mock()
        graph.show_grid = False
        graph.update_grid_visibility()
        self.assertEqual(show_grid.call_args[1], dict(x=False, y=False))

        graph.show_grid = True
        graph.update_grid_visibility()
        self.assertEqual(show_grid.call_args[1], dict(x=True, y=True))

    def test_show_legend(self):
        graph = self.graph
        graph.reset_graph()

        shape_legend = self.graph.shape_legend.setVisible = Mock()
        color_legend = self.graph.color_legend.setVisible = Mock()
        shape_labels = color_labels = None  # Avoid pylint warning
        self.master.get_shape_labels = lambda: shape_labels
        self.master.get_color_labels = lambda: color_labels
        for shape_labels in (None, ["a", "b"]):
            for color_labels in (None, ["c", "d"], None):
                for visible in (True, False, True):
                    graph.show_legend = visible
                    graph.update_legends()
                    self.assertIs(
                        shape_legend.call_args[0][0],
                        visible and bool(shape_labels),
                        msg="error at {}, {}".format(visible, shape_labels))
                    self.assertIs(
                        color_legend.call_args[0][0],
                        visible and bool(color_labels),
                        msg="error at {}, {}".format(visible, color_labels))

    def test_show_legend_no_data(self):
        graph = self.graph
        self.master.get_shape_labels = lambda: ["a", "b"]
        self.master.get_color_labels = lambda: ["c", "d"]
        self.master.get_shape_data = lambda: np.arange(10) % 2
        self.master.get_color_data = lambda: np.arange(10) < 6
        graph.reset_graph()

        shape_legend = self.graph.shape_legend.setVisible = Mock()
        color_legend = self.graph.color_legend.setVisible = Mock()
        self.master.get_coordinates_data = lambda: (None, None)
        graph.reset_graph()
        self.assertFalse(shape_legend.call_args[0][0])
        self.assertFalse(color_legend.call_args[0][0])

    def test_legend_combine(self):
        master = self.master
        graph = self.graph
        graph.reset_graph()

        shape_legend = self.graph.shape_legend.setVisible = Mock()
        color_legend = self.graph.color_legend.setVisible = Mock()

        master.get_shape_labels = lambda: ["a", "b"]
        master.get_color_labels = lambda: ["c", "d"]
        graph.update_legends()
        self.assertTrue(shape_legend.call_args[0][0])
        self.assertTrue(color_legend.call_args[0][0])

        master.get_color_labels = lambda: ["a", "b"]
        graph.update_legends()
        self.assertTrue(shape_legend.call_args[0][0])
        self.assertFalse(color_legend.call_args[0][0])
        self.assertEqual(len(graph.shape_legend.items), 2)

        master.is_continuous_color = lambda: True
        master.get_color_data = lambda: np.arange(10, dtype=float)
        graph.update_colors()
        self.assertTrue(shape_legend.call_args[0][0])
        self.assertTrue(color_legend.call_args[0][0])
        self.assertEqual(len(graph.shape_legend.items), 2)

    def test_select_by_click(self):
        graph = self.graph
        graph.reset_graph()
        points = graph.scatterplot_item.points()
        graph.select_by_click(None, [points[2]])
        np.testing.assert_almost_equal(graph.get_selection(), [2])
        with patch("AnyQt.QtWidgets.QApplication.keyboardModifiers",
                   lambda: Qt.ShiftModifier):
            graph.select_by_click(None, points[3:6])
        np.testing.assert_almost_equal(
            list(graph.get_selection()), [2, 3, 4, 5])
        np.testing.assert_almost_equal(
            graph.selection, [0, 0, 1, 2, 2, 2, 0, 0, 0, 0])

    def test_select_by_rectangle(self):
        graph = self.graph
        coords = np.array(
            [(x, y) for y in range(10) for x in range(10)], dtype=float).T
        self.master.get_coordinates_data = lambda: coords

        graph.reset_graph()
        graph.select_by_rectangle(QRectF(3, 5, 3.9, 2.9))
        self.assertTrue(
            all(selected == (3 <= coords[0][i] <= 6 and 5 <= coords[1][i] <= 7)
                for i, selected in enumerate(graph.selection)))

    def test_select_by_indices(self):
        graph = self.graph
        graph.reset_graph()
        graph.label_only_selected = True

        def select(modifiers, indices):
            with patch("AnyQt.QtWidgets.QApplication.keyboardModifiers",
                       lambda: modifiers):
                graph.update_selection_colors = Mock()
                graph.update_labels = Mock()
                self.master.selection_changed = Mock()

                graph.select_by_indices(np.array(indices))
                graph.update_selection_colors.assert_called_with()
                if graph.label_only_selected:
                    graph.update_labels.assert_called_with()
                else:
                    graph.update_labels.assert_not_called()
                self.master.selection_changed.assert_called_with()

        select(0, [7, 8, 9])
        np.testing.assert_almost_equal(
            graph.selection, [0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

        select(Qt.ShiftModifier | Qt.ControlModifier, [5, 6])
        np.testing.assert_almost_equal(
            graph.selection, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        select(Qt.ShiftModifier, [3, 4, 5])
        np.testing.assert_almost_equal(
            graph.selection, [0, 0, 0, 2, 2, 2, 1, 1, 1, 1])

        select(Qt.AltModifier, [1, 3, 7])
        np.testing.assert_almost_equal(
            graph.selection, [0, 0, 0, 0, 2, 2, 1, 0, 1, 1])

        select(0, [1, 8])
        np.testing.assert_almost_equal(
            graph.selection, [0, 1, 0, 0, 0, 0, 0, 0, 1, 0])

        graph.label_only_selected = False
        select(0, [3, 4])

    def test_unselect_all(self):
        graph = self.graph
        graph.reset_graph()
        graph.label_only_selected = True

        graph.select_by_indices([3, 4, 5])
        np.testing.assert_almost_equal(
            graph.selection, [0, 0, 0, 1, 1, 1, 0, 0, 0, 0])

        graph.update_selection_colors = Mock()
        graph.update_labels = Mock()
        self.master.selection_changed = Mock()

        graph.unselect_all()
        self.assertIsNone(graph.selection)
        graph.update_selection_colors.assert_called_with()
        graph.update_labels.assert_called_with()
        self.master.selection_changed.assert_called_with()

        graph.update_selection_colors.reset_mock()
        graph.update_labels.reset_mock()
        self.master.selection_changed.reset_mock()

        graph.unselect_all()
        self.assertIsNone(graph.selection)
        graph.update_selection_colors.assert_not_called()
        graph.update_labels.assert_not_called()
        self.master.selection_changed.assert_not_called()


if __name__ == "__main__":
    import unittest
    unittest.main()
