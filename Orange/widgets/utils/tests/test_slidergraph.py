import numpy as np
from AnyQt.QtCore import Qt

from Orange.widgets import widget
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.slidergraph import SliderGraph


class SimpleWidget(widget.OWWidget):
    name = "Simple widget"

    def __init__(self):
        super().__init__()
        self.plot = SliderGraph(x_axis_label="label1",
                                y_axis_label="label2",
                                callback=lambda x: x)

        self.mainArea.layout().addWidget(self.plot)


class TestSliderGraph(WidgetTest):
    def setUp(self):
        self.data = [np.array([1, 2, 3, 4, 5, 6, 7])]
        self.widget = self.create_widget(SimpleWidget)

    def test_init(self):
        p = self.widget.plot

        # labels set correctly?
        self.assertEqual("label1", p.getAxis("bottom").labelText)
        self.assertEqual("label2", p.getAxis("left").labelText)

        # pylint: disable=protected-access
        self.assertIsNone(p._line)
        self.assertIsNone(p.sequences)
        self.assertIsNone(p.x)
        self.assertIsNone(p.selection_limit)
        self.assertIsNone(p.data_increasing)

        self.assertListEqual([], p.plot_horlabel)
        self.assertListEqual([], p.plot_horline)

    def test_plot(self):
        p = self.widget.plot
        x = np.arange(len(self.data[0]))
        p.update(x, self.data, [Qt.red],
                 cutpoint_x=1)

        # labels set correctly?
        self.assertEqual("label1", p.getAxis("bottom").labelText)
        self.assertEqual("label2", p.getAxis("left").labelText)

        # pylint: disable=protected-access
        self.assertIsNotNone(p._line)
        np.testing.assert_array_equal(self.data, p.sequences)
        np.testing.assert_array_equal(x, p.x)
        self.assertTrue(p.data_increasing)

        self.assertEqual(len(p.plot_horlabel), 1)
        self.assertEqual(len(p.plot_horline), 1)
        # pylint: disable=protected-access
        self.assertIsNotNone(p._line)

    def test_plot_selection_limit(self):
        p = self.widget.plot
        x = np.arange(len(self.data[0]))
        p.update(x, self.data, [Qt.red],
                 cutpoint_x=1, selection_limit=(0, 2))

        # labels set correctly?
        self.assertEqual("label1", p.getAxis("bottom").labelText)
        self.assertEqual("label2", p.getAxis("left").labelText)

        # pylint: disable=protected-access
        self.assertIsNotNone(p._line)
        np.testing.assert_array_equal(self.data, p.sequences)
        np.testing.assert_array_equal(x, p.x)
        self.assertTrue(p.data_increasing)
        self.assertTupleEqual((0, 2), p.selection_limit)
        # pylint: disable=protected-access
        self.assertEqual((0, 2), p._line.maxRange)

        self.assertEqual(len(p.plot_horlabel), 1)
        self.assertEqual(len(p.plot_horline), 1)
        # pylint: disable=protected-access
        self.assertIsNotNone(p._line)

    def test_plot_no_cutpoint(self):
        """
        When no cutpoint provided there must be no cutpoint plotted.
        """
        p = self.widget.plot
        x = np.arange(len(self.data[0]))
        p.update(x, self.data, [Qt.red])

        # pylint: disable=protected-access
        self.assertIsNone(p._line)

        # then it is set
        p.update(x, self.data, [Qt.red], cutpoint_x=1)
        # pylint: disable=protected-access
        self.assertIsNotNone(p._line)

        # and re-ploted without cutpoint again
        p.update(x, self.data, [Qt.red])
        # pylint: disable=protected-access
        self.assertIsNone(p._line)
