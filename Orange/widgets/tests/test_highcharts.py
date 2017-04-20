import time
import unittest

from AnyQt.QtCore import Qt, QPoint, QObject
from AnyQt.QtTest import QTest

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.highcharts import Highchart


class Scatter(Highchart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         options=dict(chart=dict(type='scatter')),
                         **kwargs)


class SelectionScatter(Scatter):
    def __init__(self, bridge, selected_indices_callback):
        super().__init__(bridge=bridge,
                         enable_select='xy+',
                         selection_callback=selected_indices_callback)


class HighchartTest(WidgetTest):
    def test_svg_is_svg(self):
        scatter = Scatter()
        scatter.chart(dict(series=dict(data=[[0, 1],
                                             [1, 2]])))
        svg = self.process_events(lambda: scatter.svg())

        self.assertEqual(svg[:5], '<svg ')
        self.assertEqual(svg[-6:], '</svg>')

    @unittest.skip("Does not work")
    def test_selection(self):

        class NoopBridge(QObject):
            pass

        for bridge in (NoopBridge(), None):
            self._selection_test(bridge)

    def _selection_test(self, bridge):
        data = Table('iris')
        selected_indices = []

        def selection_callback(indices):
            nonlocal selected_indices
            selected_indices = indices

        scatter = SelectionScatter(bridge, selection_callback)
        scatter.chart(options=dict(series=[dict(data=data.X[:, :2])]))
        scatter.show()

        self.process_events(lambda: not scatter.isHidden() and scatter.geometry().isValid())

        time.sleep(1)  # add some time for WM to place window or whatever
        topleft = scatter.geometry().topLeft()
        bottomright = scatter.geometry().bottomRight()
        startpos = topleft + QPoint(100, 100)
        endpos = bottomright - QPoint(300, 300)

        # Simulate selection
        QTest.mousePress(scatter, Qt.LeftButton, Qt.NoModifier, startpos, 1000)
        self.process_events()
        QTest.mouseMove(scatter, endpos)
        self.process_events()
        QTest.mouseRelease(scatter, Qt.LeftButton, Qt.NoModifier, endpos, 100)

        self.process_events(lambda: len(selected_indices))

        self.assertEqual(len(selected_indices), 1)
        self.assertGreater(len(selected_indices[0]), 0)

        # Simulate deselection
        QTest.mouseClick(scatter, Qt.LeftButton, Qt.NoModifier, startpos - QPoint(10, 10))

        self.process_events(lambda: not len(selected_indices))

        self.assertFalse(len(selected_indices))

        # Test Esc hiding
        QTest.keyClick(scatter, Qt.Key_Escape)
        self.assertTrue(scatter.isHidden())
