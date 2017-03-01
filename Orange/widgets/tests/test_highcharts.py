import time
import os
import sys
import unittest

from AnyQt.QtCore import Qt, QPoint, QObject
from AnyQt.QtWidgets import qApp
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
        while True:
            try:
                svg = scatter.svg()
                break
            except ValueError:
                qApp.processEvents()

        self.assertEqual(svg[:5], '<svg ')
        self.assertEqual(svg[-6:], '</svg>')

    @unittest.skipIf(os.environ.get('APPVEYOR'), 'test stalls on AppVeyor')
    @unittest.skipIf(sys.version_info[:2] <= (3, 4),
                     'the second iteration stalls on Travis / Py3.4')
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

        while scatter.isHidden() or not scatter.geometry().isValid():
            qApp.processEvents()
            time.sleep(.05)

        time.sleep(1)  # add some time for WM to place window or whatever
        topleft = scatter.geometry().topLeft()
        bottomright = scatter.geometry().bottomRight()
        startpos = topleft + QPoint(100, 100)
        endpos = bottomright - QPoint(300, 300)

        # Simulate selection
        QTest.mousePress(scatter, Qt.LeftButton, Qt.NoModifier, startpos, 1000)
        qApp.processEvents()
        QTest.mouseMove(scatter, endpos)
        qApp.processEvents()
        QTest.mouseRelease(scatter, Qt.LeftButton, Qt.NoModifier, endpos, 100)

        while not selected_indices:
            qApp.processEvents()
            time.sleep(.05)

        self.assertEqual(len(selected_indices), 1)
        self.assertGreater(len(selected_indices[0]), 0)

        # Simulate deselection
        QTest.mouseClick(scatter, Qt.LeftButton, Qt.NoModifier, startpos - QPoint(10, 10))

        while selected_indices:
            qApp.processEvents()
            time.sleep(.05)

        self.assertFalse(len(selected_indices))

        # Test Esc hiding
        QTest.keyClick(scatter, Qt.Key_Escape)
        self.assertTrue(scatter.isHidden())
