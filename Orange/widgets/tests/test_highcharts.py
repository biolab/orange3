import time
import os
import unittest

from AnyQt.QtCore import Qt, QPoint
from AnyQt.QtWidgets import qApp
from AnyQt.QtTest import QTest

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.highcharts import Highchart


class SelectionScatter(Highchart):
    def __init__(self, selected_indices_callback):
        super().__init__(enable_select='xy+',
                         selection_callback=selected_indices_callback,
                         options=dict(chart=dict(type='scatter')))


class HighchartTest(WidgetTest):
    @unittest.skipIf(os.environ.get('APPVEYOR'), 'test stalls on AppVeyor')
    def test_selection(self):
        data = Table('iris')
        selected_indices = []

        def selection_callback(indices):
            nonlocal selected_indices
            selected_indices = indices

        scatter = SelectionScatter(selection_callback)
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

        QTest.keyClick(scatter, Qt.Key_Escape)
        self.assertTrue(scatter.isHidden())
