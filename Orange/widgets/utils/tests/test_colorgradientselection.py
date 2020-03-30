import numpy as np

from AnyQt.QtTest import QSignalSpy
from AnyQt.QtCore import Qt, QStringListModel

from Orange.widgets.utils.colorgradientselection import ColorGradientSelection
from Orange.widgets.tests.base import GuiTest

class TestColorGradientSelection(GuiTest):
    def test_constructor(self):
        w = ColorGradientSelection(thresholds=(0.1, 0.9))
        self.assertEqual(w.thresholds(), (0.1, 0.9))

        w = ColorGradientSelection(thresholds=(-0.1, 1.1))
        self.assertEqual(w.thresholds(), (0.0, 1.0))

        w = ColorGradientSelection(thresholds=(1.0, 0.0))
        self.assertEqual(w.thresholds(), (1.0, 1.0))

    def test_setModel(self):
        w = ColorGradientSelection()
        model = QStringListModel(["A", "B"])
        w.setModel(model)
        self.assertIs(w.model(), model)
        self.assertEqual(w.findData("B", Qt.DisplayRole), 1)
        current = QSignalSpy(w.currentIndexChanged)
        w.setCurrentIndex(1)
        self.assertEqual(w.currentIndex(), 1)
        self.assertSequenceEqual(list(current), [[1]])

    def test_thresholds(self):
        w = ColorGradientSelection()
        w.setThresholds(0.2, 0.8)
        self.assertEqual(w.thresholds(), (0.2, 0.8))
        w.setThresholds(0.5, 0.5)
        self.assertEqual(w.thresholds(), (0.5, 0.5))
        w.setThresholds(0.5, np.nextafter(0.5, 0))
        self.assertEqual(w.thresholds(), (0.5, 0.5))
        w.setThresholds(-1, 2)
        self.assertEqual(w.thresholds(), (0., 1.))
        w.setThresholds(0.1, 0.0)
        self.assertEqual(w.thresholds(), (0.1, 0.1))
        w.setThresholdLow(0.2)
        self.assertEqual(w.thresholds(), (0.2, 0.2))
        self.assertEqual(w.thresholdLow(), 0.2)
        w.setThresholdHigh(0.1)
        self.assertEqual(w.thresholdHigh(), 0.1)
        self.assertEqual(w.thresholds(), (0.1, 0.1))

    def test_slider_move(self):
        w = ColorGradientSelection()
        w.adjustSize()
        w.setThresholds(0.5, 0.5)
        changed = QSignalSpy(w.thresholdsChanged)
        sl, sh = w.slider_low, w.slider_high
        sl.triggerAction(sl.SliderToMinimum)
        self.assertEqual(len(changed), 1)
        low, high = changed[-1]
        self.assertLessEqual(low, high)
        self.assertEqual(low, 0.0)
        sl.triggerAction(sl.SliderToMaximum)
        self.assertEqual(len(changed), 2)
        low, high = changed[-1]
        self.assertLessEqual(low, high)
        self.assertEqual(low, 1.0)
        sh.triggerAction(sl.SliderToMinimum)
        self.assertEqual(len(changed), 3)
        low, high = changed[-1]
        self.assertLessEqual(low, high)
        self.assertEqual(high, 0.0)
