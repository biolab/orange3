from orangewidget.tests.base import GuiTest
from Orange.widgets.utils.spinbox import DoubleSpinBox


class TestDoubleSpinBox(GuiTest):
    def test_double_spin_box(self):
        w = DoubleSpinBox(
            minimum=-1, maximum=1, value=0, singleStep=0.1, decimals=-1,
            minimumStep=1e-7,
        )
        self.assertEqual(w.minimum(), -1)
        self.assertEqual(w.maximum(), 1)
        self.assertEqual(w.value(), 0)
        self.assertEqual(w.singleStep(), 0.1)
        self.assertEqual(w.decimals(), -1)
        self.assertEqual(w.minimumStep(), 1e-7)

        w.setValue(2)
        self.assertEqual(w.value(), 1)
        w.setValue(0.999999)
        self.assertEqual(w.value(), 0.999999)
        w.stepBy(-1)
        self.assertEqual(w.value(), 0.899999)
        w.stepBy(1)
        self.assertEqual(w.value(), 0.999999)
        w.stepBy(1)
        self.assertEqual(w.value(), 1.0)

        w.setStepType(DoubleSpinBox.AdaptiveDecimalStepType)
        w.stepBy(-1)
        self.assertEqual(w.value(), 0.99)

        w.setValue(0.123456789)
        w.stepBy(1)
        self.assertEqual(w.value(), 0.133456789)
        w.stepBy(-1)
        self.assertEqual(w.value(), 0.123456789)
        w.setMinimumStep(0.001)
        w.setValue(0.00005)
        w.stepBy(1)
        w.setValue(0.00105)
        w.setDecimals(3)
        self.assertEqual(w.value(), 0.001)
        w.stepBy(1)
        self.assertEqual(w.value(), 0.002)
