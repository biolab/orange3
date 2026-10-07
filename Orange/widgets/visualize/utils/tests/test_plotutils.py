import unittest
from unittest.mock import patch

import numpy as np
from AnyQt.QtCore import QPointF

from orangewidget.tests.base import GuiTest

from Orange.widgets.visualize.utils.plotutils import (
    AxisItem, InteractiveViewBox
)


class TestInteractiveViewBox(unittest.TestCase):

    def test_update_scale_box(self):
        view_box = InteractiveViewBox(graph=None)
        with patch.object(view_box, "mapToView", lambda x: x):
            view_box.updateScaleBox(QPointF(0, 0), QPointF(2, 2))
            tr = view_box.rbScaleBox.transform()
            self.assertFalse(tr.isRotating())
            trm = [[tr.m11(), tr.m12(), tr.m13()],
                   [tr.m21(), tr.m22(), tr.m23()],
                   [tr.m31(), tr.m32(), tr.m33()]]
            np.testing.assert_equal(trm, [[2, 0, 0], [0, 2, 0], [0, 0, 1]])


class TestAxisItem(GuiTest):
    def test_si_prefix_without_units(self):
        axis = AxisItem("bottom")
        axis.setLabel("a")

        axis.setRange(0, 0.5)
        self.assertEqual(axis.autoSIPrefixScale, 1)
        axis.setRange(0, 5e5)
        self.assertEqual(axis.autoSIPrefixScale, 1)

        axis.setRange(0, 5e-4)
        self.assertEqual(axis.autoSIPrefixScale, 1e6)
        axis.setRange(0, 5e6)
        self.assertEqual(axis.autoSIPrefixScale, 1e-6)


if __name__ == '__main__':
    unittest.main()
