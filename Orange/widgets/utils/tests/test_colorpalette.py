import warnings
import unittest

import numpy as np

from AnyQt.QtCore import  Qt
from AnyQt.QtGui import QColor

from Orange.widgets.tests.base import WidgetTest

with warnings.catch_warnings():
    # This test tests an obsolete module, hence this warning is expected
    warnings.filterwarnings("ignore", ".*", DeprecationWarning)
    from Orange.widgets.utils.colorpalette import \
        ColorPaletteDlg, GradientPaletteGenerator, NAN_GREY


class TestColorPalette(WidgetTest):
    def test_colorpalette(self):
        dlg = ColorPaletteDlg(None)

        dlg.createContinuousPalette(
            "", "Gradient palette", False, QColor(Qt.white), QColor(Qt.black))

        dlg.contLeft.getColor().getRgb()
        dlg.contRight.getColor().getRgb()
        dlg.contpassThroughBlack


class GradientPaletteGeneratorTest(unittest.TestCase):
    def test_two_color(self):
        generator = GradientPaletteGenerator('#000', '#fff')
        for float_values, rgb in ((.5, (128, 128, 128)),
                                  (np.nan, NAN_GREY),
                                  ((0, .5, np.nan), ((0, 0, 0),
                                                     (128, 128, 128),
                                                     NAN_GREY))):
            np.testing.assert_equal(generator.getRGB(float_values), rgb)

    def test_three_color(self):
        generator = GradientPaletteGenerator('#f00', '#000', '#fff')
        for float_values, rgb in ((.5, (0, 0, 0)),
                                  (np.nan, NAN_GREY),
                                  ((0, .5, 1), ((255, 0, 0),
                                                (0, 0, 0),
                                                (255, 255, 255)))):
            np.testing.assert_equal(generator.getRGB(float_values), rgb)
