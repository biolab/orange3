import unittest

import numpy as np

from Orange.widgets.utils.colorpalette import GradientPaletteGenerator, NAN_GREY


class GradientPaletteGeneratorTest(unittest.TestCase):
    def test_two_color(self):
        generator = GradientPaletteGenerator('#000', '#fff')
        np.testing.assert_equal(generator.getRGB(.5), (128, 128, 128))
        np.testing.assert_equal(generator.getRGB(np.nan), NAN_GREY)
        np.testing.assert_equal(generator.getRGB([0, .5, np.nan]),
                                [(0, 0, 0),
                                 (128, 128, 128),
                                 NAN_GREY])

    def test_three_color(self):
        generator = GradientPaletteGenerator('#f00', '#000', '#fff')
        np.testing.assert_equal(generator.getRGB(.5), (0, 0, 0))
        np.testing.assert_equal(generator.getRGB(np.nan), NAN_GREY)
        np.testing.assert_equal(generator.getRGB([0, .5, 1]),
                                [(255, 0, 0),
                                 (0, 0, 0),
                                 (255, 255, 255)])

