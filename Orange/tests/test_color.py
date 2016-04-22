# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np

from Orange.widgets.utils.colorpalette import GradientPaletteGenerator, NAN_GREY


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
