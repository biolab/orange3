# pylint: disable=protected-access
import unittest
from unittest.mock import Mock, patch

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QImage, QColor, QIcon

from orangewidget.tests.base import GuiTest
from Orange.util import color_to_hex
from Orange.data import DiscreteVariable, ContinuousVariable, Variable
from Orange.preprocess.discretize import decimal_binnings
# pylint: disable=wildcard-import,unused-wildcard-import
from Orange.widgets.utils.colorpalettes import *


class PaletteTest(unittest.TestCase):
    def test_copy(self):
        palette = DiscretePalette(
            "custom", "c123", [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)])
        copy = palette.copy()
        self.assertEqual(copy.friendly_name, "custom")
        self.assertEqual(copy.name, "c123")
        np.testing.assert_equal(palette.palette, copy.palette)
        copy.palette[0, 0] += 1
        self.assertEqual(palette.palette[0, 0], 1)

    def test_qcolors(self):
        palcolors = [(1, 2, 3), (4, 5, 6)]
        nan_color = (7, 8, 9)
        palette = DiscretePalette(
            "custom", "c123", palcolors, nan_color=nan_color)
        self.assertEqual([col.getRgb()[:3] for col in palette.qcolors],
                         palcolors)
        self.assertEqual([col.getRgb()[:3] for col in palette.qcolors_w_nan],
                         palcolors + [nan_color])


class IndexPaletteTest(unittest.TestCase):
    """Tested through DiscretePalette because IndexedPalette is too abstract"""
    def setUp(self) -> None:
        self.palette = DiscretePalette(
            "custom", "c123", [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)])

    def test_len(self):
        self.assertEqual(len(self.palette), 4)

    def test_getitem(self):
        self.assertEqual(self.palette[1].getRgb()[:3], (4, 5, 6))
        self.assertEqual([col.getRgb()[:3] for col in self.palette[1:3]],
                         [(4, 5, 6), (7, 8, 9)])
        self.assertEqual([col.getRgb()[:3] for col in self.palette[0, 3, 0]],
                         [(1, 2, 3), (10, 11, 12), (1, 2, 3)])
        self.assertEqual([col.getRgb()[:3]
                          for col in self.palette[np.array([0, 3, 0])]],
                         [(1, 2, 3), (10, 11, 12), (1, 2, 3)])


class DiscretePaletteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.palette = DiscretePalette.from_colors(
            [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)])

    def test_from_colors(self):
        self.assertEqual(self.palette[2].getRgb()[:3], (7, 8, 9))

    def test_color_indices(self):
        a, nans = DiscretePalette._color_indices([1, 2, 3])
        self.assertIsInstance(a, np.ndarray)
        self.assertEqual(a.dtype, int)
        np.testing.assert_equal(a, [1, 2, 3])
        np.testing.assert_equal(nans, [False, False, False])

        a, nans = DiscretePalette._color_indices([1, 2.0, np.nan])
        self.assertIsInstance(a, np.ndarray)
        self.assertEqual(a.dtype, int)
        np.testing.assert_equal(a, [1, 2, -1])
        np.testing.assert_equal(nans, [False, False, True])

        a, nans = DiscretePalette._color_indices(np.array([1, 2, 3]))
        self.assertIsInstance(a, np.ndarray)
        self.assertEqual(a.dtype, int)
        np.testing.assert_equal(a, [1, 2, 3])
        np.testing.assert_equal(nans, [False, False, False])

        x = np.array([1, 2.0, np.nan])
        a, nans = DiscretePalette._color_indices(x)
        self.assertIsInstance(a, np.ndarray)
        self.assertEqual(a.dtype, int)
        np.testing.assert_equal(a, [1, 2, -1])
        np.testing.assert_equal(nans, [False, False, True])
        self.assertTrue(np.isnan(x[2]))

        x = np.array([])
        a, nans = DiscretePalette._color_indices(x)
        self.assertIsInstance(a, np.ndarray)
        self.assertEqual(a.dtype, int)
        np.testing.assert_equal(a, [])
        np.testing.assert_equal(nans, [])

    def test_values_to_colors(self):
        palette = self.palette

        x = np.array([1, 2.0, np.nan])
        colors = palette.values_to_colors(x)
        np.testing.assert_equal(colors, [[4, 5, 6], [7, 8, 9], NAN_COLOR])

        x = [1, 2.0, np.nan]
        colors = palette.values_to_colors(x)
        np.testing.assert_equal(colors, [[4, 5, 6], [7, 8, 9], NAN_COLOR])

    def test_values_to_qcolors(self):
        palette = self.palette

        x = np.array([1, 2.0, np.nan])
        colors = palette.values_to_qcolors(x)
        self.assertEqual([col.getRgb()[:3] for col in colors],
                         [(4, 5, 6), (7, 8, 9), NAN_COLOR])

        x = [1, 2.0, np.nan]
        colors = palette.values_to_qcolors(x)
        self.assertEqual([col.getRgb()[:3] for col in colors],
                         [(4, 5, 6), (7, 8, 9), NAN_COLOR])

    def test_value_to_color(self):
        palette = self.palette
        np.testing.assert_equal(palette.value_to_color(0), [1, 2, 3])
        np.testing.assert_equal(palette.value_to_color(1.), [4, 5, 6])
        np.testing.assert_equal(palette.value_to_color(np.nan), NAN_COLOR)

    def test_value_to_qcolor(self):
        palette = self.palette
        np.testing.assert_equal(
            palette.value_to_qcolor(0).getRgb(), (1, 2, 3, 255))
        np.testing.assert_equal(
            palette.value_to_qcolor(1.).getRgb(), (4, 5, 6, 255))
        np.testing.assert_equal(
            palette.value_to_qcolor(np.nan).getRgb()[:3], NAN_COLOR)

    def test_default(self):
        self.assertIs(DefaultDiscretePalette,
                      DiscretePalettes[DefaultDiscretePaletteName])


class LimitedDiscretePaletteTest(unittest.TestCase):
    @staticmethod
    def test_small_palettes():
        defcols = len(DefaultRGBColors.palette)

        palette = LimitedDiscretePalette(3)
        np.testing.assert_equal(palette.palette, DefaultRGBColors.palette[:3])

        palette = LimitedDiscretePalette(defcols)
        np.testing.assert_equal(palette.palette, DefaultRGBColors.palette)

        palette = LimitedDiscretePalette(defcols + 1)
        np.testing.assert_equal(palette.palette, Glasbey.palette[:defcols + 1])

        palette = LimitedDiscretePalette(100)
        np.testing.assert_equal(palette.palette, Glasbey.palette[:100])

    def test_large_palettes(self):
        palette = LimitedDiscretePalette(257)
        qcolors = palette.qcolors
        qcolors_w_nan = palette.qcolors_w_nan
        c256 = qcolors[256].getRgb()

        self.assertEqual(len(palette), 257)
        self.assertEqual(len(palette.palette), 257)
        self.assertEqual(len(qcolors), 257)
        self.assertEqual(len(qcolors_w_nan), 258)
        self.assertEqual([c.getRgb() for c in qcolors],
                         [c.getRgb() for c in qcolors_w_nan[:-1]])
        self.assertEqual(palette[256].getRgb(), c256)
        np.testing.assert_equal(palette.value_to_color(256), c256[:3])
        self.assertEqual(palette.value_to_qcolor(256).getRgb(), c256)
        np.testing.assert_equal(palette.values_to_colors([256])[0], c256[:3])
        self.assertEqual(palette.values_to_qcolors([256])[0].getRgb(), c256)

        for size in range(1020, 1030):
            self.assertEqual(len(LimitedDiscretePalette(size)), size)

    @staticmethod
    def test_forced_glasbey_palettes():
        palette = LimitedDiscretePalette(5, force_glasbey=True)
        np.testing.assert_equal(palette.palette, Glasbey.palette[:5])

    def test_deprecate_force_hsv_palettes(self):
        with self.assertWarns(DeprecationWarning):
            palette = LimitedDiscretePalette(3, force_hsv=False)
            np.testing.assert_equal(palette.palette,
                                    DefaultRGBColors.palette[:3])

        with self.assertWarns(DeprecationWarning):
            palette = LimitedDiscretePalette(5, force_hsv=True)
            np.testing.assert_equal(palette.palette, Glasbey.palette[:5])


class HuePaletteTest(unittest.TestCase):
    def test_n_colors(self):
        palette = HuePalette(42)
        self.assertEqual(len(palette), 42)


class ContinuousPaletteTest(GuiTest):
    @staticmethod
    def assert_equal_within(a, b, diff):
        a = a.astype(float)  # make sure a is a signed type
        np.testing.assert_array_less(np.abs(a - b), diff)

    @staticmethod
    def test_color_indices():
        x = [0, 1, 2, 1, 0, np.nan, 1]
        a, nans = ContinuousPalette._color_indices(x)
        np.testing.assert_equal(a, [0, 128, 255, 128, 0, -1, 128])
        np.testing.assert_equal(nans, [False] * 5 + [True, False])

        x = [np.nan, np.nan, np.nan]
        a, nans = ContinuousPalette._color_indices(x)
        np.testing.assert_equal(a, [-1, -1, -1])
        np.testing.assert_equal(nans, [True, True, True])

        x = []
        a, nans = ContinuousPalette._color_indices(x)
        np.testing.assert_equal(a, [])
        np.testing.assert_equal(nans, [])

    @staticmethod
    def test_color_indices_low_high():
        x = [0, 1, 2, 1, 4, np.nan, 3]
        a, nans = ContinuousPalette._color_indices(x)
        np.testing.assert_equal(a, [0, 64, 128, 64, 255, -1, 191])
        np.testing.assert_equal(nans, [False] * 5 + [True, False])

        x = [0, 1, 2, 1, 4, np.nan, 3]
        a, nans = ContinuousPalette._color_indices(x, low=2)
        np.testing.assert_equal(a, [0, 0, 0, 0, 255, -1, 128])
        np.testing.assert_equal(nans, [False] * 5 + [True, False])

        x = [0, 1, 2, 1, 4, np.nan, 3]
        a, nans = ContinuousPalette._color_indices(x, high=2)
        np.testing.assert_equal(a, [0, 128, 255, 128, 255, -1, 255])
        np.testing.assert_equal(nans, [False] * 5 + [True, False])

        x = [0, 1, 2, 1, 4, np.nan, 3]
        a, nans = ContinuousPalette._color_indices(x, low=1, high=3)
        np.testing.assert_equal(a, [0, 0, 128, 0, 255, -1, 255])
        np.testing.assert_equal(nans, [False] * 5 + [True, False])

        x = [0, 1, 2, 1, 4, np.nan, 3]
        a, nans = ContinuousPalette._color_indices(x, low=0, high=8)
        np.testing.assert_equal(a, [0, 32, 64, 32, 128, -1, 96])
        np.testing.assert_equal(nans, [False] * 5 + [True, False])

        x = [1, 1, 1, np.nan]
        a, nans = ContinuousPalette._color_indices(x)
        np.testing.assert_equal(a, [128, 128, 128, -1])
        np.testing.assert_equal(nans, [False] * 3 + [True])

        x = [np.nan, np.nan, np.nan]
        a, nans = ContinuousPalette._color_indices(x)
        np.testing.assert_equal(a, [-1, -1, -1])
        np.testing.assert_equal(nans, [True, True, True])

        x = []
        a, nans = ContinuousPalette._color_indices(x)
        np.testing.assert_equal(a, [])
        np.testing.assert_equal(nans, [])

    def test_values_to_colors(self):
        def assert_equal_colors(x, indices, **args):
            expected = [palette.palette[idx] if idx >= 0 else NAN_COLOR
                        for idx in indices]
            np.testing.assert_equal(
                palette.values_to_colors(x, **args),
                expected)
            np.testing.assert_equal(
                [col.getRgb()[:3]
                 for col in palette.values_to_qcolors(x, **args)],
                expected)

        palette = list(ContinuousPalettes.values())[-1]
        assert_equal_colors(
            [0, 1, 2, 1, 4, np.nan, 3],
            [0, 64, 128, 64, 255, -1, 191])

        assert_equal_colors(
            [0, 1, 2, 1, 4, np.nan, 3],
            [0, 0, 0, 0, 255, -1, 128], low=2)

        assert_equal_colors(
            [0, 1, 2, 1, 4, np.nan, 3],
            [0, 128, 255, 128, 255, -1, 255], high=2)

        assert_equal_colors(
            [0, 1, 2, 1, 4, np.nan, 3],
            [0, 0, 128, 0, 255, -1, 255], low=1, high=3)

        assert_equal_colors(
            [0, 1, 2, 1, 4, np.nan, 3],
            [0, 32, 64, 32, 128, -1, 96], low=0, high=8)

        assert_equal_colors(
            [1, 1, 1, np.nan],
            [128, 128, 128, -1])

        assert_equal_colors(
            [np.nan, np.nan, np.nan],
            [-1, -1, -1])

        self.assertEqual(len(palette.values_to_colors([])), 0)
        self.assertEqual(len(palette.values_to_qcolors([])), 0)

    def test_value_to_color(self):
        def assert_equal_color(x, index, **args):
            self.assertEqual(palette._color_index(x, **args), index)
            expected = palette.palette[index] if index != -1 else NAN_COLOR
            np.testing.assert_equal(
                palette.value_to_color(x, **args),
                expected)
            np.testing.assert_equal(
                palette.value_to_qcolor(x, **args).getRgb()[:3],
                expected)
            if not args:
                np.testing.assert_equal(
                    palette[x].getRgb()[:3],
                    expected)

        palette = list(ContinuousPalettes.values())[-1]

        assert_equal_color(1, 255)
        assert_equal_color(1, 128, high=2)
        assert_equal_color(0, 128, low=5, high=-5)
        assert_equal_color(5, 128, high=10)
        assert_equal_color(-15, 128, low=-20, high=-10)
        assert_equal_color(-10, 255, low=-20, high=-10)
        assert_equal_color(-20, 0, low=-20, high=-10)
        assert_equal_color(0, 128, low=13, high=13)

        assert_equal_color(2, 255)
        assert_equal_color(-1, 0)
        assert_equal_color(0, 0, low=0.5)
        assert_equal_color(1, 255, high=0.5)

        assert_equal_color(np.nan, -1)
        assert_equal_color(np.nan, -1, high=2)
        assert_equal_color(np.nan, -1, low=5, high=-5)
        assert_equal_color(np.nan, -1, low=5, high=5)
        assert_equal_color(np.nan, -1, low=5)

    def test_lookup_table(self):
        palette = list(ContinuousPalettes.values())[-1]
        np.testing.assert_equal(palette.lookup_table(), palette.palette)

        indices = np.r_[[0] * 12, np.arange(0, 255, 2), [255] * 116]
        colors = palette.palette[indices]
        self.assert_equal_within(
            palette.lookup_table(12 / 256, 140 / 256), colors, 5)

    def test_color_strip_horizontal(self):
        palette = list(ContinuousPalettes.values())[-1]
        img = palette.color_strip(57, 17)
        self.assertEqual(img.width(), 57)
        self.assertEqual(img.height(), 17)

        img = palette.color_strip(256, 3)
        img = img.toImage().convertToFormat(QImage.Format_RGB888)
        for i in range(3):
            ptr = img.scanLine(i)
            ptr.setsize(256 * 3)
            a = np.array(ptr).reshape(256, 3)
            np.testing.assert_equal(a, palette.palette)

        img = palette.color_strip(64, 3)
        img = img.toImage().convertToFormat(QImage.Format_RGB888)
        for i in range(3):
            ptr = img.scanLine(i)
            ptr.setsize(64 * 3)
            a = np.array(ptr).reshape(64, 3)
            # Colors differ due to rounding when computing indices
            self.assert_equal_within(a, palette.palette[::4], 15)

    def test_color_strip_vertical(self):
        palette = list(ContinuousPalettes.values())[-1]
        img = palette.color_strip(57, 13, Qt.Vertical)
        self.assertEqual(img.width(), 13)
        self.assertEqual(img.height(), 57)

        img = palette.color_strip(256, 3, Qt.Vertical)
        img = img.toImage().convertToFormat(QImage.Format_RGB888)
        for i in range(256):
            ptr = img.scanLine(i)
            ptr.setsize(3 * 3)
            a = np.array(ptr).reshape(3, 3)
            self.assertTrue(np.all(a == palette.palette[255 - i]))


    def test_from_colors(self):
        palette = ContinuousPalette.from_colors((255, 255, 0), (0, 255, 255))
        colors = palette.palette
        np.testing.assert_equal(colors[:, 0], np.arange(255, -1, -1))
        np.testing.assert_equal(colors[:, 1], 255)
        np.testing.assert_equal(colors[:, 2], np.arange(256))

        palette = ContinuousPalette.from_colors((127, 0, 0), (0, 0, 255), True)
        colors = palette.palette
        line = np.r_[np.arange(127, -1, -1), np.zeros(128)]
        self.assert_equal_within(colors[:, 0], line, 2)
        np.testing.assert_equal(colors[:, 1], 0)
        self.assert_equal_within(colors[:, 2], 2 * line[::-1], 2)

        palette = ContinuousPalette.from_colors((255, 0, 0), (0, 0, 255),
                                                pass_through=(255, 255, 0))
        colors = palette.palette
        self.assert_equal_within(
            colors[:, 0],
            np.r_[[255] * 128, np.arange(255, 0, -2)], 3)
        self.assert_equal_within(
            colors[:, 1],
            np.r_[np.arange(0, 255, 2), np.arange(255, 0, -2)], 3)
        self.assert_equal_within(
            colors[:, 2],
            np.r_[[0] * 128, np.arange(0, 255, 2)], 3)

    def test_default(self):
        self.assertIs(DefaultContinuousPalette,
                      ContinuousPalettes[DefaultContinuousPaletteName])


class BinnedPaletteTest(unittest.TestCase):
    def setUp(self):
        self.palette = list(ContinuousPalettes.values())[-1]
        self.bins = np.arange(10, 101, 10)
        self.binned = BinnedContinuousPalette.from_palette(
            self.palette, self.bins)

    def test_from_palette_continuous(self):
        np.testing.assert_equal(self.binned.bins, self.bins)
        np.testing.assert_equal(
            self.binned.palette,
            self.palette.values_to_colors([15, 25, 35, 45, 55, 65, 75, 85, 95],
                                          low=10, high=100)
        )

        bins = np.array([100, 200])
        binned = BinnedContinuousPalette.from_palette(self.palette, bins)
        np.testing.assert_equal(binned.bins, bins)
        np.testing.assert_equal(binned.palette, [self.palette.palette[128]])

    def test_from_palette_binned(self):
        binned2 = BinnedContinuousPalette.from_palette(
            self.binned, np.arange(10))

        self.assertIsNot(self.binned, binned2)
        np.testing.assert_equal(binned2.bins, self.bins)
        np.testing.assert_equal(
            binned2.palette,
            self.palette.values_to_colors([15, 25, 35, 45, 55, 65, 75, 85, 95],
                                          low=10, high=100)
        )

    def test_from_palette_discrete(self):
        self.assertRaises(
            TypeError,
            BinnedContinuousPalette.from_palette, DefaultRGBColors, [1, 2, 3])

    def test_bin_indices(self):
        for x in ([15, 61, 150, np.nan, -5],
                  np.array([15, 61, 150, np.nan, -5])):
            indices, nans = self.binned._bin_indices(x)
            np.testing.assert_equal(indices, [0, 5, 8, -1, 0])
            np.testing.assert_equal(nans, [False, False, False, True, False])

    def test_values_to_colors(self):
        for x in ([15, 61, 150, np.nan, -5],
                  np.array([15, 61, 150, np.nan, -5])):
            expected = [self.binned.palette[idx] if idx >= 0 else NAN_COLOR
                        for idx in [0, 5, 8, -1, 0]]
            np.testing.assert_equal(
                self.binned.values_to_colors(x),
                expected)
            np.testing.assert_equal(
                [col.getRgb()[:3]
                 for col in self.binned.values_to_qcolors(x)],
                expected)

            for col, exp in zip(x, expected):
                np.testing.assert_equal(
                    self.binned.value_to_color(col), exp)
                np.testing.assert_equal(
                    self.binned.value_to_qcolor(col).getRgb()[:3], exp)

    def test_copy(self):
        copy = self.binned.copy()
        np.testing.assert_equal(self.binned.palette, copy.palette)
        np.testing.assert_equal(self.binned.bins, copy.bins)
        copy.palette[0, 0] += 1
        self.assertNotEqual(self.binned.palette[0, 0], copy.palette[0, 0])
        copy.bins[0] += 1
        self.assertNotEqual(self.bins[0], copy.bins[0])

    def test_decimal_binnings(self):
        """test for consistency with binning from discretize"""
        data = np.array([1, 2])
        bins = decimal_binnings(data)[0].thresholds
        binned = BinnedContinuousPalette.from_palette(self.palette, bins)
        colors = binned.values_to_colors(data)
        assert not np.array_equal(colors[0], colors[1])


class UtilsTest(GuiTest):
    def test_coloricon(self):
        color = QColor(1, 2, 3)
        icon = ColorIcon(color, 16)
        self.assertIsInstance(icon, QIcon)
        sizes = icon.availableSizes()
        self.assertEqual(len(sizes), 1)
        size = sizes[0]
        self.assertEqual(size.width(), 16)
        self.assertEqual(size.height(), 16)
        pixmap = icon.pixmap(size)
        img = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        ptr = img.bits()
        ptr.setsize(16 * 16 * 3)
        a = np.array(ptr).reshape(256, 3)
        self.assertTrue(np.all(a == [1, 2, 3]))

    def test_get_default_curve_colors(self):
        def equal_colors(n, palette):
            colors = get_default_curve_colors(n)
            self.assertEqual(len(colors), n)
            self.assertTrue(all(color.getRgb() == palcol.getRgb()
                                for color, palcol in zip(colors, palette)))

        n_dark = len(Dark2Colors)
        n_rgb = len(DefaultRGBColors)
        equal_colors(2, Dark2Colors)
        equal_colors(n_dark, Dark2Colors)
        equal_colors(n_dark + 1, DefaultRGBColors)
        equal_colors(n_rgb, DefaultRGBColors)

        colors = get_default_curve_colors(n_rgb + 1)
        self.assertTrue(
            all(color.getRgb() == palcol.getRgb()
                for color, palcol in zip(colors,
                                         LimitedDiscretePalette(n_rgb + 1))))


class PatchedVariableTest(unittest.TestCase):
    def test_colors(self):
        var = Variable("x")
        colors = [Mock(), Mock()]
        var.colors = colors
        self.assertIs(var.colors, colors)

    def test_palette(self):
        var = Variable("x")
        palette = Mock()
        var.palette = palette
        self.assertIs(var.palette, palette)

    def test_exclusive(self):
        var = Variable("x")
        colors = [Mock(), Mock()]
        palette = Mock()
        var.colors = colors
        # set_color for variable does not set this attribute; derived methods do
        var.attributes["colors"] = colors
        var.palette = palette
        self.assertIsNone(var.colors)
        self.assertTrue("palette" in var.attributes)
        self.assertFalse("colors" in var.attributes)

        var.colors = colors
        # set_color for variable does not set this attribute; derived methods do
        var.attributes["colors"] = colors
        self.assertIsNone(var.palette)
        self.assertTrue("colors" in var.attributes)
        self.assertFalse("palette" in var.attributes)


class PatchedDiscreteVariableTest(unittest.TestCase):
    def test_colors(self):
        var = DiscreteVariable.make("a", values=("F", "M"))
        self.assertIsNone(var._colors)
        self.assertEqual(var.colors.shape, (2, 3))
        self.assertFalse(var.colors.flags.writeable)

        var.colors = np.arange(6).reshape((2, 3))
        np.testing.assert_almost_equal(var.colors, [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(var.attributes["colors"],
                         {"F": "#000102", "M": "#030405"})
        self.assertFalse(var.colors.flags.writeable)
        with self.assertRaises(ValueError):
            var.colors[0] = [42, 41, 40]

        var = DiscreteVariable.make("x", values=("A", "B"))
        var.attributes["colors"] = {"A": "#0a0b0c", "B": "#0d0e0f"}
        np.testing.assert_almost_equal(var.colors, [[10, 11, 12], [13, 14, 15]])

        # Backward compatibility with list-like attributes
        var = DiscreteVariable.make("x", values=("A", "B"))
        var.attributes["colors"] = ["#0a0b0c", "#0d0e0f"]
        np.testing.assert_almost_equal(var.colors, [[10, 11, 12], [13, 14, 15]])

        # Test ncolors adapts to nvalues
        var = DiscreteVariable.make('foo', values=('d', 'r'))
        self.assertEqual(len(var.colors), 2)
        var.add_value('e')
        self.assertEqual(len(var.colors), 3)
        var.add_value('k')
        self.assertEqual(len(var.colors), 4)

        # Missing colors are retrieved from palette
        var = DiscreteVariable.make("x", values=("A", "B", "C"))
        palette = LimitedDiscretePalette(3).palette
        var.attributes["colors"] = {"C": color_to_hex(palette[0]),
                                    "B": "#0D0E0F"}
        np.testing.assert_almost_equal(var.colors,
                                       [palette[1], [13, 14, 15], palette[0]])

        # Variable with many values
        var = DiscreteVariable("x", values=tuple(f"v{i}" for i in range(1020)))
        self.assertEqual(len(var.colors), 1020)

    def test_colors_fallback_to_palette(self):
        var = DiscreteVariable.make("a", values=("F", "M"))
        var.palette = Dark2Colors
        colors = var.colors
        self.assertEqual(len(colors), 2)
        for color, palcol in zip(colors, Dark2Colors):
            np.testing.assert_equal(color, palcol.getRgb()[:3])
        # the palette has to stay specified
        self.assertEqual(var.attributes["palette"], var.palette.name)

        var = DiscreteVariable.make("a", values=[f"{i}" for i in range(40)])
        var.palette = Dark2Colors
        colors = var.colors
        self.assertEqual(len(colors), 40)
        for color, palcol in zip(colors, LimitedDiscretePalette(40)):
            np.testing.assert_equal(color, palcol.getRgb()[:3])
        # the palette has to stay specified
        self.assertEqual(var.attributes["palette"], var.palette.name)

    def test_colors_default(self):
        var = DiscreteVariable.make("a", values=("F", "M"))
        colors = var.colors
        self.assertEqual(len(colors), 2)
        for color, palcol in zip(colors, DefaultRGBColors):
            np.testing.assert_equal(color, palcol.getRgb()[:3])

        var = DiscreteVariable.make("a", values=[f"{i}" for i in range(40)])
        colors = var.colors
        self.assertEqual(len(colors), 40)
        for color, palcol in zip(colors, LimitedDiscretePalette(40)):
            np.testing.assert_equal(color, palcol.getRgb()[:3])

        var = DiscreteVariable.make("a", values=("M", "F"))
        var.attributes["colors"] = "foo"
        colors = var.colors
        self.assertEqual(len(colors), 2)
        for color, palcol in zip(colors, DefaultRGBColors):
            np.testing.assert_equal(color, palcol.getRgb()[:3])

    def test_colors_no_values(self):
        var = DiscreteVariable.make("a", values=())
        colors = var.colors
        self.assertEqual(len(colors), 0)

        var = DiscreteVariable.make("a", values=())
        var.palette = DefaultRGBColors
        colors = var.colors
        self.assertEqual(len(colors), 0)

    def test_get_palette(self):
        var = DiscreteVariable.make("a", values=("M", "F"))
        palette = var.palette
        self.assertEqual(len(palette), 2)
        np.testing.assert_equal(palette.palette, DefaultRGBColors.palette[:2])

        var = DiscreteVariable.make("a", values=("M", "F"))
        var.attributes["palette"] = "dark"
        palette = var.palette
        self.assertIs(palette, Dark2Colors)

        var = DiscreteVariable.make("a", values=("M", "F"))
        var.attributes["colors"] = ['#0a0b0c', '#0d0e0f']
        palette = var.palette
        np.testing.assert_equal(palette.palette, [[10, 11, 12], [13, 14, 15]])

    @staticmethod
    def test_ignore_malfformed_atrtibutes():
        var = DiscreteVariable("a", values=("M", "F"))
        var.attributes["colors"] = {"F": "foo", "M": "bar"}
        palette = var.palette
        np.testing.assert_equal(palette.palette,
                                LimitedDiscretePalette(2).palette)

class PatchedContinuousVariableTest(unittest.TestCase):
    def test_colors(self):
        with self.assertWarns(DeprecationWarning):
            a = ContinuousVariable("a")
            self.assertEqual(a.colors, ((0, 0, 255), (255, 255, 0), False))

            a = ContinuousVariable("a")
            a.attributes["colors"] = ['#010203', '#040506', True]
            self.assertEqual(a.colors, ((1, 2, 3), (4, 5, 6), True))

            a.colors = ((3, 2, 1), (6, 5, 4), True)
            self.assertEqual(a.colors, ((3, 2, 1), (6, 5, 4), True))

    def test_colors_from_palette(self):
        with self.assertWarns(DeprecationWarning):
            a = ContinuousVariable("a")
            a.palette = palette = ContinuousPalettes["rainbow_bgyr_35_85_c73"]
            colors = a.colors
            self.assertEqual(colors, (tuple(palette.palette[0]),
                                      tuple(palette.palette[255]),
                                      False))

            a = ContinuousVariable("a")
            a.attributes["palette"] = "rainbow_bgyr_35_85_c73"
            colors = a.colors
            self.assertEqual(colors, (tuple(palette.palette[0]),
                                      tuple(palette.palette[255]),
                                      False))

            a = ContinuousVariable("a")
            a.palette = palette = ContinuousPalettes["diverging_bwr_40_95_c42"]
            colors = a.colors
            self.assertEqual(colors, (tuple(palette.palette[0]),
                                      tuple(palette.palette[255]),
                                      True))

    def test_palette(self):
        palette = ContinuousPalettes["rainbow_bgyr_35_85_c73"]

        a = ContinuousVariable("a")
        a.palette = palette
        self.assertIs(a.palette, palette)

        a = ContinuousVariable("a")
        a.attributes["palette"] = palette.name
        self.assertIs(a.palette, palette)

        a = ContinuousVariable("a")
        self.assertIs(a.palette, DefaultContinuousPalette)

        with patch.object(ContinuousPalette, "from_colors") as from_colors:
            a = ContinuousVariable("a")
            a.attributes["colors"] = ('#0a0b0c', '#0d0e0f', False)
            with self.assertWarns(DeprecationWarning):
                palette = a.palette
            from_colors.assert_called_with((10, 11, 12), (13, 14, 15), False)
            self.assertIs(palette, from_colors.return_value)

        with patch.object(ContinuousPalette, "from_colors") as from_colors:
            a = ContinuousVariable("a")
            a.colors = (10, 11, 12), (13, 14, 15), False
            with self.assertWarns(DeprecationWarning):
                palette = a.palette
            from_colors.assert_called_with((10, 11, 12), (13, 14, 15), False)
            self.assertIs(palette, from_colors.return_value)


    def test_proxy_has_separate_colors(self):
        abc = ContinuousVariable("abc")
        abc1 = abc.make_proxy()
        abc2 = abc1.make_proxy()

        with self.assertWarns(DeprecationWarning):
            original_colors = abc.colors
        red_to_green = (255, 0, 0), (0, 255, 0), False
        blue_to_red = (0, 0, 255), (255, 0, 0), False

        abc1.colors = red_to_green
        abc2.colors = blue_to_red
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(abc.colors, original_colors)
            self.assertEqual(abc1.colors, red_to_green)
            self.assertEqual(abc2.colors, blue_to_red)


patch_variable_colors()

if __name__ == "__main__":
    unittest.main()
