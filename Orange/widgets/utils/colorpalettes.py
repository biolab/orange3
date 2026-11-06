from collections.abc import Sequence

import numpy as np

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QImage, QPixmap, QColor, QIcon

from . import color as core

# The color module defines the core logic (color palettes and patching Variable classes)
# and does not require Qt.
# This module is an adapter over it, adding the Qt features used for the GUI.

class _QtPalette:
    @property
    def qcolors(self):
        return np.array([QColor(*col) for col in self.palette])

    @property
    def qcolors_w_nan(self):
        return np.array([QColor(*col) for col in self.palette]
                        + [QColor(*self.nan_color)])


class _QtIndexedPalette:
    def __getitem__(self, x):
        if isinstance(x, (Sequence, np.ndarray)):
            return self.values_to_qcolors(x)
        if isinstance(x, slice):
            return [QColor(*col) for col in self.palette.__getitem__(x)]
        return self.value_to_qcolor(x)


class _QtDiscretePalette:
    def values_to_qcolors(self, x):
        x, _ = self._color_indices(x)
        return self.qcolors_w_nan[x]

    def value_to_qcolor(self, x):
        color = self.nan_color if np.isnan(x) else self.palette[int(x)]
        return QColor(*color)


class _QtContinuousPalette:
    def values_to_qcolors(self, x, low=None, high=None):
        x, _ = self._color_indices(x, low, high)
        return self.qcolors_w_nan[x]

    def value_to_qcolor(self, x, low=0, high=1):
        if np.isnan(x):
            color = self.nan_color
        else:
            x = self._color_index(x, low, high)
            color = self.palette[x]
        return QColor(*color)

    __getitem__ = value_to_qcolor

    def color_strip(self, strip_length, strip_width, orientation=Qt.Horizontal):
        points = np.linspace(0, 255, strip_length, dtype=np.uint8)
        section = self.palette[np.newaxis, points, :].astype(np.ubyte)
        img = np.vstack((section,) * strip_width)
        if orientation == Qt.Horizontal:
            width, height = strip_length, strip_width
        else:
            width, height = strip_width, strip_length
            img = img.swapaxes(0, 1)[::-1].copy()
        pad_width = (-img.strides[1]) % 4
        if pad_width:
            padding = np.zeros((img.shape[0], pad_width, 3), np.ubyte)
            img = np.hstack((img, padding))
        img = QImage(img, width, height, img.strides[0], QImage.Format_RGB888)
        return QPixmap.fromImage(img)


class _QtBinnedContinuousPalette:
    def values_to_qcolors(self, x):
        binx, _ = self._bin_indices(x)
        return self.qcolors_w_nan[binx]

    def value_to_qcolor(self, x):
        if np.isnan(x):
            color = self.nan_color
        else:
            binx, _ = self._bin_indices([x])
            color = self.palette[binx[0]]
        return QColor(*color)


class Palette(_QtPalette, core.Palette):
    pass


class IndexedPalette(_QtIndexedPalette, core.IndexedPalette, Palette):
    pass


class DiscretePalette(_QtDiscretePalette, IndexedPalette, core.DiscretePalette):
    pass


class LimitedDiscretePalette(
        DiscretePalette, core.LimitedDiscretePalette):
    pass


class HuePalette(DiscretePalette, core.HuePalette):
    pass


class ContinuousPalette(
        _QtContinuousPalette, Palette, core.ContinuousPalette):
    pass


class BinnedContinuousPalette(
        _QtBinnedContinuousPalette, IndexedPalette,
        core.BinnedContinuousPalette):
    pass


def _adapt_palette(palette):
    if isinstance(palette, core.BinnedContinuousPalette):
        cls = BinnedContinuousPalette
        args = (palette.friendly_name, palette.name, palette.palette,
                palette.bins, palette.nan_color)
    elif isinstance(palette, core.ContinuousPalette):
        cls = ContinuousPalette
        args = (palette.friendly_name, palette.name, palette.palette,
                palette.nan_color)
    else:
        cls = DiscretePalette
        args = (palette.friendly_name, palette.name, palette.palette,
                palette.nan_color)
    return cls(*args, category=palette.category, flags=palette.flags)


DiscretePalettes = {
    name: _adapt_palette(palette)
    for name, palette in core.DiscretePalettes.items()
}
DefaultDiscretePaletteName = core.DefaultDiscretePaletteName
DefaultDiscretePalette = DiscretePalettes[DefaultDiscretePaletteName]
DefaultRGBColors = DiscretePalettes["default"]
Dark2Colors = DiscretePalettes["dark"]
Glasbey = DiscretePalettes["glasbey"]

ContinuousPalettes = {
    name: _adapt_palette(palette)
    for name, palette in core.ContinuousPalettes.items()
}
DefaultContinuousPaletteName = core.DefaultContinuousPaletteName
DefaultContinuousPalette = ContinuousPalettes[DefaultContinuousPaletteName]


class ColorIcon(QIcon):
    def __init__(self, color, size=12):
        pixmap = QPixmap(size, size)
        pixmap.fill(color)
        super().__init__(pixmap)


def get_default_curve_colors(n):
    if n <= len(Dark2Colors):
        return list(Dark2Colors)[:n]
    if n <= len(DefaultRGBColors):
        return list(DefaultRGBColors)[:n]
    return list(LimitedDiscretePalette(n))


def patch_variable_colors():
    core.patch_variable_colors_with_palettes(
        ContinuousPalettes, DefaultContinuousPalette, ContinuousPalette,
        DiscretePalettes, DefaultDiscretePalette, DiscretePalette,
        LimitedDiscretePalette)


NAN_COLOR = core.NAN_COLOR

__all__ = [
    "Palette", "IndexedPalette", "DiscretePalette", "LimitedDiscretePalette",
    "DiscretePalettes", "DefaultDiscretePalette", "DefaultDiscretePaletteName",
    "HuePalette", "DefaultRGBColors", "Dark2Colors", "Glasbey",
    "ContinuousPalette", "ContinuousPalettes", "BinnedContinuousPalette",
    "DefaultContinuousPalette", "DefaultContinuousPaletteName", "ColorIcon",
    "get_default_curve_colors", "patch_variable_colors", "NAN_COLOR",
]
