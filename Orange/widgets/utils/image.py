from typing import Sequence

import numpy as np

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QImage


def qimage_from_array(arr: np.ndarray) -> QImage:
    """
    Create and return an QImage from a (N, M, C) uint8 array where C is 3 for
    RGB and 4 for RGBA channels.

    Parameters
    ----------
    arr: (N, M C) uint8 array

    Returns
    -------
    image: QImage
        An QImage with size (M, N) in ARGB32 format format depending on `C`.
    """
    h, w, c = arr.shape
    if c == 4:
        format = QImage.Format_ARGB32
    elif c == 3:
        format = QImage.Format_RGB32
    else:
        raise ValueError(f"Wrong number of channels (need 3 or 4, got {c}")
    channels = arr.transpose((2, 0, 1))
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(Qt.white)
    if img.size().isEmpty():
        return img
    buffer = img.bits().asarray(w * h * 4)
    view = np.frombuffer(buffer, np.uint32).reshape((h, w))
    if format == QImage.Format_ARGB32:
        view[:, :] = qrgba(*channels)
    elif format == QImage.Format_RGB32:
        view[:, :] = qrgb(*channels)
    return img


def qimage_indexed_from_array(
        arr: np.ndarray, colortable: Sequence[Sequence[int]]
) -> QImage:
    arr = np.asarray(arr, dtype=np.uint8)
    h, w = arr.shape
    colortable = np.asarray(colortable, dtype=np.uint8)
    ncolors, nchannels = colortable.shape
    img = QImage(w, h, QImage.Format_Indexed8)
    img.setColorCount(ncolors)
    if nchannels == 4:
        qrgb_ = qrgba
    elif nchannels == 3:
        qrgb_ = qrgb
    else:
        raise ValueError

    for i, c in enumerate(colortable):
        img.setColor(i, qrgb_(*c))
    if img.size().isEmpty():
        return img
    buffer = img.bits().asarray(w * h)
    view = np.frombuffer(buffer, np.uint8).reshape((h, w))
    view[:, :] = arr
    return img


def qrgb(
        r: Sequence[int], g: Sequence[int], b: Sequence[int]
) -> Sequence[int]:
    """A vectorized `qRgb`."""
    r, g, b = map(lambda a: np.asarray(a, dtype=np.uint32), (r, g, b))
    return (0xff << 24) | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff)


def qrgba(
        r: Sequence[int], g: Sequence[int], b: Sequence[int], a: Sequence[int]
) -> Sequence[int]:
    """A vectorized `qRgba`."""
    r, g, b, a = map(lambda a: np.asarray(a, dtype=np.uint32), (r, g, b, a))
    return ((a & 0xff) << 24) | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff)
