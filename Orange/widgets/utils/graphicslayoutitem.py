from fractions import Fraction
from typing import Optional

from AnyQt.QtCore import Qt, QPointF, QRectF, QSizeF, QMarginsF
from AnyQt.QtWidgets import (
    QGraphicsItem, QGraphicsLayoutItem, QSizePolicy, QGraphicsScale,
    QWIDGETSIZE_MAX,
)

__all__ = [
    'SimpleLayoutItem', 'scaled'
]


class SimpleLayoutItem(QGraphicsLayoutItem):
    """
    A graphics layout item wrapping a QGraphicsItem instance to be
    managed by a layout.

    The item is positioned at the layout geometry top left corner and its
    boundingRect().size() is used as the preferred size hint

    Parameters
    ----------
    item: QGraphicsItem
    parent: Optional[QGraphicsLayoutItem]
        The parent layout item.
    anchor: Tuple[float, float]
        The anchor in this layout item's geometry relative coord. system
        (0, 0) corresponds to top left corner and (1, 1) corresponds to
        bottom right corner).
    anchorItem: Tuple[float, float]
        The relative anchor in `item` 's bounding rect.
    """
    __slots__ = (
        "__anchorThis",
        "__anchorItem",
        "item",
        "__resizeContents",
        "__aspectMode",
        "__transform",
        "__scale",
    )

    def __init__(
            self,
            item: QGraphicsItem,
            parent: Optional[QGraphicsLayoutItem] = None,
            anchor=(0., 0.),
            anchorItem=(0., 0.),
            resizeContents=False,
            aspectMode=Qt.IgnoreAspectRatio,
            **kwargs
    ) -> None:
        sizePolicy: Optional[QSizePolicy] = kwargs.pop("sizePolicy", None)
        super().__init__(parent, **kwargs)
        self.__anchorThis = anchor
        self.__anchorItem = anchorItem
        self.__resizeContents = resizeContents
        self.__aspectMode = aspectMode
        self.__transform = None  # type: Optional[QGraphicsScale]
        self.__scale = (Fraction(1), Fraction(1))

        if resizeContents:
            self.__transform = QGraphicsScale()
            trs = item.transformations()
            item.setTransformations(trs + [self.__transform])
        self.item = item
        self.setGraphicsItem(item)
        if sizePolicy is not None:
            self.setSizePolicy(sizePolicy)
        self.__layout()

    def setGeometry(self, rect: QRectF) -> None:
        resized = rect.size() != self.geometry()
        super().setGeometry(rect)
        if resized and self.__resizeContents:
            self.__updateScale()
        self.__layout()

    def sizeHint(self, which: Qt.SizeHint, constraint=QSizeF(-1, -1)) -> QSizeF:
        if which == Qt.PreferredSize:
            brect = self.item.boundingRect()
            brect = self.item.mapRectToParent(brect)
            scale = self.__transform
            size = brect.size()
            if scale is not None:
                # undo the scaling
                sx, sy = self.__scale
                size = QSizeF(float(Fraction(size.width()) / sx),
                              float(Fraction(size.height()) / sy))
            if constraint != QSizeF(-1, -1):
                size = scaled(size, constraint, self.__aspectMode)
            return size
        else:
            return QSizeF()

    def updateGeometry(self):
        super().updateGeometry()
        parent = self.parentLayoutItem()
        if parent is not None:
            parent.updateGeometry()

    def __updateScale(self):
        if self.__transform is None:
            return
        geom = self.geometry()
        if geom.size().isEmpty():
            return
        itemsize = self.sizeHint(Qt.PreferredSize)
        scaledsize = scaled(itemsize, geom.size(), self.__aspectMode)
        if not itemsize.isEmpty():
            sx = Fraction(scaledsize.width()) / Fraction(itemsize.width())
            sy = Fraction(scaledsize.height()) / Fraction(itemsize.height())
        else:
            sx = sy = Fraction(1)
        self.__scale = (sx, sy)
        self.__transform.setXScale(float(sx))
        self.__transform.setYScale(float(sy))

    def __layout(self):
        item = self.item
        geom = self.geometry()
        margins = QMarginsF(*self.getContentsMargins())
        crect = geom.marginsRemoved(margins)
        anchorpos = qrect_pos_relative(crect, *self.__anchorThis)
        brect = self.item.boundingRect()
        anchorpositem = qrect_pos_relative(brect, *self.__anchorItem)
        anchorpositem = item.mapToParent(anchorpositem)
        item.setPos(item.pos() + (anchorpos - anchorpositem))


def qrect_pos_relative(rect: QRectF, rx: float, ry: float) -> QPointF:
    return QPointF(rect.x() + rect.width() * rx, rect.y() + rect.height() * ry)


def scaled(size: QSizeF, constraint: QSizeF, mode=Qt.KeepAspectRatio) -> QSizeF:
    """
    Return size scaled to fit in the constrains using the aspect mode `mode`.

    If  width or height of constraint are negative they are ignored,
    ie. the result is not constrained in that dimension.
    """
    size, constraint = QSizeF(size), QSizeF(constraint)
    if size.isEmpty():
        return size
    if constraint.width() < 0 and constraint.height() < 0:
        return size
    if mode == Qt.IgnoreAspectRatio:
        if constraint.width() >= 0:
            size.setWidth(constraint.width())
        if constraint.height() >= 0:
            size.setHeight(constraint.height())
    elif mode == Qt.KeepAspectRatio:
        if constraint.width() < 0:
            constraint.setWidth(QWIDGETSIZE_MAX)
        if constraint.height() < 0:
            constraint.setHeight(QWIDGETSIZE_MAX)
        size.scale(constraint, mode)
    elif mode == Qt.KeepAspectRatioByExpanding:
        if constraint.width() < 0:
            constraint.setWidth(0)
        if constraint.height() < 0:
            constraint.setHeight(0)
        size.scale(constraint, mode)
    return size
