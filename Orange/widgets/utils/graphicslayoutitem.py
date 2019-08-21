from typing import Optional

from AnyQt.QtCore import Qt, QPointF, QRectF, QSizeF, QMarginsF
from AnyQt.QtWidgets import QGraphicsItem, QGraphicsLayoutItem, QSizePolicy

__all__ = [
    'SimpleLayoutItem'
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
        "item"
    )

    def __init__(
            self,
            item: QGraphicsItem,
            parent: Optional[QGraphicsLayoutItem] = None,
            anchor=(0., 0.),
            anchorItem=(0., 0.),
            **kwargs
    ) -> None:
        sizePolicy: Optional[QSizePolicy] = kwargs.pop("sizePolicy", None)
        super().__init__(parent, **kwargs)
        self.__anchorThis = anchor
        self.__anchorItem = anchorItem
        self.item = item
        if sizePolicy is not None:
            self.setSizePolicy(sizePolicy)
        self.__layout()

    def setGeometry(self, rect: QRectF) -> None:
        super().setGeometry(rect)
        self.__layout()

    def sizeHint(self, which: Qt.SizeHint, constraint=QSizeF(-1, -1)) -> QSizeF:
        if which == Qt.PreferredSize:
            brect = self.item.boundingRect()
            brect = self.item.mapRectToParent(brect)
            return brect.size()
        else:
            return QSizeF()

    def updateGeometry(self):
        super().updateGeometry()
        parent = self.parentLayoutItem()
        if parent is not None:
            parent.updateGeometry()

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
