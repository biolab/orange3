from functools import reduce
from types import SimpleNamespace
from typing import Optional, List, Iterable, Tuple

import numpy as np

from AnyQt.QtCore import QRectF, QSizeF, Qt, QPointF, QMarginsF
from AnyQt.QtWidgets import QGraphicsLayout, QGraphicsLayoutItem
from AnyQt import sip

FLT_MAX = np.finfo(np.float32).max


class _FlowLayoutItem(SimpleNamespace):
    item: QGraphicsLayoutItem
    geom: QRectF
    size: QSizeF
    row: int = 0
    alignment: Qt.Alignment = 0


class GraphicsFlowLayout(QGraphicsLayout):
    def __init__(self, parent: Optional[QGraphicsLayoutItem] = None):
        self.__items: List[QGraphicsLayoutItem] = []
        self.__spacing: Tuple[float, float] = (1., 1.)
        super().__init__(parent)
        sp = self.sizePolicy()
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)

    def setVerticalSpacing(self, spacing: float) -> None:
        new = (self.__spacing[0], spacing)
        if new != self.__spacing:
            self.__spacing = new
            self.invalidate()

    def verticalSpacing(self) -> float:
        return self.__spacing[1]

    def setHorizontalSpacing(self, spacing: float) -> None:
        new = (spacing, self.__spacing[1])
        if new != self.__spacing:
            self.__spacing = new
            self.invalidate()

    def horizontalSpacing(self) -> float:
        return self.__spacing[0]

    def setGeometry(self, rect: QRectF) -> None:
        super().setGeometry(rect)
        margins = QMarginsF(*self.getContentsMargins())
        rect = rect.marginsRemoved(margins)
        for item, r in zip(self.__items, self.__doLayout(rect)):
            item.setGeometry(r)

    def invalidate(self) -> None:
        self.updateGeometry()
        super().invalidate()

    def __doLayout(self, rect: QRectF) -> Iterable[QRectF]:
        x = y = 0
        rowheight = 0
        width = rect.width()
        spacing_x, spacing_y = self.__spacing
        first_in_row = True
        rows: List[List[QRectF]] = [[]]

        def break_():
            nonlocal x, y, rowheight, first_in_row
            y += rowheight + spacing_y
            x = 0
            rowheight = 0
            first_in_row = True
            rows.append([])

        items = [_FlowLayoutItem(item=item, geom=QRectF(), size=QSizeF())
                 for item in self.__items]

        for flitem in items:
            item = flitem.item
            sh = item.effectiveSizeHint(Qt.PreferredSize)
            if x + sh.width() > width and not first_in_row:
                break_()
            r = QRectF(rect.x() + x, rect.y() + y, sh.width(), sh.height())
            flitem.geom = r
            flitem.size = sh
            flitem.row = len(rows) - 1
            rowheight = max(rowheight, sh.height())
            x += sh.width() + spacing_x
            first_in_row = False
            rows[-1].append(flitem.geom)

        alignment = Qt.AlignVCenter | Qt.AlignLeft
        for flitem in items:
            row = rows[flitem.row]
            row_rect = reduce(QRectF.united, row, QRectF())
            if row_rect.isEmpty():
                continue
            flitem.geom = qrect_aligned_to(
                flitem.geom, row_rect, alignment & Qt.AlignVertical_Mask)
        return [fli.geom for fli in items]

    def sizeHint(self, which: Qt.SizeHint, constraint=QSizeF(-1, -1)) -> QSizeF:
        left, top, right, bottom = self.getContentsMargins()
        extra_margins = QSizeF(left + right, top + bottom)
        if constraint.width() >= 0:
            constraint.setWidth(
                max(constraint.width() - extra_margins.width(), 0.0))

        if which == Qt.PreferredSize:
            if constraint.width() >= 0:
                rect = QRectF(0, 0, constraint.width(), FLT_MAX)
            else:
                rect = QRectF(0, 0, FLT_MAX, FLT_MAX)
            res = self.__doLayout(rect)
            sh = reduce(QRectF.united, res, QRectF()).size()
            return sh + extra_margins
        if which == Qt.MinimumSize:
            return reduce(QSizeF.expandedTo,
                          (item.minimumSize() for item in self.__items),
                          QSizeF()) + extra_margins
        return QSizeF()

    def count(self) -> int:
        return len(self.__items)

    def itemAt(self, i: int) -> QGraphicsLayoutItem:
        try:
            return self.__items[i]
        except IndexError:
            return None  # type: ignore

    def removeAt(self, index: int) -> None:
        try:
            item = self.__items.pop(index)
        except IndexError:
            pass
        else:
            item.setParentLayoutItem(None)
            self.invalidate()

    def removeItem(self, item: QGraphicsLayoutItem):
        try:
            self.__items.remove(item)
        except ValueError:
            pass
        else:
            item.setParentLayoutItem(None)
            self.invalidate()

    def addItem(self, item: QGraphicsLayoutItem) -> None:
        self.insertItem(self.count(), item)

    def insertItem(self, index: int, item: QGraphicsLayoutItem, ) -> None:
        self.addChildLayoutItem(item)
        if 0 <= index < self.count():
            self.__items.insert(index, item)
        else:
            self.__items.append(item)
        self.updateGeometry()
        self.invalidate()

    def __dtor__(self):
        items = self.__items
        self.__items = []
        for item in items:
            item.setParentLayoutItem(None)
            if item.ownedByLayout():
                sip.delete(item)


def qrect_aligned_to(
        rect_a: QRectF, rect_b: QRectF, alignment: Qt.Alignment) -> QRectF:
    res = QRectF(rect_a)
    valign = alignment & Qt.AlignVertical_Mask
    halign = alignment & Qt.AlignHorizontal_Mask
    if valign == Qt.AlignTop:
        res.moveTop(rect_b.top())
    if valign == Qt.AlignVCenter:
        res.moveCenter(QPointF(res.center().x(), rect_b.center().y()))
    if valign == Qt.AlignBottom:
        res.moveBottom(rect_b.bottom())

    if halign == Qt.AlignLeft:
        res.moveLeft(rect_b.left())
    if halign == Qt.AlignHCenter:
        res.moveCenter(QPointF(rect_b.center().x(), res.center().y()))
    if halign == Qt.AlignRight:
        res.moveRight(rect_b.right())
    return res
