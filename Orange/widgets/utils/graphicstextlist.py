from AnyQt.QtCore import Qt, QSizeF, QEvent
from AnyQt.QtGui import QFontMetrics
from AnyQt.QtWidgets import (
    QGraphicsWidget, QSizePolicy, QGraphicsItemGroup, QGraphicsSimpleTextItem,
    QWIDGETSIZE_MAX,
)

__all__ = ["TextListWidget"]


class TextListWidget(QGraphicsWidget):
    def __init__(self, parent=None, items=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setFlag(QGraphicsWidget.ItemClipsChildrenToShape, True)
        self.__items = []
        self.__textitems = []
        self.__group = None
        self.__spacing = 0

        sp = QSizePolicy(QSizePolicy.Preferred,
                         QSizePolicy.Preferred)
        sp.setWidthForHeight(True)
        self.setSizePolicy(sp)

        if items is not None:
            self.setItems(items)

    def setItems(self, items):
        self.__clear()
        self.__items = list(items)
        self.__setup()
        self.__layout()
        self.updateGeometry()

    def clear(self):
        self.__clear()
        self.__items = []
        self.updateGeometry()

    def sizeHint(self, which, constraint=QSizeF()):
        if which == Qt.PreferredSize:
            sh = self.__naturalsh()
            if 0 < constraint.height() < sh.height():
                sh = scaled(sh, constraint, Qt.KeepAspectRatioByExpanding)
            return sh

        return super().sizeHint(which, constraint)

    def __naturalsh(self):
        fm = QFontMetrics(self.font())
        spacing = self.__spacing
        N = len(self.__items)
        width = max((fm.width(text) for text in self.__items),
                    default=0)
        height = N * fm.height() + (N - 1) * spacing
        return QSizeF(width, height)

    def event(self, event):
        if event.type() == QEvent.LayoutRequest:
            self.__layout()
        elif event.type() == QEvent.GraphicsSceneResize:
            self.__layout()

        return super().event(event)

    def changeEvent(self, event):
        if event.type() == QEvent.FontChange:
            self.updateGeometry()
            font = self.font()
            for item in self.__textitems:
                item.setFont(font)

    def __setup(self):
        self.__clear()
        font = self.font()
        group = QGraphicsItemGroup(self)

        for text in self.__items:
            t = QGraphicsSimpleTextItem(text, group)
            t.setData(0, text)
            t.setFont(font)
            t.setToolTip(text)
            self.__textitems.append(t)

    def __layout(self):
        crect = self.contentsRect()
        spacing = self.__spacing
        N = len(self.__items)

        if not N:
            return

        fm = QFontMetrics(self.font())
        naturalheight = fm.height()
        th = (crect.height() - (N - 1) * spacing) / N
        if th > naturalheight and N > 1:
            th = naturalheight
            spacing = (crect.height() - N * th) / (N - 1)

        for i, item in enumerate(self.__textitems):
            item.setPos(crect.left(), crect.top() + i * (th + spacing))

    def __clear(self):
        def remove(items, scene):
            for item in items:
                item.setParentItem(None)
                if scene is not None:
                    scene.removeItem(item)

        remove(self.__textitems, self.scene())
        if self.__group is not None:
            remove([self.__group], self.scene())

        self.__textitems = []


def scaled(size, constraint, mode=Qt.KeepAspectRatio):
    if constraint.width() < 0 and constraint.height() < 0:
        return size

    size, constraint = QSizeF(size), QSizeF(constraint)
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
