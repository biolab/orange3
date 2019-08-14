from typing import Optional

from AnyQt.QtCore import Qt, QSizeF, QRectF, QPointF
from AnyQt.QtGui import QPixmap, QTransform, QPainter
from AnyQt.QtWidgets import (
    QGraphicsWidget, QGraphicsItem, QStyleOptionGraphicsItem, QWidget,
)
from Orange.widgets.utils.graphicslayoutitem import scaled


class GraphicsPixmapWidget(QGraphicsWidget):
    def __init__(
            self,
            parent: Optional[QGraphicsItem] = None,
            pixmap: Optional[QPixmap] = None,
            scaleContents=False,
            aspectMode=Qt.KeepAspectRatio,
            **kwargs
    ) -> None:
        self.__scaleContents = scaleContents
        self.__aspectMode = aspectMode
        self.__pixmap = QPixmap(pixmap) if pixmap is not None else QPixmap()
        super().__init__(None, **kwargs)
        self.setFlag(QGraphicsWidget.ItemUsesExtendedStyleOption, True)
        self.setContentsMargins(0, 0, 0, 0)
        if parent is not None:
            self.setParentItem(parent)

    def setPixmap(self, pixmap: QPixmap) -> None:
        self.prepareGeometryChange()
        self.__pixmap = QPixmap(pixmap)
        self.updateGeometry()

    def pixmap(self) -> QPixmap:
        return QPixmap(self.__pixmap)

    def setAspectRatioMode(self, mode: Qt.AspectRatioMode) -> None:
        if self.__aspectMode != mode:
            self.__aspectMode = mode
            sp = self.sizePolicy()
            sp.setHeightForWidth(
                self.__aspectMode != Qt.IgnoreAspectRatio and self.__scaleContents
            )
            self.setSizePolicy(sp)
            self.updateGeometry()

    def aspectRatioMode(self) -> Qt.AspectRatioMode:
        return self.__aspectMode

    def setScaleContents(self, scale: bool) -> None:
        if self.__scaleContents != scale:
            self.__scaleContents = bool(scale)
            sp = self.sizePolicy()
            sp.setHeightForWidth(
                self.__aspectMode != Qt.IgnoreAspectRatio and self.__scaleContents
            )
            self.setSizePolicy(sp)
            self.updateGeometry()

    def scaleContents(self) -> bool:
        return self.__scaleContents

    def sizeHint(self, which, constraint=QSizeF(-1, -1)) -> QSizeF:
        if which == Qt.PreferredSize:
            sh = QSizeF(self.__pixmap.size())
            if self.__scaleContents:
                sh = scaled(sh, constraint, self.__aspectMode)
            return sh
        elif which == Qt.MinimumSize:
            if self.__scaleContents:
                return QSizeF(0, 0)
            else:
                return QSizeF(self.__pixmap.size())
        elif which == Qt.MaximumSize:
            if self.__scaleContents:
                return QSizeF()
            else:
                return QSizeF(self.__pixmap.size())
        else:
            # Qt.MinimumDescent
            return QSizeF()

    def pixmapTransform(self) -> QTransform:
        if self.__pixmap.isNull():
            return QTransform()

        pxsize = QSizeF(self.__pixmap.size())
        crect = self.contentsRect()
        transform = QTransform()
        transform = transform.translate(crect.left(), crect.top())

        if self.__scaleContents:
            csize = scaled(pxsize, crect.size(), self.__aspectMode)
        else:
            csize = pxsize

        xscale = csize.width() / pxsize.width()
        yscale = csize.height() / pxsize.height()

        return transform.scale(xscale, yscale)

    def paint(
            self, painter: QPainter, option: QStyleOptionGraphicsItem,
            widget: Optional[QWidget] = None
    ) -> None:
        if self.__pixmap.isNull():
            return
        pixmap = self.__pixmap
        crect = self.contentsRect()

        exposed = option.exposedRect
        exposedcrect = crect.intersected(exposed)
        pixmaptransform = self.pixmapTransform()
        # map exposed rect to exposed pixmap coords
        assert pixmaptransform.type() <= QTransform.TxRotate
        pixmaptransform, ok = pixmaptransform.inverted()
        if not ok:
            painter.drawPixmap(
                crect, pixmap, QRectF(QPointF(0, 0), QSizeF(pixmap.size()))
            )
        else:
            exposedpixmap = pixmaptransform.mapRect(exposed)
            painter.drawPixmap(exposedcrect, pixmap, exposedpixmap)
