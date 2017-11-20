import numpy as np

from AnyQt.QtWidgets import QGraphicsLineItem
from AnyQt.QtCore import QRectF, QLineF
from AnyQt.QtGui import QTransform

import pyqtgraph as pg


class TextItem(pg.TextItem):
    if not hasattr(pg.TextItem, "setAnchor"):
        # Compatibility with pyqtgraph <= 0.9.10; in (as of yet unreleased)
        # 0.9.11 the TextItem has a `setAnchor`, but not `updateText`
        def setAnchor(self, anchor):
            self.anchor = pg.Point(anchor)
            self.updateText()


class AnchorItem(pg.GraphicsObject):
    def __init__(self, parent=None, line=QLineF(), text="", **kwargs):
        super().__init__(parent, **kwargs)
        self._text = text
        self.setFlag(pg.GraphicsObject.ItemHasNoContents)

        self._spine = QGraphicsLineItem(line, self)
        angle = line.angle()

        self._arrow = pg.ArrowItem(parent=self, angle=0)
        self._arrow.setPos(self._spine.line().p2())
        self._arrow.setRotation(angle)
        self._arrow.setStyle(headLen=10)

        self._label = TextItem(text=text, color=(10, 10, 10))
        self._label.setParentItem(self)
        self._label.setPos(self._spine.line().p2())

        if parent is not None:
            self.setParentItem(parent)

    def setText(self, text):
        if text != self._text:
            self._text = text
            self._label.setText(text)
            self._label.setVisible(bool(text))

    def text(self):
        return self._text

    def setLine(self, *line):
        line = QLineF(*line)
        if line != self._spine.line():
            self._spine.setLine(line)
            self.__updateLayout()

    def line(self):
        return self._spine.line()

    def setPen(self, pen):
        self._spine.setPen(pen)

    def setArrowVisible(self, visible):
        self._arrow.setVisible(visible)

    def paint(self, painter, option, widget):
        pass

    def boundingRect(self):
        return QRectF()

    def viewTransformChanged(self):
        self.__updateLayout()

    def __updateLayout(self):
        T = self.sceneTransform()
        if T is None:
            T = QTransform()

        # map the axis spine to scene coord. system.
        viewbox_line = T.map(self._spine.line())
        angle = viewbox_line.angle()
        assert not np.isnan(angle)
        # note in Qt the y axis is inverted (90 degree angle 'points' down)
        left_quad = 270 < angle <= 360 or -0.0 <= angle < 90

        # position the text label along the viewbox_line
        label_pos = self._spine.line().pointAt(0.90)

        if left_quad:
            # Anchor the text under the axis spine
            anchor = (0.5, -0.1)
        else:
            # Anchor the text over the axis spine
            anchor = (0.5, 1.1)

        self._label.setPos(label_pos)
        self._label.setAnchor(pg.Point(*anchor))
        self._label.setRotation(-angle if left_quad else 180 - angle)

        self._arrow.setPos(self._spine.line().p2())
        self._arrow.setRotation(180 - angle)
