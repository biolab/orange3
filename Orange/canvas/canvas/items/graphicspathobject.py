from PyQt4.QtGui import QGraphicsObject, QPainterPath, QPainterPathStroker, \
                        QBrush, QPen

from PyQt4.QtCore import Qt, QPointF, pyqtSignal as Signal


class GraphicsPathObject(QGraphicsObject):
    """A QGraphicsObject subclass implementing an interface similar to
    QGraphicsPathItem, and also adding a positionChanged() signal

    """

    positionChanged = Signal([], ["QPointF"])

    def __init__(self, parent=None, **kwargs):
        QGraphicsObject.__init__(self, parent, **kwargs)
        self.setFlag(QGraphicsObject.ItemSendsGeometryChanges)

        self.__path = QPainterPath()
        self.__brush = QBrush(Qt.NoBrush)
        self.__pen = QPen()
        self.__boundingRect = None

    def setPath(self, path):
        """Set the items `path` (:class:`QPainterPath`).
        """
        if not isinstance(path, QPainterPath):
            raise TypeError("%r, 'QPainterPath' expected" % type(path))

        if self.__path != path:
            self.prepareGeometryChange()
            # Need to store a copy of object so the shape can't be mutated
            # without properly updating the geometry.
            self.__path = QPainterPath(path)
            self.__boundingRect = None
            self.update()

    def path(self):
        """Return the items path.
        """
        return QPainterPath(self.__path)

    def setBrush(self, brush):
        """Set the items `brush` (:class:`QBrush`)
        """
        if not isinstance(brush, QBrush):
            brush = QBrush(brush)

        if self.__brush != brush:
            self.__brush = QBrush(brush)
            self.update()

    def brush(self):
        """Return the items brush.
        """
        return QBrush(self.__brush)

    def setPen(self, pen):
        """Set the items outline `pen` (:class:`QPen`).
        """
        if not isinstance(pen, QPen):
            pen = QPen(pen)

        if self.__pen != pen:
            self.prepareGeometryChange()
            self.__pen = QPen(pen)
            self.__boundingRect = None
            self.update()

    def pen(self):
        """Return the items pen.
        """
        return QPen(self.__pen)

    def paint(self, painter, option, widget=None):
        if self.__path.isEmpty():
            return

        painter.save()
        painter.setPen(self.__pen)
        painter.setBrush(self.__brush)
        painter.drawPath(self.__path)
        painter.restore()

    def boundingRect(self):
        if self.__boundingRect is None:
            br = self.__path.controlPointRect()
            pen_w = self.__pen.widthF()
            self.__boundingRect = br.adjusted(-pen_w, -pen_w, pen_w, pen_w)

        return self.__boundingRect

    def shape(self):
        return shapeFromPath(self.__path, self.__pen)

    def itemChange(self, change, value):
        if change == QGraphicsObject.ItemPositionHasChanged:
            self.positionChanged.emit()
            self.positionChanged[QPointF].emit(value.toPyObject())

        return QGraphicsObject.itemChange(self, change, value)


def shapeFromPath(path, pen):
    """Create a QPainterPath shape from the `path` drawn with `pen`.
    """
    stroker = QPainterPathStroker()
    stroker.setCapStyle(pen.capStyle())
    stroker.setJoinStyle(pen.joinStyle())
    stroker.setMiterLimit(pen.miterLimit())
    stroker.setWidth(max(pen.widthF(), 1e-9))

    shape = stroker.createStroke(path)
    shape.addPath(path)

    return shape
