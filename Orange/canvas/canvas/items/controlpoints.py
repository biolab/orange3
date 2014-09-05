import logging

from PyQt4.QtGui import QGraphicsItem, QGraphicsObject, QBrush, QPainterPath
from PyQt4.QtCore import Qt, QPointF, QLineF, QRectF, QMargins, QEvent

from PyQt4.QtCore import pyqtSignal as Signal, pyqtProperty as Property

from .graphicspathobject import GraphicsPathObject
from .utils import toGraphicsObjectIfPossible

log = logging.getLogger(__name__)


class ControlPoint(GraphicsPathObject):
    """A control point for annotations in the canvas.
    """
    Free = 0

    Left, Top, Right, Bottom, Center = 1, 2, 4, 8, 16

    TopLeft = Top | Left
    TopRight = Top | Right
    BottomRight = Bottom | Right
    BottomLeft = Bottom | Left

    def __init__(self, parent=None, anchor=0, **kwargs):
        GraphicsPathObject.__init__(self, parent, **kwargs)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, False)
        self.setAcceptedMouseButtons(Qt.LeftButton)

        self.__constraint = 0
        self.__constraintFunc = None
        self.__anchor = 0
        self.setAnchor(anchor)

        path = QPainterPath()
        path.addEllipse(QRectF(-4, -4, 8, 8))
        self.setPath(path)

        self.setBrush(QBrush(Qt.lightGray, Qt.SolidPattern))

    def setAnchor(self, anchor):
        """Set anchor position
        """
        self.__anchor = anchor

    def anchor(self):
        return self.__anchor

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Enable ItemPositionChange (and pos constraint) only when
            # this is the mouse grabber item
            self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        return GraphicsPathObject.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, False)
        return GraphicsPathObject.mouseReleaseEvent(self, event)

    def itemChange(self, change, value):

        if change == QGraphicsItem.ItemPositionChange:
            newpos = self.constrain(value)
            return newpos

        return GraphicsPathObject.itemChange(self, change, value)

    def hasConstraint(self):
        return self.__constraintFunc is not None or self.__constraint != 0

    def setConstraint(self, constraint):
        """Set the constraint for the point (Qt.Vertical Qt.Horizontal or 0)

        .. note:: Clears the constraintFunc if it was previously set

        """
        if self.__constraint != constraint:
            self.__constraint = constraint

        self.__constraintFunc = None

    def constrain(self, pos):
        """Constrain the pos.
        """
        if self.__constraintFunc:
            return self.__constraintFunc(pos)
        elif self.__constraint == Qt.Vertical:
            return QPointF(self.pos().x(), pos.y())
        elif self.__constraint == Qt.Horizontal:
            return QPointF(pos.x(), self.pos().y())
        else:
            return pos

    def setConstraintFunc(self, func):
        if self.__constraintFunc != func:
            self.__constraintFunc = func


class ControlPointRect(QGraphicsObject):
    Free = 0
    KeepAspectRatio = 1
    KeepCenter = 2

    rectChanged = Signal(QRectF)
    rectEdited = Signal(QRectF)

    def __init__(self, parent=None, rect=None, constraints=0, **kwargs):
        QGraphicsObject.__init__(self, parent, **kwargs)
        self.setFlag(QGraphicsItem.ItemHasNoContents)
        self.setFlag(QGraphicsItem.ItemIsFocusable)

        self.__rect = rect if rect is not None else QRectF()
        self.__margins = QMargins()
        points = \
            [ControlPoint(self, ControlPoint.Left),
             ControlPoint(self, ControlPoint.Top),
             ControlPoint(self, ControlPoint.TopLeft),
             ControlPoint(self, ControlPoint.Right),
             ControlPoint(self, ControlPoint.TopRight),
             ControlPoint(self, ControlPoint.Bottom),
             ControlPoint(self, ControlPoint.BottomLeft),
             ControlPoint(self, ControlPoint.BottomRight)
             ]
        assert(points == sorted(points, key=lambda p: p.anchor()))

        self.__points = dict((p.anchor(), p) for p in points)

        if self.scene():
            self.__installFilter()

        for p in points:
            p.setFlag(QGraphicsItem.ItemIsFocusable)
            p.setFocusProxy(self)

        self.controlPoint(ControlPoint.Top).setConstraint(Qt.Vertical)
        self.controlPoint(ControlPoint.Bottom).setConstraint(Qt.Vertical)
        self.controlPoint(ControlPoint.Left).setConstraint(Qt.Horizontal)
        self.controlPoint(ControlPoint.Right).setConstraint(Qt.Horizontal)

        self.__constraints = constraints
        self.__activeControl = None

        self.__pointsLayout()

    def controlPoint(self, anchor):
        """
        Return the anchor point (:class:`ControlPoint`) at anchor position
        or `None` if an anchor point is not set.

        """
        return self.__points.get(anchor)

    def setRect(self, rect):
        """
        Set the control point rectangle (:class:`QRectF`)
        """
        if self.__rect != rect:
            self.__rect = QRectF(rect)
            self.__pointsLayout()
            self.prepareGeometryChange()
            self.rectChanged.emit(rect.normalized())

    def rect(self):
        """
        Return the control point rectangle.
        """
        # Return the rect normalized. During the control point move the
        # rect can change to an invalid size, but the layout must still
        # know to which point does an unnormalized rect side belong,
        # so __rect is left unnormalized.
        # NOTE: This means all signal emits (rectChanged/Edited) must
        #       also emit normalized rects
        return self.__rect.normalized()

    rect_ = Property(QRectF, fget=rect, fset=setRect, user=True)

    def setControlMargins(self, *margins):
        """Set the controls points on the margins around `rect`
        """
        if len(margins) > 1:
            margins = QMargins(*margins)
        else:
            margins = margins[0]
            if isinstance(margins, int):
                margins = QMargins(margins, margins, margins, margins)

        if self.__margins != margins:
            self.__margins = margins
            self.__pointsLayout()

    def controlMargins(self):
        return self.__margins

    def setConstraints(self, constraints):
        raise NotImplementedError

    def isControlActive(self):
        """Return the state of the control. True if the control is
        active (user is dragging one of the points) False otherwise.

        """
        return self.__activeControl is not None

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSceneHasChanged and self.scene():
            self.__installFilter()

        return QGraphicsObject.itemChange(self, change, value)

    def sceneEventFilter(self, obj, event):
        try:
            obj = toGraphicsObjectIfPossible(obj)
            if isinstance(obj, ControlPoint):
                etype = event.type()
                if etype == QEvent.GraphicsSceneMousePress and \
                        event.button() == Qt.LeftButton:
                    self.__setActiveControl(obj)

                elif etype == QEvent.GraphicsSceneMouseRelease and \
                        event.button() == Qt.LeftButton:
                    self.__setActiveControl(None)

        except Exception:
            log.error("Error in 'ControlPointRect.sceneEventFilter'",
                      exc_info=True)

        return QGraphicsObject.sceneEventFilter(self, obj, event)

    def __installFilter(self):
        # Install filters on the control points.
        try:
            for p in self.__points.values():
                p.installSceneEventFilter(self)
        except Exception:
            log.error("Error in ControlPointRect.__installFilter",
                      exc_info=True)

    def __pointsLayout(self):
        """Layout the control points
        """
        rect = self.__rect
        margins = self.__margins
        rect = rect.adjusted(-margins.left(), -margins.top(),
                             margins.right(), margins.bottom())
        center = rect.center()
        cx, cy = center.x(), center.y()
        left, top, right, bottom = \
                rect.left(), rect.top(), rect.right(), rect.bottom()

        self.controlPoint(ControlPoint.Left).setPos(left, cy)
        self.controlPoint(ControlPoint.Right).setPos(right, cy)
        self.controlPoint(ControlPoint.Top).setPos(cx, top)
        self.controlPoint(ControlPoint.Bottom).setPos(cx, bottom)

        self.controlPoint(ControlPoint.TopLeft).setPos(left, top)
        self.controlPoint(ControlPoint.TopRight).setPos(right, top)
        self.controlPoint(ControlPoint.BottomLeft).setPos(left, bottom)
        self.controlPoint(ControlPoint.BottomRight).setPos(right, bottom)

    def __setActiveControl(self, control):
        if self.__activeControl != control:
            if self.__activeControl is not None:
                self.__activeControl.positionChanged[QPointF].disconnect(
                    self.__activeControlMoved
                )

            self.__activeControl = control

            if control is not None:
                control.positionChanged[QPointF].connect(
                    self.__activeControlMoved
                )

    def __activeControlMoved(self, pos):
        # The active control point has moved, update the control
        # rectangle
        control = self.__activeControl
        pos = control.pos()
        rect = QRectF(self.__rect)
        margins = self.__margins

        # TODO: keyboard modifiers and constraints.

        anchor = control.anchor()
        if anchor & ControlPoint.Top:
            rect.setTop(pos.y() + margins.top())
        elif anchor & ControlPoint.Bottom:
            rect.setBottom(pos.y() - margins.bottom())

        if anchor & ControlPoint.Left:
            rect.setLeft(pos.x() + margins.left())
        elif anchor & ControlPoint.Right:
            rect.setRight(pos.x() - margins.right())

        changed = self.__rect != rect

        self.blockSignals(True)
        self.setRect(rect)
        self.blockSignals(False)

        if changed:
            self.rectEdited.emit(rect.normalized())

    def boundingRect(self):
        return QRectF()


class ControlPointLine(QGraphicsObject):

    lineChanged = Signal(QLineF)
    lineEdited = Signal(QLineF)

    def __init__(self, parent=None, **kwargs):
        QGraphicsObject.__init__(self, parent, **kwargs)
        self.setFlag(QGraphicsItem.ItemHasNoContents)
        self.setFlag(QGraphicsItem.ItemIsFocusable)

        self.__line = QLineF()
        self.__points = \
            [ControlPoint(self, ControlPoint.TopLeft),  # TopLeft is line start
             ControlPoint(self, ControlPoint.BottomRight)  # line end
             ]

        self.__activeControl = None

        if self.scene():
            self.__installFilter()

        for p in self.__points:
            p.setFlag(QGraphicsItem.ItemIsFocusable)
            p.setFocusProxy(self)

    def setLine(self, line):
        if not isinstance(line, QLineF):
            raise TypeError()

        if line != self.__line:
            self.__line = line
            self.__pointsLayout()
            self.lineChanged.emit(line)

    def line(self):
        return self.__line

    def isControlActive(self):
        """Return the state of the control. True if the control is
        active (user is dragging one of the points) False otherwise.

        """
        return self.__activeControl is not None

    def __installFilter(self):
        for p in self.__points:
            p.installSceneEventFilter(self)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSceneHasChanged:
            if self.scene():
                self.__installFilter()
        return QGraphicsObject.itemChange(self, change, value)

    def sceneEventFilter(self, obj, event):
        try:
            obj = toGraphicsObjectIfPossible(obj)
            if isinstance(obj, ControlPoint):
                etype = event.type()
                if etype == QEvent.GraphicsSceneMousePress:
                    self.__setActiveControl(obj)
                elif etype == QEvent.GraphicsSceneMouseRelease:
                    self.__setActiveControl(None)

            return QGraphicsObject.sceneEventFilter(self, obj, event)
        except Exception:
            log.error("", exc_info=True)

    def __pointsLayout(self):
        self.__points[0].setPos(self.__line.p1())
        self.__points[1].setPos(self.__line.p2())

    def __setActiveControl(self, control):
        if self.__activeControl != control:
            if self.__activeControl is not None:
                self.__activeControl.positionChanged[QPointF].disconnect(
                    self.__activeControlMoved
                )

            self.__activeControl = control

            if control is not None:
                control.positionChanged[QPointF].connect(
                    self.__activeControlMoved
                )

    def __activeControlMoved(self, pos):
        line = QLineF(self.__line)
        control = self.__activeControl
        if control.anchor() == ControlPoint.TopLeft:
            line.setP1(pos)
        elif control.anchor() == ControlPoint.BottomRight:
            line.setP2(pos)

        if self.__line != line:
            self.blockSignals(True)
            self.setLine(line)
            self.blockSignals(False)
            self.lineEdited.emit(line)

    def boundingRect(self):
        return QRectF()
