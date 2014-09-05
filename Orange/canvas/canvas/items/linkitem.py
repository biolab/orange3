"""
=========
Link Item
=========

"""
import math

from PyQt4.QtGui import (
    QGraphicsItem, QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsObject,
    QGraphicsTextItem, QGraphicsDropShadowEffect, QPen, QBrush, QColor,
    QPainterPath, QTransform
)

from PyQt4.QtCore import Qt, QPointF, QEvent

from .nodeitem import SHADOW_COLOR
from .utils import stroke_path


class LinkCurveItem(QGraphicsPathItem):
    """
    Link curve item. The main component of a :class:`LinkItem`.
    """
    def __init__(self, parent):
        QGraphicsPathItem.__init__(self, parent)
        if not isinstance(parent, LinkItem):
            raise TypeError("'LinkItem' expected")

        self.setAcceptedMouseButtons(Qt.NoButton)
        self.__canvasLink = parent
        self.setAcceptHoverEvents(True)

        self.shadow = QGraphicsDropShadowEffect(
            blurRadius=5, color=QColor(SHADOW_COLOR),
            offset=QPointF(0, 0)
        )

        self.normalPen = QPen(QBrush(QColor("#9CACB4")), 2.0)
        self.hoverPen = QPen(QBrush(QColor("#7D7D7D")), 2.1)
        self.setPen(self.normalPen)
        self.setGraphicsEffect(self.shadow)
        self.shadow.setEnabled(False)

        self.__hover = False
        self.__enabled = True
        self.__shape = None

    def linkItem(self):
        """
        Return the :class:`LinkItem` instance this curve belongs to.
        """
        return self.__canvasLink

    def setHoverState(self, state):
        self.prepareGeometryChange()
        self.__shape = None
        self.__hover = state
        self.__update()

    def setLinkEnabled(self, state):
        self.prepareGeometryChange()
        self.__shape = None
        self.__enabled = state
        self.__update()

    def isLinkEnabled(self):
        return self.__enabled

    def setCurvePenSet(self, pen, hoverPen):
        self.prepareGeometryChange()
        if pen is not None:
            self.normalPen = pen
        if hoverPen is not None:
            self.hoverPen = hoverPen
        self.__shape = None
        self.__update()

    def shape(self):
        if self.__shape is None:
            path = self.path()
            pen = QPen(self.pen())
            pen.setWidthF(max(pen.widthF(), 7.0))
            pen.setStyle(Qt.SolidLine)
            self.__shape = stroke_path(path, pen)
        return self.__shape

    def setPath(self, path):
        self.__shape = None
        QGraphicsPathItem.setPath(self, path)

    def __update(self):
        shadow_enabled = self.__hover
        if self.shadow.isEnabled() != shadow_enabled:
            self.shadow.setEnabled(shadow_enabled)

        link_enabled = self.__enabled
        if link_enabled:
            pen_style = Qt.SolidLine
        else:
            pen_style = Qt.DashLine

        if self.__hover:
            pen = self.hoverPen
        else:
            pen = self.normalPen

        pen.setStyle(pen_style)
        self.setPen(pen)


class LinkAnchorIndicator(QGraphicsEllipseItem):
    """
    A visual indicator of the link anchor point at both ends
    of the :class:`LinkItem`.

    """
    def __init__(self, *args):
        QGraphicsEllipseItem.__init__(self, *args)
        self.setRect(-3, -3, 6, 6)
        self.setPen(QPen(Qt.NoPen))
        self.normalBrush = QBrush(QColor("#9CACB4"))
        self.hoverBrush = QBrush(QColor("#7D7D7D"))
        self.setBrush(self.normalBrush)
        self.__hover = False

    def setHoverState(self, state):
        """The hover state is set by the LinkItem.
        """
        self.__hover = state
        if state:
            self.setBrush(self.hoverBrush)
        else:
            self.setBrush(self.normalBrush)


class LinkItem(QGraphicsObject):
    """
    A Link item in the canvas that connects two :class:`.NodeItem`\s in the
    canvas.

    The link curve connects two `Anchor` items (see :func:`setSourceItem`
    and :func:`setSinkItem`). Once the anchors are set the curve
    automatically adjusts its end points whenever the anchors move.

    An optional source/sink text item can be displayed above the curve's
    central point (:func:`setSourceName`, :func:`setSinkName`)

    """

    #: Z value of the item
    Z_VALUE = 0

    def __init__(self, *args):
        QGraphicsObject.__init__(self, *args)
        self.setFlag(QGraphicsItem.ItemHasNoContents, True)
        self.setAcceptedMouseButtons(Qt.RightButton | Qt.LeftButton)
        self.setAcceptHoverEvents(True)

        self.setZValue(self.Z_VALUE)

        self.sourceItem = None
        self.sourceAnchor = None
        self.sinkItem = None
        self.sinkAnchor = None

        self.curveItem = LinkCurveItem(self)

        self.sourceIndicator = LinkAnchorIndicator(self)
        self.sinkIndicator = LinkAnchorIndicator(self)
        self.sourceIndicator.hide()
        self.sinkIndicator.hide()

        self.linkTextItem = QGraphicsTextItem(self)

        self.__sourceName = ""
        self.__sinkName = ""

        self.__dynamic = False
        self.__dynamicEnabled = False

        self.hover = False

    def setSourceItem(self, item, anchor=None):
        """
        Set the source `item` (:class:`.NodeItem`). Use `anchor`
        (:class:`.AnchorPoint`) as the curve start point (if ``None`` a new
        output anchor will be created using ``item.newOutputAnchor()``).

        Setting item to ``None`` and a valid anchor is a valid operation
        (for instance while mouse dragging one end of the link).

        """
        if item is not None and anchor is not None:
            if anchor not in item.outputAnchors():
                raise ValueError("Anchor must be belong to the item")

        if self.sourceItem != item:
            if self.sourceAnchor:
                # Remove a previous source item and the corresponding anchor
                self.sourceAnchor.scenePositionChanged.disconnect(
                    self._sourcePosChanged
                )

                if self.sourceItem is not None:
                    self.sourceItem.removeOutputAnchor(self.sourceAnchor)

                self.sourceItem = self.sourceAnchor = None

            self.sourceItem = item

            if item is not None and anchor is None:
                # Create a new output anchor for the item if none is provided.
                anchor = item.newOutputAnchor()

            # Update the visibility of the start point indicator.
            self.sourceIndicator.setVisible(bool(item))

        if anchor != self.sourceAnchor:
            if self.sourceAnchor is not None:
                self.sourceAnchor.scenePositionChanged.disconnect(
                    self._sourcePosChanged
                )

            self.sourceAnchor = anchor

            if self.sourceAnchor is not None:
                self.sourceAnchor.scenePositionChanged.connect(
                    self._sourcePosChanged
                )

        self.__updateCurve()

    def setSinkItem(self, item, anchor=None):
        """
        Set the sink `item` (:class:`.NodeItem`). Use `anchor`
        (:class:`.AnchorPoint`) as the curve end point (if ``None`` a new
        input anchor will be created using ``item.newInputAnchor()``).

        Setting item to ``None`` and a valid anchor is a valid operation
        (for instance while mouse dragging one and of the link).

        """
        if item is not None and anchor is not None:
            if anchor not in item.inputAnchors():
                raise ValueError("Anchor must be belong to the item")

        if self.sinkItem != item:
            if self.sinkAnchor:
                # Remove a previous source item and the corresponding anchor
                self.sinkAnchor.scenePositionChanged.disconnect(
                    self._sinkPosChanged
                )

                if self.sinkItem is not None:
                    self.sinkItem.removeInputAnchor(self.sinkAnchor)

                self.sinkItem = self.sinkAnchor = None

            self.sinkItem = item

            if item is not None and anchor is None:
                # Create a new input anchor for the item if none is provided.
                anchor = item.newInputAnchor()

            # Update the visibility of the end point indicator.
            self.sinkIndicator.setVisible(bool(item))

        if self.sinkAnchor != anchor:
            if self.sinkAnchor is not None:
                self.sinkAnchor.scenePositionChanged.disconnect(
                    self._sinkPosChanged
                )

            self.sinkAnchor = anchor

            if self.sinkAnchor is not None:
                self.sinkAnchor.scenePositionChanged.connect(
                    self._sinkPosChanged
                )

        self.__updateCurve()

    def setFont(self, font):
        """
        Set the font for the channel names text item.
        """
        if font != self.font():
            self.linkTextItem.setFont(font)
            self.__updateText()

    def font(self):
        """
        Return the font for the channel names text.
        """
        return self.linkTextItem.font()

    def setChannelNamesVisible(self, visible):
        """
        Set the visibility of the channel name text.
        """
        self.linkTextItem.setVisible(visible)

    def setSourceName(self, name):
        """
        Set the name of the source (used in channel name text).
        """
        if self.__sourceName != name:
            self.__sourceName = name
            self.__updateText()

    def sourceName(self):
        """
        Return the source name.
        """
        return self.__sourceName

    def setSinkName(self, name):
        """
        Set the name of the sink (used in channel name text).
        """
        if self.__sinkName != name:
            self.__sinkName = name
            self.__updateText()

    def sinkName(self):
        """
        Return the sink name.
        """
        return self.__sinkName

    def _sinkPosChanged(self, *arg):
        self.__updateCurve()

    def _sourcePosChanged(self, *arg):
        self.__updateCurve()

    def __updateCurve(self):
        self.prepareGeometryChange()
        if self.sourceAnchor and self.sinkAnchor:
            source_pos = self.sourceAnchor.anchorScenePos()
            sink_pos = self.sinkAnchor.anchorScenePos()
            source_pos = self.curveItem.mapFromScene(source_pos)
            sink_pos = self.curveItem.mapFromScene(sink_pos)

            # Adaptive offset for the curve control points to avoid a
            # cusp when the two points have the same y coordinate
            # and are close together
            delta = source_pos - sink_pos
            dist = math.sqrt(delta.x() ** 2 + delta.y() ** 2)
            cp_offset = min(dist / 2.0, 60.0)

            # TODO: make the curve tangent orthogonal to the anchors path.
            path = QPainterPath()
            path.moveTo(source_pos)
            path.cubicTo(source_pos + QPointF(cp_offset, 0),
                         sink_pos - QPointF(cp_offset, 0),
                         sink_pos)

            self.curveItem.setPath(path)
            self.sourceIndicator.setPos(source_pos)
            self.sinkIndicator.setPos(sink_pos)
            self.__updateText()
        else:
            self.setHoverState(False)
            self.curveItem.setPath(QPainterPath())

    def __updateText(self):
        self.prepareGeometryChange()

        if self.__sourceName or self.__sinkName:
            if self.__sourceName != self.__sinkName:
                text = "{0} \u2192 {1}".format(self.__sourceName,
                                                self.__sinkName)
            else:
                # If the names are the same show only one.
                # Is this right? If the sink has two input channels of the
                # same type having the name on the link help elucidate
                # the scheme.
                text = self.__sourceName
        else:
            text = ""

        self.linkTextItem.setPlainText(text)

        path = self.curveItem.path()
        if not path.isEmpty():
            center = path.pointAtPercent(0.5)
            angle = path.angleAtPercent(0.5)

            brect = self.linkTextItem.boundingRect()

            transform = QTransform()
            transform.translate(center.x(), center.y())
            transform.rotate(-angle)

            # Center and move above the curve path.
            transform.translate(-brect.width() / 2, -brect.height())

            self.linkTextItem.setTransform(transform)

    def removeLink(self):
        self.setSinkItem(None)
        self.setSourceItem(None)
        self.__updateCurve()

    def setHoverState(self, state):
        if self.hover != state:
            self.prepareGeometryChange()
            self.hover = state
            self.sinkIndicator.setHoverState(state)
            self.sourceIndicator.setHoverState(state)
            self.curveItem.setHoverState(state)

    def hoverEnterEvent(self, event):
        # Hover enter event happens when the mouse enters any child object
        # but we only want to show the 'hovered' shadow when the mouse
        # is over the 'curveItem', so we install self as an event filter
        # on the LinkCurveItem and listen to its hover events.
        self.curveItem.installSceneEventFilter(self)
        return QGraphicsObject.hoverEnterEvent(self, event)

    def hoverLeaveEvent(self, event):
        # Remove the event filter to prevent unnecessary work in
        # scene event filter when not needed
        self.curveItem.removeSceneEventFilter(self)
        return QGraphicsObject.hoverLeaveEvent(self, event)

    def sceneEventFilter(self, obj, event):
        if obj is self.curveItem:
            if event.type() == QEvent.GraphicsSceneHoverEnter:
                self.setHoverState(True)
            elif event.type() == QEvent.GraphicsSceneHoverLeave:
                self.setHoverState(False)

        return QGraphicsObject.sceneEventFilter(self, obj, event)

    def boundingRect(self):
        return self.childrenBoundingRect()

    def shape(self):
        return self.curveItem.shape()

    def setEnabled(self, enabled):
        """
        Reimplemented from :class:`QGraphicsObject`

        Set link enabled state. When disabled the link is rendered with a
        dashed line.

        """
        # This getter/setter pair override a property from the base class.
        # They should be renamed to e.g. setLinkEnabled/linkEnabled
        self.curveItem.setLinkEnabled(enabled)

    def isEnabled(self):
        return self.curveItem.isLinkEnabled()

    def setDynamicEnabled(self, enabled):
        """
        Set the link's dynamic enabled state.

        If the link is `dynamic` it will be rendered in red/green color
        respectively depending on the state of the dynamic enabled state.

        """
        if self.__dynamicEnabled != enabled:
            self.__dynamicEnabled = enabled
            if self.__dynamic:
                self.__updatePen()

    def isDynamicEnabled(self):
        """
        Is the link dynamic enabled.
        """
        return self.__dynamicEnabled

    def setDynamic(self, dynamic):
        """
        Mark the link as dynamic (i.e. it responds to
        :func:`setDynamicEnabled`).

        """
        if self.__dynamic != dynamic:
            self.__dynamic = dynamic
            self.__updatePen()

    def isDynamic(self):
        """
        Is the link dynamic.
        """
        return self.__dynamic

    def __updatePen(self):
        self.prepareGeometryChange()
        if self.__dynamic:
            if self.__dynamicEnabled:
                color = QColor(0, 150, 0, 150)
            else:
                color = QColor(150, 0, 0, 150)

            normal = QPen(QBrush(color), 2.0)
            hover = QPen(QBrush(color.darker(120)), 2.1)
        else:
            normal = QPen(QBrush(QColor("#9CACB4")), 2.0)
            hover = QPen(QBrush(QColor("#7D7D7D")), 2.1)

        self.curveItem.setCurvePenSet(normal, hover)
