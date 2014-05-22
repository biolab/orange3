"""
=========
Node Item
=========

"""

from xml.sax.saxutils import escape

from PyQt4.QtGui import (
    QGraphicsItem, QGraphicsObject, QGraphicsTextItem,
    QGraphicsDropShadowEffect, QGraphicsView,
    QPen, QBrush, QColor, QPalette, QIcon, QStyle, QPainter,
    QPainterPath, QPainterPathStroker, QApplication
)

from PyQt4.QtCore import Qt, QPointF, QRectF, QSize, QTimer, QPropertyAnimation
from PyQt4.QtCore import pyqtSignal as Signal
from PyQt4.QtCore import pyqtProperty as Property

from .graphicspathobject import GraphicsPathObject
from .utils import saturated, radial_gradient

from ...scheme.node import UserMessage
from ...registry import NAMED_COLORS
from ...resources import icon_loader
from .utils import uniform_linear_layout


def create_palette(light_color, color):
    """
    Return a new :class:`QPalette` from for the :class:`NodeBodyItem`.
    """
    palette = QPalette()

    palette.setColor(QPalette.Inactive, QPalette.Light,
                     saturated(light_color, 50))
    palette.setColor(QPalette.Inactive, QPalette.Midlight,
                     saturated(light_color, 90))
    palette.setColor(QPalette.Inactive, QPalette.Button,
                     light_color)

    palette.setColor(QPalette.Active, QPalette.Light,
                     saturated(color, 50))
    palette.setColor(QPalette.Active, QPalette.Midlight,
                     saturated(color, 90))
    palette.setColor(QPalette.Active, QPalette.Button,
                     color)
    palette.setColor(QPalette.ButtonText, QColor("#515151"))
    return palette


def default_palette():
    """
    Create and return a default palette for a node.
    """
    return create_palette(QColor(NAMED_COLORS["light-yellow"]),
                          QColor(NAMED_COLORS["yellow"]))


def animation_restart(animation):
    if animation.state() == QPropertyAnimation.Running:
        animation.pause()
    animation.start()


SHADOW_COLOR = "#9CACB4"
FOCUS_OUTLINE_COLOR = "#609ED7"


class NodeBodyItem(GraphicsPathObject):
    """
    The central part (body) of the `NodeItem`.
    """
    def __init__(self, parent=None):
        GraphicsPathObject.__init__(self, parent)
        assert(isinstance(parent, NodeItem))

        self.__processingState = 0
        self.__progress = -1
        self.__animationEnabled = False
        self.__isSelected = False
        self.__hasFocus = False
        self.__hover = False
        self.__shapeRect = QRectF(-10, -10, 20, 20)

        self.setAcceptHoverEvents(True)

        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

        self.setPen(QPen(Qt.NoPen))

        self.setPalette(default_palette())

        self.shadow = QGraphicsDropShadowEffect(
            blurRadius=3,
            color=QColor(SHADOW_COLOR),
            offset=QPointF(0, 0),
            )

        self.setGraphicsEffect(self.shadow)
        self.shadow.setEnabled(True)

        self.__blurAnimation = QPropertyAnimation(self.shadow, "blurRadius",
                                                  self)
        self.__blurAnimation.setDuration(100)
        self.__blurAnimation.finished.connect(self.__on_finished)

        self.__pingAnimation = QPropertyAnimation(self, "scale", self)
        self.__pingAnimation.setDuration(250)
        self.__pingAnimation.setKeyValues([(0.0, 1.0), (0.5, 1.1), (1.0, 1.0)])

    # TODO: The body item should allow the setting of arbitrary painter
    # paths (for instance rounded rect, ...)
    def setShapeRect(self, rect):
        """
        Set the item's shape `rect`. The item should be confined within
        this rect.

        """
        path = QPainterPath()
        path.addEllipse(rect)
        self.setPath(path)
        self.__shapeRect = rect

    def setPalette(self, palette):
        """
        Set the body color palette (:class:`QPalette`).
        """
        self.palette = palette
        self.__updateBrush()

    def setAnimationEnabled(self, enabled):
        """
        Set the node animation enabled.
        """
        if self.__animationEnabled != enabled:
            self.__animationEnabled = enabled

    def setProcessingState(self, state):
        """
        Set the processing state of the node.
        """
        if self.__processingState != state:
            self.__processingState = state
            if not state and self.__animationEnabled:
                self.ping()

    def setProgress(self, progress):
        """
        Set the progress indicator state of the node. `progress` should
        be a number between 0 and 100.

        """
        self.__progress = progress
        self.update()

    def ping(self):
        """
        Trigger a 'ping' animation.
        """
        animation_restart(self.__pingAnimation)

    def hoverEnterEvent(self, event):
        self.__hover = True
        self.__updateShadowState()
        return GraphicsPathObject.hoverEnterEvent(self, event)

    def hoverLeaveEvent(self, event):
        self.__hover = False
        self.__updateShadowState()
        return GraphicsPathObject.hoverLeaveEvent(self, event)

    def paint(self, painter, option, widget):
        """
        Paint the shape and a progress meter.
        """
        # Let the default implementation draw the shape
        if option.state & QStyle.State_Selected:
            # Prevent the default bounding rect selection indicator.
            option.state = option.state ^ QStyle.State_Selected
        GraphicsPathObject.paint(self, painter, option, widget)

        if self.__progress >= 0:
            # Draw the progress meter over the shape.
            # Set the clip to shape so the meter does not overflow the shape.
            painter.setClipPath(self.shape(), Qt.ReplaceClip)
            color = self.palette.color(QPalette.ButtonText)
            pen = QPen(color, 5)
            painter.save()
            painter.setPen(pen)
            painter.setRenderHints(QPainter.Antialiasing)
            span = int(self.__progress * 57.60)
            painter.drawArc(self.__shapeRect, 90 * 16, -span)
            painter.restore()

    def __updateShadowState(self):
        if self.__hasFocus:
            color = QColor(FOCUS_OUTLINE_COLOR)
            self.setPen(QPen(color, 1.5))
        else:
            self.setPen(QPen(Qt.NoPen))

        radius = 3
        enabled = False

        if self.__isSelected:
            enabled = True
            radius = 7

        if self.__hover:
            radius = 17
            enabled = True

        if enabled and not self.shadow.isEnabled():
            self.shadow.setEnabled(enabled)

        if self.__animationEnabled:
            if self.__blurAnimation.state() == QPropertyAnimation.Running:
                self.__blurAnimation.pause()

            self.__blurAnimation.setStartValue(self.shadow.blurRadius())
            self.__blurAnimation.setEndValue(radius)
            self.__blurAnimation.start()
        else:
            self.shadow.setBlurRadius(radius)

    def __updateBrush(self):
        palette = self.palette
        if self.__isSelected:
            cg = QPalette.Active
        else:
            cg = QPalette.Inactive

        palette.setCurrentColorGroup(cg)
        c1 = palette.color(QPalette.Light)
        c2 = palette.color(QPalette.Button)
        grad = radial_gradient(c2, c1)
        self.setBrush(QBrush(grad))

    # TODO: The selected and focus states should be set using the
    # QStyle flags (State_Selected. State_HasFocus)

    def setSelected(self, selected):
        """
        Set the `selected` state.

        .. note:: The item does not have `QGraphicsItem.ItemIsSelectable` flag.
                  This property is instead controlled by the parent NodeItem.

        """
        self.__isSelected = selected
        self.__updateBrush()

    def setHasFocus(self, focus):
        """
        Set the `has focus` state.

        .. note:: The item does not have `QGraphicsItem.ItemIsFocusable` flag.
                  This property is instead controlled by the parent NodeItem.

        """
        self.__hasFocus = focus
        self.__updateShadowState()

    def __on_finished(self):
        if self.shadow.blurRadius() == 0:
            self.shadow.setEnabled(False)


class AnchorPoint(QGraphicsObject):
    """
    A anchor indicator on the :class:`NodeAnchorItem`.
    """

    #: Signal emitted when the item's scene position changes.
    scenePositionChanged = Signal(QPointF)

    #: Signal emitted when the item's `anchorDirection` changes.
    anchorDirectionChanged = Signal(QPointF)

    def __init__(self, *args):
        QGraphicsObject.__init__(self, *args)
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        self.setFlag(QGraphicsItem.ItemHasNoContents, True)

        self.__direction = QPointF()

    def anchorScenePos(self):
        """
        Return anchor position in scene coordinates.
        """
        return self.mapToScene(QPointF(0, 0))

    def setAnchorDirection(self, direction):
        """
        Set the preferred direction (QPointF) in item coordinates.
        """
        if self.__direction != direction:
            self.__direction = direction
            self.anchorDirectionChanged.emit(direction)

    def anchorDirection(self):
        """
        Return the preferred anchor direction.
        """
        return self.__direction

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemScenePositionHasChanged:
            self.scenePositionChanged.emit(value)

        return QGraphicsObject.itemChange(self, change, value)

    def boundingRect(self,):
        return QRectF()


class NodeAnchorItem(GraphicsPathObject):
    """
    The left/right widget input/output anchors.
    """

    def __init__(self, parent, *args):
        GraphicsPathObject.__init__(self, parent, *args)
        self.setAcceptHoverEvents(True)
        self.setPen(QPen(Qt.NoPen))
        self.normalBrush = QBrush(QColor("#CDD5D9"))
        self.connectedBrush = QBrush(QColor("#9CACB4"))
        self.setBrush(self.normalBrush)

        self.shadow = QGraphicsDropShadowEffect(
            blurRadius=10,
            color=QColor(SHADOW_COLOR),
            offset=QPointF(0, 0)
        )

        self.setGraphicsEffect(self.shadow)
        self.shadow.setEnabled(False)

        # Does this item have any anchored links.
        self.anchored = False

        if isinstance(parent, NodeItem):
            self.__parentNodeItem = parent
        else:
            self.__parentNodeItem = None

        self.__anchorPath = QPainterPath()
        self.__points = []
        self.__pointPositions = []

        self.__fullStroke = None
        self.__dottedStroke = None
        self.__shape = None

    def parentNodeItem(self):
        """
        Return a parent :class:`NodeItem` or ``None`` if this anchor's
        parent is not a :class:`NodeItem` instance.

        """
        return self.__parentNodeItem

    def setAnchorPath(self, path):
        """
        Set the anchor's curve path as a :class:`QPainterPath`.
        """
        self.__anchorPath = path
        # Create a stroke of the path.
        stroke_path = QPainterPathStroker()
        stroke_path.setCapStyle(Qt.RoundCap)

        # Shape is wider (bigger mouse hit area - should be settable)
        stroke_path.setWidth(12)
        self.__shape = stroke_path.createStroke(path)

        # The full stroke
        stroke_path.setWidth(3)
        self.__fullStroke = stroke_path.createStroke(path)

        # The dotted stroke (when not connected to anything)
        stroke_path.setDashPattern(Qt.DotLine)
        self.__dottedStroke = stroke_path.createStroke(path)

        if self.anchored:
            self.setPath(self.__fullStroke)
            self.setBrush(self.connectedBrush)
        else:
            self.setPath(self.__dottedStroke)
            self.setBrush(self.normalBrush)

    def anchorPath(self):
        """
        Return the anchor path (:class:`QPainterPath`). This is a curve on
        which the anchor points lie.

        """
        return self.__anchorPath

    def setAnchored(self, anchored):
        """
        Set the items anchored state. When ``False`` the item draws it self
        with a dotted stroke.

        """
        self.anchored = anchored
        if anchored:
            self.setPath(self.__fullStroke)
            self.setBrush(self.connectedBrush)
        else:
            self.setPath(self.__dottedStroke)
            self.setBrush(self.normalBrush)

    def setConnectionHint(self, hint=None):
        """
        Set the connection hint. This can be used to indicate if
        a connection can be made or not.

        """
        raise NotImplementedError

    def count(self):
        """
        Return the number of anchor points.
        """
        return len(self.__points)

    def addAnchor(self, anchor, position=0.5):
        """
        Add a new :class:`AnchorPoint` to this item and return it's index.

        The `position` specifies where along the `anchorPath` is the new
        point inserted.

        """
        return self.insertAnchor(self.count(), anchor, position)

    def insertAnchor(self, index, anchor, position=0.5):
        """
        Insert a new :class:`AnchorPoint` at `index`.

        See also
        --------
        NodeAnchorItem.addAnchor

        """
        if anchor in self.__points:
            raise ValueError("%s already added." % anchor)

        self.__points.insert(index, anchor)
        self.__pointPositions.insert(index, position)

        anchor.setParentItem(self)
        anchor.setPos(self.__anchorPath.pointAtPercent(position))
        anchor.destroyed.connect(self.__onAnchorDestroyed)

        self.__updatePositions()

        self.setAnchored(bool(self.__points))

        return index

    def removeAnchor(self, anchor):
        """
        Remove and delete the anchor point.
        """
        anchor = self.takeAnchor(anchor)

        anchor.hide()
        anchor.setParentItem(None)
        anchor.deleteLater()

    def takeAnchor(self, anchor):
        """
        Remove the anchor but don't delete it.
        """
        index = self.__points.index(anchor)

        del self.__points[index]
        del self.__pointPositions[index]

        anchor.destroyed.disconnect(self.__onAnchorDestroyed)

        self.__updatePositions()

        self.setAnchored(bool(self.__points))

        return anchor

    def __onAnchorDestroyed(self, anchor):
        try:
            index = self.__points.index(anchor)
        except ValueError:
            return

        del self.__points[index]
        del self.__pointPositions[index]

    def anchorPoints(self):
        """
        Return a list of anchor points.
        """
        return list(self.__points)

    def anchorPoint(self, index):
        """
        Return the anchor point at `index`.
        """
        return self.__points[index]

    def setAnchorPositions(self, positions):
        """
        Set the anchor positions in percentages (0..1) along the path curve.
        """
        if self.__pointPositions != positions:
            self.__pointPositions = list(positions)

            self.__updatePositions()

    def anchorPositions(self):
        """
        Return the positions of anchor points as a list of floats where
        each float is between 0 and 1 and specifies where along the anchor
        path does the point lie (0 is at start 1 is at the end).

        """
        return list(self.__pointPositions)

    def shape(self):
        if self.__shape is not None:
            return self.__shape
        else:
            return GraphicsPathObject.shape(self)

    def boundingRect(self):
        if self.__shape is not None:
            return self.__shape.controlPointRect()
        else:
            return GraphicsPathObject.boundingRect(self)

    def hoverEnterEvent(self, event):
        self.shadow.setEnabled(True)
        return GraphicsPathObject.hoverEnterEvent(self, event)

    def hoverLeaveEvent(self, event):
        self.shadow.setEnabled(False)
        return GraphicsPathObject.hoverLeaveEvent(self, event)

    def __updatePositions(self):
        """Update anchor points positions.
        """
        for point, t in zip(self.__points, self.__pointPositions):
            pos = self.__anchorPath.pointAtPercent(t)
            point.setPos(pos)


class SourceAnchorItem(NodeAnchorItem):
    """
    A source anchor item
    """
    pass


class SinkAnchorItem(NodeAnchorItem):
    """
    A sink anchor item.
    """
    pass


def standard_icon(standard_pixmap):
    """
    Return return the application style's standard icon for a
    `QStyle.StandardPixmap`.

    """
    style = QApplication.instance().style()
    return style.standardIcon(standard_pixmap)


class GraphicsIconItem(QGraphicsItem):
    """
    A graphics item displaying an :class:`QIcon`.
    """
    def __init__(self, parent=None, icon=None, iconSize=None, **kwargs):
        QGraphicsItem.__init__(self, parent, **kwargs)
        self.setFlag(QGraphicsItem.ItemUsesExtendedStyleOption, True)

        if icon is None:
            icon = QIcon()

        if iconSize is None:
            style = QApplication.instance().style()
            size = style.pixelMetric(style.PM_LargeIconSize)
            iconSize = QSize(size, size)

        self.__transformationMode = Qt.SmoothTransformation

        self.__iconSize = QSize(iconSize)
        self.__icon = QIcon(icon)

    def setIcon(self, icon):
        """
        Set the icon (:class:`QIcon`).
        """
        if self.__icon != icon:
            self.__icon = QIcon(icon)
            self.update()

    def icon(self):
        """
        Return the icon (:class:`QIcon`).
        """
        return QIcon(self.__icon)

    def setIconSize(self, size):
        """
        Set the icon (and this item's) size (:class:`QSize`).
        """
        if self.__iconSize != size:
            self.prepareGeometryChange()
            self.__iconSize = QSize(size)
            self.update()

    def iconSize(self):
        """
        Return the icon size (:class:`QSize`).
        """
        return QSize(self.__iconSize)

    def setTransformationMode(self, mode):
        """
        Set pixmap transformation mode. (`Qt.SmoothTransformation` or
        `Qt.FastTransformation`).

        """
        if self.__transformationMode != mode:
            self.__transformationMode = mode
            self.update()

    def transformationMode(self):
        """
        Return the pixmap transformation mode.
        """
        return self.__transformationMode

    def boundingRect(self):
        return QRectF(0, 0, self.__iconSize.width(), self.__iconSize.height())

    def paint(self, painter, option, widget=None):
        if not self.__icon.isNull():
            if option.state & QStyle.State_Selected:
                mode = QIcon.Selected
            elif option.state & QStyle.State_Enabled:
                mode = QIcon.Normal
            elif option.state & QStyle.State_Active:
                mode = QIcon.Active
            else:
                mode = QIcon.Disabled

            transform = self.sceneTransform()

            if widget is not None:
                # 'widget' is the QGraphicsView.viewport()
                view = widget.parent()
                if isinstance(view, QGraphicsView):
                    # Combine the scene transform with the view transform.
                    view_transform = view.transform()
                    transform = view_transform * view_transform

            lod = option.levelOfDetailFromTransform(transform)

            w, h = self.__iconSize.width(), self.__iconSize.height()
            target = QRectF(0, 0, w, h)
            source = QRectF(0, 0, w * lod, w * lod).toRect()

            # The actual size of the requested pixmap can be smaller.
            size = self.__icon.actualSize(source.size(), mode=mode)
            source.setSize(size)

            pixmap = self.__icon.pixmap(source.size(), mode=mode)

            painter.setRenderHint(
                QPainter.SmoothPixmapTransform,
                self.__transformationMode == Qt.SmoothTransformation
            )

            painter.drawPixmap(target, pixmap, QRectF(source))


class NameTextItem(QGraphicsTextItem):
    def __init__(self, *args, **kwargs):
        super(NameTextItem, self).__init__(*args, **kwargs)
        self.__selected = False
        self.__palette = None

    def paint(self, painter, option, widget=None):
        if self.__selected:
            painter.save()
            painter.setPen(QPen(Qt.NoPen))
            painter.setBrush(self.palette().color(QPalette.Highlight))
            doc = self.document()
            margin = doc.documentMargin()
            painter.translate(margin, margin)
            offset = min(margin, 2)
            for line in self._lines(doc):
                rect = line.naturalTextRect()
                painter.drawRoundedRect(
                    rect.adjusted(-offset, -offset, offset, offset),
                    3, 3
                )

            painter.restore()

        super(NameTextItem, self).paint(painter, option, widget)

    def _blocks(self, doc):
        block = doc.begin()
        while block != doc.end():
            yield block
            block = block.next()

    def _lines(self, doc):
        for block in self._blocks(doc):
            blocklayout = block.layout()
            for i in range(blocklayout.lineCount()):
                yield blocklayout.lineAt(i)

    def setSelectionState(self, state):
        if self.__selected != state:
            self.__selected = state
            self.__updateDefaultTextColor()
            self.update()

    def setPalatte(self, palette):
        if self.__palette != palette:
            self.__palette = palette
            self.__updateDefaultTextColor()
            self.update()

    def palette(self):
        if self.__palette is None:
            scene = self.scene()
            if scene is not None:
                return scene.palette()
            else:
                return QPalette()
        else:
            return self.__palette

    def __updateDefaultTextColor(self):
        if self.__selected:
            role = QPalette.HighlightedText
        else:
            role = QPalette.WindowText
        self.setDefaultTextColor(self.palette().color(role))


class NodeItem(QGraphicsObject):
    """
    An widget node item in the canvas.
    """

    #: Signal emitted when the scene position of the node has changed.
    positionChanged = Signal()

    #: Signal emitted when the geometry of the channel anchors changes.
    anchorGeometryChanged = Signal()

    #: Signal emitted when the item has been activated (by a mouse double
    #: click or a keyboard)
    activated = Signal()

    #: The item is under the mouse.
    hovered = Signal()

    #: Span of the anchor in degrees
    ANCHOR_SPAN_ANGLE = 90

    #: Z value of the item
    Z_VALUE = 100

    def __init__(self, widget_description=None, parent=None, **kwargs):
        QGraphicsObject.__init__(self, parent, **kwargs)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemHasNoContents, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)

        # central body shape item
        self.shapeItem = None

        # in/output anchor items
        self.inputAnchorItem = None
        self.outputAnchorItem = None

        # title text item
        self.captionTextItem = None

        # error, warning, info items
        self.errorItem = None
        self.warningItem = None
        self.infoItem = None

        self.__title = ""
        self.__processingState = 0
        self.__progress = -1

        self.__error = None
        self.__warning = None
        self.__info = None

        self.__anchorLayout = None
        self.__animationEnabled = False

        self.setZValue(self.Z_VALUE)
        self.setupGraphics()

        self.setWidgetDescription(widget_description)

    @classmethod
    def from_node(cls, node):
        """
        Create an :class:`NodeItem` instance and initialize it from a
        :class:`SchemeNode` instance.

        """
        self = cls()
        self.setWidgetDescription(node.description)
#        self.setCategoryDescription(node.category)
        return self

    @classmethod
    def from_node_meta(cls, meta_description):
        """
        Create an `NodeItem` instance from a node meta description.
        """
        self = cls()
        self.setWidgetDescription(meta_description)
        return self

    def setupGraphics(self):
        """
        Set up the graphics.
        """
        shape_rect = QRectF(-24, -24, 48, 48)

        self.shapeItem = NodeBodyItem(self)
        self.shapeItem.setShapeRect(shape_rect)
        self.shapeItem.setAnimationEnabled(self.__animationEnabled)

        # Rect for widget's 'ears'.
        anchor_rect = QRectF(-31, -31, 62, 62)
        self.inputAnchorItem = SinkAnchorItem(self)
        input_path = QPainterPath()
        start_angle = 180 - self.ANCHOR_SPAN_ANGLE / 2
        input_path.arcMoveTo(anchor_rect, start_angle)
        input_path.arcTo(anchor_rect, start_angle, self.ANCHOR_SPAN_ANGLE)
        self.inputAnchorItem.setAnchorPath(input_path)

        self.outputAnchorItem = SourceAnchorItem(self)
        output_path = QPainterPath()
        start_angle = self.ANCHOR_SPAN_ANGLE / 2
        output_path.arcMoveTo(anchor_rect, start_angle)
        output_path.arcTo(anchor_rect, start_angle, - self.ANCHOR_SPAN_ANGLE)
        self.outputAnchorItem.setAnchorPath(output_path)

        self.inputAnchorItem.hide()
        self.outputAnchorItem.hide()

        # Title caption item
        self.captionTextItem = NameTextItem(self)

        self.captionTextItem.setPlainText("")
        self.captionTextItem.setPos(0, 33)

        def iconItem(standard_pixmap):
            item = GraphicsIconItem(self, icon=standard_icon(standard_pixmap),
                                    iconSize=QSize(16, 16))
            item.hide()
            return item

        self.errorItem = iconItem(QStyle.SP_MessageBoxCritical)
        self.warningItem = iconItem(QStyle.SP_MessageBoxWarning)
        self.infoItem = iconItem(QStyle.SP_MessageBoxInformation)

    # TODO: Remove the set[Widget|Category]Description. The user should
    # handle setting of icons, title, ...
    def setWidgetDescription(self, desc):
        """
        Set widget description.
        """
        self.widget_description = desc
        if desc is None:
            return

        icon = icon_loader.from_description(desc).get(desc.icon)
        if icon:
            self.setIcon(icon)

        if not self.title():
            self.setTitle(desc.name)

        if desc.inputs:
            self.inputAnchorItem.show()
        if desc.outputs:
            self.outputAnchorItem.show()

        tooltip = NodeItem_toolTipHelper(self)
        self.setToolTip(tooltip)

    def setWidgetCategory(self, desc):
        """
        Set the widget category.
        """
        self.category_description = desc
        if desc and desc.background:
            background = NAMED_COLORS.get(desc.background, desc.background)
            color = QColor(background)
            if color.isValid():
                self.setColor(color)

    def setIcon(self, icon):
        """
        Set the node item's icon (:class:`QIcon`).
        """
        if isinstance(icon, QIcon):
            self.icon_item = GraphicsIconItem(self.shapeItem, icon=icon,
                                              iconSize=QSize(36, 36))
            self.icon_item.setPos(-18, -18)
        else:
            raise TypeError

    def setColor(self, color, selectedColor=None):
        """
        Set the widget color.
        """
        if selectedColor is None:
            selectedColor = saturated(color, 150)
        palette = create_palette(color, selectedColor)
        self.shapeItem.setPalette(palette)

    def setPalette(self, palette):
        # TODO: The palette should override the `setColor`
        raise NotImplementedError

    def setTitle(self, title):
        """
        Set the node title. The title text is displayed at the bottom of the
        node.

        """
        self.__title = title
        self.__updateTitleText()

    def title(self):
        """
        Return the node title.
        """
        return self.__title

    title_ = Property(str, fget=title, fset=setTitle,
                      doc="Node title text.")

    def setFont(self, font):
        """
        Set the title text font (:class:`QFont`).
        """
        if font != self.font():
            self.prepareGeometryChange()
            self.captionTextItem.setFont(font)
            self.__updateTitleText()

    def font(self):
        """
        Return the title text font.
        """
        return self.captionTextItem.font()

    def setAnimationEnabled(self, enabled):
        """
        Set the node animation enabled state.
        """
        if self.__animationEnabled != enabled:
            self.__animationEnabled = enabled
            self.shapeItem.setAnimationEnabled(enabled)

    def animationEnabled(self):
        """
        Are node animations enabled.
        """
        return self.__animationEnabled

    def setProcessingState(self, state):
        """
        Set the node processing state i.e. the node is processing
        (is busy) or is idle.

        """
        if self.__processingState != state:
            self.__processingState = state
            self.shapeItem.setProcessingState(state)
            if not state:
                # Clear the progress meter.
                self.setProgress(-1)
                if self.__animationEnabled:
                    self.shapeItem.ping()

    def processingState(self):
        """
        The node processing state.
        """
        return self.__processingState

    processingState_ = Property(int, fget=processingState,
                                fset=setProcessingState)

    def setProgress(self, progress):
        """
        Set the node work progress state (number between 0 and 100).
        """
        if progress is None or progress < 0 or not self.__processingState:
            progress = -1

        progress = max(min(progress, 100), -1)
        if self.__progress != progress:
            self.__progress = progress
            self.shapeItem.setProgress(progress)
            self.__updateTitleText()

    def progress(self):
        """
        Return the node work progress state.
        """
        return self.__progress

    progress_ = Property(float, fget=progress, fset=setProgress,
                         doc="Node progress state.")

    def setProgressMessage(self, message):
        """
        Set the node work progress message.

        .. note:: Not yet implemented

        """
        pass

    def setStateMessage(self, message):
        """
        Set a state message to display over the item.

        Parameters
        ----------
        message : UserMessage
            Message to display. `message.severity` is used to determine
            the icon and `message.contents` is used as a tool tip.

        """
        # TODO: Group messages by message_id not by severity
        # and deprecate set[Error|Warning|Error]Message
        if message.severity == UserMessage.Info:
            self.setInfoMessage(message.contents)
        elif message.severity == UserMessage.Warning:
            self.setWarningMessage(message.contents)
        elif message.severity == UserMessage.Error:
            self.setErrorMessage(message.contents)

    def setErrorMessage(self, message):
        if self.__error != message:
            self.__error = message
            self.__updateMessages()

    def setWarningMessage(self, message):
        if self.__warning != message:
            self.__warning = message
            self.__updateMessages()

    def setInfoMessage(self, message):
        if self.__info != message:
            self.__info = message
            self.__updateMessages()

    def newInputAnchor(self):
        """
        Create and return a new input :class:`AnchorPoint`.
        """
        if not (self.widget_description and self.widget_description.inputs):
            raise ValueError("Widget has no inputs.")

        anchor = AnchorPoint()
        self.inputAnchorItem.addAnchor(anchor, position=1.0)

        positions = self.inputAnchorItem.anchorPositions()
        positions = uniform_linear_layout(positions)
        self.inputAnchorItem.setAnchorPositions(positions)

        return anchor

    def removeInputAnchor(self, anchor):
        """
        Remove input anchor.
        """
        self.inputAnchorItem.removeAnchor(anchor)

        positions = self.inputAnchorItem.anchorPositions()
        positions = uniform_linear_layout(positions)
        self.inputAnchorItem.setAnchorPositions(positions)

    def newOutputAnchor(self):
        """
        Create and return a new output :class:`AnchorPoint`.
        """
        if not (self.widget_description and self.widget_description.outputs):
            raise ValueError("Widget has no outputs.")

        anchor = AnchorPoint(self)
        self.outputAnchorItem.addAnchor(anchor, position=1.0)

        positions = self.outputAnchorItem.anchorPositions()
        positions = uniform_linear_layout(positions)
        self.outputAnchorItem.setAnchorPositions(positions)

        return anchor

    def removeOutputAnchor(self, anchor):
        """
        Remove output anchor.
        """
        self.outputAnchorItem.removeAnchor(anchor)

        positions = self.outputAnchorItem.anchorPositions()
        positions = uniform_linear_layout(positions)
        self.outputAnchorItem.setAnchorPositions(positions)

    def inputAnchors(self):
        """
        Return a list of all input anchor points.
        """
        return self.inputAnchorItem.anchorPoints()

    def outputAnchors(self):
        """
        Return a list of all output anchor points.
        """
        return self.outputAnchorItem.anchorPoints()

    def setAnchorRotation(self, angle):
        """
        Set the anchor rotation.
        """
        self.inputAnchorItem.setRotation(angle)
        self.outputAnchorItem.setRotation(angle)
        self.anchorGeometryChanged.emit()

    def anchorRotation(self):
        """
        Return the anchor rotation.
        """
        return self.inputAnchorItem.rotation()

    def boundingRect(self):
        # TODO: Important because of this any time the child
        # items change geometry the self.prepareGeometryChange()
        # needs to be called.
        return self.childrenBoundingRect()

    def shape(self):
        # Shape for mouse hit detection.
        # TODO: Should this return the union of all child items?
        return self.shapeItem.shape()

    def __updateTitleText(self):
        """
        Update the title text item.
        """
        title_safe = escape(self.title())
        if self.progress() > 0:
            text = '<div align="center">%s<br/>%i%%</div>' % \
                   (title_safe, int(self.progress()))
        else:
            text = '<div align="center">%s</div>' % \
                   (title_safe)

        # The NodeItems boundingRect could change.
        self.prepareGeometryChange()
        self.captionTextItem.setHtml(text)
        self.captionTextItem.document().adjustSize()
        width = self.captionTextItem.textWidth()
        self.captionTextItem.setPos(-width / 2.0, 33)

    def __updateMessages(self):
        """
        Update message items (position, visibility and tool tips).
        """
        items = [self.errorItem, self.warningItem, self.infoItem]
        messages = [self.__error, self.__warning, self.__info]
        for message, item in zip(messages, items):
            item.setVisible(bool(message))
            item.setToolTip(message or "")
        shown = [item for item in items if item.isVisible()]
        count = len(shown)
        if count:
            spacing = 3
            rects = [item.boundingRect() for item in shown]
            width = sum(rect.width() for rect in rects)
            width += spacing * max(0, count - 1)
            height = max(rect.height() for rect in rects)
            origin = self.shapeItem.boundingRect().top() - spacing - height
            origin = QPointF(-width / 2, origin)
            for item, rect in zip(shown, rects):
                item.setPos(origin)
                origin = origin + QPointF(rect.width() + spacing, 0)

    def mousePressEvent(self, event):
        if self.shapeItem.path().contains(event.pos()):
            return QGraphicsObject.mousePressEvent(self, event)
        else:
            event.ignore()

    def mouseDoubleClickEvent(self, event):
        if self.shapeItem.path().contains(event.pos()):
            QGraphicsObject.mouseDoubleClickEvent(self, event)
            QTimer.singleShot(0, self.activated.emit)
        else:
            event.ignore()

    def contextMenuEvent(self, event):
        if self.shapeItem.path().contains(event.pos()):
            return QGraphicsObject.contextMenuEvent(self, event)
        else:
            event.ignore()

    def focusInEvent(self, event):
        self.shapeItem.setHasFocus(True)
        return QGraphicsObject.focusInEvent(self, event)

    def focusOutEvent(self, event):
        self.shapeItem.setHasFocus(False)
        return QGraphicsObject.focusOutEvent(self, event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange:
            self.shapeItem.setSelected(value)
            self.captionTextItem.setSelectionState(value)
        elif change == QGraphicsItem.ItemPositionHasChanged:
            self.positionChanged.emit()

        return QGraphicsObject.itemChange(self, change, value)


TOOLTIP_TEMPLATE = """\
<html>
<head>
<style type="text/css">
{style}
</style>
</head>
<body>
{tooltip}
</body>
</html>
"""


def NodeItem_toolTipHelper(node, links_in=[], links_out=[]):
    """
    A helper function for constructing a standard tooltip for the node
    in on the canvas.

    Parameters:
    ===========
    node : NodeItem
        The node item instance.
    links_in : list of LinkItem instances
        A list of input links for the node.
    links_out : list of LinkItem instances
        A list of output links for the node.

    """
    desc = node.widget_description
    channel_fmt = "<li>{0}</li>"

    title_fmt = "<b>{title}</b><hr/>"
    title = title_fmt.format(title=escape(node.title()))
    inputs_list_fmt = "Inputs:<ul>{inputs}</ul><hr/>"
    outputs_list_fmt = "Outputs:<ul>{outputs}</ul>"
    if desc.inputs:
        inputs = [channel_fmt.format(inp.name) for inp in desc.inputs]
        inputs = inputs_list_fmt.format(inputs="".join(inputs))
    else:
        inputs = "No inputs<hr/>" 

    if desc.outputs:
        outputs = [channel_fmt.format(out.name) for out in desc.outputs]
        outputs = outputs_list_fmt.format(outputs="".join(outputs))
    else:
        outputs = "No outputs" 

    tooltip = title + inputs + outputs
    style = "ul { margin-top: 1px; margin-bottom: 1px; }"
    return TOOLTIP_TEMPLATE.format(style=style, tooltip=tooltip)
