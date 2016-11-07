
import logging
from collections import OrderedDict
from xml.sax.saxutils import escape

import docutils.core
import CommonMark

from AnyQt.QtWidgets import (
    QGraphicsItem, QGraphicsPathItem, QGraphicsWidget, QGraphicsTextItem,
    QGraphicsDropShadowEffect, QMenu
)
from AnyQt.QtGui import (
    QPainterPath, QPainterPathStroker, QPolygonF, QColor, QPen
)
from AnyQt.QtCore import (
    Qt, QPointF, QSizeF, QRectF, QLineF, QEvent, QMetaObject, QT_VERSION
)
from AnyQt.QtCore import (
    pyqtSignal as Signal, pyqtProperty as Property, pyqtSlot as Slot
)

log = logging.getLogger(__name__)

from .graphicspathobject import GraphicsPathObject


class Annotation(QGraphicsWidget):
    """Base class for annotations in the canvas scheme.
    """
    def __init__(self, parent=None, **kwargs):
        QGraphicsWidget.__init__(self, parent, **kwargs)

    if QT_VERSION < 0x40700:
        geometryChanged = Signal()
        def setGeometry(self, rect):
            QGraphicsWidget.setGeometry(self, rect)
            self.geometryChanged.emit()
    else:
        def setGeometry(self, rect):
            QGraphicsWidget.setGeometry(self, rect)


class GraphicsTextEdit(QGraphicsTextItem):
    """
    QGraphicsTextItem subclass defining an additional placeholderText
    property (text displayed when no text is set).

    """
    #: Signal emitted when editing operation starts (the item receives edit
    #: focus)
    editingStarted = Signal()
    #: Signal emitted when editing operation ends (the item loses edit focus)
    editingFinished = Signal()

    def __init__(self, *args, **kwargs):
        QGraphicsTextItem.__init__(self, *args, **kwargs)
        self.setAcceptHoverEvents(True)
        self.__placeholderText = ""
        self.__editing = False  # text editing in progress

    def setPlaceholderText(self, text):
        """
        Set the placeholder text. This is shown when the item has no text,
        i.e when `toPlainText()` returns an empty string.

        """
        if self.__placeholderText != text:
            self.__placeholderText = text
            if not self.toPlainText():
                self.update()

    def placeholderText(self):
        """
        Return the placeholder text.
        """
        return str(self.__placeholderText)

    placeholderText_ = Property(str, placeholderText, setPlaceholderText,
                                doc="Placeholder text")

    def paint(self, painter, option, widget=None):
        QGraphicsTextItem.paint(self, painter, option, widget)

        # Draw placeholder text if necessary
        if not (self.toPlainText() and self.toHtml()) and \
                self.__placeholderText and \
                not (self.hasFocus() and \
                     self.textInteractionFlags() & Qt.TextEditable):
            brect = self.boundingRect()
            painter.setFont(self.font())
            metrics = painter.fontMetrics()
            text = metrics.elidedText(self.__placeholderText, Qt.ElideRight,
                                      brect.width())
            color = self.defaultTextColor()
            color.setAlpha(min(color.alpha(), 150))
            painter.setPen(QPen(color))
            painter.drawText(brect, Qt.AlignTop | Qt.AlignLeft, text)

    def hoverMoveEvent(self, event):
        layout = self.document().documentLayout()
        if layout.anchorAt(event.pos()):
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        flags = self.textInteractionFlags()
        if flags & Qt.LinksAccessibleByMouse \
                and not flags & Qt.TextSelectableByMouse \
                and self.document().documentLayout().anchorAt(event.pos()):
            # QGraphicsTextItem ignores the press event without
            # Qt.TextSelectableByMouse flag set. This causes the
            # corresponding mouse release to never get to this item
            # and therefore no linkActivated/openUrl ...
            super().mousePressEvent(event)
            if not event.isAccepted():
                event.accept()
        else:
            super().mousePressEvent(event)

    def setTextInteractionFlags(self, flags):
        super().setTextInteractionFlags(flags)
        if self.hasFocus() and flags & Qt.TextEditable and not self.__editing:
            self.__editing = True
            self.editingStarted.emit()

    def focusInEvent(self, event):
        super().focusInEvent(event)
        if self.textInteractionFlags() & Qt.TextEditable and \
                not self.__editing:
            self.__editing = True
            self.editingStarted.emit()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        if self.__editing and \
                event.reason() not in {Qt.ActiveWindowFocusReason,
                                       Qt.PopupFocusReason}:
            self.__editing = False
            self.editingFinished.emit()


def render_plain(content):
    """
    Return a html fragment for a plain pre-formatted text

    Parameters
    ----------
    content : str
        Plain text content

    Returns
    -------
    html : str
    """
    return '<p style="white-space: pre-wrap;">' + escape(content) + "</p>"


def render_html(content):
    """
    Return a html fragment unchanged.

    Parameters
    ----------
    content : str
        Html text.

    Returns
    -------
    html : str
    """
    return content


def render_markdown(content):
    """
    Return a html fragment from markdown text content

    Parameters
    ----------
    content : str
        A markdown formatted text

    Returns
    -------
    html : str
    """
    return CommonMark.commonmark(content)


def render_rst(content):
    """
    Return a html fragment from a RST text content

    Parameters
    ----------
    content : str
        A RST formatted text content

    Returns
    -------
    html : str
    """
    overrides = {
        "report_level": 10,  # suppress errors from appearing in the html
        "output-encoding": "utf-8"
    }
    html = docutils.core.publish_string(
        content, writer_name="html",
        settings_overrides=overrides
    )
    return html.decode("utf-8")


class TextAnnotation(Annotation):
    """
    Text annotation item for the canvas scheme.

    Text interaction (if enabled) is started by double clicking the item.
    """
    #: Emitted when the editing is finished (i.e. the item loses edit focus).
    editingFinished = Signal()

    #: Emitted when the text content changes on user interaction.
    textEdited = Signal()

    #: Emitted when the text annotation's contents change
    #: (`content` or `contentType` changed)
    contentChanged = Signal()

    #: Mapping of supported content types to corresponding
    #: content -> html transformer.
    ContentRenderer = OrderedDict([
        ("text/plain", render_plain),
        ("text/rst", render_rst),
        ("text/markdown", render_markdown),
        ("text/html", render_html),
    ])  # type: Dict[str, Callable[[str], [str]]]

    def __init__(self, parent=None, **kwargs):
        super().__init__(None, **kwargs)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)

        self.setFocusPolicy(Qt.ClickFocus)

        self.__contentType = "text/plain"
        self.__content = ""
        self.__renderer = render_plain

        self.__textMargins = (2, 2, 2, 2)
        self.__textInteractionFlags = Qt.NoTextInteraction
        self.__defaultInteractionFlags = (
            Qt.LinksAccessibleByMouse | Qt.LinksAccessibleByKeyboard)

        rect = self.geometry().translated(-self.pos())
        self.__framePen = QPen(Qt.NoPen)
        self.__framePathItem = QGraphicsPathItem(self)
        self.__framePathItem.setPen(self.__framePen)

        self.__textItem = GraphicsTextEdit(self)
        self.__textItem.setOpenExternalLinks(True)
        self.__textItem.setPlaceholderText(self.tr("Enter text here"))
        self.__textItem.setPos(2, 2)
        self.__textItem.setTextWidth(rect.width() - 4)
        self.__textItem.setTabChangesFocus(True)
        self.__textItem.setTextInteractionFlags(self.__defaultInteractionFlags)
        self.__textItem.setFont(self.font())
        self.__textItem.editingFinished.connect(self.__textEditingFinished)
        if self.__textItem.scene() is not None:
            self.__textItem.installSceneEventFilter(self)
        layout = self.__textItem.document().documentLayout()
        layout.documentSizeChanged.connect(self.__onDocumentSizeChanged)

        self.__updateFrame()
        # set parent item at the end in order to ensure
        # QGraphicsItem.ItemSceneHasChanged is delivered after initialization
        if parent is not None:
            self.setParentItem(parent)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSceneHasChanged:
            if self.__textItem.scene() is not None:
                self.__textItem.installSceneEventFilter(self)
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.__updateFrameStyle()
        return super().itemChange(change, value)

    def adjustSize(self):
        """Resize to a reasonable size.
        """
        self.__textItem.setTextWidth(-1)
        self.__textItem.adjustSize()
        size = self.__textItem.boundingRect().size()
        left, top, right, bottom = self.textMargins()
        geom = QRectF(self.pos(), size + QSizeF(left + right, top + bottom))
        self.setGeometry(geom)

    def setFramePen(self, pen):
        """Set the frame pen. By default Qt.NoPen is used (i.e. the frame
        is not shown).
        """
        if pen != self.__framePen:
            self.__framePen = QPen(pen)
            self.__updateFrameStyle()

    def framePen(self):
        """Return the frame pen.
        """
        return QPen(self.__framePen)

    def setFrameBrush(self, brush):
        """Set the frame brush.
        """
        self.__framePathItem.setBrush(brush)

    def frameBrush(self):
        """Return the frame brush.
        """
        return self.__framePathItem.brush()

    def __updateFrameStyle(self):
        if self.isSelected():
            pen = QPen(QColor(96, 158, 215), 1.25, Qt.DashDotLine)
        else:
            pen = self.__framePen

        self.__framePathItem.setPen(pen)

    def contentType(self):
        return self.__contentType

    def setContent(self, content, contentType="text/plain"):
        if self.__content != content or self.__contentType != contentType:
            self.__contentType = contentType
            self.__content = content
            self.__updateRenderedContent()
            self.contentChanged.emit()

    def content(self):
        return self.__content

    def setPlainText(self, text):
        """Set the annotation text as plain text.
        """
        self.setContent(text, "text/plain")

    def toPlainText(self):
        return self.__textItem.toPlainText()

    def setHtml(self, text):
        """Set the annotation text as html.
        """
        self.setContent(text, "text/html")

    def toHtml(self):
        return self.__textItem.toHtml()

    def setDefaultTextColor(self, color):
        """Set the default text color.
        """
        self.__textItem.setDefaultTextColor(color)

    def defaultTextColor(self):
        return self.__textItem.defaultTextColor()

    def setTextMargins(self, left, top, right, bottom):
        """Set the text margins.
        """
        margins = (left, top, right, bottom)
        if self.__textMargins != margins:
            self.__textMargins = margins
            self.__textItem.setPos(left, top)
            self.__textItem.setTextWidth(
                max(self.geometry().width() - left - right, 0)
            )

    def textMargins(self):
        """Return the text margins.
        """
        return self.__textMargins

    def document(self):
        """Return the QTextDocument instance used internally.
        """
        return self.__textItem.document()

    def setTextCursor(self, cursor):
        self.__textItem.setTextCursor(cursor)

    def textCursor(self):
        return self.__textItem.textCursor()

    def setTextInteractionFlags(self, flags):
        self.__textInteractionFlags = flags

    def textInteractionFlags(self):
        return self.__textInteractionFlags

    def setDefaultStyleSheet(self, stylesheet):
        self.document().setDefaultStyleSheet(stylesheet)

    def mouseDoubleClickEvent(self, event):
        Annotation.mouseDoubleClickEvent(self, event)

        if event.buttons() == Qt.LeftButton and \
                self.__textInteractionFlags & Qt.TextEditable:
            self.startEdit()

    def startEdit(self):
        """Start the annotation text edit process.
        """
        self.__textItem.setPlainText(self.__content)
        self.__textItem.setTextInteractionFlags(self.__textInteractionFlags)
        self.__textItem.setFocus(Qt.MouseFocusReason)
        self.__textItem.document().contentsChanged.connect(
            self.textEdited
        )

    def endEdit(self):
        """End the annotation edit.
        """
        content = self.__textItem.toPlainText()

        self.__textItem.setTextInteractionFlags(self.__defaultInteractionFlags)
        self.__textItem.document().contentsChanged.disconnect(
            self.textEdited
        )
        cursor = self.__textItem.textCursor()
        cursor.clearSelection()
        self.__textItem.setTextCursor(cursor)
        self.__content = content

        self.editingFinished.emit()
        # Cannot change the textItem's html immediately, this method is
        # invoked from it.
        # TODO: Separate the editor from the view.
        QMetaObject.invokeMethod(
            self, "__updateRenderedContent", Qt.QueuedConnection)

    def __onDocumentSizeChanged(self, size):
        # The size of the text document has changed. Expand the text
        # control rect's height if the text no longer fits inside.
        rect = self.geometry()
        _, top, _, bottom = self.textMargins()
        if rect.height() < (size.height() + bottom + top):
            rect.setHeight(size.height() + bottom + top)
            self.setGeometry(rect)

    def __updateFrame(self):
        rect = self.geometry()
        rect.moveTo(0, 0)
        path = QPainterPath()
        path.addRect(rect)
        self.__framePathItem.setPath(path)

    def resizeEvent(self, event):
        width = event.newSize().width()
        left, _, right, _ = self.textMargins()
        self.__textItem.setTextWidth(max(width - left - right, 0))
        self.__updateFrame()
        QGraphicsWidget.resizeEvent(self, event)

    def __textEditingFinished(self):
        self.endEdit()

    def sceneEventFilter(self, obj, event):
        if obj is self.__textItem and \
                not (self.__textItem.hasFocus() and
                     self.__textItem.textInteractionFlags() & Qt.TextEditable) and \
                event.type() in {QEvent.GraphicsSceneContextMenu} and \
                event.modifiers() & Qt.AltModifier:
            # Handle Alt + context menu events here
            self.contextMenuEvent(event)
            event.accept()
            return True
        return super().sceneEventFilter(obj, event)

    def changeEvent(self, event):
        if event.type() == QEvent.FontChange:
            self.__textItem.setFont(self.font())

        Annotation.changeEvent(self, event)

    @Slot()
    def __updateRenderedContent(self):
        try:
            renderer = TextAnnotation.ContentRenderer[self.__contentType]
        except KeyError:
            renderer = render_plain
        self.__textItem.setHtml(renderer(self.__content))

    def contextMenuEvent(self, event):
        if event.modifiers() & Qt.AltModifier:
            menu = QMenu(event.widget())
            menu.setAttribute(Qt.WA_DeleteOnClose)

            menu.addAction("text/plain")
            menu.addAction("text/markdown")
            menu.addAction("text/rst")
            menu.addAction("text/html")

            for action in menu.actions():
                action.setCheckable(True)
                action.setChecked(action.text() == self.__contentType.lower())

            @menu.triggered.connect
            def ontriggered(action):
                self.setContent(self.content(), action.text())

            menu.popup(event.screenPos())
            event.accept()
        else:
            event.ignore()


class ArrowItem(GraphicsPathObject):

    #: Arrow Style
    Plain, Concave = 1, 2

    def __init__(self, parent=None, line=None, lineWidth=4, **kwargs):
        GraphicsPathObject.__init__(self, parent, **kwargs)

        if line is None:
            line = QLineF(0, 0, 10, 0)

        self.__line = line

        self.__lineWidth = lineWidth

        self.__arrowStyle = ArrowItem.Plain

        self.__updateArrowPath()

    def setLine(self, line):
        """Set the baseline of the arrow (:class:`QLineF`).
        """
        if self.__line != line:
            self.__line = QLineF(line)
            self.__updateArrowPath()

    def line(self):
        """Return the baseline of the arrow.
        """
        return QLineF(self.__line)

    def setLineWidth(self, lineWidth):
        """Set the width of the arrow.
        """
        if self.__lineWidth != lineWidth:
            self.__lineWidth = lineWidth
            self.__updateArrowPath()

    def lineWidth(self):
        """Return the width of the arrow.
        """
        return self.__lineWidth

    def setArrowStyle(self, style):
        """Set the arrow style (`ArrowItem.Plain` or `ArrowItem.Concave`)
        """
        if self.__arrowStyle != style:
            self.__arrowStyle = style
            self.__updateArrowPath()

    def arrowStyle(self):
        """Return the arrow style
        """
        return self.__arrowStyle

    def __updateArrowPath(self):
        if self.__arrowStyle == ArrowItem.Plain:
            path = arrow_path_plain(self.__line, self.__lineWidth)
        else:
            path = arrow_path_concave(self.__line, self.__lineWidth)
        self.setPath(path)


def arrow_path_plain(line, width):
    """
    Return an :class:`QPainterPath` of a plain looking arrow.
    """
    path = QPainterPath()
    p1, p2 = line.p1(), line.p2()

    if p1 == p2:
        return path

    baseline = QLineF(line)
    # Require some minimum length.
    baseline.setLength(max(line.length() - width * 3, width * 3))
    path.moveTo(baseline.p1())
    path.lineTo(baseline.p2())

    stroker = QPainterPathStroker()
    stroker.setWidth(width)
    path = stroker.createStroke(path)

    arrow_head_len = width * 4
    arrow_head_angle = 50
    line_angle = line.angle() - 180

    angle_1 = line_angle - arrow_head_angle / 2.0
    angle_2 = line_angle + arrow_head_angle / 2.0

    points = [p2,
              p2 + QLineF.fromPolar(arrow_head_len, angle_1).p2(),
              p2 + QLineF.fromPolar(arrow_head_len, angle_2).p2(),
              p2]

    poly = QPolygonF(points)
    path_head = QPainterPath()
    path_head.addPolygon(poly)
    path = path.united(path_head)
    return path


def arrow_path_concave(line, width):
    """
    Return a :class:`QPainterPath` of a pretty looking arrow.
    """
    path = QPainterPath()
    p1, p2 = line.p1(), line.p2()

    if p1 == p2:
        return path

    baseline = QLineF(line)
    # Require some minimum length.
    baseline.setLength(max(line.length() - width * 3, width * 3))

    start, end = baseline.p1(), baseline.p2()
    mid = (start + end) / 2.0
    normal = QLineF.fromPolar(1.0, baseline.angle() + 90).p2()

    path.moveTo(start)
    path.lineTo(start + (normal * width / 4.0))

    path.quadTo(mid + (normal * width / 4.0),
                end + (normal * width / 1.5))

    path.lineTo(end - (normal * width / 1.5))
    path.quadTo(mid - (normal * width / 4.0),
                start - (normal * width / 4.0))
    path.closeSubpath()

    arrow_head_len = width * 4
    arrow_head_angle = 50
    line_angle = line.angle() - 180

    angle_1 = line_angle - arrow_head_angle / 2.0
    angle_2 = line_angle + arrow_head_angle / 2.0

    points = [p2,
              p2 + QLineF.fromPolar(arrow_head_len, angle_1).p2(),
              baseline.p2(),
              p2 + QLineF.fromPolar(arrow_head_len, angle_2).p2(),
              p2]

    poly = QPolygonF(points)
    path_head = QPainterPath()
    path_head.addPolygon(poly)
    path = path.united(path_head)
    return path


class ArrowAnnotation(Annotation):
    def __init__(self, parent=None, line=None, **kwargs):
        Annotation.__init__(self, parent, **kwargs)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)

        self.setFocusPolicy(Qt.ClickFocus)

        if line is None:
            line = QLineF(0, 0, 20, 0)

        self.__line = line
        self.__color = QColor(Qt.red)
        self.__arrowItem = ArrowItem(self)
        self.__arrowItem.setLine(line)
        self.__arrowItem.setBrush(self.__color)
        self.__arrowItem.setPen(QPen(Qt.NoPen))
        self.__arrowItem.setArrowStyle(ArrowItem.Concave)
        self.__arrowItem.setLineWidth(5)

        self.__shadow = QGraphicsDropShadowEffect(
            blurRadius=5, offset=QPointF(1.0, 2.0),
        )

        self.__arrowItem.setGraphicsEffect(self.__shadow)
        self.__shadow.setEnabled(True)

        self.__autoAdjustGeometry = True

    def setAutoAdjustGeometry(self, autoAdjust):
        """
        If set to `True` then the geometry will be adjusted whenever
        the arrow is changed with `setLine`. Otherwise the geometry
        of the item is only updated so the `line` lies within the
        `geometry()` rect (i.e. it only grows). True by default

        """
        self.__autoAdjustGeometry = autoAdjust
        if autoAdjust:
            self.adjustGeometry()

    def autoAdjustGeometry(self):
        """
        Should the geometry of the item be adjusted automatically when
        `setLine` is called.

        """
        return self.__autoAdjustGeometry

    def setLine(self, line):
        """
        Set the arrow base line (a `QLineF` in object coordinates).
        """
        if self.__line != line:
            self.__line = line

            # local item coordinate system
            geom = self.geometry().translated(-self.pos())

            if geom.isNull() and not line.isNull():
                geom = QRectF(0, 0, 1, 1)

            arrow_shape = arrow_path_concave(line, self.lineWidth())
            arrow_rect = arrow_shape.boundingRect()

            if not (geom.contains(arrow_rect)):
                geom = geom.united(arrow_rect)

            if self.__autoAdjustGeometry:
                # Shrink the geometry if required.
                geom = geom.intersected(arrow_rect)

            # topLeft can move changing the local coordinates.
            diff = geom.topLeft()
            line = QLineF(line.p1() - diff, line.p2() - diff)
            self.__arrowItem.setLine(line)
            self.__line = line

            # parent item coordinate system
            geom.translate(self.pos())
            self.setGeometry(geom)

    def line(self):
        """
        Return the arrow base line (`QLineF` in object coordinates).
        """
        return QLineF(self.__line)

    def setColor(self, color):
        """
        Set arrow brush color.
        """
        if self.__color != color:
            self.__color = QColor(color)
            self.__updateStyleState()

    def color(self):
        """
        Return the arrow brush color.
        """
        return QColor(self.__color)

    def setLineWidth(self, lineWidth):
        """
        Set the arrow line width.
        """
        self.__arrowItem.setLineWidth(lineWidth)

    def lineWidth(self):
        """
        Return the arrow line width.
        """
        return self.__arrowItem.lineWidth()

    def adjustGeometry(self):
        """
        Adjust the widget geometry to exactly fit the arrow inside
        while preserving the arrow path scene geometry.

        """
        # local system coordinate
        geom = self.geometry().translated(-self.pos())
        line = self.__line

        arrow_rect = self.__arrowItem.shape().boundingRect()

        if geom.isNull() and not line.isNull():
            geom = QRectF(0, 0, 1, 1)

        if not (geom.contains(arrow_rect)):
            geom = geom.united(arrow_rect)

        geom = geom.intersected(arrow_rect)
        diff = geom.topLeft()
        line = QLineF(line.p1() - diff, line.p2() - diff)
        geom.translate(self.pos())
        self.setGeometry(geom)
        self.setLine(line)

    def shape(self):
        arrow_shape = self.__arrowItem.shape()
        return self.mapFromItem(self.__arrowItem, arrow_shape)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.__updateStyleState()

        return Annotation.itemChange(self, change, value)

    def __updateStyleState(self):
        """
        Update the arrows' brush, pen, ... based on it's state
        """
        if self.isSelected():
            color = self.__color.darker(150)
            pen = QPen(QColor(96, 158, 215), Qt.DashDotLine)
            pen.setWidthF(1.25)
            pen.setCosmetic(True)
            self.__shadow.setColor(pen.color().darker(150))
        else:
            color = self.__color
            pen = QPen(Qt.NoPen)
            self.__shadow.setColor(QColor(63, 63, 63, 180))

        self.__arrowItem.setBrush(color)
        self.__arrowItem.setPen(pen)
