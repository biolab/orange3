"""
===========
Link Editor
===========

An Dialog to edit links between two nodes in the scheme.

"""

from collections import namedtuple

from xml.sax.saxutils import escape

from AnyQt.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QDialogButtonBox, QGraphicsScene,
    QGraphicsView, QGraphicsWidget, QGraphicsRectItem,
    QGraphicsLineItem, QGraphicsTextItem, QGraphicsLayoutItem,
    QGraphicsLinearLayout, QGraphicsGridLayout, QGraphicsPixmapItem,
    QGraphicsDropShadowEffect, QSizePolicy
)
from AnyQt.QtGui import QPalette, QPen, QPainter, QIcon

from AnyQt.QtCore import (
    Qt, QObject, QSize, QSizeF, QPointF, QRectF, QT_VERSION
)
from AnyQt.QtCore import pyqtSignal as Signal

from ..scheme import SchemeNode, SchemeLink, compatible_channels
from ..registry import InputSignal, OutputSignal

from ..resources import icon_loader

# This is a special value defined in Qt4 but does not seem to be exported
# by PyQt4
QWIDGETSIZE_MAX = ((1 << 24) - 1)


class EditLinksDialog(QDialog):
    """
    A dialog for editing links.

    >>> dlg = EditLinksDialog()
    >>> dlg.setNodes(file_node, test_learners_node)
    >>> dlg.setLinks([(file_node.output_channel("Data"),
    ...               (test_learners_node.input_channel("Data")])
    >>> if dlg.exec_() == EditLinksDialog.Accpeted:
    ...     new_links = dlg.links()
    ...

    """
    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)

        self.setModal(True)

        self.__setupUi()

    def __setupUi(self):
        layout = QVBoxLayout()

        # Scene with the link editor.
        self.scene = LinksEditScene()
        self.view = QGraphicsView(self.scene)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setRenderHint(QPainter.Antialiasing)

        self.scene.editWidget.geometryChanged.connect(self.__onGeometryChanged)

        # Ok/Cancel/Clear All buttons.
        buttons = QDialogButtonBox(QDialogButtonBox.Ok |
                                   QDialogButtonBox.Cancel |
                                   QDialogButtonBox.Reset,
                                   Qt.Horizontal)

        clear_button = buttons.button(QDialogButtonBox.Reset)
        clear_button.setText(self.tr("Clear All"))

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        clear_button.clicked.connect(self.scene.editWidget.clearLinks)

        layout.addWidget(self.view)
        layout.addWidget(buttons)

        self.setLayout(layout)
        layout.setSizeConstraint(QVBoxLayout.SetFixedSize)

        self.setSizeGripEnabled(False)

    def setNodes(self, source_node, sink_node):
        """
        Set the source/sink nodes (:class:`.SchemeNode` instances)
        between which to edit the links.

        .. note:: This should be called before :func:`setLinks`.

        """
        self.scene.editWidget.setNodes(source_node, sink_node)

    def setLinks(self, links):
        """
        Set a list of links to display between the source and sink
        nodes. The `links` is a list of (`OutputSignal`, `InputSignal`)
        tuples where the first element is an output signal of the source
        node and the second an input signal of the sink node.

        """
        self.scene.editWidget.setLinks(links)

    def links(self):
        """
        Return the links between the source and sink node.
        """
        return self.scene.editWidget.links()

    def __onGeometryChanged(self):
        size = self.scene.editWidget.size()
        left, top, right, bottom = self.getContentsMargins()
        self.view.setFixedSize(size.toSize() + \
                               QSize(left + right + 4, top + bottom + 4))


def find_item_at(scene, pos, order=Qt.DescendingOrder, type=None,
                 name=None):
    """
    Find an object in a :class:`QGraphicsScene` `scene` at `pos`.
    If `type` is not `None` the it must specify  the type of the item.
    I `name` is not `None` it must be a name of the object
    (`QObject.objectName()`).

    """
    items = scene.items(pos, Qt.IntersectsItemShape, order)
    for item in items:
        if type is not None and \
                not isinstance(item, type):
            continue

        if name is not None and isinstance(item, QObject) and \
                item.objectName() != name:
            continue
        return item
    else:
        return None


class LinksEditScene(QGraphicsScene):
    """
    A :class:`QGraphicsScene` used by the :class:`LinkEditWidget`.
    """
    def __init__(self, *args, **kwargs):
        QGraphicsScene.__init__(self, *args, **kwargs)

        self.editWidget = LinksEditWidget()
        self.addItem(self.editWidget)

    findItemAt = find_item_at


_Link = namedtuple(
    "_Link",
    ["output",    # OutputSignal
     "input",     # InputSignal
     "lineItem",  # QGraphicsLineItem connecting the input to output
     ])


class LinksEditWidget(QGraphicsWidget):
    """
    A Graphics Widget for editing the links between two nodes.
    """
    def __init__(self, *args, **kwargs):
        QGraphicsWidget.__init__(self, *args, **kwargs)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)

        self.source = None
        self.sink = None

        # QGraphicsWidget/Items in the scene.
        self.sourceNodeWidget = None
        self.sourceNodeTitle = None
        self.sinkNodeWidget = None
        self.sinkNodeTitle = None

        self.__links = []

        self.__textItems = []
        self.__iconItems = []
        self.__tmpLine = None
        self.__dragStartItem = None

        self.setLayout(QGraphicsLinearLayout(Qt.Vertical))
        self.layout().setContentsMargins(0, 0, 0, 0)

    def removeItems(self, items):
        """
        Remove child items from the widget and scene.
        """
        scene = self.scene()
        for item in items:
            item.setParentItem(None)
            if scene is not None:
                scene.removeItem(item)

    def clear(self):
        """
        Clear the editor state (source and sink nodes, channels ...).
        """
        if self.layout().count():
            widget = self.layout().takeAt(0).graphicsItem()
            self.removeItems([widget])

        self.source = None
        self.sink = None

    def setNodes(self, source, sink):
        """
        Set the source/sink nodes (:class:`SchemeNode` instances) between
        which to edit the links.

        .. note:: Call this before :func:`setLinks`.

        """
        self.clear()

        self.source = source
        self.sink = sink

        self.__updateState()

    def setLinks(self, links):
        """
        Set a list of links to display between the source and sink
        nodes. `links` must be a list of (`OutputSignal`, `InputSignal`)
        tuples where the first element refers to the source node
        and the second to the sink node (as set by `setNodes`).

        """
        self.clearLinks()
        for output, input in links:
            self.addLink(output, input)

    def links(self):
        """
        Return the links between the source and sink node.
        """
        return [(link.output, link.input) for link in self.__links]

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            startItem = find_item_at(self.scene(), event.pos(),
                                     type=ChannelAnchor)
            if startItem is not None:
                # Start a connection line drag.
                self.__dragStartItem = startItem
                self.__tmpLine = None
                event.accept()
                return

            lineItem = find_item_at(self.scene(), event.scenePos(),
                                    type=QGraphicsLineItem)
            if lineItem is not None:
                # Remove a connection under the mouse
                for link in self.__links:
                    if link.lineItem == lineItem:
                        self.removeLink(link.output, link.input)
                event.accept()
                return

        QGraphicsWidget.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:

            downPos = event.buttonDownPos(Qt.LeftButton)
            if not self.__tmpLine and self.__dragStartItem and \
                    (downPos - event.pos()).manhattanLength() > \
                        QApplication.instance().startDragDistance():
                # Start a line drag
                line = QGraphicsLineItem(self)
                start = self.__dragStartItem.boundingRect().center()
                start = self.mapFromItem(self.__dragStartItem, start)
                line.setLine(start.x(), start.y(),
                             event.pos().x(), event.pos().y())

                pen = QPen(Qt.black, 4)
                pen.setCapStyle(Qt.RoundCap)
                line.setPen(pen)
                line.show()

                self.__tmpLine = line

            if self.__tmpLine:
                # Update the temp line
                line = self.__tmpLine.line()
                line.setP2(event.pos())
                self.__tmpLine.setLine(line)

        QGraphicsWidget.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.__tmpLine:
            endItem = find_item_at(self.scene(), event.scenePos(),
                                     type=ChannelAnchor)

            if endItem is not None:
                startItem = self.__dragStartItem
                startChannel = startItem.channel()
                endChannel = endItem.channel()
                possible = False

                # Make sure the drag was from input to output (or reversed) and
                # not between input -> input or output -> output
                if type(startChannel) != type(endChannel):
                    if isinstance(startChannel, InputSignal):
                        startChannel, endChannel = endChannel, startChannel

                    possible = compatible_channels(startChannel, endChannel)

                if possible:
                    self.addLink(startChannel, endChannel)

            self.scene().removeItem(self.__tmpLine)
            self.__tmpLine = None
            self.__dragStartItem = None

        QGraphicsWidget.mouseReleaseEvent(self, event)

    def addLink(self, output, input):
        """
        Add a link between `output` (:class:`OutputSignal`) and `input`
        (:class:`InputSignal`).

        """
        if not compatible_channels(output, input):
            return

        if output not in self.source.output_channels():
            raise ValueError("%r is not an output channel of %r" % \
                             (output, self.source))

        if input not in self.sink.input_channels():
            raise ValueError("%r is not an input channel of %r" % \
                             (input, self.sink))

        if input.single:
            # Remove existing link if it exists.
            for s1, s2, _ in self.__links:
                if s2 == input:
                    self.removeLink(s1, s2)

        line = QGraphicsLineItem(self)

        source_anchor = self.sourceNodeWidget.anchor(output)
        sink_anchor = self.sinkNodeWidget.anchor(input)

        source_pos = source_anchor.boundingRect().center()
        source_pos = self.mapFromItem(source_anchor, source_pos)

        sink_pos = sink_anchor.boundingRect().center()
        sink_pos = self.mapFromItem(sink_anchor, sink_pos)
        line.setLine(source_pos.x(), source_pos.y(),
                     sink_pos.x(), sink_pos.y())
        pen = QPen(Qt.black, 4)
        pen.setCapStyle(Qt.RoundCap)
        line.setPen(pen)

        self.__links.append(_Link(output, input, line))

    def removeLink(self, output, input):
        """
        Remove a link between the `output` and `input` channels.
        """
        for link in list(self.__links):
            if link.output == output and link.input == input:
                self.scene().removeItem(link.lineItem)
                self.__links.remove(link)
                break
        else:
            raise ValueError("No such link {0.name!r} -> {1.name!r}." \
                             .format(output, input))

    def clearLinks(self):
        """
        Clear (remove) all the links.
        """
        for output, input, _ in list(self.__links):
            self.removeLink(output, input)

    def __updateState(self):
        """
        Update the widget with the new source/sink node signal descriptions.
        """
        widget = QGraphicsWidget()
        widget.setLayout(QGraphicsGridLayout())

        # Space between left and right anchors
        widget.layout().setHorizontalSpacing(50)

        left_node = EditLinksNode(self, direction=Qt.LeftToRight,
                                  node=self.source)

        left_node.setSizePolicy(QSizePolicy.MinimumExpanding,
                                QSizePolicy.MinimumExpanding)

        right_node = EditLinksNode(self, direction=Qt.RightToLeft,
                                   node=self.sink)

        right_node.setSizePolicy(QSizePolicy.MinimumExpanding,
                                 QSizePolicy.MinimumExpanding)

        left_node.setMinimumWidth(150)
        right_node.setMinimumWidth(150)

        widget.layout().addItem(left_node, 0, 0,)
        widget.layout().addItem(right_node, 0, 1,)

        title_template = "<center><b>{0}<b></center>"

        left_title = GraphicsTextWidget(self)
        left_title.setHtml(title_template.format(escape(self.source.title)))
        left_title.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        right_title = GraphicsTextWidget(self)
        right_title.setHtml(title_template.format(escape(self.sink.title)))
        right_title.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        widget.layout().addItem(left_title, 1, 0,
                                alignment=Qt.AlignHCenter | Qt.AlignTop)
        widget.layout().addItem(right_title, 1, 1,
                                alignment=Qt.AlignHCenter | Qt.AlignTop)

        widget.setParentItem(self)

        max_w = max(left_node.sizeHint(Qt.PreferredSize).width(),
                    right_node.sizeHint(Qt.PreferredSize).width())

        # fix same size
        left_node.setMinimumWidth(max_w)
        right_node.setMinimumWidth(max_w)
        left_title.setMinimumWidth(max_w)
        right_title.setMinimumWidth(max_w)

        self.layout().addItem(widget)
        self.layout().activate()

        self.sourceNodeWidget = left_node
        self.sinkNodeWidget = right_node
        self.sourceNodeTitle = left_title
        self.sinkNodeTitle = right_title

    if QT_VERSION < 0x40700:
        geometryChanged = Signal()

        def setGeometry(self, rect):
            QGraphicsWidget.setGeometry(self, rect)
            self.geometryChanged.emit()


class EditLinksNode(QGraphicsWidget):
    """
    A Node representation with channel anchors.

    `direction` specifies the layout (default `Qt.LeftToRight` will
    have icon on the left and channels on the right).

    """

    def __init__(self, parent=None, direction=Qt.LeftToRight,
                 node=None, icon=None, iconSize=None, **args):
        QGraphicsWidget.__init__(self, parent, **args)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.__direction = direction

        self.setLayout(QGraphicsLinearLayout(Qt.Horizontal))

        # Set the maximum size, otherwise the layout can't grow beyond its
        # sizeHint (and we need it to grow so the widget can grow and keep the
        # contents centered vertically.
        self.layout().setMaximumSize(QSizeF(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX))

        self.setSizePolicy(QSizePolicy.MinimumExpanding,
                           QSizePolicy.MinimumExpanding)

        self.__iconSize = iconSize or QSize(64, 64)
        self.__icon = icon

        self.__iconItem = QGraphicsPixmapItem(self)
        self.__iconLayoutItem = GraphicsItemLayoutItem(item=self.__iconItem)

        self.__channelLayout = QGraphicsGridLayout()
        self.__channelAnchors = []

        if self.__direction == Qt.LeftToRight:
            self.layout().addItem(self.__iconLayoutItem)
            self.layout().addItem(self.__channelLayout)
            channel_alignemnt = Qt.AlignRight

        else:
            self.layout().addItem(self.__channelLayout)
            self.layout().addItem(self.__iconLayoutItem)
            channel_alignemnt = Qt.AlignLeft

        self.layout().setAlignment(self.__iconLayoutItem, Qt.AlignCenter)
        self.layout().setAlignment(self.__channelLayout,
                                   Qt.AlignVCenter | channel_alignemnt)

        if node is not None:
            self.setSchemeNode(node)

    def setIconSize(self, size):
        """
        Set the icon size for the node.
        """
        if size != self.__iconSize:
            self.__iconSize = QSize(size)
            if self.__icon:
                self.__iconItem.setPixmap(self.__icon.pixmap(size))
                self.__iconLayoutItem.updateGeometry()

    def iconSize(self):
        """
        Return the icon size.
        """
        return QSize(self.__iconSize)

    def setIcon(self, icon):
        """
        Set the icon to display.
        """
        if icon != self.__icon:
            self.__icon = QIcon(icon)
            self.__iconItem.setPixmap(icon.pixmap(self.iconSize()))
            self.__iconLayoutItem.updateGeometry()

    def icon(self):
        """
        Return the icon.
        """
        return QIcon(self.__icon)

    def setSchemeNode(self, node):
        """
        Set an instance of `SchemeNode`. The widget will be initialized
        with its icon and channels.

        """
        self.node = node

        if self.__direction == Qt.LeftToRight:
            channels = node.output_channels()
        else:
            channels = node.input_channels()
        self.channels = channels

        loader = icon_loader.from_description(node.description)
        icon = loader.get(node.description.icon)

        self.setIcon(icon)

        label_template = ('<div align="{align}">'
                          '<span class="channelname">{name}</span>'
                          '</div>')

        if self.__direction == Qt.LeftToRight:
            align = "right"
            label_alignment = Qt.AlignVCenter | Qt.AlignRight
            anchor_alignment = Qt.AlignVCenter | Qt.AlignLeft
            label_row = 0
            anchor_row = 1
        else:
            align = "left"
            label_alignment = Qt.AlignVCenter | Qt.AlignLeft
            anchor_alignment = Qt.AlignVCenter | Qt.AlignLeft
            label_row = 1
            anchor_row = 0

        self.__channelAnchors = []
        grid = self.__channelLayout

        for i, channel in enumerate(channels):
            text = label_template.format(align=align,
                                         name=escape(channel.name))

            text_item = GraphicsTextWidget(self)
            text_item.setHtml(text)
            text_item.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            text_item.setToolTip(
                escape(getattr(channel, 'description', channel.type)))

            grid.addItem(text_item, i, label_row,
                         alignment=label_alignment)

            anchor = ChannelAnchor(self, channel=channel,
                                   rect=QRectF(0, 0, 20, 20))

            anchor.setBrush(self.palette().brush(QPalette.Mid))

            layout_item = GraphicsItemLayoutItem(grid, item=anchor)
            grid.addItem(layout_item, i, anchor_row,
                         alignment=anchor_alignment)
            anchor.setToolTip(escape(channel.type))

            self.__channelAnchors.append(anchor)

    def anchor(self, channel):
        """
        Return the anchor item for the `channel` name.
        """
        for anchor in self.__channelAnchors:
            if anchor.channel() == channel:
                return anchor

        raise ValueError(channel.name)

    def paint(self, painter, option, widget=None):
        painter.save()
        palette = self.palette()
        border = palette.brush(QPalette.Mid)
        pen = QPen(border, 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(palette.brush(QPalette.Window))
        brect = self.boundingRect()
        painter.drawRoundedRect(brect, 4, 4)
        painter.restore()


class GraphicsItemLayoutItem(QGraphicsLayoutItem):
    """
    A graphics layout that handles the position of a general QGraphicsItem
    in a QGraphicsLayout. The items boundingRect is used as this items fixed
    sizeHint and the item is positioned at the top left corner of the this
    items geometry.

    """

    def __init__(self, parent=None, item=None, ):
        self.__item = None

        QGraphicsLayoutItem.__init__(self, parent, isLayout=False)

        self.setOwnedByLayout(True)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        if item is not None:
            self.setItem(item)

    def setItem(self, item):
        self.__item = item
        self.setGraphicsItem(item)

    def setGeometry(self, rect):
        # TODO: specifiy if the geometry should be set relative to the
        # bounding rect top left corner
        if self.__item:
            self.__item.setPos(rect.topLeft())

        QGraphicsLayoutItem.setGeometry(self, rect)

    def sizeHint(self, which, constraint):
        if self.__item:
            return self.__item.boundingRect().size()
        else:
            return QGraphicsLayoutItem.sizeHint(self, which, constraint)


class ChannelAnchor(QGraphicsRectItem):
    """
    A rectangular Channel Anchor indicator.
    """
    def __init__(self, parent=None, channel=None, rect=None, **kwargs):
        QGraphicsRectItem.__init__(self, **kwargs)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.__channel = None

        if rect is None:
            rect = QRectF(0, 0, 20, 20)

        self.setRect(rect)

        if channel:
            self.setChannel(channel)

        self.__shadow = QGraphicsDropShadowEffect(blurRadius=5,
                                                  offset=QPointF(0, 0))
        self.setGraphicsEffect(self.__shadow)
        self.__shadow.setEnabled(False)

    def setChannel(self, channel):
        """
        Set the channel description.
        """
        if channel != self.__channel:
            self.__channel = channel

            if hasattr(channel, "description"):
                self.setToolTip(channel.description)
            # TODO: Should also include name, type, flags, dynamic in the
            #       tool tip as well as add visual clues to the anchor

    def channel(self):
        """
        Return the channel description.
        """
        return self.__channel

    def hoverEnterEvent(self, event):
        self.__shadow.setEnabled(True)
        QGraphicsRectItem.hoverEnterEvent(self, event)

    def hoverLeaveEvent(self, event):
        self.__shadow.setEnabled(False)
        QGraphicsRectItem.hoverLeaveEvent(self, event)


class GraphicsTextWidget(QGraphicsWidget):
    """
    A QGraphicsWidget subclass that manages a `QGraphicsTextItem`.
    """

    def __init__(self, parent=None, textItem=None):
        QGraphicsLayoutItem.__init__(self, parent)
        if textItem is None:
            textItem = QGraphicsTextItem()

        self.__textItem = textItem
        self.__textItem.setParentItem(self)
        self.__textItem.setPos(0, 0)

        doc_layout = self.document().documentLayout()
        doc_layout.documentSizeChanged.connect(self._onDocumentSizeChanged)

    def sizeHint(self, which, constraint=QSizeF()):
        if which == Qt.PreferredSize:
            doc = self.document()
            textwidth = doc.textWidth()
            if textwidth != constraint.width():
                cloned = doc.clone(self)
                cloned.setTextWidth(constraint.width())
                sh = cloned.size()
                cloned.deleteLater()
            else:
                sh = doc.size()
            return sh
        else:
            return QGraphicsWidget.sizeHint(self, which, constraint)

    def setGeometry(self, rect):
        QGraphicsWidget.setGeometry(self, rect)
        self.__textItem.setTextWidth(rect.width())

    def setPlainText(self, text):
        self.__textItem.setPlainText(text)
        self.updateGeometry()

    def setHtml(self, text):
        self.__textItem.setHtml(text)

    def adjustSize(self):
        self.__textItem.adjustSize()
        self.updateGeometry()

    def setDefaultTextColor(self, color):
        self.__textItem.setDefaultTextColor(color)

    def document(self):
        return self.__textItem.document()

    def setDocument(self, doc):
        doc_layout = self.document().documentLayout()
        doc_layout.documentSizeChanged.disconnect(self._onDocumentSizeChanged)

        self.__textItem.setDocument(doc)

        doc_layout = self.document().documentLayout()
        doc_layout.documentSizeChanged.connect(self._onDocumentSizeChanged)

        self.updateGeometry()

    def _onDocumentSizeChanged(self, size):
        """The doc size has changed"""
        self.updateGeometry()
