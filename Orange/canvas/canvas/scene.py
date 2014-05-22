"""
=====================
Canvas Graphics Scene
=====================

"""

import logging
import itertools

from operator import attrgetter

from xml.sax.saxutils import escape

from PyQt4.QtGui import QGraphicsScene, QPainter, QBrush, QColor, QFont, \
                        QGraphicsItem

from PyQt4.QtCore import Qt, QPointF, QRectF, QSizeF, QLineF, QBuffer, QEvent

from PyQt4.QtCore import pyqtSignal as Signal
from PyQt4.QtCore import PYQT_VERSION_STR


from .. import scheme

from . import items
from .layout import AnchorLayout
from .items.utils import toGraphicsObjectIfPossible, typed_signal_mapper

log = logging.getLogger(__name__)


NodeItemSignalMapper = typed_signal_mapper(items.NodeItem)


class CanvasScene(QGraphicsScene):
    """
    A Graphics Scene for displaying an :class:`~.scheme.Scheme` instance.
    """

    #: Signal emitted when a :class:`NodeItem` has been added to the scene.
    node_item_added = Signal(items.NodeItem)

    #: Signal emitted when a :class:`NodeItem` has been removed from the
    #: scene.
    node_item_removed = Signal(items.LinkItem)

    #: Signal emitted when a new :class:`LinkItem` has been added to the
    #: scene.
    link_item_added = Signal(items.LinkItem)

    #: Signal emitted when a :class:`LinkItem` has been removed.
    link_item_removed = Signal(items.LinkItem)

    #: Signal emitted when a :class:`Annotation` item has been added.
    annotation_added = Signal(items.annotationitem.Annotation)

    #: Signal emitted when a :class:`Annotation` item has been removed.
    annotation_removed = Signal(items.annotationitem.Annotation)

    #: Signal emitted when the position of a :class:`NodeItem` has changed.
    node_item_position_changed = Signal(items.NodeItem, QPointF)

    #: Signal emitted when an :class:`NodeItem` has been double clicked.
    node_item_double_clicked = Signal(items.NodeItem)

    #: An node item has been activated (clicked)
    node_item_activated = Signal(items.NodeItem)

    #: An node item has been hovered
    node_item_hovered = Signal(items.NodeItem)

    #: Link item has been hovered
    link_item_hovered = Signal(items.LinkItem)

    def __init__(self, *args, **kwargs):
        QGraphicsScene.__init__(self, *args, **kwargs)

        self.scheme = None
        self.registry = None

        # All node items
        self.__node_items = []
        # Mapping from SchemeNodes to canvas items
        self.__item_for_node = {}
        # All link items
        self.__link_items = []
        # Mapping from SchemeLinks to canvas items.
        self.__item_for_link = {}

        # All annotation items
        self.__annotation_items = []
        # Mapping from SchemeAnnotations to canvas items.
        self.__item_for_annotation = {}

        # Is the scene editable
        self.editable = True

        # Anchor Layout
        self.__anchor_layout = AnchorLayout()
        self.addItem(self.__anchor_layout)

        self.__channel_names_visible = True
        self.__node_animation_enabled = True

        self.user_interaction_handler = None

        self.activated_mapper = NodeItemSignalMapper(self)
        self.activated_mapper.pyMapped.connect(
            self.node_item_activated
        )

        self.hovered_mapper = NodeItemSignalMapper(self)
        self.hovered_mapper.pyMapped.connect(
            self.node_item_hovered
        )

        self.position_change_mapper = NodeItemSignalMapper(self)
        self.position_change_mapper.pyMapped.connect(
            self._on_position_change
        )

        log.info("'%s' intitialized." % self)

    def clear_scene(self):
        """
        Clear (reset) the scene.
        """
        if self.scheme is not None:
            self.scheme.node_added.disconnect(self.add_node)
            self.scheme.node_removed.disconnect(self.remove_node)

            self.scheme.link_added.disconnect(self.add_link)
            self.scheme.link_removed.disconnect(self.remove_link)

            self.scheme.annotation_added.disconnect(self.add_annotation)
            self.scheme.annotation_removed.disconnect(self.remove_annotation)

            self.scheme.node_state_changed.disconnect(
                self.on_widget_state_change
            )
            self.scheme.channel_state_changed.disconnect(
                self.on_link_state_change
            )

            # Remove all items to make sure all signals from scheme items
            # to canvas items are disconnected.

            for annot in self.scheme.annotations:
                if annot in self.__item_for_annotation:
                    self.remove_annotation(annot)

            for link in self.scheme.links:
                if link in self.__item_for_link:
                    self.remove_link(link)

            for node in self.scheme.nodes:
                if node in self.__item_for_node:
                    self.remove_node(node)

        self.scheme = None
        self.__node_items = []
        self.__item_for_node = {}
        self.__link_items = []
        self.__item_for_link = {}
        self.__annotation_items = []
        self.__item_for_annotation = {}

        self.__anchor_layout.deleteLater()

        self.user_interaction_handler = None

        self.clear()
        log.info("'%s' cleared." % self)

    def set_scheme(self, scheme):
        """
        Set the scheme to display. Populates the scene with nodes and links
        already in the scheme. Any further change to the scheme will be
        reflected in the scene.

        Parameters
        ----------
        scheme : :class:`~.scheme.Scheme`

        """
        if self.scheme is not None:
            # Clear the old scheme
            self.clear_scene()

        log.info("Setting scheme '%s' on '%s'" % (scheme, self))

        self.scheme = scheme
        if self.scheme is not None:
            self.scheme.node_added.connect(self.add_node)
            self.scheme.node_removed.connect(self.remove_node)

            self.scheme.link_added.connect(self.add_link)
            self.scheme.link_removed.connect(self.remove_link)

            self.scheme.annotation_added.connect(self.add_annotation)
            self.scheme.annotation_removed.connect(self.remove_annotation)

            self.scheme.node_state_changed.connect(
                self.on_widget_state_change
            )
            self.scheme.channel_state_changed.connect(
                self.on_link_state_change
            )

            self.scheme.topology_changed.connect(self.on_scheme_change)

        for node in scheme.nodes:
            self.add_node(node)

        for link in scheme.links:
            self.add_link(link)

        for annot in scheme.annotations:
            self.add_annotation(annot)

    def set_registry(self, registry):
        """
        Set the widget registry.
        """
        # TODO: Remove/Deprecate. Is used only to get the category/background
        # color. That should be part of the SchemeNode/WidgetDescription.
        log.info("Setting registry '%s on '%s'." % (registry, self))
        self.registry = registry

    def set_anchor_layout(self, layout):
        """
        Set an :class:`~.layout.AnchorLayout`
        """
        if self.__anchor_layout != layout:
            if self.__anchor_layout:
                self.__anchor_layout.deleteLater()
                self.__anchor_layout = None

            self.__anchor_layout = layout

    def anchor_layout(self):
        """
        Return the anchor layout instance.
        """
        return self.__anchor_layout

    def set_channel_names_visible(self, visible):
        """
        Set the channel names visibility.
        """
        self.__channel_names_visible = visible
        for link in self.__link_items:
            link.setChannelNamesVisible(visible)

    def channel_names_visible(self):
        """
        Return the channel names visibility state.
        """
        return self.__channel_names_visible

    def set_node_animation_enabled(self, enabled):
        """
        Set node animation enabled state.
        """
        if self.__node_animation_enabled != enabled:
            self.__node_animation_enabled = enabled

            for node in self.__node_items:
                node.setAnimationEnabled(enabled)

    def add_node_item(self, item):
        """
        Add a :class:`.NodeItem` instance to the scene.
        """
        if item in self.__node_items:
            raise ValueError("%r is already in the scene." % item)

        if item.pos().isNull():
            if self.__node_items:
                pos = self.__node_items[-1].pos() + QPointF(150, 0)
            else:
                pos = QPointF(150, 150)

            item.setPos(pos)

        item.setFont(self.font())

        # Set signal mappings
        self.activated_mapper.setPyMapping(item, item)
        item.activated.connect(self.activated_mapper.pyMap)

        self.hovered_mapper.setPyMapping(item, item)
        item.hovered.connect(self.hovered_mapper.pyMap)

        self.position_change_mapper.setPyMapping(item, item)
        item.positionChanged.connect(self.position_change_mapper.pyMap)

        self.addItem(item)

        self.__node_items.append(item)

        self.node_item_added.emit(item)

        log.info("Added item '%s' to '%s'" % (item, self))
        return item

    def add_node(self, node):
        """
        Add and return a default constructed :class:`.NodeItem` for a
        :class:`SchemeNode` instance `node`. If the `node` is already in
        the scene do nothing and just return its item.

        """
        if node in self.__item_for_node:
            # Already added
            return self.__item_for_node[node]

        item = self.new_node_item(node.description)

        if node.position:
            pos = QPointF(*node.position)
            item.setPos(pos)

        item.setTitle(node.title)
        item.setProcessingState(node.processing_state)
        item.setProgress(node.progress)

        for message in node.state_messages():
            item.setStateMessage(message)

        self.__item_for_node[node] = item

        node.position_changed.connect(self.__on_node_pos_changed)
        node.title_changed.connect(item.setTitle)
        node.progress_changed.connect(item.setProgress)
        node.processing_state_changed.connect(item.setProcessingState)
        node.state_message_changed.connect(item.setStateMessage)

        return self.add_node_item(item)

    def new_node_item(self, widget_desc, category_desc=None):
        """
        Construct an new :class:`.NodeItem` from a `WidgetDescription`.
        Optionally also set `CategoryDescription`.

        """
        item = items.NodeItem()
        item.setWidgetDescription(widget_desc)

        if category_desc is None and self.registry and widget_desc.category:
            category_desc = self.registry.category(widget_desc.category)

        if category_desc is None and self.registry is not None:
            try:
                category_desc = self.registry.category(widget_desc.category)
            except KeyError:
                pass

        if category_desc is not None:
            item.setWidgetCategory(category_desc)

        item.setAnimationEnabled(self.__node_animation_enabled)
        return item

    def remove_node_item(self, item):
        """
        Remove `item` (:class:`.NodeItem`) from the scene.
        """
        self.activated_mapper.removePyMappings(item)
        self.hovered_mapper.removePyMappings(item)

        item.hide()
        self.removeItem(item)
        self.__node_items.remove(item)

        self.node_item_removed.emit(item)

        log.info("Removed item '%s' from '%s'" % (item, self))

    def remove_node(self, node):
        """
        Remove the :class:`.NodeItem` instance that was previously
        constructed for a :class:`SchemeNode` `node` using the `add_node`
        method.

        """
        item = self.__item_for_node.pop(node)

        node.position_changed.disconnect(self.__on_node_pos_changed)
        node.title_changed.disconnect(item.setTitle)
        node.progress_changed.disconnect(item.setProgress)
        node.processing_state_changed.disconnect(item.setProcessingState)
        node.state_message_changed.disconnect(item.setStateMessage)

        self.remove_node_item(item)

    def node_items(self):
        """
        Return all :class:`.NodeItem` instances in the scene.
        """
        return list(self.__node_items)

    def add_link_item(self, item):
        """
        Add a link (:class:`.LinkItem`) to the scene.
        """
        if item.scene() is not self:
            self.addItem(item)

        item.setFont(self.font())
        self.__link_items.append(item)

        self.link_item_added.emit(item)

        log.info("Added link %r -> %r to '%s'" % \
                 (item.sourceItem.title(), item.sinkItem.title(), self))

        self.__anchor_layout.invalidateLink(item)

        return item

    def add_link(self, scheme_link):
        """
        Create and add a :class:`.LinkItem` instance for a
        :class:`SchemeLink` instance. If the link is already in the scene
        do nothing and just return its :class:`.LinkItem`.

        """
        if scheme_link in self.__item_for_link:
            return self.__item_for_link[scheme_link]

        source = self.__item_for_node[scheme_link.source_node]
        sink = self.__item_for_node[scheme_link.sink_node]

        item = self.new_link_item(source, scheme_link.source_channel,
                                  sink, scheme_link.sink_channel)

        item.setEnabled(scheme_link.enabled)
        scheme_link.enabled_changed.connect(item.setEnabled)

        if scheme_link.is_dynamic():
            item.setDynamic(True)
            item.setDynamicEnabled(scheme_link.dynamic_enabled)
            scheme_link.dynamic_enabled_changed.connect(item.setDynamicEnabled)

        self.add_link_item(item)
        self.__item_for_link[scheme_link] = item
        return item

    def new_link_item(self, source_item, source_channel,
                      sink_item, sink_channel):
        """
        Construct and return a new :class:`.LinkItem`
        """
        item = items.LinkItem()
        item.setSourceItem(source_item)
        item.setSinkItem(sink_item)

        def channel_name(channel):
            if isinstance(channel, str):
                return channel
            else:
                return channel.name

        source_name = channel_name(source_channel)
        sink_name = channel_name(sink_channel)

        fmt = "<b>{0}</b>&nbsp; \u2192 &nbsp;<b>{1}</b>"
        item.setToolTip(
            fmt.format(escape(source_name),
                       escape(sink_name))
        )

        item.setSourceName(source_name)
        item.setSinkName(sink_name)
        item.setChannelNamesVisible(self.__channel_names_visible)

        return item

    def remove_link_item(self, item):
        """
        Remove a link (:class:`.LinkItem`) from the scene.
        """
        # Invalidate the anchor layout.
        self.__anchor_layout.invalidateAnchorItem(
            item.sourceItem.outputAnchorItem
        )
        self.__anchor_layout.invalidateAnchorItem(
            item.sinkItem.inputAnchorItem
        )

        self.__link_items.remove(item)

        # Remove the anchor points.
        item.removeLink()
        self.removeItem(item)

        self.link_item_removed.emit(item)

        log.info("Removed link '%s' from '%s'" % (item, self))

        return item

    def remove_link(self, scheme_link):
        """
        Remove a :class:`.LinkItem` instance that was previously constructed
        for a :class:`SchemeLink` instance `link` using the `add_link` method.

        """
        item = self.__item_for_link.pop(scheme_link)
        scheme_link.enabled_changed.disconnect(item.setEnabled)

        if scheme_link.is_dynamic():
            scheme_link.dynamic_enabled_changed.disconnect(
                item.setDynamicEnabled
            )

        self.remove_link_item(item)

    def link_items(self):
        """
        Return all :class:`.LinkItem`\s in the scene.
        """
        return list(self.__link_items)

    def add_annotation_item(self, annotation):
        """
        Add an :class:`.Annotation` item to the scene.
        """
        self.__annotation_items.append(annotation)
        self.addItem(annotation)
        self.annotation_added.emit(annotation)
        return annotation

    def add_annotation(self, scheme_annot):
        """
        Create a new item for :class:`SchemeAnnotation` and add it
        to the scene. If the `scheme_annot` is already in the scene do
        nothing and just return its item.

        """
        if scheme_annot in self.__item_for_annotation:
            # Already added
            return self.__item_for_annotation[scheme_annot]

        if isinstance(scheme_annot, scheme.SchemeTextAnnotation):
            item = items.TextAnnotation()
            item.setPlainText(scheme_annot.text)
            x, y, w, h = scheme_annot.rect
            item.setPos(x, y)
            item.resize(w, h)
            item.setTextInteractionFlags(Qt.TextEditorInteraction)

            font = font_from_dict(scheme_annot.font, item.font())
            item.setFont(font)
            scheme_annot.text_changed.connect(item.setPlainText)

        elif isinstance(scheme_annot, scheme.SchemeArrowAnnotation):
            item = items.ArrowAnnotation()
            start, end = scheme_annot.start_pos, scheme_annot.end_pos
            item.setLine(QLineF(QPointF(*start), QPointF(*end)))
            item.setColor(QColor(scheme_annot.color))

        scheme_annot.geometry_changed.connect(
            self.__on_scheme_annot_geometry_change
        )

        self.add_annotation_item(item)
        self.__item_for_annotation[scheme_annot] = item

        return item

    def remove_annotation_item(self, annotation):
        """
        Remove an :class:`.Annotation` instance from the scene.

        """
        self.__annotation_items.remove(annotation)
        self.removeItem(annotation)
        self.annotation_removed.emit(annotation)

    def remove_annotation(self, scheme_annotation):
        """
        Remove an :class:`.Annotation` instance that was previously added
        using :func:`add_anotation`.

        """
        item = self.__item_for_annotation.pop(scheme_annotation)

        scheme_annotation.geometry_changed.disconnect(
            self.__on_scheme_annot_geometry_change
        )

        if isinstance(scheme_annotation, scheme.SchemeTextAnnotation):
            scheme_annotation.text_changed.disconnect(
                item.setPlainText
            )

        self.remove_annotation_item(item)

    def annotation_items(self):
        """
        Return all :class:`.Annotation` items in the scene.
        """
        return self.__annotation_items

    def item_for_annotation(self, scheme_annotation):
        return self.__item_for_annotation[scheme_annotation]

    def annotation_for_item(self, item):
        rev = dict(reversed(item) \
                   for item in self.__item_for_annotation.items())
        return rev[item]

    def commit_scheme_node(self, node):
        """
        Commit the `node` into the scheme.
        """
        if not self.editable:
            raise Exception("Scheme not editable.")

        if node not in self.__item_for_node:
            raise ValueError("No 'NodeItem' for node.")

        item = self.__item_for_node[node]

        try:
            self.scheme.add_node(node)
        except Exception:
            log.error("An error occurred while committing node '%s'",
                      node, exc_info=True)
            # Cleanup (remove the node item)
            self.remove_node_item(item)
            raise

        log.info("Commited node '%s' from '%s' to '%s'" % \
                 (node, self, self.scheme))

    def commit_scheme_link(self, link):
        """
        Commit a scheme link.
        """
        if not self.editable:
            raise Exception("Scheme not editable")

        if link not in self.__item_for_link:
            raise ValueError("No 'LinkItem' for link.")

        self.scheme.add_link(link)
        log.info("Commited link '%s' from '%s' to '%s'" % \
                 (link, self, self.scheme))

    def node_for_item(self, item):
        """
        Return the `SchemeNode` for the `item`.
        """
        rev = dict([(v, k) for k, v in self.__item_for_node.items()])
        return rev[item]

    def item_for_node(self, node):
        """
        Return the :class:`NodeItem` instance for a :class:`SchemeNode`.
        """
        return self.__item_for_node[node]

    def link_for_item(self, item):
        """
        Return the `SchemeLink for `item` (:class:`LinkItem`).
        """
        rev = dict([(v, k) for k, v in self.__item_for_link.items()])
        return rev[item]

    def item_for_link(self, link):
        """
        Return the :class:`LinkItem` for a :class:`SchemeLink`
        """
        return self.__item_for_link[link]

    def selected_node_items(self):
        """
        Return the selected :class:`NodeItem`'s.
        """
        return [item for item in self.__node_items if item.isSelected()]

    def selected_annotation_items(self):
        """
        Return the selected :class:`Annotation`'s
        """
        return [item for item in self.__annotation_items if item.isSelected()]

    def node_links(self, node_item):
        """
        Return all links from the `node_item` (:class:`NodeItem`).
        """
        return self.node_output_links(node_item) + \
               self.node_input_links(node_item)

    def node_output_links(self, node_item):
        """
        Return a list of all output links from `node_item`.
        """
        return [link for link in self.__link_items
                if link.sourceItem == node_item]

    def node_input_links(self, node_item):
        """
        Return a list of all input links for `node_item`.
        """
        return [link for link in self.__link_items
                if link.sinkItem == node_item]

    def neighbor_nodes(self, node_item):
        """
        Return a list of `node_item`'s (class:`NodeItem`) neighbor nodes.
        """
        neighbors = list(map(attrgetter("sourceItem"),
                             self.node_input_links(node_item)))

        neighbors.extend(map(attrgetter("sinkItem"),
                             self.node_output_links(node_item)))
        return neighbors

    def on_widget_state_change(self, widget, state):
        pass

    def on_link_state_change(self, link, state):
        pass

    def on_scheme_change(self, ):
        pass

    def _on_position_change(self, item):
        # Invalidate the anchor point layout and schedule a layout.
        self.__anchor_layout.invalidateNode(item)

        self.node_item_position_changed.emit(item, item.pos())

    def __on_node_pos_changed(self, pos):
        node = self.sender()
        item = self.__item_for_node[node]
        item.setPos(*pos)

    def __on_scheme_annot_geometry_change(self):
        annot = self.sender()
        item = self.__item_for_annotation[annot]
        if isinstance(annot, scheme.SchemeTextAnnotation):
            item.setGeometry(QRectF(*annot.rect))
        elif isinstance(annot, scheme.SchemeArrowAnnotation):
            p1 = item.mapFromScene(QPointF(*annot.start_pos))
            p2 = item.mapFromScene(QPointF(*annot.end_pos))
            item.setLine(QLineF(p1, p2))
        else:
            pass

    def item_at(self, pos, type_or_tuple=None, buttons=0):
        """Return the item at `pos` that is an instance of the specified
        type (`type_or_tuple`). If `buttons` (`Qt.MouseButtons`) is given
        only return the item if it is the top level item that would
        accept any of the buttons (`QGraphicsItem.acceptedMouseButtons`).

        """
        rect = QRectF(pos, QSizeF(1, 1))
        items = self.items(rect)

        if buttons:
            items = itertools.dropwhile(
                lambda item: not item.acceptedMouseButtons() & buttons,
                items
            )
            items = list(items)[:1]

        if type_or_tuple:
            items = [i for i in items if isinstance(i, type_or_tuple)]

        return items[0] if items else None

    if list(map(int, PYQT_VERSION_STR.split('.'))) < [4, 9]:
        # For QGraphicsObject subclasses items, itemAt ... return a
        # QGraphicsItem wrapper instance and not the actual class instance.
        def itemAt(self, *args, **kwargs):
            item = QGraphicsScene.itemAt(self, *args, **kwargs)
            return toGraphicsObjectIfPossible(item)

        def items(self, *args, **kwargs):
            items = QGraphicsScene.items(self, *args, **kwargs)
            return list(map(toGraphicsObjectIfPossible, items))

        def selectedItems(self, *args, **kwargs):
            return list(map(toGraphicsObjectIfPossible,
                       QGraphicsScene.selectedItems(self, *args, **kwargs)))

        def collidingItems(self, *args, **kwargs):
            return list(map(toGraphicsObjectIfPossible,
                       QGraphicsScene.collidingItems(self, *args, **kwargs)))

        def focusItem(self, *args, **kwargs):
            item = QGraphicsScene.focusItem(self, *args, **kwargs)
            return toGraphicsObjectIfPossible(item)

        def mouseGrabberItem(self, *args, **kwargs):
            item = QGraphicsScene.mouseGrabberItem(self, *args, **kwargs)
            return toGraphicsObjectIfPossible(item)

    def mousePressEvent(self, event):
        if self.user_interaction_handler and \
                self.user_interaction_handler.mousePressEvent(event):
            return

        # Right (context) click on the node item. If the widget is not
        # in the current selection then select the widget (only the widget).
        # Else simply return and let customContextMenuReqested signal
        # handle it
        shape_item = self.item_at(event.scenePos(), items.NodeItem)
        if shape_item and event.button() == Qt.RightButton and \
                shape_item.flags() & QGraphicsItem.ItemIsSelectable:
            if not shape_item.isSelected():
                self.clearSelection()
                shape_item.setSelected(True)

        return QGraphicsScene.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        if self.user_interaction_handler and \
                self.user_interaction_handler.mouseMoveEvent(event):
            return

        return QGraphicsScene.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        if self.user_interaction_handler and \
                self.user_interaction_handler.mouseReleaseEvent(event):
            return
        return QGraphicsScene.mouseReleaseEvent(self, event)

    def mouseDoubleClickEvent(self, event):
        if self.user_interaction_handler and \
                self.user_interaction_handler.mouseDoubleClickEvent(event):
            return

        return QGraphicsScene.mouseDoubleClickEvent(self, event)

    def keyPressEvent(self, event):
        if self.user_interaction_handler and \
                self.user_interaction_handler.keyPressEvent(event):
            return
        return QGraphicsScene.keyPressEvent(self, event)

    def keyReleaseEvent(self, event):
        if self.user_interaction_handler and \
                self.user_interaction_handler.keyReleaseEvent(event):
            return
        return QGraphicsScene.keyReleaseEvent(self, event)

    def set_user_interaction_handler(self, handler):
        if self.user_interaction_handler and \
                not self.user_interaction_handler.isFinished():
            self.user_interaction_handler.cancel()

        log.info("Setting interaction '%s' to '%s'" % (handler, self))

        self.user_interaction_handler = handler
        if handler:
            handler.start()

    def event(self, event):
        # TODO: change the base class of Node/LinkItem to QGraphicsWidget.
        # It already handles font changes.
        if event.type() == QEvent.FontChange:
            self.__update_font()

        return QGraphicsScene.event(self, event)

    def __update_font(self):
        font = self.font()
        for item in self.__node_items + self.__link_items:
            item.setFont(font)

    def __str__(self):
        return "%s(objectName=%r, ...)" % \
                (type(self).__name__, str(self.objectName()))


def font_from_dict(font_dict, font=None):
    if font is None:
        font = QFont()
    else:
        font = QFont(font)

    if "family" in font_dict:
        font.setFamily(font_dict["family"])

    if "size" in font_dict:
        font.setPixelSize(font_dict["size"])

    return font


def grab_svg(scene):
    """
    Return a SVG rendering of the scene contents.

    Parameters
    ----------
    scene : :class:`CanvasScene`

    """
    from PyQt4.QtSvg import QSvgGenerator
    svg_buffer = QBuffer()
    gen = QSvgGenerator()
    gen.setOutputDevice(svg_buffer)

    items_rect = scene.itemsBoundingRect().adjusted(-10, -10, 10, 10)

    if items_rect.isNull():
        items_rect = QRectF(0, 0, 10, 10)

    width, height = items_rect.width(), items_rect.height()
    rect_ratio = float(width) / height

    # Keep a fixed aspect ratio.
    aspect_ratio = 1.618
    if rect_ratio > aspect_ratio:
        height = int(height * rect_ratio / aspect_ratio)
    else:
        width = int(width * aspect_ratio / rect_ratio)

    target_rect = QRectF(0, 0, width, height)
    source_rect = QRectF(0, 0, width, height)
    source_rect.moveCenter(items_rect.center())

    gen.setSize(target_rect.size().toSize())
    gen.setViewBox(target_rect)

    painter = QPainter(gen)

    # Draw background.
    painter.setBrush(QBrush(Qt.white))
    painter.drawRect(target_rect)

    # Render the scene
    scene.render(painter, target_rect, source_rect)
    painter.end()

    buffer_str = bytes(svg_buffer.buffer())
    return buffer_str.decode("utf-8")
