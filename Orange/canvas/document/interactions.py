"""
User interaction handlers for CanvasScene.

"""
import logging

from PyQt4.QtGui import (
    QApplication, QGraphicsRectItem, QPen, QBrush, QColor, QFontMetrics
)

from PyQt4.QtCore import Qt, QObject, QSizeF, QPointF, QRect, QRectF, QLineF
from PyQt4.QtCore import pyqtSignal as Signal

from ..registry.qt import QtWidgetRegistry
from .. import scheme
from ..canvas import items
from ..canvas.items import controlpoints
from . import commands

log = logging.getLogger(__name__)


class UserInteraction(QObject):
    # cancel reason flags
    NoReason = 0  # No specified reason
    UserCancelReason = 1  # User canceled the operation (e.g. pressing ESC)
    InteractionOverrideReason = 3  # Another interaction was set
    ErrorReason = 4  # An internal error occurred
    OtherReason = 5

    # Emitted when the interaction is set on the scene.
    started = Signal()

    # Emitted when the interaction finishes successfully.
    finished = Signal()

    # Emitted when the interaction ends (canceled or finished)
    ended = Signal()

    # Emitted when the interaction is canceled.
    canceled = Signal([], [int])

    def __init__(self, document, parent=None, deleteOnEnd=True):
        QObject.__init__(self, parent)
        self.document = document
        self.scene = document.scene()
        self.scheme = document.scheme()
        self.deleteOnEnd = deleteOnEnd

        self.cancelOnEsc = False

        self.__finished = False
        self.__canceled = False
        self.__cancelReason = self.NoReason

    def start(self):
        """Start the interaction. This is called by the scene when
        the interaction is installed.

        Must be called from subclass implementations.

        """
        self.started.emit()

    def end(self):
        """Finish the interaction. Restore any leftover state in
        this method.

        .. note:: This gets called from the default `cancel` implementation.

        """
        self.__finished = True

        if self.scene.user_interaction_handler is self:
            self.scene.set_user_interaction_handler(None)

        if self.__canceled:
            self.canceled.emit()
            self.canceled[int].emit(self.__cancelReason)
        else:
            self.finished.emit()

        self.ended.emit()

        if self.deleteOnEnd:
            self.deleteLater()

    def cancel(self, reason=OtherReason):
        """Cancel the interaction for `reason`.
        """

        self.__canceled = True
        self.__cancelReason = reason

        self.end()

    def isFinished(self):
        """Has the interaction finished.
        """
        return self.__finished

    def isCanceled(self):
        """Was the interaction canceled.
        """
        return self.__canceled

    def cancelReason(self):
        """Return the reason the interaction was canceled.
        """
        return self.__cancelReason

    def mousePressEvent(self, event):
        return False

    def mouseMoveEvent(self, event):
        return False

    def mouseReleaseEvent(self, event):
        return False

    def mouseDoubleClickEvent(self, event):
        return False

    def keyPressEvent(self, event):
        if self.cancelOnEsc and event.key() == Qt.Key_Escape:
            self.cancel(self.UserCancelReason)
        return False

    def keyReleaseEvent(self, event):
        return False


class NoPossibleLinksError(ValueError):
    pass


def reversed_arguments(func):
    """Return a function with reversed argument order.
    """
    def wrapped(*args):
        return func(*reversed(args))
    return wrapped


class NewLinkAction(UserInteraction):
    """User drags a new link from an existing node anchor item to create
    a connection between two existing nodes or to a new node if the release
    is over an empty area, in which case a quick menu for new node selection
    is presented to the user.

    """
    # direction of the drag
    FROM_SOURCE = 1
    FROM_SINK = 2

    def __init__(self, document, *args, **kwargs):
        UserInteraction.__init__(self, document, *args, **kwargs)
        self.source_item = None
        self.sink_item = None
        self.from_item = None
        self.direction = None

        self.current_target_item = None
        self.tmp_link_item = None
        self.tmp_anchor_point = None
        self.cursor_anchor_point = None

    def remove_tmp_anchor(self):
        """Remove a temp anchor point from the current target item.
        """
        if self.direction == self.FROM_SOURCE:
            self.current_target_item.removeInputAnchor(self.tmp_anchor_point)
        else:
            self.current_target_item.removeOutputAnchor(self.tmp_anchor_point)
        self.tmp_anchor_point = None

    def create_tmp_anchor(self, item):
        """Create a new tmp anchor at the item (`NodeItem`).
        """
        assert(self.tmp_anchor_point is None)
        if self.direction == self.FROM_SOURCE:
            self.tmp_anchor_point = item.newInputAnchor()
        else:
            self.tmp_anchor_point = item.newOutputAnchor()

    def can_connect(self, target_item):
        """Is the connection between `self.from_item` (item where the drag
        started) and `target_item`.

        """
        node1 = self.scene.node_for_item(self.from_item)
        node2 = self.scene.node_for_item(target_item)

        if self.direction == self.FROM_SOURCE:
            return bool(self.scheme.propose_links(node1, node2))
        else:
            return bool(self.scheme.propose_links(node2, node1))

    def set_link_target_anchor(self, anchor):
        """Set the temp line target anchor
        """
        if self.direction == self.FROM_SOURCE:
            self.tmp_link_item.setSinkItem(None, anchor)
        else:
            self.tmp_link_item.setSourceItem(None, anchor)

    def target_node_item_at(self, pos):
        """Return a suitable NodeItem on which a link can be dropped.
        """
        # Test for a suitable NodeAnchorItem or NodeItem at pos.
        if self.direction == self.FROM_SOURCE:
            anchor_type = items.SinkAnchorItem
        else:
            anchor_type = items.SourceAnchorItem

        item = self.scene.item_at(pos, (anchor_type, items.NodeItem))

        if isinstance(item, anchor_type):
            item = item.parentNodeItem()

        return item

    def mousePressEvent(self, event):
        anchor_item = self.scene.item_at(event.scenePos(),
                                         items.NodeAnchorItem,
                                         buttons=Qt.LeftButton)
        if anchor_item and event.button() == Qt.LeftButton:
            # Start a new link starting at item
            self.from_item = anchor_item.parentNodeItem()
            if isinstance(anchor_item, items.SourceAnchorItem):
                self.direction = NewLinkAction.FROM_SOURCE
                self.source_item = self.from_item
            else:
                self.direction = NewLinkAction.FROM_SINK
                self.sink_item = self.from_item

            event.accept()
            return True
        else:
            # Whoever put us in charge did not know what he was doing.
            self.cancel(self.ErrorReason)
            return False

    def mouseMoveEvent(self, event):
        if not self.tmp_link_item:
            # On first mouse move event create the temp link item and
            # initialize it to follow the `cursor_anchor_point`.
            self.tmp_link_item = items.LinkItem()
            # An anchor under the cursor for the duration of this action.
            self.cursor_anchor_point = items.AnchorPoint()
            self.cursor_anchor_point.setPos(event.scenePos())

            # Set the `fixed` end of the temp link (where the drag started).
            if self.direction == self.FROM_SOURCE:
                self.tmp_link_item.setSourceItem(self.source_item)
            else:
                self.tmp_link_item.setSinkItem(self.sink_item)

            self.set_link_target_anchor(self.cursor_anchor_point)
            self.scene.addItem(self.tmp_link_item)

        # `NodeItem` at the cursor position
        item = self.target_node_item_at(event.scenePos())

        if self.current_target_item is not None and \
                (item is None or item is not self.current_target_item):
            # `current_target_item` is no longer under the mouse cursor
            # (was replaced by another item or the the cursor was moved over
            # an empty scene spot.
            log.info("%r is no longer the target.", self.current_target_item)
            self.remove_tmp_anchor()
            self.current_target_item = None

        if item is not None and item is not self.from_item:
            # The mouse is over an node item (different from the starting node)
            if self.current_target_item is item:
                # Avoid reseting the points
                pass
            elif self.can_connect(item):
                # Grab a new anchor
                log.info("%r is the new target.", item)
                self.create_tmp_anchor(item)
                self.set_link_target_anchor(self.tmp_anchor_point)
                self.current_target_item = item
            else:
                log.info("%r does not have compatible channels", item)
                self.set_link_target_anchor(self.cursor_anchor_point)
                # TODO: How to indicate that the connection is not possible?
                #       The node's anchor could be drawn with a 'disabled'
                #       palette
        else:
            self.set_link_target_anchor(self.cursor_anchor_point)

        self.cursor_anchor_point.setPos(event.scenePos())

        return True

    def mouseReleaseEvent(self, event):
        if self.tmp_link_item:
            item = self.target_node_item_at(event.scenePos())
            node = None
            stack = self.document.undoStack()
            stack.beginMacro("Add link")

            if item:
                # If the release was over a widget item
                # then connect them
                node = self.scene.node_for_item(item)
            else:
                # Release on an empty canvas part
                # Show a quick menu popup for a new widget creation.
                try:
                    node = self.create_new(event)
                except Exception:
                    log.error("Failed to create a new node, ending.",
                              exc_info=True)
                    node = None

                if node is not None:
                    self.document.addNode(node)

            if node is not None:
                self.connect_existing(node)
            else:
                self.end()

            stack.endMacro()
        else:
            self.end()
            return False

    def create_new(self, event):
        """Create and return a new node with a QuickWidgetMenu.
        """
        pos = event.screenPos()
        menu = self.document.quickMenu()
        node = self.scene.node_for_item(self.from_item)
        from_desc = node.description

        def is_compatible(source, sink):
            return any(scheme.compatible_channels(output, input) \
                       for output in source.outputs \
                       for input in sink.inputs)

        if self.direction == self.FROM_SINK:
            # Reverse the argument order.
            is_compatible = reversed_arguments(is_compatible)

        def filter(index):
            desc = index.data(QtWidgetRegistry.WIDGET_DESC_ROLE)
            if desc.isValid():
                return is_compatible(from_desc, desc)
            else:
                return False

        menu.setFilterFunc(filter)
        try:
            action = menu.exec_(pos)
        finally:
            menu.setFilterFunc(None)

        if action:
            item = action.property("item")
            desc = item.data(QtWidgetRegistry.WIDGET_DESC_ROLE)
            pos = event.scenePos()
            node = scheme.SchemeNode(desc, position=(pos.x(), pos.y()))
            return node

    def connect_existing(self, node):
        """Connect anchor_item to `node`.
        """
        if self.direction == self.FROM_SOURCE:
            source_item = self.source_item
            source_node = self.scene.node_for_item(source_item)
            sink_node = node
        else:
            source_node = node
            sink_item = self.sink_item
            sink_node = self.scene.node_for_item(sink_item)

        try:
            possible = self.scheme.propose_links(source_node, sink_node)

            log.debug("proposed (weighted) links: %r",
                      [(s1.name, s2.name, w) for s1, s2, w in possible])

            if not possible:
                raise NoPossibleLinksError

            source, sink, w = possible[0]
            links_to_add = [(source, sink)]

            show_link_dialog = False

            # Ambiguous new link request.
            if len(possible) >= 2:
                # Check for possible ties in the proposed link weights
                _, _, w2 = possible[1]
                if w == w2:
                    show_link_dialog = True

                # Check for destructive action (i.e. would the new link
                # replace a previous link)
                if sink.single and self.scheme.find_links(sink_node=sink_node,
                                                          sink_channel=sink):
                    show_link_dialog = True

                if show_link_dialog:
                    links_action = EditNodeLinksAction(
                                    self.document, source_node, sink_node)
                    try:
                        links_action.edit_links()
                    except Exception:
                        log.error("'EditNodeLinksAction' failed",
                                  exc_info=True)
                        raise
                    # EditNodeLinksAction already commits the links on accepted
                    links_to_add = []

            for source, sink in links_to_add:
                if sink.single:
                    # Remove an existing link to the sink channel if present.
                    existing_link = self.scheme.find_links(
                        sink_node=sink_node, sink_channel=sink
                    )

                    if existing_link:
                        self.document.removeLink(existing_link[0])

                # Check if the new link is a duplicate of an existing link
                duplicate = self.scheme.find_links(
                    source_node, source, sink_node, sink
                )

                if duplicate:
                    # Do nothing.
                    continue

                # Remove temp items before creating a new link
                self.cleanup()

                link = scheme.SchemeLink(source_node, source, sink_node, sink)
                self.document.addLink(link)

        except scheme.IncompatibleChannelTypeError:
            log.info("Cannot connect: invalid channel types.")
            self.cancel()
        except scheme.SchemeTopologyError:
            log.info("Cannot connect: connection creates a cycle.")
            self.cancel()
        except NoPossibleLinksError:
            log.info("Cannot connect: no possible links.")
            self.cancel()
        except Exception:
            log.error("An error occurred during the creation of a new link.",
                      exc_info=True)
            self.cancel()

        if not self.isFinished():
            self.end()

    def end(self):
        self.cleanup()
        UserInteraction.end(self)

    def cancel(self, reason=UserInteraction.OtherReason):
        self.cleanup()
        UserInteraction.cancel(self, reason)

    def cleanup(self):
        """Cleanup all temp items in the scene that are left.
        """
        if self.tmp_link_item:
            self.tmp_link_item.setSinkItem(None)
            self.tmp_link_item.setSourceItem(None)

            if self.tmp_link_item.scene():
                self.scene.removeItem(self.tmp_link_item)

            self.tmp_link_item = None

        if self.current_target_item:
            self.remove_tmp_anchor()
            self.current_target_item = None

        if self.cursor_anchor_point and self.cursor_anchor_point.scene():
            self.scene.removeItem(self.cursor_anchor_point)
            self.cursor_anchor_point = None


class NewNodeAction(UserInteraction):
    """Present the user with a quick menu for node selection and
    create the selected node.

    """

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.create_new(event.screenPos())
            self.end()

    def create_new(self, pos):
        """Create a new widget with a QuickWidgetMenu at `pos`
        (in screen coordinates).

        """
        menu = self.document.quickMenu()
        menu.setFilterFunc(None)

        action = menu.exec_(pos)
        if action:
            item = action.property("item")
            desc = item.data(QtWidgetRegistry.WIDGET_DESC_ROLE)
            # Get the scene position
            view = self.document.view()
            pos = view.mapToScene(view.mapFromGlobal(pos))
            node = scheme.SchemeNode(desc, position=(pos.x(), pos.y()))
            self.document.addNode(node)
            return node


class RectangleSelectionAction(UserInteraction):
    """Select items in the scene using a Rectangle selection
    """
    def __init__(self, document, *args, **kwargs):
        UserInteraction.__init__(self, document, *args, **kwargs)
        self.initial_selection = None
        self.last_selection = None
        self.selection_rect = None
        self.modifiers = 0

    def mousePressEvent(self, event):
        pos = event.scenePos()
        any_item = self.scene.item_at(pos)
        if not any_item and event.button() & Qt.LeftButton:
            self.modifiers = event.modifiers()
            self.selection_rect = QRectF(pos, QSizeF(0, 0))
            self.rect_item = QGraphicsRectItem(
                self.selection_rect.normalized()
            )

            self.rect_item.setPen(
                QPen(QBrush(QColor(51, 153, 255, 192)),
                     0.4, Qt.SolidLine, Qt.RoundCap)
            )

            self.rect_item.setBrush(
                QBrush(QColor(168, 202, 236, 192))
            )

            self.rect_item.setZValue(-100)

            # Clear the focus if necessary.
            if not self.scene.stickyFocus():
                self.scene.clearFocus()

            if not self.modifiers & Qt.ControlModifier:
                self.scene.clearSelection()

            event.accept()
            return True
        else:
            self.cancel(self.ErrorReason)
            return False

    def mouseMoveEvent(self, event):
        if not self.rect_item.scene():
            self.scene.addItem(self.rect_item)
        self.update_selection(event)
        return True

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.initial_selection is None:
                # A single click.
                self.scene.clearSelection()
            else:
                self.update_selection(event)
        self.end()
        return True

    def update_selection(self, event):
        if self.initial_selection is None:
            self.initial_selection = set(self.scene.selectedItems())
            self.last_selection = self.initial_selection

        pos = event.scenePos()
        self.selection_rect = QRectF(self.selection_rect.topLeft(), pos)

        rect = self._bound_selection_rect(self.selection_rect.normalized())

        # Need that constant otherwise the sceneRect will still grow
        pw = self.rect_item.pen().width() + 0.5

        self.rect_item.setRect(rect.adjusted(pw, pw, -pw, -pw))

        selected = self.scene.items(self.selection_rect.normalized(),
                                    Qt.IntersectsItemShape,
                                    Qt.AscendingOrder)

        selected = set([item for item in selected if \
                        item.flags() & Qt.ItemIsSelectable])

        if self.modifiers & Qt.ControlModifier:
            for item in selected | self.last_selection | \
                    self.initial_selection:
                item.setSelected(
                    (item in selected) ^ (item in self.initial_selection)
                )
        else:
            for item in selected.union(self.last_selection):
                item.setSelected(item in selected)

        self.last_selection = set(self.scene.selectedItems())

    def end(self):
        self.initial_selection = None
        self.last_selection = None
        self.modifiers = 0

        self.rect_item.hide()
        if self.rect_item.scene() is not None:
            self.scene.removeItem(self.rect_item)
        UserInteraction.end(self)

    def viewport_rect(self):
        """Return the bounding rect of the document's viewport on the
        scene.

        """
        view = self.document.view()
        vsize = view.viewport().size()
        viewportrect = QRect(0, 0, vsize.width(), vsize.height())
        return view.mapToScene(viewportrect).boundingRect()

    def _bound_selection_rect(self, rect):
        """Bound the selection `rect` to a sensible size.
        """
        srect = self.scene.sceneRect()
        vrect = self.viewport_rect()
        maxrect = srect.united(vrect)
        return rect.intersected(maxrect)


class EditNodeLinksAction(UserInteraction):
    def __init__(self, document, source_node, sink_node, *args, **kwargs):
        UserInteraction.__init__(self, document, *args, **kwargs)
        self.source_node = source_node
        self.sink_node = sink_node

    def edit_links(self):
        from ..canvas.editlinksdialog import EditLinksDialog

        log.info("Constructing a Link Editor dialog.")

        parent = self.scene.views()[0]
        dlg = EditLinksDialog(parent)

        links = self.scheme.find_links(source_node=self.source_node,
                                       sink_node=self.sink_node)
        existing_links = [(link.source_channel, link.sink_channel)
                          for link in links]

        dlg.setNodes(self.source_node, self.sink_node)
        dlg.setLinks(existing_links)

        log.info("Executing a Link Editor Dialog.")
        rval = dlg.exec_()

        if rval == EditLinksDialog.Accepted:
            links = dlg.links()

            links_to_add = set(links) - set(existing_links)
            links_to_remove = set(existing_links) - set(links)

            stack = self.document.undoStack()
            stack.beginMacro("Edit Links")

            for source_channel, sink_channel in links_to_remove:
                links = self.scheme.find_links(source_node=self.source_node,
                                               source_channel=source_channel,
                                               sink_node=self.sink_node,
                                               sink_channel=sink_channel)

                self.document.removeLink(links[0])

            for source_channel, sink_channel in links_to_add:
                link = scheme.SchemeLink(self.source_node, source_channel,
                                         self.sink_node, sink_channel)

                self.document.addLink(link)
            stack.endMacro()


def point_to_tuple(point):
    return point.x(), point.y()


class NewArrowAnnotation(UserInteraction):
    """Create a new arrow annotation.
    """
    def __init__(self, document, *args, **kwargs):
        UserInteraction.__init__(self, document, *args, **kwargs)
        self.down_pos = None
        self.arrow_item = None
        self.annotation = None
        self.color = "red"

    def start(self):
        self.document.view().setCursor(Qt.CrossCursor)
        UserInteraction.start(self)

    def setColor(self, color):
        self.color = color

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.down_pos = event.scenePos()
            event.accept()
            return True

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if self.arrow_item is None and \
                    (self.down_pos - event.scenePos()).manhattanLength() > \
                    QApplication.instance().startDragDistance():

                annot = scheme.SchemeArrowAnnotation(
                    point_to_tuple(self.down_pos),
                    point_to_tuple(event.scenePos())
                )
                annot.set_color(self.color)
                item = self.scene.add_annotation(annot)

                self.arrow_item = item
                self.annotation = annot

            if self.arrow_item is not None:
                p1, p2 = map(self.arrow_item.mapFromScene,
                             (self.down_pos, event.scenePos()))
                self.arrow_item.setLine(QLineF(p1, p2))
                self.arrow_item.adjustGeometry()

            event.accept()
            return True

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.arrow_item is not None:
                p1, p2 = self.down_pos, event.scenePos()

                # Commit the annotation to the scheme
                self.annotation.set_line(point_to_tuple(p1),
                                         point_to_tuple(p2))

                self.document.addAnnotation(self.annotation)

                p1, p2 = map(self.arrow_item.mapFromScene, (p1, p2))
                self.arrow_item.setLine(QLineF(p1, p2))
                self.arrow_item.adjustGeometry()

            self.end()
            return True

    def end(self):
        self.down_pos = None
        self.arrow_item = None
        self.annotation = None
        self.document.view().setCursor(Qt.ArrowCursor)
        UserInteraction.end(self)


def rect_to_tuple(rect):
    return rect.x(), rect.y(), rect.width(), rect.height()


class NewTextAnnotation(UserInteraction):
    def __init__(self, document, *args, **kwargs):
        UserInteraction.__init__(self, document, *args, **kwargs)
        self.down_pos = None
        self.annotation_item = None
        self.annotation = None
        self.control = None
        self.font = document.font()

    def setFont(self, font):
        self.font = font

    def start(self):
        self.document.view().setCursor(Qt.CrossCursor)
        UserInteraction.start(self)

    def createNewAnnotation(self, rect):
        """Create a new TextAnnotation at with `rect` as the geometry.
        """
        annot = scheme.SchemeTextAnnotation(rect_to_tuple(rect))
        font = {"family": str(self.font.family()),
                "size": self.font.pointSize()}
        annot.set_font(font)

        item = self.scene.add_annotation(annot)
        item.setTextInteractionFlags(Qt.TextEditorInteraction)
        item.setFramePen(QPen(Qt.DashLine))

        self.annotation_item = item
        self.annotation = annot
        self.control = controlpoints.ControlPointRect()
        self.control.rectChanged.connect(
            self.annotation_item.setGeometry
        )
        self.scene.addItem(self.control)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.down_pos = event.scenePos()
            return True

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if self.annotation_item is None and \
                    (self.down_pos - event.scenePos()).manhattanLength() > \
                    QApplication.instance().startDragDistance():
                rect = QRectF(self.down_pos, event.scenePos()).normalized()
                self.createNewAnnotation(rect)

            if self.annotation_item is not None:
                rect = QRectF(self.down_pos, event.scenePos()).normalized()
                self.control.setRect(rect)

            return True

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.annotation_item is None:
                self.createNewAnnotation(QRectF(event.scenePos(),
                                                event.scenePos()))
                rect = self.defaultTextGeometry(event.scenePos())

            else:
                rect = QRectF(self.down_pos, event.scenePos()).normalized()

            # Commit the annotation to the scheme.
            self.annotation.rect = rect_to_tuple(rect)

            self.document.addAnnotation(self.annotation)

            self.annotation_item.setGeometry(rect)

            self.control.rectChanged.disconnect(
                self.annotation_item.setGeometry
            )
            self.control.hide()

            # Move the focus to the editor.
            self.annotation_item.setFramePen(QPen(Qt.NoPen))
            self.annotation_item.setFocus(Qt.OtherFocusReason)
            self.annotation_item.startEdit()

            self.end()

    def defaultTextGeometry(self, point):
        """Return the default text geometry. Used in case the user
        single clicked in the scene.

        """
        font = self.annotation_item.font()
        metrics = QFontMetrics(font)
        spacing = metrics.lineSpacing()
        margin = self.annotation_item.document().documentMargin()

        rect = QRectF(QPointF(point.x(), point.y() - spacing - margin),
                      QSizeF(150, spacing + 2 * margin))
        return rect

    def end(self):
        if self.control is not None:
            self.scene.removeItem(self.control)

        self.control = None
        self.down_pos = None
        self.annotation_item = None
        self.annotation = None
        self.document.view().setCursor(Qt.ArrowCursor)
        UserInteraction.end(self)


class ResizeTextAnnotation(UserInteraction):
    def __init__(self, document, *args, **kwargs):
        UserInteraction.__init__(self, document, *args, **kwargs)
        self.item = None
        self.annotation = None
        self.control = None
        self.savedFramePen = None
        self.savedRect = None

    def mousePressEvent(self, event):
        pos = event.scenePos()
        if self.item is None:
            item = self.scene.item_at(pos, items.TextAnnotation)
            if item is not None and not item.hasFocus():
                self.editItem(item)
                return False

        return UserInteraction.mousePressEvent(self, event)

    def editItem(self, item):
        annotation = self.scene.annotation_for_item(item)
        rect = item.geometry()  # TODO: map to scene if item has a parent.
        control = controlpoints.ControlPointRect(rect=rect)
        self.scene.addItem(control)

        self.savedFramePen = item.framePen()
        self.savedRect = rect

        control.rectEdited.connect(item.setGeometry)
        control.setFocusProxy(item)

        item.setFramePen(QPen(Qt.DashDotLine))
        item.geometryChanged.connect(self.__on_textGeometryChanged)

        self.item = item

        self.annotation = annotation
        self.control = control

    def commit(self):
        """Commit the current item geometry state to the document.
        """
        rect = self.item.geometry()
        if self.savedRect != rect:
            command = commands.SetAttrCommand(
                self.annotation, "rect",
                (rect.x(), rect.y(), rect.width(), rect.height()),
                name="Edit text geometry"
            )
            self.document.undoStack().push(command)
            self.savedRect = rect

    def __on_editingFinished(self):
        self.commit()
        self.end()

    def __on_rectEdited(self, rect):
        self.item.setGeometry(rect)

    def __on_textGeometryChanged(self):
        if not self.control.isControlActive():
            rect = self.item.geometry()
            self.control.setRect(rect)

    def cancel(self, reason=UserInteraction.OtherReason):
        log.debug("ResizeArrowAnnotation.cancel(%s)", reason)
        if self.item is not None and self.savedRect is not None:
            self.item.setGeometry(self.savedRect)

        UserInteraction.cancel(self, reason)

    def end(self):
        if self.control is not None:
            self.scene.removeItem(self.control)

        if self.item is not None:
            self.item.setFramePen(self.savedFramePen)

        self.item = None
        self.annotation = None
        self.control = None

        UserInteraction.end(self)


class ResizeArrowAnnotation(UserInteraction):
    def __init__(self, document, *args, **kwargs):
        UserInteraction.__init__(self, document, *args, **kwargs)
        self.item = None
        self.annotation = None
        self.control = None
        self.savedLine = None

    def mousePressEvent(self, event):
        pos = event.scenePos()
        if self.item is None:
            item = self.scene.item_at(pos, items.ArrowAnnotation)
            if item is not None and not item.hasFocus():
                self.editItem(item)
                return False

        return UserInteraction.mousePressEvent(self, event)

    def editItem(self, item):
        annotation = self.scene.annotation_for_item(item)
        control = controlpoints.ControlPointLine()
        self.scene.addItem(control)

        line = item.line()
        self.savedLine = line

        p1, p2 = map(item.mapToScene, (line.p1(), line.p2()))

        control.setLine(QLineF(p1, p2))
        control.setFocusProxy(item)
        control.lineEdited.connect(self.__on_lineEdited)

        item.geometryChanged.connect(self.__on_lineGeometryChanged)

        self.item = item
        self.annotation = annotation
        self.control = control

    def commit(self):
        """Commit the current geometry of the item to the document.

        .. note:: Does nothing if the actual geometry is not changed.

        """
        line = self.control.line()
        p1, p2 = line.p1(), line.p2()

        if self.item.line() != self.savedLine:
            command = commands.SetAttrCommand(
                self.annotation,
                "geometry",
                ((p1.x(), p1.y()), (p2.x(), p2.y())),
                name="Edit arrow geometry",
            )
            self.document.undoStack().push(command)
            self.savedLine = self.item.line()

    def __on_editingFinished(self):
        self.commit()
        self.end()

    def __on_lineEdited(self, line):
        p1, p2 = map(self.item.mapFromScene, (line.p1(), line.p2()))
        self.item.setLine(QLineF(p1, p2))
        self.item.adjustGeometry()

    def __on_lineGeometryChanged(self):
        # Possible geometry change from out of our control, for instance
        # item move as a part of a selection group.
        if not self.control.isControlActive():
            line = self.item.line()
            p1, p2 = map(self.item.mapToScene, (line.p1(), line.p2()))
            self.control.setLine(QLineF(p1, p2))

    def cancel(self, reason=UserInteraction.OtherReason):
        log.debug("ResizeArrowAnnotation.cancel(%s)", reason)
        if self.item is not None and self.savedLine is not None:
            self.item.setLine(self.savedLine)

        UserInteraction.cancel(self, reason)

    def end(self):
        if self.control is not None:
            self.scene.removeItem(self.control)

        if self.item is not None:
            self.item.geometryChanged.disconnect(self.__on_lineGeometryChanged)

        self.control = None
        self.item = None
        self.annotation = None

        UserInteraction.end(self)
