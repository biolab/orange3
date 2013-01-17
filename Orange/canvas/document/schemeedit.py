"""
Scheme Edit widget.

"""

import sys
import logging

from operator import attrgetter

from PyQt4.QtGui import (
    QWidget, QVBoxLayout, QInputDialog, QMenu, QAction, QActionGroup,
    QKeySequence, QUndoStack, QGraphicsItem, QGraphicsObject,
    QGraphicsTextItem, QCursor, QFont, QPainter, QPixmap, QColor,
    QIcon, QDesktopServices
)

from PyQt4.QtCore import (
    Qt, QObject, QEvent, QSignalMapper, QRectF, QUrl, QCoreApplication
)

from PyQt4.QtCore import pyqtProperty as Property, pyqtSignal as Signal

from ..registry.qt import whats_this_helper
from ..gui.quickhelp import QuickHelpTipEvent
from ..gui.utils import message_information, disabled
from ..scheme import scheme
from ..canvas.scene import CanvasScene
from ..canvas.view import CanvasView
from ..canvas import items
from . import interactions
from . import commands
from . import quickmenu


log = logging.getLogger(__name__)


# TODO: Should this be moved to CanvasScene?
class GraphicsSceneFocusEventListener(QGraphicsObject):

    itemFocusedIn = Signal(QGraphicsItem)
    itemFocusedOut = Signal(QGraphicsItem)

    def __init__(self, parent=None):
        QGraphicsObject.__init__(self, parent)
        self.setFlag(QGraphicsItem.ItemHasNoContents)

    def sceneEventFilter(self, obj, event):
        if event.type() == QEvent.FocusIn and \
                obj.flags() & QGraphicsItem.ItemIsFocusable:
            obj.focusInEvent(event)
            if obj.hasFocus():
                self.itemFocusedIn.emit(obj)
            return True
        elif event.type() == QEvent.FocusOut:
            obj.focusOutEvent(event)
            if not obj.hasFocus():
                self.itemFocusedOut.emit(obj)
            return True

        return QGraphicsObject.sceneEventFilter(self, obj, event)

    def boundingRect(self):
        return QRectF()


class SchemeEditWidget(QWidget):
    undoAvailable = Signal(bool)
    redoAvailable = Signal(bool)
    modificationChanged = Signal(bool)
    undoCommandAdded = Signal()
    selectionChanged = Signal()

    titleChanged = Signal(str)

    pathChanged = Signal(str)

    # Quick Menu triggers
    (NoTriggers,
     Clicked,
     DoubleClicked,
     SpaceKey,
     AnyKey) = [0, 1, 2, 4, 8]

    def __init__(self, parent=None, ):
        QWidget.__init__(self, parent)

        self.__modified = False
        self.__registry = None
        self.__scheme = None
        self.__path = ""
        self.__quickMenuTriggers = SchemeEditWidget.SpaceKey | \
                                   SchemeEditWidget.DoubleClicked
        self.__emptyClickButtons = 0
        self.__channelNamesVisible = True
        self.__possibleSelectionHandler = None
        self.__possibleMouseItemsMove = False
        self.__itemsMoving = {}
        self.__contextMenuTarget = None
        self.__quickMenu = None
        self.__quickTip = ""

        self.__undoStack = QUndoStack(self)
        self.__undoStack.cleanChanged[bool].connect(self.__onCleanChanged)

        # OWBaseWidget properties when set to clean state
        self.__cleanSettings = []

        self.__editFinishedMapper = QSignalMapper(self)
        self.__editFinishedMapper.mapped[QObject].connect(
            self.__onEditingFinished
        )

        self.__annotationGeomChanged = QSignalMapper(self)

        self.__setupActions()
        self.__setupUi()

        self.__editMenu = QMenu(self.tr("&Edit"), self)
        self.__editMenu.addAction(self.__undoAction)
        self.__editMenu.addAction(self.__redoAction)
        self.__editMenu.addSeparator()
        self.__editMenu.addAction(self.__selectAllAction)

        self.__widgetMenu = QMenu(self.tr("&Widget"), self)
        self.__widgetMenu.addAction(self.__openSelectedAction)
        self.__widgetMenu.addSeparator()
        self.__widgetMenu.addAction(self.__renameAction)
        self.__widgetMenu.addAction(self.__removeSelectedAction)
        self.__widgetMenu.addSeparator()
        self.__widgetMenu.addAction(self.__helpAction)

        self.__linkMenu = QMenu(self.tr("Link"), self)
        self.__linkMenu.addAction(self.__linkEnableAction)
        self.__linkMenu.addSeparator()
        self.__linkMenu.addAction(self.__linkRemoveAction)
        self.__linkMenu.addAction(self.__linkResetAction)

    def __setupActions(self):

        self.__zoomAction = \
            QAction(self.tr("Zoom"), self,
                    objectName="zoom-action",
                    checkable=True,
                    shortcut=QKeySequence.ZoomIn,
                    toolTip=self.tr("Zoom in the scheme."),
                    toggled=self.toogleZoom,
                    )

        self.__cleanUpAction = \
            QAction(self.tr("Clean Up"), self,
                    objectName="cleanup-action",
                    toolTip=self.tr("Align widget to a grid."),
                    triggered=self.alignToGrid,
                    )

        self.__newTextAnnotationAction = \
            QAction(self.tr("Text"), self,
                    objectName="new-text-action",
                    toolTip=self.tr("Add a text annotation to the scheme."),
                    checkable=True,
                    toggled=self.__toggleNewTextAnnotation,
                    )

        # Create a font size menu for the new annotation action.
        self.__fontMenu = QMenu("Font Size", self)
        self.__fontActionGroup = group = \
            QActionGroup(self, exclusive=True,
                         triggered=self.__onFontSizeTriggered)

        def font(size):
            return QFont("Helvetica", size)

        for size in [12, 14, 16, 18, 20, 22, 24]:
            action = QAction("%ip" % size, group,
                             checkable=True,
                             font=font(size))

            self.__fontMenu.addAction(action)

        group.actions()[2].setChecked(True)

        self.__newTextAnnotationAction.setMenu(self.__fontMenu)

        self.__newArrowAnnotationAction = \
            QAction(self.tr("Arrow"), self,
                    objectName="new-arrow-action",
                    toolTip=self.tr("Add a arrow annotation to the scheme."),
                    checkable=True,
                    toggled=self.__toggleNewArrowAnnotation,
                    )

        # Create a color menu for the arrow annotation action
        self.__arrowColorMenu = QMenu("Arrow Color",)
        self.__arrowColorActionGroup = group = \
            QActionGroup(self, exclusive=True,
                         triggered=self.__onArrowColorTriggered)

        def color_icon(color):
            icon = QIcon()
            for size in [16, 24, 32]:
                pixmap = QPixmap(size, size)
                pixmap.fill(QColor(0, 0, 0, 0))
                p = QPainter(pixmap)
                p.setRenderHint(QPainter.Antialiasing)
                p.setBrush(color)
                p.setPen(Qt.NoPen)
                p.drawEllipse(1, 1, size - 2, size - 2)
                p.end()
                icon.addPixmap(pixmap)
            return icon

        for color in ["#000", "#C1272D", "#662D91", "#1F9CDF", "#39B54A"]:
            icon = color_icon(QColor(color))
            action = QAction(group, icon=icon, checkable=True,
                             iconVisibleInMenu=True)
            action.setData(color)
            self.__arrowColorMenu.addAction(action)

        group.actions()[1].setChecked(True)

        self.__newArrowAnnotationAction.setMenu(self.__arrowColorMenu)

        self.__undoAction = self.__undoStack.createUndoAction(self)
        self.__undoAction.setShortcut(QKeySequence.Undo)
        self.__undoAction.setObjectName("undo-action")

        self.__redoAction = self.__undoStack.createRedoAction(self)
        self.__redoAction.setShortcut(QKeySequence.Redo)
        self.__redoAction.setObjectName("redo-action")

        self.__selectAllAction = \
            QAction(self.tr("Select all"), self,
                    objectName="select-all-action",
                    toolTip=self.tr("Select all items."),
                    triggered=self.selectAll,
                    shortcut=QKeySequence.SelectAll
                    )

        self.__openSelectedAction = \
            QAction(self.tr("Open"), self,
                    objectName="open-action",
                    toolTip=self.tr("Open selected widget"),
                    triggered=self.openSelected,
                    enabled=False)

        self.__removeSelectedAction = \
            QAction(self.tr("Remove"), self,
                    objectName="remove-selected",
                    toolTip=self.tr("Remove selected items"),
                    triggered=self.removeSelected,
                    enabled=False
                    )

        shortcuts = [Qt.Key_Delete,
                     Qt.ControlModifier + Qt.Key_Backspace]

        if sys.platform == "darwin":
            # Command Backspace should be the first
            # (visible shortcut in the menu)
            shortcuts.reverse()

        self.__removeSelectedAction.setShortcuts(shortcuts)

        self.__renameAction = \
            QAction(self.tr("Rename"), self,
                    objectName="rename-action",
                    toolTip=self.tr("Rename selected widget"),
                    triggered=self.__onRenameAction,
                    shortcut=QKeySequence(Qt.Key_F2),
                    enabled=False)

        self.__helpAction = \
            QAction(self.tr("Help"), self,
                    objectName="help-action",
                    toolTip=self.tr("Show widget help"),
                    triggered=self.__onHelpAction,
                    shortcut=QKeySequence.HelpContents
                    )

        self.__linkEnableAction = \
            QAction(self.tr("Enabled"), self,
                    objectName="link-enable-action",
                    triggered=self.__toggleLinkEnabled,
                    checkable=True,
                    )

        self.__linkRemoveAction = \
            QAction(self.tr("Remove"), self,
                    objectName="link-remove-action",
                    triggered=self.__linkRemove,
                    toolTip=self.tr("Remove link."),
                    )

        self.__linkResetAction = \
            QAction(self.tr("Reset Signals"), self,
                    objectName="link-reset-action",
                    triggered=self.__linkReset,
                    )

        self.addActions([self.__newTextAnnotationAction,
                         self.__newArrowAnnotationAction,
                         self.__linkEnableAction,
                         self.__linkRemoveAction,
                         self.__linkResetAction])

        # Actions which should be disabled while a multistep
        # interaction is in progress.
        self.__disruptiveActions = \
                [self.__undoAction,
                 self.__redoAction,
                 self.__removeSelectedAction,
                 self.__selectAllAction]

    def __setupUi(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scene = CanvasScene()
        scene.set_channel_names_visible(self.__channelNamesVisible)

        view = CanvasView(scene)
        view.setFrameStyle(CanvasView.NoFrame)
        view.setRenderHint(QPainter.Antialiasing)
        view.setContextMenuPolicy(Qt.CustomContextMenu)
        view.customContextMenuRequested.connect(
            self.__onCustomContextMenuRequested
        )

        self.__view = view
        self.__scene = scene

        self.__focusListener = GraphicsSceneFocusEventListener()
        self.__focusListener.itemFocusedIn.connect(self.__onItemFocusedIn)
        self.__focusListener.itemFocusedOut.connect(self.__onItemFocusedOut)
        self.__scene.addItem(self.__focusListener)

        self.__scene.selectionChanged.connect(
            self.__onSelectionChanged
        )

        layout.addWidget(view)
        self.setLayout(layout)

    def toolbarActions(self):
        """Return a list of actions that can be inserted into a toolbar.
        """
        return [self.__zoomAction,
                self.__cleanUpAction,
                self.__newTextAnnotationAction,
                self.__newArrowAnnotationAction]

    def menuBarActions(self):
        """Return a list of actions that can be inserted into a QMenuBar.
        These actions should have a menu.

        """
        return [self.__editMenu.menuAction(), self.__widgetMenu.menuAction()]

    def isModified(self):
        """Is the document modified.
        """
        return self.__modified or not self.__undoStack.isClean()

    def setModified(self, modified):
        if self.__modified != modified:
            self.__modified = modified

        if not modified:
            self.__cleanSettings = self.__scheme.widget_settings()
            self.__undoStack.setClean()
        else:
            self.__cleanSettings = []

    modified = Property(bool, fget=isModified, fset=setModified)

    def isModifiedStrict(self):
        """Is the document modified. Run a strict check against all node
        properties as they were at the time when the last call to
        `setModified(True)` was made.

        """
        settingsChanged = self.__cleanSettings != \
                          self.__scheme.widget_settings()

        log.debug("Modified strict check (modified flag: %s, "
                  "undo stack clean: %s, properties: %s)",
                  self.__modified,
                  self.__undoStack.isClean(),
                  settingsChanged)

        return self.isModified() or settingsChanged

    def setQuickMenuTriggers(self, triggers):
        """Set quick menu triggers.
        """
        if self.__quickMenuTriggers != triggers:
            self.__quickMenuTriggers = triggers

    def quickMenuTriggres(self):
        return self.__quickMenuTriggers

    def setChannelNamesVisible(self, visible):
        """Set channel names visibility state. When enabled the links
        in the view will have a visible source/sink channel names
        displayed over them.

        """
        if self.__channelNamesVisible != visible:
            self.__channelNamesVisible = visible
            self.__scene.set_channel_names_visible(visible)

    def channelNamesVisible(self):
        """Return the channel name visibility state.
        """
        return self.__channelNamesVisible

    def undoStack(self):
        """Return the undo stack.
        """
        return self.__undoStack

    def setPath(self, path):
        """Set the path associated with the current scheme.

        .. note:: Calling `setScheme` will invalidate the path (i.e. set it
                  to an empty string)

        """
        if self.__path != path:
            self.__path = str(path)
            self.pathChanged.emit(self.__path)

    def path(self):
        """Return the path associated with the scene
        """
        return self.__path

    def setScheme(self, scheme):
        if self.__scheme is not scheme:
            if self.__scheme:
                self.__scheme.title_changed.disconnect(self.titleChanged)
                self.__scheme.node_added.disconnect(self.__onNodeAdded)
                self.__scheme.node_removed.disconnect(self.__onNodeRemoved)

            self.__scheme = scheme

            self.setPath("")

            if self.__scheme:
                self.__scheme.title_changed.connect(self.titleChanged)
                self.__scheme.node_added.connect(self.__onNodeAdded)
                self.__scheme.node_removed.connect(self.__onNodeRemoved)
                self.titleChanged.emit(scheme.title)
                self.__cleanSettings = scheme.widget_settings()
            else:
                self.__cleanSettings = []

            # Clear the current item selection in the scene so edit action
            # states are updated accordingly.
            self.__scene.clearSelection()

            self.__annotationGeomChanged.deleteLater()
            self.__annotationGeomChanged = QSignalMapper(self)

            self.__undoStack.clear()

            self.__focusListener.itemFocusedIn.disconnect(
                self.__onItemFocusedIn
            )
            self.__focusListener.itemFocusedOut.disconnect(
                self.__onItemFocusedOut
            )

            self.__scene.selectionChanged.disconnect(
                self.__onSelectionChanged
            )

            self.__scene.clear()
            self.__scene.removeEventFilter(self)
            self.__scene.deleteLater()

            self.__scene = CanvasScene()
            self.__view.setScene(self.__scene)
            self.__scene.set_channel_names_visible(self.__channelNamesVisible)

            self.__scene.installEventFilter(self)

            self.__scene.set_registry(self.__registry)

            # Focus listener
            self.__focusListener = GraphicsSceneFocusEventListener()
            self.__focusListener.itemFocusedIn.connect(
                self.__onItemFocusedIn
            )
            self.__focusListener.itemFocusedOut.connect(
                self.__onItemFocusedOut
            )
            self.__scene.addItem(self.__focusListener)

            self.__scene.selectionChanged.connect(
                self.__onSelectionChanged
            )

            self.__scene.node_item_activated.connect(
                self.__onNodeActivate
            )

            self.__scene.annotation_added.connect(
                self.__onAnnotationAdded
            )

            self.__scene.annotation_removed.connect(
                self.__onAnnotationRemoved
            )

            self.__scene.set_scheme(scheme)

    def scheme(self):
        """Return the :class:`Scheme` edited by the widget.
        """
        return self.__scheme

    def scene(self):
        """Return the QGraphicsScene instance used to display the scheme.
        """
        return self.__scene

    def view(self):
        """Return the QGraphicsView instance used to display the scene.
        """
        return self.__view

    def setRegistry(self, registry):
        # Is this method necessary
        self.__registry = registry
        if self.__scene:
            self.__scene.set_registry(registry)
            self.__quickMenu = None

    def quickMenu(self):
        """Return a quick menu instance for quick new node creation.
        """
        if self.__quickMenu is None:
            menu = quickmenu.QuickMenu(self)
            if self.__registry is not None:
                menu.setModel(self.__registry.model())
            self.__quickMenu = menu
        return self.__quickMenu

    def setTitle(self, title):
        self.__undoStack.push(
            commands.SetAttrCommand(self.__scheme, "title", title)
        )

    def setDescription(self, description):
        self.__undoStack.push(
            commands.SetAttrCommand(self.__scheme, "description", description)
        )

    def addNode(self, node):
        """Add a new node to the scheme.
        """
        command = commands.AddNodeCommand(self.__scheme, node)
        self.__undoStack.push(command)

    def createNewNode(self, description):
        """Create a new `SchemeNode` and add it to the document at left of the
        last added node.

        """
        node = scheme.SchemeNode(description)

        if self.scheme().nodes:
            x, y = self.scheme().nodes[-1].position
            node.position = (x + 150, y)
        else:
            node.position = (150, 150)

        self.addNode(node)

    def removeNode(self, node):
        """Remove a `node` (:class:`SchemeNode`) from the scheme
        """
        command = commands.RemoveNodeCommand(self.__scheme, node)
        self.__undoStack.push(command)

    def renameNode(self, node, title):
        """Rename a `node` (:class:`SchemeNode`) to `title`.
        """
        command = commands.RenameNodeCommand(self.__scheme, node, title)
        self.__undoStack.push(command)

    def addLink(self, link):
        """Add a `link` (:class:`SchemeLink`) to the scheme.
        """
        command = commands.AddLinkCommand(self.__scheme, link)
        self.__undoStack.push(command)

    def removeLink(self, link):
        """Remove a link (:class:`SchemeLink`) from the scheme.
        """
        command = commands.RemoveLinkCommand(self.__scheme, link)
        self.__undoStack.push(command)

    def addAnnotation(self, annotation):
        """Add `annotation` (:class:`BaseSchemeAnnotation`) to the scheme
        """
        command = commands.AddAnnotationCommand(self.__scheme, annotation)
        self.__undoStack.push(command)

    def removeAnnotation(self, annotation):
        """Remove `annotation` (:class:`BaseSchemeAnnotation`) from the scheme.
        """
        command = commands.RemoveAnnotationCommand(self.__scheme, annotation)
        self.__undoStack.push(command)

    def removeSelected(self):
        """Remove all selected items in the scheme.
        """
        selected = self.scene().selectedItems()
        if not selected:
            return

        self.__undoStack.beginMacro(self.tr("Remove"))
        for item in selected:
            if isinstance(item, items.NodeItem):
                node = self.scene().node_for_item(item)
                self.__undoStack.push(
                    commands.RemoveNodeCommand(self.__scheme, node)
                )
            elif isinstance(item, items.annotationitem.Annotation):
                annot = self.scene().annotation_for_item(item)
                self.__undoStack.push(
                    commands.RemoveAnnotationCommand(self.__scheme, annot)
                )
        self.__undoStack.endMacro()

    def selectAll(self):
        """Select all selectable items in the scheme.
        """
        for item in self.__scene.items():
            if item.flags() & QGraphicsItem.ItemIsSelectable:
                item.setSelected(True)

    def toogleZoom(self, zoom):
        view = self.view()
        if zoom:
            view.scale(1.5, 1.5)
        else:
            view.resetTransform()

    def alignToGrid(self):
        """Align nodes to a grid.
        """
        tile_size = 150
        tiles = {}

        nodes = sorted(self.scheme().nodes, key=attrgetter("position"))

        if nodes:
            self.__undoStack.beginMacro(self.tr("Align To Grid"))

            for node in nodes:
                x, y = node.position
                x = int(round(float(x) / tile_size) * tile_size)
                y = int(round(float(y) / tile_size) * tile_size)
                while (x, y) in tiles:
                    x += tile_size

                self.__undoStack.push(
                    commands.MoveNodeCommand(self.scheme(), node,
                                             node.position, (x, y))
                )

                tiles[x, y] = node
                self.__scene.item_for_node(node).setPos(x, y)

            self.__undoStack.endMacro()

    def selectedNodes(self):
        """Return all selected `SchemeNode` items.
        """
        return list(map(self.scene().node_for_item,
                        self.scene().selected_node_items()))

    def selectedAnnotations(self):
        """Return all selected `SchemeAnnotation` items.
        """
        return list(map(self.scene().annotation_for_item,
                   self.scene().selected_annotation_items()))

    def openSelected(self):
        """Open (show and raise) all widgets for selected nodes.
        """
        selected = self.scene().selected_node_items()
        for item in selected:
            self.__onNodeActivate(item)

    def editNodeTitle(self, node):
        """Edit the `node`'s title.
        """
        name, ok = QInputDialog.getText(
                    self, self.tr("Rename"),
                    str(self.tr("Enter a new name for the %r widget")) \
                    % node.title,
                    text=node.title
                    )

        if ok:
            self.__undoStack.push(
                commands.RenameNodeCommand(self.__scheme, node, node.title,
                                           str(name))
            )

    def __onCleanChanged(self, clean):
        if self.isWindowModified() != (not clean):
            self.setWindowModified(not clean)
            self.modificationChanged.emit(not clean)

    def eventFilter(self, obj, event):
        # Filter the scene's drag/drop events.
        if obj is self.scene():
            etype = event.type()
            if  etype == QEvent.GraphicsSceneDragEnter or \
                    etype == QEvent.GraphicsSceneDragMove:
                mime_data = event.mimeData()
                if mime_data.hasFormat(
                        "application/vnv.orange-canvas.registry.qualified-name"
                        ):
                    event.acceptProposedAction()
                return True
            elif etype == QEvent.GraphicsSceneDrop:
                data = event.mimeData()
                qname = data.data(
                    "application/vnv.orange-canvas.registry.qualified-name"
                )
                desc = self.__registry.widget(bytes(qname).decode())
                pos = event.scenePos()
                node = scheme.SchemeNode(desc, position=(pos.x(), pos.y()))
                self.addNode(node)
                return True

            elif etype == QEvent.GraphicsSceneMousePress:
                return self.sceneMousePressEvent(event)
            elif etype == QEvent.GraphicsSceneMouseMove:
                return self.sceneMouseMoveEvent(event)
            elif etype == QEvent.GraphicsSceneMouseRelease:
                return self.sceneMouseReleaseEvent(event)
            elif etype == QEvent.GraphicsSceneMouseDoubleClick:
                return self.sceneMouseDoubleClickEvent(event)
            elif etype == QEvent.KeyRelease:
                return self.sceneKeyPressEvent(event)
            elif etype == QEvent.KeyRelease:
                return self.sceneKeyReleaseEvent(event)
            elif etype == QEvent.GraphicsSceneContextMenu:
                return self.sceneContextMenuEvent(event)

        return QWidget.eventFilter(self, obj, event)

    def sceneMousePressEvent(self, event):
        scene = self.__scene
        if scene.user_interaction_handler:
            return False

        pos = event.scenePos()

        anchor_item = scene.item_at(pos, items.NodeAnchorItem,
                                    buttons=Qt.LeftButton)
        if anchor_item and event.button() == Qt.LeftButton:
            # Start a new link starting at item
            scene.clearSelection()
            handler = interactions.NewLinkAction(self)
            self._setUserInteractionHandler(handler)
            return handler.mousePressEvent(event)

        any_item = scene.item_at(pos)
        if not any_item and event.button() == Qt.LeftButton:
            self.__emptyClickButtons |= Qt.LeftButton
            # Create a RectangleSelectionAction but do not set in on the scene
            # just yet (instead wait for the mouse move event).
            handler = interactions.RectangleSelectionAction(self)
            rval = handler.mousePressEvent(event)
            if rval == True:
                self.__possibleSelectionHandler = handler
            return rval

        if any_item and event.button() == Qt.LeftButton:
            self.__possibleMouseItemsMove = True
            self.__itemsMoving.clear()
            self.__scene.node_item_position_changed.connect(
                self.__onNodePositionChanged
            )
            self.__annotationGeomChanged.mapped[QObject].connect(
                self.__onAnnotationGeometryChanged
            )

            set_enabled_all(self.__disruptiveActions, False)

        return False

    def sceneMouseMoveEvent(self, event):
        scene = self.__scene
        if scene.user_interaction_handler:
            return False

        if self.__emptyClickButtons & Qt.LeftButton and \
                event.buttons() & Qt.LeftButton and \
                self.__possibleSelectionHandler:
            # Set the RectangleSelection (initialized in mousePressEvent)
            # on the scene
            handler = self.__possibleSelectionHandler
            self._setUserInteractionHandler(handler)
            return handler.mouseMoveEvent(event)

        return False

    def sceneMouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.__possibleMouseItemsMove:
            self.__possibleMouseItemsMove = False
            self.__scene.node_item_position_changed.disconnect(
                self.__onNodePositionChanged
            )
            self.__annotationGeomChanged.mapped[QObject].disconnect(
                self.__onAnnotationGeometryChanged
            )

            set_enabled_all(self.__disruptiveActions, True)

            if self.__itemsMoving:
                self.__scene.mouseReleaseEvent(event)
                stack = self.undoStack()
                stack.beginMacro(self.tr("Move"))
                for scheme_item, (old, new) in self.__itemsMoving.items():
                    if isinstance(scheme_item, scheme.SchemeNode):
                        command = commands.MoveNodeCommand(
                            self.scheme(), scheme_item, old, new
                        )
                    elif isinstance(scheme_item, scheme.BaseSchemeAnnotation):
                        command = commands.AnnotationGeometryChange(
                            self.scheme(), scheme_item, old, new
                        )
                    else:
                        continue

                    stack.push(command)
                stack.endMacro()

                self.__itemsMoving.clear()
                return True

        if self.__emptyClickButtons & Qt.LeftButton and \
                event.button() & Qt.LeftButton:
            self.__emptyClickButtons &= ~Qt.LeftButton

            if self.__quickMenuTriggers & SchemeEditWidget.Clicked and \
                    mouse_drag_distance(event, Qt.LeftButton) < 1:
                action = interactions.NewNodeAction(self)

                with (disabled(self.__undoAction),
                      disabled(self.__redoAction)):
                    action.create_new(event.screenPos())

                event.accept()
                return True

        return False

    def sceneMouseDoubleClickEvent(self, event):
        scene = self.__scene
        if scene.user_interaction_handler:
            return False

        item = scene.item_at(event.scenePos())
        if not item and self.__quickMenuTriggers & \
                SchemeEditWidget.DoubleClicked:
            # Double click on an empty spot
            # Create a new node using QuickMenu
            action = interactions.NewNodeAction(self)

            with (disabled(self.__undoAction),
                  disabled(self.__redoAction)):
                action.create_new(event.screenPos())

            event.accept()
            return True

        item = scene.item_at(event.scenePos(), items.LinkItem,
                             buttons=Qt.LeftButton)

        if item is not None and event.button() == Qt.LeftButton:
            link = self.scene().link_for_item(item)
            action = interactions.EditNodeLinksAction(self, link.source_node,
                                                      link.sink_node)
            action.edit_links()
            event.accept()
            return True

        return False

    def sceneKeyPressEvent(self, event):
        scene = self.__scene
        if scene.user_interaction_handler:
            return False

        # If a QGraphicsItem is in text editing mode, don't interrupt it
        focusItem = scene.focusItem()
        if focusItem and isinstance(focusItem, QGraphicsTextItem) and \
                focusItem.textInteractionFlags() & Qt.TextEditable:
            return False

        # If the mouse is not over out view
        if not self.view().underMouse():
            return False

        handler = None
        if (event.key() == Qt.Key_Space and \
                self.__quickMenuTriggers & SchemeEditWidget.SpaceKey):
            handler = interactions.NewNodeAction(self)

        elif len(event.text()) and \
                self.__quickMenuTriggers & SchemeEditWidget.AnyKey:
            handler = interactions.NewNodeAction(self)
            # TODO: set the search text to event.text() and set focus on the
            # search line

        if handler is not None:
            # Control + Backspace (remove widget action on Mac OSX) conflicts
            # with the 'Clear text' action in the search widget (there might
            # be selected items in the canvas), so we disable the
            # remove widget action so the text editing follows standard
            # 'look and feel'
            with (disabled(self.__removeSelectedAction),
                  disabled(self.__undoAction),
                  disabled(self.__redoAction)):
                handler.create_new(QCursor.pos())

            event.accept()
            return True

        return False

    def sceneKeyReleaseEvent(self, event):
        return False

    def sceneContextMenuEvent(self, event):
        return False

    def _setUserInteractionHandler(self, handler):
        """Helper method for setting the user interaction handlers.
        """
        if self.__scene.user_interaction_handler:
            if isinstance(self.__scene.user_interaction_handler,
                          (interactions.ResizeArrowAnnotation,
                           interactions.ResizeTextAnnotation)):
                self.__scene.user_interaction_handler.commit()

            self.__scene.user_interaction_handler.ended.disconnect(
                self.__onInteractionEnded
            )

        if handler:
            handler.ended.connect(self.__onInteractionEnded)
            # Disable actions which could change the model
            set_enabled_all(self.__disruptiveActions, False)

        self.__scene.set_user_interaction_handler(handler)

    def __onInteractionEnded(self):
        self.sender().ended.disconnect(self.__onInteractionEnded)
        set_enabled_all(self.__disruptiveActions, True)

    def __onSelectionChanged(self):
        nodes = self.selectedNodes()
        annotations = self.selectedAnnotations()

        self.__openSelectedAction.setEnabled(bool(nodes))
        self.__removeSelectedAction.setEnabled(
            bool(nodes) or bool(annotations)
        )

        self.__helpAction.setEnabled(len(nodes) == 1)
        self.__renameAction.setEnabled(len(nodes) == 1)

        if len(nodes) > 1:
            self.__openSelectedAction.setText(self.tr("Open All"))
        else:
            self.__openSelectedAction.setText(self.tr("Open"))

        if len(nodes) + len(annotations) > 1:
            self.__removeSelectedAction.setText(self.tr("Remove All"))
        else:
            self.__removeSelectedAction.setText(self.tr("Remove"))

        if len(nodes) == 0:
            self.__openSelectedAction.setText(self.tr("Open"))
            self.__removeSelectedAction.setText(self.tr("Remove"))

        focus = self.__scene.focusItem()
        if isinstance(focus, items.NodeItem):
            node = self.__scene.node_for_item(focus)
            desc = node.description
            tip = whats_this_helper(desc)
        else:
            tip = ""

        if tip != self.__quickTip:
            self.__quickTip = tip
            ev = QuickHelpTipEvent("", self.__quickTip,
                                   priority=QuickHelpTipEvent.Permanent)

            QCoreApplication.sendEvent(self, ev)

    def __onNodeAdded(self, node):
        widget = self.__scheme.widget_for_node[node]
        widget.widgetStateChanged.connect(self.__onWidgetStateChanged)

    def __onNodeRemoved(self, node):
        widget = self.__scheme.widget_for_node[node]
        widget.widgetStateChanged.disconnect(self.__onWidgetStateChanged)

    def __onWidgetStateChanged(self, *args):
        widget = self.sender()
        self.scheme()
        widget_to_node = dict((v, k) for k,v in
                              self.__scheme.widget_for_node.items())
        node = widget_to_node[widget]
        item = self.__scene.item_for_node(node)

        info = widget.widgetStateToHtml(True, False, False)
        warning = widget.widgetStateToHtml(False, True, False)
        error = widget.widgetStateToHtml(False, False, True)

        item.setInfoMessage(info or None)
        item.setWarningMessage(warning or None)
        item.setErrorMessage(error or None)

    def __onNodeActivate(self, item):
        node = self.__scene.node_for_item(item)
        widget = self.scheme().widget_for_node[node]
        widget.show()
        widget.raise_()

    def __onNodePositionChanged(self, item, pos):
        node = self.__scene.node_for_item(item)
        new = (pos.x(), pos.y())
        if node not in self.__itemsMoving:
            self.__itemsMoving[node] = (node.position, new)
        else:
            old, _ = self.__itemsMoving[node]
            self.__itemsMoving[node] = (old, new)

    def __onAnnotationGeometryChanged(self, item):
        annot = self.scene().annotation_for_item(item)
        if annot not in self.__itemsMoving:
            self.__itemsMoving[annot] = (annot.geometry,
                                         geometry_from_annotation_item(item))
        else:
            old, _ = self.__itemsMoving[annot]
            self.__itemsMoving[annot] = (old,
                                         geometry_from_annotation_item(item))

    def __onAnnotationAdded(self, item):
        log.debug("Annotation added (%r)", item)
        item.setFlag(QGraphicsItem.ItemIsSelectable)
        item.setFlag(QGraphicsItem.ItemIsMovable)
        item.setFlag(QGraphicsItem.ItemIsFocusable)

        item.installSceneEventFilter(self.__focusListener)

        if isinstance(item, items.ArrowAnnotation):
            pass
        elif isinstance(item, items.TextAnnotation):
            # Make the annotation editable.
            item.setTextInteractionFlags(Qt.TextEditorInteraction)

            self.__editFinishedMapper.setMapping(item, item)
            item.editingFinished.connect(
                self.__editFinishedMapper.map
            )

        self.__annotationGeomChanged.setMapping(item, item)
        item.geometryChanged.connect(
            self.__annotationGeomChanged.map
        )

    def __onAnnotationRemoved(self, item):
        log.debug("Annotation removed (%r)", item)
        if isinstance(item, items.ArrowAnnotation):
            pass
        elif isinstance(item, items.TextAnnotation):
            item.editingFinished.disconnect(
                self.__editFinishedMapper.map
            )

        item.removeSceneEventFilter(self.__focusListener)

        self.__annotationGeomChanged.removeMappings(item)
        item.geometryChanged.disconnect(
            self.__annotationGeomChanged.map
        )

    def __onItemFocusedIn(self, item):
        """Annotation item has gained focus.
        """
        if not self.__scene.user_interaction_handler:
            self.__startControlPointEdit(item)

    def __onItemFocusedOut(self, item):
        """Annotation item lost focus.
        """
        self.__endControlPointEdit()

    def __onEditingFinished(self, item):
        """Text annotation editing has finished.
        """
        annot = self.__scene.annotation_for_item(item)
        text = str(item.toPlainText())
        if annot.text != text:
            self.__undoStack.push(
                commands.TextChangeCommand(self.scheme(), annot,
                                           annot.text, text)
            )

    def __toggleNewArrowAnnotation(self, checked):
        if self.__newTextAnnotationAction.isChecked():
            self.__newTextAnnotationAction.setChecked(not checked)

        action = self.__newArrowAnnotationAction

        if not checked:
            handler = self.__scene.user_interaction_handler
            if isinstance(handler, interactions.NewArrowAnnotation):
                # Cancel the interaction and restore the state
                handler.ended.disconnect(action.toggle)
                handler.cancel(interactions.UserInteraction.UserCancelReason)
                log.info("Canceled new arrow annotation")

        else:
            handler = interactions.NewArrowAnnotation(self)
            checked = self.__arrowColorActionGroup.checkedAction()
            handler.setColor(checked.data())

            handler.ended.connect(action.toggle)

            self._setUserInteractionHandler(handler)

    def __onFontSizeTriggered(self, action):
        if not self.__newTextAnnotationAction.isChecked():
            # Trigger the action
            self.__newTextAnnotationAction.trigger()
        else:
            # just update the preferred font on the interaction handler
            handler = self.__scene.user_interaction_handler
            if isinstance(handler, interactions.NewTextAnnotation):
                handler.setFont(action.font())

    def __toggleNewTextAnnotation(self, checked):
        if self.__newArrowAnnotationAction.isChecked():
            self.__newArrowAnnotationAction.setChecked(not checked)

        action = self.__newTextAnnotationAction

        if not checked:
            handler = self.__scene.user_interaction_handler
            if isinstance(handler, interactions.NewTextAnnotation):
                # cancel the interaction and restore the state
                handler.ended.disconnect(action.toggle)
                handler.cancel(interactions.UserInteraction.UserCancelReason)
                log.info("Canceled new text annotation")

        else:
            handler = interactions.NewTextAnnotation(self)
            checked = self.__fontActionGroup.checkedAction()
            handler.setFont(checked.font())

            handler.ended.connect(action.toggle)

            self._setUserInteractionHandler(handler)

    def __onArrowColorTriggered(self, action):
        if not self.__newArrowAnnotationAction.isChecked():
            # Trigger the action
            self.__newArrowAnnotationAction.trigger()
        else:
            # just update the preferred color on the interaction handler
            handler = self.__scene.user_interaction_handler
            if isinstance(handler, interactions.NewArrowAnnotation):
                handler.setColor(action.data())

    def __onCustomContextMenuRequested(self, pos):
        scenePos = self.view().mapToScene(pos)
        globalPos = self.view().mapToGlobal(pos)

        item = self.scene().item_at(scenePos, items.NodeItem)
        if item is not None:
            self.__widgetMenu.popup(globalPos)
            return

        item = self.scene().item_at(scenePos, items.LinkItem,
                                    buttons=Qt.RightButton)
        if item is not None:
            link = self.scene().link_for_item(item)
            self.__linkEnableAction.setChecked(link.enabled)
            self.__contextMenuTarget = link
            self.__linkMenu.popup(globalPos)
            return

    def __onRenameAction(self):
        selected = self.selectedNodes()
        if len(selected) == 1:
            self.editNodeTitle(selected[0])

    def __onHelpAction(self):
        nodes = self.selectedNodes()
        help_url = None
        if len(nodes) == 1:
            node = nodes[0]
            desc = node.description
            if desc.help:
                help_url = desc.help

        if help_url is not None:
            QDesktopServices.openUrl(QUrl(help_url))
        else:
            message_information(
                self.tr("Sorry there is documentation available for "
                        "this widget."),
                parent=self)

    def __toggleLinkEnabled(self, enabled):
        """Link enabled state was toggled in the context menu.
        """
        if self.__contextMenuTarget:
            link = self.__contextMenuTarget
            command = commands.SetAttrCommand(
                link, "enabled", enabled, name=self.tr("Set enabled"),
            )
            self.__undoStack.push(command)

    def __linkRemove(self):
        """Remove link was requested from the context menu.
        """
        if self.__contextMenuTarget:
            self.removeLink(self.__contextMenuTarget)

    def __linkReset(self):
        """Link reset from the context menu was requested.
        """
        if self.__contextMenuTarget:
            link = self.__contextMenuTarget
            action = interactions.EditNodeLinksAction(
                self, link.source_node, link.sink_node
            )
            action.edit_links()

    def __startControlPointEdit(self, item):
        """Start a control point edit interaction for item.
        """
        if isinstance(item, items.ArrowAnnotation):
            handler = interactions.ResizeArrowAnnotation(self)
        elif isinstance(item, items.TextAnnotation):
            handler = interactions.ResizeTextAnnotation(self)
        else:
            log.warning("Unknown annotation item type %r" % item)
            return

        handler.editItem(item)
        self._setUserInteractionHandler(handler)

        log.info("Control point editing started (%r)." % item)

    def __endControlPointEdit(self):
        """End the current control point edit interaction.
        """
        handler = self.__scene.user_interaction_handler
        if isinstance(handler, (interactions.ResizeArrowAnnotation,
                                interactions.ResizeTextAnnotation)) and \
                not handler.isFinished() and not handler.isCanceled():
            handler.commit()
            handler.end()

            log.info("Control point editing finished.")


def geometry_from_annotation_item(item):
    if isinstance(item, items.ArrowAnnotation):
        line = item.line()
        p1 = item.mapToScene(line.p1())
        p2 = item.mapToScene(line.p2())
        return ((p1.x(), p1.y()), (p2.x(), p2.y()))
    elif isinstance(item, items.TextAnnotation):
        geom = item.geometry()
        return (geom.x(), geom.y(), geom.width(), geom.height())


def mouse_drag_distance(event, button=Qt.LeftButton):
    """Return the (manhattan) distance between the (screen position)
    when the `button` was pressed and the current mouse position.

    """
    diff = (event.buttonDownScreenPos(button) - event.screenPos())
    return diff.manhattanLength()


def set_enabled_all(objects, enable):
    """Set enabled properties on all objects (QObjects with setEnabled).
    """
    for obj in objects:
        obj.setEnabled(enable)
