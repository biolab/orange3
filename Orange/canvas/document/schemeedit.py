"""
====================
Scheme Editor Widget
====================


"""

import sys
import logging
import itertools
import unicodedata
import copy

from operator import attrgetter
from urllib.parse import urlencode

from typing import List

from AnyQt.QtWidgets import (
    QWidget, QVBoxLayout, QInputDialog, QMenu, QAction, QActionGroup,
    QUndoStack, QUndoCommand, QGraphicsItem, QGraphicsObject,
    QGraphicsTextItem, QFormLayout, QComboBox, QDialog, QDialogButtonBox,
    QMessageBox
)
from AnyQt.QtGui import (
    QKeySequence, QCursor, QFont, QPainter, QPixmap, QColor, QIcon,
    QWhatsThisClickedEvent, QPalette
)

from AnyQt.QtCore import (
    Qt, QObject, QEvent, QSignalMapper, QRectF, QCoreApplication
)

from AnyQt.QtCore import pyqtProperty as Property, pyqtSignal as Signal

from ..registry.qt import whats_this_helper
from ..gui.quickhelp import QuickHelpTipEvent
from ..gui.utils import message_information, disabled
from ..scheme import (
    scheme, signalmanager, Scheme, SchemeNode, SchemeLink, BaseSchemeAnnotation
)
from ..scheme import widgetsscheme
from ..canvas.scene import CanvasScene
from ..canvas.view import CanvasView
from ..canvas import items
from . import interactions
from . import commands
from . import quickmenu


log = logging.getLogger(__name__)


# TODO: Should this be moved to CanvasScene?
class GraphicsSceneFocusEventListener(QGraphicsObject):

    itemFocusedIn = Signal(object)
    itemFocusedOut = Signal(object)

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
    """
    A widget for editing a :class:`~.scheme.Scheme` instance.

    """
    #: Undo command has become available/unavailable.
    undoAvailable = Signal(bool)

    #: Redo command has become available/unavailable.
    redoAvailable = Signal(bool)

    #: Document modified state has changed.
    modificationChanged = Signal(bool)

    #: Undo command was added to the undo stack.
    undoCommandAdded = Signal()

    #: Item selection has changed.
    selectionChanged = Signal()

    #: Document title has changed.
    titleChanged = Signal(str)

    #: Document path has changed.
    pathChanged = Signal(str)

    # Quick Menu triggers
    (NoTriggers,
     RightClicked,
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
        self.__nodeAnimationEnabled = True
        self.__possibleSelectionHandler = None
        self.__possibleMouseItemsMove = False
        self.__itemsMoving = {}
        self.__contextMenuTarget = None
        self.__quickMenu = None
        self.__quickTip = ""

        self.__undoStack = QUndoStack(self)
        self.__undoStack.cleanChanged[bool].connect(self.__onCleanChanged)

        # scheme node properties when set to a clean state
        self.__cleanProperties = []

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
        self.__editMenu.addAction(self.__duplicateSelectedAction)
        self.__editMenu.addAction(self.__selectAllAction)

        self.__widgetMenu = QMenu(self.tr("&Widget"), self)
        self.__widgetMenu.addAction(self.__openSelectedAction)
        self.__widgetMenu.addSeparator()
        self.__widgetMenu.addAction(self.__renameAction)
        self.__widgetMenu.addAction(self.__removeSelectedAction)
        self.__widgetMenu.addSeparator()
        self.__widgetMenu.addAction(self.__helpAction)
        if log.isEnabledFor(logging.DEBUG):
            self.__widgetMenu.addSeparator()
            self.__widgetMenu.addAction(self.__showSettingsAction)

        self.__linkMenu = QMenu(self.tr("Link"), self)
        self.__linkMenu.addAction(self.__linkEnableAction)
        self.__linkMenu.addSeparator()
        self.__linkMenu.addAction(self.__linkRemoveAction)
        self.__linkMenu.addAction(self.__linkResetAction)

    def __setupActions(self):
        self.__cleanUpAction = \
            QAction(self.tr("Clean Up"), self,
                    objectName="cleanup-action",
                    toolTip=self.tr("Align widgets to a grid."),
                    triggered=self.alignToGrid,
                    )

        self.__newTextAnnotationAction = \
            QAction(self.tr("Text"), self,
                    objectName="new-text-action",
                    toolTip=self.tr("Add a text annotation to the workflow."),
                    checkable=True,
                    toggled=self.__toggleNewTextAnnotation,
                    )

        # Create a font size menu for the new annotation action.
        self.__fontMenu = QMenu("Font Size", self)
        self.__fontActionGroup = group = \
            QActionGroup(self, exclusive=True,
                         triggered=self.__onFontSizeTriggered)

        def font(size):
            f = QFont(self.font())
            f.setPixelSize(size)
            return f

        for size in [12, 14, 16, 18, 20, 22, 24]:
            action = QAction("%ipx" % size, group,
                             checkable=True,
                             font=font(size))

            self.__fontMenu.addAction(action)

        group.actions()[2].setChecked(True)

        self.__newTextAnnotationAction.setMenu(self.__fontMenu)

        self.__newArrowAnnotationAction = \
            QAction(self.tr("Arrow"), self,
                    objectName="new-arrow-action",
                    toolTip=self.tr("Add an arrow annotation to the workflow."),
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
            QAction(self.tr("Select All"), self,
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

        self.__showSettingsAction = \
            QAction(self.tr("Show settings"), self,
                    objectName="show-settings",
                    toolTip=self.tr("Show widget settings"),
                    triggered=self.showSettings,
                    enabled=False)

        shortcuts = [Qt.Key_Backspace,
                     Qt.Key_Delete,
                     Qt.ControlModifier + Qt.Key_Backspace]

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
                    shortcut=QKeySequence("F1"),
                    enabled=False,
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

        self.__duplicateSelectedAction = \
            QAction(self.tr("Duplicate Selected"), self,
                    objectName="duplicate-action",
                    enabled=False,
                    shortcut=QKeySequence(Qt.ControlModifier + Qt.Key_D),
                    triggered=self.__duplicateSelected,
                    )

        self.addActions([self.__newTextAnnotationAction,
                         self.__newArrowAnnotationAction,
                         self.__linkEnableAction,
                         self.__linkRemoveAction,
                         self.__linkResetAction,
                         self.__duplicateSelectedAction])

        # Actions which should be disabled while a multistep
        # interaction is in progress.
        self.__disruptiveActions = \
                [self.__undoAction,
                 self.__redoAction,
                 self.__removeSelectedAction,
                 self.__selectAllAction,
                 self.__duplicateSelectedAction]

        #: Top 'Window Groups' action
        self.__windowGroupsAction = QAction(
            self.tr("Window Groups"), self, objectName="window-groups-action",
            toolTip="Manage preset widget groups"
        )
        #: Action group containing action for every window group
        self.__windowGroupsActionGroup = QActionGroup(
            self.__windowGroupsAction, objectName="window-groups-action-group",
        )
        self.__windowGroupsActionGroup.triggered.connect(
            self.__activateWindowGroup
        )
        self.__saveWindowGroupAction = QAction(
            self.tr("Save Window Group..."), self,
            toolTip="Create and save a new window group."
        )
        self.__saveWindowGroupAction.triggered.connect(self.__saveWindowGroup)
        self.__clearWindowGroupsAction = QAction(
            self.tr("Delete All Groups"), self,
            toolTip="Delete all saved widget presets"
        )
        self.__clearWindowGroupsAction.triggered.connect(
            self.__clearWindowGroups
        )

        groups_menu = QMenu(self)
        sep = groups_menu.addSeparator()
        sep.setObjectName("groups-separator")
        groups_menu.addAction(self.__saveWindowGroupAction)
        groups_menu.addSeparator()
        groups_menu.addAction(self.__clearWindowGroupsAction)
        self.__windowGroupsAction.setMenu(groups_menu)

        # the counterpart to Control + Key_Up to raise the containing workflow
        # view (maybe move that shortcut here)
        self.__raiseWidgetsAction = QAction(
            self.tr("Bring Widgets to Front"), self,
            objectName="bring-widgets-to-front-action",
            shortcut=QKeySequence(Qt.ControlModifier + Qt.Key_Down),
            shortcutContext=Qt.WindowShortcut,
        )
        self.__raiseWidgetsAction.triggered.connect(self.__raiseToFont)
        self.addAction(self.__raiseWidgetsAction)

    def __setupUi(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scene = CanvasScene(self)
        scene.setItemIndexMethod(CanvasScene.NoIndex)
        self.__setupScene(scene)

        view = CanvasView(scene)
        view.setFrameStyle(CanvasView.NoFrame)
        view.setRenderHint(QPainter.Antialiasing)

        self.__view = view
        self.__scene = scene

        layout.addWidget(view)
        self.setLayout(layout)

    def __setupScene(self, scene):
        """
        Set up a :class:`CanvasScene` instance for use by the editor.

        .. note:: If an existing scene is in use it must be teared down using
            __teardownScene

        """
        scene.set_channel_names_visible(self.__channelNamesVisible)
        scene.set_node_animation_enabled(
            self.__nodeAnimationEnabled
        )

        scene.setFont(self.font())
        scene.setPalette(self.palette())
        scene.installEventFilter(self)

        scene.set_registry(self.__registry)

        # Focus listener
        self.__focusListener = GraphicsSceneFocusEventListener()
        self.__focusListener.itemFocusedIn.connect(
            self.__onItemFocusedIn
        )
        self.__focusListener.itemFocusedOut.connect(
            self.__onItemFocusedOut
        )
        scene.addItem(self.__focusListener)

        scene.selectionChanged.connect(
            self.__onSelectionChanged
        )

        scene.node_item_activated.connect(
            self.__onNodeActivate
        )

        scene.annotation_added.connect(
            self.__onAnnotationAdded
        )

        scene.annotation_removed.connect(
            self.__onAnnotationRemoved
        )

        self.__annotationGeomChanged = QSignalMapper(self)

    def __teardownScene(self, scene):
        """
        Tear down an instance of :class:`CanvasScene` that was used by the
        editor.

        """
        # Clear the current item selection in the scene so edit action
        # states are updated accordingly.
        scene.clearSelection()

        # Clear focus from any item.
        scene.setFocusItem(None)

        # Clear the annotation mapper
        self.__annotationGeomChanged.deleteLater()
        self.__annotationGeomChanged = None

        self.__focusListener.itemFocusedIn.disconnect(
            self.__onItemFocusedIn
        )
        self.__focusListener.itemFocusedOut.disconnect(
            self.__onItemFocusedOut
        )

        scene.selectionChanged.disconnect(
            self.__onSelectionChanged
        )

        scene.removeEventFilter(self)

        # Clear all items from the scene
        scene.blockSignals(True)
        scene.clear_scene()

    def toolbarActions(self):
        """
        Return a list of actions that can be inserted into a toolbar.
        At the moment these are:

            - 'Clean up' action (align to grid)
            - 'New text annotation' action (with a size menu)
            - 'New arrow annotation' action (with a color menu)

        """
        return [self.__cleanUpAction,
                self.__newTextAnnotationAction,
                self.__newArrowAnnotationAction]

    def menuBarActions(self):
        """
        Return a list of actions that can be inserted into a `QMenuBar`.

        """
        return [self.__editMenu.menuAction(), self.__widgetMenu.menuAction()]

    def isModified(self):
        """
        Is the document is a modified state.
        """
        return self.__modified or not self.__undoStack.isClean()

    def setModified(self, modified):
        """
        Set the document modified state.
        """
        if self.__modified != modified:
            self.__modified = modified

        if not modified:
            self.__cleanProperties = node_properties(self.__scheme)
            self.__undoStack.setClean()
        else:
            self.__cleanProperties = []

    modified = Property(bool, fget=isModified, fset=setModified)

    def isModifiedStrict(self):
        """
        Is the document modified.

        Run a strict check against all node properties as they were
        at the time when the last call to `setModified(True)` was made.

        """
        propertiesChanged = self.__cleanProperties != \
                            node_properties(self.__scheme)

        log.debug("Modified strict check (modified flag: %s, "
                  "undo stack clean: %s, properties: %s)",
                  self.__modified,
                  self.__undoStack.isClean(),
                  propertiesChanged)

        return self.isModified() or propertiesChanged

    def setQuickMenuTriggers(self, triggers):
        """
        Set quick menu trigger flags.

        Flags can be a bitwise `or` of:

            - `SchemeEditWidget.NoTrigeres`
            - `SchemeEditWidget.RightClicked`
            - `SchemeEditWidget.DoubleClicked`
            - `SchemeEditWidget.SpaceKey`
            - `SchemeEditWidget.AnyKey`

        """
        if self.__quickMenuTriggers != triggers:
            self.__quickMenuTriggers = triggers

    def quickMenuTriggers(self):
        """
        Return quick menu trigger flags.
        """
        return self.__quickMenuTriggers

    def setChannelNamesVisible(self, visible):
        """
        Set channel names visibility state. When enabled the links
        in the view will have a source/sink channel names displayed over
        them.

        """
        if self.__channelNamesVisible != visible:
            self.__channelNamesVisible = visible
            self.__scene.set_channel_names_visible(visible)

    def channelNamesVisible(self):
        """
        Return the channel name visibility state.
        """
        return self.__channelNamesVisible

    def setNodeAnimationEnabled(self, enabled):
        """
        Set the node item animation enabled state.
        """
        if self.__nodeAnimationEnabled != enabled:
            self.__nodeAnimationEnabled = enabled
            self.__scene.set_node_animation_enabled(enabled)

    def nodeAnimationEnabled(self):
        """
        Return the node item animation enabled state.
        """
        return self.__nodeAnimationEnabled

    def undoStack(self):
        """
        Return the undo stack.
        """
        return self.__undoStack

    def setPath(self, path):
        """
        Set the path associated with the current scheme.

        .. note:: Calling `setScheme` will invalidate the path (i.e. set it
                  to an empty string)

        """
        if self.__path != path:
            self.__path = str(path)
            self.pathChanged.emit(self.__path)

    def path(self):
        """
        Return the path associated with the scheme
        """
        return self.__path

    def setScheme(self, scheme):
        """
        Set the :class:`~.scheme.Scheme` instance to display/edit.
        """
        if self.__scheme is not scheme:
            if self.__scheme:
                self.__scheme.title_changed.disconnect(self.titleChanged)
                self.__scheme.removeEventFilter(self)
                sm = self.__scheme.findChild(signalmanager.SignalManager)
                if sm:
                    sm.stateChanged.disconnect(
                        self.__signalManagerStateChanged)

            self.__scheme = scheme

            self.setPath("")

            if self.__scheme:
                self.__scheme.title_changed.connect(self.titleChanged)
                self.titleChanged.emit(scheme.title)
                self.__cleanProperties = node_properties(scheme)
                sm = scheme.findChild(signalmanager.SignalManager)
                if sm:
                    sm.stateChanged.connect(self.__signalManagerStateChanged)
            else:
                self.__cleanProperties = []

            self.__teardownScene(self.__scene)
            self.__scene.deleteLater()

            self.__undoStack.clear()

            self.__scene = CanvasScene(self)
            self.__scene.setItemIndexMethod(CanvasScene.NoIndex)
            self.__setupScene(self.__scene)

            self.__scene.set_scheme(scheme)
            self.__view.setScene(self.__scene)

            if self.__scheme:
                self.__scheme.installEventFilter(self)
                nodes = self.__scheme.nodes
                if nodes:
                    self.ensureVisible(nodes[0])

        group = self.__windowGroupsActionGroup
        menu = self.__windowGroupsAction.menu()
        actions = group.actions()
        for a in actions:
            group.removeAction(a)
            menu.removeAction(a)
            a.deleteLater()

        if scheme:
            presets = scheme.window_group_presets()
            sep = menu.findChild(QAction, "groups-separator")
            for g in presets:
                a = QAction(g.name, menu)
                a.setShortcut(
                    QKeySequence("Meta+P, Ctrl+{}"
                                 .format(len(group.actions()) + 1))
                )
                a.setData(g)
                group.addAction(a)
                menu.insertAction(sep, a)

    def ensureVisible(self, node):
        """
        Scroll the contents of the viewport so that `node` is visible.

        Parameters
        ----------
        node: SchemeNode
        """
        if self.__scheme is None:
            return
        item = self.__scene.item_for_node(node)
        self.__view.ensureVisible(item)

    def scheme(self):
        """
        Return the :class:`~.scheme.Scheme` edited by the widget.
        """
        return self.__scheme

    def scene(self):
        """
        Return the :class:`QGraphicsScene` instance used to display the
        current scheme.

        """
        return self.__scene

    def view(self):
        """
        Return the :class:`QGraphicsView` instance used to display the
        current scene.

        """
        return self.__view

    def setRegistry(self, registry):
        # Is this method necessary?
        # It should be removed when the scene (items) is fixed
        # so all information regarding the visual appearance is
        # included in the node/widget description.
        self.__registry = registry
        if self.__scene:
            self.__scene.set_registry(registry)
            self.__quickMenu = None

    def quickMenu(self):
        """
        Return a :class:`~.quickmenu.QuickMenu` popup menu instance for
        new node creation.

        """
        if self.__quickMenu is None:
            menu = quickmenu.QuickMenu(self)
            if self.__registry is not None:
                menu.setModel(self.__registry.model())
            self.__quickMenu = menu
        return self.__quickMenu

    def setTitle(self, title):
        """
        Set the scheme title.
        """
        self.__undoStack.push(
            commands.SetAttrCommand(self.__scheme, "title", title)
        )

    def setDescription(self, description):
        """
        Set the scheme description string.
        """
        self.__undoStack.push(
            commands.SetAttrCommand(self.__scheme, "description", description)
        )

    def addNode(self, node):
        """
        Add a new node (:class:`.SchemeNode`) to the document.
        """
        command = commands.AddNodeCommand(self.__scheme, node)
        self.__undoStack.push(command)

    def createNewNode(self, description, title=None, position=None):
        """
        Create a new :class:`.SchemeNode` and add it to the document.
        The new node is constructed using :func:`newNodeHelper` method.

        """
        node = self.newNodeHelper(description, title, position)
        self.addNode(node)

        return node

    def newNodeHelper(self, description, title=None, position=None):
        """
        Return a new initialized :class:`.SchemeNode`. If `title`
        and `position` are not supplied they are initialized to sensible
        defaults.

        """
        if title is None:
            title = self.enumerateTitle(description.name)

        if position is None:
            position = self.nextPosition()

        return SchemeNode(description, title=title, position=position)

    def enumerateTitle(self, title):
        """
        Enumerate a `title` string (i.e. add a number in parentheses) so
        it is not equal to any node title in the current scheme.

        """
        curr_titles = set([node.title for node in self.scheme().nodes])
        template = title + " ({0})"

        enumerated = map(template.format, itertools.count(1))
        candidates = itertools.chain([title], enumerated)

        seq = itertools.dropwhile(curr_titles.__contains__, candidates)
        return next(seq)

    def nextPosition(self):
        """
        Return the next default node position as a (x, y) tuple. This is
        a position left of the last added node.

        """
        nodes = self.scheme().nodes
        if nodes:
            x, y = nodes[-1].position
            position = (x + 150, y)
        else:
            position = (150, 150)
        return position

    def removeNode(self, node):
        """
        Remove a `node` (:class:`.SchemeNode`) from the scheme
        """
        command = commands.RemoveNodeCommand(self.__scheme, node)
        self.__undoStack.push(command)

    def renameNode(self, node, title):
        """
        Rename a `node` (:class:`.SchemeNode`) to `title`.
        """
        command = commands.RenameNodeCommand(self.__scheme, node, title)
        self.__undoStack.push(command)

    def addLink(self, link):
        """
        Add a `link` (:class:`.SchemeLink`) to the scheme.
        """
        command = commands.AddLinkCommand(self.__scheme, link)
        self.__undoStack.push(command)

    def removeLink(self, link):
        """
        Remove a link (:class:`.SchemeLink`) from the scheme.
        """
        command = commands.RemoveLinkCommand(self.__scheme, link)
        self.__undoStack.push(command)

    def addAnnotation(self, annotation):
        """
        Add `annotation` (:class:`.BaseSchemeAnnotation`) to the scheme
        """
        command = commands.AddAnnotationCommand(self.__scheme, annotation)
        self.__undoStack.push(command)

    def removeAnnotation(self, annotation):
        """
        Remove `annotation` (:class:`.BaseSchemeAnnotation`) from the scheme.
        """
        command = commands.RemoveAnnotationCommand(self.__scheme, annotation)
        self.__undoStack.push(command)

    def removeSelected(self):
        """
        Remove all selected items in the scheme.
        """
        selected = self.scene().selectedItems()
        if not selected:
            return
        scene = self.scene()
        self.__undoStack.beginMacro(self.tr("Remove"))
        for item in selected:
            if isinstance(item, items.NodeItem):
                node = self.scene().node_for_item(item)
                self.__undoStack.push(
                    commands.RemoveNodeCommand(self.__scheme, node)
                )
            elif isinstance(item, items.annotationitem.Annotation):
                if item.hasFocus() or item.isAncestorOf(scene.focusItem()):
                    # Clear input focus from the item to be removed.
                    scene.focusItem().clearFocus()
                annot = self.scene().annotation_for_item(item)
                self.__undoStack.push(
                    commands.RemoveAnnotationCommand(self.__scheme, annot)
                )
        self.__undoStack.endMacro()

    def showSettings(self):
        """
        Dump settings of selected items to the standard output.
        """
        selected = self.scene().selectedItems()
        for item in selected:
            node = self.scene().node_for_item(item)
            self.scheme().dump_settings(node)

    def selectAll(self):
        """
        Select all selectable items in the scheme.
        """
        for item in self.__scene.items():
            if item.flags() & QGraphicsItem.ItemIsSelectable:
                item.setSelected(True)

    def alignToGrid(self):
        """
        Align nodes to a grid.
        """
        # TODO: The the current layout implementation is BAD (fix is urgent).
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

    def focusNode(self):
        """
        Return the current focused :class:`.SchemeNode` or ``None`` if no
        node has focus.

        """
        focus = self.__scene.focusItem()
        node = None
        if isinstance(focus, items.NodeItem):
            try:
                node = self.__scene.node_for_item(focus)
            except KeyError:
                # in case the node has been removed but the scene was not
                # yet fully updated.
                node = None
        return node

    def selectedNodes(self):
        """
        Return all selected :class:`.SchemeNode` items.
        """
        return list(map(self.scene().node_for_item,
                        self.scene().selected_node_items()))

    def selectedAnnotations(self):
        """
        Return all selected :class:`.BaseSchemeAnnotation` items.
        """
        return list(map(self.scene().annotation_for_item,
                        self.scene().selected_annotation_items()))

    def openSelected(self):
        """
        Open (show and raise) all widgets for the current selected nodes.
        """
        selected = self.scene().selected_node_items()
        for item in selected:
            self.__onNodeActivate(item)

    def editNodeTitle(self, node):
        """
        Edit (rename) the `node`'s title. Opens an input dialog.
        """
        name, ok = QInputDialog.getText(
                    self, self.tr("Rename"),
                    str(self.tr("Enter a new name for the '%s' widget")) \
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

    def changeEvent(self, event):
        if event.type() == QEvent.FontChange:
            self.__updateFont()
        elif event.type() == QEvent.PaletteChange:
            if self.__scene is not None:
                self.__scene.setPalette(self.palette())

        QWidget.changeEvent(self, event)

    def eventFilter(self, obj, event):
        # Filter the scene's drag/drop events.
        if obj is self.scene():
            etype = event.type()
            if etype == QEvent.GraphicsSceneDragEnter or \
                    etype == QEvent.GraphicsSceneDragMove:
                mime_data = event.mimeData()
                if mime_data.hasFormat(
                        "application/vnv.orange-canvas.registry.qualified-name"
                        ):
                    event.acceptProposedAction()
                else:
                    event.ignore()
                return True
            elif etype == QEvent.GraphicsSceneDrop:
                data = event.mimeData()
                qname = data.data(
                    "application/vnv.orange-canvas.registry.qualified-name"
                )
                try:
                    desc = self.__registry.widget(bytes(qname).decode())
                except KeyError:
                    log.error("Unknown qualified name '%s'", qname)
                else:
                    pos = event.scenePos()
                    self.createNewNode(desc, position=(pos.x(), pos.y()))
                return True

            elif etype == QEvent.GraphicsSceneMousePress:
                return self.sceneMousePressEvent(event)
            elif etype == QEvent.GraphicsSceneMouseMove:
                return self.sceneMouseMoveEvent(event)
            elif etype == QEvent.GraphicsSceneMouseRelease:
                return self.sceneMouseReleaseEvent(event)
            elif etype == QEvent.GraphicsSceneMouseDoubleClick:
                return self.sceneMouseDoubleClickEvent(event)
            elif etype == QEvent.KeyPress:
                return self.sceneKeyPressEvent(event)
            elif etype == QEvent.KeyRelease:
                return self.sceneKeyReleaseEvent(event)
            elif etype == QEvent.GraphicsSceneContextMenu:
                return self.sceneContextMenuEvent(event)

        elif obj is self.__scheme:
            if event.type() == QEvent.WhatsThisClicked:
                # Re post the event
                self.__showHelpFor(event.href())

            elif event.type() == \
                    widgetsscheme.ActivateParentEvent.ActivateParent:
                self.window().activateWindow()
                self.window().raise_()

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
        if not any_item:
            self.__emptyClickButtons |= event.button()

        if not any_item and event.button() == Qt.LeftButton:
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
            self.__possibleSelectionHandler = None
            return handler.mouseMoveEvent(event)

        return False

    def sceneMouseReleaseEvent(self, event):
        scene = self.__scene
        if scene.user_interaction_handler:
            return False

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
                    if isinstance(scheme_item, SchemeNode):
                        command = commands.MoveNodeCommand(
                            self.scheme(), scheme_item, old, new
                        )
                    elif isinstance(scheme_item, BaseSchemeAnnotation):
                        command = commands.AnnotationGeometryChange(
                            self.scheme(), scheme_item, old, new
                        )
                    else:
                        continue

                    stack.push(command)
                stack.endMacro()

                self.__itemsMoving.clear()
                return True
        elif event.button() == Qt.LeftButton:
            self.__possibleSelectionHandler = None

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

            with disabled(self.__undoAction), disabled(self.__redoAction):
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
        searchText = ""
        if (event.key() == Qt.Key_Space and \
                self.__quickMenuTriggers & SchemeEditWidget.SpaceKey):
            handler = interactions.NewNodeAction(self)

        elif len(event.text()) and \
                self.__quickMenuTriggers & SchemeEditWidget.AnyKey and \
                is_printable(str(event.text())[0]):
            handler = interactions.NewNodeAction(self)
            searchText = str(event.text())

            # TODO: set the search text to event.text() and set focus on the
            # search line

        if handler is not None:
            # Control + Backspace (remove widget action on Mac OSX) conflicts
            # with the 'Clear text' action in the search widget (there might
            # be selected items in the canvas), so we disable the
            # remove widget action so the text editing follows standard
            # 'look and feel'
            with disabled(self.__removeSelectedAction), \
                 disabled(self.__undoAction), \
                 disabled(self.__redoAction):
                handler.create_new(QCursor.pos(), searchText)

            event.accept()
            return True

        return False

    def sceneKeyReleaseEvent(self, event):
        return False

    def sceneContextMenuEvent(self, event):
        scenePos = event.scenePos()
        globalPos = event.screenPos()

        item = self.scene().item_at(scenePos, items.NodeItem)
        if item is not None:
            self.__widgetMenu.popup(globalPos)
            return True

        item = self.scene().item_at(scenePos, items.LinkItem,
                                    buttons=Qt.RightButton)
        if item is not None:
            link = self.scene().link_for_item(item)
            self.__linkEnableAction.setChecked(link.enabled)
            self.__contextMenuTarget = link
            self.__linkMenu.popup(globalPos)
            return True

        item = self.scene().item_at(scenePos)
        if not item and \
                self.__quickMenuTriggers & SchemeEditWidget.RightClicked:
            action = interactions.NewNodeAction(self)

            with disabled(self.__undoAction), disabled(self.__redoAction):
                action.create_new(globalPos)
            return True

        return False

    def _setUserInteractionHandler(self, handler):
        """
        Helper method for setting the user interaction handlers.
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
        self.__duplicateSelectedAction.setEnabled(bool(nodes))
        self.__showSettingsAction.setEnabled(len(nodes) == 1)

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

        focus = self.focusNode()
        if focus is not None:
            desc = focus.description
            tip = whats_this_helper(desc, include_more_link=True)
        else:
            tip = ""

        if tip != self.__quickTip:
            self.__quickTip = tip
            ev = QuickHelpTipEvent("", self.__quickTip,
                                   priority=QuickHelpTipEvent.Permanent)

            QCoreApplication.sendEvent(self, ev)

    def __onNodeActivate(self, item):
        node = self.__scene.node_for_item(item)
        widget = self.scheme().widget_for_node(node)
        widget.showNormal()
        widget.raise_()
        widget.activateWindow()

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
        """
        Annotation item has gained focus.
        """
        if not self.__scene.user_interaction_handler:
            self.__startControlPointEdit(item)

    def __onItemFocusedOut(self, item):
        """
        Annotation item lost focus.
        """
        self.__endControlPointEdit()

    def __onEditingFinished(self, item):
        """
        Text annotation editing has finished.
        """
        annot = self.__scene.annotation_for_item(item)

        content_type = item.contentType()
        content = item.content()

        if annot.text != content or annot.content_type != content_type:
            self.__undoStack.push(
                commands.TextChangeCommand(
                    self.scheme(), annot,
                    annot.text, annot.content_type,
                    content, content_type
                )
            )

    def __toggleNewArrowAnnotation(self, checked):
        if self.__newTextAnnotationAction.isChecked():
            # Uncheck the text annotation action if needed.
            self.__newTextAnnotationAction.setChecked(not checked)

        action = self.__newArrowAnnotationAction

        if not checked:
            # The action was unchecked (canceled by the user)
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
            # When selecting from the (font size) menu the 'Text'
            # action does not get triggered automatically.
            self.__newTextAnnotationAction.trigger()
        else:
            # Update the preferred font on the interaction handler.
            handler = self.__scene.user_interaction_handler
            if isinstance(handler, interactions.NewTextAnnotation):
                handler.setFont(action.font())

    def __toggleNewTextAnnotation(self, checked):
        if self.__newArrowAnnotationAction.isChecked():
            # Uncheck the arrow annotation if needed.
            self.__newArrowAnnotationAction.setChecked(not checked)

        action = self.__newTextAnnotationAction

        if not checked:
            # The action was unchecked (canceled by the user)
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
            # When selecting from the (color) menu the 'Arrow'
            # action does not get triggered automatically.
            self.__newArrowAnnotationAction.trigger()
        else:
            # Update the preferred color on the interaction handler
            handler = self.__scene.user_interaction_handler
            if isinstance(handler, interactions.NewArrowAnnotation):
                handler.setColor(action.data())

    def __onRenameAction(self):
        """
        Rename was requested for the selected widget.
        """
        selected = self.selectedNodes()
        if len(selected) == 1:
            self.editNodeTitle(selected[0])

    def __onHelpAction(self):
        """
        Help was requested for the selected widget.
        """
        nodes = self.selectedNodes()
        help_url = None
        if len(nodes) == 1:
            node = nodes[0]
            desc = node.description

            help_url = "help://search?" + urlencode({"id": desc.qualified_name})
            self.__showHelpFor(help_url)

    def __showHelpFor(self, help_url):
        """
        Show help for an "help" url.
        """
        # Notify the parent chain and let them respond
        ev = QWhatsThisClickedEvent(help_url)
        handled = QCoreApplication.sendEvent(self, ev)

        if not handled:
            message_information(
                self.tr("There is no documentation for this widget yet."),
                parent=self)

    def __toggleLinkEnabled(self, enabled):
        """
        Link 'enabled' state was toggled in the context menu.
        """
        if self.__contextMenuTarget:
            link = self.__contextMenuTarget
            command = commands.SetAttrCommand(
                link, "enabled", enabled, name=self.tr("Set enabled"),
            )
            self.__undoStack.push(command)

    def __linkRemove(self):
        """
        Remove link was requested from the context menu.
        """
        if self.__contextMenuTarget:
            self.removeLink(self.__contextMenuTarget)

    def __linkReset(self):
        """
        Link reset from the context menu was requested.
        """
        if self.__contextMenuTarget:
            link = self.__contextMenuTarget
            action = interactions.EditNodeLinksAction(
                self, link.source_node, link.sink_node
            )
            action.edit_links()

    def __duplicateSelected(self):
        """
        Duplicate currently selected nodes.
        """
        def copy_node(node):
            x, y = node.position
            return SchemeNode(
                node.description, node.title, position=(x + 20, y + 20),
                properties=copy.deepcopy(node.properties))

        def copy_link(link, source=None, sink=None):
            source = link.source_node if source is None else source
            sink = link.sink_node if sink is None else sink
            return SchemeLink(
                source, link.source_channel,
                sink, link.sink_channel,
                enabled=link.enabled,
                properties=copy.deepcopy(link.properties))

        scheme = self.scheme()
        # ensure up to date node properties (settings)
        scheme.sync_node_properties()

        selection = self.selectedNodes()

        links = [link for link in scheme.links
                 if link.source_node in selection and
                    link.sink_node in selection]
        nodedups = [copy_node(node) for node in selection]
        allnames = {node.title for node in scheme.nodes + nodedups}
        for nodedup in nodedups:
            nodedup.title = uniquify(
                nodedup.title, allnames, pattern="{item} ({_})", start=1)

        node_to_dup = dict(zip(selection, nodedups))

        linkdups = [copy_link(link, source=node_to_dup[link.source_node],
                              sink=node_to_dup[link.sink_node])
                    for link in links]

        command = QUndoCommand(self.tr("Duplicate"))
        macrocommands = []
        for nodedup in nodedups:
            macrocommands.append(
                commands.AddNodeCommand(scheme, nodedup, parent=command))
        for linkdup in linkdups:
            macrocommands.append(
                commands.AddLinkCommand(scheme, linkdup, parent=command))

        self.__undoStack.push(command)
        scene = self.__scene

        for node in selection:
            item = scene.item_for_node(node)
            item.setSelected(False)

        for node in nodedups:
            item = scene.item_for_node(node)
            item.setSelected(True)

    def __startControlPointEdit(self, item):
        """
        Start a control point edit interaction for `item`.
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
        """
        End the current control point edit interaction.
        """
        handler = self.__scene.user_interaction_handler
        if isinstance(handler, (interactions.ResizeArrowAnnotation,
                                interactions.ResizeTextAnnotation)) and \
                not handler.isFinished() and not handler.isCanceled():
            handler.commit()
            handler.end()

            log.info("Control point editing finished.")

    def __updateFont(self):
        """
        Update the font for the "Text size' menu and the default font
        used in the `CanvasScene`.

        """
        actions = self.__fontActionGroup.actions()
        font = self.font()
        for action in actions:
            size = action.font().pixelSize()
            action_font = QFont(font)
            action_font.setPixelSize(size)
            action.setFont(action_font)

        if self.__scene:
            self.__scene.setFont(font)

    def __signalManagerStateChanged(self, state):
        if state == signalmanager.SignalManager.Running:
            role = QPalette.Base
        else:
            role = QPalette.Window
        self.__view.viewport().setBackgroundRole(role)

    def __saveWindowGroup(self):
        # Run a 'Save Window Group' dialog
        workflow = self.__scheme  # type: widgetsscheme.WidgetsScheme
        state = workflow.widget_manager.save_window_state()
        presets = workflow.window_group_presets()
        items = [g.name for g in presets]

        dlg = SaveWindowGroup(
            self, windowTitle="Save Group as...")
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setItems(items)

        menu = self.__windowGroupsAction.menu()  # type: QMenu
        group = self.__windowGroupsActionGroup

        def store_group():
            text = dlg.selectedText()
            actions = group.actions()  # type: List[QAction]
            try:
                idx = items.index(text)
            except ValueError:
                idx = -1
            newpresets = list(presets)
            newpreset = Scheme.WindowGroup(text, False, state)
            if idx == -1:
                # new group slot
                newpresets.append(newpreset)
                action = QAction(text, menu)
                action.setShortcut(
                    QKeySequence("Meta+P, Ctrl+{}".format(len(newpresets)))
                )
                oldpreset = None
            else:
                newpresets[idx] = newpreset
                action = actions[idx]
                # store old state for undo
                oldpreset = presets[idx]

            sep = menu.findChild(QAction, "groups-separator")
            assert isinstance(sep, QAction) and sep.isSeparator()

            def redo():
                action.setData(newpreset)
                workflow.set_window_group_presets(newpresets)
                if idx == -1:
                    group.addAction(action)
                    menu.insertAction(sep, action)

            def undo():
                action.setData(oldpreset)
                workflow.set_window_group_presets(presets)
                if idx == -1:
                    group.removeAction(action)
                    menu.removeAction(action)
            if idx == -1:
                text = "Store Window Group"
            else:
                text = "Update Window Group"
            self.__undoStack.push(
                commands.SimpleUndoCommand(redo, undo, text)
            )
        dlg.accepted.connect(store_group)
        dlg.show()
        dlg.raise_()

    def __activateWindowGroup(self, action):
        # type: (QAction) -> None
        data = action.data()  # type: Scheme.WindowGroup
        workflow = self.__scheme
        if not isinstance(workflow, widgetsscheme.WidgetsScheme):
            return
        workflow.widget_manager.activate_window_group(data)

    def __clearWindowGroups(self):
        workflow = self.__scheme  # type: Scheme
        presets = workflow.window_group_presets()
        menu = self.__windowGroupsAction.menu()  # type: QMenu
        group = self.__windowGroupsActionGroup
        actions = group.actions()

        def redo():
            workflow.set_window_group_presets([])
            for action in reversed(actions):
                group.removeAction(action)
                menu.removeAction(action)

        def undo():
            workflow.set_window_group_presets(presets)
            sep = menu.findChild(QAction, "groups-separator")
            for action in actions:
                group.addAction(action)
                menu.insertAction(sep, action)

        self.__undoStack.push(
            commands.SimpleUndoCommand(redo, undo, "Delete All Window Groups")
        )

    def __raiseToFont(self):
        # Raise current visible widgets to front
        wf = self.__scheme
        if wf is not None:
            wf.widget_manager.raise_widgets_to_front()


class SaveWindowGroup(QDialog):
    """
    A dialog for saving window groups.

    The user can select an existing group to overwrite or enter a new group
    name.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QVBoxLayout()
        form = QFormLayout(
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow)
        layout.addLayout(form)
        self._combobox = cb = QComboBox(
            editable=True, minimumContentsLength=16,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLength,
            insertPolicy=QComboBox.NoInsert,
        )
        # default text if no items are present
        cb.setEditText(self.tr("Window Group 1"))
        cb.lineEdit().selectAll()
        form.addRow(self.tr("Save As:"), cb)
        bb = QDialogButtonBox(
            standardButtons=QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.__accept_check)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)
        layout.setSizeConstraint(QVBoxLayout.SetFixedSize)
        self.setLayout(layout)
        self.setWhatsThis(
            "Save the current open widgets' window arrangement to the "
            "workflow view presets."
        )
        cb.setFocus(Qt.NoFocusReason)

    def __accept_check(self):
        cb = self._combobox
        text = cb.currentText()
        if cb.findText(text) == -1:
            self.accept()
            return
        # Ask for overwrite confirmation
        mb = QMessageBox(
            self, windowTitle=self.tr("Confirm Overwrite"),
            icon=QMessageBox.Question,
            standardButtons=QMessageBox.Yes | QMessageBox.Cancel,
            text=self.tr("The window group '{}' already exists. Do you want " +
                         "to replace it?").format(text),
        )
        mb.setDefaultButton(QMessageBox.Yes)
        mb.setEscapeButton(QMessageBox.Cancel)
        mb.setWindowModality(Qt.WindowModal)
        button = mb.button(QMessageBox.Yes)
        button.setText(self.tr("Replace"))
        mb.finished.connect(
            lambda status: status == QMessageBox.Yes and self.accept()
        )
        mb.show()

    def setItems(self, items):
        # type: (List[str]) -> None
        """Set a list of existing items/names to present to the user"""
        self._combobox.clear()
        self._combobox.addItems(items)
        if items:
            self._combobox.setCurrentIndex(len(items) - 1)

    def selectedText(self):
        # type: () -> str
        """Return the current entered text."""
        return self._combobox.currentText()


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
    """
    Return the (manhattan) distance between the mouse position
    when the `button` was pressed and the current mouse position.

    """
    diff = (event.buttonDownScreenPos(button) - event.screenPos())
    return diff.manhattanLength()


def set_enabled_all(objects, enable):
    """
    Set `enabled` properties on all objects (objects with `setEnabled` method).
    """
    for obj in objects:
        obj.setEnabled(enable)


# All control character categories.
_control = set(["Cc", "Cf", "Cs", "Co", "Cn"])


def is_printable(unichar):
    """
    Return True if the unicode character `unichar` is a printable character.
    """
    return unicodedata.category(unichar) not in _control


def node_properties(scheme):
    scheme.sync_node_properties()
    return [dict(node.properties) for node in scheme.nodes]


def uniquify(item, names, pattern="{item}-{_}", start=0):
    candidates = (pattern.format(item=item, _=i)
                  for i in itertools.count(start))
    candidates = itertools.dropwhile(lambda item: item in names, candidates)
    return next(candidates)
