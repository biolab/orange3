"""
Orange Canvas Tool Dock widget

"""
import sys

from PyQt4.QtGui import (
    QWidget, QSplitter, QVBoxLayout, QTextEdit, QAction, QPalette,
    QSizePolicy, QApplication, QDrag
)

from PyQt4.QtCore import (
    Qt, QSize, QObject, QPropertyAnimation, QEvent, QRect, QPoint,
    QModelIndex, QPersistentModelIndex, QEventLoop, QMimeData
)

from PyQt4.QtCore import pyqtProperty as Property, pyqtSignal as Signal

from ..gui.toolgrid import ToolGrid
from ..gui.toolbar import DynamicResizeToolBar
from ..gui.quickhelp import QuickHelp
from ..gui.framelesswindow import FramelessWindow
from ..document.quickmenu import MenuPage
from ..document.quickmenu import create_css_gradient
from .widgettoolbox import WidgetToolBox, iter_item

from ..registry.qt import QtWidgetRegistry
from ..utils.qtcompat import toPyObject


class SplitterResizer(QObject):
    """An object able to control the size of a widget in a
    QSpliter instance.

    """

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.__splitter = None
        self.__widget = None
        self.__animationEnabled = True
        self.__size = -1
        self.__expanded = False
        self.__animation = QPropertyAnimation(self, "size_", self)

        self.__action = QAction("toogle-expanded", self, checkable=True)
        self.__action.triggered[bool].connect(self.setExpanded)

    def setSize(self, size):
        """Set the size of the controlled widget (either width or height
        depending on the orientation).

        """
        if self.__size != size:
            self.__size = size
            self.__update()

    def size(self):
        """Return the size of the widget in the splitter (either height of
        width) depending on the splitter orientation.

        """
        if self.__splitter and self.__widget:
            index = self.__splitter.indexOf(self.__widget)
            sizes = self.__splitter.sizes()
            return sizes[index]
        else:
            return -1

    size_ = Property(int, fget=size, fset=setSize)

    def setAnimationEnabled(self, enable):
        """Enable/disable animation.
        """
        self.__animation.setDuration(0 if enable else 200)

    def animationEnabled(self):
        return self.__animation.duration() == 0

    def setSplitterAndWidget(self, splitter, widget):
        """Set the QSplitter and QWidget instance the resizer should control.

        .. note:: the widget must be in the splitter.

        """
        if splitter and widget and not splitter.indexOf(widget) > 0:
            raise ValueError("Widget must be in a spliter.")

        if self.__widget:
            self.__widget.removeEventFilter()
        self.__splitter = splitter
        self.__widget = widget

        if widget:
            widget.installEventFilter(self)

        self.__update()

    def toogleExpandedAction(self):
        """Return a QAction that can be used to toggle expanded state.
        """
        return self.__action

    def open(self):
        """Open the controlled widget (expand it to it sizeHint).
        """
        self.__expanded = True
        self.__action.setChecked(True)

        if not (self.__splitter and self.__widget):
            return

        size = self.size()
        if size > 0:
            # Already has non zero size.
            return

        hint = self.__widget.sizeHint()

        if self.__splitter.orientation() == Qt.Vertical:
            end = hint.height()
        else:
            end = hint.width()

        self.__animation.setStartValue(0)
        self.__animation.setEndValue(end)
        self.__animation.start()

    def close(self):
        """Close the controlled widget (shrink to size 0).
        """
        self.__expanded = False
        self.__action.setChecked(False)

        if not (self.__splitter and self.__widget):
            return

        self.__animation.setStartValue(self.size())
        self.__animation.setEndValue(0)
        self.__animation.start()

    def setExpanded(self, expanded):
        """Set the expanded state.

        """
        if self.__expanded != expanded:
            if expanded:
                self.open()
            else:
                self.close()

    def expanded(self):
        """Return the expanded state.
        """
        return self.__expanded

    def __update(self):
        """Update the splitter sizes.
        """
        if self.__splitter and self.__widget:
            splitter = self.__splitter
            index = splitter.indexOf(self.__widget)
            sizes = splitter.sizes()
            current = sizes[index]
            diff = current - self.__size
            sizes[index] = self.__size
            sizes[index - 1] = sizes[index - 1] + diff

            self.__splitter.setSizes(sizes)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Resize:
            if self.__splitter.orientation() == Qt.Vertical:
                size = event.size().height()
            else:
                size = event.size().width()

            if self.__expanded and size == 0:
                self.__action.setChecked(False)
                self.__expanded = False
            elif not self.__expanded and size > 0:
                self.__action.setChecked(True)
                self.__expanded = True

        return QObject.eventFilter(self, obj, event)


class QuickHelpWidget(QuickHelp):
    def minimumSizeHint(self):
        """Reimplemented to allow the Splitter to resize the widget
        with a continuous animation.

        """
        hint = QTextEdit.minimumSizeHint(self)
        return QSize(hint.width(), 0)


class CanvasToolDock(QWidget):
    """Canvas dock widget with widget toolbox, quick help and
    canvas actions.

    """
    def __init__(self, parent=None, **kwargs):
        QWidget.__init__(self, parent, **kwargs)

        self.__setupUi()

    def __setupUi(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.toolbox = WidgetToolBox()

        self.help = QuickHelpWidget(objectName="quick-help")

        self.__splitter = QSplitter()
        self.__splitter.setOrientation(Qt.Vertical)

        self.__splitter.addWidget(self.toolbox)
        self.__splitter.addWidget(self.help)

        self.toolbar = DynamicResizeToolBar()
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)

        self.toolbar.setSizePolicy(QSizePolicy.Ignored,
                                   QSizePolicy.Preferred)

        layout.addWidget(self.__splitter, 10)
        layout.addWidget(self.toolbar)

        self.setLayout(layout)
        self.__splitterResizer = SplitterResizer()
        self.__splitterResizer.setSplitterAndWidget(self.__splitter, self.help)

    def setQuickHelpVisible(self, state):
        """Set the quick help box visibility status.
        """
        self.__splitterResizer.setExpanded(state)

    def quickHelpVisible(self):
        return self.__splitterResizer.expanded()

    def setQuickHelpAnimationEnabled(self, enabled):
        """Enable/disable the quick help animation.
        """
        self.__splitterResizer.setAnimationEnabled(enabled)

    def toogleQuickHelpAction(self):
        """Return a checkable QAction for help show/hide.
        """
        return self.__splitterResizer.toogleExpandedAction()


class QuickCategoryToolbar(ToolGrid):
    """A toolbar with category buttons.
    """
    def __init__(self, parent=None, buttonSize=None, iconSize=None):
        ToolGrid.__init__(self, parent, 1, buttonSize, iconSize,
                          Qt.ToolButtonIconOnly)
        self.__model = None

    def setColumnCount(self, count):
        raise Exception("Cannot set the column count on a Toolbar")

    def setModel(self, model):
        """Set the registry model.
        """
        if self.__model is not None:
            self.__model.itemChanged.disconnect(self.__on_itemChanged)
            self.__model.rowsInserted.disconnect(self.__on_rowsInserted)
            self.__model.rowsRemoved.disconnect(self.__on_rowsRemoved)
            self.clear()

        self.__model = model
        if self.__model is not None:
            self.__model.itemChanged.connect(self.__on_itemChanged)
            self.__model.rowsInserted.connect(self.__on_rowsInserted)
            self.__model.rowsRemoved.connect(self.__on_rowsRemoved)
            self.__initFromModel(model)

    def __initFromModel(self, model):
        """Initialize the toolbar from the model.
        """
        root = model.invisibleRootItem()
        for item in iter_item(root):
            action = self.createActionForItem(item)
            self.addAction(action)

    def createActionForItem(self, item):
        """Create the QAction instance for item.
        """
        action = QAction(item.icon(), item.text(), self,
                         toolTip=item.toolTip())
        action.setData(item)
        return action

    def createButtonForAction(self, action):
        """Create a button for the action.
        """
        button = ToolGrid.createButtonForAction(self, action)

        item = action.data()
        if item.data(Qt.BackgroundRole) is not None:
            brush = item.background()
        elif item.data(QtWidgetRegistry.BACKGROUND_ROLE) is not None:
            brush = item.data(QtWidgetRegistry.BACKGROUND_ROLE)
        else:
            brush = self.palette().brush(QPalette.Button)

        palette = button.palette()
        palette.setColor(QPalette.Button, brush.color())
        palette.setColor(QPalette.Window, brush.color())
        button.setPalette(palette)
        button.setProperty("quick-category-toolbutton", True)

        style_sheet = ("QToolButton {\n"
                       "    background: %s;\n"
                       "    border: none;\n"
                       "    border-bottom: 1px solid palette(mid);\n"
                       "}")
        button.setStyleSheet(style_sheet % create_css_gradient(brush.color()))

        return button

    def __on_itemChanged(self, item):
        root = self.__model.invisibleRootItem()
        if item.parentItem() == root:
            row = item.row()
            action = self._gridSlots[row].action
            action.setText(item.text())
            action.setIcon(item.icon())
            action.setToolTip(item.toolTip())

    def __on_rowsInserted(self, parent, start, end):
        root = self.__model.invisibleRootItem()
        if root == parent:
            for index in range(start, end + 1):
                item = parent.child(index)
                self.addAction(self.createActionForItem(item))

    def __on_rowsRemoved(self, parent, start, end):
        root = self.__model.invisibleRootItem()
        if root == parent:
            for index in range(end, start - 1, -1):
                action = self._gridSlots[index].action
                self.removeAction(action)


class CategoryPopupMenu(FramelessWindow):
    triggered = Signal(QAction)
    hovered = Signal(QAction)

    def __init__(self, parent=None, **kwargs):
        FramelessWindow.__init__(self, parent, **kwargs)
        self.setWindowFlags(self.windowFlags() | Qt.Popup)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)

        self.__menu = MenuPage()
        self.__menu.setActionRole(QtWidgetRegistry.WIDGET_ACTION_ROLE)

        if sys.platform == "darwin":
            self.__menu.view().setAttribute(Qt.WA_MacShowFocusRect, False)

        self.__menu.triggered.connect(self.__onTriggered)
        self.__menu.hovered.connect(self.hovered)

        self.__dragListener = ItemViewDragStartEventListener(self)
        self.__dragListener.dragStarted.connect(self.__onDragStarted)

        self.__menu.view().viewport().installEventFilter(self.__dragListener)

        layout.addWidget(self.__menu)

        self.setLayout(layout)

        self.__action = None
        self.__loop = None
        self.__item = None

    def setCategoryItem(self, item):
        """
        Set the category root item (:class:`QStandardItem`).
        """
        self.__item = item
        model = item.model()
        self.__menu.setModel(model)
        self.__menu.setRootIndex(item.index())

    def popup(self, pos=None):
        if pos is None:
            pos = self.pos()
        geom = widget_popup_geometry(pos, self)
        self.setGeometry(geom)
        self.show()

    def exec_(self, pos=None):
        self.popup(pos)
        self.__loop = QEventLoop()

        self.__action = None
        self.__loop.exec_()
        self.__loop = None

        if self.__action is not None:
            action = self.__action
        else:
            action = None
        return action

    def hideEvent(self, event):
        if self.__loop is not None:
            self.__loop.exit(0)

        return FramelessWindow.hideEvent(self, event)

    def __onTriggered(self, action):
        self.__action = action
        self.triggered.emit(action)
        self.hide()

        if self.__loop:
            self.__loop.exit(0)

    def __onDragStarted(self, index):
        desc = index.data(QtWidgetRegistry.WIDGET_DESC_ROLE)
        icon = index.data(Qt.DecorationRole)

        drag_data = QMimeData()
        drag_data.setData(
            "application/vnv.orange-canvas.registry.qualified-name",
            desc.qualified_name
        )
        drag = QDrag(self)
        drag.setPixmap(icon.pixmap(38))
        drag.setMimeData(drag_data)

        # TODO: Should animate (accept) hide.
        self.hide()

        # When a drag is started and the menu hidden the item's tool tip
        # can still show for a short time UNDER the cursor preventing a
        # drop.
        viewport = self.__menu.view().viewport()
        filter = ToolTipEventFilter()
        viewport.installEventFilter(filter)

        drag.exec_(Qt.CopyAction)

        viewport.removeEventFilter(filter)


class ItemViewDragStartEventListener(QObject):
    dragStarted = Signal(QModelIndex)

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self._pos = None
        self._index = None

    def eventFilter(self, viewport, event):
        view = viewport.parent()

        if event.type() == QEvent.MouseButtonPress and \
                event.button() == Qt.LeftButton:

            index = view.indexAt(event.pos())

            if index is not None:
                self._pos = event.pos()
                self._index = QPersistentModelIndex(index)

        elif event.type() == QEvent.MouseMove and self._pos is not None and \
                ((self._pos - event.pos()).manhattanLength() >=
                 QApplication.startDragDistance()):

            if self._index.isValid():
                # Map to a QModelIndex in the model.
                index = self._index
                index = index.model().index(index.row(), index.column(),
                                            index.parent())
                self._pos = None
                self._index = None

                self.dragStarted.emit(index)

        return QObject.eventFilter(self, view, event)


class ToolTipEventFilter(QObject):
    def eventFilter(self, receiver, event):
        if event.type() == QEvent.ToolTip:
            return True

        return QObject.eventFilter(self, receiver, event)


def widget_popup_geometry(pos, widget):
    widget.ensurePolished()

    if widget.testAttribute(Qt.WA_Resized):
        size = widget.size()
    else:
        size = widget.sizeHint()

    desktop = QApplication.desktop()
    screen_geom = desktop.availableGeometry(pos)

    # Adjust the size to fit inside the screen.
    if size.height() > screen_geom.height():
        size.setHeight(screen_geom.height())
    if size.width() > screen_geom.width():
        size.setWidth(screen_geom.width())

    geom = QRect(pos, size)

    if geom.top() < screen_geom.top():
        geom.setTop(screen_geom.top())

    if geom.left() < screen_geom.left():
        geom.setLeft(screen_geom.left())

    bottom_margin = screen_geom.bottom() - geom.bottom()
    right_margin = screen_geom.right() - geom.right()
    if bottom_margin < 0:
        # Falls over the bottom of the screen, move it up.
        geom.translate(0, bottom_margin)

    # TODO: right to left locale
    if right_margin < 0:
        # Falls over the right screen edge, move the menu to the
        # other side of pos.
        geom.translate(-size.width(), 0)

    return geom


def popup_position_from_source(popup, source, orientation=Qt.Vertical):
    popup.ensurePolished()
    source.ensurePolished()

    if popup.testAttribute(Qt.WA_Resized):
        size = popup.size()
    else:
        size = popup.sizeHint()

    desktop = QApplication.desktop()
    screen_geom = desktop.availableGeometry(source)
    source_rect = QRect(source.mapToGlobal(QPoint(0, 0)), source.size())

    if orientation == Qt.Vertical:
        if source_rect.right() + size.width() < screen_geom.right():
            x = source_rect.right()
        else:
            x = source_rect.left() - size.width()

        # bottom overflow
        dy = source_rect.top() + size.height() - screen_geom.bottom()
        if dy < 0:
            y = source_rect.top()
        else:
            y = source_rect.top() - dy
    else:
        # right overflow
        dx = source_rect.left() + size.width() - screen_geom.right()
        if dx < 0:
            x = source_rect.left()
        else:
            x = source_rect.left() - dx

        if source_rect.bottom() + size.height() < screen_geom.bottom():
            y = source_rect.bottom()
        else:
            y = source_rect.top() - size.height()

    return QPoint(x, y)
