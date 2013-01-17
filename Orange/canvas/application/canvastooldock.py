"""
Orange Canvas Tool Dock widget

"""
from PyQt4.QtGui import (
    QWidget, QSplitter, QVBoxLayout, QTextEdit, QAction, QPalette,
    QSizePolicy
)

from PyQt4.QtCore import Qt, QSize, QObject, QPropertyAnimation, QEvent
from PyQt4.QtCore import pyqtProperty as Property

from ..gui.toolgrid import ToolGrid
from ..gui.toolbar import DynamicResizeToolBar
from ..gui.quickhelp import QuickHelp
from .widgettoolbox import WidgetToolBox, iter_item

from ..registry.qt import QtWidgetRegistry


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
                       "    background-color: %s;\n"
                       "    border: none;\n"
                       "    border-bottom: 1px solid palette(dark);\n"
                       "}")
        button.setStyleSheet(style_sheet % brush.color().name())

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
