"""
Welcome Screen Dialog
"""
from types import SimpleNamespace
from typing import List  # pylint: disable=unused-import

from AnyQt.QtWidgets import (
    QDialog, QWidget, QToolButton, QCheckBox, QAction,
    QHBoxLayout, QVBoxLayout, QSizePolicy, QLabel,
    QListView, QDialogButtonBox, QStackedWidget,
    QStyle, QStyledItemDelegate, QStyleOption, QStyleOptionViewItem,
    QFrame
)
from AnyQt.QtGui import (
    QFont, QIcon, QPixmap, QPainter, QColor, QBrush,
    QStandardItemModel, QStandardItem
)

from AnyQt.QtCore import (  # pylint: disable=unused-import
    Qt, QEvent, QRect, QSize, QPoint, QModelIndex, QItemSelectionModel
)
from AnyQt.QtCore import pyqtSignal as Signal

from ..gui.iconview import LinearIconView
from ..canvas.items.utils import radial_gradient
from ..registry import NAMED_COLORS


def decorate_welcome_icon(icon, background_color):
    """Return a `QIcon` with a circle shaped background.
    """
    welcome_icon = QIcon()
    sizes = [32, 48, 64, 80, 128, 256]
    background_color = NAMED_COLORS.get(background_color, background_color)
    background_color = QColor(background_color)
    grad = radial_gradient(background_color)
    for size in sizes:
        icon_size = QSize(5 * size / 8, 5 * size / 8)
        icon_rect = QRect(QPoint(0, 0), icon_size)
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setBrush(QBrush(grad))
        p.setPen(Qt.NoPen)
        ellipse_rect = QRect(0, 0, size, size)
        p.drawEllipse(ellipse_rect)
        icon_rect.moveCenter(ellipse_rect.center())
        icon.paint(p, icon_rect, Qt.AlignCenter, )
        p.end()

        welcome_icon.addPixmap(pixmap)

    return welcome_icon


WELCOME_WIDGET_BUTTON_STYLE = \
"""

WelcomeActionButton {
    border: none;
    icon-size: 75px;
    /*font: bold italic 14px "Helvetica";*/
}

WelcomeActionButton:pressed {
    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #dadbde, stop: 1 #f6f7fa);
    border-radius: 10px;
}

WelcomeActionButton:focus {
    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #dadbde, stop: 1 #f6f7fa);
    border-radius: 10px;
}

"""


class WelcomeActionButton(QToolButton):
    def __init__(self, parent=None):
        QToolButton.__init__(self, parent)

    def paintEvent(self, event):
        QToolButton.paintEvent(self, event)


class WelcomeDialog(QDialog):
    """A welcome widget shown at startup presenting a series
    of buttons (actions) for a beginner to choose from.

    """
    triggered = Signal(QAction)

    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)

        self.__triggeredAction = None

        self.setupUi()

    def setupUi(self):
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self.__mainLayout = QVBoxLayout()
        self.__mainLayout.setContentsMargins(0, 40, 0, 40)
        self.__mainLayout.setSpacing(65)

        self.layout().addLayout(self.__mainLayout)

        self.setStyleSheet(WELCOME_WIDGET_BUTTON_STYLE)

        bottom_bar = QWidget(objectName="bottom-bar")
        bottom_bar_layout = QHBoxLayout()
        bottom_bar_layout.setContentsMargins(20, 10, 20, 10)
        bottom_bar.setLayout(bottom_bar_layout)
        bottom_bar.setSizePolicy(QSizePolicy.MinimumExpanding,
                                 QSizePolicy.Maximum)

        check = QCheckBox(self.tr("Show at startup"), bottom_bar)
        check.setChecked(False)

        self.__showAtStartupCheck = check

        feedback = QLabel(
            '<a href="http://orange.biolab.si/survey/long.html">Help us improve!</a>')
        feedback.setTextInteractionFlags(Qt.TextBrowserInteraction)
        feedback.setOpenExternalLinks(True)

        bottom_bar_layout.addWidget(check, alignment=Qt.AlignVCenter | \
                                    Qt.AlignLeft)
        bottom_bar_layout.addWidget(feedback, alignment=Qt.AlignVCenter | \
                                    Qt.AlignRight)

        self.layout().addWidget(bottom_bar, alignment=Qt.AlignBottom,
                                stretch=1)

        self.setSizeGripEnabled(False)
        self.setFixedSize(620, 390)

    def setShowAtStartup(self, show):
        """
        Set the 'Show at startup' check box state.
        """
        if self.__showAtStartupCheck.isChecked() != show:
            self.__showAtStartupCheck.setChecked(show)

    def showAtStartup(self):
        """
        Return the 'Show at startup' check box state.
        """
        return self.__showAtStartupCheck.isChecked()

    def addRow(self, actions, background="light-orange"):
        """Add a row with `actions`.
        """
        count = self.__mainLayout.count()
        self.insertRow(count, actions, background)

    def insertRow(self, index, actions, background="light-orange"):
        """Insert a row with `actions` at `index`.
        """
        widget = QWidget(objectName="icon-row")
        layout = QHBoxLayout()
        layout.setContentsMargins(40, 0, 40, 0)
        layout.setSpacing(65)
        widget.setLayout(layout)

        self.__mainLayout.insertWidget(index, widget, stretch=10,
                                       alignment=Qt.AlignCenter)

        for i, action in enumerate(actions):
            self.insertAction(index, i, action, background)

    def insertAction(self, row, index, action,
                      background="light-orange"):
        """Insert `action` in `row` in position `index`.
        """
        button = self.createButton(action, background)
        self.insertButton(row, index, button)

    def insertButton(self, row, index, button):
        """Insert `button` in `row` in position `index`.
        """
        item = self.__mainLayout.itemAt(row)
        layout = item.widget().layout()
        layout.insertWidget(index, button)
        button.triggered.connect(self.__on_actionTriggered)

    def createButton(self, action, background="light-orange"):
        """Create a tool button for action.
        """
        button = WelcomeActionButton(self)
        button.setDefaultAction(action)
        button.setText(action.iconText())
        button.setIcon(decorate_welcome_icon(action.icon(), background))
        button.setToolTip(action.toolTip())
        button.setFixedSize(100, 100)
        button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        font = QFont(button.font())
        font.setPointSize(13)
        button.setFont(font)

        return button

    def buttonAt(self, i, j):
        """Return the button at i-t row and j-th column.
        """
        item = self.__mainLayout.itemAt(i)
        row = item.widget()
        item = row.layout().itemAt(j)
        return item.widget()

    def triggeredAction(self):
        """Return the action that was triggered by the user.
        """
        return self.__triggeredAction

    def showEvent(self, event):
        # Clear the triggered action before show.
        self.__triggeredAction = None
        QDialog.showEvent(self, event)

    def __on_actionTriggered(self, action):
        """Called when the button action is triggered.
        """
        self.triggered.emit(action)
        self.__triggeredAction = action


class PagedWidget(QFrame):
    class Page(SimpleNamespace):
        icon = ...  # type: QIcon
        text = ...  # type: str
        toolTip = ...  # type: str
        widget = ...   # type: QWidget

    class TabView(LinearIconView):
        def __init__(self, *args, focusPolicy=Qt.TabFocus, **kwargs):
            super().__init__(*args, focusPolicy=focusPolicy, **kwargs)

        def viewOptions(self):
            # type: () -> QStyleOptionViewItem
            option = super().viewOptions()
            # by default items in views are active only if the view is focused
            if self.isActiveWindow():
                option.state |= QStyle.State_Active
            return option

        def selectionCommand(self, index, event=None):
            # type: (QModelIndex, QEvent) -> QItemSelectionModel.SelectionFlags
            command = super().selectionCommand(index, event)
            if not index.isValid():
                # Prevent deselection on click/drag in an empty view part
                return QItemSelectionModel.NoUpdate
            else:
                # Prevent deselect on click + ctrl modifier
                return command & ~QItemSelectionModel.Deselect

    class TabViewDelegate(QStyledItemDelegate):
        def sizeHint(self, option, index):
            # type: (QStyleOptionViewItem, QModelIndex) -> QSize
            sh = super().sizeHint(option, index)
            widget = option.widget
            if isinstance(widget, PagedWidget.TabView):
                if widget.flow() == QListView.TopToBottom:
                    return sh.expandedTo(QSize(82, 100))
                else:
                    return sh.expandedTo(QSize(100, 82))
            else:
                return sh

        def initStyleOption(self, option, index):
            # type: (QStyleOptionViewItem, QModelIndex) -> None
            super().initStyleOption(option, index)
            widget = option.widget
            if isinstance(widget, PagedWidget.TabView):
                # extend the item rect to cover the whole viewport
                # (probably not a good idea).
                if widget.flow() == QListView.TopToBottom:
                    option.rect.setLeft(0)
                    option.rect.setRight(widget.viewport().width())
                else:
                    option.rect.setTop(0)
                    option.rect.setBottom(widget.viewport().height())

            if option.state & QStyle.State_Selected:
                # make sure the selection highlights cover the whole area
                option.showDecorationSelected = True

    #: Signal emitted when the current displayed widget changes
    currentIndexChanged = Signal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__pages = []  # type: List[PagedWidget.Page]
        self.__currentIndex = -1

        self.setContentsMargins(0, 0, 0, 0)
        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self.__tabview = PagedWidget.TabView(
            viewMode=QListView.IconMode,
            flow=QListView.TopToBottom,
            editTriggers=QListView.NoEditTriggers,
            uniformItemSizes=True,
            horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOff
        )
        self.__tabview.setAttribute(Qt.WA_LayoutUsesWidgetRect)
        self.__tabview.setContentsMargins(0, 0, 0, 0)
        self.__tabview.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Expanding)

        self.__tabview.setItemDelegate(PagedWidget.TabViewDelegate())
        self.__tabview.setModel(QStandardItemModel(self))
        self.__tabview.selectionModel().selectionChanged.connect(
            self.__on_activated, Qt.UniqueConnection
        )
        iconsize = self.style().pixelMetric(QStyle.PM_LargeIconSize) * 3 // 2
        self.__tabview.setIconSize(QSize(iconsize, iconsize))
        self.__tabview.setAttribute(Qt.WA_MacShowFocusRect, False)

        self.__stack = QStackedWidget(objectName="contents")
        self.__stack.setContentsMargins(0, 0, 0, 0)
        self.__stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.layout().addWidget(self.__tabview)
        self.layout().addWidget(self.__stack)

    def currentIndex(self):
        # type: () -> int
        return self.__currentIndex

    def setCurrentIndex(self, index):
        # type: (int) -> None
        assert index < self.count()
        if self.__currentIndex != index:
            self.__currentIndex = index
            if index < 0:
                self.__tabview.selectionModel().clearSelection()
            else:
                self.__tabview.selectionModel().select(
                    self.__tabview.model().index(index, 0),
                    QItemSelectionModel.ClearAndSelect
                )
            self.__stack.setCurrentIndex(index)
            self.currentIndexChanged.emit(index)

    def count(self):
        # type: () -> int
        return len(self.__pages)

    def addPage(self, icon, text, widget):
        # type: (QIcon, str, QWidget) -> int
        return self.insertPage(len(self.__pages), icon, text, widget)

    def insertPage(self, index, icon, text, widget):
        # type: (int, QIcon, str, QWidget) -> int
        if not 0 <= index < self.count():
            index = self.count()

        page = PagedWidget.Page(
            icon=QIcon(icon), text=text, toolTip="", widget=widget
        )
        item = QStandardItem()
        item.setIcon(icon)
        item.setText(text)

        self.__pages.insert(index, page)
        self.__tabview.model().insertRow(index, item)
        self.__stack.insertWidget(index, page.widget)

        if len(self.__pages) == 1:
            self.setCurrentIndex(0)
        elif index <= self.__currentIndex:
            self.__currentIndex += 1
        return index

    def removePage(self, index):
        # type: (int) -> None
        if 0 <= index < len(self.__pages):
            page = self.__pages[index]
            model = self.__tabview.model()  # type: QStandardItemModel
            currentIndex = self.__currentIndex
            if index < currentIndex:
                newCurrent = currentIndex - 1
            else:
                newCurrent = currentIndex
            selmodel = self.__tabview.selectionModel()
            selmodel.selectionChanged.disconnect(self.__on_activated)
            model.removeRow(index)
            del self.__pages[index]
            self.__stack.removeWidget(page.widget)
            selmodel.selectionChanged.connect(
                self.__on_activated, Qt.UniqueConnection)
            self.setCurrentIndex(newCurrent)

    def widget(self, index):
        # type: (int) -> QWidget
        return self.__pages[index].widget

    def setPageEnabled(self, index, enabled):
        # type: (int, bool) -> None
        item = self.__tabview.model().item(index)  # type: QStandardItem
        if item is not None:
            flags = item.flags()
            if enabled:
                flags = flags | Qt.ItemIsEnabled | Qt.ItemIsSelectable
            else:
                flags = flags & ~(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item.setFlags(flags)

    def isPageEnabled(self, index):
        # type: (int) -> bool
        item = self.__tabview.model().item(index)
        return bool(item.flags() & Qt.ItemIsEnabled)

    def setPageToolTip(self, index, toolTip):
        # type: (int, str) -> None
        if 0 <= index < self.count():
            model = self.__tabview.model()  # type: QStandardItemModel
            item = model.item(index, 0)
            item.setToolTip(toolTip)

    def pageToolTip(self, index):
        model = self.__tabview.model()  # type: QStandardItemModel
        return model.item(index, 0).toolTip()

    def __on_activated(self, selected, deselected):
        indexes = selected.indexes()
        if len(indexes) == 1:
            self.setCurrentIndex(indexes[0].row())
        elif len(indexes) == 0:
            self.setCurrentIndex(-1)
        else:
            assert False, "Invalid selection mode"


class PagedDialog(QDialog):
    """
    A paged dialog widget.

    A paged widget dialog displays a tabbed paged interface
    """
    currentIndexChanged = Signal(int)

    class BottomBar(QWidget):
        def paintEvent(self, event):
            style = self.style()  # type: QStyle
            option = QStyleOption()
            option.initFrom(self)
            p = QPainter(self)
            style.drawPrimitive(QStyle.PE_PanelStatusBar, option, p, self)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setContentsMargins(0, 0, 0, 0)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        self.__pageview = PagedWidget()
        self.__pageview.currentIndexChanged.connect(self.currentIndexChanged)

        self.__bottom = PagedDialog.BottomBar(objectName="bottom-area")
        self.__bottom.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.__bottom.setLayout(QHBoxLayout())

        self.__buttons = QDialogButtonBox(objectName="dialog-buttons")
        self.__buttons.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.__buttons.setVisible(False)
        self.__buttons.rejected.connect(self.reject)

        self.__bottom.layout().addWidget(self.__buttons)

        self.layout().addWidget(self.__pageview)
        self.layout().addWidget(self.__bottom)

    def currentIndex(self):
        # type: () -> int
        return self.__pageview.currentIndex()

    def setCurrentIndex(self, index):
        # type: (int) -> None
        self.__pageview.setCurrentIndex(index)

    def count(self):
        # type: () -> int
        return self.__pageview.count()

    def addPage(self, icon, text, widget):
        # type: (QIcon, str, QWidget) -> int
        return self.__pageview.addPage(icon, text, widget)

    def insertPage(self, index, icon, text, widget):
        # type: (int, QIcon, str, QWidget) -> int
        return self.__pageview.insertPage(index, icon, text, widget)

    def removePage(self, index):
        # type: (int) -> None
        return self.__pageview.removePage(index)

    def widget(self, index):
        # type: (int) -> QWidget
        return self.__pageview.widget(index)

    def setPageEnabled(self, index, enabled):
        # type: (int, bool) -> None
        self.__pageview.setPageEnabled(index, enabled)

    def isPageEnabled(self, index):
        # type: (int) -> bool
        return self.__pageview.isPageEnabled(index)

    def buttonBox(self):
        # type: () -> QDialogButtonBox
        """
        Return a QDialogButtonBox instance.
        """
        return self.__buttons
