"""
A widget containing a grid of clickable actions/buttons.

"""
from collections import namedtuple, deque

from AnyQt.QtWidgets import (
    QFrame, QAction, QToolButton, QGridLayout,  QSizePolicy,
    QStyleOptionToolButton, QStylePainter, QStyle
)
from AnyQt.QtGui import QFontMetrics
from AnyQt.QtCore import Qt, QObject, QSize, QEvent, QSignalMapper
from AnyQt.QtCore import pyqtSignal as Signal

from . import utils


_ToolGridSlot = namedtuple(
    "_ToolGridSlot",
    ["button",
     "action",
     "row",
     "column"
     ]
    )


class _ToolGridButton(QToolButton):
    def __init__(self, *args, **kwargs):
        QToolButton.__init__(self, *args, **kwargs)

        self.__text = ""

    def actionEvent(self, event):
        QToolButton.actionEvent(self, event)
        if event.type() == QEvent.ActionChanged or \
                event.type() == QEvent.ActionAdded:
            self.__textLayout()

    def resizeEvent(self, event):
        QToolButton.resizeEvent(self, event)
        self.__textLayout()

    def __textLayout(self):
        fm = QFontMetrics(self.font())
        text = str(self.defaultAction().text())
        words = deque(text.split())

        lines = []
        curr_line = ""
        curr_line_word_count = 0

        option = QStyleOptionToolButton()
        option.initFrom(self)

        margin = self.style().pixelMetric(QStyle.PM_ButtonMargin, option, self)
        width = self.width() - 2 * margin

        while words:
            w = words.popleft()

            if curr_line_word_count:
                line_extended = " ".join([curr_line, w])
            else:
                line_extended = w

            line_w = fm.boundingRect(line_extended).width()

            if line_w >= width:
                if curr_line_word_count == 0 or len(lines) == 1:
                    # A single word that is too long must be elided.
                    # Also if the text overflows 2 lines
                    # Warning: hardcoded max lines
                    curr_line = fm.elidedText(line_extended, Qt.ElideRight,
                                              width)
                    curr_line = str(curr_line)
                else:
                    # Put the word back
                    words.appendleft(w)

                lines.append(curr_line)
                curr_line = ""
                curr_line_word_count = 0
                if len(lines) == 2:
                    break
            else:
                curr_line = line_extended
                curr_line_word_count += 1

        if curr_line:
            lines.append(curr_line)

        text = "\n".join(lines)
        text = text.replace('&', '&&')  # Need escaped ampersand to show

        self.__text = text

    def paintEvent(self, event):
        p = QStylePainter(self)
        opt = QStyleOptionToolButton()
        self.initStyleOption(opt)
        if self.__text:
            # Replace the text
            opt.text = self.__text
        p.drawComplexControl(QStyle.CC_ToolButton, opt)
        p.end()


class ToolGrid(QFrame):
    """
    A widget containing a grid of actions/buttons.

    Actions can be added using standard :func:`QWidget.addAction(QAction)`
    and :func:`QWidget.insertAction(int, QAction)` methods.

    Parameters
    ----------
    parent : :class:`QWidget`
        Parent widget.
    columns : int
        Number of columns in the grid layout.
    buttonSize : :class:`QSize`, optional
        Size of tool buttons in the grid.
    iconSize : :class:`QSize`, optional
        Size of icons in the buttons.
    toolButtonStyle : :class:`Qt.ToolButtonStyle`
        Tool button style.

    """

    actionTriggered = Signal(QAction)
    actionHovered = Signal(QAction)

    def __init__(self, parent=None, columns=4, buttonSize=None,
                 iconSize=None, toolButtonStyle=Qt.ToolButtonTextUnderIcon):
        QFrame.__init__(self, parent)

        if buttonSize is not None:
            buttonSize = QSize(buttonSize)

        if iconSize is not None:
            iconSize = QSize(iconSize)

        self.__columns = columns
        self.__buttonSize = buttonSize or QSize(50, 50)
        self.__iconSize = iconSize or QSize(26, 26)
        self.__toolButtonStyle = toolButtonStyle

        self.__gridSlots = []
        self.__mapper = QSignalMapper()
        self.__mapper.mapped[QObject].connect(self.__onClicked)

        self.__setupUi()

    def __setupUi(self):
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setSizeConstraint(QGridLayout.SetFixedSize)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.MinimumExpanding)

    def setButtonSize(self, size):
        """
        Set the button size.
        """
        if self.__buttonSize != size:
            self.__buttonSize = size
            for slot in self.__gridSlots:
                slot.button.setFixedSize(size)

    def buttonSize(self):
        """
        Return the button size.
        """
        return QSize(self.__buttonSize)

    def setIconSize(self, size):
        """
        Set the button icon size.
        """
        if self.__iconSize != size:
            self.__iconSize = size
            for slot in self.__gridSlots:
                slot.button.setIconSize(size)

    def iconSize(self):
        """
        Return the icon size
        """
        return QSize(self.__iconSize)

    def setToolButtonStyle(self, style):
        """
        Set the tool button style.
        """
        if self.__toolButtonStyle != style:
            self.__toolButtonStyle = style
            for slot in self.__gridSlots:
                slot.button.setToolButtonStyle(style)

    def toolButtonStyle(self):
        """
        Return the tool button style.
        """
        return self.__toolButtonStyle

    def setColumnCount(self, columns):
        """
        Set the number of button/action columns.
        """
        if self.__columns != columns:
            self.__columns = columns
            self.__relayout()

    def columns(self):
        """
        Return the number of columns in the grid.
        """
        return self.__columns

    def clear(self):
        """
        Clear all actions/buttons.
        """
        for slot in reversed(list(self.__gridSlots)):
            self.removeAction(slot.action)
        self.__gridSlots = []

    def insertAction(self, before, action):
        """
        Insert a new action at the position currently occupied
        by `before` (can also be an index).

        Parameters
        ----------
        before : :class:`QAction` or int
            Position where the `action` should be inserted.
        action : :class:`QAction`
            Action to insert

        """
        if isinstance(before, int):
            actions = list(self.actions())
            if len(actions) == 0 or before >= len(actions):
                # Insert as the first action or the last action.
                return self.addAction(action)

            before = actions[before]

        return QFrame.insertAction(self, before, action)

    def setActions(self, actions):
        """
        Clear the grid and add `actions`.
        """
        self.clear()

        for action in actions:
            self.addAction(action)

    def buttonForAction(self, action):
        """
        Return the :class:`QToolButton` instance button for `action`.
        """
        actions = [slot.action for slot in self.__gridSlots]
        index = actions.index(action)
        return self.__gridSlots[index].button

    def createButtonForAction(self, action):
        """
        Create and return a :class:`QToolButton` for action.
        """
        button = _ToolGridButton(self)
        button.setDefaultAction(action)

        if self.__buttonSize.isValid():
            button.setFixedSize(self.__buttonSize)
        if self.__iconSize.isValid():
            button.setIconSize(self.__iconSize)

        button.setToolButtonStyle(self.__toolButtonStyle)
        button.setProperty("tool-grid-button", True)
        return button

    def count(self):
        """
        Return the number of buttons/actions in the grid.
        """
        return len(self.__gridSlots)

    def actionEvent(self, event):
        QFrame.actionEvent(self, event)

        if event.type() == QEvent.ActionAdded:
            # Note: the action is already in the self.actions() list.
            actions = list(self.actions())
            index = actions.index(event.action())
            self.__insertActionButton(index, event.action())

        elif event.type() == QEvent.ActionRemoved:
            self.__removeActionButton(event.action())

    def __insertActionButton(self, index, action):
        """Create a button for the action and add it to the layout
        at index.

        """
        self.__shiftGrid(index, 1)
        button = self.createButtonForAction(action)

        row = index // self.__columns
        column = index % self.__columns

        self.layout().addWidget(
            button, row, column,
            Qt.AlignLeft | Qt.AlignTop
        )

        self.__gridSlots.insert(
            index, _ToolGridSlot(button, action, row, column)
        )

        self.__mapper.setMapping(button, action)
        button.clicked.connect(self.__mapper.map)
        button.installEventFilter(self)

    def __removeActionButton(self, action):
        """Remove the button for the action from the layout and delete it.
        """
        actions = [slot.action for slot in self.__gridSlots]
        index = actions.index(action)
        slot = self.__gridSlots.pop(index)

        slot.button.removeEventFilter(self)
        self.__mapper.removeMappings(slot.button)

        self.layout().removeWidget(slot.button)
        self.__shiftGrid(index + 1, -1)

        slot.button.deleteLater()

    def __shiftGrid(self, start, count=1):
        """Shift all buttons starting at index `start` by `count` cells.
        """
        button_count = self.layout().count()
        direction = 1 if count >= 0 else -1
        if direction == 1:
            start, end = button_count - 1, start - 1
        else:
            start, end = start, button_count

        for index in range(start, end, -direction):
            item = self.layout().itemAtPosition(index / self.__columns,
                                                index % self.__columns)
            if item:
                button = item.widget()
                new_index = index + count
                self.layout().addWidget(button, new_index / self.__columns,
                                        new_index % self.__columns,
                                        Qt.AlignLeft | Qt.AlignTop)

    def __relayout(self):
        """Relayout the buttons.
        """
        for i in reversed(range(self.layout().count())):
            self.layout().takeAt(i)

        self.__gridSlots = [_ToolGridSlot(slot.button, slot.action,
                                          i / self.__columns,
                                          i % self.__columns)
                            for i, slot in enumerate(self.__gridSlots)]

        for slot in self.__gridSlots:
            self.layout().addWidget(slot.button, slot.row, slot.column,
                                    Qt.AlignLeft | Qt.AlignTop)

    def __indexOf(self, button):
        """Return the index of button widget.
        """
        buttons = [slot.button for slot in self.__gridSlots]
        return buttons.index(button)

    def __onButtonEnter(self, button):
        action = button.defaultAction()
        self.actionHovered.emit(action)

    def __onClicked(self, action):
        self.actionTriggered.emit(action)

    def paintEvent(self, event):
        return utils.StyledWidget_paintEvent(self, event)

    def eventFilter(self, obj, event):
        etype = event.type()
        if etype == QEvent.KeyPress and obj.hasFocus():
            key = event.key()
            if key in [Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right]:
                if self.__focusMove(obj, key):
                    event.accept()
                    return True
        elif etype == QEvent.HoverEnter and obj.parent() is self:
            self.__onButtonEnter(obj)
        return QFrame.eventFilter(self, obj, event)

    def __focusMove(self, focus, key):
        assert(focus is self.focusWidget())
        try:
            index = self.__indexOf(focus)
        except IndexError:
            return False

        if key == Qt.Key_Down:
            index += self.__columns
        elif key == Qt.Key_Up:
            index -= self.__columns
        elif key == Qt.Key_Left:
            index -= 1
        elif key == Qt.Key_Right:
            index += 1

        if index >= 0 and index < self.count():
            button = self.__gridSlots[index].button
            button.setFocus(Qt.TabFocusReason)
            return True
        else:
            return False
