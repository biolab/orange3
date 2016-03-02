"""
A LineEdit class with a button on left/right side.
"""
from collections import namedtuple

from AnyQt.QtWidgets import (
    QLineEdit, QToolButton, QStyleOptionToolButton, QStylePainter,
    QStyle, QAction
)
from AnyQt.QtGui import QPalette, QFontMetrics
from AnyQt.QtCore import Qt, QSize, QRect, QT_VERSION
from AnyQt.QtCore import pyqtSignal as Signal, pyqtProperty as Property


_ActionSlot = \
    namedtuple(
        "_AcitonSlot",
        ["position",  # Left/Right position
         "action",    # QAction
         "button",    # LineEditButton instance
         "autoHide"]  # Auto hide when line edit is empty.
        )


class LineEditButton(QToolButton):
    """
    A button in the :class:`LineEdit`.
    """
    def __init__(self, parent=None, flat=True, **kwargs):
        QToolButton.__init__(self, parent, **kwargs)

        self.__flat = flat

    def setFlat(self, flat):
        if self.__flat != flat:
            self.__flat = flat
            self.update()

    def flat(self):
        return self.__flat

    flat_ = Property(bool, fget=flat, fset=setFlat,
                     designable=True)

    def paintEvent(self, event):
        if self.__flat:
            opt = QStyleOptionToolButton()
            self.initStyleOption(opt)
            p = QStylePainter(self)
            p.drawControl(QStyle.CE_ToolButtonLabel, opt)
        else:
            QToolButton.paintEvent(self, event)


class LineEdit(QLineEdit):
    """
    A line edit widget with support for adding actions (buttons) to
    the left/right of the edited text

    """
    #: Position flags
    LeftPosition, RightPosition = 1, 2

    #: Emitted when the action is triggered.
    triggered = Signal(QAction)

    #: The left action was triggered.
    leftTriggered = Signal()

    #: The right action was triggered.
    rightTriggered = Signal()

    def __init__(self, parent=None, **kwargs):
        QLineEdit.__init__(self, parent, **kwargs)
        self.__actions = [None, None]

    def setAction(self, action, position=LeftPosition):
        """
        Set `action` to be displayed at `position`. Existing action
        (if present) will be removed.

        Parameters
        ----------
        action : :class:`QAction`
        position : int
            Position where to set the action (default: ``LeftPosition``).

        """

        curr = self.actionAt(position)
        if curr is not None:
            self.removeAction(position)

        # Add the action using QWidget.addAction (for shortcuts)
        QLineEdit.addAction(self, action)

        button = LineEditButton(self)
        button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        button.setDefaultAction(action)
        button.setVisible(self.isVisible())
        button.show()
        button.setCursor(Qt.ArrowCursor)

        button.triggered.connect(self.triggered)
        button.triggered.connect(self.__onTriggered)

        slot = _ActionSlot(position, action, button, False)
        self.__actions[position - 1] = slot

        if not self.testAttribute(Qt.WA_Resized):
            # Need some sensible height to do the layout.
            self.adjustSize()

        self.__layoutActions()

    def actionAt(self, position):
        """
        Return :class:`QAction` at `position`.
        """
        self._checkPosition(position)
        slot = self.__actions[position - 1]
        if slot:
            return slot.action
        else:
            return None

    def removeActionAt(self, position):
        """
        Remove the action at position.
        """
        self._checkPosition(position)

        slot = self.__actions[position - 1]
        self.__actions[position - 1] = None

        slot.button.hide()
        slot.button.deleteLater()
        QLineEdit.removeAction(self, slot.action)
        self.__layoutActions()

    def button(self, position):
        """
        Return the button (:class:`LineEditButton`) for the action
        at `position`.

        """
        self._checkPosition(position)
        slot = self.__actions[position - 1]
        if slot:
            return slot.button
        else:
            return None

    def _checkPosition(self, position):
        if position not in [self.LeftPosition, self.RightPosition]:
            raise ValueError("Invalid position")

    def resizeEvent(self, event):
        QLineEdit.resizeEvent(self, event)
        self.__layoutActions()

    if QT_VERSION < 0x40700:
        # Qt 4.6 does not yet have placeholder text
        def setPlaceholderText(self, text):
            self.__placeholderText = text
            self.update()

        def placeholderText(self):
            try:
                return self.__placeholderText
            except AttributeError:
                return ""

        def paintEvent(self, event):
            QLineEdit.paintEvent(self, event)
            if not self.text() and self.placeholderText() and \
                    not self.hasFocus():
                p = QStylePainter(self)
                font = self.font()
                metrics = QFontMetrics(font)
                p.setFont(font)
                color = self.palette().color(QPalette.Mid)
                p.setPen(color)
                left, top, right, bottom = self.getTextMargins()
                contents = self.contentsRect()
                contents = contents.adjusted(left, top, -right, -bottom)
                text = metrics.elidedText(self.placeholderText(),
                                          Qt.ElideMiddle,
                                          contents.width())
                p.drawText(contents, Qt.AlignLeft | Qt.AlignVCenter, text)

    def __layoutActions(self):
        left, right = self.__actions

        contents = self.contentsRect()
        buttonSize = QSize(contents.height(), contents.height())

        margins = self.textMargins()

        if left:
            geom = QRect(contents.topLeft(), buttonSize)
            left.button.setGeometry(geom)
            margins.setLeft(buttonSize.width())

        if right:
            geom = QRect(contents.topRight(), buttonSize)
            right.button.setGeometry(geom.translated(-buttonSize.width(), 0))
            margins.setLeft(buttonSize.width())

        self.setTextMargins(margins)

    def __onTriggered(self, action):
        left, right = self.__actions
        if left and action == left.action:
            self.leftTriggered.emit()
        elif right and action == right.action:
            self.rightTriggered.emit()
