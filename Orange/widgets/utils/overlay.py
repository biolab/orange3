"""
Overlay Message Widget
----------------------

A Widget to display a temporary dismissable message over another widget.

"""

import sys
import enum
import functools
import operator
from collections import namedtuple

from PyQt4.QtGui import (
    QIcon, QHBoxLayout, QPushButton, QLabel, QSizePolicy, QStyle,
    QAbstractButton, QStyleOptionButton, QStylePainter, QFocusFrame,
    QWidget, QPixmap, QStyleOption, QPainter
)
from PyQt4.QtCore import Qt, QSize, QRect, QPoint, QEvent, QTimer
from PyQt4.QtCore import pyqtSignal as Signal, pyqtSlot as Slot


class OverlayWidget(QWidget):
    """
    A widget positioned on top of another widget.
    """
    def __init__(self, parent=None, alignment=Qt.AlignCenter, **kwargs):
        super().__init__(parent, **kwargs)
        self.setContentsMargins(0, 0, 0, 0)
        self.__alignment = alignment
        self.__widget = None

    def setWidget(self, widget):
        """
        Set the widget over which this overlay should be displayed (anchored).

        :type widget: QWidget
        """
        if self.__widget is not None:
            self.__widget.removeEventFilter(self)
            self.__widget.destroyed.disconnect(self.__on_destroyed)
        self.__widget = widget
        if self.__widget is not None:
            self.__widget.installEventFilter(self)
            self.__widget.destroyed.connect(self.__on_destroyed)

        if self.__widget is None:
            self.hide()
        else:
            self.__layout()

    def widget(self):
        """
        Return the overlaid widget.

        :rtype: QWidget | None
        """
        return self.__widget

    def setAlignment(self, alignment):
        """
        Set overlay alignment.

        :type alignment: Qt.Alignment
        """
        if self.__alignment != alignment:
            self.__alignment = alignment
            if self.__widget is not None:
                self.__layout()

    def alignment(self):
        """
        Return the overlay alignment.

        :rtype: Qt.Alignment
        """
        return self.__alignment

    def eventFilter(self, recv, event):
        # reimplemented
        if recv is self.__widget:
            if event.type() == QEvent.Resize or event.type() == QEvent.Move:
                self.__layout()
            elif event.type() == QEvent.Show:
                self.show()
            elif event.type() == QEvent.Hide:
                self.hide()
        return super().eventFilter(recv, event)

    def event(self, event):
        # reimplemented
        if event.type() == QEvent.LayoutRequest:
            self.__layout()
            return True
        else:
            return super().event(event)

    def paintEvent(self, event):
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, painter, self)

    def __layout(self):
        # position itself over `widget`
        widget = self.__widget
        if widget is None:
            return

        alignment = self.__alignment
        policy = self.sizePolicy()

        if widget.isWindow():
            bounds = widget.geometry()
        else:

            bounds = QRect(widget.mapToGlobal(QPoint(0, 0)),
                           widget.size())
        if self.isWindow():
            bounds = bounds
        else:
            bounds = QRect(self.parent().mapFromGlobal(bounds.topLeft()),
                           bounds.size())

        sh = self.sizeHint()
        minsh = self.minimumSizeHint()
        minsize = self.minimumSize()
        if minsize.isNull():
            minsize = minsh
        maxsize = bounds.size().boundedTo(self.maximumSize())
        minsize = minsize.boundedTo(maxsize)
        effectivesh = sh.expandedTo(minsize).boundedTo(maxsize)

        hpolicy = policy.horizontalPolicy()
        vpolicy = policy.verticalPolicy()

        def getsize(hint, minimum, maximum, policy):
            if policy == QSizePolicy.Ignored:
                return maximum
            elif policy & QSizePolicy.ExpandFlag:
                return maximum
            else:
                return max(hint, minimum)

        width = getsize(effectivesh.width(), minsize.width(),
                        maxsize.width(), hpolicy)

        heightforw = self.heightForWidth(width)
        if heightforw > 0:
            height = getsize(heightforw, minsize.height(),
                             maxsize.height(), vpolicy)
        else:
            height = getsize(effectivesh.height(), minsize.height(),
                             maxsize.height(), vpolicy)

        size = QSize(width, height)
        if alignment & Qt.AlignLeft:
            x = bounds.x()
        elif alignment & Qt.AlignRight:
            x = bounds.right() - size.width()
        else:
            x = bounds.x() + max(0, bounds.width() - size.width()) // 2

        if alignment & Qt.AlignTop:
            y = bounds.y()
        elif alignment & Qt.AlignBottom:
            y = bounds.bottom() - size.height()
        else:
            y = bounds.y() + max(0, bounds.height() - size.height()) // 2

        geom = QRect(QPoint(x, y), size)
        self.setGeometry(geom)

    @Slot()
    def __on_destroyed(self):
        self.__widget = None
        if self.isVisible():
            self.hide()


class SimpleButton(QAbstractButton):
    """
    A simple icon button widget.
    """
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__focusframe = None

    def focusInEvent(self, event):
        # reimplemented
        event.accept()
        self.__focusframe = QFocusFrame(self)
        self.__focusframe.setWidget(self)

    def focusOutEvent(self, event):
        # reimplemented
        event.accept()
        self.__focusframe.deleteLater()
        self.__focusframe = None

    def sizeHint(self):
        # reimplemented
        self.ensurePolished()
        iconsize = self.iconSize()
        icon = self.icon()
        if not icon.isNull():
            iconsize = icon.actualSize(iconsize)
        return iconsize

    def minimumSizeHint(self):
        # reimplemented
        return self.sizeHint()

    def paintEvent(self, event):
        # reimplemented
        painter = QStylePainter(self)
        option = QStyleOptionButton()
        option.initFrom(self)
        option.icon = self.icon()
        option.iconSize = self.iconSize()

        icon = self.icon()

        if not icon.isNull():
            if option.state & QStyle.State_Active:
                mode = (QIcon.Normal if option.state & QStyle.State_MouseOver
                        else QIcon.Active)
            else:
                mode = QIcon.Disabled
            pixmap = icon.pixmap(option.iconSize, mode, )

            painter.drawItemPixmap(option.rect, Qt.AlignCenter, pixmap)


class MessageWidget(QWidget):
    """
    A widget displaying a simple message to the user.

    This is an alternative to a full QMessageBox intended for inline
    modeless messages.

    [[icon] {Message text} (Ok) (Cancel)]
    """
    #: Emitted when a button with the AcceptRole is clicked
    accepted = Signal()
    #: Emitted when a button with the RejectRole is clicked
    rejected = Signal()
    #: Emitted when a button with the HelpRole is clicked
    helpRequested = Signal()
    #: Emitted when a button is clicked
    clicked = Signal(QAbstractButton)

    class StandardButton(enum.IntEnum):
        NoButton, Ok, Close, Help = 0x0, 0x1, 0x2, 0x4
    NoButton, Ok, Close, Help = list(StandardButton)

    class ButtonRole(enum.IntEnum):
        InvalidRole, AcceptRole, RejectRole, HelpRole = 0, 1, 2, 3

    InvalidRole, AcceptRole, RejectRole, HelpRole = list(ButtonRole)

    _Button = namedtuple("_Button", ["button", "role", "stdbutton"])

    def __init__(self, parent=None, icon=QIcon(), text="", wordWrap=False,
                 textFormat=Qt.AutoText, standardButtons=NoButton, **kwargs):
        super().__init__(parent, **kwargs)
        self.__text = text
        self.__icon = QIcon()
        self.__wordWrap = wordWrap
        self.__standardButtons = MessageWidget.NoButton
        self.__buttons = []

        layout = QHBoxLayout()
        layout.setContentsMargins(8, 0, 8, 0)

        self.__iconlabel = QLabel(objectName="icon-label")
        self.__iconlabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.__textlabel = QLabel(objectName="text-label", text=text,
                                  wordWrap=wordWrap, textFormat=textFormat)

        if sys.platform == "darwin":
            self.__textlabel.setAttribute(Qt.WA_MacSmallSize)

        layout.addWidget(self.__iconlabel)
        layout.addWidget(self.__textlabel)

        self.setLayout(layout)
        self.setIcon(icon)
        self.setStandardButtons(standardButtons)

    def setText(self, text):
        """
        Set the current message text.

        :type message: str
        """
        if self.__text != text:
            self.__text = text
            self.__textlabel.setText(text)

    def text(self):
        """
        Return the current message text.

        :rtype: str
        """
        return self.__text

    def setIcon(self, icon):
        """
        Set the message icon.

        :type icon: QIcon | QPixmap | QString | QStyle.StandardPixmap
        """
        if isinstance(icon, QStyle.StandardPixmap):
            icon = self.style().standardIcon(icon)
        else:
            icon = QIcon(icon)

        if self.__icon != icon:
            self.__icon = QIcon(icon)
            if not self.__icon.isNull():
                size = self.style().pixelMetric(
                    QStyle.PM_SmallIconSize, None, self)
                pm = self.__icon.pixmap(QSize(size, size))
            else:
                pm = QPixmap()

            self.__iconlabel.setPixmap(pm)
            self.__iconlabel.setVisible(not pm.isNull())

    def icon(self):
        """
        Return the current icon.

        :rtype: QIcon
        """
        return QIcon(self.__icon)

    def setWordWrap(self, wordWrap):
        """
        Set the message text wrap property

        :type wordWrap: bool
        """
        if self.__wordWrap != wordWrap:
            self.__wordWrap = wordWrap
            self.__textlabel.setWordWrap(wordWrap)

    def wordWrap(self):
        """
        Return the message text wrap property.

        :rtype: bool
        """
        return self.__wordWrap

    def setTextFormat(self, textFormat):
        """
        Set message text format

        :type textFormat: Qt.TextFormat
        """
        self.__textlabel.setTextFormat(textFormat)

    def textFormat(self):
        """
        Return the message text format.

        :rtype: Qt.TextFormat
        """
        return self.__textlabel.textFormat()

    def changeEvent(self, event):
        # reimplemented
        if event.type() == 177:  # QEvent.MacSizeChange:
            ...
        super().changeEvent(event)

    def setStandardButtons(self, buttons):
        for button in MessageWidget.StandardButton:
            existing = self.button(button)
            if button & buttons and existing is None:
                self.addButton(button)
            elif existing is not None:
                self.removeButton(existing)

    def standardButtons(self):
        return functools.reduce(
            operator.ior,
            (slot.stdbutton for slot in self.__buttons
             if slot.stdbutton is not None),
            MessageWidget.NoButton)

    def addButton(self, button, *rolearg):
        """
        addButton(QAbstractButton, ButtonRole)
        addButton(str, ButtonRole)
        addButton(StandardButton)

        Add and return a button
        """
        stdbutton = None
        if isinstance(button, QAbstractButton):
            if len(rolearg) != 1:
                raise TypeError("Wrong number of arguments for "
                                "addButton(QAbstractButton, role)")
            role = rolearg[0]
        elif isinstance(button, MessageWidget.StandardButton):
            if len(rolearg) != 0:
                raise TypeError("Wrong number of arguments for "
                                "addButton(StandardButton)")
            stdbutton = button
            if button == MessageWidget.Ok:
                role = MessageWidget.AcceptRole
                button = QPushButton("Ok", default=False, autoDefault=False)
            elif button == MessageWidget.Close:
                role = MessageWidget.RejectRole
#                 button = QPushButton(
#                     default=False, autoDefault=False, flat=True,
#                     icon=QIcon(self.style().standardIcon(
#                                QStyle.SP_TitleBarCloseButton)))
                button = SimpleButton(
                    icon=QIcon(self.style().standardIcon(
                               QStyle.SP_TitleBarCloseButton)))
            elif button == MessageWidget.Help:
                role = MessageWidget.HelpRole
                button = QPushButton("Help", default=False, autoDefault=False)
        elif isinstance(button, str):
            if len(rolearg) != 1:
                raise TypeError("Wrong number of arguments for "
                                "addButton(str, ButtonRole)")
            role = rolearg[0]
            button = QPushButton(button, default=False, autoDefault=False)

        if sys.platform == "darwin":
            button.setAttribute(Qt.WA_MacSmallSize)
        self.__buttons.append(MessageWidget._Button(button, role, stdbutton))
        button.clicked.connect(self.__button_clicked)
        self.__relayout()

        return button

    def removeButton(self, button):
        """
        Remove a `button`.

        :type button: QAbstractButton
        """
        slot = [s for s in self.__buttons if s.button is button]
        if slot:
            slot = slot[0]
            self.__buttons.remove(slot)
            self.layout().removeWidget(slot.button)
            slot.button.setParent(None)

    def buttonRole(self, button):
        """
        Return the ButtonRole for button

        :type button: QAbsstractButton
        """
        for slot in self.__buttons:
            if slot.button is button:
                return slot.role
        else:
            return MessageWidget.InvalidRole

    def button(self, standardButton):
        """
        Return the button for the StandardButton.

        :type standardButton: StandardButton
        """
        for slot in self.__buttons:
            if slot.stdbutton == standardButton:
                return slot.button
        else:
            return None

    def __button_clicked(self):
        button = self.sender()
        role = self.buttonRole(button)
        self.clicked.emit(button)

        if role == MessageWidget.AcceptRole:
            self.accepted.emit()
            self.close()
        elif role == MessageWidget.RejectRole:
            self.rejected.emit()
            self.close()
        elif role == MessageWidget.HelpRole:
            self.helpRequested.emit()

    def __relayout(self):
        for slot in self.__buttons:
            self.layout().removeWidget(slot.button)
        order = {
            MessageOverlayWidget.HelpRole: 0,
            MessageOverlayWidget.AcceptRole: 2,
            MessageOverlayWidget.RejectRole: 3,
        }
        orderd = sorted(self.__buttons,
                        key=lambda slot: order.get(slot.role, -1))

        prev = self.__textlabel
        for slot in orderd:
            self.layout().addWidget(slot.button)
            QWidget.setTabOrder(prev, slot.button)


def proxydoc(func):
    return functools.wraps(func, assigned=["__doc__"], updated=[])


class MessageOverlayWidget(OverlayWidget):
    #: Emitted when a button with an Accept role is clicked
    accepted = Signal()
    #: Emitted when a button with a RejectRole is clicked
    rejected = Signal()
    #: Emitted when a button is clicked
    clicked = Signal(QAbstractButton)
    #: Emitted when a button with HelpRole is clicked
    helpRequested = Signal()

    NoButton, Ok, Close, Help = list(MessageWidget.StandardButton)
    InvalidRole, AcceptRole, RejectRole, HelpRole = \
        list(MessageWidget.ButtonRole)

    def __init__(self, parent=None, text="", icon=QIcon(),
                 alignment=Qt.AlignTop, wordWrap=False,
                 standardButtons=NoButton, **kwargs):
        super().__init__(parent, alignment=alignment, **kwargs)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.__msgwidget = MessageWidget(
            parent=self, text=text, icon=icon, wordWrap=wordWrap,
            standardButtons=standardButtons
        )
        self.__msgwidget.accepted.connect(self.accepted)
        self.__msgwidget.rejected.connect(self.rejected)
        self.__msgwidget.clicked.connect(self.clicked)
        self.__msgwidget.helpRequested.connect(self.helpRequested)

        self.__msgwidget.accepted.connect(self.hide)
        self.__msgwidget.rejected.connect(self.hide)
        layout.addWidget(self.__msgwidget)
        self.setLayout(layout)

    @proxydoc(MessageWidget.setText)
    def setText(self, text):
        self.__msgwidget.setText(text)

    @proxydoc(MessageWidget.text)
    def text(self):
        return self.__msgwidget.text()

    @proxydoc(MessageWidget.setIcon)
    def setIcon(self, icon):
        self.__msgwidget.setIcon(icon)

    @proxydoc(MessageWidget.icon)
    def icon(self):
        return self.__msgwidget.icon()

    @proxydoc(MessageWidget.textFormat)
    def textFromat(self):
        return self.__msgwidget.textFormat()

    @proxydoc(MessageWidget.setTextFormat)
    def setTextFormat(self, textFormat):
        self.__msgwidget.setTextFormat(textFormat)

    @proxydoc(MessageWidget.setStandardButtons)
    def setStandardButtons(self, buttons):
        self.__msgwidget.setStandardButtons(buttons)

    @proxydoc(MessageWidget.addButton)
    def addButton(self, *args):
        return self.__msgwidget.addButton(*args)

    @proxydoc(MessageWidget.removeButton)
    def removeButton(self, button):
        self.__msgwidget.removeButton(button)

    @proxydoc(MessageWidget.buttonRole)
    def buttonRole(self, button):
        return self.__msgwidget.buttonRole(button)

    @proxydoc(MessageWidget.button)
    def button(self, standardButton):
        return self.__msgwidget.button(standardButton)


import unittest


class TestOverlay(unittest.TestCase):
    def setUp(self):
        from PyQt4.QtGui import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        self.app = app

    def _exec(self, timeout):
        QTimer.singleShot(timeout, self.app.quit)
        return self.app.exec_()

    def tearDown(self):
        del self.app

    def test_overlay(self):
        container = QWidget()
        overlay = MessageOverlayWidget(parent=container)
        overlay.setWidget(container)
        overlay.setIcon(QStyle.SP_MessageBoxInformation)
        container.show()
        container.raise_()
        self._exec(500)
        self.assertTrue(overlay.isVisible())

        overlay.setText("Hello world! It's so nice here")
        self._exec(500)
        button_ok = overlay.addButton(MessageOverlayWidget.Ok)
        button_close = overlay.addButton(MessageOverlayWidget.Close)
        button_help = overlay.addButton(MessageOverlayWidget.Help)

        self.assertTrue(all([button_ok, button_close, button_help]))
        self.assertIs(overlay.button(MessageOverlayWidget.Ok), button_ok)
        self.assertIs(overlay.button(MessageOverlayWidget.Close), button_close)
        self.assertIs(overlay.button(MessageOverlayWidget.Help), button_help)

        button = overlay.addButton("Click Me!",
                                   MessageOverlayWidget.AcceptRole)
        self.assertIsNot(button, None)
        self.assertTrue(overlay.buttonRole(button),
                        MessageOverlayWidget.AcceptRole)

        self._exec(10000)
