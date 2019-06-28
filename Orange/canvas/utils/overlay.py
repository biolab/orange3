import enum
import functools
import operator
import sys
from collections import namedtuple

from AnyQt.QtCore import Signal, Qt, QSize
from AnyQt.QtGui import QIcon, QPixmap, QPainter
from AnyQt.QtWidgets import QAbstractButton, QHBoxLayout, QPushButton, QStyle, QWidget, \
    QVBoxLayout, QLabel, QSizePolicy, QStyleOption

from orangecanvas.gui.stackedwidget import StackLayout
from Orange.widgets.utils.buttons import SimpleButton
from Orange.widgets.utils.overlay import OverlayWidget


class NotificationMessageWidget(QWidget):
    #: Emitted when a button with the AcceptRole is clicked
    accepted = Signal()
    #: Emitted when a button with the RejectRole is clicked
    rejected = Signal()
    #: Emitted when a button is clicked
    clicked = Signal(QAbstractButton)

    class StandardButton(enum.IntEnum):
        NoButton, Ok, Close = 0x0, 0x1, 0x2

    NoButton, Ok, Close = list(StandardButton)

    class ButtonRole(enum.IntEnum):
        InvalidRole, AcceptRole, RejectRole, DismissRole = 0, 1, 2, 3

    InvalidRole, AcceptRole, RejectRole, DismissRole = list(ButtonRole)

    _Button = namedtuple("_Button", ["button", "role", "stdbutton"])

    def __init__(self, parent=None, icon=QIcon(), title="", text="", wordWrap=False,
                 textFormat=Qt.PlainText, standardButtons=NoButton, acceptLabel="Ok",
                 rejectLabel="No", **kwargs):
        super().__init__(parent, **kwargs)
        self._title = title
        self._text = text
        self._icon = QIcon()
        self._wordWrap = wordWrap
        self._standardButtons = NotificationMessageWidget.NoButton
        self._buttons = []
        self._acceptLabel = acceptLabel
        self._rejectLabel = rejectLabel

        self._iconlabel = QLabel(objectName="icon-label")
        self._titlelabel = QLabel(objectName="title-label", text=title,
                                  wordWrap=wordWrap, textFormat=textFormat)
        self._textlabel = QLabel(objectName="text-label", text=text,
                                 wordWrap=wordWrap, textFormat=textFormat)
        self._textlabel.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self._textlabel.setOpenExternalLinks(True)

        if sys.platform == "darwin":
            self._titlelabel.setAttribute(Qt.WA_MacSmallSize)
            self._textlabel.setAttribute(Qt.WA_MacSmallSize)

        layout = QHBoxLayout()
        self._iconlabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(self._iconlabel)
        layout.setAlignment(self._iconlabel, Qt.AlignTop)

        message_layout = QVBoxLayout()
        self._titlelabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if sys.platform == "darwin":
            self._titlelabel.setContentsMargins(0, 1, 0, 0)
        else:
            self._titlelabel.setContentsMargins(0, 0, 0, 0)
        message_layout.addWidget(self._titlelabel)
        self._textlabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        message_layout.addWidget(self._textlabel)

        self.button_layout = QHBoxLayout()
        self.button_layout.setAlignment(Qt.AlignLeft)
        message_layout.addLayout(self.button_layout)

        layout.addLayout(message_layout)
        layout.setSpacing(7)
        self.setLayout(layout)
        self.setIcon(icon)
        self.setStandardButtons(standardButtons)

    def setText(self, text):
        """
        Set the current message text.

        :type message: str
        """
        if self._text != text:
            self._text = text
            self._textlabel.setText(text)

    def text(self):
        """
        Return the current message text.

        :rtype: str
        """
        return self._text

    def setTitle(self, title):
        """
        Set the current title text.

        :type title: str
        """
        if self._title != title:
            self._title = title
            self._titleLabel.setText(title)

    def title(self):
        """
        Return the current title text.

        :rtype: str
        """
        return self._title

    def setIcon(self, icon):
        """
        Set the message icon.

        :type icon: QIcon | QPixmap | QString | QStyle.StandardPixmap
        """
        if isinstance(icon, QStyle.StandardPixmap):
            icon = self.style().standardIcon(icon)
        else:
            icon = QIcon(icon)

        if self._icon != icon:
            self._icon = QIcon(icon)
            if not self._icon.isNull():
                size = self.style().pixelMetric(
                    QStyle.PM_SmallIconSize, None, self)
                pm = self._icon.pixmap(QSize(size, size))
            else:
                pm = QPixmap()

            self._iconlabel.setPixmap(pm)
            self._iconlabel.setVisible(not pm.isNull())

    def icon(self):
        """
        Return the current icon.

        :rtype: QIcon
        """
        return QIcon(self._icon)

    def setWordWrap(self, wordWrap):
        """
        Set the message text wrap property

        :type wordWrap: bool
        """
        if self._wordWrap != wordWrap:
            self._wordWrap = wordWrap
            self._textlabel.setWordWrap(wordWrap)

    def wordWrap(self):
        """
        Return the message text wrap property.

        :rtype: bool
        """
        return self._wordWrap

    def setTextFormat(self, textFormat):
        """
        Set message text format

        :type textFormat: Qt.TextFormat
        """
        self._textlabel.setTextFormat(textFormat)

    def textFormat(self):
        """
        Return the message text format.

        :rtype: Qt.TextFormat
        """
        return self._textlabel.textFormat()

    def setAcceptLabel(self, label):
        """
        Set the accept button label.
        :type label: str
        """
        self._acceptLabel = label

    def acceptLabel(self):
        """
        Return the accept button label.
        :rtype str
        """
        return self._acceptLabel

    def setRejectLabel(self, label):
        """
        Set the reject button label.
        :type label: str
        """
        self._rejectLabel = label

    def rejectLabel(self):
        """
        Return the reject button label.
        :rtype str
        """
        return self._rejectLabel

    def setStandardButtons(self, buttons):
        for button in NotificationMessageWidget.StandardButton:
            existing = self.button(button)
            if button & buttons and existing is None:
                self.addButton(button)
            elif existing is not None:
                self.removeButton(existing)

    def standardButtons(self):
        return functools.reduce(
            operator.ior,
            (slot.stdbutton for slot in self._buttons
             if slot.stdbutton is not None),
            NotificationMessageWidget.NoButton)

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
        elif isinstance(button, NotificationMessageWidget.StandardButton):
            if rolearg:
                raise TypeError("Wrong number of arguments for "
                                "addButton(StandardButton)")
            stdbutton = button
            if button == NotificationMessageWidget.Ok:
                role = NotificationMessageWidget.AcceptRole
                button = QPushButton(self._acceptLabel, default=False, autoDefault=False)
            elif button == NotificationMessageWidget.Close:
                role = NotificationMessageWidget.RejectRole
                button = QPushButton(self._rejectLabel, default=False, autoDefault=False)
        elif isinstance(button, str):
            if len(rolearg) != 1:
                raise TypeError("Wrong number of arguments for "
                                "addButton(str, ButtonRole)")
            role = rolearg[0]
            button = QPushButton(button, default=False, autoDefault=False)

        if sys.platform == "darwin":
            button.setAttribute(Qt.WA_MacSmallSize)

        self._buttons.append(NotificationMessageWidget._Button(button, role, stdbutton))
        button.clicked.connect(self._button_clicked)
        self._relayout()

        return button

    def _relayout(self):
        for slot in self._buttons:
            self.button_layout.removeWidget(slot.button)
        order = {
            NotificationWidget.AcceptRole: 0,
            NotificationWidget.RejectRole: 1,
        }
        ordered = sorted([b for b in self._buttons if
                          self.buttonRole(b.button) != NotificationMessageWidget.DismissRole],
                         key=lambda slot: order.get(slot.role, -1))

        prev = self._textlabel
        for slot in ordered:
            self.button_layout.addWidget(slot.button)
            QWidget.setTabOrder(prev, slot.button)

    def removeButton(self, button):
        """
        Remove a `button`.

        :type button: QAbstractButton
        """
        slot = [s for s in self._buttons if s.button is button]
        if slot:
            slot = slot[0]
            self._buttons.remove(slot)
            self.layout().removeWidget(slot.button)
            slot.button.setParent(None)

    def buttonRole(self, button):
        """
        Return the ButtonRole for button

        :type button: QAbstractButton
        """
        for slot in self._buttons:
            if slot.button is button:
                return slot.role
        return NotificationMessageWidget.InvalidRole

    def button(self, standardButton):
        """
        Return the button for the StandardButton.

        :type standardButton: StandardButton
        """
        for slot in self._buttons:
            if slot.stdbutton == standardButton:
                return slot.button
        return None

    def _button_clicked(self):
        button = self.sender()
        role = self.buttonRole(button)
        self.clicked.emit(button)

        if role == NotificationMessageWidget.AcceptRole:
            self.accepted.emit()
            self.close()
        elif role == NotificationMessageWidget.RejectRole:
            self.rejected.emit()
            self.close()


def proxydoc(func):
    return functools.wraps(func, assigned=["__doc__"], updated=[])


class NotificationWidget(QWidget):
    #: Emitted when a button with an Accept role is clicked
    accepted = Signal()
    #: Emitted when a button with a Reject role is clicked
    rejected = Signal()
    #: Emitted when a button with a Dismiss role is clicked
    dismissed = Signal()
    #: Emitted when a button is clicked
    clicked = Signal(QAbstractButton)

    NoButton, Ok, Close = list(NotificationMessageWidget.StandardButton)
    InvalidRole, AcceptRole, RejectRole, DismissRole = \
        list(NotificationMessageWidget.ButtonRole)

    def __init__(self, parent=None, title="", text="", textFormat=Qt.AutoText, icon=QIcon(),
                 wordWrap=True, standardButtons=NoButton, acceptLabel="Ok", rejectLabel="No",
                 **kwargs):
        super().__init__(parent, **kwargs)
        self._margin = 10  # used in stylesheet and for dismiss button

        layout = QHBoxLayout()
        if sys.platform == "darwin":
            layout.setContentsMargins(6, 6, 6, 6)
        else:
            layout.setContentsMargins(9, 9, 9, 9)

        self.setStyleSheet("""
                            NotificationWidget {
                                margin: """ + str(self._margin) + """px;
                                background: #626262;
                                border: 1px solid #999999;
                                border-radius: 8px;
                            }
                            NotificationWidget QLabel#text-label {
                                color: white;
                            }
                            NotificationWidget QLabel#title-label {
                                color: white;
                                font-weight: bold;
                            }""")

        self._msgwidget = NotificationMessageWidget(
            parent=self, title=title, text=text, textFormat=textFormat, icon=icon,
            wordWrap=wordWrap, standardButtons=standardButtons, acceptLabel=acceptLabel,
            rejectLabel=rejectLabel
        )
        self._msgwidget.accepted.connect(self.accepted)
        self._msgwidget.rejected.connect(self.rejected)
        self._msgwidget.clicked.connect(self.clicked)

        self._dismiss_button = SimpleButton(parent=self,
                                            icon=QIcon(self.style().standardIcon(
                                                QStyle.SP_TitleBarCloseButton)))
        self._dismiss_button.setFixedSize(18, 18)
        self._dismiss_button.clicked.connect(self.dismissed)

        def dismiss_handler():
            self.clicked.emit(self._dismiss_button)
        self._dismiss_button.clicked.connect(dismiss_handler)

        layout.addWidget(self._msgwidget)
        self.setLayout(layout)

        self.setFixedWidth(400)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if sys.platform == "darwin":
            corner_margin = 6
        else:
            corner_margin = 7
        x = self.width() - self._dismiss_button.width() - self._margin - corner_margin
        y = self._margin + corner_margin
        self._dismiss_button.move(x, y)

    def clone(self):
        cloned = NotificationWidget(parent=self.parent(),
                                    title=self.title(),
                                    text=self.text(),
                                    textFormat=self._msgwidget.textFormat(),
                                    icon=self.icon(),
                                    standardButtons=self._msgwidget.standardButtons(),
                                    acceptLabel=self._msgwidget.acceptLabel(),
                                    rejectLabel=self._msgwidget.rejectLabel())
        cloned.accepted.connect(self.accepted)
        cloned.rejected.connect(self.rejected)
        cloned.dismissed.connect(self.dismissed)

        # each canvas displays a clone of the original notification,
        # therefore the cloned buttons' events are connected to the original's
        # pylint: disable=protected-access
        button_map = dict(zip(
            [b.button for b in cloned._msgwidget._buttons] + [cloned._dismiss_button],
            [b.button for b in self._msgwidget._buttons] + [self._dismiss_button]))
        cloned.clicked.connect(lambda b:
                               self.clicked.emit(button_map[b]))

        return cloned

    def paintEvent(self, event):
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, painter, self)

    @proxydoc(NotificationMessageWidget.setText)
    def setText(self, text):
        self._msgwidget.setText(text)

    @proxydoc(NotificationMessageWidget.text)
    def text(self):
        return self._msgwidget.text()

    @proxydoc(NotificationMessageWidget.setTitle)
    def setTitle(self, title):
        self._msgwidget.setTitle(title)

    @proxydoc(NotificationMessageWidget.title)
    def title(self):
        return self._msgwidget.title()

    @proxydoc(NotificationMessageWidget.setIcon)
    def setIcon(self, icon):
        self._msgwidget.setIcon(icon)

    @proxydoc(NotificationMessageWidget.icon)
    def icon(self):
        return self._msgwidget.icon()

    @proxydoc(NotificationMessageWidget.textFormat)
    def textFormat(self):
        return self._msgwidget.textFormat()

    @proxydoc(NotificationMessageWidget.setTextFormat)
    def setTextFormat(self, textFormat):
        self._msgwidget.setTextFormat(textFormat)

    @proxydoc(NotificationMessageWidget.setStandardButtons)
    def setStandardButtons(self, buttons):
        self._msgwidget.setStandardButtons(buttons)

    @proxydoc(NotificationMessageWidget.addButton)
    def addButton(self, *args):
        return self._msgwidget.addButton(*args)

    @proxydoc(NotificationMessageWidget.removeButton)
    def removeButton(self, button):
        self._msgwidget.removeButton(button)

    @proxydoc(NotificationMessageWidget.buttonRole)
    def buttonRole(self, button):
        if button is self._dismiss_button:
            return NotificationWidget.DismissRole
        return self._msgwidget.buttonRole(button)

    @proxydoc(NotificationMessageWidget.button)
    def button(self, standardButton):
        return self._msgwidget.button(standardButton)


class NotificationOverlay(OverlayWidget):
    # each canvas instance has its own overlay instance
    overlayInstances = []

    # list of queued notifications, overlay instances retain a clone of each notification
    notifQueue = []

    def __init__(self, parent=None, alignment=Qt.AlignRight | Qt.AlignBottom, **kwargs):
        """
        Registers a new canvas instance to simultaneously display notifications in.
        The parent parameter should be the canvas' scheme widget.
        """
        super().__init__(parent, alignment=alignment, **kwargs)

        layout = StackLayout()
        self.setLayout(layout)

        self._widgets = []

        for notif in self.notifQueue:
            cloned = notif.clone()
            cloned.setParent(self)
            cloned.clicked.connect(NotificationOverlay.nextNotification)
            self.addWidget(cloned)

        self.overlayInstances.append(self)

        self.setWidget(parent)

    @staticmethod
    def registerNotification(notif):
        """
        Queues notification in all canvas instances (shows it if no other notifications present).
        """
        notif.hide()
        NotificationOverlay.notifQueue.append(notif)

        overlays = NotificationOverlay.overlayInstances
        for overlay in overlays:
            # each canvas requires its own instance of each notification
            cloned = notif.clone()
            cloned.setParent(overlay)
            cloned.clicked.connect(NotificationOverlay.nextNotification)
            overlay.addWidget(cloned)

    @staticmethod
    def nextNotification():
        NotificationOverlay.notifQueue.pop(0)

        overlays = NotificationOverlay.overlayInstances
        for overlay in overlays:
            overlay.nextWidget()

    def addWidget(self, widget):
        """
        Append the widget to the stack.
        """
        self._widgets.append(widget)
        self.layout().addWidget(widget)

    def nextWidget(self):
        """
        Removes first widget from the stack.

        .. note:: The widget is hidden but is not deleted.

        """
        widget = self._widgets[0]
        self._widgets.pop(0)
        self.layout().removeWidget(widget)

    def currentWidget(self):
        """
        Return the currently displayed widget.
        """
        if not self._widgets:
            return None
        return self._widgets[0]

    def close(self):
        self.overlayInstances.remove(self)
        super().close()
