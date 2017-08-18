import sys
import enum
import base64
from itertools import chain
from operator import attrgetter
from xml.sax.saxutils import escape
from collections import OrderedDict
# pylint: disable=unused-import
from typing import (
    NamedTuple, Tuple, List, Dict, Iterable, Union, Optional, Hashable
)

from AnyQt.QtCore import Qt, QSize, QBuffer
from AnyQt.QtGui import (
    QIcon, QPixmap, QPainter, QPalette, QLinearGradient, QBrush, QPen
)
from AnyQt.QtWidgets import (
    QWidget, QLabel, QSizePolicy, QStyle, QHBoxLayout, QMessageBox,
    QMenu, QWidgetAction, QStyleOption, QStylePainter, QApplication
)
from AnyQt.QtCore import pyqtSignal as Signal

__all__ = ["Message", "MessagesWidget"]


def image_data(pm):
    # type: (QPixmap) -> str
    """
    Render the contents of the pixmap as a data URL (RFC-2397)

    Parameters
    ----------
    pm : QPixmap

    Returns
    -------
    datauri : str
    """
    pm = QPixmap(pm)
    device = QBuffer()
    assert device.open(QBuffer.ReadWrite)
    pm.save(device, b'png')
    device.close()
    data = bytes(device.data())
    payload = base64.b64encode(data).decode("ascii")
    return "data:image/png;base64," + payload


class Severity(enum.IntEnum):
    """
    Message severity level.
    """
    Information = QMessageBox.Information
    Warning = QMessageBox.Warning
    Error = QMessageBox.Critical


class Message(
        NamedTuple(
            "Message", [
                ("severity", Severity),
                ("icon", QIcon),
                ("text", str),
                ("informativeText", str),
                ("detailedText", str),
                ("textFormat", Qt.TextFormat)
            ])):
    """
    A stateful message/notification.

    Parameters
    ----------
    severity : Message.Severity
        Severity level (default: Information).
    icon : QIcon
        Associated icon. If empty the `QStyle.standardIcon` will be used based
        on severity.
    text : str
        Short message text.
    informativeText : str
        Extra informative text to append to `text` (space permitting).
    detailedText : str
        Extra detailed text (e.g. exception traceback)
    textFormat : Qt.TextFormat
        If `Qt.RichText` then the contents of `text`, `informativeText` and
        `detailedText` will be rendered as html instead of plain text.

    """
    Severity = Severity
    Warning = Severity.Warning
    Information = Severity.Information
    Error = Severity.Error

    def __new__(cls, severity=Severity.Information, icon=QIcon(), text="",
                informativeText="", detailedText="", textFormat=Qt.PlainText):
        return super().__new__(cls, Severity(severity), icon, text,
                               informativeText, detailedText, textFormat)

    def asHtml(self):
        # type: () -> str
        """
        Render the message as an HTML fragment.
        """
        if self.textFormat == Qt.RichText:
            render = lambda t: t
        else:
            render = lambda t: ('<span style="white-space: pre">{}</span>'
                                .format(escape(t)))

        def iconsrc(message):
            # type: (Message) -> str
            """
            Return an image src url for message icon.
            """
            icon = message_icon(message)
            pm = icon.pixmap(12, 12)
            return image_data(pm)

        parts = [
            ('<div style="white-space:pre" class="message {}">'
             .format(self.severity.name.lower())),
            ('<div class="field-text">'
             '<img src="{iconurl}" width="12" height="12" />{text}</div>'
             .format(iconurl=iconsrc(self), text=render(self.text)))
        ]
        if self.informativeText:
            parts += ['<div class="field-informative-text">{}</div>'
                      .format(render(self.informativeText))]
        if self.detailedText:
            parts += ['<blockquote class="field-detailed-text">{}</blockquote>'
                      .format(render(self.detailedText))]
        parts += ['</div>']
        return "\n".join(parts)

    def isEmpty(self):
        # type: () -> bool
        """
        Is this message instance empty (has no text or icon)
        """
        return (not self.text and self.icon.isNull() and
                not self.informativeText and not self.detailedText)


def standard_pixmap(severity):
    # type: (Severity) -> QStyle.StandardPixmap
    mapping = {
        Severity.Information: QStyle.SP_MessageBoxInformation,
        Severity.Warning: QStyle.SP_MessageBoxWarning,
        Severity.Error: QStyle.SP_MessageBoxCritical,
    }
    return mapping[severity]


def message_icon(message, style=None):
    # type: (Message, Optional[QStyle]) -> QIcon
    """
    Return the resolved icon for the message.

    If `message.icon` is a valid icon then it is used. Otherwise the
    appropriate style icon is used based on the `message.severity`

    Parameters
    ----------
    message : Message
    style : Optional[QStyle]

    Returns
    -------
    icon : QIcon
    """
    if style is None and QApplication.instance() is not None:
        style = QApplication.style()
    if message.icon.isNull():
        icon = style.standardIcon(standard_pixmap(message.severity))
    else:
        icon = message.icon
    return icon


def categorize(messages):
    # type: (List[Message]) -> Tuple[Optional[Message], List[Message], List[Message], List[Message]]
    """
    Categorize the messages by severity picking the message leader if
    possible.

    The leader is a message with the highest severity iff it is the only
    representative of that severity.

    Parameters
    ----------
    messages : List[Messages]

    Returns
    -------
    r : Tuple[Optional[Message], List[Message], List[Message], List[Message]]
    """
    errors = [m for m in messages if m.severity == Severity.Error]
    warnings = [m for m in messages if m.severity == Severity.Warning]
    info = [m for m in messages if m.severity == Severity.Information]
    lead = None
    if len(errors) == 1:
        lead = errors.pop(-1)
    elif not errors and len(warnings) == 1:
        lead = warnings.pop(-1)
    elif not errors and not warnings and len(info) == 1:
        lead = info.pop(-1)
    return lead, errors, warnings, info


# pylint: disable=too-many-branches
def summarize(messages):
    # type: (List[Message]) -> Message
    """
    Summarize a list of messages into a single message instance

    Parameters
    ----------
    messages: List[Message]

    Returns
    -------
    message: Message
    """
    if not messages:
        return Message()

    if len(messages) == 1:
        return messages[0]

    lead, errors, warnings, info = categorize(messages)
    severity = Severity.Information
    icon = QIcon()
    leading_text = ""
    text_parts = []
    if lead is not None:
        severity = lead.severity
        icon = lead.icon
        leading_text = lead.text
    elif errors:
        severity = Severity.Error
    elif warnings:
        severity = Severity.Warning

    def format_plural(fstr, items, *args, **kwargs):
        return fstr.format(len(items), *args,
                           s="s" if len(items) != 1 else "",
                           **kwargs)
    if errors:
        text_parts.append(format_plural("{} error{s}", errors))
    if warnings:
        text_parts.append(format_plural("{} warning{s}", warnings))
    if info:
        if not (errors and warnings and lead):
            text_parts.append(format_plural("{} message{s}", info))
        else:
            text_parts.append(format_plural("{} other", info))

    if leading_text:
        text = leading_text
        if text_parts:
            text = text + " (" + ", ".join(text_parts) + ")"
    else:
        text = ", ".join(text_parts)
    detailed = "<hr/>".join(m.asHtml()
                            for m in chain([lead], errors, warnings, info)
                            if m is not None and not m.isEmpty())
    return Message(severity, icon, text, detailedText=detailed,
                   textFormat=Qt.RichText)


class MessagesWidget(QWidget):
    """
    An iconified multiple message display area.

    `MessagesWidget` displays a short message along with an icon. If there
    are multiple messages they are summarized. The user can click on the
    widget to display the full message text in a popup view.
    """
    #: Signal emitted when an embedded html link is clicked
    #: (if `openExternalLinks` is `False`).
    linkActivated = Signal(str)

    #: Signal emitted when an embedded html link is hovered.
    linkHovered = Signal(str)

    Severity = Severity
    #: General informative message.
    Information = Severity.Information
    #: A warning message severity.
    Warning = Severity.Warning
    #: An error message severity.
    Error = Severity.Error

    Message = Message

    def __init__(self, parent=None, openExternalLinks=False, **kwargs):
        kwargs.setdefault(
            "sizePolicy",
            QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        )
        super().__init__(parent, **kwargs)
        self.__openExternalLinks = openExternalLinks  # type: bool
        self.__messages = OrderedDict()  # type: Dict[Hashable, Message]
        #: The full (joined all messages text - rendered as html), displayed
        #: in a tooltip.
        self.__fulltext = ""
        #: The full text displayed in a popup. Is empty if the message is
        #: short
        self.__popuptext = ""
        #: Leading icon
        self.__iconwidget = IconWidget(
            sizePolicy=QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        )
        #: Inline  message text
        self.__textlabel = QLabel(
            wordWrap=False,
            textInteractionFlags=Qt.LinksAccessibleByMouse,
            openExternalLinks=self.__openExternalLinks,
            sizePolicy=QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        )
        #: Indicator that extended contents are accessible with a click on the
        #: widget.
        self.__popupicon = QLabel(
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum),
            text="\N{VERTICAL ELLIPSIS}",
            visible=False,
        )
        self.__textlabel.linkActivated.connect(self.linkActivated)
        self.__textlabel.linkHovered.connect(self.linkHovered)
        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(2, 1, 2, 1)
        self.layout().setSpacing(0)
        self.layout().addWidget(self.__iconwidget)
        self.layout().addSpacing(4)
        self.layout().addWidget(self.__textlabel)
        self.layout().addWidget(self.__popupicon)
        self.__textlabel.setAttribute(Qt.WA_MacSmallSize)

    def sizeHint(self):
        sh = super().sizeHint()
        h = self.style().pixelMetric(QStyle.PM_SmallIconSize)
        return sh.expandedTo(QSize(0, h + 2))

    def openExternalLinks(self):
        # type: () -> bool
        """
        If True then linkActivated signal will be emitted when the user
        clicks on an html link in a message, otherwise links are opened
        using `QDesktopServices.openUrl`
        """
        return self.__openExternalLinks

    def setOpenExternalLinks(self, state):
        # type: (bool) -> None
        """
        """
        # TODO: update popup if open
        self.__openExternalLinks = state
        self.__textlabel.setOpenExternalLinks(state)

    def setMessage(self, message_id, message):
        # type: (Hashable, Message) -> None
        """
        Add a `message` for `message_id` to the current display.

        Note
        ----
        Set an empty `Message` instance to clear the message display but
        retain the relative ordering in the display should a message for
        `message_id` reactivate.
        """
        self.__messages[message_id] = message
        self.__update()

    def removeMessage(self, message_id):
        # type: (Hashable) -> None
        """
        Remove message for `message_id` from the display.

        Note
        ----
        Setting an empty `Message` instance will also clear the display,
        however the relative ordering of the messages will be retained,
        should the `message_id` 'reactivate'.
        """
        del self.__messages[message_id]
        self.__update()

    def setMessages(self, messages):
        # type: (Union[Iterable[Tuple[Hashable, Message]], Dict[Hashable, Message]]) -> None
        """
        Set multiple messages in a single call.
        """
        messages = OrderedDict(messages)
        self.__messages.update(messages)
        self.__update()

    def clear(self):
        # type: () -> None
        """
        Clear all messages.
        """
        self.__messages.clear()
        self.__update()

    def messages(self):
        # type: () -> List[Message]
        return list(self.__messages.values())

    def summarize(self):
        # type: () -> Message
        """
        Summarize all the messages into a single message.
        """
        messages = [m for m in self.__messages.values() if not m.isEmpty()]
        if messages:
            return summarize(messages)
        else:
            return Message()

    def __update(self):
        """
        Update the current display state.
        """
        self.ensurePolished()
        summary = self.summarize()
        icon = message_icon(summary)
        self.__iconwidget.setIcon(icon)
        self.__iconwidget.setVisible(not (summary.isEmpty() or icon.isNull()))
        self.__textlabel.setTextFormat(summary.textFormat)
        self.__textlabel.setText(summary.text)
        messages = [m for m in self.__messages.values() if not m.isEmpty()]
        if messages:
            messages = sorted(messages, key=attrgetter("severity"),
                              reverse=True)
            fulltext = "<hr/>".join(m.asHtml() for m in messages)
        else:
            fulltext = ""
        self.__fulltext = fulltext
        self.setToolTip(fulltext)

        def is_short(m):
            return not (m.informativeText or m.detailedText)

        if not messages or len(messages) == 1 and is_short(messages[0]):
            self.__popuptext = ""
        else:
            self.__popuptext = fulltext
        self.__popupicon.setVisible(bool(self.__popuptext))
        self.layout().activate()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.__popuptext:
                popup = QMenu(self)
                label = QLabel(
                    self, textInteractionFlags=Qt.TextBrowserInteraction,
                    openExternalLinks=self.__openExternalLinks,
                    text=self.__popuptext
                )
                label.linkActivated.connect(self.linkActivated)
                label.linkHovered.connect(self.linkHovered)
                action = QWidgetAction(popup)
                action.setDefaultWidget(label)
                popup.addAction(action)
                popup.popup(event.globalPos(), action)
                event.accept()
            return
        else:
            super().mousePressEvent(event)

    def enterEvent(self, event):
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.update()

    def changeEvent(self, event):
        super().changeEvent(event)
        self.update()

    def paintEvent(self, event):
        opt = QStyleOption()
        opt.initFrom(self)
        if not self.__popupicon.isVisible():
            return

        if not (opt.state & QStyle.State_MouseOver or
                opt.state & QStyle.State_HasFocus):
            return

        palette = opt.palette  # type: QPalette
        if opt.state & QStyle.State_HasFocus:
            pen = QPen(palette.color(QPalette.Highlight))
        else:
            pen = QPen(palette.color(QPalette.Dark))

        if self.__fulltext and \
                opt.state & QStyle.State_MouseOver and \
                opt.state & QStyle.State_Active:
            g = QLinearGradient()
            g.setCoordinateMode(QLinearGradient.ObjectBoundingMode)
            base = palette.color(QPalette.Window)
            base.setAlpha(90)
            g.setColorAt(0, base.lighter(200))
            g.setColorAt(0.6, base)
            g.setColorAt(1.0, base.lighter(200))
            brush = QBrush(g)
        else:
            brush = QBrush(Qt.NoBrush)
        p = QPainter(self)
        p.setBrush(brush)
        p.setPen(pen)
        p.drawRect(opt.rect.adjusted(0, 0, -1, -1))


class IconWidget(QWidget):
    """
    A widget displaying an `QIcon`
    """
    def __init__(self, parent=None, icon=QIcon(), iconSize=QSize(), **kwargs):
        sizePolicy = kwargs.pop("sizePolicy", QSizePolicy(QSizePolicy.Fixed,
                                                          QSizePolicy.Fixed))
        super().__init__(parent, **kwargs)
        self.__icon = QIcon(icon)
        self.__iconSize = QSize(iconSize)
        self.setSizePolicy(sizePolicy)

    def setIcon(self, icon):
        # type: (QIcon) -> None
        if self.__icon != icon:
            self.__icon = QIcon(icon)
            self.updateGeometry()
            self.update()

    def icon(self):
        # type: () -> QIcon
        return QIcon(self.__icon)

    def iconSize(self):
        # type: () -> QSize
        if not self.__iconSize.isValid():
            size = self.style().pixelMetric(QStyle.PM_ButtonIconSize)
            return QSize(size, size)
        else:
            return QSize(self.__iconSize)

    def setIconSize(self, iconSize):
        # type: (QSize) -> None
        if self.__iconSize != iconSize:
            self.__iconSize = QSize(iconSize)
            self.updateGeometry()
            self.update()

    def sizeHint(self):
        sh = self.iconSize()
        m = self.contentsMargins()
        return QSize(sh.width() + m.left() + m.right(),
                     sh.height() + m.top() + m.bottom())

    def paintEvent(self, event):
        painter = QStylePainter(self)
        opt = QStyleOption()
        opt.initFrom(self)
        painter.drawPrimitive(QStyle.PE_Widget, opt)
        if not self.__icon.isNull():
            rect = self.contentsRect()
            if opt.state & QStyle.State_Active:
                mode = QIcon.Active
            else:
                mode = QIcon.Disabled
            self.__icon.paint(painter, rect, Qt.AlignCenter, mode, QIcon.Off)
        painter.end()


def main(argv=None):  # pragma: no cover
    from AnyQt.QtWidgets import QVBoxLayout, QCheckBox, QStatusBar
    app = QApplication(list(argv) if argv else [])
    l1 = QVBoxLayout()
    l1.setContentsMargins(0, 0, 0, 0)
    blayout = QVBoxLayout()
    l1.addLayout(blayout)
    sb = QStatusBar()

    w = QWidget()
    w.setLayout(l1)
    messages = [
        Message(Severity.Error, text="Encountered a HCF",
                detailedText="<em>AAA! It burns.</em>",
                textFormat=Qt.RichText),
        Message(Severity.Warning,
                text="ACHTUNG!",
                detailedText=(
                    "<div style=\"color: red\">DAS KOMPUTERMASCHINE IST "
                    "NICHT FÜR DER GEFINGERPOKEN</div>"
                ),
                textFormat=Qt.RichText),
        Message(Severity.Information,
                text="The rain in spain falls mostly on the plain",
                informativeText=(
                    "<a href=\"https://www.google.si/search?q="
                    "Average+Yearly+Precipitation+in+Spain\">Link</a>"
                ),
                textFormat=Qt.RichText),
        Message(Severity.Error,
                text="I did not do this!",
                informativeText="The computer made suggestions...",
                detailedText="... and the default options was yes."),
        Message(),
    ]
    mw = MessagesWidget(openExternalLinks=True)
    for i, m in enumerate(messages):
        cb = QCheckBox(m.text)

        def toogled(state, i=i, m=m):
            if state:
                mw.setMessage(i, m)
            else:
                mw.removeMessage(i)
        cb.toggled[bool].connect(toogled)
        blayout.addWidget(cb)

    sb.addWidget(mw)
    w.layout().addWidget(sb, 0)
    w.show()
    return app.exec_()

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
