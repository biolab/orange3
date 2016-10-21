from collections import Callable

from AnyQt.QtWidgets import QTextBrowser
from AnyQt.QtGui import QStatusTipEvent, QWhatsThisClickedEvent
from AnyQt.QtCore import QObject, QCoreApplication, QEvent, QTimer, QUrl
from AnyQt.QtCore import pyqtSignal as Signal


class QuickHelp(QTextBrowser):

    #: Emitted when the shown text changes.
    textChanged = Signal()

    def __init__(self, *args, **kwargs):
        QTextBrowser.__init__(self, *args, **kwargs)

        self.setOpenExternalLinks(False)
        self.setOpenLinks(False)

        self.__text = ""
        self.__permanentText = ""

        self.__timer = QTimer(self, timeout=self.__on_timeout,
                              singleShot=True)
        self.anchorClicked.connect(self.__on_anchorClicked)

    def showHelp(self, text, timeout=0):
        """
        Show help for `timeout` milliseconds. if timeout is 0 then
        show the text until it is cleared with clearHelp or showHelp is
        called with an empty string.

        """
        if self.__text != text:
            self.__text = str(text)
            self.__update()
            self.textChanged.emit()

        if timeout > 0:
            self.__timer.start(timeout)

    def clearHelp(self):
        """
        Clear help text previously set with `showHelp`.
        """
        self.__timer.stop()
        self.showHelp("")

    def showPermanentHelp(self, text):
        """
        Set permanent help text. The text may be temporarily overridden
        by showHelp but will be shown again when that is cleared.

        """
        if self.__permanentText != text:
            self.__permanentText = text
            self.__update()
            self.textChanged.emit()

    def currentText(self):
        """
        Return the current shown text.
        """
        return self.__text or self.__permanentText

    def __update(self):
        if self.__text:
            self.setHtml(self.__text)
        else:
            self.setHtml(self.__permanentText)

    def __on_timeout(self):
        if self.__text:
            self.__text = ""
            self.__update()
            self.textChanged.emit()

    def __on_anchorClicked(self, anchor):
        ev = QuickHelpDetailRequestEvent(anchor.toString(), anchor)
        QCoreApplication.postEvent(self, ev)


class QuickHelpTipEvent(QStatusTipEvent):
    Temporary, Normal, Permanent = range(1, 4)

    def __init__(self, tip, html=None, priority=Normal, timeout=None):
        QStatusTipEvent.__init__(self, tip)
        self.__html = html or ""
        self.__priority = priority
        self.__timeout = timeout

    def html(self):
        return self.__html

    def priority(self):
        return self.__priority

    def timeout(self):
        return self.__timeout


class QuickHelpDetailRequestEvent(QWhatsThisClickedEvent):
    def __init__(self, href, url):
        QWhatsThisClickedEvent.__init__(self, href)
        self.__url = QUrl(url)

    def url(self):
        return QUrl(self.__url)


class StatusTipPromoter(QObject):
    """
    Promotes `QStatusTipEvent` to `QuickHelpTipEvent` using ``whatsThis``
    property of the object.

    """
    def eventFilter(self, obj, event):
        if event.type() == QEvent.StatusTip and \
                not isinstance(event, QuickHelpTipEvent) and \
                hasattr(obj, "whatsThis") and \
                isinstance(obj.whatsThis, Callable):
            tip = event.tip()

            try:
                text = obj.whatsThis()
            except Exception:
                text = None

            if text:
                ev = QuickHelpTipEvent(tip, text if tip else "")
                return QCoreApplication.sendEvent(obj, ev)

        return QObject.eventFilter(self, obj, event)
