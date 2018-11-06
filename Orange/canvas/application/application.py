"""
Orange Canvas Application

"""

from AnyQt.QtWidgets import QApplication

from AnyQt.QtCore import Qt, QUrl, QEvent, pyqtSignal as Signal


class CanvasApplication(QApplication):
    fileOpenRequest = Signal(QUrl)

    def __init__(self, argv):
        if hasattr(Qt, "AA_EnableHighDpiScaling"):
            # Turn on HighDPI support when available
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QApplication.__init__(self, argv)
        self.setAttribute(Qt.AA_DontShowIconsInMenus, True)

    def event(self, event):
        if event.type() == QEvent.FileOpen:
            self.fileOpenRequest.emit(event.url())

        return QApplication.event(self, event)
