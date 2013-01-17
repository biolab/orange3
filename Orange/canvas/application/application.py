"""
Orange Canvas Application

"""

from PyQt4.QtGui import QApplication

from PyQt4.QtCore import Qt


class CanvasApplication(QApplication):
    def __init__(self, argv):
        QApplication.__init__(self, argv)
        self.setAttribute(Qt.AA_DontShowIconsInMenus, True)
