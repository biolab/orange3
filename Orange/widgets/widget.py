#
# OWWidget.py
# Orange Widget
# A General Orange Widget, from which all the Orange Widgets are derived
#
import os, sys

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from Orange.widgets import basewidget
from Orange.widgets import gui as OWGUI

class OWWidget(basewidget.OWBaseWidget):
    def __init__(self, parent=None, signalManager=None, title="Orange Widget", wantGraph=False, wantStatusBar=False, savePosition=True, wantMainArea=1, noReport=False, showSaveGraph=1, resizingEnabled=1, wantStateInfoWidget=None, **args):
        """
        Initialization
        Parameters:
            title - The title of the\ widget, including a "&" (for shortcut in about box)
            wantGraph - displays a save graph button or not
        """

        super().__init__(parent, signalManager, title, savePosition=savePosition, resizingEnabled=resizingEnabled, **args)

        self.setLayout(QVBoxLayout())
        self.layout().setMargin(2)

        self.topWidgetPart = OWGUI.widgetBox(self, orientation="horizontal", margin=0)
        self.leftWidgetPart = OWGUI.widgetBox(self.topWidgetPart, orientation="vertical", margin=0)
        if wantMainArea:
            self.leftWidgetPart.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.MinimumExpanding))
            self.leftWidgetPart.updateGeometry()
            self.mainArea = OWGUI.widgetBox(self.topWidgetPart, orientation="vertical", sizePolicy=QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding), margin=0)
            self.mainArea.layout().setMargin(4)
            self.mainArea.updateGeometry()

        self.controlArea = OWGUI.widgetBox(self.leftWidgetPart, orientation="vertical", margin=4)# if wantMainArea else 1)

        self.space = self.controlArea

        self.buttonBackground = OWGUI.widgetBox(self.leftWidgetPart, orientation="horizontal", margin=4)# if wantMainArea else 1)
        self.buttonBackground.hide()

        if wantGraph and showSaveGraph:
            self.buttonBackground.show()
            self.graphButton = OWGUI.button(self.buttonBackground, self, "&Save Graph")
            self.graphButton.setAutoDefault(0)

        if wantStateInfoWidget is None:
            wantStateInfoWidget = self._owShowStatus

        if wantStateInfoWidget:
            # Widget for error, warnings, info.
            self.widgetStateInfoBox = OWGUI.widgetBox(self.leftWidgetPart, "Widget state")
            self.widgetStateInfo = OWGUI.widgetLabel(self.widgetStateInfoBox, "\n")
            self.widgetStateInfo.setWordWrap(True)
            self.widgetStateInfo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            self.widgetStateInfo.setFixedHeight(self.widgetStateInfo.height())
            self.widgetStateInfoBox.hide()

            self.connect(self, SIGNAL("widgetStateChanged(QString, int, QString)"), self.updateWidgetStateInfo)


        self.__reportData = None

        if wantStatusBar:
            self.widgetStatusArea = QFrame(self)
            self.statusBarIconArea = QFrame(self)
            self.widgetStatusBar = QStatusBar(self)

            self.layout().addWidget(self.widgetStatusArea)

            self.widgetStatusArea.setLayout(QHBoxLayout(self.widgetStatusArea))
            self.widgetStatusArea.layout().addWidget(self.statusBarIconArea)
            self.widgetStatusArea.layout().addWidget(self.widgetStatusBar)
            self.widgetStatusArea.layout().setMargin(0)
            self.widgetStatusArea.setFrameShape(QFrame.StyledPanel)

            self.statusBarIconArea.setLayout(QHBoxLayout())
            self.widgetStatusBar.setSizeGripEnabled(0)

            self.statusBarIconArea.hide()

            warning_icon = os.path.join(self.widgetDir + "icons/triangle-orange.png")
            self._warningWidget = self.createPixmapWidget(self.statusBarIconArea, warning_icon)
            error_icon = os.path.join(self.widgetDir + "icons/triangle-red.png")
            self._errorWidget = self.createPixmapWidget(self.statusBarIconArea, error_icon)

    # status bar handler functions
    def createPixmapWidget(self, parent, iconName):
        w = QLabel(parent)
        parent.layout().addWidget(w)
        w.setFixedSize(16, 16)
        w.hide()
        if os.path.exists(iconName):
            w.setPixmap(QPixmap(iconName))
        return w

    def setState(self, stateType, id, text):
        stateChanged = super().setState(stateType, id, text)
        if not stateChanged or not hasattr(self, "widgetStatusArea"):
            return

        iconsShown = 0
        for state, widget, use in [("Warning", self._warningWidget, self._owWarning), ("Error", self._errorWidget, self._owError)]:
            if not widget: continue
            if use and self.widgetState[state] != {}:
                widget.setToolTip("\n".join(self.widgetState[state].values()))
                widget.show()
                iconsShown = 1
            else:
                widget.setToolTip("")
                widget.hide()

        if iconsShown:
            self.statusBarIconArea.show()
        else:
            self.statusBarIconArea.hide()

        if (stateType == "Warning" and self._owWarning) or (stateType == "Error" and self._owError):
            if text:
                self.setStatusBarText(stateType + ": " + text)
            else:
                self.setStatusBarText("")
        self.updateStatusBarState()

    def updateWidgetStateInfo(self, stateType, id, text):
        html = self.widgetStateToHtml(self._owInfo, self._owWarning, self._owError)
        if html:
            self.widgetStateInfoBox.show()
            self.widgetStateInfo.setText(html)
            self.widgetStateInfo.setToolTip(html)
        else:
            if not self.widgetStateInfoBox.isVisible():
                dHeight = - self.widgetStateInfoBox.height()
            else:
                dHeight = 0
            self.widgetStateInfoBox.hide()
            self.widgetStateInfo.setText("")
            self.widgetStateInfo.setToolTip("")
            width, height = self.width(), self.height() + dHeight
            self.resize(width, height)

    def updateStatusBarState(self):
        if not hasattr(self, "widgetStatusArea"):
            return
        if self._owShowStatus and (self.widgetState["Warning"] != {} or self.widgetState["Error"] != {}):
            self.widgetStatusArea.show()
        else:
            self.widgetStatusArea.hide()

    def setStatusBarText(self, text, timeout=5000):
        if hasattr(self, "widgetStatusBar"):
            self.widgetStatusBar.showMessage(" " + text, timeout)

if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWWidget()
    ow.show()
    a.exec_()
