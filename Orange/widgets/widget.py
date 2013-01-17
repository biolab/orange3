#
# OWWidget.py
# Orange Widget
# A General Orange Widget, from which all the Orange Widgets are derived
#

from OWBaseWidget import *

class OWWidget(OWBaseWidget):
    def __init__(self, parent=None, signalManager=None, title="Orange Widget", wantGraph=False, wantStatusBar=False, savePosition=True, wantMainArea=1, noReport=False, showSaveGraph=1, resizingEnabled=1, wantStateInfoWidget=None, **args):
        """
        Initialization
        Parameters:
            title - The title of the\ widget, including a "&" (for shortcut in about box)
            wantGraph - displays a save graph button or not
        """

        OWBaseWidget.__init__(self, parent, signalManager, title, savePosition=savePosition, resizingEnabled=resizingEnabled, **args)

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
        if OWReport.get_instance() and not noReport and hasattr(self, "sendReport"):
            self.buttonBackground.show()
            self.reportButton = OWGUI.button(self.buttonBackground, self, "&Report", self.reportAndFinish, debuggingEnabled=0)
            self.reportButton.setAutoDefault(0)

        if wantStatusBar:
            #self.widgetStatusArea = OWGUI.widgetBox(self, orientation = "horizontal", margin = 2)
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
            #self.statusBarIconArea.setFrameStyle (QFrame.Panel + QFrame.Sunken)
            #self.widgetStatusBar.setFrameStyle (QFrame.Panel + QFrame.Sunken)
            #self.widgetStatusBar.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred))
            #self.widgetStatusBar.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred))
            #self.widgetStatusBar.updateGeometry()
            #self.statusBarIconArea.setFixedSize(16*2,18)
            self.statusBarIconArea.hide()


            # create pixmaps used in statusbar to show info, warning and error messages
            #self._infoWidget, self._infoPixmap = self.createPixmapWidget(self.statusBarIconArea, os.path.join(self.widgetDir + "icons/triangle-blue.png"))
            self._warningWidget = self.createPixmapWidget(self.statusBarIconArea, os.path.join(self.widgetDir + "icons/triangle-orange.png"))
            self._errorWidget = self.createPixmapWidget(self.statusBarIconArea, os.path.join(self.widgetDir + "icons/triangle-red.png"))



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
        stateChanged = OWBaseWidget.setState(self, stateType, id, text)
        if not stateChanged or not hasattr(self, "widgetStatusArea"):
            return

        iconsShown = 0
        #for state, widget, icon, use in [("Info", self._infoWidget, self._owInfo), ("Warning", self._warningWidget, self._owWarning), ("Error", self._errorWidget, self._owError)]:
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

        #if (stateType == "Info" and self._owInfo) or (stateType == "Warning" and self._owWarning) or (stateType == "Error" and self._owError):
        if (stateType == "Warning" and self._owWarning) or (stateType == "Error" and self._owError):
            if text:
                self.setStatusBarText(stateType + ": " + text)
            else:
                self.setStatusBarText("")
        self.updateStatusBarState()
        #qApp.processEvents()

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
#            QTimer.singleShot(1, lambda :self.resize(width, height))

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

    def reportAndFinish(self):
        self.sendReport()
        self.finishReport()

    def startReport(self, name=None, needDirectory=False):
        if self.__reportData is not None:
            print("Cannot open a new report when an old report is still active")
            return False
        self.reportName = name or self.windowTitle()
        self.__reportData = ""
        if needDirectory:
            return OWReport.get_instance().createDirectory()
        else:
            return True

    def reportSection(self, title):
        if self.__reportData is None:
            self.startReport()
        self.__reportData += "\n\n<h2>%s</h2>\n\n" % title

    def reportSubsection(self, title):
        if self.__reportData is None:
            self.startReport()
        self.__reportData += "\n\n  <h3>%s</h3>\n\n" % title

    def reportList(self, items):
        if self.__reportData is None:
            self.startReport()
        self.startReportList()
        for item in items:
            self.addToReportList(item)
        self.finishReportList()

    def getUniqueFileName(self, patt):
        return OWReport.get_instance().getUniqueFileName(patt)

    def getUniqueImageName(self, nm="img", ext=".png"):
        return OWReport.get_instance().getUniqueFileName(nm + "%06i" + ext)

    def reportImage(self, filenameOrFunc, *args):
        if self.__reportData is None:
            self.startReport()

        if isinstance(filenameOrFunc, str):
            self.__reportData += '    <IMG src="%s"/>\n' % filenameOrFunc
        else:
            sfn, ffn = self.getUniqueImageName()
            filenameOrFunc(ffn, *args)
            self.reportImage(sfn)

    svg_type = "image/svg+xml"
    def reportObject(self, type, data, **attrs):
        if self.__reportData is None:
            self.startReport()
        self.__reportData += '<object type="%s" data="%s" %s></object>' % (type, data, " ".join('%s="%s"' % attr for attr in attrs.items()))

    def startReportList(self):
        if self.__reportData is None:
            self.startReport()
        self.__reportData += "    <UL>\n"

    def addToReportList(self, item):
        self.__reportData += "      <LI>%s</LI>\n" % item

    def finishReportList(self):
        self.__reportData += "    </UL>\n"

    def reportSettings(self, sectionName="", settingsList=None, closeList=True):
        if sectionName:
            self.reportSection(sectionName)
        elif self.__reportData is None:
            self.startReport()
        self.__reportData += "    <ul>%s</ul>\n" % "".join("<b>%s: </b>%s<br/>" % item for item in settingsList if item)

    def reportRaw(self, text):
        if self.__reportData is None:
            self.startReport()
        self.__reportData += text

    def prepareDataReport(self, data, listAttributes=True, exampleCount=True):
        if data:
            res = []
            if exampleCount:
                res.append(("Examples", str(len(data))))
            if listAttributes:
                if data.domain.attributes:
                    res.append(("Attributes", "%i %s" % (
                                len(data.domain.attributes),
                                 "(%s%s)" % (", ".join(x.name for foo, x in zip(range(30), data.domain.attributes)), "..." if len(data.domain.attributes) > 30 else "")
                              )))
                else:
                    res.append(("Attributes", "0"))
                metas = data.domain.getmetas()
                if metas:
                  if len(metas) <= 100:
                      res.append(("Meta attributes", "%i (%s)" % (len(metas), ", ".join(x.name for x in metas.values()))))
                  else:
                      res.append(("Meta attributes", str(len(metas))))
                res.append(("Class", data.domain.classVar.name if data.domain.classVar else "<none>"))
            return res

    def reportData(self, settings, sectionName="Data", ifNone="None", listAttributes = True, exampleCount=True):
        haveSettings = False
        try:
            haveSettings = isinstance(settings, list) and len(settings[0])==2
        except:
            pass
        if not haveSettings:
            settings = self.prepareDataReport(settings, listAttributes, exampleCount)
        if not self.__reportData:
            self.startReport()
        if sectionName is not None:
            self.reportSection(sectionName)
        if settings:
            self.reportSettings("", settings)
        elif ifNone is not None:
            self.reportRaw(ifNone)



    def finishReport(self):
        if self.__reportData is not None:
            OWReport.get_instance()(self.reportName, self.__reportData or "", self.widgetId, self.windowIcon())#, self.getSettings(False))
            self.__reportData = None

import OWReport

if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWWidget()
    ow.show()
    a.exec_()
