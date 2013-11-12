"""
<name>Parallel Coordinates (Qt)</name>
<description>Parallel coordinates (multiattribute) visualization.</description>
<contact>Gregor Leban (gregor.leban@fri.uni-lj.si)</contact>
<icon>icons/ParallelCoordinates.svg</icon>
<priority>170</priority>
"""
# ParallelCoordinates.py
#
# Show data using parallel coordinates visualization method
#
from OWVisWidget import *
from OWParallelGraphQt import *
import OWToolbars, OWGUI, OWColorPalette, orngVisFuncts
from sys import getrecursionlimit, setrecursionlimit

###########################################################################################
##### WIDGET : Parallel coordinates visualization
###########################################################################################
class OWParallelCoordinatesQt(OWVisWidget):
    settingsList = ["graph.jitterSize", "graph.showDistributions",
                    "graph.showAttrValues",
                    "graph.useSplines", "graph.alphaValue", "graph.alphaValue2", "graph.show_legend", "autoSendSelection",
                    "toolbarSelection", "graph.showStatistics", "colorSettings", "selectedSchemaIndex", "showAllAttributes"]
    jitterSizeNums = [0, 2,  5,  10, 15, 20, 30]
    contextHandlers = {"": DomainContextHandler("", [ContextField("shownAttributes", DomainContextHandler.RequiredList, selected="selectedShown", reservoir="hiddenAttributes")])}

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Parallel Coordinates (Qt)", TRUE)

        #add a graph widget
        self.graph = OWParallelGraph(self, self.mainArea)
        self.mainArea.layout().addWidget(self.graph)

        self.showAllAttributes = 0

        self.inputs = [("Data", ExampleTable, self.setData, Default), ("Data Subset", ExampleTable, self.setSubsetData), ("Features", AttributeList, self.setShownAttributes)]
        self.outputs = [("Selected Data", ExampleTable), ("Other Data", ExampleTable), ("Features", AttributeList)]

        #set default settings
        self.data = None
        self.subsetData = None
        self.autoSendSelection = 1
        self.attrDiscOrder = "Unordered"
        self.attrContOrder = "Unordered"
        self.projections = None
        self.correlationDict = {}
        self.middleLabels = "Correlations"
        self.attributeSelectionList = None
        self.toolbarSelection = 0
        self.colorSettings = None
        self.selectedSchemaIndex = 0

        self.graph.jitterSize = 10
        self.graph.showDistributions = 1
        self.graph.showStatistics = 0
        self.graph.showAttrValues = 1
        self.graph.useSplines = 0
        self.graph.show_legend = 1

        #load settings
        self.loadSettings()

        #GUI
        self.tabs = OWGUI.tabWidget(self.controlArea)
        self.GeneralTab = OWGUI.createTabPage(self.tabs, "Main")
        self.SettingsTab = OWGUI.createTabPage(self.tabs, "Settings")

        self.createShowHiddenLists(self.GeneralTab, callback = self.updateGraph)
        self.connect(self.shownAttribsLB, SIGNAL('itemDoubleClicked(QListWidgetItem*)'), self.flipAttribute)

        self.optimizationDlg = ParallelOptimization(self, signalManager = self.signalManager)
        self.optimizationDlgButton = OWGUI.button(self.GeneralTab, self, "Optimization Dialog", callback = self.optimizationDlg.reshow, debuggingEnabled = 0)

        self.zoomSelectToolbar = OWToolbars.ZoomSelectToolbar(self, self.GeneralTab, self.graph, self.autoSendSelection, buttons = (1, 2, 0, 7, 8))
        self.connect(self.zoomSelectToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)

        #connect controls to appropriate functions
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        # ####################################
        # SETTINGS functionality
        box = OWGUI.widgetBox(self.SettingsTab, "Transparency")
        OWGUI.hSlider(box, self, 'graph.alphaValue', label = "Examples: ", minValue=0, maxValue=255, step=10, callback = self.updateGraph, tooltip = "Alpha value used for drawing example lines")
        OWGUI.hSlider(box, self, 'graph.alphaValue2', label = "Rest:     ", minValue=0, maxValue=255, step=10, callback = self.updateGraph, tooltip = "Alpha value used to draw statistics, example subsets, ...")

        box = OWGUI.widgetBox(self.SettingsTab, "Jittering Options")
        OWGUI.comboBox(box, self, "graph.jitterSize", label = 'Jittering size (% of size):  ', orientation='horizontal', callback = self.setJitteringSize, items = self.jitterSizeNums, sendSelectedValue = 1, valueType = float)

        # visual settings
        box = OWGUI.widgetBox(self.SettingsTab, "Visual Settings")

        OWGUI.checkBox(box, self, 'graph.showAttrValues', 'Show attribute values', callback = self.updateGraph)
        OWGUI.checkBox(box, self, 'graph.useSplines', 'Show splines', callback = self.updateGraph, tooltip  = "Show lines using splines")
        
        self.graph.gui.show_legend_check_box(box)

        box = OWGUI.widgetBox(self.SettingsTab, "Axis Distance")
        resizeColsBox = OWGUI.widgetBox(box, 0, "horizontal", 0)
        OWGUI.label(resizeColsBox, self, "Increase/decrease distance: ")
        b = OWGUI.toolButton(resizeColsBox, self, "+", callback=self.increaseAxesDistance, tooltip = "Increase the distance between the axes", width=30, height = 20)
        b = OWGUI.toolButton(resizeColsBox, self, "-", callback=self.decreaseAxesDistance, tooltip = "Decrease the distance between the axes", width=30, height = 20)
        OWGUI.rubber(resizeColsBox)
        OWGUI.checkBox(box, self, "graph.autoUpdateAxes", "Auto scale X axis", tooltip = "Auto scale X axis to show all visualized attributes", callback = self.updateGraph)

        box = OWGUI.widgetBox(self.SettingsTab, "Statistical Information")
        OWGUI.comboBox(box, self, "graph.showStatistics", label = "Statistics: ", orientation = "horizontal", labelWidth=90, items = ["No statistics", "Means, deviations", "Median, quartiles"], callback = self.updateGraph, sendSelectedValue = 0, valueType = int)
        OWGUI.comboBox(box, self, "middleLabels", label = "Middle labels: ", orientation="horizontal", labelWidth=90, items = ["No labels", "Correlations", "VizRank"], callback = self.updateGraph, tooltip = "The information do you wish to view on top in the middle of coordinate axes", sendSelectedValue = 1, valueType = str)
        OWGUI.checkBox(box, self, 'graph.showDistributions', 'Show distributions', callback = self.updateGraph, tooltip = "Show bars with distribution of class values (only for discrete attributes)")

        box = OWGUI.widgetBox(self.SettingsTab, "Colors", orientation = "horizontal")
        OWGUI.button(box, self, "Set colors", self.setColors, tooltip = "Set the canvas background color and color palette for coloring continuous variables", debuggingEnabled = 0)

        box = OWGUI.widgetBox(self.SettingsTab, "Auto Send Selected Data When...")
        OWGUI.checkBox(box, self, 'autoSendSelection', 'Adding/Removing selection areas', callback = self.selectionChanged, tooltip = "Send selected data whenever a selection area is added or removed")
        OWGUI.checkBox(box, self, 'graph.sendSelectionOnUpdate', 'Moving/Resizing selection areas', tooltip = "Send selected data when a user moves or resizes an existing selection area")
        self.graph.autoSendSelectionCallback = self.selectionChanged

        self.SettingsTab.layout().addStretch(100)
        self.icons = self.createAttributeIconDict()

        dlg = self.createColorDialog()
        self.graph.contPalette = dlg.getContinuousPalette("contPalette")
        self.graph.discPalette = dlg.getDiscretePalette("discPalette")
        self.graph.setCanvasBackground(dlg.getColor("Canvas"))
        [self.zoomSelectToolbar.actionZooming, self.zoomSelectToolbar.actionRectangleSelection, self.zoomSelectToolbar.actionPolygonSelection][self.toolbarSelection](*[])
        self.cbShowAllAttributes()

        self.resize(900, 700)

    def flipAttribute(self, item):
        if self.graph.flipAttribute(str(item.text())):
            self.updateGraph()
            self.information(0)
        else:
            self.information(0, "Didn't flip the attribute. To flip a continuous attribute uncheck 'Global value scaling' checkbox.")

    def updateGraph(self, *args):
        attrs = self.getShownAttributeList()
        self.graph.updateData(attrs, self.buildMidLabels(attrs))


    def increaseAxesDistance(self):
	m, M = self.graph.bounds_for_axis(xBottom)
        if (M-m) == 0:
            return      # we have not yet updated the axes (self.graph.updateAxes())
        self.graph.setAxisScale(xBottom, m, M - (M-m)/10., 1)
        self.graph.replot()

    def decreaseAxesDistance(self):
	m, M = self.graph.bounds_for_axis(xBottom)
	if (M-m) == 0:
            return      # we have not yet updated the axes (self.graph.updateAxes())

        self.graph.setAxisScale(xBottom, m, min(len(self.graph.visualizedAttributes)-1, M + (M-m)/10.), 1)
        self.graph.replot()


    # build a list of strings that will be shown in the middle of the parallel axis
    def buildMidLabels(self, attrs):
        labels = []
        if self.middleLabels == "No labels" or not self.graph.haveData: return None
        elif self.middleLabels == "Correlations":
            for i in range(len(attrs)-1):
                corr = None
                if (attrs[i], attrs[i+1]) in self.correlationDict:   corr = self.correlationDict[(attrs[i], attrs[i+1])]
                elif (attrs[i+1], attrs[i]) in self.correlationDict: corr = self.correlationDict[(attrs[i+1], attrs[i])]
                else:
                    try:
                        corr = orngVisFuncts.computeCorrelation(self.graph.rawData, attrs[i], attrs[i+1])
                    except:
                        corr = None
                    self.correlationDict[(attrs[i], attrs[i+1])] = corr
                if corr and (self.graph.attributeFlipInfo.get(attrs[i], 0) != self.graph.attributeFlipInfo.get(attrs[i+1], 0)): corr = -corr
                if corr: labels.append("%2.3f" % (corr))
                else: labels.append("")
        elif self.middleLabels == "VizRank":
            for i in range(len(attrs)-1):
                val = self.optimizationDlg.getVizRankVal(attrs[i], attrs[i+1])
                if val: labels.append("%2.2f%%" % (val))
                else: labels.append("")
        return labels


    # ------------- SIGNALS --------------------------
    # receive new data and update all fields
    def setData(self, data):
        if data and (len(data) == 0 or len(data.domain) == 0):
            data = None
        if self.data != None and data != None and self.data.checksum() == data.checksum():
            return    # check if the new data set is the same as the old one

        self.closeContext()
        sameDomain = self.data and data and data.domain.checksum() == self.data.domain.checksum() # preserve attribute choice if the domain is the same
        self.projections = None
        self.correlationDict = {}
        self.data = data
        self.optimizationDlg.clearResults()
        if not sameDomain:
            self.setShownAttributeList(self.attributeSelectionList)
        self.openContext("", self.data)
        self.resetAttrManipulation()


    def setSubsetData(self, subData):
        self.subsetData = subData


    # attribute selection signal - list of attributes to show
    def setShownAttributes(self, attributeSelectionList):
        self.attributeSelectionList = attributeSelectionList


    # this is called by OWBaseWidget after setData and setSubsetData are called. this way the graph is updated only once
    def handleNewSignals(self):
        self.graph.setData(self.data, self.subsetData)
        if self.attributeSelectionList and 0 not in [attr in self.graph.attributeNameIndex for attr in self.attributeSelectionList]:
            self.setShownAttributeList(self.attributeSelectionList)
        else:
            self.setShownAttributeList()
        self.attributeSelectionList = None
        self.updateGraph()
        self.sendSelections()


    def sendShownAttributes(self, attrList = None):
        if attrList == None:
            attrList = self.getShownAttributeList()
        self.send("Features", attrList)

    def selectionChanged(self):
        self.zoomSelectToolbar.buttonSendSelections.setEnabled(not self.autoSendSelection)
        if self.autoSendSelection:
            self.sendSelections()

    # send signals with selected and unselected examples as two datasets
    def sendSelections(self):
        (selected, unselected) = self.graph.getSelectionsAsExampleTables()
        self.send("Selected Data", selected)
        self.send("Other Data", unselected)


    # jittering options
    def setJitteringSize(self):
        self.graph.rescaleData()
        self.updateGraph()

    def setColors(self):
        dlg = self.createColorDialog()
        if dlg.exec_():
            self.colorSettings = dlg.getColorSchemas()
            self.selectedSchemaIndex = dlg.selectedSchemaIndex
            self.graph.contPalette = dlg.getContinuousPalette("contPalette")
            self.graph.discPalette = dlg.getDiscretePalette("discPalette")
            self.graph.setCanvasBackground(dlg.getColor("Canvas"))
            self.updateGraph()

    def createColorDialog(self):
        c = OWColorPalette.ColorPaletteDlg(self, "Color Palette")
        c.createDiscretePalette("discPalette", "Discrete Palette")
        c.createContinuousPalette("contPalette", "Continuous Palette")
        box = c.createBox("otherColors", "Other Colors")
        c.createColorButton(box, "Canvas", "Canvas color", self.graph.color(OWPalette.Canvas))
        c.setColorSchemas(self.colorSettings, self.selectedSchemaIndex)
        return c

    def saveSettings(self):
        OWWidget.saveSettings(self)
        self.optimizationDlg.saveSettings()

    def closeEvent(self, ce):
        self.optimizationDlg.hide()
        OWWidget.closeEvent(self, ce)

    def sendReport(self):
        self.reportImage(self.graph.saveToFileDirect, QSize(500, 500))



CORRELATION = 0
VIZRANK = 1
#
# Find attribute subsets that are interesting to visualize using parallel coordinates
class ParallelOptimization(OWWidget):
    resultListList = [50, 100, 200, 500, 1000]
    qualityMeasure =  ["Classification accuracy", "Average correct", "Brier score"]
    testingMethod = ["Leave one out", "10-fold cross validation", "Test on learning set"]

    settingsList = ["attributeCount", "fileBuffer", "lastSaveDirName", "optimizationMeasure",
                    "numberOfAttributes", "orderAllAttributes", "optimizationMeasure"]

    def __init__(self, parallelWidget, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Parallel Optimization Dialog", FALSE)
        self.setCaption("Parallel Optimization Dialog")
        self.parallelWidget = parallelWidget

        self.optimizationMeasure = 0
        self.attributeCount = 5
        self.numberOfAttributes = 6
        self.fileName = ""
        self.lastSaveDirName = os.getcwd() + "/"
        self.fileBuffer = []
        self.projections = []
        self.allResults = []
        self.canOptimize = 0
        self.orderAllAttributes = 1 # do we wish to order all attributes or find just an interesting subset
        self.worstVal = -1  # used in heuristics to stop the search in uninteresting parts of the graph

        self.loadSettings()

        self.measureBox = OWGUI.radioButtonsInBox(self.controlArea, self, "optimizationMeasure", ["Correlation", "VizRank"], box = "Select optimization measure", callback = self.updateGUI)
        self.vizrankSettingsBox = OWGUI.widgetBox(self.controlArea, "VizRank settings")
        self.optimizeBox = OWGUI.widgetBox(self.controlArea, "Optimize")
        self.manageBox = OWGUI.widgetBox(self.controlArea, "Manage results")
        self.resultsBox = OWGUI.widgetBox(self.mainArea, "Results")

        self.resultList = OWGUI.listBox(self.resultsBox, self)
        self.resultList.setMinimumSize(200,200)
        self.connect(self.resultList, SIGNAL("itemSelectionChanged()"), self.showSelectedAttributes)

        # remove non-existing files
        names = []
        for i in range(len(self.fileBuffer)-1, -1, -1):
            (short, longName) = self.fileBuffer[i]
            if not os.path.exists(longName):
                self.fileBuffer.remove((short, longName))
            else: names.append(short)
        names.append("(None)")
        self.fileName = "(None)"

        self.hbox1 = OWGUI.widgetBox(self.vizrankSettingsBox, "VizRank projections file", orientation = "horizontal")
        self.vizrankFileCombo = OWGUI.comboBox(self.hbox1, self, "fileName", items = names, tooltip = "File that contains information about interestingness of scatterplots \ngenerated by VizRank method in scatterplot widget", callback = self.changeProjectionFile, sendSelectedValue = 1, valueType = str)
        self.browseButton = OWGUI.button(self.hbox1, self, "...", callback = self.loadProjections)
        self.browseButton.setMaximumWidth(20)

        self.resultsInfoBox = OWGUI.widgetBox(self.vizrankSettingsBox, "VizRank parameters")
        self.kNeighborsLabel = OWGUI.widgetLabel(self.resultsInfoBox, "Number of neighbors (k):")
        self.percentDataUsedLabel = OWGUI.widgetLabel(self.resultsInfoBox, "Percent of data used:")
        self.testingMethodLabel = OWGUI.widgetLabel(self.resultsInfoBox, "Testing method used:")
        self.qualityMeasureLabel = OWGUI.widgetLabel(self.resultsInfoBox, "Quality measure used:")

        #self.numberOfAttributesCombo = OWGUI.comboBoxWithCaption(self.optimizeBox, self, "numberOfAttributes", "Number of visualized attributes: ", tooltip = "Projections with this number of attributes will be evaluated", items = [x for x in range(3, 12)], sendSelectedValue = 1, valueType = int)
        self.allAttributesRadio = QRadioButton("Order all attributes", self.optimizeBox)
        self.optimizeBox.layout().addWidget(self.allAttributesRadio)
        self.connect(self.allAttributesRadio, SIGNAL("clicked()"), self.setAllAttributeRadio)
        box = OWGUI.widgetBox(self.optimizeBox, orientation = "horizontal")
        self.subsetAttributeRadio = QRadioButton("Find subsets of", box)
#        self.optimizeBox.layout().addWidget(self.subsetAttributeRadio)
        box.layout().addWidget(self.subsetAttributeRadio)
        self.connect(self.subsetAttributeRadio, SIGNAL("clicked()"), self.setSubsetAttributeRadio)
        self.subsetAttributeEdit = OWGUI.lineEdit(box, self, "numberOfAttributes", valueType = int)
        self.subsetAttributeEdit.setMaximumWidth(30)
        label  = OWGUI.widgetLabel(box, "attributes")

        self.startOptimizationButton = OWGUI.button(self.optimizeBox, self, "Start Optimization", callback = self.startOptimization)
        f = self.startOptimizationButton.font()
        f.setBold(1)
        self.startOptimizationButton.setFont(f)
        self.stopOptimizationButton = OWGUI.button(self.optimizeBox, self, "Stop Evaluation", callback = self.stopOptimizationClick)
        self.stopOptimizationButton.setFont(f)
        self.stopOptimizationButton.hide()
        self.connect(self.stopOptimizationButton , SIGNAL("clicked()"), self.stopOptimizationClick)


        self.clearButton = OWGUI.button(self.manageBox, self, "Clear Results", self.clearResults)
        self.loadButton  = OWGUI.button(self.manageBox, self, "Load", self.loadResults)
        self.saveButton  = OWGUI.button(self.manageBox, self, "Save", self.saveResults)
        self.closeButton = OWGUI.button(self.manageBox, self, "Close Dialog", self.hide)

        self.changeProjectionFile()
        self.updateGUI()
        if self.orderAllAttributes: self.setAllAttributeRadio()
        else:                       self.setSubsetAttributeRadio()

    def updateGUI(self):
        self.vizrankSettingsBox.setEnabled(self.optimizationMeasure)

    # if user clicks new attribute list in optimization dialog, we update shown attributes
    def showSelectedAttributes(self):
        attrList = self.getSelectedAttributes()
        if not attrList: return

        self.parallelWidget.setShownAttributeList(attrList)
        self.parallelWidget.graph.removeAllSelections()

        self.parallelWidget.middleLabels = (self.optimizationMeasure == VIZRANK and "VizRank") or "Correlations"
        self.parallelWidget.updateGraph()

    def setAllAttributeRadio(self):
        self.orderAllAttributes = 1
        self.allAttributesRadio.setChecked(1)
        self.subsetAttributeRadio.setChecked(0)
        self.subsetAttributeEdit.setEnabled(0)

    def setSubsetAttributeRadio(self):
        self.orderAllAttributes = 0
        self.allAttributesRadio.setChecked(0)
        self.subsetAttributeRadio.setChecked(1)
        self.subsetAttributeEdit.setEnabled(1)

    # return list of selected attributes
    def getSelectedAttributes(self):
        if self.resultList.count() == 0 or self.allResults == []:
            return None
        return self.allResults[self.resultList.currentRow()][1]

    # called when optimization is in progress
    def canContinueOptimization(self):
        return self.canOptimize

    def getWorstVal(self):
        return self.worstVal

    def stopOptimizationClick(self):
        self.canOptimize = 0

    # get vizrank value for attributes attr1 and attr2
    def getVizRankVal(self, attr1, attr2):
        if not self.projections: return None
        for (val, [a1, a2]) in self.projections:
            if (attr1 == a1 and attr2 == a2) or (attr1 == a2 and attr2 == a1): return val
        return None

    def changeProjectionFile(self):
        for (short, int) in self.fileBuffer:
            if short == self.fileName:
                self.loadProjections(int)
                return

    # load projections from a file
    def loadProjections(self, name = None):
        self.projections = []
        self.kNeighborsLabel.setText("Number of neighbors (k): " )
        self.percentDataUsedLabel.setText("Percent of data used:" )
        self.testingMethodLabel.setText("Testing method used:" )
        self.qualityMeasureLabel.setText("Quality measure used:" )

        if name == None:
            name = str(QFileDialog.getOpenFileName(self, "Open Projections",  self.lastSaveDirName, "Interesting projections (*.proj)"))
            if name == "": return

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        file = open(name, "rt")
        settings = eval(file.readline()[:-1])
        if "parentName" in settings and settings["parentName"].lower() != "scatterplot":
            QMessageBox.critical( None, "Optimization Dialog", 'Unable to load projection file. Only projection file generated by scatterplot is compatible. \nThis file was created using %s method'%(settings["parentName"]), QMessageBox.Ok)
            file.close()
            return

        if type(eval(file.readline()[:-1])) != list:    # second line must contain a list of classes that we tried to separate
            QMessageBox.critical(None,'Old version of projection file','This file was saved with an older version of k-NN Optimization Dialog. The new version of dialog offers \nsome additional functionality and therefore you have to compute the projection quality again.',QMessageBox.Ok)
            file.close()
            return

        try:
            line = file.readline()[:-1]; ind = 0    # first line is a settings line
            (acc, other_results, lenTable, attrList, tryIndex, strList) = eval(line)
            if len(attrList) != 2:
                QMessageBox.information(self, "Incorrect file", "File should contain projections with 2 attributes!", QMessageBox.Ok)
                file.close()
                return

            while (line != ""):
                (acc, other_results, lenTable, attrList, tryIndex, strList) = eval(line)
                self.projections += [(acc, attrList)]
                line = file.readline()[:-1]
        except:
            self.projections = []
            file.close()
            QMessageBox.information(self, "Incorrect file", "Incorrect file format!", QMessageBox.Ok)
            return

        file.close()

        if (shortFileName, name) in self.fileBuffer:
            self.fileBuffer.remove((shortFileName, name))

        self.fileBuffer.insert(0, (shortFileName, name))


        if len(self.fileBuffer) > 10:
            self.fileBuffer.remove(self.fileBuffer[-1])

        self.vizrankFileCombo.clear()
        for i in range(len(self.fileBuffer)):
            self.vizrankFileCombo.addItem(self.fileBuffer[i][0])
        self.fileName = shortFileName

        self.kNeighborsLabel.setText("Number of neighbors (k): " + str(settings["kValue"]))
        self.percentDataUsedLabel.setText("Percent of data used: " + "%d %%" % (settings["percentDataUsed"]))
        self.testingMethodLabel.setText("Testing method used: " + self.testingMethod[settings["testingMethod"]])
        self.qualityMeasureLabel.setText("Quality measure used: " + self.qualityMeasure[settings["qualityMeasure"]])


    def addProjection(self, val, attrList):
        index = self.findTargetIndex(val)
        self.allResults.insert(index, (val, attrList))
        self.resultList.insertItem(index, "%.3f - %s" % (val, str(attrList)))


    def findTargetIndex(self, accuracy):
        # use bisection to find correct index
        top = 0; bottom = len(self.allResults)

        while (bottom-top) > 1:
            mid  = (bottom + top)/2
            if max(accuracy, self.allResults[mid][0]) == accuracy: bottom = mid
            else: top = mid

        if len(self.allResults) == 0: return 0
        if max(accuracy, self.allResults[top][0]) == accuracy:
            return top
        else:
            return bottom


    def startOptimization(self):
        self.clearResults()
        if self.parallelWidget.data == None: return

        if self.optimizationMeasure == VIZRANK and self.fileName == "":
            QMessageBox.information(self, "No projection file", "If you wish to optimize using VizRank you first have to load a projection file \ncreated by VizRank using Scatterplot widget.", QMessageBox.Ok)
            return
        if self.parallelWidget.data == None:
            QMessageBox.information(self, "Missing data set", "A data set has to be loaded in order to perform optimization.", QMessageBox.Ok)
            return

        attrInfo = []
        self.progressBarInit()
        if self.optimizationMeasure == CORRELATION:
            attrList = [attr.name for attr in self.parallelWidget.data.domain.attributes]
            self.startOptimizationButton.hide()
            self.stopOptimizationButton.show()
            self.canOptimize = 1
            class StopOptimizationException(Exception):
                pass
            def progressSetWithStop(value):
                if not self.canContinueOptimization():
                    raise StopOptimizationException()
                else:
                    self.progressBarSet(value * 0.9)
            try: 
                attrInfo = orngVisFuncts.computeCorrelationBetweenAttributes(self.parallelWidget.data, attrList, progressCallback=progressSetWithStop)
            except StopOptimizationException:
                attrInfo = []
                self.startOptimizationButton.show()
                self.stopOptimizationButton.hide()
                
#            self.progressBarFinished()
            #attrInfo = orngVisFuncts.computeCorrelationInsideClassesBetweenAttributes(self.parallelWidget.data, attrList)
        elif self.optimizationMeasure == VIZRANK:
            for (val, [a1, a2]) in self.projections:
                attrInfo.append((val, a1, a2))

            # check if all attributes in loaded projection file are actually present in this data set
            attrs = [attr.name for attr in self.parallelWidget.data.domain.attributes]
            for (v, a1, a2) in attrInfo:
                if a1 not in attrs:
                    print("attribute " + a1 + " was not found in the data set. You probably loaded wrong file with VizRank projections.")
                    return
                if a2 not in attrs:
                    print("attribute " + a2 + " was not found in the data set. You probably loaded wrong file with VizRank projections.")
                    return

        if len(attrInfo) == 0:
            print("len(attrInfo) == 0. No attribute pairs. Unable to optimize."); return

        self.worstVal = -1
        self.canOptimize = 1
        self.startOptimizationButton.hide()
        self.stopOptimizationButton.show()
        #qApp.processEvents()        # allow processing of other events

        if self.orderAllAttributes:
            orngVisFuncts.optimizeAttributeOrder(attrInfo, len(self.parallelWidget.data.domain.attributes), self, qApp)
        else:
            orngVisFuncts.optimizeAttributeOrder(attrInfo, self.numberOfAttributes, self, qApp)

        self.stopOptimizationButton.hide()
        self.startOptimizationButton.show()
        
        self.progressBarFinished()


    # ################################
    # MANAGE RESULTS
    def updateShownProjections(self, *args):
        self.resultList.clear()
        for i in range(len(self.allResults)):
            self.resultList.addItem("%.2f - %s" % (self.allResults[i][0], str(self.allResults[i][1])), i)
        if self.resultList.count() > 0: self.resultList.setCurrentRow(0)

    def clearResults(self):
        self.allResults = []
        self.resultList.clear()


    def saveResults(self, filename = None):
        if filename == None:
            filename = ""
            datasetName = getattr(self.parallelWidget.graph.rawData, "name", "")
            if datasetName != "":
                filename = os.path.splitext(os.path.split(datasetName)[1])[0]
            if self.optimizationMeasure == CORRELATION: filename += " - " + "correlation"
            else:                                       filename += " - " + "vizrank"

            name = str(QFileDialog.getSaveFileName(self, "Save Parallel Projections",  os.path.join(self.lastSaveDirName, filename), "Parallel projections (*.papr)"))
            if name == "": return
        else:
            name = filename

        # take care of extension
        if os.path.splitext(name)[1] != ".papr": name += ".papr"

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        # open, write and save file
        file = open(name, "wt")
        for val in self.allResults:
            file.write(str(val) + "\n")
        file.close()

    def loadResults(self):
        self.clearResults()

        name = str(QFileDialog.getOpenFileName(self, "Open Parallel Projections",  self.lastSaveDirName, "Parallel projections (*.papr)"))
        if name == "": return

        dirName, shortFileName = os.path.split(name)
        self.lastSaveDirName = dirName

        file = open(name, "rt")
        line = file.readline()[:-1]; ind = 0
        while (line != ""):
            (val, attrList) = eval(line)
            self.allResults.insert(ind, (val, attrList))
            self.resultList.addItem("%.2f - %s" % (val, str(attrList)), ind)
            line = file.readline()[:-1]
            ind+=1
        file.close()


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWParallelCoordinates()
    ow.show()
    ow.graph.discPalette = ColorPaletteGenerator(rgbColors = [(127, 201, 127), (190, 174, 212), (253, 192, 134)])
    data = orange.ExampleTable("../../doc/datasets/iris.tab")
#    data = orange.ExampleTable(r"e:\Development\Orange Datasets\UCI\wine.tab")
    #data = orange.ExampleTable(r"e:\Development\Orange Datasets\UCI\zoo.tab")
    ow.setData(data)
    ow.handleNewSignals()
    
    a.exec_()
