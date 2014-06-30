import sys

import numpy
from PyQt4.QtCore import SIGNAL, QSize
from PyQt4.QtGui import QApplication

import Orange
from Orange.data import Table, Variable
from Orange.data.sql.table import SqlTable
from Orange.widgets import gui
from Orange.widgets.settings import DomainContextHandler
from Orange.widgets.utils.colorpalette import ColorPaletteDlg
from Orange.widgets.utils.plot import OWPlot, OWPalette, OWPlotGUI
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotGraphQt_test
from Orange.widgets.widget import OWWidget, Default, AttributeList


VarTypes = Variable.VarTypes


class OWScatterPlotQt(OWWidget):
    """
    <name>Scatterplot (Qt)</name>
    <description>Scatterplot visualization.</description>
    <contact>Gregor Leban (gregor.leban@fri.uni-lj.si)</contact>
    <icon>icons/ScatterPlot.svg</icon>
    <priority>130</priority>
    """
    name = 'Scatterplot'
    description = 'Scatterplot visualization'

    inputs =  [("Data", Table, "setData", Default), ("Data Subset", Table, "setSubsetData"), ("Features", AttributeList, "setShownAttributes")] #, ("Evaluation Results", ExperimentResults, "setTestResults"), ("VizRank Learner", Learner, "setVizRankLearner")]
    outputs = [("Selected Data", Table), ("Other Data", Table)]

    settingsList = ["graph." + s for s in OWPlot.point_settings + OWPlot.appearance_settings] + [
                    "graph.showXaxisTitle", "graph.showYLaxisTitle", "showGridlines",
                    "graph.showLegend", "graph.jitterSize", "graph.jitterContinuous", "graph.showFilledSymbols", "graph.showProbabilities",
                    "graph.showDistributions", "autoSendSelection", "toolbarSelection", "graph.sendSelectionOnUpdate",
                    "colorSettings", "selectedSchemaIndex", "VizRankLearnerName"]
    jitterSizeNums = [0.0, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50]

    settingsHandler = DomainContextHandler()
    # contextHandlers = {"": DomainContextHandler("", ["attrX", "attrY",
    #                                                  (["attrColor", "attrShape", "attrSize"], DomainContextHandler.Optional),
    #                                                  ("attrLabel", DomainContextHandler.Optional + DomainContextHandler.IncludeMetaAttributes)])}

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Scatterplot (Qt)", True)

        ##TODO tukaj mas testni graf!
        self.graph = OWScatterPlotGraphQt_test(self, self.mainArea, "ScatterPlotQt_test")

        #add a graph widget
        ##TODO pazi
        # self.mainArea.layout().addWidget(self.graph.pgPlotWidget)             # tale je zaresni
        self.mainArea.layout().addWidget(self.graph.glw)     # tale je testni

        ## TODO spodaj je se en POZOR, kjer nastavis palette

        # self.vizrank = OWVizRank(self, self.signalManager, self.graph, orngVizRank.SCATTERPLOT, "ScatterPlotQt")
        # self.optimizationDlg = self.vizrank

        # local variables
        self.showGridlines = 1
        self.autoSendSelection = 1
        self.toolbarSelection = 0
        self.classificationResults = None
        self.outlierValues = None
        self.colorSettings = None
        self.selectedSchemaIndex = 0
        self.graph.sendSelectionOnUpdate = 0
        self.attributeSelectionList = None

        self.data = None
        self.subsetData = None

        #load settings
        # self.loadSettings()
        self.graph.setShowXaxisTitle()
        self.graph.setShowYLaxisTitle()

        # self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        box1 = gui.widgetBox(self.controlArea, "Axis Variables")
        #x attribute
        self.attrX = ""
        self.attrXCombo = gui.comboBox(box1, self, "attrX", label="X-Axis:", labelWidth=50, orientation="horizontal", callback = self.majorUpdateGraph, sendSelectedValue = 1, valueType = str)
        # y attribute
        self.attrY = ""
        self.attrYCombo = gui.comboBox(box1, self, "attrY", label="Y-Axis:", labelWidth=50, orientation="horizontal", callback = self.majorUpdateGraph, sendSelectedValue = 1, valueType = str)

        box2 = gui.widgetBox(self.controlArea, "Point Properties")
        self.attrColor = ""
        self.attrColorCombo = gui.comboBox(box2, self, "attrColor", label="Color:", labelWidth=50, orientation="horizontal", callback = self.updateGraph, sendSelectedValue=1, valueType = str, emptyString = "(Same color)")
        # labelling
        self.attrLabel = ""
        self.attrLabelCombo = gui.comboBox(box2, self, "attrLabel", label="Label:", labelWidth=50, orientation="horizontal", callback = self.updateGraph, sendSelectedValue = 1, valueType = str, emptyString = "(No labels)")
        # shaping
        self.attrShape = ""
        self.attrShapeCombo = gui.comboBox(box2, self, "attrShape", label="Shape:", labelWidth=50, orientation="horizontal", callback = self.updateGraph, sendSelectedValue=1, valueType = str, emptyString = "(Same shape)")
        # sizing
        self.attrSize = ""
        self.attrSizeCombo = gui.comboBox(box2, self, "attrSize", label="Size:", labelWidth=50, orientation="horizontal", callback = self.updateGraph, sendSelectedValue=1, valueType = str, emptyString = "(Same size)")

        g = self.graph.gui

        box3 = g.point_properties_box(self.controlArea)
        # self.jitterSizeCombo = gui.comboBox(box3, self, "graph.jitter_size", label = 'Jittering size (% of size):'+'  ', orientation = "horizontal", callback = self.resetGraphData, items = self.jitterSizeNums, sendSelectedValue = 1, valueType = float)
        ## TODO: jitter size slider ima samo interger values -> ali lahko slajda po self.jitterSizeNums
        gui.hSlider(box3, self, value='graph.jitter_size', label='Jittering (%): ', minValue=1, maxValue=10, callback=self.resetGraphData)

        gui.checkBox(gui.indentedBox(box3), self, 'graph.jitter_continuous', 'Jitter continuous values', callback = self.resetGraphData, tooltip = "Does jittering apply also on continuous attributes?")
        gui.button(box3, self, "Set Colors", self.setColors, tooltip = "Set the canvas background color, grid color and color palette for coloring continuous variables")

        box4 = gui.widgetBox(self.controlArea, "Plot Properties")
        g.add_widgets([g.ShowLegend, g.ShowGridLines], box4)
        # gui.comboBox(box4, self, "graph.tooltipKind", items = ["Don't Show Tooltips", "Show Visible Attributes", "Show All Attributes"], callback = self.updateGraph)
        gui.checkBox(box4, self, value='graph.tooltipShowsAllAttributes', label='Show all attributes in tooltip')

        box5 = gui.widgetBox(self.controlArea, "Auto Send Selected Data When...")
        gui.checkBox(box5, self, 'autoSendSelection', 'Adding/Removing selection areas', callback = self.selectionChanged, tooltip = "Send selected data whenever a selection area is added or removed")
        gui.checkBox(box5, self, 'graph.sendSelectionOnUpdate', 'Moving/Resizing selection areas', tooltip = "Send selected data when a user moves or resizes an existing selection area")
        self.graph.selection_changed.connect(self.selectionChanged)

        # zooming / selection
        self.zoomSelectToolbar = g.zoom_select_toolbar(self.controlArea, buttons = g.default_zoom_select_buttons + [g.Spacing, g.ShufflePoints])
        self.connect(self.zoomSelectToolbar.buttons[g.SendSelection], SIGNAL("clicked()"), self.sendSelections)
        self.connect(self.zoomSelectToolbar.buttons[g.Zoom], SIGNAL("clicked()",), self.graph.zoomButtonClicked)
        self.connect(self.zoomSelectToolbar.buttons[g.Pan], SIGNAL("clicked()",), self.graph.panButtonClicked)
        self.connect(self.zoomSelectToolbar.buttons[g.Select], SIGNAL("clicked()",), self.graph.selectButtonClicked)

        self.controlArea.layout().addStretch(100)
        self.icons = gui.attributeIconDict

        self.debugSettings = ["attrX", "attrY", "attrColor", "attrLabel", "attrShape", "attrSize"]
        # self.wdChildDialogs = [self.vizrank]        # used when running widget debugging

        dlg = self.createColorDialog()
        self.graph.contPalette = dlg.getContinuousPalette("contPalette")
        self.graph.discPalette = dlg.getDiscretePalette("discPalette")


        ##TODO POZOR!
        # p = self.graph.pgPlotWidget.palette()
        p = self.graph.glw.palette()



        p.setColor(OWPalette.Canvas, dlg.getColor("Canvas"))
        p.setColor(OWPalette.Grid, dlg.getColor("Grid"))
        self.graph.set_palette(p)

        self.graph.enableGridXB(self.showGridlines)
        self.graph.enableGridYL(self.showGridlines)

        # self.graph.resize(700, 550)
        self.mainArea.setMinimumWidth(700)
        self.mainArea.setMinimumHeight(550)
        ## TODO tole je zdej minimum size --> najdi drug nacin za resize


    # def settingsFromWidgetCallback(self, handler, context):
    #     context.selectionPolygons = []
    #     for curve in self.graph.selectionCurveList:
    #         xs = [curve.x(i) for i in range(curve.dataSize())]
    #         ys = [curve.y(i) for i in range(curve.dataSize())]
    #         context.selectionPolygons.append((xs, ys))

    # def settingsToWidgetCallback(self, handler, context):
    #     selections = getattr(context, "selectionPolygons", [])
    #     for (xs, ys) in selections:
    #         c = SelectionCurve("")
    #         c.setData(xs,ys)
    #         c.attach(self.graph)
    #         self.graph.selectionCurveList.append(c)

    # ##############################################################################################################################################################
    # SCATTERPLOT SIGNALS
    # ##############################################################################################################################################################

    def resetGraphData(self):
        self.graph.rescale_data()
        self.majorUpdateGraph()

    # receive new data and update all fields
    def setData(self, data):
        if data is not None and (len(data) == 0 or len(data.domain) == 0):
            data = None
        if self.data and data and self.data.checksum() == data.checksum():
            return    # check if the new data set is the same as the old one

        self.closeContext()
        sameDomain = self.data and data and data.domain.checksum() == self.data.domain.checksum() # preserve attribute choice if the domain is the same
        self.data = data

        # Temporary hack for SqlTables, which don't have X and Y
        # TODO: adapt scatter plot to work on SqlTables (avoid use of X and Y)
        if type(self.data) is SqlTable:
            self.data.X = numpy.zeros((len(self.data), len(self.data.domain.attributes)))
            self.data.Y = numpy.zeros((len(self.data), len(self.data.domain.class_vars)))
            for (i, row) in enumerate(data):
                self.data.X[i] = [row[attr] for attr in self.data.domain.attributes]
                if self.data.domain.class_vars:
                    self.data.Y[i] = [row[cv] for cv in self.data.domain.class_vars]

        # self.vizrank.clearResults()
        if not sameDomain:
            self.initAttrValues()
        self.graph.insideColors = None
        self.classificationResults = None
        self.outlierValues = None
        self.openContext(self.data)

    # set an example table with a data subset subset of the data. if called by a visual classifier, the update parameter will be 0
    def setSubsetData(self, subsetData):
        self.subsetData = subsetData
        # self.vizrank.clearArguments()

    # this is called by OWBaseWidget after setData and setSubsetData are called. this way the graph is updated only once
    def handleNewSignals(self):
        self.graph.setData(self.data, self.subsetData)
        # self.vizrank.resetDialog()
        if self.attributeSelectionList and 0 not in [self.graph.attribute_name_index.has_key(attr) for attr in self.attributeSelectionList]:
            self.attrX = self.attributeSelectionList[0]
            self.attrY = self.attributeSelectionList[1]
        self.attributeSelectionList = None
        self.updateGraph()
        self.sendSelections()

    # receive information about which attributes we want to show on x and y axis
    def setShownAttributes(self, list):
        if list and len(list[:2]) == 2:
            self.attributeSelectionList = list[:2]
        else:
            self.attributeSelectionList = None


    # TODO: Add support for visualizing ExperimentResults or remove code for it
    # # visualize the results of the classification
    # def setTestResults(self, results):
    #     self.classificationResults = None
    #     if isinstance(results, ExperimentResults) and len(results.results) > 0 and len(results.results[0].probabilities) > 0:
    #         self.classificationResults = [results.results[i].probabilities[0][results.results[i].actualClass] for i in range(len(results.results))]
    #         self.classificationResults = (self.classificationResults, "Probability of correct classification = %.2f%%")


    # set the learning method to be used in VizRank
    def setVizRankLearner(self, learner):
        self.vizrank.externalLearner = learner

    # send signals with selected and unselected examples as two datasets
    def sendSelections(self):
        (selected, unselected) = self.graph.getSelectionsAsExampleTables([self.attrX, self.attrY])
        self.send("Selected Data", selected)
        self.send("Other Data", unselected)
        print('\nselected data:\n', selected)
        print('unselected data:\n', unselected)


    # ##############################################################################################################################################################
    # CALLBACKS FROM VIZRANK DIALOG
    # ##############################################################################################################################################################

    def showSelectedAttributes(self):
        val = self.vizrank.getSelectedProjection()
        if not val: return
        if self.data.domain.class_var:
            self.attrColor = self.data.domain.class_var.name
        self.majorUpdateGraph(val[3])

    # ##############################################################################################################################################################
    # ATTRIBUTE SELECTION
    # ##############################################################################################################################################################

    def getShownAttributeList(self):
        return [self.attrX, self.attrY]

    def initAttrValues(self):
        self.attrXCombo.clear()
        self.attrYCombo.clear()
        self.attrColorCombo.clear()
        self.attrLabelCombo.clear()
        self.attrShapeCombo.clear()
        self.attrSizeCombo.clear()

        if not self.data: return

        self.attrColorCombo.addItem("(Same color)")
        self.attrLabelCombo.addItem("(No labels)")
        self.attrShapeCombo.addItem("(Same shape)")
        self.attrSizeCombo.addItem("(Same size)")

        #labels are usually chosen from meta variables, put them on top
        for metavar in self.data.domain._metas:
            self.attrLabelCombo.addItem(self.icons[metavar.var_type], metavar.name)

        contList = []
        discList = []
        for attr in self.data.domain:
            if attr.var_type in [VarTypes.Discrete, VarTypes.Continuous]:
                self.attrXCombo.addItem(self.icons[attr.var_type], attr.name)
                self.attrYCombo.addItem(self.icons[attr.var_type], attr.name)
                self.attrColorCombo.addItem(self.icons[attr.var_type], attr.name)
                self.attrSizeCombo.addItem(self.icons[attr.var_type], attr.name)
            if attr.var_type == VarTypes.Discrete:
                self.attrShapeCombo.addItem(self.icons[attr.var_type], attr.name)
            self.attrLabelCombo.addItem(self.icons[attr.var_type], attr.name)

        self.attrX = str(self.attrXCombo.itemText(0))
        if self.attrYCombo.count() > 1: self.attrY = str(self.attrYCombo.itemText(1))
        else:                           self.attrY = str(self.attrYCombo.itemText(0))

        if self.data.domain.class_var and self.data.domain.class_var.var_type in [VarTypes.Discrete, VarTypes.Continuous]:
            self.attrColor = self.data.domain.class_var.name
        else:
            self.attrColor = ""
        self.attrShape = ""
        self.attrSize= ""
        self.attrLabel = ""

    def majorUpdateGraph(self, attrList = None, insideColors = None, **args):
        self.graph.clear_selection()
        self.updateGraph(attrList, insideColors, **args)

    def updateGraph(self, attrList = None, insideColors = None, **args):
        self.graph.zoomStack = []
        if not self.graph.have_data:
            return

        if attrList and len(attrList) == 2:
            self.attrX = attrList[0]
            self.attrY = attrList[1]

        # if self.graph.dataHasDiscreteClass and (self.vizrank.showKNNCorrectButton.isChecked() or self.vizrank.showKNNWrongButton.isChecked()):
        #     kNNExampleAccuracy, probabilities = self.vizrank.kNNClassifyData(self.graph.createProjectionAsExampleTable([self.graph.attributeNameIndex[self.attrX], self.graph.attributeNameIndex[self.attrY]]))
        #     if self.vizrank.showKNNCorrectButton.isChecked(): kNNExampleAccuracy = ([1.0 - val for val in kNNExampleAccuracy], "Probability of wrong classification = %.2f%%")
        #     else: kNNExampleAccuracy = (kNNExampleAccuracy, "Probability of correct classification = %.2f%%")
        # else:
        kNNExampleAccuracy = None

        self.graph.insideColors = insideColors or self.classificationResults or kNNExampleAccuracy or self.outlierValues
        self.graph.updateData(self.attrX, self.attrY, self.attrColor, self.attrShape, self.attrSize, self.attrLabel)


    # ##############################################################################################################################################################
    # SCATTERPLOT SETTINGS
    # ##############################################################################################################################################################
    def saveSettings(self):
        OWWidget.saveSettings(self)
        # self.vizrank.saveSettings()

    #update status on progress bar - gets called by OWScatterplotGraph
    def updateProgress(self, current, total):
        self.progressBar.setTotalSteps(total)
        self.progressBar.setProgress(current)

    def setShowGridlines(self):
        self.graph.enableGridXB(self.showGridlines)
        self.graph.enableGridYL(self.showGridlines)

    def selectionChanged(self):
        self.zoomSelectToolbar.buttons[OWPlotGUI.SendSelection].setEnabled(not self.autoSendSelection)
        if self.autoSendSelection:
            self.sendSelections()

    def setColors(self):
        dlg = self.createColorDialog()
        if dlg.exec_():
            self.colorSettings = dlg.getColorSchemas()
            self.selectedSchemaIndex = dlg.selectedSchemaIndex
            self.graph.contPalette = dlg.getContinuousPalette("contPalette")
            self.graph.discPalette = dlg.getDiscretePalette("discPalette")
            self.graph.setCanvasBackground(dlg.getColor("Canvas"))
            self.graph.setGridColor(dlg.getColor("Grid"))
            self.updateGraph()

    def createColorDialog(self):
        c = ColorPaletteDlg(self, "Color Palette")
        c.createDiscretePalette("discPalette", "Discrete Palette")
        c.createContinuousPalette("contPalette", "Continuous Palette")
        box = c.createBox("otherColors", "Other Colors")
        c.createColorButton(box, "Canvas", "Canvas color", self.graph.color(OWPalette.Canvas))
        box.layout().addSpacing(5)
        c.createColorButton(box, "Grid", "Grid color", self.graph.color(OWPalette.Grid))
        box.layout().addSpacing(5)
        c.setColorSchemas(self.colorSettings, self.selectedSchemaIndex)
        return c

    def closeEvent(self, ce):
        # self.vizrank.close()
        OWWidget.closeEvent(self, ce)


    def sendReport(self):
        self.startReport("%s [%s - %s]" % (self.windowTitle(), self.attrX, self.attrY))
        self.reportSettings("Visualized attributes",
                            [("X", self.attrX),
                             ("Y", self.attrY),
                             self.attrColor and ("Color", self.attrColor),
                             self.attrLabel and ("Label", self.attrLabel),
                             self.attrShape and ("Shape", self.attrShape),
                             self.attrSize and ("Size", self.attrSize)])
        self.reportSettings("Settings",
                            [("Symbol size", self.graph.pointWidth),
                             ("Transparency", self.graph.alphaValue),
                             ("Jittering", self.graph.jitter_size),
                             ("Jitter continuous attributes", gui.YesNo[self.graph.jitter_continuous])])
        self.reportSection("Graph")
        self.reportImage(self.graph.saveToFileDirect, QSize(400, 400))

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWScatterPlotQt()
    ow.show()
    data = Orange.data.Table(r"iris.tab")
    ow.setData(data)
    #ow.setData(orange.ExampleTable("wine.tab"))
    ow.handleNewSignals()
    a.exec_()
    #save settings
    ow.saveSettings()
