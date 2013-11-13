import sys

from PyQt4.QtCore import SIGNAL, QSize
from PyQt4.QtGui import QApplication

from Orange.canvas.registry.description import Default
import Orange.data
from Orange.data import Table
from Orange.widgets.gui import attributeIconDict
from Orange.widgets.settings import DomainContextHandler
from Orange.widgets.utils.colorpalette import ColorPaletteDlg, ColorPaletteGenerator
from Orange.widgets.utils.plot import xBottom, OWPalette
from Orange.widgets.visualize.owparallelgraph import OWParallelGraph
from Orange.widgets.visualize.owviswidget import OWVisWidget
from Orange.widgets.widget import OWWidget, AttributeList
from Orange.widgets import gui


class OWParallelCoordinatesQt(OWVisWidget):
    settingsList = ["graph.jitterSize", "graph.showDistributions",
                    "graph.showAttrValues",
                    "graph.useSplines", "graph.alphaValue", "graph.alphaValue2", "graph.show_legend",
                    "autoSendSelection",
                    "toolbarSelection", "graph.showStatistics", "colorSettings", "selectedSchemaIndex",
                    "showAllAttributes"]
    jitterSizeNums = [0, 2, 5, 10, 15, 20, 30]
    contextHandlers = {"": DomainContextHandler("", [
        ContextField("shownAttributes", DomainContextHandler.RequiredList, selected="selectedShown",
                     reservoir="hiddenAttributes")])}

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, "Parallel Coordinates (Qt)", TRUE)

        #add a graph widget
        self.graph = OWParallelGraph(self, self.mainArea)
        self.mainArea.layout().addWidget(self.graph)

        self.showAllAttributes = 0

        self.inputs = [("Data", ExampleTable, self.setData, Default), ("Data Subset", ExampleTable, self.setSubsetData),
                       ("Features", AttributeList, self.setShownAttributes)]
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
        self.graph.show_legend = 0

        #GUI
        self.tabs = gui.tabWidget(self.controlArea)
        self.GeneralTab = gui.createTabPage(self.tabs, "Main")
        self.SettingsTab = gui.createTabPage(self.tabs, "Settings")

        self.createShowHiddenLists(self.GeneralTab, callback=self.updateGraph)
        self.connect(self.shownAttribsLB, SIGNAL('itemDoubleClicked(QListWidgetItem*)'), self.flipAttribute)

        # FIXME: self.zoomSelectToolbar = OWToolbars.ZoomSelectToolbar(self, self.GeneralTab, self.graph, self.autoSendSelection,
        #                                                      buttons=(1, 2, 0, 7, 8))
        # self.connect(self.zoomSelectToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)

        #connect controls to appropriate functions
        #self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        # ####################################
        # SETTINGS functionality
        box = gui.widgetBox(self.SettingsTab, "Transparency")
        gui.hSlider(box, self, 'graph.alphaValue', label="Examples: ", minValue=0, maxValue=255, step=10,
                    callback=self.updateGraph, tooltip="Alpha value used for drawing example lines")
        gui.hSlider(box, self, 'graph.alphaValue2', label="Rest:     ", minValue=0, maxValue=255, step=10,
                    callback=self.updateGraph, tooltip="Alpha value used to draw statistics, example subsets, ...")

        box = gui.widgetBox(self.SettingsTab, "Jittering Options")
        gui.comboBox(box, self, "graph.jitterSize", label='Jittering size (% of size):  ', orientation='horizontal',
                     callback=self.setJitteringSize, items=self.jitterSizeNums, sendSelectedValue=True, valueType=float)

        # visual settings
        box = gui.widgetBox(self.SettingsTab, "Visual Settings")

        gui.checkBox(box, self, 'graph.showAttrValues', 'Show attribute values', callback=self.updateGraph)
        gui.checkBox(box, self, 'graph.useSplines', 'Show splines', callback=self.updateGraph,
                     tooltip="Show lines using splines")

        self.graph.gui.show_legend_check_box(box)

        box = gui.widgetBox(self.SettingsTab, "Axis Distance")
        resizeColsBox = gui.widgetBox(box, 0, "horizontal", 0)
        gui.label(resizeColsBox, self, "Increase/decrease distance: ")
        gui.toolButton(resizeColsBox, self, "+", callback=self.increaseAxesDistance,
                       tooltip="Increase the distance between the axes", width=30, height=20)
        gui.toolButton(resizeColsBox, self, "-", callback=self.decreaseAxesDistance,
                       tooltip="Decrease the distance between the axes", width=30, height=20)
        gui.rubber(resizeColsBox)
        gui.checkBox(box, self, "graph.autoUpdateAxes", "Auto scale X axis",
                     tooltip="Auto scale X axis to show all visualized attributes", callback=self.updateGraph)

        box = gui.widgetBox(self.SettingsTab, "Statistical Information")
        gui.comboBox(box, self, "graph.showStatistics", label="Statistics: ", orientation="horizontal", labelWidth=90,
                     items=["No statistics", "Means, deviations", "Median, quartiles"], callback=self.updateGraph,
                     sendSelectedValue=False, valueType=int)
        gui.comboBox(box, self, "middleLabels", label="Middle labels: ", orientation="horizontal", labelWidth=90,
                     items=["No labels", "Correlations", "VizRank"], callback=self.updateGraph,
                     tooltip="The information do you wish to view on top in the middle of coordinate axes",
                     sendSelectedValue=True, valueType=str)
        gui.checkBox(box, self, 'graph.showDistributions', 'Show distributions', callback=self.updateGraph,
                     tooltip="Show bars with distribution of class values (only for discrete attributes)")

        box = gui.widgetBox(self.SettingsTab, "Colors", orientation="horizontal")
        gui.button(box, self, "Set colors", self.setColors,
                   tooltip="Set the canvas background color and color palette for coloring continuous variables")

        box = gui.widgetBox(self.SettingsTab, "Auto Send Selected Data When...")
        gui.checkBox(box, self, 'autoSendSelection', 'Adding/Removing selection areas',
                     callback=self.selectionChanged,
                     tooltip="Send selected data whenever a selection area is added or removed")
        gui.checkBox(box, self, 'graph.sendSelectionOnUpdate', 'Moving/Resizing selection areas',
                     tooltip="Send selected data when a user moves or resizes an existing selection area")
        self.graph.autoSendSelectionCallback = self.selectionChanged

        self.SettingsTab.layout().addStretch(100)
        self.icons = attributeIconDict

        dlg = self.createColorDialog()
        self.graph.contPalette = dlg.getContinuousPalette("contPalette")
        self.graph.discPalette = dlg.getDiscretePalette("discPalette")
        self.graph.setCanvasBackground(dlg.getColor("Canvas"))
        #[self.zoomSelectToolbar.actionZooming, self.zoomSelectToolbar.actionRectangleSelection,
        # self.zoomSelectToolbar.actionPolygonSelection][self.toolbarSelection](*[])
        self.cbShowAllAttributes()

        self.resize(900, 700)

    def flipAttribute(self, item):
        if self.graph.flipAttribute(str(item.text())):
            self.updateGraph()
            self.information(0)
        else:
            self.information(0, "Didn't flip the attribute. To flip a continuous "
                                "attribute uncheck 'Global value scaling' checkbox.")

    def updateGraph(self, *args):
        attrs = self.getShownAttributeList()
        self.graph.updateData(attrs)


    def increaseAxesDistance(self):
        m, M = self.graph.bounds_for_axis(xBottom)
        if (M - m) == 0:
            return # we have not yet updated the axes (self.graph.updateAxes())
        self.graph.setAxisScale(xBottom, m, M - (M - m) / 10., 1)
        self.graph.replot()

    def decreaseAxesDistance(self):
        m, M = self.graph.bounds_for_axis(xBottom)
        if (M - m) == 0:
            return # we have not yet updated the axes (self.graph.updateAxes())

        self.graph.setAxisScale(xBottom, m, min(len(self.graph.visualizedAttributes) - 1, M + (M - m) / 10.), 1)
        self.graph.replot()

    # ------------- SIGNALS --------------------------
    # receive new data and update all fields
    def setData(self, data):
        if data and (len(data) == 0 or len(data.domain) == 0):
            data = None
        if self.data is not None and data is not None and self.data.checksum() == data.checksum():
            return # check if the new data set is the same as the old one

        self.closeContext()
        same_domain = self.data and data and data.domain.checksum() == self.data.domain.checksum() # preserve attribute choice if the domain is the same
        self.projections = None
        self.correlationDict = {}
        self.data = data
        if not same_domain:
            self.setShownAttributeList(self.attributeSelectionList)
        self.openContext(self.data)
        self.resetAttrManipulation()


    def setSubsetData(self, subData):
        self.subsetData = subData


    # attribute selection signal - list of attributes to show
    def setShownAttributes(self, attributeSelectionList):
        self.attributeSelectionList = attributeSelectionList


    # this is called by OWBaseWidget after setData and setSubsetData are called. this way the graph is updated only once
    def handleNewSignals(self):
        self.graph.setData(self.data, self.subsetData)
        if self.attributeSelectionList and 0 not in [attr in self.graph.attributeNameIndex for attr in
                                                     self.attributeSelectionList]:
            self.setShownAttributeList(self.attributeSelectionList)
        else:
            self.setShownAttributeList()
        self.attributeSelectionList = None
        self.updateGraph()
        self.sendSelections()


    def sendShownAttributes(self, attrList=None):
        if attrList == None:
            attrList = self.getShownAttributeList()
        self.send("Features", attrList)

    def selectionChanged(self):
        #FIXME:
        #self.zoomSelectToolbar.buttonSendSelections.setEnabled(not self.autoSendSelection)
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
        c = ColorPaletteDlg(self, "Color Palette")
        c.createDiscretePalette("discPalette", "Discrete Palette")
        c.createContinuousPalette("contPalette", "Continuous Palette")
        box = c.createBox("otherColors", "Other Colors")
        c.createColorButton(box, "Canvas", "Canvas color", self.graph.color(OWPalette.Canvas))
        c.setColorSchemas(self.colorSettings, self.selectedSchemaIndex)
        return c

    def saveSettings(self):
        OWWidget.saveSettings(self)

    def closeEvent(self, ce):
        OWWidget.closeEvent(self, ce)

    def sendReport(self):
        self.reportImage(self.graph.saveToFileDirect, QSize(500, 500))


#test widget appearance
if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWParallelCoordinates()
    ow.show()
    ow.graph.discPalette = ColorPaletteGenerator(rgbColors=[(127, 201, 127), (190, 174, 212), (253, 192, 134)])
    data = Orange.data.Table("iris")
    ow.setData(data)
    ow.handleNewSignals()

    a.exec_()
