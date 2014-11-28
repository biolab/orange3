import os
import sys
from collections import defaultdict
from functools import reduce
from itertools import product
from math import sqrt

import numpy
from PyQt4.QtCore import QPoint, Qt, QRectF
from PyQt4.QtGui import (QGraphicsRectItem, QGraphicsView, QColor,
                         QGraphicsScene, QPainter, QIcon, QDialog, QPen,
                         QVBoxLayout, QListWidget, QSizePolicy, QApplication,
                         QGraphicsTextItem, QBrush, QGraphicsLineItem,
                         QGraphicsEllipseItem)

from Orange.widgets.settings import (Setting, DomainContextHandler,
                                     ContextSetting)
from Orange.canvas.utils import environ
from Orange.classification import Fitter
from Orange.data import Table, Variable, filter, DiscreteVariable, ContinuousVariable
from Orange.data.discretization import DiscretizeTable
from Orange.data.sql.table import SqlTable, LARGE_TABLE, DEFAULT_SAMPLE_TIME
from Orange.feature.discretization import EqualFreq
from Orange.statistics.distribution import get_distribution
from Orange.widgets import gui
from Orange.widgets.settings import DomainContextHandler
from Orange.widgets.utils import getHtmlCompatibleString
from Orange.widgets.utils.colorpalette import ColorPaletteDlg, DefaultRGBColors
from Orange.widgets.utils.scaling import get_variable_values_sorted
from Orange.widgets.widget import OWWidget, Default


PEARSON = 0
CLASS_DISTRIBUTION = 1

BOTTOM = 0
LEFT = 1
TOP = 2
RIGHT = 3

# using function with same name from owtools.py
# def get_variable_values_sorted(param):
#     if hasattr(param, "values"):
#         return param.values
#     return []


class SelectionRectangle(QGraphicsRectItem):
    pass


class MosaicSceneView(QGraphicsView):
    def __init__(self, widget, *args):
        QGraphicsView.__init__(self, *args)
        self.widget = widget
        self.bMouseDown = False
        self.mouseDownPosition = QPoint(0, 0)
        self.tempRect = None

    # mouse button was pressed
    def mousePressEvent(self, ev):
        QGraphicsView.mousePressEvent(self, ev)
        self.mouseDownPosition = QPoint(ev.pos().x(), ev.pos().y())
        self.bMouseDown = True
        self.mouseMoveEvent(ev)

    # mouse button was pressed and mouse is moving ######################
    def mouseMoveEvent(self, ev):
        QGraphicsView.mouseMoveEvent(self, ev)
        if ev.button() == Qt.RightButton:
            return

        if not self.bMouseDown:
            if self.tempRect:
                self.scene().removeItem(self.tempRect)
                self.tempRect = None
        else:
            if not self.tempRect:
                self.tempRect = SelectionRectangle(None, self.scene())
            rect = QRectF(min(self.mouseDownPosition.x(), ev.pos().x()),
                          min(self.mouseDownPosition.y(), ev.pos().y()),
                          max(abs(self.mouseDownPosition.x() - ev.pos().x()), 1),
                          max(abs(self.mouseDownPosition.y() - ev.pos().y()), 1))
            self.tempRect.setRect(rect)


    # mouse button was released #########################################
    def mouseReleaseEvent(self, ev):
        self.bMouseDown = False

        if ev.button() == Qt.RightButton:
            self.widget.removeLastSelection()
        elif self.tempRect:
            self.widget.addSelection(self.tempRect)
            self.scene().removeItem(self.tempRect)
            self.tempRect = None


class OWMosaicDisplay(OWWidget):
    name = "Mosaic Display"
    description = "Shows mosaic displays"
    icon = "icons/MosaicDisplay.svg"

    inputs = [("Data", Table, "setData", Default),
              ("Data Subset", Table, "setSubsetData")]
    outputs = [("Selected Data", Table), ("Learner", Fitter)]

    settingsHandler = DomainContextHandler()
    show_apriori_distribution_lines = Setting(False)
    show_apriori_distribution_boxes = Setting(True)
    use_boxes = Setting(True)
    interior_coloring = Setting(0)
    color_settings = Setting(None)
    selected_schema_index = Setting(0)
    show_subset_data_boxes = Setting(True)
    remove_unused_values = Setting(True)
    variable1 = ContextSetting("")
    variable2 = ContextSetting("")
    variable3 = ContextSetting("")
    variable4 = ContextSetting("")

    interior_coloring_opts = ["Pearson residuals",
                            "Class distribution"]
    subboxesOpts = ["Expected distribution",
                    "Apriori distribution"]

    _apriori_pen_color = QColor(255, 255, 255, 128)
    _box_size = 5
    _cellspace = 4

    def __init__(self, parent=None):
        super().__init__(self, parent)

        #set default settings
        self.data = None
        self.unprocessed_subset_data = None
        self.subset_data = None
        self.names = []  # class values

        self.exploreAttrPermutations = 0

        self.attributeNameOffset = 30
        self.attributeValueOffset = 15
        self.residuals = []  # residual values if the residuals are visualized
        self.aprioriDistributions = []
        self.colorPalette = None
        self.permutationDict = {}
        self.manualAttributeValuesDict = {}
        self.conditionalDict = None
        self.conditionalSubsetDict = None
        self.activeRule = None

        self.selectionRectangle = None
        self.selectionConditionsHistorically = []
        self.selectionConditions = []

        # color paletes for visualizing pearsons residuals
        self.blue_colors = [QColor(255, 255, 255), QColor(210, 210, 255),
                            QColor(110, 110, 255), QColor(0, 0, 255)]
        self.red_colors = [QColor(255, 255, 255), QColor(255, 200, 200),
                           QColor(255, 100, 100), QColor(255, 0, 0)]

        self.canvas = QGraphicsScene()
        self.canvas_view = MosaicSceneView(self, self.canvas, self.mainArea)
        self.mainArea.layout().addWidget(self.canvas_view)
        self.canvas_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.canvas_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.canvas_view.setRenderHint(QPainter.Antialiasing)
        #self.canvasView.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        #GUI
        #add controls to self.controlArea widget
        #self.controlArea.setMinimumWidth(235)

        box = gui.widgetBox(self.controlArea, "Variables")
        for i in range(1, 5):
            inbox = gui.widgetBox(box, orientation="horizontal")
            combo = gui.comboBox(inbox, self, value="variable{}".format(i),
                                 #label="Variable {}".format(i),
                                 orientation="horizontal",
                                 callback=self.updateGraphAndPermList,
                                 sendSelectedValue=True, valueType=str)

            butt = gui.button(inbox, self, "", callback=self.orderAttributeValues,
                              tooltip="Change the order of attribute values")
            butt.setFixedSize(26, 24)
            butt.setCheckable(1)
            butt.setIcon(QIcon(os.path.join(environ.widget_install_dir,
                                            "icons/Dlg_sort.png")))
            setattr(self, "sort{}".format(i), butt)
            setattr(self, "attr{}".format(i) + "Combo", combo)

        # self.optimizationDlg = OWMosaicOptimization(self, self.signalManager)

        # optimizationButtons = gui.widgetBox(self.GeneralTab, "Dialogs", orientation="horizontal")
        # gui.button(optimizationButtons, self, "VizRank", callback=self.optimizationDlg.reshow, debuggingEnabled=0,
        #            tooltip="Find attribute combinations that will separate different classes as clearly as possible.")

        # self.collapsableWBox = gui.collapsableWidgetBox(self.GeneralTab, "Explore Attribute Permutations", self,
        #                                                 "exploreAttrPermutations",
        #                                                 callback=self.permutationListToggle)
        # self.permutationList = gui.listBox(self.collapsableWBox, self, callback=self.setSelectedPermutation)
        #self.permutationList.hide()

        box5 = gui.widgetBox(self.controlArea, "Visual Settings")
        gui.comboBox(box5, self, "interior_coloring",
                     label="Color", orientation="horizontal",
                     items=self.interior_coloring_opts,
                     callback=self.updateGraph)

        gui.checkBox(box5, self, "remove_unused_values",
                     "Remove unused attribute values")

        gui.checkBox(box5, self, 'show_apriori_distribution_lines',
                     'Show apriori distribution with lines',
                     callback=self.updateGraph)

        self.box8 = gui.widgetBox(self.controlArea, "Boxes in Cells")
        self.cb_show_subset = gui.checkBox(
            self.box8, self, 'show_subset_data_boxes',
            'Show subset data distribution', callback=self.updateGraph)
        self.cb_show_subset.setDisabled(self.subset_data is None)
        cb = gui.checkBox(self.box8, self, 'use_boxes', 'Display sub-box...',
                          callback=self.updateGraph)
        ind_box = gui.indentedBox(self.box8, sep=gui.checkButtonOffsetHint(cb))
        gui.comboBox(ind_box, self, 'show_apriori_distribution_boxes',
                     items=self.subboxesOpts, callback=self.updateGraph)

        hbox = gui.widgetBox(self.controlArea, "Colors", addSpace=1)
        gui.button(hbox, self, "Set Colors", self.setColors,
                   tooltip="Set the color palette for class values")

        #self.box6.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        self.controlArea.layout().addStretch(1)

        # self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFileCanvas)
        self.icons = gui.attributeIconDict
        self.resize(830, 550)

        # self.VizRankLearner = MosaicTreeLearner(self.optimizationDlg)
        # self.send("Learner", self.VizRankLearner)

        # self.wdChildDialogs = [self.optimizationDlg]  # used when running widget debugging

        # self.collapsableWBox.updateControls()
        dlg = self.createColorDialog()
        self.colorPalette = dlg.getDiscretePalette("discPalette")
        self.selectionColorPalette = [QColor(*col) for col in DefaultRGBColors]

        gui.rubber(self.controlArea)


    def permutationListToggle(self):
        if self.exploreAttrPermutations:
            self.updateGraphAndPermList()

    def setSelectedPermutation(self):
        newRow = self.permutationList.currentRow()
        if self.permutationList.count() > 0 and self.bestPlacements and newRow < len(self.bestPlacements):
            self.removeAllSelections()
            val, attrList, valueOrder = self.bestPlacements[newRow]
            if len(attrList) > 0: self.variable1 = attrList[0]
            if len(attrList) > 1: self.variable2 = attrList[1]
            if len(attrList) > 2: self.variable3 = attrList[2]
            if len(attrList) > 3: self.variable4 = attrList[3]
            self.updateGraph(
                customValueOrderDict=dict([(attrList[i], tuple(valueOrder[i])) for i in range(len(attrList))]))

    def orderAttributeValues(self):
        attr = None
        if self.sort1.isChecked():
            attr = self.variable1
        elif self.sort2.isChecked():
            attr = self.variable2
        elif self.sort3.isChecked():
            attr = self.variable3
        elif self.sort4.isChecked():
            attr = self.variable4

        if self.data and attr != "" and attr != "(None)":
            dlg = SortAttributeValuesDlg(attr,
                                         self.manualAttributeValuesDict.get(attr, None) or get_variable_values_sorted(
                                             self.data.domain[attr]))
            if dlg.exec_() == QDialog.Accepted:
                self.manualAttributeValuesDict[attr] = [str(dlg.attributeList.item(i).text()) for i in
                                                        range(dlg.attributeList.count())]

        for control in [self.sort1, self.sort2, self.sort3, self.sort4]:
            control.setChecked(0)
        self.updateGraph()

    # initialize combo boxes with discrete attributes
    def initCombos(self, data):
        for combo in [self.attr1Combo, self.attr2Combo, self.attr3Combo, self.attr4Combo]:
            combo.clear()

        if data == None: return

        self.attr2Combo.addItem("(None)")
        self.attr3Combo.addItem("(None)")
        self.attr4Combo.addItem("(None)")

        for attr in data.domain:
            if isinstance(attr, DiscreteVariable):
                for combo in [self.attr1Combo, self.attr2Combo, self.attr3Combo, self.attr4Combo]:
                    combo.addItem(self.icons[attr], attr.name)

        if self.attr1Combo.count() > 0:
            self.variable1 = str(self.attr1Combo.itemText(0))
            self.variable2 = str(self.attr2Combo.itemText(0 + 2 * (self.attr2Combo.count() > 2)))
        self.variable3 = str(self.attr3Combo.itemText(0))
        self.variable4 = str(self.attr4Combo.itemText(0))

    #  when we resize the widget, we have to redraw the data
    def resizeEvent(self, e):
        OWWidget.resizeEvent(self, e)
        self.updateGraph()

    def showEvent(self, ev):
        OWWidget.showEvent(self, ev)
        self.updateGraph()

    def closeEvent(self, ce):
        # self.optimizationDlg.hide()
        QDialog.closeEvent(self, ce)

    # ------------- SIGNALS --------------------------
    # # DATA signal - receive new data and update all fields
    def setData(self, data):
        if type(data) == SqlTable and data.approx_len() > LARGE_TABLE:
            data = data.sample_time(DEFAULT_SAMPLE_TIME)

        self.closeContext()
        self.data = data
        self.bestPlacements = None
        self.manualAttributeValuesDict = {}
        self.attributeValuesDict = {}
        self.information([0, 1, 2])

        if not self.data:
            return

        if any(isinstance(attr, ContinuousVariable)
               for attr in self.data.domain):
            self.information(0, "Data contains continuous variables. " +
                             "Discretize the data to use them.")

        """ TODO: check
        if data.has_missing_class():
            self.information(1, "Examples with missing classes were removed.")
        if self.removeUnusedValues and len(data) != len(self.data):
            self.information(2, "Unused attribute values were removed.")
        """

        if isinstance(self.data.domain.class_var, DiscreteVariable):
            self.interior_coloring = CLASS_DISTRIBUTION
            self.colorPalette.set_number_of_colors(
                len(self.data.domain.class_var.values))
        else:
            self.interior_coloring = PEARSON

        self.initCombos(self.data)
        self.openContext(self.data)

        # if we first received subset data
        # we now have to call setSubsetData to process it
        if self.unprocessed_subset_data:
            self.setSubsetData(self.unprocessed_subset_data)
            self.unprocessed_subset_data = None

    def setSubsetData(self, data):
        if not self.data:
            self.unprocessed_subset_data = data
            self.warning(10)
        else:
            try:
                self.subset_data = data.from_table(self.data.domain, data)
                self.warning(10)
            except:
                self.subset_data = None
                self.warning(10, data and "'Data' and 'Data Subset'" +
                             " do not have compatible domains." or "")
        self.cb_show_subset.setDisabled(self.subset_data is None)



    # this is called by OWBaseWidget after setData and setSubsetData are called.
    # this way the graph is updated only once
    def handleNewSignals(self):
        self.updateGraphAndPermList()

    # ------------------------------------------------

    def setShownAttributes(self, attrList, **args):
        if not attrList: return
        self.variable1 = attrList[0]

        if len(attrList) > 1:
            self.variable2 = attrList[1]
        else:
            self.variable2 = "(None)"

        if len(attrList) > 2:
            self.variable3 = attrList[2]
        else:
            self.variable3 = "(None)"

        if len(attrList) > 3:
            self.variable4 = attrList[3]
        else:
            self.variable4 = "(None)"

        self.attributeValuesDict = args.get("customValueOrderDict", None)
        self.updateGraphAndPermList()

    def getShownAttributeList(self):
        attrList = [self.variable1, self.variable2, self.variable3, self.variable4]
        while "(None)" in attrList: attrList.remove("(None)")
        while "" in attrList:       attrList.remove("")
        return attrList

    def updateGraphAndPermList(self, **args):
        self.removeAllSelections()
        # self.permutationList.clear()

        if self.exploreAttrPermutations:
            attrList = self.getShownAttributeList()
            if not getattr(self, "bestPlacements", []) or 0 in [attr in self.bestPlacements[0][1] for attr in
                                                                attrList]:  # we might have bestPlacements for a different set of attributes
                self.setStatusBarText(
                    "Evaluating different attribute permutations. You can stop evaluation by opening VizRank dialog and pressing 'Stop optimization' button.")
                self.bestPlacements = self.optimizationDlg.optimizeCurrentAttributeOrder(attrList, updateGraph=0)
                self.setStatusBarText("")

            if self.bestPlacements:
                self.permutationList.addItems(
                    ["%.2f - %s" % (val, attrs) for (val, attrs, order) in self.bestPlacements])
                attrList, valueOrder = self.bestPlacements[0][1], self.bestPlacements[0][2]
                self.attributeValuesDict = dict([(attrList[i], tuple(valueOrder[i])) for i in range(len(attrList))])

        self.updateGraph(**args)

    # ############################################################################
    # updateGraph - gets called every time the graph has to be updated
    def updateGraph(self, data=-1, subsetData=-1, attrList=-1, **args):
        # do we want to erase previous diagram?
        if args.get("erasePrevious", 1):
            for item in list(self.canvas.items()):
                if not isinstance(item, SelectionRectangle):
                    self.canvas.removeItem(item)  # remove all canvas items, except SelectionCurves
            self.names = []

        if data == -1:
            data = self.data

        if subsetData == -1:
            subsetData = self.subset_data

        if attrList == -1:
            attrList = [self.variable1, self.variable2, self.variable3, self.variable4]

        if data == None: return

        while "(None)" in attrList: attrList.remove("(None)")
        while "" in attrList:       attrList.remove("")
        if attrList == []:
            return

        selectList = attrList
        if type(data) == SqlTable and data.domain.class_var:
            cv = data.domain.class_var # shranim class_var, ker se v naslednji vrstici zbrise (v primeru da je SqlTable)
            data = data[:, attrList + [data.domain.class_var]]
            data.domain.class_var = cv
        elif data.domain.class_var:
            cv = data.domain.class_var # shranim class_var, ker se v naslednji vrstici zbrise (v primeru da si izbral atribut, ki je class_var)
            name = data.name
            data = data[:, attrList + [data.domain.class_var]]
            data.domain.class_var = cv
            data.name = name
        else:
            data = data[:, attrList]
        # TODO: preveri kaj je stem
        # data = Preprocessor_dropMissing(data)

        if len(data) == 0:
            self.warning(5,
                         "No data instances with valid values for currently visualized attributes.")
            return
        else:
            self.warning(5)

        self.aprioriDistributions = []
        if self.interior_coloring == PEARSON:
            self.aprioriDistributions = [get_distribution(data, attr) for attr in attrList]

        if args.get("positions"):
            xOff, yOff, squareSize = args.get("positions")
        else:
            # get the maximum width of rectangle
            xOff = 50
            width = 50
            if len(attrList) > 1:
                text = OWCanvasText(self.canvas, attrList[1], bold=1, show=0)
                width = text.boundingRect().height() + 30 + 20
                xOff = width
                if len(attrList) == 4:
                    text = OWCanvasText(self.canvas, attrList[3], bold=1, show=0)
                    width += text.boundingRect().height() + 30 + 20

            # get the maximum height of rectangle
            height = 90
            yOff = 40
            squareSize = min(self.canvas_view.width() - width - 20, self.canvas_view.height() - height - 20)

        if squareSize < 0: return  # canvas is too small to draw rectangles
        self.canvas_view.setSceneRect(0, 0, self.canvas_view.width(), self.canvas_view.height())

        self.legend = {}  # dictionary that tells us, for what attributes did we already show the legend
        for attr in attrList:
            self.legend[attr] = 0

        self.drawnSides = dict([(0, 0), (1, 0), (2, 0), (3, 0)])
        self.drawPositions = {}

        if not getattr(self, "attributeValuesDict", None):
            self.attributeValuesDict = self.manualAttributeValuesDict

        # compute distributions

        self.conditionalDict = self.getConditionalDistributions(data, attrList)
        self.conditionalDict[""] = len(data)
        self.conditionalSubsetDict = None
        if subsetData:
            self.conditionalSubsetDict = self.getConditionalDistributions(subsetData, attrList)
            self.conditionalSubsetDict[""] = len(subsetData)

        # draw rectangles
        self.DrawData(attrList, (xOff, xOff + squareSize), (yOff, yOff + squareSize), 0, "", len(attrList), **args)
        if args.get("drawLegend", 1):
            self.DrawLegend(data, (xOff, xOff + squareSize), (yOff, yOff + squareSize))  # draw class legend

        if args.get("drillUpdateSelection", 1):
            # self.optimizationDlg.mtUpdateState()
            pass

            # self.canvas.update()

    # create a dictionary with all possible pairs of "combination-of-attr-values" : count
    ## TODO: this function is used both in owmosaic and owsieve --> where to put it?
    def getConditionalDistributions(self, data, attrs):
        cond_dist = defaultdict(int)
        all_attrs = [data.domain[a] for a in attrs]
        if data.domain.class_var is not None:
            all_attrs.append(data.domain.class_var)

        for i in range(1, len(all_attrs)+1):
            attr = all_attrs[:i]
            if type(data) == SqlTable:
                # make all possible pairs of attributes + class_var
                attr = [a.to_sql() for a in attr]
                fields = attr + ["COUNT(*)"]
                query = data._sql_query(fields, group_by=attr)
                with data._execute_sql_query(query) as cur:
                    res = cur.fetchall()
                for r in res:
                    str_values =[a.repr_val(a.to_val(x)) for a, x in zip(all_attrs, r[:-1])]
                    str_values = [x if x != '?' else 'None' for x in str_values]
                    cond_dist['-'.join(str_values)] = r[-1]
            else:
                for indices in product(*(range(len(a.values)) for a in attr)):
                    vals = []
                    conditions = []
                    for k, ind in enumerate(indices):
                        vals.append(attr[k].values[ind])
                        fd = filter.FilterDiscrete(column=attr[k], values=[attr[k].values[ind]])
                        conditions.append(fd)
                    filt = filter.Values(conditions)
                    filtdata = filt(data)
                    cond_dist['-'.join(vals)] = len(filtdata)
        return cond_dist


    # ############################################################################
    # ############################################################################

    ##  DRAW DATA - draw rectangles for attributes in attrList inside rect (x0,x1), (y0,y1)
    def DrawData(self, attrList, x0_x1, y0_y1, side, condition, totalAttrs, used_attrs=[], used_vals=[],
                 attrVals="", **args):
        x0, x1 = x0_x1
        y0, y1 = y0_y1
        if self.conditionalDict[attrVals] == 0:
            self.addRect(x0, x1, y0, y1, "", used_attrs, used_vals, attrVals=attrVals)
            self.DrawText(side, attrList[0], (x0, x1), (y0, y1), totalAttrs, used_attrs, used_vals,
                          attrVals)  # store coordinates for later drawing of labels
            return

        attr = attrList[0]
        edge = len(attrList) * self._cellspace  # how much smaller rectangles do we draw
        values = self.attributeValuesDict.get(attr, None) or get_variable_values_sorted(self.data.domain[attr])
        if side % 2: values = values[::-1]  # reverse names if necessary

        if side % 2 == 0:  # we are drawing on the x axis
            whole = max(0, (x1 - x0) - edge * (
                len(values) - 1))  # we remove the space needed for separating different attr. values
            if whole == 0: edge = (x1 - x0) / float(len(values) - 1)
        else:  # we are drawing on the y axis
            whole = max(0, (y1 - y0) - edge * (len(values) - 1))
            if whole == 0: edge = (y1 - y0) / float(len(values) - 1)

        if attrVals == "":
            counts = [self.conditionalDict[val] for val in values]
        else:
            counts = [self.conditionalDict[attrVals + "-" + val] for val in values]
        total = sum(counts)

        # if we are visualizing the third attribute and the first attribute has the last value, we have to reverse the order in which the boxes will be drawn
        # otherwise, if the last cell, nearest to the labels of the fourth attribute, is empty, we wouldn't be able to position the labels
        valRange = list(range(len(values)))
        if len(attrList + used_attrs) == 4 and len(used_attrs) == 2:
            attr1Values = self.attributeValuesDict.get(used_attrs[0], None) or get_variable_values_sorted(
                self.data.domain[used_attrs[0]])
            if used_vals[0] == attr1Values[-1]:
                valRange = valRange[::-1]

        for i in valRange:
            start = i * edge + whole * float(sum(counts[:i]) / float(total))
            end = i * edge + whole * float(sum(counts[:i + 1]) / float(total))
            val = values[i]
            htmlVal = getHtmlCompatibleString(val)
            if attrVals != "":
                newAttrVals = attrVals + "-" + val
            else:
                newAttrVals = val

            if side % 2 == 0:  # if we are moving horizontally
                if len(attrList) == 1:
                    self.addRect(x0 + start, x0 + end, y0, y1,
                                 condition + 4 * "&nbsp;" + attr + ": <b>" + htmlVal + "</b><br>", used_attrs + [attr],
                                 used_vals + [val], newAttrVals, **args)
                else:
                    self.DrawData(attrList[1:], (x0 + start, x0 + end), (y0, y1), side + 1,
                                  condition + 4 * "&nbsp;" + attr + ": <b>" + htmlVal + "</b><br>", totalAttrs,
                                  used_attrs + [attr], used_vals + [val], newAttrVals, **args)
            else:
                if len(attrList) == 1:
                    self.addRect(x0, x1, y0 + start, y0 + end,
                                 condition + 4 * "&nbsp;" + attr + ": <b> " + htmlVal + "</b><br>", used_attrs + [attr],
                                 used_vals + [val], newAttrVals, **args)
                else:
                    self.DrawData(attrList[1:], (x0, x1), (y0 + start, y0 + end), side + 1,
                                  condition + 4 * "&nbsp;" + attr + ": <b>" + htmlVal + "</b><br>", totalAttrs,
                                  used_attrs + [attr], used_vals + [val], newAttrVals, **args)

        self.DrawText(side, attrList[0], (x0, x1), (y0, y1), totalAttrs, used_attrs, used_vals, attrVals)


    ######################################################################
    ## DRAW TEXT - draw legend for all attributes in attrList and their possible values
    def DrawText(self, side, attr, x0_x1, y0_y1, totalAttrs, used_attrs, used_vals, attrVals):
        x0, x1 = x0_x1
        y0, y1 = y0_y1
        if self.drawnSides[side]: return

        # the text on the right will be drawn when we are processing visualization of the last value of the first attribute
        if side == RIGHT:
            attr1Values = self.attributeValuesDict.get(used_attrs[0], None) or get_variable_values_sorted(
                self.data.domain[used_attrs[0]])
            if used_vals[0] != attr1Values[-1]:
                return

        if not self.conditionalDict[attrVals]:
            if side not in self.drawPositions: self.drawPositions[side] = (x0, x1, y0, y1)
            return
        else:
            if side in self.drawPositions: (x0, x1, y0, y1) = self.drawPositions[
                side]  # restore the positions where we have to draw the attribute values and attribute name

        self.drawnSides[side] = 1

        values = self.attributeValuesDict.get(attr, None) or get_variable_values_sorted(self.data.domain[attr])
        if side % 2:  values = values[::-1]

        width = x1 - x0 - (side % 2 == 0) * self._cellspace * (totalAttrs - side) * (len(values) - 1)
        height = y1 - y0 - (side % 2 == 1) * self._cellspace * (totalAttrs - side) * (len(values) - 1)

        #calculate position of first attribute
        currPos = 0

        if attrVals == "":
            counts = [self.conditionalDict.get(val, 1) for val in values]
        else:
            counts = [self.conditionalDict.get(attrVals + "-" + val, 1) for val in values]
        total = sum(counts)
        if total == 0:
            counts = [1] * len(values)
            total = sum(counts)

        max_ylabel_w1 = 0
        max_ylabel_w2 = 0

        for i in range(len(values)):
            val = values[i]
            perc = counts[i] / float(total)
            if side == 0:
                OWCanvasText(self.canvas, str(val), x0 + currPos + width * 0.5 * perc, y1 + self.attributeValueOffset,
                             Qt.AlignCenter, bold=0)
            elif side == 1:
                t = OWCanvasText(self.canvas, str(val), x0 - self.attributeValueOffset, y0 + currPos + height * 0.5 * perc,
                             Qt.AlignRight | Qt.AlignVCenter, bold=0)
                max_ylabel_w1 = max(int(t.boundingRect().width()), max_ylabel_w1)
            elif side == 2:
                OWCanvasText(self.canvas, str(val), x0 + currPos + width * perc * 0.5, y0 - self.attributeValueOffset,
                             Qt.AlignCenter, bold=0)
            else:
                t = OWCanvasText(self.canvas, str(val), x1 + self.attributeValueOffset, y0 + currPos + height * 0.5 * perc,
                             Qt.AlignLeft | Qt.AlignVCenter, bold=0)
                max_ylabel_w2 = max(int(t.boundingRect().width()), max_ylabel_w2)

            if side % 2 == 0:
                currPos += perc * width + self._cellspace * (totalAttrs - side)
            else:
                currPos += perc * height + self._cellspace * (totalAttrs - side)

        if side == 0:
            OWCanvasText(self.canvas, attr, x0 + (x1 - x0) / 2, y1 + self.attributeNameOffset, Qt.AlignCenter, bold=1)
        elif side == 1:
            OWCanvasText(self.canvas, attr, max(x0 - max_ylabel_w1 - self.attributeValueOffset, 20), y0 + (y1 - y0) / 2,
                         Qt.AlignRight | Qt.AlignVCenter, bold=1, vertical=True)
        elif side == 2:
            OWCanvasText(self.canvas, attr, x0 + (x1 - x0) / 2, y0 - self.attributeNameOffset, Qt.AlignCenter, bold=1)
        else:
            OWCanvasText(self.canvas, attr, min(x1+50, x1 + max_ylabel_w2 + self.attributeValueOffset), y0 + (y1 - y0) / 2,
                         Qt.AlignLeft | Qt.AlignVCenter, bold=1, vertical=True)



    # draw a rectangle, set it to back and add it to rect list
    def addRect(self, x0, x1, y0, y1, condition="", used_attrs=[], used_vals=[], attrVals="", **args):
        if x0 == x1:
            x1 += 1
        if y0 == y1:
            y1 += 1

        if x1 - x0 + y1 - y0 == 2:
            y1 += 1  # if we want to show a rectangle of width and height 1 it doesn't show anything. in such cases we therefore have to increase size of one edge

        if ("selectionDict" in args and
            tuple(used_vals) in args["selectionDict"]):
            d = 2
            OWCanvasRectangle(self.canvas, x0 - d, y0 - d, x1 - x0 + 1 + 2 * d, y1 - y0 + 1 + 2 * d,
                              penColor=args["selectionDict"][tuple(used_vals)], penWidth=2, z=-100)

        # if we have selected a rule that contains this combination of attr values then show a kind of selection of this rectangle
        if self.activeRule and len(used_attrs) == len(self.activeRule[0]) and sum(
                [v in used_attrs for v in self.activeRule[0]]) == len(self.activeRule[0]):
            for vals in self.activeRule[1]:
                if used_vals == [vals[self.activeRule[0].index(a)] for a in used_attrs]:
                    values = list(
                        self.attributeValuesDict.get(self.data.domain.classVar.name, [])) or get_variable_values_sorted(
                        self.data.domain.class_var)
                    counts = [self.conditionalDict[attrVals + "-" + val] for val in values]
                    d = 2
                    r = OWCanvasRectangle(self.canvas, x0 - d, y0 - d, x1 - x0 + 2 * d + 1, y1 - y0 + 2 * d + 1, z=50)
                    r.setPen(QPen(self.colorPalette[counts.index(max(counts))], 2, Qt.DashLine))

        aprioriDist = ()
        pearson = None
        expected = None
        outerRect = OWCanvasRectangle(self.canvas, x0, y0, x1 - x0, y1 - y0, z=30)

        if not self.conditionalDict[attrVals]: return

        # we have to remember which conditions were new in this update so that when we right click we can only remove the last added selections
        if self.selectionRectangle != None and self.selectionRectangle.collidesWithItem(outerRect) and tuple(
                used_vals) not in self.selectionConditions:
            self.recentlyAdded = getattr(self, "recentlyAdded", []) + [tuple(used_vals)]
            self.selectionConditions = self.selectionConditions + [tuple(used_vals)]

        # show rectangle selected or not
        if tuple(used_vals) in self.selectionConditions:
            outerRect.setPen(QPen(Qt.black, 3, Qt.DotLine))

        if self.interior_coloring == CLASS_DISTRIBUTION and (
                    not self.data.domain.class_var or not isinstance(self.data.domain.class_var, DiscreteVariable)):
            return

        # draw pearsons residuals
        if self.interior_coloring == PEARSON or not self.data.domain.class_var or not isinstance(self.data.domain.class_var, DiscreteVariable):
            s = sum(self.aprioriDistributions[0])
            expected = s * reduce(lambda x, y: x * y,
                                  [self.aprioriDistributions[i][used_vals[i]] / float(s) for i in range(len(used_vals))])
            actual = self.conditionalDict[attrVals]
            pearson = float(actual - expected) / sqrt(expected)
            if abs(pearson) < 2:
                ind = 0
            elif abs(pearson) < 4:
                ind = 1
            elif abs(pearson) < 8:
                ind = 2
            else:
                ind = 3

            if pearson > 0:
                color = self.blue_colors[ind]
            else:
                color = self.red_colors[ind]
            OWCanvasRectangle(self.canvas, x0, y0, x1 - x0, y1 - y0, color, color, z=-20)

        # draw class distribution - actual and apriori
        # we do have a discrete class
        else:
            clsValues = list(
                self.attributeValuesDict.get(self.data.domain.class_var.name, [])) or get_variable_values_sorted(
                self.data.domain.class_var)
            aprioriDist = get_distribution(self.data, self.data.domain.class_var.name)
            total = 0
            for i in range(len(clsValues)):
                val = self.conditionalDict[attrVals + "-" + clsValues[i]]
                if val == 0:
                    continue
                if i == len(clsValues) - 1:
                    v = y1 - y0 - total
                else:
                    v = ((y1 - y0) * val) / self.conditionalDict[attrVals]
                OWCanvasRectangle(self.canvas, x0, y0 + total, x1 - x0, v, self.colorPalette[i],
                                  self.colorPalette[i], z=-20)
                total += v

            # show apriori boxes and lines
            if (self.show_apriori_distribution_lines or self.use_boxes) and \
                    abs(x1 - x0) > self._box_size and \
                    abs(y1 - y0) > self._box_size:
                apriori = [aprioriDist[val] / float(len(self.data))
                           for val in clsValues]
                if self.show_apriori_distribution_boxes or \
                        self.data.domain.class_var.name in used_attrs:
                    box_counts = apriori
                else:
                    contingencies = \
                        self.optimizationDlg.getContingencys(used_attrs)
                    box_counts = []
                    for clsVal in clsValues:
                        # compute: P(c_i) * prod (P(c_i|attr_k) / P(c_i))
                        # for each class value
                        pci = aprioriDist[clsVal] / float(sum(aprioriDist.values()))
                        tempVal = pci
                        if pci > 0:
                            #tempVal = 1.0 / Pci
                            for ua, uv in zip(used_attrs, used_vals):
                                tempVal *= contingencies[ua][uv] / pci
                        box_counts.append(tempVal)
                        #boxCounts.append(aprioriDist[val]/float(sum(aprioriDist.values())) * reduce(operator.mul, [contingencies[used_attrs[i]][used_vals[i]][clsVal]/float(sum(contingencies[used_attrs[i]][used_vals[i]].values())) for i in range(len(used_attrs))]))

                total1 = 0
                total2 = 0
                if self.use_boxes:
                    OWCanvasLine(self.canvas, x0 + self._box_size, y0, x0 + self._box_size, y1, z=30)

                for i in range(len(clsValues)):
                    val1 = apriori[i]
                    if self.show_apriori_distribution_boxes:
                        val2 = apriori[i]
                    else:
                        val2 = box_counts[i] / float(sum(box_counts))
                    if i == len(clsValues) - 1:
                        v1 = y1 - y0 - total1
                        v2 = y1 - y0 - total2
                    else:
                        v1 = (y1 - y0) * val1
                        v2 = (y1 - y0) * val2
                    x, y, w, h, xL1, yL1, xL2, yL2 = x0, y0 + total2, self._box_size, v2, x0, y0 + total1 + v1, x1, y0 + total1 + v1

                    if self.use_boxes:
                        OWCanvasRectangle(self.canvas, x, y, w, h, self.colorPalette[i], self.colorPalette[i], z=20)
                    if i < len(clsValues) - 1 and self.show_apriori_distribution_lines:
                        OWCanvasLine(self.canvas, xL1, yL1, xL2, yL2, z=10, penColor=self._apriori_pen_color)

                    total1 += v1
                    total2 += v2

            # show subset distribution
            if self.conditionalSubsetDict:
                # show a rect around the box if subset examples belong to this box
                if self.conditionalSubsetDict[attrVals]:
                    #counts = [self.conditionalSubsetDict[attrVals + "-" + val] for val in clsValues]
                    #if sum(counts) == 1:    color = self.colorPalette[counts.index(1)]
                    #else:                   color = Qt.black
                    #OWCanvasRectangle(self.canvas, x0-2, y0-2, x1-x0+5, y1-y0+5, color, QColor(Qt.white), penWidth = 2, z=-50, penStyle = Qt.DashLine)
                    counts = [self.conditionalSubsetDict[attrVals + "-" + val] for val in clsValues]
                    if sum(counts) == 1:
                        OWCanvasRectangle(self.canvas, x0 - 2, y0 - 2, x1 - x0 + 5, y1 - y0 + 5,
                                          self.colorPalette[counts.index(1)], QColor(Qt.white), penWidth=2, z=-50,
                                          penStyle=Qt.DashLine)

                    if self.show_subset_data_boxes:  # do we want to show exact distribution in the right edge of each cell
                        OWCanvasLine(self.canvas, x1 - self._box_size, y0, x1 - self._box_size, y1, z=30)
                        total = 0
                        for i in range(len(aprioriDist)):
                            val = self.conditionalSubsetDict[attrVals + "-" + clsValues[i]]
                            if not self.conditionalSubsetDict[attrVals] or val == 0: continue
                            if i == len(aprioriDist) - 1:
                                v = y1 - y0 - total
                            else:
                                v = ((y1 - y0) * val) / float(self.conditionalSubsetDict[attrVals])
                            OWCanvasRectangle(self.canvas, x1 - self._box_size, y0 + total, self._box_size, v,
                                              self.colorPalette[i], self.colorPalette[i], z=15)
                            total += v

        tooltipText = "Examples in this area have:<br>" + condition

        if any(aprioriDist):
            clsValues = list(
                self.attributeValuesDict.get(self.data.domain.class_var.name, [])) or get_variable_values_sorted(
                self.data.domain.class_var)
            actual = [self.conditionalDict[attrVals + "-" + clsValues[i]] for i in range(len(aprioriDist))]
            if sum(actual) > 0:
                apriori = [aprioriDist[key] for key in clsValues]
                aprioriText = ""
                actualText = ""
                text = ""
                for i in range(len(clsValues)):
                    text += 4 * "&nbsp;" + "<b>%s</b>: %d / %.1f%% (Expected %.1f / %.1f%%)<br>" % (
                        clsValues[i], actual[i], 100.0 * actual[i] / float(sum(actual)),
                        (apriori[i] * sum(actual)) / float(sum(apriori)), 100.0 * apriori[i] / float(sum(apriori)))
                tooltipText += "Number of examples: " + str(int(sum(actual))) + "<br> Class distribution:<br>" + text[
                                                                                                                 :-4]
        elif pearson and expected:
            tooltipText += "<hr>Expected number of examples: %.1f<br>Actual number of examples: %d<br>Standardized (Pearson) residual: %.1f" % (
                expected, self.conditionalDict[attrVals], pearson)
        outerRect.setToolTip(tooltipText)


    # draw the class legend below the square
    def DrawLegend(self, data, x0_x1, y0_y1):
        x0, x1 = x0_x1
        y0, y1 = y0_y1
        if self.interior_coloring == CLASS_DISTRIBUTION and (
                    not data.domain.class_var or isinstance(data.domain.class_var, ContinuousVariable)):
            return

        if self.interior_coloring == PEARSON:
            names = ["<-8", "-8:-4", "-4:-2", "-2:2", "2:4", "4:8", ">8", "Residuals:"]
            colors = self.red_colors[::-1] + self.blue_colors[1:]
        else:
            names = (list(self.attributeValuesDict.get(data.domain.class_var.name, [])) or get_variable_values_sorted(
                data.domain.class_var)) + [data.domain.class_var.name + ":"]
            colors = [self.colorPalette[i] for i in range(len(data.domain.class_var.values))]

        self.names = [OWCanvasText(self.canvas, name, alignment=Qt.AlignVCenter) for name in names]
        totalWidth = sum([text.boundingRect().width() for text in self.names])

        # compute the x position of the center of the legend
        y = y1 + self.attributeNameOffset + 20
        distance = 30
        startX = (x0 + x1) / 2 - (totalWidth + (len(names)) * distance) / 2

        self.names[-1].setPos(startX + 15, y)
        self.names[-1].show()
        xOffset = self.names[-1].boundingRect().width() + distance

        size = 8  # 8 + 8*(self.interiorColoring == PEARSON)

        for i in range(len(names) - 1):
            if self.interior_coloring == PEARSON:
                edgeColor = Qt.black
            else:
                edgeColor = colors[i]

            OWCanvasRectangle(self.canvas, startX + xOffset, y - size / 2, size, size, edgeColor, colors[i])
            self.names[i].setPos(startX + xOffset + 10, y)
            xOffset += distance + self.names[i].boundingRect().width()

    # def saveToFileCanvas(self):
    #     sizeDlg = OWDlgs.OWChooseImageSizeDlg(self.canvas, parent=self)
    #     sizeDlg.exec_()

    def setColors(self):
        dlg = self.createColorDialog()
        if dlg.exec_():
            self.color_settings = dlg.getColorSchemas()
            self.selected_schema_index = dlg.selectedSchemaIndex
            self.colorPalette = dlg.getDiscretePalette("discPalette")
            if self.data and self.data.domain.class_var and isinstance(self.data.domain.class_var, DiscreteVariable):
                self.colorPalette.set_number_of_colors(len(self.data.domain.class_var.values))
            self.updateGraph()

    def createColorDialog(self):
        c = ColorPaletteDlg(self, "Color Palette")
        c.createDiscretePalette("discPalette", "Discrete Palette",
                                DefaultRGBColors)  #defaultColorBrewerPalette)
        c.setColorSchemas(self.color_settings, self.selected_schema_index)
        return c

    # ########################################
    # cell/example selection
    def sendSelectedData(self):
        # send the selected examples
        self.send("Selected Data", self.getSelectedExamples())

    # add a new rectangle. update the graph and see which mosaics does it intersect. add this mosaics to the recentlyAdded list
    def addSelection(self, rect):
        self.selectionRectangle = rect
        self.updateGraph(drillUpdateSelection=0)
        self.sendSelectedData()

        if getattr(self, "recentlyAdded", []):
            self.selectionConditionsHistorically = self.selectionConditionsHistorically + [self.recentlyAdded]
            self.recentlyAdded = []

        # self.optimizationDlg.mtUpdateState()  # we have already called this in self.updateGraph() call
        self.selectionRectangle = None

    # remove the mosaics that were added with the last selection rectangle
    def removeLastSelection(self):
        if self.selectionConditionsHistorically:
            vals = self.selectionConditionsHistorically.pop()
            for val in vals:
                if tuple(val) in self.selectionConditions:
                    self.selectionConditions.remove(tuple(val))

        self.updateGraph()
        ##        self.optimizationDlg.mtUpdateState()       # we have already called this in self.updateGraph() call
        self.sendSelectedData()

    def removeAllSelections(self):
        self.selectionConditions = []
        self.selectionConditionsHistorically = []
        ##        self.optimizationDlg.mtUpdateState()       # removeAllSelections is always called before updateGraph() - where mtUpdateState is called
        self.sendSelectedData()

    # return examples in currently selected boxes as example table or array of 0/1 values
    def getSelectedExamples(self, asExampleTable=1, negate=0, selectionConditions=None, data=None, attrs=None):
        if attrs == None:     attrs = self.getShownAttributeList()
        if data == None:      data = self.data
        if selectionConditions == None:    selectionConditions = self.selectionConditions

        if attrs == [] or not data:
            return None

        # TODO: poglej kaj je s tem
        # pp = orange.Preprocessor_take()
        sumIndices = numpy.zeros(len(data))
        # for val in selectionConditions:
        #     for i, attr in enumerate(attrs):
        #         pp.values[data.domain[attr]] = val[i]
        #     indices = numpy.array(pp.selectionVector(data))
        #     sumIndices += indices
        selectedIndices = list(numpy.where(sumIndices > 0, 1 - negate, 0 + negate))

        # if asExampleTable:
        #     return data.selectref(selectedIndices)
        # else:
        #     return selectedIndices

    def saveSettings(self):
        OWWidget.saveSettings(self)
        # self.optimizationDlg.saveSettings()


class SortAttributeValuesDlg(OWWidget):
    def __init__(self, attr="", valueList=[]):
        super().__init__(self)

        self.setLayout(QVBoxLayout())
        #self.space = QWidget(self)
        #self.layout = QVBoxLayout(self, 4)
        #self.layout.addWidget(self.space)

        box1 = gui.widgetBox(self, "Select Value Order for Attribute \"" + attr + '"', orientation="horizontal")

        self.attributeList = gui.listBox(box1, self, selectionMode=QListWidget.ExtendedSelection, enableDragDrop=1)
        self.attributeList.addItems(valueList)

        vbox = gui.widgetBox(box1, "", orientation="vertical")
        self.buttonUPAttr = gui.button(vbox, self, "", callback=self.moveAttrUP,
                                       tooltip="Move selected attribute values up")
        self.buttonDOWNAttr = gui.button(vbox, self, "", callback=self.moveAttrDOWN,
                                         tooltip="Move selected attribute values down")
        self.buttonUPAttr.setIcon(QIcon(os.path.join(environ.widget_install_dir, "icons/Dlg_up3.png")))
        self.buttonUPAttr.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding))
        self.buttonUPAttr.setFixedWidth(40)
        self.buttonDOWNAttr.setIcon(QIcon(os.path.join(environ.widget_install_dir, "icons/Dlg_down3.png")))
        self.buttonDOWNAttr.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding))
        self.buttonDOWNAttr.setFixedWidth(40)

        box2 = gui.widgetBox(self, 1, orientation="horizontal")
        self.okButton = gui.button(box2, self, "OK", callback=self.accept)
        self.cancelButton = gui.button(box2, self, "Cancel", callback=self.reject)

        self.resize(300, 300)

    # move selected attribute values
    def moveAttrUP(self):
        for i in range(1, self.attributeList.count()):
            if self.attributeList.item(i).isSelected():
                self.attributeList.insertItem(i - 1, self.attributeList.item(i).text())
                self.attributeList.takeItem(i + 1)
                self.attributeList.item(i - 1).setSelected(True)

    def moveAttrDOWN(self):
        for i in range(self.attributeList.count() - 2, -1, -1):
            if self.attributeList.item(i).isSelected():
                self.attributeList.insertItem(i + 2, self.attributeList.item(i).text())
                self.attributeList.item(i + 2).setSelected(True)
                self.attributeList.takeItem(i)


class OWCanvasText(QGraphicsTextItem):
    def __init__(self, canvas, text="", x=0, y=0, alignment=Qt.AlignLeft | Qt.AlignTop, bold=0, font=None, z=0,
                 htmlText=None, tooltip=None, show=1, vertical=False):
        QGraphicsTextItem.__init__(self, text, None, canvas)

        if font:
            self.setFont(font)
        if bold:
            font = self.font()
            font.setBold(bold)
            self.setFont(font)
        if htmlText:
            self.setHtml(htmlText)

        self.alignment = alignment
        self.vertical = vertical
        if vertical:
            self.setRotation(-90)

        self.setPos(x, y)
        self.x, self.y = x, y
        self.setZValue(z)
        if tooltip: self.setToolTip(tooltip)

        if show:
            self.show()
        else:
            self.hide()

    def setPos(self, x, y):
        self.x, self.y = x, y
        rect = QGraphicsTextItem.boundingRect(self)
        if self.vertical:
            h, w = rect.height(), rect.width()
            rect.setWidth(h)
            rect.setHeight(-w)
        if int(self.alignment & Qt.AlignRight):
            x -= rect.width()
        elif int(self.alignment & Qt.AlignHCenter):
            x -= rect.width() / 2.
        if int(self.alignment & Qt.AlignBottom):
            y -= rect.height()
        elif int(self.alignment & Qt.AlignVCenter):
            y -= rect.height() / 2.
        QGraphicsTextItem.setPos(self, x, y)


def OWCanvasRectangle(canvas, x=0, y=0, width=0, height=0, penColor=QColor(128, 128, 128), brushColor=None, penWidth=1, z=0,
                      penStyle=Qt.SolidLine, pen=None, tooltip=None, show=1):
    rect = QGraphicsRectItem(x, y, width, height, None, canvas)
    if brushColor: rect.setBrush(QBrush(brushColor))
    if pen:
        rect.setPen(pen)
    else:
        rect.setPen(QPen(penColor, penWidth, penStyle))
    rect.setZValue(z)
    if tooltip: rect.setToolTip(tooltip)
    if show:
        rect.show()
    else:
        rect.hide()
    return rect


def OWCanvasLine(canvas, x1=0, y1=0, x2=0, y2=0, penWidth=2, penColor=QColor(255, 255, 255, 128), pen=None, z=0, tooltip=None, show=1):
    r = QGraphicsLineItem(x1, y1, x2, y2, None, canvas)
    if pen != None:
        r.setPen(pen)
    else:
        r.setPen(QPen(penColor, penWidth))
    r.setZValue(z)
    if tooltip: r.setToolTip(tooltip)

    if show:
        r.show()
    else:
        r.hide()

    return r


def OWCanvasEllipse(canvas, x=0, y=0, width=0, height=0, penWidth=1, startAngle=0, angles=360, penColor=Qt.black,
                    brushColor=None, z=0, penStyle=Qt.SolidLine, pen=None, tooltip=None, show=1):
    e = QGraphicsEllipseItem(x, y, width, height, None, canvas)
    e.setZValue(z)
    if brushColor != None:
        e.setBrush(QBrush(brushColor))
    if pen != None:
        e.setPen(pen)
    else:
        e.setPen(QPen(penColor, penWidth))
    e.setStartAngle(startAngle)
    e.setSpanAngle(angles * 16)
    if tooltip: e.setToolTip(tooltip)

    if show:
        e.show()
    else:
        e.hide()

    return e


#test widget appearance
if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWMosaicDisplay()
    ow.show()
    #    data = orange.ExampleTable(r"e:\Development\Orange Datasets\UCI\zoo.tab")
    data = Table("zoo.tab")
    ow.setData(data)
    ow.handleNewSignals()
    #    for d in ["zoo.tab", "iris.tab", "zoo.tab"]:
    #        data = orange.ExampleTable(r"e:\Development\Orange Datasets\UCI\\" + d)
    #        ow.setData(data)
    #        ow.handleNewSignals()
    a.exec_()
