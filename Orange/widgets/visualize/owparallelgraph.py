#
# OWParallelGraph.py
#
import os
import sys
import math

import numpy as np

from PyQt4.QtCore import QLineF, Qt, QEvent, QRect, QPoint, QPointF
from PyQt4.QtGui import QGraphicsPathItem, QPixmap, QColor, QBrush, QPen, QToolTip, QPainterPath

from Orange.canvas.utils import environ

from Orange.statistics.contingency import get_contingencies, get_contingency
from Orange.statistics.distribution import get_distribution
from Orange.widgets.settings import SettingProvider, Setting
from Orange.widgets.tests.test_settings import VarTypes
from Orange.widgets.utils.plot import OWPlot, UserAxis, AxisStart, AxisEnd, OWCurve, OWPoint, PolygonCurve, \
    xBottom, yLeft, ZOOMING
from Orange.widgets.utils.scaling import get_variable_value_indices, get_variable_values_sorted, ScaleData

NO_STATISTICS = 0
MEANS = 1
MEDIAN = 2


class OWParallelGraph(OWPlot, ScaleData, SettingProvider):
    show_distributions = Setting(False)
    show_attr_values = Setting(True)
    show_statistics = Setting(default=False)

    use_splines = Setting(False)
    alpha_value = Setting(150)
    alpha_value_2 = Setting(150)

    def __init__(self, widget, parent=None, name=None):
        widget.settingsHandler.initialize(self)
        OWPlot.__init__(self, parent, name, axes=[], widget=widget)
        ScaleData.__init__(self)

        self.update_antialiasing(False)

        self.parallelDlg = widget
        self.toolRects = []
        self.show_statistics = 0
        self.lastSelectedCurve = None
        self.enableGridXB(0)
        self.enableGridYL(0)
        self.domainContingency = None
        self.autoUpdateAxes = 1
        self.oldLegendKeys = []
        self.selectionConditions = {}
        self.visualized_attributes = []
        self.visualized_mid_labels = []
        self.selectedExamples = []
        self.unselectedExamples = []
        self.bottomPixmap = QPixmap(os.path.join(environ.widget_install_dir, "icons/upgreenarrow.png"))
        self.topPixmap = QPixmap(os.path.join(environ.widget_install_dir, "icons/downgreenarrow.png"))

    def setData(self, data, subsetData=None, **args):
        OWPlot.setData(self, data)
        ScaleData.setData(self, data, subsetData, **args)
        self.domainContingency = None

    # update shown data. Set attributes, coloring by className ....
    def updateData(self, attributes, midLabels=None, updateAxisScale=1):
        if attributes != self.visualized_attributes:
            self.selectionConditions = {}       # reset selections

        self.clear()

        self.visualized_attributes = []
        self.visualized_mid_labels = []
        self.selectedExamples = []
        self.unselectedExamples = []

        if not (self.have_data or self.have_subset_data):
            return
        if len(attributes) < 2:
            return

        self.visualized_attributes = attributes
        self.visualized_mid_labels = midLabels
        for name in list(
                self.selectionConditions.keys()):        # keep only conditions that are related to the currently visualized attributes
            if name not in self.visualized_attributes:
                self.selectionConditions.pop(name)

        # set the limits for panning
        self.xPanningInfo = (1, 0, len(attributes) - 1)
        self.yPanningInfo = (0, 0, 0)

        length = len(attributes)
        indices = [self.attribute_name_index[label] for label in attributes]

        xs = list(range(length))
        dataSize = len(self.scaled_data[0])

        if self.data_has_discrete_class:
            self.discPalette.setNumberOfColors(len(self.data_domain.class_var.values))


        # ############################################
        # draw the data
        # ############################################
        subsetIdsToDraw = self.have_subset_data and dict(
            [(self.rawSubsetData[i].id, 1) for i in self.getValidSubsetIndices(indices)]) or {}
        validData = self.getValidList(indices)
        mainCurves = {}
        subCurves = {}
        conditions = dict([(name, attributes.index(name)) for name in list(self.selectionConditions.keys())])

        for i in range(dataSize):
            if not validData[i]:
                continue

            if not self.data_has_class:
                newColor = (0, 0, 0)
            elif self.data_has_continuous_class:
                newColor = self.contPalette.getRGB(self.noJitteringScaledData[self.data_class_index][i])
            else:
                newColor = self.discPalette.getRGB(self.original_data[self.data_class_index][i])

            data = [self.scaled_data[index][i] for index in indices]

            # if we have selected some conditions and the example does not match it we show it as a subset data
            if 0 in [self.selectionConditions[name][0] <= data[index] <= self.selectionConditions[name][1]
                     for (name, index) in list(conditions.items())]:
                alpha = self.alpha_value_2
                curves = subCurves
                self.unselectedExamples.append(i)
            # if we have subset data then use alpha2 for main data and alpha for subset data
            elif self.have_subset_data and self.raw_data[i].id not in subsetIdsToDraw:
                alpha = self.alpha_value_2
                curves = subCurves
                self.unselectedExamples.append(i)
            else:
                alpha = self.alpha_value
                curves = mainCurves
                self.selectedExamples.append(i)
                #FIXME:
                #if self.raw_data[i].id in subsetIdsToDraw:
                #    subsetIdsToDraw.pop(self.raw_data[i].id)

            newColor += (alpha,)

            if newColor not in curves:
                curves[newColor] = []
            curves[newColor].extend(data)

        # if we have a data subset that contains examples that don't exist in the original dataset we show them here
        if subsetIdsToDraw != {}:
            validSubsetData = self.getValidSubsetList(indices)

            for i in range(len(self.rawSubsetData)):
                if not validSubsetData[i]:
                    continue
                if self.rawSubsetData[i].id not in subsetIdsToDraw:
                    continue

                data = [self.scaledSubsetData[index][i] for index in indices]
                if not self.data_domain.class_var or self.rawSubsetData[i].getclass().isSpecial():
                    newColor = (0, 0, 0)
                elif self.data_has_continuous_class:
                    newColor = self.contPalette.getRGB(self.noJitteringScaledSubsetData[self.dataClassIndex][i])
                else:
                    newColor = self.discPalette.getRGB(self.originalSubsetData[self.dataClassIndex][i])

                if 0 in [self.selectionConditions[name][0] <= data[index] <= self.selectionConditions[name][1]
                         for (name, index) in list(conditions.items())]:
                    newColor += (self.alpha_value_2,)
                    curves = subCurves
                else:
                    newColor += (self.alpha_value,)
                    curves = mainCurves

                if newColor not in curves:
                    curves[newColor] = []
                curves[newColor].extend(data)

        # add main curves
        keys = sorted(mainCurves.keys())
        for key in keys:
            curve = ParallelCoordinatesCurve(len(attributes), mainCurves[key], key)
            curve.fitted = self.use_splines
            curve.attach(self)

        # add sub curves
        keys = sorted(subCurves.keys())
        for key in keys:
            curve = ParallelCoordinatesCurve(len(attributes), subCurves[key], key)
            curve.fitted = self.use_splines
            curve.attach(self)

        # ############################################
        # do we want to show distributions with discrete attributes
        if self.show_distributions and self.data_has_discrete_class and self.have_data:
            self.showDistributionValues(validData, indices)

        self.draw_axes(attributes)

        # ##############################################
        # show lines that represent standard deviation or quartiles
        # ##############################################
        if self.show_statistics and self.have_data:
            data = []
            for i in range(length):
                if self.data_domain[indices[i]].var_type != VarTypes.Continuous:
                    data.append([()])
                    continue  # only for continuous attributes
                array = np.compress(np.equal(self.validDataArray[indices[i]], 1),
                                    self.scaledData[indices[i]])  # remove missing values

                if not self.data_has_class or self.data_has_continuous_class:    # no class
                    if self.show_statistics == MEANS:
                        m = array.mean()
                        dev = array.std()
                        data.append([(m - dev, m, m + dev)])
                    elif self.show_statistics == MEDIAN:
                        sorted_array = np.sort(array)
                        if len(sorted_array) > 0:
                            data.append([(sorted_array[int(len(sorted_array) / 4.0)],
                                          sorted_array[int(len(sorted_array) / 2.0)],
                                          sorted_array[int(len(sorted_array) * 0.75)])])
                        else:
                            data.append([(0, 0, 0)])
                else:
                    curr = []
                    classValues = get_variable_values_sorted(self.data_domain.class_var)
                    classValueIndices = get_variable_value_indices(self.data_domain.class_var)
                    for c in range(len(classValues)):
                        scaledVal = ((classValueIndices[classValues[c]] * 2) + 1) / float(2 * len(classValueIndices))
                        nonMissingValues = np.compress(np.equal(self.validDataArray[indices[i]], 1),
                                                       self.noJitteringScaledData[self.dataClassIndex])
                        # remove missing values
                        arr_c = np.compress(np.equal(nonMissingValues, scaledVal), array)
                        if len(arr_c) == 0:
                            curr.append((0, 0, 0))
                            continue
                        if self.show_statistics == MEANS:
                            m = arr_c.mean()
                            dev = arr_c.std()
                            curr.append((m - dev, m, m + dev))
                        elif self.show_statistics == MEDIAN:
                            sorted_array = np.sort(arr_c)
                            curr.append((sorted_array[int(len(arr_c) / 4.0)], sorted_array[int(len(arr_c) / 2.0)],
                                         sorted_array[int(len(arr_c) * 0.75)]))
                    data.append(curr)

            # draw vertical lines
            for i in range(len(data)):
                for c in range(len(data[i])):
                    if data[i][c] == (): continue
                    x = i - 0.03 * (len(data[i]) - 1) / 2.0 + c * 0.03
                    col = QColor(self.discPalette[c])
                    col.setAlpha(self.alpha_value_2)
                    self.addCurve("", col, col, 3, OWCurve.Lines, OWPoint.NoSymbol, xData=[x, x, x],
                                  yData=[data[i][c][0], data[i][c][1], data[i][c][2]], lineWidth=4)
                    self.addCurve("", col, col, 1, OWCurve.Lines, OWPoint.NoSymbol, xData=[x - 0.03, x + 0.03],
                                  yData=[data[i][c][0], data[i][c][0]], lineWidth=4)
                    self.addCurve("", col, col, 1, OWCurve.Lines, OWPoint.NoSymbol, xData=[x - 0.03, x + 0.03],
                                  yData=[data[i][c][1], data[i][c][1]], lineWidth=4)
                    self.addCurve("", col, col, 1, OWCurve.Lines, OWPoint.NoSymbol, xData=[x - 0.03, x + 0.03],
                                  yData=[data[i][c][2], data[i][c][2]], lineWidth=4)

            # draw lines with mean/median values
            classCount = 1
            if not self.data_has_class or self.data_has_continuous_class:
                classCount = 1 # no class
            else:
                classCount = len(self.data_domain.class_var.values)
            for c in range(classCount):
                diff = - 0.03 * (classCount - 1) / 2.0 + c * 0.03
                ys = []
                xs = []
                for i in range(len(data)):
                    if data[i] != [()]:
                        ys.append(data[i][c][1])
                        xs.append(i + diff)
                    else:
                        if len(xs) > 1:
                            col = QColor(self.discPalette[c])
                            col.setAlpha(self.alpha_value_2)
                            self.addCurve("", col, col, 1, OWCurve.Lines,
                                          OWPoint.NoSymbol, xData=xs, yData=ys, lineWidth=4)
                        xs = []
                        ys = []
                col = QColor(self.discPalette[c])
                col.setAlpha(self.alpha_value_2)
                self.addCurve("", col, col, 1, OWCurve.Lines,
                              OWPoint.NoSymbol, xData=xs, yData=ys, lineWidth=4)


        # ##################################################
        # show labels in the middle of the axis
        if midLabels:
            for j in range(len(midLabels)):
                self.addMarker(midLabels[j], j + 0.5, 1.0, alignment=Qt.AlignCenter | Qt.AlignTop)

        # show the legend
        if self.data_has_class:
            if self.data_domain.class_var.var_type == VarTypes.Discrete:
                self.legend().clear()
                varValues = get_variable_values_sorted(self.data_domain.class_var)
                for ind in range(len(varValues)):
                    self.legend().add_item(self.data_domain.class_var.name, varValues[ind],
                                           OWPoint(OWPoint.Rect, self.discPalette[ind], self.point_width))
            else:
                values = self.attr_values[self.data_domain.class_var.name]
                decimals = self.data_domain.class_var.numberOfDecimals
                self.legend().add_color_gradient(self.data_domain.class_var.name,
                                                 ["%%.%df" % decimals % v for v in values])
        else:
            self.legend().clear()
            self.oldLegendKeys = []

        self.replot()

    def draw_axes(self, attributes):
        self.remove_all_axes()
        for i in range(len(attributes)):
            id = UserAxis + i
            a = self.add_axis(id, line=QLineF(i, 0, i, 1), arrows=AxisStart | AxisEnd, zoomable=True)
            a.always_horizontal_text = True
            a.max_text_width = 100
            a.title_margin = -10
            a.text_margin = 0
            a.setZValue(5)
            self.set_axis_title(id, self.data_domain[attributes[i]].name)
            self.set_show_axis_title(id, self.show_attr_values)
            if self.show_attr_values == 1:
                attr = self.data_domain[attributes[i]]
                if attr.var_type == VarTypes.Continuous:
                    self.set_axis_scale(id, self.attr_values[attr.name][0], self.attr_values[attr.name][1])
                elif attr.var_type == VarTypes.Discrete:
                    attrVals = get_variable_values_sorted(self.data_domain[attributes[i]])
                    self.set_axis_labels(id, attrVals)

    # ##########################################
    # SHOW DISTRIBUTION BAR GRAPH
    def showDistributionValues(self, validData, indices):
        # create color table
        clsCount = len(self.data_domain.class_var.values)
        #if clsCount < 1: clsCount = 1.0

        # we create a hash table of possible class values (happens only if we have a discrete class)
        classValueSorted = get_variable_values_sorted(self.data_domain.class_var)
        if self.domainContingency == None:
            self.domainContingency = get_contingencies(self.raw_data)

        maxVal = 1
        for attr in indices:
            if self.data_domain[attr].var_type != VarTypes.Discrete:
                continue
            if self.data_domain[attr] == self.data_domain.class_var:
                maxVal = max(maxVal, max(get_distribution(self.raw_data, attr)))
            else:
                maxVal = max(maxVal,
                             max([max(val or [1]) for val in list(self.domainContingency[attr].values())] or [1]))

        for graphAttrIndex, index in enumerate(indices):
            attr = self.data_domain[index]
            if attr.var_type != VarTypes.Discrete:
                continue
            if self.data_domain[index] == self.data_domain.class_var:
                contingency = get_contingency(self.raw_data, self.data_domain[index], self.data_domain[index])
            else:
                contingency = self.domainContingency[index]

            attrLen = len(attr.values)

            # we create a hash table of variable values and their indices
            variableValueIndices = get_variable_value_indices(self.data_domain[index])
            variableValueSorted = get_variable_values_sorted(self.data_domain[index])

            # create bar curve
            for j in range(attrLen):
                attrVal = variableValueSorted[j]
                try:
                    attrValCont = contingency[attrVal]
                except IndexError as ex:
                    print(ex, attrVal, contingency, file=sys.stderr)
                    continue

                for i in range(clsCount):
                    clsVal = classValueSorted[i]

                    newColor = QColor(self.discPalette[i])
                    newColor.setAlpha(self.alpha_value)

                    width = float(attrValCont[clsVal] * 0.5) / float(maxVal)
                    interval = 1.0 / float(2 * attrLen)
                    yOff = float(1.0 + 2.0 * j) / float(2 * attrLen)
                    height = 0.7 / float(clsCount * attrLen)

                    yLowBott = yOff + float(clsCount * height) / 2.0 - i * height
                    curve = PolygonCurve(QPen(newColor),
                                         QBrush(newColor),
                                         xData=[graphAttrIndex, graphAttrIndex + width,
                                                graphAttrIndex + width, graphAttrIndex],
                                         yData=[yLowBott, yLowBott, yLowBott - height,
                                                yLowBott - height],
                                         tooltip=self.data_domain[index].name)
                    curve.attach(self)


    # handle tooltip events
    def event(self, ev):
        if ev.type() == QEvent.ToolTip:
            x = self.inv_transform(xBottom, ev.pos().x())
            y = self.inv_transform(yLeft, ev.pos().y())

            canvasPos = self.mapToScene(ev.pos())
            xFloat = self.inv_transform(xBottom, canvasPos.x())
            contact, (index, pos) = self.testArrowContact(int(round(xFloat)), canvasPos.x(), canvasPos.y())
            if contact:
                attr = self.data_domain[self.visualized_attributes[index]]
                if attr.var_type == VarTypes.Continuous:
                    condition = self.selectionConditions.get(attr.name, [0, 1])
                    val = self.attr_values[attr.name][0] + condition[pos] * (
                        self.attr_values[attr.name][1] - self.attr_values[attr.name][0])
                    strVal = attr.name + "= %%.%df" % (attr.numberOfDecimals) % (val)
                    QToolTip.showText(ev.globalPos(), strVal)
            else:
                for curve in self.items():
                    if type(curve) == PolygonCurve and \
                            curve.boundingRect().contains(x, y) and \
                            getattr(curve, "tooltip", None):
                        (name, value, total, dist) = curve.tooltip
                        count = sum([v[1] for v in dist])
                        if count == 0:
                            continue
                        tooltipText = "Attribute: <b>%s</b><br>Value: <b>%s</b><br>" \
                                      "Total instances: <b>%i</b> (%.1f%%)<br>" \
                                      "Class distribution:<br>" % (
                                          name, value, count, 100.0 * count / float(total))
                        for (val, n) in dist:
                            tooltipText += "&nbsp; &nbsp; <b>%s</b> : <b>%i</b> (%.1f%%)<br>" % (
                                val, n, 100.0 * float(n) / float(count))
                        QToolTip.showText(ev.globalPos(), tooltipText[:-4])

        elif ev.type() == QEvent.MouseMove:
            QToolTip.hideText()

        return OWPlot.event(self, ev)


    def testArrowContact(self, indices, x, y):
        if type(indices) != list: indices = [indices]
        for index in indices:
            if index >= len(self.visualized_attributes) or index < 0: continue
            intX = self.transform(xBottom, index)
            bottom = self.transform(yLeft,
                                    self.selectionConditions.get(self.visualized_attributes[index], [0, 1])[0])
            bottomRect = QRect(intX - self.bottomPixmap.width() / 2, bottom, self.bottomPixmap.width(),
                               self.bottomPixmap.height())
            if bottomRect.contains(QPoint(x, y)):
                return 1, (index, 0)
            top = self.transform(yLeft,
                                 self.selectionConditions.get(self.visualized_attributes[index], [0, 1])[1])
            topRect = QRect(intX - self.topPixmap.width() / 2, top - self.topPixmap.height(), self.topPixmap.width(),
                            self.topPixmap.height())
            if topRect.contains(QPoint(x, y)):
                return 1, (index, 1)
        return 0, (0, 0)

    def mousePressEvent(self, e):
        canvasPos = self.mapToScene(e.pos())
        xFloat = self.inv_transform(xBottom, canvasPos.x())
        contact, info = self.testArrowContact(int(round(xFloat)), canvasPos.x(), canvasPos.y())

        if contact:
            self.pressedArrow = info
        else:
            OWPlot.mousePressEvent(self, e)


    def mouseMoveEvent(self, e):
        if hasattr(self, "pressedArrow"):
            canvasPos = self.mapToScene(e.pos())
            yFloat = min(1, max(0, self.inv_transform(yLeft, canvasPos.y())))
            index, pos = self.pressedArrow
            attr = self.data_domain[self.visualized_attributes[index]]
            oldCondition = self.selectionConditions.get(attr.name, [0, 1])
            oldCondition[pos] = yFloat
            self.selectionConditions[attr.name] = oldCondition
            self.updateData(self.visualized_attributes, self.visualized_mid_labels, updateAxisScale=0)

            if attr.var_type == VarTypes.Continuous:
                val = self.attr_values[attr.name][0] + oldCondition[pos] * (
                    self.attr_values[attr.name][1] - self.attr_values[attr.name][0])
                strVal = attr.name + "= %%.%df" % (attr.numberOfDecimals) % (val)
                QToolTip.showText(e.globalPos(), strVal)
            if self.sendSelectionOnUpdate and self.autoSendSelectionCallback:
                self.autoSendSelectionCallback()

        else:
            OWPlot.mouseMoveEvent(self, e)

    def mouseReleaseEvent(self, e):
        if hasattr(self, "pressedArrow"):
            del self.pressedArrow
            if self.autoSendSelectionCallback and not (self.sendSelectionOnUpdate and self.autoSendSelectionCallback):
                self.autoSendSelectionCallback() # send the new selection
        else:
            OWPlot.mouseReleaseEvent(self, e)


    def staticMouseClick(self, e):
        if e.button() == Qt.LeftButton and self.state == ZOOMING:
            if self.tempSelectionCurve: self.tempSelectionCurve.detach()
            self.tempSelectionCurve = None
            canvasPos = self.mapToScene(e.pos())
            x = self.inv_transform(xBottom, canvasPos.x())
            y = self.inv_transform(yLeft, canvasPos.y())
            diffX = (self.axisScaleDiv(xBottom).interval().maxValue() - self.axisScaleDiv(
                xBottom).interval().minValue()) / 2.

            xmin = x - (diffX / 2.) * (x - self.axisScaleDiv(xBottom).interval().minValue()) / diffX
            xmax = x + (diffX / 2.) * (self.axisScaleDiv(xBottom).interval().maxValue() - x) / diffX
            ymin = self.axisScaleDiv(yLeft).interval().maxValue()
            ymax = self.axisScaleDiv(yLeft).interval().minValue()

            self.zoomStack.append((
                self.axisScaleDiv(xBottom).interval().minValue(), self.axisScaleDiv(xBottom).interval().maxValue(),
                self.axisScaleDiv(yLeft).interval().minValue(), self.axisScaleDiv(yLeft).interval().maxValue()))
            self.setNewZoom(xmin, xmax, ymax, ymin)
            return 1

        # if the user clicked between two lines send a list with the names of the two attributes
        elif self.parallelDlg:
            x1 = int(self.inv_transform(xBottom, e.x()))
            axis = self.axisScaleDraw(xBottom)
            self.parallelDlg.sendShownAttributes([str(axis.label(x1)), str(axis.label(x1 + 1))])
        return 0

    def removeAllSelections(self, send=1):
        self.selectionConditions = {}
        self.updateData(self.visualized_attributes, self.visualized_mid_labels, updateAxisScale=0)
        if send and self.autoSendSelectionCallback:
            self.autoSendSelectionCallback() # do we want to send new selection

    # draw the curves and the selection conditions
    def drawCanvas(self, painter):
        OWPlot.drawCanvas(self, painter)
        for i in range(
                int(max(0, math.floor(self.axisScaleDiv(xBottom).interval().minValue()))),
                int(min(len(self.visualized_attributes),
                        math.ceil(self.axisScaleDiv(xBottom).interval().maxValue()) + 1))):
            bottom, top = self.selectionConditions.get(self.visualized_attributes[i], (0, 1))
            painter.drawPixmap(self.transform(xBottom, i) - self.bottomPixmap.width() / 2,
                               self.transform(yLeft, bottom), self.bottomPixmap)
            painter.drawPixmap(self.transform(xBottom, i) - self.topPixmap.width() / 2,
                               self.transform(yLeft, top) - self.topPixmap.height(), self.topPixmap)

    # get selected examples
    # this function must be called after calling self.updateGraph
    def getSelectionsAsExampleTables(self):
        # FIXME:
        return (None, None)

        if not self.have_data:
            return (None, None)

        selected = self.raw_data.getitemsref(self.selectedExamples)
        unselected = self.raw_data.getitemsref(self.unselectedExamples)

        if len(selected) == 0: selected = None
        if len(unselected) == 0: unselected = None
        return (selected, unselected)


# ####################################################################
# a curve that is able to draw several series of lines
class ParallelCoordinatesCurve(OWCurve):
    def __init__(self, n_attributes, y_values, color, name=""):
        OWCurve.__init__(self, tooltip=name)
        self._item = QGraphicsPathItem(self)
        self.path = QPainterPath()
        self.fitted = False

        self.n_attributes = n_attributes
        self.n_rows = int(len(y_values) / n_attributes)

        self.set_style(OWCurve.Lines)
        if isinstance(color,  tuple):
            self.set_pen(QPen(QColor(*color)))
        else:
            self.set_pen(QPen(QColor(color)))

        x_values = list(range(n_attributes)) * self.n_rows
        self.set_data(x_values, y_values)

    def update_properties(self):
        self.redraw_path()

    def redraw_path(self):
        self.path = QPainterPath()
        for segment in self.segment(self.data()):
            if self.fitted:
                self.draw_cubic_path(segment)
            else:
                self.draw_normal_path(segment)
        self._item.setPath(self.graph_transform().map(self.path))
        self._item.setPen(self.pen())

    def segment(self, data):
        for i in range(self.n_rows):
            yield data[i * self.n_attributes:(i + 1) * self.n_attributes]

    def draw_cubic_path(self, segment):
        for (x1, y1), (x2, y2) in zip(segment, segment[1:]):
            self.path.moveTo(x1, y1)
            self.path.cubicTo(QPointF(x1 + 0.5, y1),
                              QPointF(x2 - 0.5, y2), QPointF(x2, y2))

    def draw_normal_path(self, segment):
        if not segment:
            return

        x, y = segment[0]
        self.path.moveTo(x, y)
        for x, y in segment[1:]:
            self.path.lineTo(x, y)
