#
# OWScatterPlotGraph.py
#
from PyQt4.QtGui import QColor, QImage, QApplication
import numpy
import sys
from Orange.data import DiscreteVariable, ContinuousVariable
from Orange.data.sql.table import SqlTable
from Orange.widgets.utils.plot import OWPlot, xBottom, yLeft, OWPalette, OWPoint, ProbabilitiesItem
import Orange
from Orange.widgets.utils.scaling import get_variable_values_sorted, ScaleScatterPlotData

DONT_SHOW_TOOLTIPS = 0
VISIBLE_ATTRIBUTES = 1
ALL_ATTRIBUTES = 2

MIN_SHAPE_SIZE = 6


###########################################################################################
##### CLASS : OWSCATTERPLOTGRAPH
###########################################################################################
class OWScatterPlotGraphQt(OWPlot, ScaleScatterPlotData):
    def __init__(self, scatterWidget, parent = None, name = "None"):
        OWPlot.__init__(self, parent, name, widget = scatterWidget)
        ScaleScatterPlotData.__init__(self)
        # zgoraj zamenjal orngScaleScatterPlotData z ScaleScatterPlotData

        self.pointWidth = 8
        self.jitterContinuous = 0
        self.jitterSize = 5
        self.showXaxisTitle = 1
        self.showYLaxisTitle = 1
        self.showLegend = 1
        self.tooltipKind = 1
        self.showFilledSymbols = 1
        self.showProbabilities = 0

        self.tooltipData = []
        self.scatterWidget = scatterWidget
        self.insideColors = None
        self.shownAttributeIndices = []
        self.shownXAttribute = ""
        self.shownYAttribute = ""
        self.squareGranularity = 3
        self.spaceBetweenCells = 1
        self.oldLegendKeys = {}

        self.enableWheelZoom = 1
        self.potentialsCurve = None

    def setData(self, data, subsetData = None, **args):
        OWPlot.setData(self, data)
        self.oldLegendKeys = {}
        ScaleScatterPlotData.set_data(self, data, subsetData, **args)

    #########################################################
    # update shown data. Set labels, coloring by className ....
    def updateData(self, xAttr, yAttr, colorAttr, shapeAttr = "", sizeShapeAttr = "", labelAttr = None, **args):
        self.legend().clear()
        self.tooltipData = []
        self.potentialsClassifier = None
        self.potentialsImage = None
        # self.canvas().invalidatePaintCache()
        self.shownXAttribute = xAttr
        self.shownYAttribute = yAttr

        if self.scaled_data == None or len(self.scaled_data) == 0:
            self.set_axis_scale(xBottom, 0, 1, 1)
            self.set_axis_scale(yLeft, 0, 1, 1)
            self.setXaxisTitle("")
            self.setYLaxisTitle("")
            self.oldLegendKeys = {}
            return

        self.__dict__.update(args)      # set value from args dictionary

        colorIndex = -1
        if colorAttr != "" and colorAttr != "(Same color)":
            colorIndex = self.attribute_name_index[colorAttr]
            if self.data_domain[colorAttr].var_type == Orange.data.Variable.VarTypes.Discrete:
                self.discPalette.setNumberOfColors(len(self.data_domain[colorAttr].values))

        shapeIndex = -1
        if shapeAttr != "" and shapeAttr != "(Same shape)" and len(self.dataDomain[shapeAttr].values) < 11:
            shapeIndex = self.attribute_name_index[shapeAttr]

        sizeIndex = -1
        if sizeShapeAttr != "" and sizeShapeAttr != "(Same size)":
            sizeIndex = self.attribute_name_index[sizeShapeAttr]

        showContinuousColorLegend = colorIndex != -1 and self.data_domain[colorIndex].var_type == Orange.data.Variable.VarTypes.Continuous

        (xVarMin, xVarMax) = self.attr_values[xAttr]
        (yVarMin, yVarMax) = self.attr_values[yAttr]
        xVar = max(xVarMax - xVarMin, 1e-10)
        yVar = max(yVarMax - yVarMin, 1e-10)
        xAttrIndex = self.attribute_name_index[xAttr]
        yAttrIndex = self.attribute_name_index[yAttr]

        attrIndices = [xAttrIndex, yAttrIndex, colorIndex, shapeIndex, sizeIndex]
        while -1 in attrIndices: attrIndices.remove(-1)
        self.shownAttributeIndices = attrIndices

        # set axis for x attribute
        discreteX = self.data_domain[xAttrIndex].var_type == Orange.data.Variable.VarTypes.Discrete
        if discreteX:
            xVarMax -= 1; xVar -= 1
            xmin = xVarMin - (self.jitterSize + 10.)/100.
            xmax = xVarMax + (self.jitterSize + 10.)/100.
            labels = get_variable_values_sorted(self.dataDomain[xAttrIndex])
        else:
            off  = (xVarMax - xVarMin) * (self.jitterSize * self.jitterContinuous + 2) / 100.0
            xmin = xVarMin - off
            xmax = xVarMax + off
            labels = None
        self.setXlabels(labels)
        self.set_axis_scale(xBottom, xmin, xmax,  discreteX)

        # set axis for y attribute
        discreteY = self.data_domain[yAttrIndex].var_type == Orange.data.Variable.VarTypes.Discrete
        if discreteY:
            yVarMax -= 1; yVar -= 1
            ymin = yVarMin - (self.jitterSize + 10.)/100.
            ymax = yVarMax + (self.jitterSize + 10.)/100.
            labels = get_variable_values_sorted(self.data_domain[yAttrIndex])
        else:
            off  = (yVarMax - yVarMin) * (self.jitterSize * self.jitterContinuous + 2) / 100.0
            ymin = yVarMin - off
            ymax = yVarMax + off
            labels = None
        self.setYLlabels(labels)
        self.set_axis_scale(yLeft, ymin, ymax, discreteY)

        self.setXaxisTitle(xAttr)
        self.setYLaxisTitle(yAttr)

        # compute x and y positions of the points in the scatterplot
        xData, yData = self.get_xy_data_positions(xAttr, yAttr)
        validData = self.get_valid_list(attrIndices)      # get examples that have valid data for each used attribute

        # #######################################################
        # show probabilities
        if self.potentialsCurve:
            self.potentialsCurve.detach()
            self.potentialsCurve = None
        if self.showProbabilities and colorIndex >= 0 and self.data_domain[colorIndex].var_type in [Orange.data.Variable.VarTypes.Discrete, Orange.data.Variable.VarTypes.Continuous]:
            if self.data_domain[colorIndex].var_type == Orange.data.Variable.VarTypes.Discrete:
                domain = Orange.data.Domain([self.data_domain[xAttrIndex], self.data_domain[yAttrIndex], DiscreteVariable(self.attribute_names[colorIndex], values = get_variable_values_sorted(self.data_domain[colorIndex]))])
            else:
                domain = Orange.Domain([self.data_domain[xAttrIndex], self.data_domain[yAttrIndex], ContinuousVariable(self.attributeNames[colorIndex])])
            xdiff = xmax-xmin; ydiff = ymax-ymin
            scX = xData/xdiff
            scY = yData/ydiff
            classData = self.original_data[colorIndex]

            probData = numpy.transpose(numpy.array([scX, scY, classData]))
            probData= numpy.compress(validData, probData, axis = 0)

            sys.stderr.flush()
            self.xmin = xmin; self.xmax = xmax
            self.ymin = ymin; self.ymax = ymax

            if probData.any():
                self.potentialsClassifier = Orange.P2NN(domain, probData, None, None, None, None)
                self.potentialsCurve = ProbabilitiesItem(self.potentialsClassifier, self.squareGranularity, 1., self.spaceBetweenCells)
                self.potentialsCurve.attach(self)
            else:
                self.potentialsClassifier = None

        """
            Create a single curve with different points
        """

        def_color = self.color(OWPalette.Data)
        def_size = self.point_width
        def_shape = self.curveSymbols[0]

        if colorIndex != -1:
            if self.data_domain[colorIndex].var_type == Orange.data.Variable.VarTypes.Continuous:
                c_data = self.no_jittering_scaled_data[colorIndex]
                palette = self.continuous_palette
            else:
                c_data = self.original_data[colorIndex]
                palette = self.discrete_palette
            checked_color_data = [(c_data[i] if validData[i] else 0) for i in range(len(c_data))]
            colorData = [QColor(*palette.getRGB(i)) for i in checked_color_data]
        else:
            colorData = [def_color]

        if sizeIndex != -1:
            sizeData = [MIN_SHAPE_SIZE + round(i * self.pointWidth) for i in self.no_jittering_scaled_data[sizeIndex]]
        else:
            sizeData = [def_size]

        if shapeIndex != -1 and self.data_domain[shapeIndex].var_type == Orange.data.Variable.VarTypes.Discrete:
            shapeData = [self.curveSymbols[int(i)] for i in self.original_data[shapeIndex]]
        else:
            shapeData = [def_shape]

        if labelAttr and labelAttr in [self.raw_data.domain.getmeta(mykey).name for mykey in self.raw_data.domain.getmetas().keys()] + [var.name for var in self.raw_data.domain]:
            if self.raw_data[0][labelAttr].var_type == Orange.data.Variable.VarTypes.Continuous:
                labelData = ["%4.1f" % Orange.Value(i[labelAttr]) if not i[labelAttr].isSpecial() else "" for i in self.raw_data]
            else:
                labelData = [str(i[labelAttr].value) if not i[labelAttr].isSpecial() else "" for i in self.raw_data]
        else:
            labelData = [""]

        if self.have_subset_data:
            subset_ids = [example.id for example in self.raw_subset_data]
            marked_data = [example.id in subset_ids for example in self.raw_data]
            showFilled = 0
        else:
            marked_data = []
        self.set_main_curve_data(xData, yData, colorData, labelData, sizeData, shapeData, marked_data, validData)

        '''
            Create legend items in any case
            so that show/hide legend only
        '''
        discColorIndex = colorIndex if colorIndex != -1 and self.data_domain[colorIndex].var_type == Orange.data.Variable.VarTypes.Discrete else -1
        discShapeIndex = shapeIndex if shapeIndex != -1 and self.data_domain[shapeIndex].var_type == Orange.data.Variable.VarTypes.Discrete else -1
        discSizeIndex = sizeIndex if sizeIndex != -1 and self.data_domain[sizeIndex].var_type == Orange.data.Variable.VarTypes.Discrete else -1

        if discColorIndex != -1:
            num = len(self.data_domain[discColorIndex].values)
            varValues = get_variable_values_sorted(self.data_domain[discColorIndex])
            for ind in range(num):
                self.legend().add_item(self.data_domain[discColorIndex].name, varValues[ind], OWPoint(def_shape, self.discPalette[ind], def_size))

        if discShapeIndex != -1:
            num = len(self.data_domain[discShapeIndex].values)
            varValues = get_variable_values_sorted(self.data_domain[discShapeIndex])
            for ind in range(num):
                self.legend().add_item(self.data_domain[discShapeIndex].name, varValues[ind], OWPoint(self.curveSymbols[ind], def_color, def_size))

        if discSizeIndex != -1:
            num = len(self.data_domain[discSizeIndex].values)
            varValues = get_variable_values_sorted(self.data_domain[discSizeIndex])
            for ind in range(num):
                self.legend().add_item(self.data_domain[discSizeIndex].name, varValues[ind], OWPoint(def_shape, def_color, MIN_SHAPE_SIZE + round(ind*self.pointWidth/len(varValues))))

        # ##############################################################
        # draw color scale for continuous coloring attribute
        if colorIndex != -1 and showContinuousColorLegend:
            self.legend().add_color_gradient(colorAttr, [("%%.%df" % self.data_domain[colorAttr].number_of_decimals % v) for v in self.attr_values[colorAttr]])

        self.replot()

##    # ##############################################################
##    # ######  SHOW CLUSTER LINES  ##################################
##    # ##############################################################
##    def showClusterLines(self, xAttr, yAttr, width = 1):
##        classIndices = getVariableValueIndices(self.rawData, self.attributeNameIndex[self.rawData.domain.classVar.name])
##
##        shortData = self.rawData.select([self.rawData.domain[xAttr], self.rawData.domain[yAttr], self.rawData.domain.classVar])
##        shortData = orange.Preprocessor_dropMissing(shortData)
##
##        (closure, enlargedClosure, classValue) = self.clusterClosure
##
##        (xVarMin, xVarMax) = self.attrValues[xAttr]
##        (yVarMin, yVarMax) = self.attrValues[yAttr]
##        xVar = xVarMax - xVarMin
##        yVar = yVarMax - yVarMin
##
##        if type(closure) == dict:
##            for key in closure.keys():
##                clusterLines = closure[key]
##                color = self.discPalette[classIndices[self.rawData.domain.classVar[classValue[key]].value]]
##                for (p1, p2) in clusterLines:
##                    self.addCurve("", color, color, 1, QwtPlotCurve.Lines, OWPoint.NoSymbol, xData = [float(shortData[p1][0]), float(shortData[p2][0])], yData = [float(shortData[p1][1]), float(shortData[p2][1])], lineWidth = width)
##        else:
##            colorIndex = self.discPalette[classIndices[self.rawData.domain.classVar[classValue].value]]
##            for (p1, p2) in closure:
##                self.addCurve("", color, color, 1, QwtPlotCurve.Lines, OWPoint.NoSymbol, xData = [float(shortData[p1][0]), float(shortData[p2][0])], yData = [float(shortData[p1][1]), float(shortData[p2][1])], lineWidth = width)

    def update_point_size(self):
        if self.scatterWidget.attrSize:
            self.scatterWidget.updateGraph()
        else:
            self.main_curve.set_point_sizes([self.point_width])
            self.update_curves()


    def addTip(self, x, y, attrIndices = None, dataindex = None, text = None):
        if self.tooltipKind == DONT_SHOW_TOOLTIPS: return
        if text == None:
            if self.tooltipKind == VISIBLE_ATTRIBUTES:  text = self.getExampleTooltipText(self.rawData[dataindex], attrIndices)
            elif self.tooltipKind == ALL_ATTRIBUTES:    text = self.getExampleTooltipText(self.rawData[dataindex], range(len(self.attributeNames)))
        self.tips.addToolTip(x, y, text)


    # override the default buildTooltip function defined in OWPlot
    def buildTooltip(self, exampleIndex):
        if exampleIndex < 0:
            example = self.rawSubsetData[-exampleIndex - 1]
        else:
            example = self.rawData[exampleIndex]

        if self.tooltipKind == VISIBLE_ATTRIBUTES:
            text = self.getExampleTooltipText(example, self.shownAttributeIndices)
        elif self.tooltipKind == ALL_ATTRIBUTES:
            text = self.getExampleTooltipText(example)
        return text


    # ##############################################################
    # send 2 example tables. in first is the data that is inside selected rects (polygons), in the second is unselected data
    def getSelectionsAsExampleTables(self, attrList):
        [xAttr, yAttr] = attrList
        #if not self.rawData: return (None, None, None)
        if not self.have_data: return (None, None)

        selIndices, unselIndices = self.getSelectionsAsIndices(attrList)

        if type(self.raw_data) is SqlTable:
            selected = [self.raw_data[i] for (i, val) in enumerate(selIndices) if val]
            unselected = [self.raw_data[i] for (i, val) in enumerate(unselIndices) if val]
        else:
            selected = self.raw_data[numpy.array(selIndices)]
            unselected = self.raw_data[numpy.array(unselIndices)]

        if len(selected) == 0: selected = None
        if len(unselected) == 0: unselected = None

        return (selected, unselected)


    def getSelectionsAsIndices(self, attrList, validData = None):
        [xAttr, yAttr] = attrList
        if not self.have_data: return [], []

        attrIndices = [self.attribute_name_index[attr] for attr in attrList]
        if validData == None:
            validData = self.get_valid_list(attrIndices)

        (xArray, yArray) = self.get_xy_data_positions(xAttr, yAttr)

        return self.get_selected_points(xArray, yArray, validData)


    def onMouseReleased(self, e):
        OWPlot.onMouseReleased(self, e)
        self.updateLayout()

    def computePotentials(self):
        # import orangeom
        s = self.graph_area.toRect().size()
        if not s.isValid():
            self.potentialsImage = QImage()
            return
        rx = s.width()
        ry = s.height()
        rx -= rx % self.squareGranularity
        ry -= ry % self.squareGranularity

        ox = int(self.transform(xBottom, 0) - self.transform(xBottom, self.xmin))
        oy = int(self.transform(yLeft, self.ymin) - self.transform(yLeft, 0))

        if not getattr(self, "potentialsImage", None) or getattr(self, "potentialContext", None) != (rx, ry, self.shownXAttribute, self.shownYAttribute, self.squareGranularity, self.jitterSize, self.jitterContinuous, self.spaceBetweenCells):
            self.potentialContext = (rx, ry, self.shownXAttribute, self.shownYAttribute, self.squareGranularity, self.jitterSize, self.jitterContinuous, self.spaceBetweenCells)
            self.potentialsImageFromClassifier = self.potentialsClassifier

if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)
    c = OWScatterPlotGraphQt(None)
    c.show()
    a.exec_()
