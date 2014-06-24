import sys

import numpy
import pyqtgraph as pg
from pyqtgraph.graphicsItems.GraphicsWidgetAnchor import GraphicsWidgetAnchor
from pyqtgraph.graphicsItems.ScatterPlotItem import SpotItem
from PyQt4 import QtCore
from PyQt4.QtCore import Qt, QRectF, QPointF, QPoint
from PyQt4.QtGui import (QColor, QImage, QApplication, QTransform,
                         QGraphicsObject, QGraphicsTextItem, QLinearGradient,
                         QPen, QBrush, QGraphicsRectItem, QGraphicsItem)

import Orange
from Orange.data import DiscreteVariable, ContinuousVariable
from Orange.data.sql.table import SqlTable
from Orange.widgets.utils.colorpalette import (ColorPaletteGenerator,
                                               ContinuousPaletteGenerator)
from Orange.widgets.utils.plot import (OWPalette, ProbabilitiesItem, OWPlotGUI,
                                       TooltipManager, NOTHING, SELECT, PANNING,
                                       ZOOMING, SELECTION_ADD, SELECTION_REMOVE,
                                       SELECTION_TOGGLE, xBottom, yLeft,
                                       move_item_xy)
from Orange.widgets.utils.scaling import (get_variable_values_sorted,
                                          ScaleScatterPlotData)
from Orange.widgets.widget import OWWidget


DONT_SHOW_TOOLTIPS = 0
VISIBLE_ATTRIBUTES = 1
ALL_ATTRIBUTES = 2

MIN_SHAPE_SIZE = 6

LIGHTER_VALUE = 160
INITIAL_ALPHA_VALUE = 150

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOptions(antialias=False)

def isSelected(point):
    if not isinstance(point, SpotItem):
        raise TypeError('Expected SpotItem instead of %s' % point.__class__)
    return point.pen().color() == point.brush().color()

def toSelectedColor(point):
    if not isinstance(point, SpotItem):
        raise TypeError('Expected SpotItem instead of %s' % point.__class__)
    point.setBrush(point.pen().color())

def toUnselectedColor(point):
    if not isinstance(point, SpotItem):
        raise TypeError('Expected SpotItem instead of %s' % point.__class__)
    color = point.pen().color()
    lighter_color = color.lighter(LIGHTER_VALUE)
    lighter_color.setAlpha(INITIAL_ALPHA_VALUE)
    point.setBrush(lighter_color)


class GradientLegendItem(QGraphicsObject, GraphicsWidgetAnchor):
    """
    Kopija OWLegendGradient, prilagojen za prikaz na pyqtgraph-u.
    """

    gradient_width = 20

    def __init__(self, title, palette, values, parent):
        QGraphicsObject.__init__(self, parent)
        GraphicsWidgetAnchor.__init__(self)
        self.parent = parent
        self.legend = parent

        self.palette = palette
        self.values = values

        self.title = QGraphicsTextItem('%s:' % title, self)
        f = self.title.font()
        f.setBold(True)
        self.title.setFont(f)
        self.title_item = QGraphicsRectItem(self.title.boundingRect(), self)
        self.title_item.setPen(QPen(Qt.NoPen))
        self.title_item.stackBefore(self.title)

        self.label_items = [QGraphicsTextItem(text, self) for text in values]
        for i in self.label_items:
            i.setTextWidth(50)

        self.rect = QRectF()

        self.gradient_item = QGraphicsRectItem(self)
        self.gradient = QLinearGradient()
        self.gradient.setStops([(v*0.1, self.palette[v*0.1]) for v in range(11) ])
        self.orientation = Qt.Horizontal
        self.set_orientation(Qt.Vertical)

        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)

    def set_orientation(self, orientation):
        if self.orientation == orientation:
            return

        self.orientation = orientation

        if self.orientation == Qt.Vertical:
            height = max([item.boundingRect().height() for item in self.label_items])
            total_height = height * max(5, len(self.label_items))
            interval = (total_height - self.label_items[-1].boundingRect().height()) / (len(self.label_items) -1)
            self.gradient_item.setRect(10, 0, self.gradient_width, total_height)
            self.gradient.setStart(10, 0)
            self.gradient.setFinalStop(10, total_height)
            self.gradient_item.setBrush(QBrush(self.gradient))
            self.gradient_item.setPen(QPen(Qt.NoPen))
            y = -20   # hja, no; dela --> pri boundingRect() zato pristejem +20
            x = 0
            move_item_xy(self.title, x, y, False)
            y = 10
            x = 30
            for item in self.label_items:
                move_item_xy(item, x, y, False) # self.parent.graph.animate_plot)
                y += interval
            self.rect = QRectF(10, 0, self.gradient_width + max([item.boundingRect().width() for item in self.label_items]), self.label_items[0].boundingRect().height() * max(5, len(self.label_items)))
        else:
            # za horizontalno orientacijo nisem dodajal title-a
            width = 50
            height = max([item.boundingRect().height() for item in self.label_items])
            total_width = width * max(5, len(self.label_items))
            interval = (total_width - self.label_items[-1].boundingRect().width()) / (len(self.label_items) -1)

            self.gradient_item.setRect(0, 0, total_width, self.gradient_width)
            self.gradient.setStart(0, 0)
            self.gradient.setFinalStop(total_width, 0)
            self.gradient_item.setBrush(QBrush(self.gradient))
            self.gradient_item.setPen(QPen(Qt.NoPen))
            x = 0
            y = 30
            for item in self.label_items:
                move_item_xy(item, x, y, False) #self.parent.graph.animate_plot)
                x += interval
            self.rect = QRectF(0, 0, total_width, self.gradient_width + height)

    def boundingRect(self):
        width = max(self.rect.width(), self.title_item.boundingRect().width())
        height = self.rect.height() + self.title_item.boundingRect().height()
        return QRectF(self.rect.left(), self.rect.top(), width, height)

    def paint(self, painter, option, widget):
        pass


class ScatterViewBox(pg.ViewBox):
    def __init__(self, graph):
        pg.ViewBox.__init__(self)
        self.graph = graph
        self.setMouseMode(self.PanMode)
        self.setMenuEnabled(False) # should disable the right click menu

    def mouseDragEvent(self, ev):
        if self.graph.state == SELECT:
            ev.accept()
            pos = ev.pos()
            if ev.button() == Qt.LeftButton:
                self.updateScaleBox(ev.buttonDownPos(), ev.pos())
                if ev.isFinish():
                    self.rbScaleBox.hide()
                    selectionPixelRect = QRectF(ev.buttonDownPos(ev.button()), pos)
                    points = self.calculatePointsInRect(selectionPixelRect)
                    self.togglePointsSelection(points)
                    self.graph.selection_changed.emit()
                else:
                    self.updateScaleBox(ev.buttonDownPos(), ev.pos())
        elif self.graph.state == ZOOMING or self.graph.state == PANNING:
            ev.ignore()
            super().mouseDragEvent(ev)
        else:
            ev.ignore()

    def calculatePointsInRect(self, pixelRect):
        # Get the data from the GraphicsItem
        item = self.getDataItem() # pg...ScatterPlotItem
        all_points = item.points()
        valueRect = self.childGroup.mapRectFromParent(pixelRect)
        points_in_rect = [all_points[i] for i in range(len(all_points)) if valueRect.contains(QPointF(all_points[i].pos()))]
        return points_in_rect

    def getDataItem(self):
        return self.graph.spi

    def unselectAllPoints(self):
        item = self.getDataItem()
        points = item.points()
        for p in points:
            toUnselectedColor(p)
        self.graph.selectedPoints = []

    def togglePointsSelection(self, points):
        if self.graph.selection_behavior == SELECTION_ADD:
            for p in points:
                #p.setBrush(p.pen().color())
                toSelectedColor(p)
                self.graph.selectedPoints.append(p)
        elif self.graph.selection_behavior == SELECTION_REMOVE:
            for p in points:
                toUnselectedColor(p)
                if p in self.graph.selectedPoints:
                    self.graph.selectedPoints.remove(p)
        elif self.graph.selection_behavior == SELECTION_TOGGLE:
            for p in points:
                if isSelected(p):
                    toUnselectedColor(p)
                    self.graph.selectedPoints.remove(p)
                else:
                    toSelectedColor(p)
                    self.graph.selectedPoints.append(p)

    def shuffle_points(self):
        pass


###########################################################################################
###########################################################################################
##### CLASS : OWSCATTERPLOTGRAPH
class OWScatterPlotGraphQt(OWWidget, ScaleScatterPlotData):
    selection_changed = QtCore.Signal()

    def __init__(self, scatterWidget, parent = None, name = "None"):
        OWWidget.__init__(self)
        svb = ScatterViewBox(self)
        self.pgPlotWidget = pg.PlotWidget(viewBox=svb, parent=parent)
        self.replot = self.pgPlotWidget
        ScaleScatterPlotData.__init__(self)

        self.pgScatterPlotItem = None

        self.jitter_continuous = 0
        self.jitter_size = 5
        self.showXaxisTitle = 1
        self.showYLaxisTitle = 1
        self.tooltipShowsAllAttributes = False  # when False, tooltip shows only visible attributes
        self.showProbabilities = 0

        self.tooltipData = []
        self.tooltipTextColor = pg.mkColor(70, 70, 70)
        self.tooltip = pg.TextItem(text='', color=self.tooltipTextColor, border=pg.mkPen(200, 200, 200), fill=pg.mkBrush(250, 250, 150, 200), anchor=(0,0))
        self.tooltip.hide()

        self.labels = []

        self.scatterWidget = scatterWidget
        self.insideColors = None
        self.shownAttributeIndices = []
        self.shownXAttribute = ""
        self.shownYAttribute = ""
        self.squareGranularity = 3
        self.spaceBetweenCells = 1
        self.oldLegendKeys = {}     # TODO kaj dela tale

        self.enableWheelZoom = 1
        self.potentialsCurve = None

        # spodnje dodal ko sem vzel stran class OWPlot
        self.gui = OWPlotGUI(self)
        self.continuous_palette = ContinuousPaletteGenerator(QColor(200, 200, 200), QColor(0, 0, 0), True)
        self.discrete_palette = ColorPaletteGenerator()

        self.animate_plot = True        # TODO plot se ne animira vec?
        self.selection_behavior = 0 # Add selection

        self.show_legend = 1
        self._legend = None
        self._gradient_legend = None
        self._legend_position = None
        self._gradient_legend_position = None

        # OWScatterPlot needs these:
        self.point_width = 10
        self.show_filled_symbols = True
        self.alpha_value = 255
        self.show_grid = True

        self.curveSymbols = list(range(13))
        self.tips = TooltipManager(self)
        # self.setMouseTracking(True)
        # self.grabGesture(QPinchGesture)
        # self.grabGesture(QPanGesture)

        self.state = NOTHING
        self._pressed_mouse_button = 0 # Qt.NoButton
        self._pressed_point = None
        self.selection_items = []
        self._current_rs_item = None
        self._current_ps_item = None
        self.polygon_close_treshold = 10
        self.sendSelectionOnUpdate = True
        self.auto_send_selection_callback = None

        self.data_range = {}
        self.map_transform = QTransform()
        self.graph_area = QRectF()

        self.selectedPoints = []

        # iz OWWidget; zdaj zakomentirano ker je OWWidget baseclass
        # self.callbackDeposit = []
        # self.controlledAttributes = ControlledAttributesDict

        self.update_grid()

    def spotItemClicked(self, plot, points):
        self.pgScatterPlotItem.getViewBox().unselectAllPoints()
        for p in points:
            toSelectedColor(p) ## TODO tale nastavi points[i] na None?
            self.selectedPoints.append(p)
        self.selection_changed.emit()
        ## TODO ko je graf v pobarvan po continuous atributu in ko selectam samo eno tocko, se zgodi cudnost
        ## TODO vcasih je tezko kliknit na tocko?

    def mouseMoved(self, pos):
        ## TODO: tooltip naj se prikaze na visible area --> ce npr. pokazes na tocko cisto spodaj, naj bo tooltip nad tocko
        act_pos = self.pgScatterPlotItem.mapFromScene(pos)
        points = self.pgScatterPlotItem.pointsAt(act_pos)
        text = []
        if len(points):
            for (i, p) in enumerate(points):
                index = p.data()
                if self.tooltipShowsAllAttributes:
                    text.append('Attributes:')
                    for attr in self.data_domain.attributes:
                        text.append('\n   %s = %4.1f' % (attr.name,
                                                         self.raw_data[index][attr]))
                    if self.data_domain.class_var:
                        text.append('\nClass:\n   %s = %s' % (self.data_domain.class_var.name,
                                                      self.raw_data[index][self.data_domain.class_var]))
                else:
                    text.append('Attributes:\n   %s = %4.1f\n   %s = %4.1f' % (self.shownXAttribute,
                                                                   self.raw_data[index][self.shownXAttribute],
                                                                   self.shownYAttribute,
                                                                   self.raw_data[index][self.shownYAttribute]))
                    text.append('\nClass:\n   %s = %s' % (self.data_domain.class_var.name,
                                                          self.raw_data[index][self.raw_data.domain.class_var]))
                if len(points) > 1 and i < len(points) - 1:
                    text.append('\n------------------\n')
            self.tooltip.setText(''.join(text), color=self.tooltipTextColor)
            self.tooltip.setPos(act_pos)
            self.tooltip.show()
            self.tooltip.setZValue(10)   # to make it appear on top of spots
        else:
            self.tooltip.hide()

    def zoomButtonClicked(self):
        self.pgScatterPlotItem.getViewBox().setMouseMode(self.pgScatterPlotItem.getViewBox().RectMode)

    def panButtonClicked(self):
        self.pgScatterPlotItem.getViewBox().setMouseMode(self.pgScatterPlotItem.getViewBox().PanMode)

    def selectButtonClicked(self):
        self.pgScatterPlotItem.getViewBox().setMouseMode(self.pgScatterPlotItem.getViewBox().RectMode)

    def setData(self, data, subsetData = None, **args):
        # OWPlot.setData(self, data)
        self.pgPlotWidget.clear() # OWPlot.setData() je v bistvu naredil samo clear() ???
        self.oldLegendKeys = {}
        ScaleScatterPlotData.set_data(self, data, subsetData, **args)

    #########################################################
    # update shown data. Set labels, coloring by className ....
    def updateData(self, xAttr, yAttr, colorAttr, shapeAttr = "", sizeShapeAttr = "", labelAttr = None, **args):
        # self.legend().clear()
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
        if shapeAttr != "" and shapeAttr != "(Same shape)" and len(self.data_domain[shapeAttr].values) < 11:
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
            xmin = xVarMin - (self.jitter_size + 10.)/100.
            xmax = xVarMax + (self.jitter_size + 10.)/100.
            labels = get_variable_values_sorted(self.data_domain[xAttrIndex])
        else:
            off  = (xVarMax - xVarMin) * (self.jitter_size * self.jitter_continuous + 2) / 100.0
            xmin = xVarMin - off
            xmax = xVarMax + off
            labels = None
        self.setXlabels(labels)
        self.set_axis_scale(xBottom, xmin, xmax,  discreteX)

        # set axis for y attribute
        discreteY = self.data_domain[yAttrIndex].var_type == Orange.data.Variable.VarTypes.Discrete
        if discreteY:
            yVarMax -= 1; yVar -= 1
            ymin = yVarMin - (self.jitter_size + 10.)/100.
            ymax = yVarMax + (self.jitter_size + 10.)/100.
            labels = get_variable_values_sorted(self.data_domain[yAttrIndex])
        else:
            off  = (yVarMax - yVarMin) * (self.jitter_size * self.jitter_continuous + 2) / 100.0
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

        ####################################################
        ###  Create a single curve with different points ###
        ####################################################

        def_color = self.color(OWPalette.Data) #QColor(200, 200, 200)
        def_size = self.point_width
        def_shape = self.curveSymbols[0]
        n = len(xData)   # predvidevam da so vsi enako dolgi (tudi yData itd)

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
            colorData = [def_color for i in range(n)]

        if sizeIndex != -1:
            sizeData = [MIN_SHAPE_SIZE + round(i * self.point_width) for i in self.no_jittering_scaled_data[sizeIndex]]
        else:
            sizeData = [def_size for i in range(n)]

        if shapeIndex != -1 and self.data_domain[shapeIndex].var_type == Orange.data.Variable.VarTypes.Discrete:
            shapeData = [self.curveSymbols[int(i)] for i in self.original_data[shapeIndex]]
        else:
            shapeData = [def_shape for i in range(n)]

        # if labelAttr and labelAttr in [self.raw_data.domain.getmeta(mykey).name for mykey in self.raw_data.domain.metas().keys()] + [var.name for var in self.raw_data.domain]:
        if labelAttr and labelAttr in [meta for meta in list(self.raw_data.domain.metas)] + [var.name for var in self.raw_data.domain]:
            if self.raw_data[0][labelAttr].variable.var_type == Orange.data.Variable.VarTypes.Continuous:
                # labelData = ["%4.1f" % Orange.Value(i[labelAttr]) if not i[labelAttr].isSpecial() else "" for i in self.raw_data]
                ## TODO kaj je delala isSpecial()?
                labelData = ["%4.1f" % float(i[labelAttr]) for i in self.raw_data]
            else:
                # labelData = [str(i[labelAttr].value) if not i[labelAttr].isSpecial() else "" for i in self.raw_data]
                labelData = [str(i[labelAttr].value) for i in self.raw_data]
        else:
            labelData = []

        if self.have_subset_data:
            subset_ids = [example.id for example in self.raw_subset_data]
            marked_data = [example.id in subset_ids for example in self.raw_data]   ## TODO kaj je marked_data
        else:
            marked_data = []

        # self.set_main_curve_data(xData, yData, colorData, labelData, sizeData, shapeData, marked_data, validData)
        if self.pgScatterPlotItem:
            self.pgPlotWidget.removeItem(self.pgScatterPlotItem)
        # brushData = [color.lighter(LIGHTER_ARG) if color != def_color else QColor(200, 200, 200) for color in colorData]
        brushData = [color.lighter(LIGHTER_VALUE) for color in colorData]
        # brushData = [color for color in colorData]
        for i in range(len(brushData)):
            brushData[i].setAlpha(INITIAL_ALPHA_VALUE)
        self.pgScatterPlotItem = pg.ScatterPlotItem() # x=xData, y=yData, brush=brushData, pen=colorData)
        spots = [{'pos': (xData[i], yData[i]), 'size': sizeData[i], 'pen': colorData[i], 'brush': brushData[i], 'symbol': shapeData[i], 'data': i} for i in range(n)]
        self.pgScatterPlotItem.addPoints(spots)
        self.pgScatterPlotItem.sigClicked.connect(self.spotItemClicked)
        self.pgScatterPlotItem.selectedPoints = []
        self.pgPlotWidget.addItem(self.pgScatterPlotItem)
        self.pgPlotWidget.addItem(self.tooltip)
        self.pgScatterPlotItem.scene().sigMouseMoved.connect(self.mouseMoved)    # scene() se nastavi sele ko dodas scatterplotitem na plotwidget
        self.main_curve = self.pgScatterPlotItem   ## TODO: kaj je s tem main_curve?

        if labelData:
            for label in self.labels:
                self.pgPlotWidget.removeItem(label)
            self.labels = []
            d = self.get_xy_data_positions(self.shownXAttribute, self.shownYAttribute)
            for (ind, label) in enumerate(labelData):
                ti = pg.TextItem(text=label, color=pg.mkColor(200, 200, 200))
                self.pgPlotWidget.addItem(ti)
                ti.setPos(xData[ind], yData[ind])
                self.labels.append(ti)
        else:
            for label in self.labels:
                self.pgPlotWidget.removeItem(label)
            self.labels = []


        #################################################################
        # Create legend items in any case so that show/hide legend only

        discColorIndex = colorIndex if colorIndex != -1 and self.data_domain[colorIndex].var_type == Orange.data.Variable.VarTypes.Discrete else -1
        discShapeIndex = shapeIndex if shapeIndex != -1 and self.data_domain[shapeIndex].var_type == Orange.data.Variable.VarTypes.Discrete else -1
        discSizeIndex = sizeIndex if sizeIndex != -1 and self.data_domain[sizeIndex].var_type == Orange.data.Variable.VarTypes.Discrete else -1

        self.remove_legend()
        if any([discColorIndex != -1, discShapeIndex != -1, discSizeIndex != -1]):
            self.create_legend(offset=QPoint(self.pgPlotWidget.size().width() - 150, 10))
            index = max(discColorIndex, discShapeIndex, discSizeIndex)
            num = len(self.data_domain[index].values)
            varValues = get_variable_values_sorted(self.data_domain[index])
            for ind in range(num):
                # construct an item to pass to ItemSample
                if discColorIndex != -1:
                    p = QColor(*palette.getRGB(ind))
                    b = p.lighter(LIGHTER_VALUE)
                else:
                    p = def_color
                    b = p.lighter(LIGHTER_VALUE)
                if discSizeIndex != -1:
                    sz = MIN_SHAPE_SIZE + round(ind*self.point_width/len(varValues))
                else:
                    sz = self.point_width
                if discShapeIndex != -1:
                    sym = self.curveSymbols[ind]
                else:
                    sym = self.pgScatterPlotItem.opts['symbol']
                sample = lambda: None
                sample.opts = {'pen': p,
                        'brush': b,
                        'size': sz,
                        'symbol': sym
                }
                self.legend().addItem(item=sample, name=varValues[ind])#, varValues[ind], OWPoint(def_shape, self.discPalette[ind], def_size))

        # ##############################################################
        # draw color scale for continuous coloring attribute
        self.remove_gradient_legend()
        if colorIndex != -1 and showContinuousColorLegend:
            values = [("%%.%df" % self.data_domain[colorAttr].number_of_decimals % v) for v in self.attr_values[colorAttr]]
            self.create_gradient_legend(colorAttr, values=values, parentSize=self.pgPlotWidget.size())

        self.pgPlotWidget.replot()

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
            # self.pgScatterPlotItem.setSize(size=self.point_width)     ## TODO: ne dela?
            points = self.pgScatterPlotItem.points()
            for p in points:
                p.setSize(self.point_width)

    def update_alpha_value(self):
        for p in self.pgScatterPlotItem.points():
            brush = p.brush().color()
            brush.setAlpha(self.alpha_value * (INITIAL_ALPHA_VALUE/255))
            pen = p.pen().color()
            pen.setAlpha(self.alpha_value)
            p.setBrush(brush)
            p.setPen(pen)

        # self.pgPlotWidget.plotItem.setAlpha(self.alpha_value)  ## TODO: zakaj ne dela?

    def update_filled_symbols(self):
        ## TODO: Implement this in Curve.cpp
        pass

    def update_grid(self):
        self.pgPlotWidget.showGrid(x=self.show_grid, y=self.show_grid)

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

    def get_selected_points(self, xData, yData, validData):
        # hoping that the indices will be in same order as raw_data
        ## TODO check if actually selecting the right points
        selectedIndices = [p.data() for p in self.selectedPoints]
        selected = [i in selectedIndices for i in range(len(self.raw_data))]
        unselected = [i not in selectedIndices for i in range(len(self.raw_data))]
        return selected, unselected

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

        if not getattr(self, "potentialsImage", None) or getattr(self, "potentialContext", None) != (rx, ry, self.shownXAttribute, self.shownYAttribute, self.squareGranularity, self.jitter_size, self.jitter_continuous, self.spaceBetweenCells):
            self.potentialContext = (rx, ry, self.shownXAttribute, self.shownYAttribute, self.squareGranularity, self.jitter_size, self.jitter_continuous, self.spaceBetweenCells)
            self.potentialsImageFromClassifier = self.potentialsClassifier

    def set_axis_scale(self, axis_id, min, max, step_size=0):
        # done automagically in pyqtGraph for continuous values; for discrete values, it is handled in setXlabels() and setYLlabels()
        # orange2pyqtgraph_map = { yLeft: 'left', yRight: 'right', xBottom: 'bottom', xTop: 'top'}
        # axis_id = orange2pyqtgraph_map[axis_id]
        # axis = self.pgPlotWidget.getAxis(axis_id)
        pass

    def setXlabels(self, labels):
        # orange labels are the pyqtgraph ticks (values displayed on axes)
        """The format of ticks looks like this:
            [
                [ (majorTickValue1, majorTickString1), (majorTickValue2, majorTickString2), ... ],
                [ (minorTickValue1, minorTickString1), (minorTickValue2, minorTickString2), ... ],
                ...
            ]"""
        axis = self.pgPlotWidget.getAxis('bottom')
        if labels:
            ticks = [[(i, labels[i]) for i in range(len(labels))]]
            axis.setTicks(ticks)
        else:
            axis.setTicks(None)

    def setYLlabels(self, labels):
        # orange labels are the pyqtgraph ticks (values displayed on axes)
        axis = self.pgPlotWidget.getAxis('left')
        if labels:
            ticks = [[(i, labels[i]) for i in range(len(labels))]]
            axis.setTicks(ticks)
        else:
            axis.setTicks(None)

    def setXaxisTitle(self, title):
        self.pgPlotWidget.setLabel(axis='bottom', text=title)

    def setYLaxisTitle(self, title):
        self.pgPlotWidget.setLabel(axis='left', text=title)

    def setShowXaxisTitle(self):
        self.pgPlotWidget.showLabel(axis='bottom', show=self.showXaxisTitle)

    def setShowYLaxisTitle(self):
        self.pgPlotWidget.showLabel(axis='left', show=self.showYLaxisTitle)

    def color(self, role, group = None):
        if group:
            return self.pgPlotWidget.palette().color(group, role)
        else:
            return self.pgPlotWidget.palette().color(role)

    def set_palette(self, p):
        self.pgPlotWidget.setPalette(p)

    def enableGridXB(self, b):
        self.show_grid = b

    def enableGridYL(self, b):
        self.show_grid = b

    def legend(self):
        if hasattr(self, '_legend'):
            return self._legend
        else:
            return None

    def gradient_legend(self):
        if hasattr(self, '_gradient_legend'):
            return self._gradient_legend
        else:
            return None

    def create_legend(self, offset):
        self._legend = pg.graphicsItems.LegendItem.LegendItem(offset=offset)
        self._legend.setParentItem(self.pgPlotWidget.plotItem)
        if self._legend_position:
            # if legend was moved from default position, show on the same position as before
            item_pos = (0, 0)
            parent_pos = (0, 0)
            offset = self._legend_position
        else:
            item_pos = (1, 0)   # 0 - left and top; 1 - right and bottom
            parent_pos = (1, 0) # 0 - left and top; 1 - right and bottom
            offset = (-10, 10)
        self._legend.anchor(itemPos=item_pos, parentPos=parent_pos, offset=offset)
        self.update_legend()

    def remove_legend(self):
        if self._legend:
            self._legend_position = self._legend.pos() # restore position next time it is shown
            self._legend.items = []
            while self._legend.layout.count() > 0:
                self._legend.removeAt(0)
            if self._legend.scene and self._legend.scene():
                self._legend.scene().removeItem(self._legend)
            self._legend = None

    def create_gradient_legend(self, title, values, parentSize):
        ## TODO: legendi se prekrijeta, ce se malo potrudis
        self._gradient_legend = GradientLegendItem(title, self.contPalette, [str(v) for v in values], self.pgPlotWidget.plotItem)
        if self._gradient_legend_position:
            # if legend was moved from default position, show on the same position as before
            item_pos = (0, 0)
            parent_pos = (0, 0)
            offset = self._gradient_legend_position
        else:
            y = 20 if not self._legend else self._legend.boundingRect().y() + self._legend.boundingRect().height() + 25 # shown beneath _legend
            item_pos = (1, 0)   # 0 - left and top; 1 - right and bottom
            parent_pos = (1, 0) # 0 - left and top; 1 - right and bottom
            offset = (-10, y)
        self._gradient_legend.anchor(itemPos=item_pos, parentPos=parent_pos, offset=offset)
        self.update_legend()

    def remove_gradient_legend(self):
        if self._gradient_legend:
            self._gradient_legend_position = self._gradient_legend.pos() # restore position next time it is shown
            parent = self._gradient_legend.parentItem()
            parent.removeItem(self._gradient_legend)   # tole nic ne naredi
            self._gradient_legend.hide()               # tale skrije, rajsi bi vidu ce izbrise
            self._gradient_legend = None

    def update_legend(self):
        if self._legend:
            if (self._legend.isVisible() == self.show_legend):
                return
            self._legend.setVisible(self.show_legend)

        if self._gradient_legend:
            # if (self._gradient_legend.isVisible() == self.show_legend):
            #     return
            self._gradient_legend.setVisible(self.show_legend)

    def send_selection(self):
        if self.auto_send_selection_callback:
            self.auto_send_selection_callback()

    def clear_selection(self):
        # called from zoom/select toolbar button 'clear selection'
        self.pgScatterPlotItem.getViewBox().unselectAllPoints()

    def shuffle_points(self):
        pass
        # if self.main_curve:
        #     self.main_curve.shuffle_points()

    def update_animations(self, use_animations=None):
        if use_animations is not None:
            self.animate_plot = use_animations
            self.animate_points = use_animations

    def setCanvasBackground(self, color):
        # called when closing set colors dialog (ColorPalleteDlg)
        print('setCanvasBackground - color=%s' % color)

    def setGridColor(self, color):
        # called when closing set colors dialog (ColorPalleteDlg)
        print('setGridColor - color=%s' % color)






















##O#O#O#O#O#O#O#O#O#O#OO#O#O#O#O#O#O#O#O#O#OO#O#O#O#O#O#O#O#O#O#O#O#O#OO#O#O#O#O#O#O#O#O#O#O#O#O#O#
#
#
#           CLASS FOR TESTING NEW FEATURES
#               add it to owscatterplot to test
#
##O#O#O#O#O#O#O#O#O#O#OO#O#O#O#O#O#O#O#O#O#OO#O#O#O#O#O#O#O#O#O#O#O#O#OO#O#O#O#O#O#O#O#O#O#O#O#O#O#
class OWScatterPlotGraphQt_test(OWWidget, ScaleScatterPlotData):
    selection_changed = QtCore.Signal()

    def __init__(self, scatterWidget, parent = None, name = "None"):
        OWWidget.__init__(self)
        svb = ScatterViewBox(self)
        self.glw = pg.GraphicsLayoutWidget()
        self.plot = self.glw.addPlot(viewBox=svb)
        self.replot = self.plot ##TODO kaj je to? spodi je metoda z istim imenom
        self.spi = None

        ScaleScatterPlotData.__init__(self)

        # create ViewBox to hold the legends
        self.default_legend_width = 140
        self.add_legend_viewbox()
        self.show_legend = True
        self._legend = None
        self._gradient_legend = None

        self.jitter_continuous = 0
        self.jitter_size = 5
        self.showXaxisTitle = 1
        self.showYLaxisTitle = 1
        self.tooltipShowsAllAttributes = False  # when False, tooltip shows only visible attributes
        self.showProbabilities = 0

        self.tooltipData = []
        self.tooltipTextColor = pg.mkColor(70, 70, 70)
        self.tooltip = pg.TextItem(text='', color=self.tooltipTextColor, border=pg.mkPen(200, 200, 200), fill=pg.mkBrush(250, 250, 150, 200), anchor=(0,0))
        self.tooltip.hide()

        self.labels = []

        self.scatterWidget = scatterWidget
        self.insideColors = None
        self.shownAttributeIndices = []
        self.shownXAttribute = ""
        self.shownYAttribute = ""
        self.squareGranularity = 3
        self.spaceBetweenCells = 1
        self.oldLegendKeys = {}     # TODO kaj dela tale

        self.enableWheelZoom = 1
        self.potentialsCurve = None

        # spodnje dodal ko sem vzel stran class OWPlot
        self.gui = OWPlotGUI(self)
        self.continuous_palette = ContinuousPaletteGenerator(QColor(200, 200, 200), QColor(0, 0, 0), True)
        self.discrete_palette = ColorPaletteGenerator()

        self.animate_plot = True        # TODO plot se animira  -> naredi na koncu
        self.selection_behavior = 0 # Add selection


        # OWScatterPlot needs these:
        self.point_width = 10
        self.show_filled_symbols = True
        self.alpha_value = 255
        self.show_grid = True

        self.curveSymbols = list(range(13))
        self.tips = TooltipManager(self)
        # self.setMouseTracking(True)
        # self.grabGesture(QPinchGesture)
        # self.grabGesture(QPanGesture)

        self.state = NOTHING
        self._pressed_mouse_button = 0 # Qt.NoButton
        self._pressed_point = None
        self.selection_items = []
        self._current_rs_item = None
        self._current_ps_item = None
        self.polygon_close_treshold = 10
        self.sendSelectionOnUpdate = True
        self.auto_send_selection_callback = None

        self.data_range = {}
        self.map_transform = QTransform()
        self.graph_area = QRectF()

        self.selectedPoints = []

        # iz OWWidget; zdaj zakomentirano ker je OWWidget baseclass
        # self.callbackDeposit = []
        # self.controlledAttributes = ControlledAttributesDict

        self.update_grid()

    def spotItemClicked(self, plot, points):
        self.spi.getViewBox().unselectAllPoints()
        for p in points:
            toSelectedColor(p) ## TODO tale nastavi points[i] na None?
            self.selectedPoints.append(p)
        self.selection_changed.emit()
        ## TODO ko je graf v pobarvan po continuous atributu in ko selectam samo eno tocko, se zgodi cudnost
        ## TODO vcasih je tezko kliknit na tocko?

    def mouseMoved(self, pos):
        ## TODO: tooltip naj se prikaze na visible area --> ce npr. pokazes na tocko cisto spodaj, naj bo tooltip nad tocko
        act_pos = self.spi.mapFromScene(pos)
        points = self.spi.pointsAt(act_pos)
        text = []
        if len(points):
            for (i, p) in enumerate(points):
                index = p.data()
                if self.tooltipShowsAllAttributes:
                    text.append('Attributes:')
                    for attr in self.data_domain.attributes:
                        text.append('\n   %s = %4.1f' % (attr.name,
                                                         self.raw_data[index][attr]))
                    if self.data_domain.class_var:
                        text.append('\nClass:\n   %s = %s' % (self.data_domain.class_var.name,
                                                              self.raw_data[index][self.data_domain.class_var]))
                else:
                    text.append('Attributes:\n   %s = %4.1f\n   %s = %4.1f' % (self.shownXAttribute,
                                                                               self.raw_data[index][self.shownXAttribute],
                                                                               self.shownYAttribute,
                                                                               self.raw_data[index][self.shownYAttribute]))
                    if self.data_domain.class_var:
                        text.append('\nClass:\n   %s = %s' % (
                            self.data_domain.class_var.name,
                            self.raw_data[index][self.raw_data.domain.class_var]))
                if len(points) > 1 and i < len(points) - 1:
                    text.append('\n------------------\n')
            self.tooltip.setText(''.join(text), color=self.tooltipTextColor)
            self.tooltip.setPos(act_pos)
            self.tooltip.show()
            self.tooltip.setZValue(10)   # to make it appear on top of spots
        else:
            self.tooltip.hide()

    def zoomButtonClicked(self):
        self.spi.getViewBox().setMouseMode(self.spi.getViewBox().RectMode)

    def panButtonClicked(self):
        self.spi.getViewBox().setMouseMode(self.spi.getViewBox().PanMode)

    def selectButtonClicked(self):
        self.spi.getViewBox().setMouseMode(self.spi.getViewBox().RectMode)

    def setData(self, data, subsetData = None, **args):
        # OWPlot.setData(self, data)
        self.plot.clear() # OWPlot.setData() je v bistvu naredil samo clear() ???
        self.oldLegendKeys = {}
        ScaleScatterPlotData.set_data(self, data, subsetData, **args)

    #########################################################
    # update shown data. Set labels, coloring by className ....
    def updateData(self, xAttr, yAttr, colorAttr, shapeAttr = "", sizeShapeAttr = "", labelAttr = None, **args):
        # self.legend().clear()
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
        if shapeAttr != "" and shapeAttr != "(Same shape)" and len(self.data_domain[shapeAttr].values) < 11:
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
            xmin = xVarMin - (self.jitter_size + 10.)/100.
            xmax = xVarMax + (self.jitter_size + 10.)/100.
            labels = get_variable_values_sorted(self.data_domain[xAttrIndex])
        else:
            off  = (xVarMax - xVarMin) * (self.jitter_size * self.jitter_continuous + 2) / 100.0
            xmin = xVarMin - off
            xmax = xVarMax + off
            labels = None
        self.setXlabels(labels)
        self.set_axis_scale(xBottom, xmin, xmax,  discreteX)

        # set axis for y attribute
        discreteY = self.data_domain[yAttrIndex].var_type == Orange.data.Variable.VarTypes.Discrete
        if discreteY:
            yVarMax -= 1; yVar -= 1
            ymin = yVarMin - (self.jitter_size + 10.)/100.
            ymax = yVarMax + (self.jitter_size + 10.)/100.
            labels = get_variable_values_sorted(self.data_domain[yAttrIndex])
        else:
            off  = (yVarMax - yVarMin) * (self.jitter_size * self.jitter_continuous + 2) / 100.0
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

        ####################################################
        ###  Create a single curve with different points ###
        ####################################################

        def_color = self.color(OWPalette.Data) # if 'Same color' is chosen
        def_size = self.point_width
        def_shape = self.curveSymbols[0]
        n = len(xData)   # predvidevam da so vsi enako dolgi (tudi yData itd)

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
            colorData = [def_color for i in range(n)]

        if sizeIndex != -1:
            sizeData = [MIN_SHAPE_SIZE + round(i * self.point_width) for i in self.no_jittering_scaled_data[sizeIndex]]
        else:
            sizeData = [def_size for i in range(n)]

        if shapeIndex != -1 and self.data_domain[shapeIndex].var_type == Orange.data.Variable.VarTypes.Discrete:
            shapeData = [self.curveSymbols[int(i)] for i in self.original_data[shapeIndex]]
        else:
            shapeData = [def_shape for i in range(n)]

        # if labelAttr and labelAttr in [self.raw_data.domain.getmeta(mykey).name for mykey in self.raw_data.domain.metas().keys()] + [var.name for var in self.raw_data.domain]:
        if labelAttr and labelAttr in [meta for meta in list(self.raw_data.domain.metas)] + [var.name for var in self.raw_data.domain]:
            if self.raw_data[0][labelAttr].variable.var_type == Orange.data.Variable.VarTypes.Continuous:
                # labelData = ["%4.1f" % Orange.Value(i[labelAttr]) if not i[labelAttr].isSpecial() else "" for i in self.raw_data]
                ## TODO kaj je delala isSpecial()?
                labelData = ["%4.1f" % float(i[labelAttr]) for i in self.raw_data]
            else:
                # labelData = [str(i[labelAttr].value) if not i[labelAttr].isSpecial() else "" for i in self.raw_data]
                labelData = [str(i[labelAttr].value) for i in self.raw_data]
        else:
            labelData = []

        if self.have_subset_data:
            subset_ids = [example.id for example in self.raw_subset_data]
            marked_data = [example.id in subset_ids for example in self.raw_data]   ## TODO kaj je marked_data
        else:
            marked_data = []

        # self.set_main_curve_data(xData, yData, colorData, labelData, sizeData, shapeData, marked_data, validData)

        # brushData = [color.lighter(LIGHTER_ARG) if color != def_color else QColor(200, 200, 200) for color in colorData]
        brushData = [color.lighter(LIGHTER_VALUE) for color in colorData]
        # brushData = [color for color in colorData]
        for i in range(len(brushData)):
            brushData[i].setAlpha(INITIAL_ALPHA_VALUE)
        spots = [{'pos': (xData[i], yData[i]), 'size': sizeData[i], 'pen': colorData[i], 'brush': brushData[i], 'symbol': shapeData[i], 'data': i} for i in range(n)]

        if self.spi:
            self.plot.removeItem(self.spi)
        self.spi = pg.ScatterPlotItem()
        self.spi.addPoints(spots)
        self.spi.sigClicked.connect(self.spotItemClicked)
        self.spi.selectedPoints = []
        self.plot.addItem(self.spi)
        self.plot.addItem(self.tooltip)
        self.spi.scene().sigMouseMoved.connect(self.mouseMoved)    # scene() se nastavi sele ko dodas scatterplotitem na plotwidget
        self.main_curve = self.spi   ## TODO: kaj je s tem main_curve?

        if labelData:
            for label in self.labels:
                self.plot.removeItem(label)
            self.labels = []
            d = self.get_xy_data_positions(self.shownXAttribute, self.shownYAttribute)
            for (ind, label) in enumerate(labelData):
                ti = pg.TextItem(text=label, color=pg.mkColor(200, 200, 200))
                self.plot.addItem(ti)
                ti.setPos(xData[ind], yData[ind])
                self.labels.append(ti)
        else:
            for label in self.labels:
                self.plot.removeItem(label)
            self.labels = []


        #################################################################
        # Create legend items in any case so that show/hide legend only

        discColorIndex = colorIndex if colorIndex != -1 and self.data_domain[colorIndex].var_type == Orange.data.Variable.VarTypes.Discrete else -1
        discShapeIndex = shapeIndex if shapeIndex != -1 and self.data_domain[shapeIndex].var_type == Orange.data.Variable.VarTypes.Discrete else -1
        discSizeIndex = sizeIndex if sizeIndex != -1 and self.data_domain[sizeIndex].var_type == Orange.data.Variable.VarTypes.Discrete else -1


        self.remove_legend()
        if any([discColorIndex != -1, discShapeIndex != -1, discSizeIndex != -1]):
            self.create_legend(offset=QPoint(self.plot.size().width() - 150, 10))
            index = max(discColorIndex, discShapeIndex, discSizeIndex)
            num = len(self.data_domain[index].values)
            varValues = get_variable_values_sorted(self.data_domain[index])
            for ind in range(num):
                # construct an item to pass to ItemSample
                if discColorIndex != -1:
                    p = QColor(*palette.getRGB(ind))
                    b = p.lighter(LIGHTER_VALUE)
                else:
                    p = def_color
                    b = p.lighter(LIGHTER_VALUE)
                if discSizeIndex != -1:
                    sz = MIN_SHAPE_SIZE + round(ind*self.point_width/len(varValues))
                else:
                    sz = self.point_width
                if discShapeIndex != -1:
                    sym = self.curveSymbols[ind]
                else:
                    sym = self.spi.opts['symbol']
                sample = lambda: None
                sample.opts = {'pen': p,
                               'brush': b,
                               'size': sz,
                               'symbol': sym
                }
                self.legend().addItem(item=sample, name=varValues[ind])#, varValues[ind], OWPoint(def_shape, self.discPalette[ind], def_size))

        # ##############################################################
        # draw color scale for continuous coloring attribute
        self.remove_gradient_legend()
        if colorIndex != -1 and showContinuousColorLegend:
            values = [("%%.%df" % self.data_domain[colorAttr].number_of_decimals % v) for v in self.attr_values[colorAttr]]
            self.create_gradient_legend(colorAttr, values=values, parentSize=self.plot.size())

            # self.pgPlotWidget.replot()     #TODO še zmeri ne vem kaj je tale replot()

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
            # self.pgScatterPlotItem.setSize(size=self.point_width)     ## TODO: ne dela?
            points = self.spi.points()
            for p in points:
                p.setSize(self.point_width)

    def update_alpha_value(self):
        for p in self.spi.points():
            brush = p.brush().color()
            brush.setAlpha(self.alpha_value * (INITIAL_ALPHA_VALUE/255))
            pen = p.pen().color()
            pen.setAlpha(self.alpha_value)
            p.setBrush(brush)
            p.setPen(pen)

            # self.pgPlotWidget.plotItem.setAlpha(self.alpha_value)  ## TODO: zakaj ne dela?

    def update_filled_symbols(self):
        ## TODO: Implement this in Curve.cpp
        pass

    def update_grid(self):
        self.plot.showGrid(x=self.show_grid, y=self.show_grid)

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

    def get_selected_points(self, xData, yData, validData):
        # hoping that the indices will be in same order as raw_data
        ## TODO check if actually selecting the right points
        selectedIndices = [p.data() for p in self.selectedPoints]
        selected = [i in selectedIndices for i in range(len(self.raw_data))]
        unselected = [i not in selectedIndices for i in range(len(self.raw_data))]
        return selected, unselected

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

        if not getattr(self, "potentialsImage", None) or getattr(self, "potentialContext", None) != (rx, ry, self.shownXAttribute, self.shownYAttribute, self.squareGranularity, self.jitter_size, self.jitter_continuous, self.spaceBetweenCells):
            self.potentialContext = (rx, ry, self.shownXAttribute, self.shownYAttribute, self.squareGranularity, self.jitter_size, self.jitter_continuous, self.spaceBetweenCells)
            self.potentialsImageFromClassifier = self.potentialsClassifier

    def set_axis_scale(self, axis_id, min, max, step_size=0):
        # done automagically in pyqtGraph for continuous values; for discrete values, it is handled in setXlabels() and setYLlabels()
        # orange2pyqtgraph_map = { yLeft: 'left', yRight: 'right', xBottom: 'bottom', xTop: 'top'}
        # axis_id = orange2pyqtgraph_map[axis_id]
        # axis = self.pgPlotWidget.getAxis(axis_id)
        pass

    def setXlabels(self, labels):
        # orange labels are the pyqtgraph ticks (values displayed on axes)
        """The format of ticks looks like this:
            [
                [ (majorTickValue1, majorTickString1), (majorTickValue2, majorTickString2), ... ],
                [ (minorTickValue1, minorTickString1), (minorTickValue2, minorTickString2), ... ],
                ...
            ]"""
        axis = self.plot.getAxis('bottom')
        if labels:
            ticks = [[(i, labels[i]) for i in range(len(labels))]]
            axis.setTicks(ticks)
        else:
            axis.setTicks(None)

    def setYLlabels(self, labels):
        # orange labels are the pyqtgraph ticks (values displayed on axes)
        axis = self.plot.getAxis('left')
        if labels:
            ticks = [[(i, labels[i]) for i in range(len(labels))]]
            axis.setTicks(ticks)
        else:
            axis.setTicks(None)

    def setXaxisTitle(self, title):
        self.plot.setLabel(axis='bottom', text=title)

    def setYLaxisTitle(self, title):
        self.plot.setLabel(axis='left', text=title)

    def setShowXaxisTitle(self):
        self.plot.showLabel(axis='bottom', show=self.showXaxisTitle)

    def setShowYLaxisTitle(self):
        self.plot.showLabel(axis='left', show=self.showYLaxisTitle)

    def color(self, role, group = None):
        if group:
            return self.plot.palette().color(group, role)
        else:
            return self.plot.palette().color(role)

    def set_palette(self, p):
        self.plot.setPalette(p)

    def enableGridXB(self, b):
        self.show_grid = b

    def enableGridYL(self, b):
        self.show_grid = b

    def add_legend_viewbox(self):
        self.lvb = self.glw.addViewBox()
        self.lvb.setMaximumWidth(self.default_legend_width) #TODO: dinamično določi širino glede na vsebino

    def remove_legend_viewbox(self):
        self.glw.removeItem(self.lvb)
        self.lvb = None

    def create_legend(self, offset):
        self._legend = pg.graphicsItems.LegendItem.LegendItem(offset=offset)
        self.update_legend()

    def remove_legend(self):
        if self._legend:
            self.lvb.removeItem(self._legend)
            self._legend = None

    def legend(self):
        if hasattr(self, '_legend'):
            return self._legend
        else:
            return None

    def create_gradient_legend(self, title, values, parentSize):
        self._gradient_legend = GradientLegendItem(title, self.contPalette, [str(v) for v in values], self.lvb)
        self.update_legend()

    def remove_gradient_legend(self):
        if self._gradient_legend:
            self._gradient_legend.hide()               # tale skrije, rajsi bi vidu ce izbrise
            self._gradient_legend = None

    def update_legend(self):
        # pazi da ne kličeš rekurzivno
        # called when show legend check box is clicked
        if self.show_legend:
            if not self.lvb:
                self.add_legend_viewbox()

            # show spots legend
            if self._legend:
                self._legend.setParentItem(self.lvb)
                item_pos = (1, 0)   # 0 - left and top; 1 - right and bottom
                parent_pos = (1, 0) # 0 - left and top; 1 - right and bottom
                offset = (-10, 10)
                self._legend.anchor(itemPos=item_pos, parentPos=parent_pos, offset=offset)

            # show gradient legend
            if self._gradient_legend:
                # show only when already created (created in updateData()
                self._gradient_legend.setParentItem(self.lvb)
                y = 20 if not self._legend else self._legend.boundingRect().y() + self._legend.boundingRect().height() + 25 # shown beneath _legend
                item_pos = (1, 0)   # 0 - left and top; 1 - right and bottom
                parent_pos = (1, 0) # 0 - left and top; 1 - right and bottom
                offset = (-10, y)
                self._gradient_legend.anchor(itemPos=item_pos, parentPos=parent_pos, offset=offset)

            self.lvb.setMaximumWidth(max(self._legend.boundingRect().width() if self._legend else self.default_legend_width,
                                         self._gradient_legend.boundingRect().width() if self._gradient_legend else self.default_legend_width))
        else:
            self.remove_legend_viewbox()

    def send_selection(self):
        if self.auto_send_selection_callback:
            self.auto_send_selection_callback()

    def clear_selection(self):
        # called from zoom/select toolbar button 'clear selection'
        self.spi.getViewBox().unselectAllPoints()

    def shuffle_points(self):
        pass
        # if self.main_curve:
        #     self.main_curve.shuffle_points()

    def update_animations(self, use_animations=None):
        if use_animations is not None:
            self.animate_plot = use_animations
            self.animate_points = use_animations

    def setCanvasBackground(self, color):
        # called when closing set colors dialog (ColorPalleteDlg)
        print('setCanvasBackground - color=%s' % color)

    def setGridColor(self, color):
        # called when closing set colors dialog (ColorPalleteDlg)
        print('setGridColor - color=%s' % color)





if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)
    c = OWScatterPlotGraphQt_test(None)
    c.show()
    a.exec_()




    # from PyQt4 import QtGui
    # # set-up a Qt Gui window to hold the PlotWidget
    # app = QtGui.QApplication([])
    # mw = QtGui.QMainWindow()
    # mw.resize(800, 800)
    # cw = QtGui.QWidget()
    # mw.setCentralWidget(cw)
    # l = QtGui.QVBoxLayout()
    # cw.setLayout(l)
    #
    # from pyqtgraph import GraphicsLayoutWidget
    # glw = GraphicsLayoutWidget()
    # l.addWidget(glw)
    # plot = glw.addPlot()
    #
    # curve1 = plot.plot([1,4,3,2,3], pen='r', name='red plot')
    # curve2 = plot.plot([-1,2,4,2,3], pen='g', name='green plot')
    #
    #
    # legend = pg.LegendItem()
    # legend.addItem(curve1, name=curve1.opts['name'])
    # legend.addItem(curve2, name=curve2.opts['name'])
    #
    # lvb = glw.addViewBox()
    # legend.setParentItem(lvb)
    # lvb.setMaximumWidth(legend.boundingRect().width() + 10)
    #
    # # show the MainWindow
    # mw.show()
