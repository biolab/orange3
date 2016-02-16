from collections import defaultdict
from itertools import product
from math import sqrt, floor, ceil
import sys

from PyQt4.QtCore import Qt, QSize
from PyQt4.QtGui import (QGraphicsScene, QGraphicsView, QColor, QPen, QBrush,
                         QDialog, QApplication, QSizePolicy)

import Orange
from Orange.data import Table
from Orange.data.sql.table import SqlTable, LARGE_TABLE, DEFAULT_SAMPLE_TIME
from Orange.statistics.contingency import get_contingency
from Orange.widgets import gui
from Orange.widgets.utils import getHtmlCompatibleString
from Orange.widgets.visualize.owmosaic import (OWCanvasText, OWCanvasRectangle,
                                               OWCanvasLine)
from Orange.widgets.widget import OWWidget, Default, AttributeList


class OWSieveDiagram(OWWidget):
    name = "Sieve Diagram"
    description = "A two-way contingency table providing information in " \
                  "relation to expected frequency of combination of feature " \
                  "values under independence."
    icon = "icons/SieveDiagram.svg"
    priority = 4200

    inputs = [("Data", Table, "setData", Default),
              ("Features", AttributeList, "setShownAttributes")]
    outputs = []

    graph_name = "canvas"

    want_control_area = False

    def __init__(self):
        super().__init__()

        self.data = None

        self.attrX = ""
        self.attrY = ""
        self.attributeSelectionList = None
        self.stopCalculating = 0

        controlArea = gui.hBox(self.mainArea)
        self.attrXCombo = gui.comboBox(
            controlArea, self, value="attrX", contentsLength=12,
            callback=self.updateGraph, sendSelectedValue=True, valueType=str)
        gui.widgetLabel(controlArea, "\u2715").setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.attrYCombo = gui.comboBox(
            controlArea, self, value="attrY", contentsLength=12,
            callback=self.updateGraph, sendSelectedValue=True, valueType=str)

        self.canvas = QGraphicsScene()
        self.canvasView = QGraphicsView(self.canvas, self.mainArea)
        self.mainArea.layout().addWidget(self.canvasView)
        self.canvasView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.canvasView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # want_control_area = False also disables creation of these buttons
        box = gui.hBox(self.mainArea)
        self.graphButton = gui.button(
                box, None, "&Save Graph", callback=self.save_graph)
        self.graphButton.setAutoDefault(0)
        self.report_button = gui.button(
                box, None, "&Report", callback=self.show_report)

    def sizeHint(self):
        return QSize(450, 550)

    # receive new data and update all fields
    def setData(self, data):
        if type(data) == SqlTable and data.approx_len() > LARGE_TABLE:
            data = data.sample_time(DEFAULT_SAMPLE_TIME)

        self.information(0)
        self.information(1)
        sameDomain = self.data and data and self.data.domain.checksum() == data.domain.checksum() # preserve attribute choice if the domain is the same
        # self.data = self.optimizationDlg.setData(data, 0)
        self.data = data

        if not sameDomain:
            self.initCombos()

        self.warning(0, "")
        if data:
            if any(attr.is_continuous for attr in data.domain):
                self.warning(0, "Data contains continuous variables. " +
                             "Discretize the data to use them.")

        self.setShownAttributes(self.attributeSelectionList)

    ## Attribute selection signal
    def setShownAttributes(self, attrList):
        self.attributeSelectionList = attrList
        if self.data and self.attributeSelectionList and len(attrList) >= 2:
            attrs = [attr.name for attr in self.data.domain]
            if attrList[0] in attrs and attrList[1] in attrs:
                self.attrX = attrList[0]
                self.attrY = attrList[1]
        self.updateGraph()

    def getData(self, xAttr=None, yAttr=None, dropMissingData=1):
        if not self.data:
            return None
        xAttr = xAttr or self.attrX
        yAttr = yAttr or self.attrY
        if not (xAttr and yAttr):
            return
        return self.data[:, [xAttr, yAttr]]

    # initialize lists for shown and hidden attributes
    def initCombos(self):
        self.attrXCombo.clear()
        self.attrYCombo.clear()

        if not self.data: return
        for i, var in enumerate(self.data.domain):
            if var.is_discrete:
                self.attrXCombo.addItem(gui.attributeIconDict[self.data.domain[i]], self.data.domain[i].name)
                self.attrYCombo.addItem(gui.attributeIconDict[self.data.domain[i]], self.data.domain[i].name)

        if self.attrXCombo.count() > 0:
            self.attrX = str(self.attrXCombo.itemText(0))
            self.attrY = str(self.attrYCombo.itemText(self.attrYCombo.count() > 1))
        else:
            self.attrX = None
            self.attrY = None

    def resizeEvent(self, e):
        OWWidget.resizeEvent(self,e)
        self.updateGraph()

    def showEvent(self, ev):
        OWWidget.showEvent(self, ev)
        self.updateGraph()

    ## updateGraph - gets called every time the graph has to be updated
    def updateGraph(self, *args):
        for item in self.canvas.items():
            self.canvas.removeItem(item)    # remove all canvas items
        if not self.data: return
        if not self.attrX or not self.attrY: return

        data = self.getData()
        if not data or len(data) == 0: return

        valsX = []
        valsY = []
        contX = get_contingency(data, self.attrX, self.attrX)
        contY = get_contingency(data, self.attrY, self.attrY)

        # compute contingency of x and y attributes
        for entry in contX:
            sum_ = 0
            try:
                for val in entry: sum_ += val
            except: pass
            valsX.append(sum_)

        for entry in contY:
            sum_ = 0
            try:
                for val in entry: sum_ += val
            except: pass
            valsY.append(sum_)

        contXY = self.getConditionalDistributions(data, [data.domain[self.attrX], data.domain[self.attrY]])

        # compute probabilities
        probs = {}
        for i in range(len(valsX)):
            valx = valsX[i]
            for j in range(len(valsY)):
                valy = valsY[j]

                actualProb = 0
                try:
                    actualProb = contXY['%s-%s' %(data.domain[self.attrX].values[i], data.domain[self.attrY].values[j])]
                    # for val in contXY['%s-%s' %(i, j)]: actualProb += val
                except:
                    actualProb = 0
                probs['%s-%s' %(data.domain[self.attrX].values[i], data.domain[self.attrY].values[j])] = ((data.domain[self.attrX].values[i], valx), (data.domain[self.attrY].values[j], valy), actualProb, len(data))

        #get text width of Y labels
        max_ylabel_w = 0
        for j in range(len(valsY)):
            xl = OWCanvasText(self.canvas, "", 0, 0, htmlText = getHtmlCompatibleString(data.domain[self.attrY].values[j]), show=False)
            max_ylabel_w = max(int(xl.boundingRect().width()), max_ylabel_w)
        max_ylabel_w = min(max_ylabel_w, 200) #upper limit for label widths

        # get text width of Y attribute name
        text = OWCanvasText(self.canvas, data.domain[self.attrY].name, x  = 0, y = 0, bold = 1, show = 0, vertical=True)
        xOff = int(text.boundingRect().height() + max_ylabel_w)
        yOff = 55
        sqareSize = min(self.canvasView.width() - xOff - 35, self.canvasView.height() - yOff - 50)
        if sqareSize < 0: return    # canvas is too small to draw rectangles
        self.canvasView.setSceneRect(0, 0, self.canvasView.width(), self.canvasView.height())

        # print graph name
        name  = "<b>P(%s, %s) &#8800; P(%s)&times;P(%s)</b>" %(self.attrX, self.attrY, self.attrX, self.attrY)
        OWCanvasText(self.canvas, "" , xOff+ sqareSize/2, 20, Qt.AlignCenter, htmlText = name)
        OWCanvasText(self.canvas, "N = " + str(len(data)), xOff+ sqareSize/2, 38, Qt.AlignCenter, bold = 0)

        ######################
        # compute chi-square
        chisquare = 0.0
        for i in range(len(valsX)):
            for j in range(len(valsY)):
                ((xAttr, xVal), (yAttr, yVal), actual, sum_) = probs['%s-%s' %(data.domain[self.attrX].values[i], data.domain[self.attrY].values[j])]
                expected = float(xVal*yVal)/float(sum_)
                if expected == 0: continue
                pearson2 = (actual - expected)*(actual - expected) / expected
                chisquare += pearson2

        ######################
        # draw rectangles
        currX = xOff
        max_xlabel_h = 0

        normX, normY = sum(valsX), sum(valsY)
        for i in range(len(valsX)):
            if valsX[i] == 0: continue
            currY = yOff
            width = int(float(sqareSize * valsX[i])/float(normX))

            for j in range(len(valsY)-1, -1, -1):   # this way we sort y values correctly
                ((xAttr, xVal), (yAttr, yVal), actual, sum_) = probs['%s-%s' %(data.domain[self.attrX].values[i], data.domain[self.attrY].values[j])]
                if valsY[j] == 0: continue
                height = int(float(sqareSize * valsY[j])/float(normY))

                # create rectangle
                rect = OWCanvasRectangle(self.canvas, currX+2, currY+2, width-4, height-4, z = -10)
                self.addRectIndependencePearson(rect, currX+2, currY+2, width-4, height-4, (xAttr, xVal), (yAttr, yVal), actual, sum_)

                expected = float(xVal*yVal)/float(sum_)
                pearson = (actual - expected) / sqrt(expected)
                tooltipText = """<b>X Attribute: %s</b><br>Value: <b>%s</b><br>Number of instances (p(x)): <b>%d (%.2f%%)</b><hr>
                                <b>Y Attribute: %s</b><br>Value: <b>%s</b><br>Number of instances (p(y)): <b>%d (%.2f%%)</b><hr>
                                <b>Number Of Instances (Probabilities):</b><br>Expected (p(x)p(y)): <b>%.1f (%.2f%%)</b><br>Actual (p(x,y)): <b>%d (%.2f%%)</b>
                                <hr><b>Statistics:</b><br>Chi-square: <b>%.2f</b><br>Standardized Pearson residual: <b>%.2f</b>""" %(self.attrX, getHtmlCompatibleString(xAttr), xVal, 100.0*float(xVal)/float(sum_), self.attrY, getHtmlCompatibleString(yAttr), yVal, 100.0*float(yVal)/float(sum_), expected, 100.0*float(xVal*yVal)/float(sum_*sum_), actual, 100.0*float(actual)/float(sum_), chisquare, pearson )
                rect.setToolTip(tooltipText)

                currY += height
                if currX == xOff:
                    OWCanvasText(self.canvas, "", xOff, currY - height/2, Qt.AlignRight | Qt.AlignVCenter, htmlText = getHtmlCompatibleString(data.domain[self.attrY].values[j]))

            xl = OWCanvasText(self.canvas, "", currX + width/2, yOff + sqareSize, Qt.AlignHCenter | Qt.AlignTop, htmlText = getHtmlCompatibleString(data.domain[self.attrX].values[i]))
            max_xlabel_h = max(int(xl.boundingRect().height()), max_xlabel_h)

            currX += width

        # show attribute names
        OWCanvasText(self.canvas, self.attrY, 0, yOff + sqareSize/2, Qt.AlignLeft | Qt.AlignVCenter, bold = 1, vertical=True)
        OWCanvasText(self.canvas, self.attrX, xOff + sqareSize/2, yOff + sqareSize + max_xlabel_h, Qt.AlignHCenter | Qt.AlignTop, bold = 1)

    # create a dictionary with all possible pairs of "combination-of-attr-values" : count
    def getConditionalDistributions(self, data, attrs):
        cond_dist = defaultdict(int)
        all_attrs = [data.domain[a] for a in attrs]
        if data.domain.class_var is not None:
            all_attrs.append(data.domain.class_var)

        for i in range(1, len(all_attrs) + 1):
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
                        fd = Orange.data.filter.FilterDiscrete(column=attr[k], values=[attr[k].values[ind]])
                        conditions.append(fd)
                    filt = Orange.data.filter.Values(conditions)
                    filtdata = filt(data)
                    cond_dist['-'.join(vals)] = len(filtdata)
        return cond_dist

    ######################################################################
    ## show deviations from attribute independence with standardized pearson residuals
    def addRectIndependencePearson(self, rect, x, y, w, h, xAttr_xVal, yAttr_yVal, actual, sum):
        xAttr, xVal = xAttr_xVal
        yAttr, yVal = yAttr_yVal
        expected = float(xVal*yVal)/float(sum)
        pearson = (actual - expected) / sqrt(expected)

        if pearson > 0:     # if there are more examples that we would expect under the null hypothesis
            intPearson = floor(pearson)
            pen = QPen(QColor(0,0,255), 1); rect.setPen(pen)
            b = 255
            r = g = 255 - intPearson*20
            r = g = max(r, 55)  #
        elif pearson < 0:
            intPearson = ceil(pearson)
            pen = QPen(QColor(255,0,0), 1)
            rect.setPen(pen)
            r = 255
            b = g = 255 + intPearson*20
            b = g = max(b, 55)
        else:
            pen = QPen(QColor(255,255,255), 1)
            r = g = b = 255         # white
        color = QColor(r,g,b)
        brush = QBrush(color); rect.setBrush(brush)

        if pearson > 0:
            pearson = min(pearson, 10)
            kvoc = 1 - 0.08 * pearson       #  if pearson in [0..10] --> kvoc in [1..0.2]
        else:
            pearson = max(pearson, -10)
            kvoc = 1 - 0.4*pearson

        self.addLines(x,y,w,h, kvoc, pen)


    ##################################################
    # add lines
    def addLines(self, x,y,w,h, diff, pen):
        if w == 0 or h == 0: return

        # create lines
        dist = 20   # original distance between two lines in pixels
        dist = dist * diff
        temp = dist
        while (temp < w):
            OWCanvasLine(self.canvas, temp+x, y, temp+x, y+h, 1, pen.color())
            temp += dist

        temp = dist
        while (temp < h):
            OWCanvasLine(self.canvas, x, y+temp, x+w, y+temp, 1, pen.color())
            temp += dist

    def closeEvent(self, ce):
        QDialog.closeEvent(self, ce)

    def get_widget_name_extension(self):
        if self.data is not None:
            return "{} vs {}".format(self.attrX, self.attrY)

    def send_report(self):
        self.report_plot()


# test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWSieveDiagram()
    ow.show()
    data = Table(r"zoo.tab")
    ow.setData(data)
    a.exec_()
    ow.saveSettings()
