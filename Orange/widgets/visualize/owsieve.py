from itertools import chain
from math import sqrt, floor, ceil

from PyQt4.QtCore import Qt, QSize
from PyQt4.QtGui import (QGraphicsScene, QColor, QPen, QBrush,
                         QDialog, QApplication, QSizePolicy, QGraphicsLineItem)

from Orange.data import Table, filter
from Orange.data.sql.table import SqlTable, LARGE_TABLE, DEFAULT_SAMPLE_TIME
from Orange.statistics.contingency import get_contingency
from Orange.widgets import gui
from Orange.widgets.settings import DomainContextHandler, ContextSetting
from Orange.widgets.utils import getHtmlCompatibleString
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.visualize.owmosaic import (
    CanvasText, CanvasRectangle, ViewWithPress, get_conditional_distribution)
from Orange.widgets.widget import OWWidget, Default, AttributeList


class OWSieveDiagram(OWWidget):
    name = "Sieve Diagram"
    description = "A two-way contingency table providing information on the " \
                  "relation between the observed and expected frequencies " \
                  "of a combination of feature values under the assumption of independence."
    icon = "icons/SieveDiagram.svg"
    priority = 4200

    inputs = [("Data", Table, "set_data", Default),
              ("Features", AttributeList, "set_input_features")]
    outputs = [("Selection", Table)]

    graph_name = "canvas"

    want_control_area = False

    settingsHandler = DomainContextHandler()
    attrX = ContextSetting("")
    attrY = ContextSetting("")
    selection = ContextSetting(set())

    def __init__(self):
        super().__init__()

        self.data = None
        self.input_features = None
        self.attrs = []

        self.attr_box = gui.hBox(self.mainArea)
        model = VariableListModel()
        model.wrap(self.attrs)
        self.attrXCombo = gui.comboBox(
            self.attr_box, self, value="attrX", contentsLength=12,
            callback=self.change_attr, sendSelectedValue=True, valueType=str)
        self.attrXCombo.setModel(model)
        gui.widgetLabel(self.attr_box, "\u2715").\
            setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.attrYCombo = gui.comboBox(
            self.attr_box, self, value="attrY", contentsLength=12,
            callback=self.change_attr, sendSelectedValue=True, valueType=str)
        self.attrYCombo.setModel(model)

        self.canvas = QGraphicsScene()
        self.canvasView = ViewWithPress(self.canvas, self.mainArea,
                                         handler=self.reset_selection)
        self.mainArea.layout().addWidget(self.canvasView)
        self.canvasView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.canvasView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        box = gui.hBox(self.mainArea)
        box.layout().addWidget(self.graphButton)
        box.layout().addWidget(self.report_button)

    def sizeHint(self):
        return QSize(450, 550)

    def set_data(self, data):
        if type(data) == SqlTable and data.approx_len() > LARGE_TABLE:
            data = data.sample_time(DEFAULT_SAMPLE_TIME)

        self.closeContext()
        self.data = data
        self.areas = []
        if self.data is None:
            self.attrs[:] = []
        else:
            self.attrs[:] = [
                var for var in chain(self.data.domain,
                                     self.data.domain.metas)
                if var.is_discrete
            ]
        if self.attrs:
            self.attrX = self.attrs[0].name
            self.attrY = self.attrs[len(self.attrs) > 1].name
        else:
            self.attrX = self.attrY = None
        self.openContext(self.data)

        self.information(0, "")
        if data and any(attr.is_continuous for attr in data.domain):
            self.information(0, "Data contains continuous variables. "
                                "Discretize the data to use them.")
        self.resolve_shown_attributes()
        self.update_selection()

    def change_attr(self):
        self.selection = set()
        self.updateGraph()
        self.update_selection()

    def set_input_features(self, attrList):
        self.input_features = attrList
        self.resolve_shown_attributes()
        self.update_selection()

    def resolve_shown_attributes(self):
        self.warning(1)
        self.attr_box.setEnabled(True)
        if self.input_features:  # non-None and non-empty!
            features = [f for f in self.input_features if f in self.attrs]
            if not features:
                self.warning(1, "Features from the input signal "
                                "are not present in the data")
            else:
                old_attrs = self.attrX, self.attrY
                self.attrX, self.attrY = [f.name for f in (features * 2)[:2]]
                self.attr_box.setEnabled(False)
                if (self.attrX, self.attrY) != old_attrs:
                    self.selection = set()
        # else: do nothing; keep current features, even if input with the
        # features just changed to None
        self.updateGraph()

    def resizeEvent(self, e):
        OWWidget.resizeEvent(self,e)
        self.updateGraph()

    def showEvent(self, ev):
        OWWidget.showEvent(self, ev)
        self.updateGraph()

    def reset_selection(self):
        self.selection = set()
        self.update_selection()

    def select_area(self, area, ev):
        if ev.button() != Qt.LeftButton:
            return
        index = self.areas.index(area)
        if ev.modifiers() & Qt.ControlModifier:
            self.selection ^= {index}
        else:
            self.selection = {index}
        self.update_selection()

    def update_selection(self):
        if self.areas is None or not self.selection:
            self.send("Selection", None)
            return

        filters = []
        for i, area in enumerate(self.areas):
            if i in self.selection:
                width = 4
                val_x, val_y = area.value_pair
                filters.append(
                    filter.Values([
                        filter.FilterDiscrete(self.attrX, [val_x]),
                        filter.FilterDiscrete(self.attrY, [val_y])
                    ]))
            else:
                width = 1
            pen = area.pen()
            pen.setWidth(width)
            area.setPen(pen)
        if len(filters) == 1:
            filters = filters[0]
        else:
            filters = filter.Values(filters, conjunction=False)
        self.send("Selection", filters(self.data))

    # -----------------------------------------------------------------------
    # Everything from here on is ancient and has been changed only according
    # to what has been changed above. Some clean-up may be in order some day
    #
    def updateGraph(self, *args):
        for item in self.canvas.items():
            self.canvas.removeItem(item)
        if self.data is None or len(self.data) == 0 or \
                self.attrX is None or self.attrY is None:
            return
        data = self.data[:, [self.attrX, self.attrY]]
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

        contXY, _ = get_conditional_distribution(
            data, [data.domain[self.attrX], data.domain[self.attrY]])
        # compute probabilities
        probs = {}
        for i in range(len(valsX)):
            valx = valsX[i]
            for j in range(len(valsY)):
                valy = valsY[j]
                try:
                    actualProb = contXY['%s-%s' %(data.domain[self.attrX].values[i], data.domain[self.attrY].values[j])]
                    # for val in contXY['%s-%s' %(i, j)]: actualProb += val
                except:
                    actualProb = 0
                probs['%s-%s' %(data.domain[self.attrX].values[i], data.domain[self.attrY].values[j])] = ((data.domain[self.attrX].values[i], valx), (data.domain[self.attrY].values[j], valy), actualProb, len(data))

        #get text width of Y labels
        max_ylabel_w = 0
        for j in range(len(valsY)):
            xl = CanvasText(self.canvas, "", 0, 0, html_text= getHtmlCompatibleString(data.domain[self.attrY].values[j]), show=False)
            max_ylabel_w = max(int(xl.boundingRect().width()), max_ylabel_w)
        max_ylabel_w = min(max_ylabel_w, 200) #upper limit for label widths
        # get text width of Y attribute name
        text = CanvasText(self.canvas, data.domain[self.attrY].name, x  = 0, y = 0, bold = 1, show = 0, vertical=True)
        xOff = int(text.boundingRect().height() + max_ylabel_w)
        yOff = 55
        sqareSize = min(self.canvasView.width() - xOff - 35, self.canvasView.height() - yOff - 50)
        sqareSize = max(sqareSize, 10)
        self.canvasView.setSceneRect(0, 0, self.canvasView.width(), self.canvasView.height())

        # print graph name
        name  = "<b>P(%s, %s) &#8800; P(%s)&times;P(%s)</b>" %(self.attrX, self.attrY, self.attrX, self.attrY)
        CanvasText(self.canvas, "", xOff + sqareSize / 2, 20, Qt.AlignCenter, html_text= name)
        CanvasText(self.canvas, "N = " + str(len(data)), xOff + sqareSize / 2, 38, Qt.AlignCenter, bold = 0)

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
        self.areas = []
        for i in range(len(valsX)):
            if valsX[i] == 0: continue
            currY = yOff
            width = int(float(sqareSize * valsX[i])/float(normX))

            for j in range(len(valsY)-1, -1, -1):   # this way we sort y values correctly
                ((xAttr, xVal), (yAttr, yVal), actual, sum_) = probs['%s-%s' %(data.domain[self.attrX].values[i], data.domain[self.attrY].values[j])]
                if valsY[j] == 0: continue
                height = int(float(sqareSize * valsY[j])/float(normY))

                # create rectangle
                selected = len(self.areas) in self.selection
                rect = CanvasRectangle(
                    self.canvas, currX+2, currY+2, width-4, height-4, z = -10,
                    onclick=self.select_area)
                rect.value_pair = i, j
                self.areas.append(rect)
                self.addRectIndependencePearson(rect, currX+2, currY+2, width-4, height-4, (xAttr, xVal), (yAttr, yVal), actual, sum_,
                    width=1 + 3 * selected,  # Ugly! This is needed since
                    # resize redraws the graph! When this is handled by resizing
                    # just the viewer, update_selection will take care of this
                    )

                expected = float(xVal*yVal)/float(sum_)
                pearson = (actual - expected) / sqrt(expected)
                tooltipText = """<b>X Attribute: %s</b><br>Value: <b>%s</b><br>Number of instances (p(x)): <b>%d (%.2f%%)</b><hr>
                                <b>Y Attribute: %s</b><br>Value: <b>%s</b><br>Number of instances (p(y)): <b>%d (%.2f%%)</b><hr>
                                <b>Number Of Instances (Probabilities):</b><br>Expected (p(x)p(y)): <b>%.1f (%.2f%%)</b><br>Actual (p(x,y)): <b>%d (%.2f%%)</b>
                                <hr><b>Statistics:</b><br>Chi-square: <b>%.2f</b><br>Standardized Pearson residual: <b>%.2f</b>""" %(self.attrX, getHtmlCompatibleString(xAttr), xVal, 100.0*float(xVal)/float(sum_), self.attrY, getHtmlCompatibleString(yAttr), yVal, 100.0*float(yVal)/float(sum_), expected, 100.0*float(xVal*yVal)/float(sum_*sum_), actual, 100.0*float(actual)/float(sum_), chisquare, pearson )
                rect.setToolTip(tooltipText)

                currY += height
                if currX == xOff:
                    CanvasText(self.canvas, "", xOff, currY - height / 2, Qt.AlignRight | Qt.AlignVCenter, html_text= getHtmlCompatibleString(data.domain[self.attrY].values[j]))

            xl = CanvasText(self.canvas, "", currX + width / 2, yOff + sqareSize, Qt.AlignHCenter | Qt.AlignTop, html_text= getHtmlCompatibleString(data.domain[self.attrX].values[i]))
            max_xlabel_h = max(int(xl.boundingRect().height()), max_xlabel_h)

            currX += width

        # show attribute names
        CanvasText(self.canvas, self.attrY, 0, yOff + sqareSize / 2, Qt.AlignLeft | Qt.AlignVCenter, bold = 1, vertical=True)
        CanvasText(self.canvas, self.attrX, xOff + sqareSize / 2, yOff + sqareSize + max_xlabel_h, Qt.AlignHCenter | Qt.AlignTop, bold = 1)


    ######################################################################
    ## show deviations from attribute independence with standardized pearson residuals
    def addRectIndependencePearson(self, rect, x, y, w, h, xAttr_xVal, yAttr_yVal, actual, sum, width):
        xAttr, xVal = xAttr_xVal
        yAttr, yVal = yAttr_yVal
        expected = float(xVal*yVal)/float(sum)
        pearson = (actual - expected) / sqrt(expected)

        if pearson > 0:     # if there are more examples that we would expect under the null hypothesis
            intPearson = floor(pearson)
            pen = QPen(QColor(0,0,255), width); rect.setPen(pen)
            b = 255
            r = g = 255 - intPearson*20
            r = g = max(r, 55)  #
        elif pearson < 0:
            intPearson = ceil(pearson)
            pen = QPen(QColor(255,0,0), width)
            rect.setPen(pen)
            r = 255
            b = g = 255 + intPearson*20
            b = g = max(b, 55)
        else:
            pen = QPen(QColor(255,255,255), width)
            r = g = b = 255         # white
        color = QColor(r,g,b)
        brush = QBrush(color)
        rect.setBrush(brush)

        if pearson > 0:
            pearson = min(pearson, 10)
            kvoc = 1 - 0.08 * pearson       #  if pearson in [0..10] --> kvoc in [1..0.2]
        else:
            pearson = max(pearson, -10)
            kvoc = 1 - 0.4*pearson

        pen.setWidth(1)
        self.addLines(x,y,w,h, kvoc, pen)


    ##################################################
    # add lines
    def addLines(self, x, y, w, h, diff, pen):
        if w == 0 or h == 0:
            return

        dist = 20 * diff  # original distance between two lines in pixels
        temp = dist
        canvas = self.canvas
        while temp < w:
            r = QGraphicsLineItem(temp + x, y, temp + x, y + h, None)
            canvas.addItem(r)
            r.setPen(pen)
            temp += dist

        temp = dist
        while temp < h:
            r = QGraphicsLineItem(x, y + temp, x + w, y + temp, None)
            canvas.addItem(r)
            r.setPen(pen)
            temp += dist

    def closeEvent(self, ce):
        QDialog.closeEvent(self, ce)

    def get_widget_name_extension(self):
        if self.data is not None:
            return "{} vs {}".format(self.attrX, self.attrY)

    def send_report(self):
        self.report_plot()


# test widget appearance
if __name__ == "__main__":
    import sys
    a=QApplication(sys.argv)
    ow=OWSieveDiagram()
    ow.show()
    data = Table(r"zoo.tab")
    ow.set_data(data)
    a.exec_()
    ow.saveSettings()
