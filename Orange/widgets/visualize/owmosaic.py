import sys
from collections import defaultdict
from functools import reduce
from itertools import product, chain
from math import sqrt, log
from operator import mul

from PyQt4.QtCore import QPoint, Qt, QRectF, QSize
from PyQt4.QtGui import (QGraphicsRectItem, QGraphicsView, QColor,
                         QGraphicsScene, QPainter, QPen, QApplication,
                         QGraphicsTextItem, QBrush, QGraphicsLineItem,
                         QGraphicsEllipseItem)

from Orange.data import Table, filter
from Orange.data.sql.table import SqlTable, LARGE_TABLE, DEFAULT_SAMPLE_TIME
from Orange.statistics.distribution import get_distribution
from Orange.widgets import gui
from Orange.widgets.settings import (Setting, DomainContextHandler,
                                     ContextSetting)
from Orange.widgets.utils import getHtmlCompatibleString
from Orange.widgets.utils.colorpalette import DefaultRGBColors
from Orange.widgets.utils.scaling import get_variable_values_sorted
from Orange.widgets.widget import OWWidget, Default


class SelectionRectangle(QGraphicsRectItem):
    pass


class MosaicSceneView(QGraphicsView):
    def __init__(self, widget, *args):
        super().__init__(*args)
        self.widget = widget
        self.bMouseDown = False
        self.mouseDownPosition = QPoint(0, 0)
        self.tempRect = None

    def mousePressEvent(self, ev):
        QGraphicsView.mousePressEvent(self, ev)
        self.mouseDownPosition = QPoint(ev.pos().x(), ev.pos().y())
        self.bMouseDown = True
        self.mouseMoveEvent(ev)

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
            rect = QRectF(
                    min(self.mouseDownPosition.x(), ev.pos().x()),
                    min(self.mouseDownPosition.y(), ev.pos().y()),
                    max(abs(self.mouseDownPosition.x() - ev.pos().x()), 1),
                    max(abs(self.mouseDownPosition.y() - ev.pos().y()), 1))
            self.tempRect.setRect(rect)

    def mouseReleaseEvent(self, ev):
        self.bMouseDown = False
        self.widget.key_modifier = ev.modifiers()
        if self.tempRect:
            if ev.button() == Qt.LeftButton and not ev.modifiers() & \
                    (Qt.AltModifier | Qt.ControlModifier | Qt.ShiftModifier):
                self.widget.selectionConditions = []
            self.widget.addSelection(self.tempRect)
            self.scene().removeItem(self.tempRect)
            self.tempRect = None


class OWMosaicDisplay(OWWidget):
    name = "Mosaic Display"
    description = "Display data in a mosaic plot."
    icon = "icons/MosaicDisplay.svg"

    inputs = [("Data", Table, "set_data", Default),
              ("Data Subset", Table, "set_subset_data")]
    outputs = [("Selected Data", Table)]

    settingsHandler = DomainContextHandler()
    use_boxes = Setting(True)
    variable1 = ContextSetting("")
    variable2 = ContextSetting("")
    variable3 = ContextSetting("")
    variable4 = ContextSetting("")
    # interior_coloring is context setting to properly reset it
    # if the widget switches to regression and back (set setData)
    interior_coloring = ContextSetting(1)

    PEARSON, CLASS_DISTRIBUTION = 0, 1
    interior_coloring_opts = ["Pearson residuals",
                              "Class distribution"]
    color_bars_opts = ["Don't show",
                       "Class distribution",
                       "Subset distribution"]

    _apriori_pen_color = QColor(255, 255, 255, 128)
    _box_size = 5
    _cellspace = 4

    graph_name = "canvas"
    want_control_area = False

    def __init__(self):
        super().__init__()

        self.data = None
        self.unprocessed_subset_data = None
        self.subset_data = None
        self.names = []  # class values

        self.attributeNameOffset = 20
        self.attributeValueOffset = 3
        self.residuals = []  # residual values if the residuals are visualized
        self.aprioriDistributions = []
        self.conditionalDict = None
        self.conditionalSubsetDict = None
        self.distributionDict = None
        self.distributionSubsetDict = None

        self.selectionRectangle = None
        self.selectionConditions = []
        self.recentlyAdded = []
        self.key_modifier = Qt.NoModifier

        # color paletes for visualizing pearsons residuals
        self.blue_colors = [QColor(255, 255, 255), QColor(210, 210, 255),
                            QColor(110, 110, 255), QColor(0, 0, 255)]
        self.red_colors = [QColor(255, 255, 255), QColor(255, 200, 200),
                           QColor(255, 100, 100), QColor(255, 0, 0)]
        self.selectionColorPalette = [QColor(*col) for col in DefaultRGBColors]

        cbox = gui.hBox(self.mainArea)
        box = gui.vBox(cbox, box=True)
        self.attr_combos = [
            gui.comboBox(
                    box, self, value="variable{}".format(i),
                    contentsLength=12,
                    callback=self.reset_graph,
                    sendSelectedValue=True, valueType=str)
            for i in range(1, 5)]

        box = gui.vBox(cbox)
        self.rb_colors = gui.radioButtonsInBox(
                box, self, "interior_coloring",
                self.interior_coloring_opts, box=True,#"Interior coloring",
                callback=self.update_graph)
        gui.checkBox(gui.indentedBox(self.rb_colors),
                     self, 'use_boxes', label='Compare with total',
                     callback=self._compare_with_total)

        bbox = gui.hBox(box)
        gui.button(bbox, None, "&Save Graph",
                   callback=self.save_graph, autoDefault=False)
        gui.button(bbox, None, "&Report",
                   callback=self.show_report, autoDefault=False)

        self.canvas = QGraphicsScene()
        self.canvas_view = MosaicSceneView(self, self.canvas, self.mainArea)
        self.mainArea.layout().addWidget(self.canvas_view)
        self.canvas_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.canvas_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.canvas_view.setRenderHint(QPainter.Antialiasing)

    def size(self):
        return QSize(500, 700)

    def _compare_with_total(self):
        self.interior_coloring = 1
        self.update_graph()

    def init_combos(self, data):
        for combo in self.attr_combos:
            combo.clear()
        if data is None:
            return
        for combo in self.attr_combos[1:]:
            combo.addItem("(None)")

        icons = gui.attributeIconDict
        for attr in chain(data.domain, data.domain.metas):
            if attr.is_discrete:
                for combo in self.attr_combos:
                    combo.addItem(icons[attr], attr.name)

        if self.attr_combos[0].count() > 0:
            self.variable1 = self.attr_combos[0].itemText(0)
            self.variable2 = self.attr_combos[1].itemText(
                    2 * (self.attr_combos[1].count() > 2))
        self.variable3 = self.attr_combos[2].itemText(0)
        self.variable4 = self.attr_combos[3].itemText(0)

    def get_attr_list(self):
        return [
            a for a in [self.variable1, self.variable2,
                        self.variable3, self.variable4]
            if a and a != "(None)"]

    def resizeEvent(self, e):
        OWWidget.resizeEvent(self, e)
        self.update_graph()

    def showEvent(self, ev):
        OWWidget.showEvent(self, ev)
        self.update_graph()

    def set_data(self, data):
        if type(data) == SqlTable and data.approx_len() > LARGE_TABLE:
            data = data.sample_time(DEFAULT_SAMPLE_TIME)

        self.closeContext()
        self.data = data
        self.init_combos(self.data)
        self.information([0, 1, 2])
        if not self.data:
            return
        if any(attr.is_continuous for attr in self.data.domain):
            self.information(0, "Data contains continuous variables. "
                                "Discretize the data to use them.")
        """ TODO: check
        if data.has_missing_class():
            self.information(1, "Examples with missing classes were removed.")
        """

        if self.data.domain.class_var is None:
            self.rb_colors.setDisabled(True)
        else:
            self.rb_colors.setDisabled(False)
            disc_class = self.data.domain.has_discrete_class
            self.rb_colors.group.button(2).setDisabled(not disc_class)
            self.interior_coloring = bool(disc_class)
        self.openContext(self.data)

        # if we first received subset we now call setSubsetData to process it
        if self.unprocessed_subset_data:
            self.set_subset_data(self.unprocessed_subset_data)
            self.unprocessed_subset_data = None

    def set_subset_data(self, data):
        if not self.data:
            self.unprocessed_subset_data = data
            self.warning(10)
            return
        try:
            self.subset_data = data.from_table(self.data.domain, data)
            self.warning(10)
        except:
            self.subset_data = None
            self.warning(
                10,
                "'Data' and 'Data Subset' are incompatible" if data is not None
                else "")

    # this is called by widget after setData and setSubsetData are called.
    # this way the graph is updated only once
    def handleNewSignals(self):
        self.reset_graph()

    def reset_graph(self):
        self.removeAllSelections()
        self.update_graph()

    def update_graph(self):
        for item in self.canvas.items():
            if not isinstance(item, SelectionRectangle):
                self.canvas.removeItem(item)

        data = self.data
        if data is None:
            return
        subset = self.subset_data
        attr_list = self.get_attr_list()
        if data.domain.class_var:
            sql = type(data) == SqlTable
            name = not sql and data.name
            # save class_var because it is removed in the next line
            cv = data.domain.class_var
            data = data[:, attr_list + [data.domain.class_var]]
            data.domain.class_var = cv
            if not sql:
                data.name = name
        else:
            data = data[:, attr_list]
        # TODO: check this
        # data = Preprocessor_dropMissing(data)
        if len(data) == 0:
            self.warning(5, "No valid data for current attributes.")
            return
        else:
            self.warning(5)

        if self.interior_coloring == self.PEARSON:
            self.aprioriDistributions = \
                [get_distribution(data, attr) for attr in attr_list]
        else:
            self.aprioriDistributions = []

        def get_max_label_width(attr):
            values = get_variable_values_sorted(self.data.domain[attr])
            maxw = 0
            for val in values:
                t = OWCanvasText(self.canvas, val, 0, 0, bold=0, show=False)
                maxw = max(int(t.boundingRect().width()), maxw)
            return maxw

        # get the maximum width of rectangle
        xoff = 20
        width = 20
        if len(attr_list) > 1:
            text = OWCanvasText(self.canvas, attr_list[1], bold=1, show=0)
            self.max_ylabel_w1 = min(get_max_label_width(attr_list[1]), 150)
            width = 5 + text.boundingRect().height() + \
                self.attributeValueOffset + self.max_ylabel_w1
            xoff = width
            if len(attr_list) == 4:
                text = OWCanvasText(self.canvas, attr_list[3], bold=1, show=0)
                self.max_ylabel_w2 = min(get_max_label_width(attr_list[3]), 150)
                width += text.boundingRect().height() + \
                    self.attributeValueOffset + self.max_ylabel_w2 - 10

        # get the maximum height of rectangle
        height = 100
        yoff = 45
        square_size = min(self.canvas_view.width() - width - 20,
                          self.canvas_view.height() - height - 20)

        if square_size < 0:
            return  # canvas is too small to draw rectangles
        self.canvas_view.setSceneRect(
                0, 0, self.canvas_view.width(), self.canvas_view.height())

        self.drawnSides = dict([(0, 0), (1, 0), (2, 0), (3, 0)])
        self.drawPositions = {}

        self.conditionalDict, self.distributionDict = \
            self.getConditionalDistributions(data, attr_list)
        self.conditionalSubsetDict = self.distributionSubsetDict = None
        if subset:
            self.conditionalSubsetDict, self.distributionSubsetDict = \
                self.getConditionalDistributions(subset, attr_list)

        # draw rectangles
        self.draw_data(
            attr_list, (xoff, xoff + square_size), (yoff, yoff + square_size),
            0, "", len(attr_list))
        self.DrawLegend(
            data, (xoff, xoff + square_size), (yoff, yoff + square_size))

    # create a dictionary "combination-of-attr-values" : count
    # TODO: this function is also used in owsieve --> where to put it?
    def getConditionalDistributions(self, data, attrs):
        cond_dist = defaultdict(int)
        dist = defaultdict(int)
        cond_dist[""] = dist[""] = len(data)
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
                    str_values = [a.repr_val(a.to_val(x))
                                  for a, x in zip(all_attrs, r[:-1])]
                    str_values = [x if x != '?' else 'None' for x in str_values]
                    cond_dist['-'.join(str_values)] = r[-1]
                    dist[str_values[-1]] += r[-1]
            else:
                for indices in product(*(range(len(a.values)) for a in attr)):
                    vals = []
                    conditions = []
                    for k, ind in enumerate(indices):
                        vals.append(attr[k].values[ind])
                        fd = filter.FilterDiscrete(
                                column=attr[k], values=[attr[k].values[ind]])
                        conditions.append(fd)
                    filt = filter.Values(conditions)
                    filtdata = filt(data)
                    cond_dist['-'.join(vals)] = len(filtdata)
                    dist[vals[-1]] += len(filtdata)
        return cond_dist, dist

    def draw_data(self, attrList, x0_x1, y0_y1, side, condition,
                  totalAttrs, used_attrs=[], used_vals=[],
                  attrVals="", **args):
        x0, x1 = x0_x1
        y0, y1 = y0_y1
        if self.conditionalDict[attrVals] == 0:
            self.addRect(x0, x1, y0, y1, "",
                         used_attrs, used_vals, attrVals=attrVals)
            # store coordinates for later drawing of labels
            self.draw_text(side, attrList[0], (x0, x1), (y0, y1), totalAttrs,
                           used_attrs, used_vals, attrVals)
            return

        attr = attrList[0]
        # how much smaller rectangles do we draw
        edge = len(attrList) * self._cellspace
        values = get_variable_values_sorted(self.data.domain[attr])
        if side % 2:
            values = values[::-1]  # reverse names if necessary

        if side % 2 == 0:  # we are drawing on the x axis
            # we remove the space needed for separating different attr. values
            whole = max(0, (x1 - x0) - edge * (
                len(values) - 1))
            if whole == 0:
                edge = (x1 - x0) / float(len(values) - 1)
        else:  # we are drawing on the y axis
            whole = max(0, (y1 - y0) - edge * (len(values) - 1))
            if whole == 0:
                edge = (y1 - y0) / float(len(values) - 1)

        if attrVals == "":
            counts = [self.conditionalDict[val] for val in values]
        else:
            counts = [self.conditionalDict[attrVals + "-" + val]
                      for val in values]
        total = sum(counts)

        # if we are visualizing the third attribute and the first attribute
        # has the last value, we have to reverse the order in which the boxes
        # will be drawn otherwise, if the last cell, nearest to the labels of
        # the fourth attribute, is empty, we wouldn't be able to position the
        # labels
        valRange = list(range(len(values)))
        if len(attrList + used_attrs) == 4 and len(used_attrs) == 2:
            attr1Values = get_variable_values_sorted(
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

            tooltip = condition + 4 * "&nbsp;" + attr + \
                ": <b>" + htmlVal + "</b><br>"
            attrs = used_attrs + [attr]
            vals = used_vals + [val]
            common_args = attrs, vals, newAttrVals
            if side % 2 == 0:  # if we are moving horizontally
                if len(attrList) == 1:
                    self.addRect(x0 + start, x0 + end, y0, y1,
                                 tooltip, *common_args, **args)
                else:
                    self.draw_data(attrList[1:], (x0 + start, x0 + end),
                                   (y0, y1), side + 1,
                                   tooltip, totalAttrs, *common_args, **args)
            else:
                if len(attrList) == 1:
                    self.addRect(x0, x1, y0 + start, y0 + end,
                                 tooltip, *common_args, **args)
                else:
                    self.draw_data(attrList[1:], (x0, x1),
                                   (y0 + start, y0 + end), side + 1,
                                   tooltip, totalAttrs, *common_args, **args)

        self.draw_text(side, attrList[0], (x0, x1), (y0, y1),
                       totalAttrs, used_attrs, used_vals, attrVals)

    def draw_text(self, side, attr, x0_x1, y0_y1,
                  totalAttrs, used_attrs, used_vals, attrVals):
        x0, x1 = x0_x1
        y0, y1 = y0_y1
        if self.drawnSides[side]:
            return

        # the text on the right will be drawn when we are processing
        # visualization of the last value of the first attribute
        if side == 3:
            attr1Values = \
                get_variable_values_sorted(self.data.domain[used_attrs[0]])
            if used_vals[0] != attr1Values[-1]:
                return

        if not self.conditionalDict[attrVals]:
            if side not in self.drawPositions:
                self.drawPositions[side] = (x0, x1, y0, y1)
            return
        else:
            if side in self.drawPositions:
                # restore the positions of attribute values and name
                (x0, x1, y0, y1) = self.drawPositions[side]

        self.drawnSides[side] = True

        values = get_variable_values_sorted(self.data.domain[attr])
        if side % 2:
            values = values[::-1]

        spaces = self._cellspace * (totalAttrs - side) * (len(values) - 1)
        width = x1 - x0 - spaces * (side % 2 == 0)
        height = y1 - y0 - spaces * (side % 2 == 1)

        # calculate position of first attribute
        currPos = 0

        if attrVals == "":
            counts = [self.conditionalDict.get(val, 1) for val in values]
        else:
            counts = [self.conditionalDict.get(attrVals + "-" + val, 1)
                      for val in values]
        total = sum(counts)
        if total == 0:
            counts = [1] * len(values)
            total = sum(counts)

        aligns = [Qt.AlignTop | Qt.AlignHCenter,
                  Qt.AlignRight | Qt.AlignVCenter,
                  Qt.AlignBottom | Qt.AlignHCenter,
                  Qt.AlignLeft | Qt.AlignVCenter]
        align = aligns[side]
        for i in range(len(values)):
            val = values[i]
            perc = counts[i] / float(total)
            if self.distributionDict[val] != 0:
                if side == 0:
                    OWCanvasText(self.canvas, str(val),
                                 x0 + currPos + width * 0.5 * perc,
                                 y1 + self.attributeValueOffset, align)
                elif side == 1:
                    OWCanvasText(self.canvas, str(val),
                                 x0 - self.attributeValueOffset,
                                 y0 + currPos + height * 0.5 * perc, align)
                elif side == 2:
                    OWCanvasText(self.canvas, str(val),
                                 x0 + currPos + width * perc * 0.5,
                                 y0 - self.attributeValueOffset, align)
                else:
                    OWCanvasText(self.canvas, str(val),
                                 x1 + self.attributeValueOffset,
                                 y0 + currPos + height * 0.5 * perc, align)

            if side % 2 == 0:
                currPos += perc * width + self._cellspace * (totalAttrs - side)
            else:
                currPos += perc * height + self._cellspace * (totalAttrs - side)

        if side == 0:
            OWCanvasText(
                self.canvas, attr,
                x0 + (x1 - x0) / 2,
                y1 + self.attributeValueOffset + self.attributeNameOffset,
                align, bold=1)
        elif side == 1:
            OWCanvasText(
                self.canvas, attr,
                x0 - self.max_ylabel_w1 - self.attributeValueOffset,
                y0 + (y1 - y0) / 2,
                align, bold=1, vertical=True)
        elif side == 2:
            OWCanvasText(
                self.canvas, attr,
                x0 + (x1 - x0) / 2,
                y0 - self.attributeValueOffset - self.attributeNameOffset,
                align, bold=1)
        else:
            OWCanvasText(
                self.canvas, attr,
                x1 + self.max_ylabel_w2 + self.attributeValueOffset,
                y0 + (y1 - y0) / 2,
                align, bold=1, vertical=True)

    # draw a rectangle, set it to back and add it to rect list
    def addRect(self, x0, x1, y0, y1, condition="",
                used_attrs=[], used_vals=[], attrVals="", **args):
        if x0 == x1:
            x1 += 1
        if y0 == y1:
            y1 += 1

        # rectangles of width and height 1 are not shown - increase
        if x1 - x0 + y1 - y0 == 2:
            y1 += 1

        if "selectionDict" in args and \
                tuple(used_vals) in args["selectionDict"]:
            d = 2
            OWCanvasRectangle(
                    self.canvas,
                    x0 - d, y0 - d, x1 - x0 + 1 + 2 * d, y1 - y0 + 1 + 2 * d,
                    penColor=args["selectionDict"][tuple(used_vals)],
                    penWidth=2, z=-100)

        if self.data.domain.has_discrete_class:
            colors = [QColor(*col) for col in self.data.domain.class_var.colors]
        else:
            colors = None

        prior = ()
        pearson = None
        expected = None
        outerRect = OWCanvasRectangle(
                self.canvas, x0, y0, x1 - x0, y1 - y0, z=30)

        if not self.conditionalDict[attrVals]:
            return

        # we have to remember which conditions were new in this update so that
        # when we right click we can only remove the last added selections
        if self.selectionRectangle is not None and \
                self.selectionRectangle.collidesWithItem(outerRect):
            if tuple(used_vals) in self.selectionConditions and \
                            self.key_modifier & (Qt.AltModifier
                                                     | Qt.ControlModifier):
                self.selectionConditions.remove(tuple(used_vals))
            elif tuple(used_vals) not in self.selectionConditions:
                self.recentlyAdded += [tuple(used_vals)]
                if self.key_modifier & (Qt.ControlModifier | Qt.ShiftModifier):
                    self.selectionConditions = self.selectionConditions \
                                               + [tuple(used_vals)]
                elif not self.key_modifier & (Qt.AltModifier | Qt.ShiftModifier
                                                  | Qt.ControlModifier):
                    self.selectionConditions = self.recentlyAdded

        # show rectangle selected or not
        if tuple(used_vals) in self.selectionConditions:
            outerRect.setPen(QPen(Qt.black, 3, Qt.DotLine))

        if (self.interior_coloring == self.CLASS_DISTRIBUTION and
                not self.data.domain.has_discrete_class):
            return

        # draw pearsons residuals
        if self.interior_coloring == self.PEARSON:
            s = sum(self.aprioriDistributions[0])
            expected = s * reduce(
                mul,
                (self.aprioriDistributions[i][used_vals[i]] / float(s)
                 for i in range(len(used_vals))))
            actual = self.conditionalDict[attrVals]
            pearson = (actual - expected) / sqrt(expected)
            ind = min(int(log(abs(pearson), 2)), 3)
            color = [self.red_colors, self.blue_colors][pearson > 0][ind]
            OWCanvasRectangle(
                self.canvas, x0, y0, x1 - x0, y1 - y0,
                color, color, z=-20)
        else:
            cls_values = get_variable_values_sorted(self.data.domain.class_var)
            prior = get_distribution(self.data, self.data.domain.class_var.name)
            total = 0
            for i, value in enumerate(cls_values):
                val = self.conditionalDict[attrVals + "-" + value]
                if val == 0:
                    continue
                if i == len(cls_values) - 1:
                    v = y1 - y0 - total
                else:
                    v = ((y1 - y0) * val) / self.conditionalDict[attrVals]
                OWCanvasRectangle(self.canvas, x0, y0 + total, x1 - x0, v,
                                  colors[i], colors[i], z=-20)
                total += v

            # show apriori boxes and lines
            if self.use_boxes and \
                            abs(x1 - x0) > self._box_size and \
                            abs(y1 - y0) > self._box_size:
                apriori = [prior[val] / float(len(self.data))
                           for val in cls_values]
                total1 = 0
                total2 = 0
                OWCanvasLine(self.canvas,
                             x0 + self._box_size, y0, x0 + self._box_size, y1,
                             z=30)

                for i in range(len(cls_values)):
                    val1 = apriori[i]
                    val2 = apriori[i]
                    if i == len(cls_values) - 1:
                        v1 = y1 - y0 - total1
                        v2 = y1 - y0 - total2
                    else:
                        v1 = (y1 - y0) * val1
                        v2 = (y1 - y0) * val2
                    OWCanvasRectangle(self.canvas,
                                      x0, y0 + total2, self._box_size, v2,
                                      colors[i], colors[i], z=20)
                    total1 += v1
                    total2 += v2

            # show subset distribution
            if self.conditionalSubsetDict:
                # show a rect around the box if subset examples belong to this box
                if self.conditionalSubsetDict[attrVals]:
                    counts = [self.conditionalSubsetDict[attrVals + "-" + val] for val in cls_values]
                    if sum(counts) == 1:
                        OWCanvasRectangle(self.canvas, x0 - 2, y0 - 2, x1 - x0 + 5, y1 - y0 + 5,
                                          colors[counts.index(1)],
                                          QColor(Qt.white), penWidth=2, z=-50,
                                          penStyle=Qt.DashLine)

                    if self.subset_data is not None:
                        OWCanvasLine(self.canvas, x1 - self._box_size, y0, x1 - self._box_size, y1, z=30)
                        total = 0
                        for i in range(len(prior)):
                            val = self.conditionalSubsetDict[attrVals + "-" + cls_values[i]]
                            if not self.conditionalSubsetDict[attrVals] or val == 0: continue
                            if i == len(prior) - 1:
                                v = y1 - y0 - total
                            else:
                                v = ((y1 - y0) * val) / float(self.conditionalSubsetDict[attrVals])
                            OWCanvasRectangle(self.canvas, x1 - self._box_size, y0 + total, self._box_size, v,
                                              colors[i], colors[i], z=15)
                            total += v

        tooltipText = condition + "<hr/>"
        if any(prior):
            clsValues = get_variable_values_sorted(self.data.domain.class_var)
            actual = [self.conditionalDict[attrVals + "-" + clsValues[i]]
                      for i in range(len(prior))]
            if sum(actual) > 0:
                apriori = [prior[key] for key in clsValues]
                text = "<br/>".join(
                    "<b>%s</b>: %d / %.1f%% (Expected %.1f / %.1f%%)" % (
                        clsValues[i], actual[i],
                        100.0 * actual[i] / float(sum(actual)),
                        (apriori[i] * sum(actual)) / float(sum(apriori)),
                        100.0 * apriori[i] / float(sum(apriori)))
                    for i in range(len(clsValues)))
                tooltipText += "Instances: " + str(int(sum(actual))) + \
                               "<br><br>" + text[:-4]
        elif pearson and expected:
            tooltipText += "Expected instances: %.1f<br>" \
                           "Actual instances: %d<br>" \
                           "Standardized (Pearson) residual: %.1f" % (
                expected, self.conditionalDict[attrVals], pearson)
        outerRect.setToolTip(tooltipText)

    # draw the class legend below the square
    def DrawLegend(self, data, x0_x1, y0_y1):
        x0, x1 = x0_x1
        y0, y1 = y0_y1
        if (self.interior_coloring == self.CLASS_DISTRIBUTION and
                data.domain.has_continuous_class):
            return

        if self.interior_coloring == self.PEARSON:
            names = ["<-8", "-8:-4", "-4:-2", "-2:2", "2:4", "4:8", ">8", "Residuals:"]
            colors = self.red_colors[::-1] + self.blue_colors[1:]
        else:
            names = get_variable_values_sorted(data.domain.class_var) + [data.domain.class_var.name + ":"]
            colors = [QColor(*col) for col in data.domain.class_var.colors]

        self.names = [OWCanvasText(self.canvas, name, alignment=Qt.AlignVCenter) for name in names]
        totalWidth = sum([text.boundingRect().width() for text in self.names])

        # compute the x position of the center of the legend
        y = y1 + self.attributeNameOffset + self.attributeValueOffset + 35
        distance = 30
        startX = (x0 + x1) / 2 - (totalWidth + (len(names)) * distance) / 2

        self.names[-1].setPos(startX + 15, y)
        self.names[-1].show()
        xOffset = self.names[-1].boundingRect().width() + distance

        size = 8

        for i in range(len(names) - 1):
            if self.interior_coloring == self.PEARSON:
                edgeColor = Qt.black
            else:
                edgeColor = colors[i]

            OWCanvasRectangle(self.canvas, startX + xOffset, y - size / 2, size, size, edgeColor, colors[i])
            self.names[i].setPos(startX + xOffset + 10, y)
            xOffset += distance + self.names[i].boundingRect().width()


    # ########################################
    # cell/example selection
    def sendSelectedData(self):
        selected_data = None
        if self.data and not isinstance(self.data, SqlTable):
            attributes = self.get_attr_list()
            row_indices = []
            for i, row in enumerate(self.data):
                for condition in self.selectionConditions:
                    if len([attr for attr, val in zip(attributes, condition)
                            if row[attr] == val]) == len(condition):
                        row_indices.append(i)
            selected_data = Table.from_table_rows(self.data, row_indices)
        self.send("Selected Data", selected_data)

    # add a new rectangle. update the graph and see which mosaics does it intersect. add this mosaics to the recentlyAdded list
    def addSelection(self, rect):
        self.selectionRectangle = rect
        self.update_graph()
        self.sendSelectedData()
        self.recentlyAdded = []

        self.selectionRectangle = None

    def removeAllSelections(self):
        self.selectionConditions = []
        self.sendSelectedData()

    def show_report(self):
        self.report_plot()

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


class OWCanvasRectangle(QGraphicsRectItem):
    def __init__(self, canvas, x=0, y=0, width=0, height=0,
                 penColor=QColor(128, 128, 128), brushColor=None, penWidth=1,
                 z=0, penStyle=Qt.SolidLine, pen=None, tooltip=None, show=1,
                 onclick=None):
        super().__init__(x, y, width, height, None, canvas)
        self.onclick = onclick
        if brushColor:
            self.setBrush(QBrush(brushColor))
        if pen:
            self.setPen(pen)
        else:
            self.setPen(QPen(QBrush(penColor), penWidth, penStyle))
        self.setZValue(z)
        if tooltip:
            self.setToolTip(tooltip)
        if show:
            self.show()
        else:
            self.hide()

    def mousePressEvent(self, ev):
        if self.onclick:
            self.onclick(self, ev)

def OWCanvasLine(canvas, x1=0, y1=0, x2=0, y2=0, penWidth=2, penColor=QColor(255, 255, 255, 128), pen=None, z=0,
                 tooltip=None, show=1):
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


# test widget appearance
if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWMosaicDisplay()
    ow.show()
    data = Table("zoo.tab")
    ow.set_data(data)
    ow.handleNewSignals()
    a.exec_()
