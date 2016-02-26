import sys
from collections import defaultdict
from functools import reduce
from itertools import product, chain
from math import sqrt, log
from operator import mul

from PyQt4.QtCore import Qt, QSize
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
    selection = ContextSetting({})
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

    # color paletes for visualizing pearsons residuals
    blue_colors = [QColor(255, 255, 255), QColor(210, 210, 255),
                   QColor(110, 110, 255), QColor(0, 0, 255)]
    red_colors = [QColor(255, 255, 255), QColor(255, 200, 200),
                  QColor(255, 100, 100), QColor(255, 0, 0)]
    selectionColorPalette = [QColor(*col) for col in DefaultRGBColors]
    attributeNameOffset = 20
    attributeValueOffset = 3

    graph_name = "canvas"

    def __init__(self):
        super().__init__()

        self.data = None
        self.unprocessed_subset_data = None
        self.subset_data = None

        self.areas = []

        self.canvas = QGraphicsScene()
        self.canvas_view = ViewWithPress(self.canvas,
                                         handler=self.clear_selection)
        self.mainArea.layout().addWidget(self.canvas_view)
        self.canvas_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.canvas_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.canvas_view.setRenderHint(QPainter.Antialiasing)

        box = gui.vBox(self.controlArea, box=True)
        self.attr_combos = [
            gui.comboBox(
                    box, self, value="variable{}".format(i),
                    orientation="horizontal", contentsLength=12,
                    callback=self.reset_graph,
                    sendSelectedValue=True, valueType=str)
            for i in range(1, 5)]
        self.rb_colors = gui.radioButtonsInBox(
                self.controlArea, self, "interior_coloring",
                self.interior_coloring_opts, box="Interior coloring",
                callback=self.update_graph)
        gui.checkBox(gui.indentedBox(self.rb_colors),
                     self, 'use_boxes', label='Compare with total',
                     callback=self._compare_with_total)
        gui.rubber(self.controlArea)

    def size(self):
        return QSize(830, 550)

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

    def clear_selection(self):
        self.selection = {}
        self.update_selection_rects()
        self.send_selection()

    def reset_graph(self):
        self.clear_selection()
        self.update_graph()

    def update_selection_rects(self):
        for i, (attr, vals, area) in enumerate(self.areas):
            if i in self.selection:
                area.setPen(QPen(Qt.black, 3, Qt.DotLine))
            else:
                area.setPen(QPen())

    def select_area(self, index, ev):
        if ev.button() != Qt.LeftButton:
            return
        if ev.modifiers() & Qt.ControlModifier:
            self.selection ^= {index}
        else:
            self.selection = {index}
        self.update_selection_rects()
        self.send_selection()

    def send_selection(self):
        if not self.selection or self.data is None:
            self.send("Selected Data", None)
            return
        filters = []
        for i in self.selection:
            cols, vals, area = self.areas[i]
            filters.append(
                    filter.Values(
                            filter.FilterDiscrete(col, val)
                            for col, val in zip(cols, vals)))
        if len(filters) > 1:
            filters = filter.Values(filters, conjunction=False)
        else:
            filters = filters[0]
        self.send("Selected Data", filters(self.data))

    def show_report(self):
        self.report_plot()

    def update_graph(self):
        def draw_data(attr_list, x0_x1, y0_y1, side, condition,
                      total_attrs, used_attrs=[], used_vals=[],
                      attr_vals=""):
            x0, x1 = x0_x1
            y0, y1 = y0_y1
            if conditionaldict[attr_vals] == 0:
                add_rect(x0, x1, y0, y1, "",
                         used_attrs, used_vals, attr_vals=attr_vals)
                # store coordinates for later drawing of labels
                draw_text(side, attr_list[0], (x0, x1), (y0, y1), total_attrs,
                          used_attrs, used_vals, attr_vals)
                return

            attr = attr_list[0]
            # how much smaller rectangles do we draw
            edge = len(attr_list) * self._cellspace
            values = get_variable_values_sorted(self.data.domain[attr])
            if side % 2:
                values = values[::-1]  # reverse names if necessary

            if side % 2 == 0:  # we are drawing on the x axis
                # remove the space needed for separating different attr. values
                whole = max(0, (x1 - x0) - edge * (
                    len(values) - 1))
                if whole == 0:
                    edge = (x1 - x0) / float(len(values) - 1)
            else:  # we are drawing on the y axis
                whole = max(0, (y1 - y0) - edge * (len(values) - 1))
                if whole == 0:
                    edge = (y1 - y0) / float(len(values) - 1)

            if attr_vals == "":
                counts = [conditionaldict[val] for val in values]
            else:
                counts = [conditionaldict[attr_vals + "-" + val]
                          for val in values]
            total = sum(counts)

            # if we are visualizing the third attribute and the first attribute
            # has the last value, we have to reverse the order in which the
            # boxes will be drawn otherwise, if the last cell, nearest to the
            # labels of the fourth attribute, is empty, we wouldn't be able to
            # position the labels
            valrange = list(range(len(values)))
            if len(attr_list + used_attrs) == 4 and len(used_attrs) == 2:
                attr1values = get_variable_values_sorted(
                        self.data.domain[used_attrs[0]])
                if used_vals[0] == attr1values[-1]:
                    valrange = valrange[::-1]

            for i in valrange:
                start = i * edge + whole * float(sum(counts[:i]) / total)
                end = i * edge + whole * float(sum(counts[:i + 1]) / total)
                val = values[i]
                htmlval = getHtmlCompatibleString(val)
                if attr_vals != "":
                    newattrvals = attr_vals + "-" + val
                else:
                    newattrvals = val

                tooltip = condition + 4 * "&nbsp;" + attr + \
                          ": <b>" + htmlval + "</b><br>"
                attrs = used_attrs + [attr]
                vals = used_vals + [val]
                common_args = attrs, vals, newattrvals
                if side % 2 == 0:  # if we are moving horizontally
                    if len(attr_list) == 1:
                        add_rect(x0 + start, x0 + end, y0, y1,
                                 tooltip, *common_args)
                    else:
                        draw_data(attr_list[1:], (x0 + start, x0 + end),
                                  (y0, y1), side + 1,
                                  tooltip, total_attrs, *common_args)
                else:
                    if len(attr_list) == 1:
                        add_rect(x0, x1, y0 + start, y0 + end,
                                 tooltip, *common_args)
                    else:
                        draw_data(attr_list[1:], (x0, x1),
                                  (y0 + start, y0 + end), side + 1,
                                  tooltip, total_attrs, *common_args)

            draw_text(side, attr_list[0], (x0, x1), (y0, y1),
                      total_attrs, used_attrs, used_vals, attr_vals)

        def draw_text(side, attr, x0_x1, y0_y1,
                      total_attrs, used_attrs, used_vals, attr_vals):
            x0, x1 = x0_x1
            y0, y1 = y0_y1
            if side in drawn_sides:
                return

            # the text on the right will be drawn when we are processing
            # visualization of the last value of the first attribute
            if side == 3:
                attr1values = \
                    get_variable_values_sorted(self.data.domain[used_attrs[0]])
                if used_vals[0] != attr1values[-1]:
                    return

            if not conditionaldict[attr_vals]:
                if side not in draw_positions:
                    draw_positions[side] = (x0, x1, y0, y1)
                return
            else:
                if side in draw_positions:
                    # restore the positions of attribute values and name
                    (x0, x1, y0, y1) = draw_positions[side]

            drawn_sides.add(side)

            values = get_variable_values_sorted(self.data.domain[attr])
            if side % 2:
                values = values[::-1]

            spaces = self._cellspace * (total_attrs - side) * (len(values) - 1)
            width = x1 - x0 - spaces * (side % 2 == 0)
            height = y1 - y0 - spaces * (side % 2 == 1)

            # calculate position of first attribute
            currpos = 0

            if attr_vals == "":
                counts = [conditionaldict.get(val, 1) for val in values]
            else:
                counts = [conditionaldict.get(attr_vals + "-" + val, 1)
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
                if distributiondict[val] != 0:
                    if side == 0:
                        OWCanvasText(self.canvas, str(val),
                                     x0 + currpos + width * 0.5 * perc,
                                     y1 + self.attributeValueOffset, align)
                    elif side == 1:
                        OWCanvasText(self.canvas, str(val),
                                     x0 - self.attributeValueOffset,
                                     y0 + currpos + height * 0.5 * perc, align)
                    elif side == 2:
                        OWCanvasText(self.canvas, str(val),
                                     x0 + currpos + width * perc * 0.5,
                                     y0 - self.attributeValueOffset, align)
                    else:
                        OWCanvasText(self.canvas, str(val),
                                     x1 + self.attributeValueOffset,
                                     y0 + currpos + height * 0.5 * perc, align)

                if side % 2 == 0:
                    currpos += perc * width + self._cellspace * \
                                              (total_attrs - side)
                else:
                    currpos += perc * height + self._cellspace * \
                                              (total_attrs - side)

            if side == 0:
                OWCanvasText(
                        self.canvas, attr,
                        x0 + (x1 - x0) / 2,
                        y1 + self.attributeValueOffset +
                        self.attributeNameOffset,
                        align, bold=1)
            elif side == 1:
                OWCanvasText(
                        self.canvas, attr,
                        x0 - max_ylabel_w1 - self.attributeValueOffset,
                        y0 + (y1 - y0) / 2,
                        align, bold=1, vertical=True)
            elif side == 2:
                OWCanvasText(
                        self.canvas, attr,
                        x0 + (x1 - x0) / 2,
                        y0 - self.attributeValueOffset -
                        self.attributeNameOffset,
                        align, bold=1)
            else:
                OWCanvasText(
                        self.canvas, attr,
                        x1 + max_ylabel_w2 + self.attributeValueOffset,
                        y0 + (y1 - y0) / 2,
                        align, bold=1, vertical=True)

        def add_rect(x0, x1, y0, y1, condition="",
                     used_attrs=[], used_vals=[], attr_vals=""):
            area_index = len(self.areas)
            if x0 == x1:
                x1 += 1
            if y0 == y1:
                y1 += 1

            # rectangles of width and height 1 are not shown - increase
            if x1 - x0 + y1 - y0 == 2:
                y1 += 1

            if class_var.is_discrete:
                colors = [QColor(*col) for col in class_var.colors]
            else:
                colors = None

            def select_area(_, ev):
                self.select_area(area_index, ev)

            def rect(x, y, w, h, z, pen_color=None, brush_color=None, **args):
                if pen_color is None:
                    return OWCanvasRectangle(
                            self.canvas, x, y, w, h, z=z, onclick=select_area,
                            **args)
                if brush_color is None:
                    brush_color = pen_color
                return OWCanvasRectangle(
                        self.canvas, x, y, w, h, pen_color, brush_color, z=z,
                        onclick=select_area, **args)

            outer_rect = rect(x0, y0, x1 - x0, y1 - y0, 30)
            self.areas.append((used_attrs, used_vals, outer_rect))
            if not conditionaldict[attr_vals]:
                return

            if self.interior_coloring == self.PEARSON:
                s = sum(apriori_dists[0])
                expected = s * reduce(
                        mul,
                        (apriori_dists[i][used_vals[i]] / float(s)
                         for i in range(len(used_vals))))
                actual = conditionaldict[attr_vals]
                pearson = (actual - expected) / sqrt(expected)
                ind = min(int(log(abs(pearson), 2)), 3)
                color = [self.red_colors, self.blue_colors][pearson > 0][ind]
                rect(x0, y0, x1 - x0, y1 - y0, -20, color)
                outer_rect.setToolTip(
                        condition + "<hr/>" +
                        "Expected instances: %.1f<br>"
                        "Actual instances: %d<br>"
                        "Standardized (Pearson) residual: %.1f" %
                        (expected, conditionaldict[attr_vals], pearson))
            else:
                cls_values = get_variable_values_sorted(class_var)
                prior = get_distribution(self.data, class_var.name)
                total = 0
                for i, value in enumerate(cls_values):
                    val = conditionaldict[attr_vals + "-" + value]
                    if val == 0:
                        continue
                    if i == len(cls_values) - 1:
                        v = y1 - y0 - total
                    else:
                        v = ((y1 - y0) * val) / conditionaldict[attr_vals]
                    rect(x0, y0 + total, x1 - x0, v, -20, colors[i])
                    total += v

                if self.use_boxes and \
                        abs(x1 - x0) > self._box_size and \
                        abs(y1 - y0) > self._box_size:
                    total = 0
                    OWCanvasLine(
                        self.canvas,
                        x0 + self._box_size, y0, x0 + self._box_size, y1, z=30)
                    n = sum(prior)
                    for i, (val, color) in enumerate(zip(prior, colors)):
                        if i == len(prior) - 1:
                            h = y1 - y0 - total
                        else:
                            h = (y1 - y0) * val / n
                        rect(x0, y0 + total, self._box_size, h, 20, color)
                        total += h

                if conditionalsubsetdict:
                    if conditionalsubsetdict[attr_vals]:
                        counts = [conditionalsubsetdict[attr_vals + "-" + val]
                                  for val in cls_values]
                        if sum(counts) == 1:
                            rect(x0 - 2, y0 - 2, x1 - x0 + 5, y1 - y0 + 5, -550,
                                 colors[counts.index(1)], Qt.white,
                                 penWidth=2, penStyle=Qt.DashLine)
                        if self.subset_data is not None:
                            OWCanvasLine(
                                self.canvas,
                                x1 - self._box_size, y0, x1 - self._box_size,
                                y1, z=30)
                            total = 0
                            n = conditionalsubsetdict[attr_vals]
                            if n:
                                for i, (cls, color) in \
                                        enumerate(zip(cls_values, colors)):
                                    val = conditionalsubsetdict[
                                        attr_vals + "-" + cls]
                                    if val == 0:
                                        continue
                                    if i == len(prior) - 1:
                                        v = y1 - y0 - total
                                    else:
                                        v = ((y1 - y0) * val) / n
                                    rect(x1 - self._box_size, y0 + total,
                                         self._box_size, v, 15, color)
                                    total += v

                actual = [conditionaldict[attr_vals + "-" + cls_values[i]]
                          for i in range(len(prior))]
                n_actual = sum(actual)
                if n_actual > 0:
                    apriori = [prior[key] for key in cls_values]
                    n_apriori = sum(apriori)
                    text = "<br/>".join(
                            "<b>%s</b>: %d / %.1f%% (Expected %.1f / %.1f%%)" %
                            (cls, act, 100.0 * act / n_actual,
                             apr / n_apriori * n_actual, 100.0 * apr / n_apriori
                            )
                            for cls, act, apr in zip(cls_values, actual, apriori
                                                     ))
                else:
                    text = ""
                outer_rect.setToolTip(
                        "{}<hr>Instances: {}<br><br>{}".format(
                                condition, n_actual, text[:-4]))

        def draw_legend(data, x0_x1, y0_y1):
            x0, x1 = x0_x1
            y0, y1 = y0_y1
            if self.interior_coloring == self.PEARSON:
                names = ["<-8", "-8:-4", "-4:-2", "-2:2", "2:4", "4:8", ">8",
                         "Residuals:"]
                colors = self.red_colors[::-1] + self.blue_colors[1:]
            else:
                names = get_variable_values_sorted(class_var) + \
                        [class_var.name + ":"]
                colors = [QColor(*col) for col in class_var.colors]

            names = [OWCanvasText(self.canvas, name, alignment=Qt.AlignVCenter)
                     for name in names]
            totalwidth = sum(text.boundingRect().width() for text in names)

            # compute the x position of the center of the legend
            y = y1 + self.attributeNameOffset + self.attributeValueOffset + 35
            distance = 30
            startx = (x0 + x1) / 2 - (totalwidth + (len(names)) * distance) / 2

            names[-1].setPos(startx + 15, y)
            names[-1].show()
            xoffset = names[-1].boundingRect().width() + distance

            size = 8

            for i in range(len(names) - 1):
                if self.interior_coloring == self.PEARSON:
                    edgecolor = Qt.black
                else:
                    edgecolor = colors[i]

                OWCanvasRectangle(self.canvas, startx + xoffset, y - size / 2,
                                  size, size, edgecolor, colors[i])
                names[i].setPos(startx + xoffset + 10, y)
                xoffset += distance + names[i].boundingRect().width()

        self.canvas.clear()
        self.areas = []

        data = self.data
        if data is None:
            return
        subset = self.subset_data
        attr_list = self.get_attr_list()
        class_var = data.domain.class_var
        if class_var:
            sql = type(data) == SqlTable
            name = not sql and data.name
            # save class_var because it is removed in the next line
            data = data[:, attr_list + [class_var]]
            data.domain.class_var = class_var
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
            apriori_dists = [get_distribution(data, attr) for attr in attr_list]
        else:
            apriori_dists = []

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
            max_ylabel_w1 = min(get_max_label_width(attr_list[1]), 150)
            width = 5 + text.boundingRect().height() + \
                self.attributeValueOffset + max_ylabel_w1
            xoff = width
            if len(attr_list) == 4:
                text = OWCanvasText(self.canvas, attr_list[3], bold=1, show=0)
                max_ylabel_w2 = min(get_max_label_width(attr_list[3]), 150)
                width += text.boundingRect().height() + \
                    self.attributeValueOffset + max_ylabel_w2 - 10

        # get the maximum height of rectangle
        height = 100
        yoff = 45
        square_size = min(self.canvas_view.width() - width - 20,
                          self.canvas_view.height() - height - 20)

        if square_size < 0:
            return  # canvas is too small to draw rectangles
        self.canvas_view.setSceneRect(
                0, 0, self.canvas_view.width(), self.canvas_view.height())

        drawn_sides = set()
        draw_positions = {}

        conditionaldict, distributiondict = \
            get_conditional_distribution(data, attr_list)
        conditionalsubsetdict = None
        if subset:
            conditionalsubsetdict, _ = \
                get_conditional_distribution(subset, attr_list)

        # draw rectangles
        draw_data(
            attr_list, (xoff, xoff + square_size), (yoff, yoff + square_size),
            0, "", len(attr_list))
        draw_legend(
            data, (xoff, xoff + square_size), (yoff, yoff + square_size))
        self.update_selection_rects()


def get_conditional_distribution(data, attrs):
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


class OWCanvasText(QGraphicsTextItem):
    def __init__(self, canvas, text="", x=0, y=0,
                 alignment=Qt.AlignLeft | Qt.AlignTop, bold=0, font=None, z=0,
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
        if tooltip:
            self.setToolTip(tooltip)
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


def OWCanvasLine(canvas, x1=0, y1=0, x2=0, y2=0, penWidth=2,
                 penColor=QColor(255, 255, 255, 128), pen=None, z=0,
                 tooltip=None, show=1):
    r = QGraphicsLineItem(x1, y1, x2, y2, None, canvas)
    if pen is not None:
        r.setPen(pen)
    else:
        r.setPen(QPen(penColor, penWidth))
    r.setZValue(z)
    if tooltip:
        r.setToolTip(tooltip)
    if show:
        r.show()
    else:
        r.hide()

    return r


def OWCanvasEllipse(canvas, x=0, y=0, width=0, height=0, penWidth=1,
                    startAngle=0, angles=360, penColor=Qt.black,
                    brushColor=None, z=0, penStyle=Qt.SolidLine, pen=None,
                    tooltip=None, show=1):
    e = QGraphicsEllipseItem(x, y, width, height, None, canvas)
    e.setZValue(z)
    if brushColor is not None:
        e.setBrush(QBrush(brushColor))
    if pen is not None:
        e.setPen(pen)
    else:
        e.setPen(QPen(penColor, penWidth))
    e.setStartAngle(startAngle)
    e.setSpanAngle(angles * 16)
    if tooltip:
        e.setToolTip(tooltip)
    if show:
        e.show()
    else:
        e.hide()

    return e


class ViewWithPress(QGraphicsView):
    def __init__(self, *args, **kwargs):
        self.handler = kwargs.pop("handler")
        super().__init__(*args, **kwargs)

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        if not ev.isAccepted():
            self.handler()


# test widget appearance
if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWMosaicDisplay()
    ow.show()
    data = Table("zoo.tab")
    ow.set_data(data)
    ow.handleNewSignals()
    a.exec_()
