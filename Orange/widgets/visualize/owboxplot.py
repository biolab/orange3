# -*- coding: utf-8 -*-

import operator
import math
from scipy import stats
import numpy as np

from PyQt4 import QtCore
from PyQt4 import QtGui
import scipy

from Orange.data import  DiscreteVariable, Table, Domain
from Orange.statistics import contingency, distribution

from Orange.widgets import widget, gui
from Orange.widgets.settings import (Setting, DomainContextHandler,
                                     ContextSetting)
from Orange.widgets.utils import datacaching
from Orange.widgets.utils.plot import owaxis

#from OWColorPalette import ColorPixmap, ColorPaletteGenerator

class BoxData:
    def __init__(self, dist, label):
        self.dist = dist
        self.label = label
        self.N = N = np.sum(dist[1])
        if N == 0:
            return
        self.a_min = dist[0, 0]
        self.a_max = dist[0, -1]
        self.mean = np.sum(dist[0] * dist[1]) / N
        self.var = np.sum(dist[1] * (dist[0] - self.mean) ** 2) / N
        s = 0
        thresholds = [N/4, N/2, 3*N/4]
        thresh_i = 0
        q = []
        for i, e in enumerate(dist[1]):
            s += e
            if s >= thresholds[thresh_i]:
                if s == thresholds[thresh_i] and i + 1 < dist.shape[1]:
                    q.append((dist[0, i] + dist[0, i + 1]) / 2)
                else:
                    q.append(dist[0, i])
                thresh_i += 1
                if thresh_i == 3:
                    break
        while len(q) < 3:
            q.append(q[-1])
        self.q25, self.median, self.q75 = q


class BoxItem(QtGui.QGraphicsItemGroup):
    pen_light = QtGui.QPen(QtGui.QBrush(QtGui.QColor(0xff, 0xff, 0xff)), 2)
    pen_dark = QtGui.QPen(QtGui.QBrush(QtGui.QColor(0x33, 0x00, 0xff)), 2)
    pen_dark_dotted = QtGui.QPen(QtGui.QBrush(QtGui.QColor(0x33, 0x00, 0xff)), 2)
    pen_dark_wide = QtGui.QPen(QtGui.QBrush(QtGui.QColor(0x33, 0x00, 0xff)), 4)
    for pen in (pen_dark, pen_light, pen_dark_wide, pen_dark_dotted):
        pen.setCosmetic(True)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
    pen_dark_dotted.setStyle(QtCore.Qt.DotLine)

    def __init__(self, stat, width=20):
        super().__init__()
        self.stat = stat
        Line = QtGui.QGraphicsLineItem
        whisker1 = Line(-width/16, stat.a_min, width/16, stat.a_min, self)
        whisker2 = Line(-width/16, stat.a_max, width/16, stat.a_max, self)
        vert_line = Line(0, stat.a_min, 0, stat.a_max, self)
        mean_line = Line(-0.7*width, stat.mean, 0.7*width, stat.mean, self)
        for it in (whisker1, whisker2, mean_line):
            it.setPen(self.pen_dark)
        vert_line.setPen(self.pen_dark_dotted)
        dev = math.sqrt(stat.var)
        var_line = Line(0, stat.mean - dev, 0,  stat.mean + dev, self)
        var_line.setPen(self.pen_dark_wide)

        box = QtGui.QGraphicsRectItem(-width/2, stat.q25, width,
                                      stat.q75 - stat.q25, self)
        box.setBrush(QtGui.QBrush(QtGui.QColor(0x33, 0x88, 0xff, 0xc0)))
        box.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        box.setZValue(100)

        median_line = Line(-width/2, stat.median, width/2, stat.median, self)
        median_line.setPen(self.pen_light)
        median_line.setZValue(200)


class OWBoxPlot(widget.OWWidget):
    _name = "Box plot"
    _description = "Shows box plots"
    _long_description = """Shows box plots, either one for or multiple
    box plots for data split by an attribute value."""
    #_icon = "icons/Boxplot.svg"
    _priority = 100
    _author = "Amela Rakanović, Janez Demšar"
    inputs = [("Data", Table, "data")]
    outputs = [("Basic statistic", Table)]

    settingsHandler = DomainContextHandler()
    sorting_select = Setting(0)
    grouping_select = ContextSetting([0])
    attributes_select = ContextSetting([0])
    stattest = Setting(0)
    sig_threshold = Setting(0.05)

    _sorting_criteria_attrs = ["", "label", "median", "mean"]

    tick_pen = QtGui.QPen(QtCore.Qt.white, 5)
    axis_pen = QtGui.QPen(QtCore.Qt.darkGray, 3)
    for pen in (tick_pen, axis_pen):
        pen.setCosmetic(True)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
    axis_pen.setCapStyle(QtCore.Qt.RoundCap)
    tick_pen.setCapStyle(QtCore.Qt.FlatCap)

    def __init__(self, parent=None, signalManager=None, settings=None):
        super().__init__(parent, signalManager, settings)

        self.grouping = []
        self.attributes = []

        self.slider_intervals = 1
        self.ddataset = None

        ## Control Area
        self.attr_list_box = gui.listBox(self.controlArea, self,
            "attributes_select", "attributes", box="Variable",
            selectionMode=QtGui.QListWidget.SingleSelection,
            callback=self.process_change)

        gb = gui.widgetBox(self.controlArea, orientation=1)
        self.attrCombo = gui.listBox(gb, self,
            'grouping_select', "grouping", box="Grouping",
            selectionMode=QtGui.QListWidget.SingleSelection,
            callback=self.process_change)

        self.sorting_combo = gui.comboBox(gb, self,
            'sorting_select', callback=self.sorting_update,
            items=["Show in original order",
                   "Sort by label", "Sort by median", "Sort by mean"])

        gui.rubber(self.controlArea)

        ## Main Area
        result = gui.widgetBox(self.mainArea, addSpace=True)
        self.boxScene = QtGui.QGraphicsScene()
        self.boxView = QtGui.QGraphicsView(self.boxScene)
        self.boxView.setRenderHints(QtGui.QPainter.Antialiasing |
                                    QtGui.QPainter.TextAntialiasing |
                                    QtGui.QPainter.SmoothPixmapTransform)
        self.mainArea.layout().addWidget(self.boxView)
        self.no_values = gui.widgetLabel(self.mainArea,
            "<center><big><b>Too many values.</b></big></center>")
        self.no_values.hide()

        e = gui.widgetBox(self.mainArea, addSpace=False, orientation=0)
        self.infot1 = gui.widgetLabel(e, "<center>No test results.</center>")

        self.warning = gui.widgetBox(self.controlArea, "Warning:")
        self.warning_info = gui.widgetLabel(self.warning, "")
        self.warning.hide()

        #self.controlArea.setFixedWidth(250)
        self.mainArea.setMinimumWidth(650)

    def data(self, dataset):
        if dataset is not None and (
                not len(dataset) or not len(dataset.domain)):
            dataset = None
        self.closeContext()
        self.ddataset = dataset
        self.grouping_select = []
        self.attributes_select = []
        self.attr_list_box.clear()
        self.attrCombo.clear()
        self.openContext(self.ddataset)
        if dataset:
            self.attrCombo.addItem("(none)")
            attributes = []
            grouping = ["None"]
            for attr in self.ddataset.domain:
                if isinstance(attr, DiscreteVariable):
                    grouping.append((attr.name, attr.var_type))
                else:
                    attributes.append((attr.name, attr.var_type))
            self.attributes = attributes
            self.grouping = grouping
            self.grouping_select = [0]
            self.attributes_select = [0]
            self.process_change()
        else:
            self.reset_all_data()

    def reset_all_data(self):
        self.attr_list_box.clear()
        self.attrCombo.clear()
        self.boxScene.clear()
        self.send("Basic statistic", None)
        self.send("Significant data", None)

    def process_change(self):
        self.compute_boxes()
        self.sorting_combo.setDisabled(len(self.stats) < 2)
        self.draw_selected()
        self.show_tests()
#        self.basic_stat = stat_basic_full_tab(self.ddataset)
#        self.send("Basic statistic", self.basic_stat)

    def compute_boxes(self):
        dataset = self.ddataset
        if dataset is None:
            self.stats = []
            return
        attr = self.attributes[self.attributes_select[0]][0]
        attr_ind = dataset.domain.index(attr)
        group_by = self.grouping_select[0]
        if group_by:
            group_attr = self.grouping[group_by][0]
            group_ind = dataset.domain.index(group_attr)
            self.conts = datacaching.getCached(dataset,
                contingency.get_contingency,
                (dataset, attr_ind, group_ind))
            self.stats = [BoxData(cont, value) for cont, value in
                          zip(self.conts, dataset.domain[group_ind].values)]
        else:
            self.dist = datacaching.getCached(dataset,
                distribution.get_distribution, (attr_ind, dataset))
            self.stats = [BoxData(self.dist, attr)]
        self.stats = [stat for stat in self.stats if stat.N > 0]

    def draw_selected(self):
        self.boxScene.clear()
        if not self.stats:
            return

        self.boxes = [BoxItem(stat) for stat in self.stats]
        for box in self.boxes:
            self.boxScene.addItem(box)

        bottom = min(stat.a_min for stat in self.stats)
        top = max(stat.a_max for stat in self.stats)
        first_val, step = owaxis.OWAxis.compute_scale(bottom, top)
        while bottom < first_val:
            first_val -= step
        bottom = first_val
        no_ticks = math.ceil((top - first_val) / step) + 1
        top = max(top, first_val + (no_ticks - 1) * step)

        r = QtCore.QRectF(-80, bottom, len(self.stats) * 60 + 60, top - bottom)
        self.boxScene.setSceneRect(r)

        # This code comes from the implementation of QGraphicsView::fitInView
        # (http://qt.gitorious.org/qt/qt/blobs/4.8/src/gui/graphicsview/
        # qgraphicsview.cpp), but modified to scale only vertically
        d = self.boxView
        d.resetTransform()
        unity = d.matrix().mapRect(QtCore.QRectF(0, 0, 1, 1))
        d.scale(1, 1 / unity.height())
        viewRect = d.viewport().rect().adjusted(15, 15, -15, -30)
        yratio = viewRect.height() / (top - bottom)
        d.scale(1, -yratio)
        d.centerOn((-80 + len(self.stats) * 60 + 60) / 2, (top - bottom) / 2)

        # This comes last because we need to position the text to the
        # appropriate transformed coordinates
        font = QtGui.QFont()
        font.setPixelSize(12)
        val = first_val
        attr = self.attributes[self.attributes_select[0]][0]
        attr_desc = self.ddataset.domain[attr]
        while True:
            l = self.boxScene.addLine(-63, val, -61, val, self.tick_pen)
            l.setZValue(100)
            t = self.boxScene.addSimpleText(attr_desc.repr_val(val), font)
            t.setFlags(t.flags() |
                       QtGui.QGraphicsItem.ItemIgnoresTransformations)
            r = t.boundingRect()
            t.setPos(-70 - r.width(), val + r.height() / 2 / yratio)
            if val >= top:
                break
            val += step
        self.boxScene.addLine(-62, bottom, -62, top, self.axis_pen)

        self.box_labels = []
        for stat in self.stats:
            t = self.boxScene.addSimpleText(stat.label)
            t.setY(stat.a_min - 2 / yratio)
            t.setFlags(t.flags() |
                       QtGui.QGraphicsItem.ItemIgnoresTransformations)
            self.box_labels.append(t)

        self.sorting_update()

    def set_positions(self):
        for pos, box_index in enumerate(self.order):
            self.boxes[box_index].setX(pos * 60)
            t = self.box_labels[box_index]
            t.setX(pos * 60 - t.boundingRect().width() / 2)

    def sorting_update(self):
        self.order = list(range(len(self.stats)))
        if self.sorting_select != 0:
            criterion = self._sorting_criteria_attrs[self.sorting_select]
            self.order.sort(key=lambda i: getattr(self.stats[i], criterion))
        self.set_positions()

    def send_to_graph(self, dataset, attr, box_label, y_label):
        if dataset:
            np_data = data_to_npcol(dataset, dataset.domain.index(attr))
            stat_graph1 = stat_graph(np_data)
            self.graph.append_data(box_label, stat_graph1, y_label)


    def show_tests(self):
        self.warning.hide()
        if len(self.stats) < 2:
            # TODO remove anything that is printed out
            self.infot1.setText("")
        elif any(s.N < 1 for s in self.stats):
            self.infot1.setText("At least one group has just one instance")
        elif len(self.stats) == 2:
            a = self.stat_ttest()
            b = self.stat_wilc()
            self.infot1.setText("<center>Student's t: %.3f (p=%.3f), "
                                "Mann-Whitney's z: %.1f (p=%.3f)</center>" %
                                (a + b))
        else:
            a = self.stat_ANOVA()
            b = self.stat_kruskal()
            self.infot1.setText("<center>ANOVA: %.3f (p=%.3f), "
                                "Kruskal Wallis's U: %.1f (p=%.3f)</center>" %
                                (a + b))

    def stat_ttest(self):
        d1, d2 = self.stats
        pooled_var = d1.var / d1.N + d2.var / d2.N
        df = pooled_var**2 / \
            ((d1.var / d1.N)**2 / (d1.N - 1) + (d2.var / d2.N)**2 / (d2.N - 1))
        t = (d1.mean - d2.mean) / math.sqrt(pooled_var)
        # TODO check!!!
        p = 2 * (1 - scipy.special.stdtr(df, t))
        return t, p

    def stat_wilc(self):
        d1, d2 = self.stats
        ni1, ni2 = d1.shape[1], d2.shape[1]
        i1 = i2 = 0
        R = 0
        rank = 0
        while i1 < ni1 and i2 < ni2:
            if d1[0, i1] < d2[0, i2]:
                R += (rank + d1[1, i1] / 2)
                rank += d1[1, i1]
                i1 += 1
            elif d1[0, i1] == d2[0, i2]:
                br = d1[1, i1] + d2[1, i2]
                R += (rank + br / 2)
                rank += br
                i1 += 1
                i2 += 1
            else:
                rank += d2[1, i2]
                i2 += 1
        if i1 < ni1:
            R = np.sum(d1[1, i1:]) / 2
        U = R - d1.N * (d1.N + 1) / 2
        m = d1.N * (d1.N + d2.N + 1) / 2
        var = d2.N * m / 6
        z = (U - m) / math.sqrt(var)
        # TODO check!!!
        p = 2 * (1 - scipy.special.ndtr(z))
        return z, p

    def stat_ANOVA(self):
        N = sum(stat.N for stat in self.stats)
        grand_avg = sum(stat.N * stat.mean for stat in self.stats) / N
        var_between = sum(stat.N * (stat.mean - grand_avg) ** 2
                          for stat in self.stats)
        df_between = len(self.stats) - 1

        var_within = sum(stat.N * stat.var for stat in self.stats)
        df_within = N - len(self.stats)
        F = (var_between / df_between) / (var_within / df_within)
        p = 1 - scipy.special.fdtr(df_between, df_within, F)
        return F, p

    def stat_kruskal(self):
        return -1, -1
