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
    def __init__(self, dist):
        self.dist = dist
        self.N = N = np.sum(dist[1])
        if N == 0:
            return
        self.a_min = float(dist[0, 0])
        self.a_max = float(dist[0, -1])
        self.mean = float(np.sum(dist[0] * dist[1]) / N)
        self.var = float(np.sum(dist[1] * (dist[0] - self.mean) ** 2) / N)
        self.dev = math.sqrt(self.var)
        s = 0
        thresholds = [N/4, N/2, 3*N/4]
        thresh_i = 0
        q = []
        for i, e in enumerate(dist[1]):
            s += e
            if s >= thresholds[thresh_i]:
                if s == thresholds[thresh_i] and i + 1 < dist.shape[1]:
                    q.append(float((dist[0, i] + dist[0, i + 1]) / 2))
                else:
                    q.append(float(dist[0, i]))
                thresh_i += 1
                if thresh_i == 3:
                    break
        while len(q) < 3:
            q.append(q[-1])
        self.q25, self.median, self.q75 = q


class BoxItem(QtGui.QGraphicsItemGroup):
    pen_med = QtGui.QPen(QtGui.QBrush(QtGui.QColor(0xff, 0xff, 0x00)), 2)
    pen_dark = QtGui.QPen(QtGui.QBrush(QtGui.QColor(0x33, 0x00, 0xff)), 2)
    pen_dark_dotted = QtGui.QPen(QtGui.QBrush(QtGui.QColor(0x33, 0x00, 0xff)), 1)
    pen_dark_wide = QtGui.QPen(QtGui.QBrush(QtGui.QColor(0x33, 0x00, 0xff)), 2)
    for pen in (pen_dark, pen_med, pen_dark_wide, pen_dark_dotted):
        pen.setCosmetic(True)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
    pen_dark_dotted.setStyle(QtCore.Qt.DotLine)

    def __init__(self, stat, height=20):
        super().__init__()
        self.stat = stat
        Line = QtGui.QGraphicsLineItem
        whisker1 = Line(stat.a_min, -height/16, stat.a_min, height/16, self)
        whisker2 = Line(stat.a_max, -height/16, stat.a_max, height/16, self)
        vert_line = Line(stat.a_min, 0, stat.a_max, 0, self)
        mean_line = Line(stat.mean, -0.35*height, stat.mean, 0.35*height, self)
        for it in (whisker1, whisker2, mean_line):
            it.setPen(self.pen_dark)
        vert_line.setPen(self.pen_dark_dotted)
        var_line = Line(stat.mean - stat.dev, 0, stat.mean + stat.dev, 0, self)
        var_line.setPen(self.pen_dark_wide)

        box = QtGui.QGraphicsRectItem(stat.q25, -height/2,
                                      stat.q75 - stat.q25, height, self)
        box.setBrush(QtGui.QBrush(QtGui.QColor(0x33, 0x88, 0xff, 0xc0)))
        box.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        box.setZValue(-200)

        median_line = Line(stat.median, -height/2, stat.median, height/2, self)
        median_line.setPen(self.pen_med)
        median_line.setZValue(-150)


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
    compare_by = Setting(0)
    grouping_select = ContextSetting([0])
    attributes_select = ContextSetting([0])
    stattest = Setting(0)
    sig_threshold = Setting(0.05)

    _sorting_criteria_attrs = ["", "median", "mean"]
    _label_positions = ["q25", "median", "mean"]

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

        self.sorting_combo = gui.radioButtonsInBox(gb, self,
            'compare_by', box='Statistics', callback=self.sorting_update,
            btnLabels=["No comparisons", "Compare medians", "Compare means"])

        gui.rubber(self.controlArea)

        ## Main Area
        result = gui.widgetBox(self.mainArea, addSpace=True)
        self.boxScene = QtGui.QGraphicsScene()
        self.boxView = QtGui.QGraphicsView(self.boxScene)
        self.boxView.setRenderHints(QtGui.QPainter.Antialiasing |
                                    QtGui.QPainter.TextAntialiasing |
                                    QtGui.QPainter.SmoothPixmapTransform)
        self.mainArea.layout().addWidget(self.boxView)
        self.posthoc_lines = []

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
        self.sorting_update()
        self.show_data()
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
            self.stats = [BoxData(cont) for cont in self.conts]
            self.label_txts = dataset.domain[group_ind].values
        else:
            self.dist = datacaching.getCached(dataset,
                distribution.get_distribution, (attr_ind, dataset))
            self.stats = [BoxData(self.dist)]
            self.label_txts = [""]
        self.stats = [stat for stat in self.stats if stat.N > 0]

    def addText(self, text, x=None, y=None):
        t = self.boxScene.addSimpleText(text)
        t.setFlags(t.flags() | QtGui.QGraphicsItem.ItemIgnoresTransformations)
        if x is not None:
            t.setX(x)
        if y is not None:
            t.setY(y)
        return t

    def draw_selected(self):
        self.boxScene.clear()
        self.posthoc_lines = []
        if not self.stats:
            return

        self.boxes = [BoxItem(stat) for stat in self.stats]
        for box in self.boxes:
            self.boxScene.addItem(box)
        self.box_labels = [self.addText(label) for label in self.label_txts]

        bottom = min(stat.a_min for stat in self.stats)
        top = max(stat.a_max for stat in self.stats)
        first_val, step = owaxis.OWAxis.compute_scale(bottom, top)
        while bottom < first_val:
            first_val -= step
        bottom = first_val
        no_ticks = math.ceil((top - first_val) / step) + 1
        top = max(top, first_val + (no_ticks - 1) * step)

        gbottom = min(bottom, min(stat.mean - stat.dev for stat in self.stats))
        gtop = max(top, max(stat.mean + stat.dev for stat in self.stats))
        r = QtCore.QRectF(gbottom, -30,
                          gtop - gbottom, len(self.stats) * 60 + 60)
        self.boxScene.setSceneRect(r)

        # This code comes from the implementation of QGraphicsView::fitInView
        # (http://qt.gitorious.org/qt/qt/blobs/4.8/src/gui/graphicsview/
        # qgraphicsview.cpp), but modified to scale only vertically
        d = self.boxView
        d.resetTransform()
        unity = d.matrix().mapRect(QtCore.QRectF(0, 0, 1, 1))
        d.scale(1 / unity.width(), 1)
        viewRect = d.viewport().rect().adjusted(15, 15, -15, -30)
        xratio = self.xratio = viewRect.width() / (gtop - gbottom)
        d.scale(xratio, 1)
        d.centerOn((gtop - gbottom) / 2, (-80 + len(self.stats) * 60 + 60) / 2)

        # This comes last because we need to position the text to the
        # appropriate transformed coordinates
        font = QtGui.QFont()
        font.setPixelSize(12)
        val = first_val
        attr = self.attributes[self.attributes_select[0]][0]
        attr_desc = self.ddataset.domain[attr]
        axis_y = len(self.stats) * 60
        while True:
            l = self.boxScene.addLine(val, axis_y - 1, val, axis_y + 1, self.tick_pen)
            l.setZValue(100)
            t = self.boxScene.addSimpleText(attr_desc.repr_val(val), font)
            t.setFlags(t.flags() |
                       QtGui.QGraphicsItem.ItemIgnoresTransformations)
            r = t.boundingRect()
            t.setPos(val - r.width() / 2 / xratio, axis_y + 8)
            if val >= top:
                break
            val += step
        self.boxScene.addLine(bottom - 4 / xratio, axis_y, top + 4 / xratio, axis_y,
                              self.axis_pen)


    def show_data(self):
        def repr_val(val):
            return "%.*f" % (attr.number_of_decimals + 1, val)

        font = QtGui.QFont()
        font.setPixelSize(11)

        def centered_text(x, d, text=None):
            l = self.boxScene.addLine(x, y + 12 * d, x, y + 20 * d)
            t = self.addText(repr_val(x) if text is None else text)
            t.setFont(font)
            bbox = t.boundingRect()
            t.setPos(x - bbox.width() / 2 / self.xratio,
                     y + 22 * d - (bbox.height() if d == -1 else 0))
            return t, l

        attr = self.attributes[self.attributes_select[0]][0]
        attr = self.ddataset.domain[attr]

        for rowi, box_index in enumerate(self.order):
            stat = self.stats[box_index]
            y = rowi * 60

            t, _ = centered_text(stat.mean, -1)
            tpm = self.addText(" \u00b1 " + repr_val(math.sqrt(stat.var)))
            tpm.setFont(font)
            tpm.setPos(stat.mean + t.boundingRect().width() / 2 / self.xratio,
                       y - 22 - tpm.boundingRect().height())

            med_t, _ = centered_text(stat.median, 1)
            med_box = med_t.boundingRect()

            t, lne = centered_text(stat.q25, 1)
            t_box = t.boundingRect()
            med_left = stat.median - (med_box.width() / 2) / self.xratio
            if stat.q25 + (t_box.width() / 2 + 5) / self.xratio >= med_left:
                if stat.q25 + 2 / self.xratio < med_left:
                    t.setX(stat.q25 - (t_box.width() + 2) / self.xratio)
                else:
                    x = med_left - (t_box.width() + 5) / self.xratio
                    t.setX(x)
                    self.boxScene.removeItem(lne)
                    path = QtGui.QPainterPath()
                    path.lineTo(0, -4)
                    x += t_box.width() / 2 / self.xratio
                    path.lineTo(stat.q25 - x - 2 / self.xratio, -4)
                    path.lineTo(stat.q25 - x - 2 / self.xratio, -8)
                    p = self.boxScene.addPath(path)
                    p.setPos(x, y + 20)

            t, lne = centered_text(stat.q75, 1)
            t_box = t.boundingRect()
            med_right = stat.median + (med_box.width() / 2) / self.xratio
            if stat.q75 - (t_box.width() / 2 + 5) / self.xratio <= med_right:
                if stat.q75 - 5 / self.xratio > med_right:
                    t.setX(stat.q75 + (t_box.width() + 5) / self.xratio)
                else:
                    x = med_right + (t_box.width() + 5) / self.xratio
                    t.setX(x)
                    self.boxScene.removeItem(lne)
                    path = QtGui.QPainterPath()
                    path.lineTo(0, -4)
                    x -= t_box.width() / 2 / self.xratio
                    path.lineTo(- (x - stat.q75 - 2 / self.xratio), -4)
                    path.lineTo(- (x - stat.q75 - 2 / self.xratio), -8)
                    p = self.boxScene.addPath(path)
                    p.setPos(x, y + 20)



    def set_positions(self):
        for pos, box_index in enumerate(self.order):
            self.boxes[box_index].setY(pos * 60)
            t = self.box_labels[box_index]
            t.setY(pos * 60 - 12 - t.boundingRect().height())

    def sorting_update(self):
        self.order = list(range(len(self.stats)))
        if self.compare_by:
            criterion = self._sorting_criteria_attrs[self.compare_by]
            self.order.sort(key=lambda i: getattr(self.stats[i], criterion))
        self.set_positions()
        self.compute_tests()
        self.show_posthoc()

    def show_posthoc(self):
        while self.posthoc_lines:
            self.boxScene.removeItem(self.posthoc_lines.pop())
        axis_y = len(self.stats) * 60 - 3
        if self.compare_by and len(self.stats) >= 2:
            crit_line = self._sorting_criteria_attrs[self.compare_by]
        else:
            crit_line = None
        post_pen = QtGui.QPen(QtCore.Qt.lightGray, 2)
        post_pen.setCosmetic(True)
        xs = []
        for pos, box_index in enumerate(self.order):
            stat = self.stats[box_index]
            t = self.box_labels[box_index]
            if crit_line:
                x = getattr(stat, crit_line)
                xs.append(x)
                it = self.boxScene.addLine(x, pos * 60 + 12, x, axis_y, post_pen)
                it.setZValue(-100)
                self.posthoc_lines.append(it)
                it = self.boxScene.addLine(x, pos * 60 - 12, x, pos * 60 - 25, post_pen)
                it.setZValue(-100)
                self.posthoc_lines.append(it)
                t.setX(x + 5 / self.xratio)
            else:
                x = (stat.q75 + stat.q25) / 2
                t.setX(x - t.boundingRect().width() / 2 / self.xratio)

        if crit_line and len(self.stats) > 2:
            grp_pen = QtGui.QPen(QtCore.Qt.lightGray, 4)
            grp_pen.setCosmetic(True)
            used_to = []
            last_to = 0
            for frm, frm_x in enumerate(xs[:-1]):
                for to in range(frm+1, len(xs)):
                    if xs[to] - frm_x > 1.5:
                        to -= 1
                        break
                if last_to == to or frm == to:
                    continue
                for rowi, used in enumerate(used_to):
                    if used < frm:
                        used_to[rowi] = to
                        break
                else:
                    rowi = len(used_to)
                    used_to.append(to)
                y = axis_y - 6 - rowi * 6
                it = self.boxScene.addLine(frm_x - 2 / self.xratio, y,
                                           xs[to] + 2 / self.xratio, y, grp_pen)
                self.posthoc_lines.append(it)
                last_to = to

    def compute_tests(self):
        self.warning.hide()
        if not self.compare_by or len(self.stats) < 2:
            t = ""
        elif any(s.N <= 1 for s in self.stats):
            t = "At least one group has just one instance, " \
                "cannot compute significance"
        elif len(self.stats) == 2:
            if self.compare_by == 1:
                z, self.p = self.stat_wilc()
                t = "Mann-Whitney's z: %.1f (p=%.3f)" % (z, self.p)
            else:
                t, self.p = self.stat_ttest()
                t = "Student's t: %.3f (p=%.3f)" % (t, self.p)
        else:
            if self.compare_by == 1:
                U, self.p = self.stat_kruskal()
                t = "Kruskal Wallis's U: %.1f (p=%.3f)" % (U, self.p)
            else:
                F, self.p = self.stat_ANOVA()
                t = "ANOVA: %.3f (p=%.3f)" % (F, self.p)
        self.infot1.setText("<center>%s</center>" % t)

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
