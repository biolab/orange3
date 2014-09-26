# -*- coding: utf-8 -*-

import math
import itertools
import numpy as np

from PyQt4 import QtCore
from PyQt4 import QtGui
import scipy
import scipy.special

from Orange.data import ContinuousVariable, DiscreteVariable, Table
from Orange.statistics import contingency, distribution, tests

from Orange.widgets import widget, gui
from Orange.widgets.settings import (Setting, DomainContextHandler,
                                     ContextSetting)
from Orange.widgets.utils import datacaching, colorpalette, vartype


def compute_scale(min, max):
    magnitude = int(3 * math.log10(abs(max - min)) + 1)
    if magnitude % 3 == 0:
        first_place = 1
    elif magnitude % 3 == 1:
        first_place = 2
    else:
        first_place = 5
    magnitude = magnitude // 3 - 1
    step = first_place * pow(10, magnitude)
    first_val = math.ceil(min / step) * step
    return first_val, step


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
        thresholds = [N / 4, N / 2, 3 * N / 4]
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


class OWBoxPlot(widget.OWWidget):
    """
    Here's how the widget's functions call each other:

    - `data` is a signal handler fills the list boxes and calls `attr_changed`.

    - `attr_changed` handles changes of attribute or grouping (callbacks for
    list boxes). It recomputes box data by calling `compute_box_data`, shows
    the appropriate display box (discrete/continuous) and then calls
    `layout_changed`

    - `layout_changed` constructs all the elements for the scene (as lists of
    QGraphicsItemGroup) and calls `display_changed`. It is called when the
    attribute or grouping is changed (by attr_changed) and on resize event.

    - `display_changed` puts the elements corresponding to the current display
    settings on the scene. It is called when the elements are reconstructed
    (layout is changed due to selection of attributes or resize event), or
    when the user changes display settings or colors.

    For discrete attributes, the flow is a bit simpler: the elements are not
    constructed in advance (by layout_changed). Instead, layout_changed and
    display_changed call display_changed_disc that draws everything.
    """
    name = "Box plot"
    description = "Shows box plots"
    long_description = """Shows box plots, either one for or multiple
    box plots for data split by an attribute value."""
    icon = "icons/BoxPlot.svg"
    priority = 100
    author = "Amela Rakanović, Janez Demšar"
    inputs = [("Data", Table, "data")]
    outputs = [("Basic statistic", Table)]

    settingsHandler = DomainContextHandler()
    display = Setting(0)
    grouping_select = ContextSetting([0])
    attributes_select = ContextSetting([0])
    stattest = Setting(0)
    sig_threshold = Setting(0.05)
    stretched = Setting(True)
    colorSettings = Setting(None)
    selectedSchemaIndex = Setting(0)

    _sorting_criteria_attrs = ["", "", "median", "mean"]
    _label_positions = ["q25", "median", "mean"]

    _pen_axis_tick = QtGui.QPen(QtCore.Qt.white, 5)
    _pen_axis = QtGui.QPen(QtCore.Qt.darkGray, 3)
    _pen_median = QtGui.QPen(QtGui.QBrush(QtGui.QColor(0xff, 0xff, 0x00)), 2)
    _pen_paramet = QtGui.QPen(QtGui.QBrush(QtGui.QColor(0x33, 0x00, 0xff)), 2)
    _pen_dotted = QtGui.QPen(QtGui.QBrush(QtGui.QColor(0x33, 0x00, 0xff)), 1)
    _pen_dotted.setStyle(QtCore.Qt.DotLine)
    _post_line_pen = QtGui.QPen(QtCore.Qt.lightGray, 2)
    _post_grp_pen = QtGui.QPen(QtCore.Qt.lightGray, 4)
    for pen in (_pen_paramet, _pen_median, _pen_dotted,
                _pen_axis, _pen_axis_tick, _post_line_pen, _post_grp_pen):
        pen.setCosmetic(True)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
    _pen_axis_tick.setCapStyle(QtCore.Qt.FlatCap)

    _box_brush = QtGui.QBrush(QtGui.QColor(0x33, 0x88, 0xff, 0xc0))

    _axis_font = QtGui.QFont()
    _axis_font.setPixelSize(12)
    _label_font = QtGui.QFont()
    _label_font.setPixelSize(11)
    _attr_brush = QtGui.QBrush(QtGui.QColor(0x33, 0x00, 0xff))

    def __init__(self):
        super().__init__()
        self.grouping = []
        self.attributes = []
        self.stats = []
        self.ddataset = None

        self.attr_list_box = gui.listBox(
            self.controlArea, self, "attributes_select", "attributes",
            box="Variable", callback=self.attr_changed)
        self.attrCombo = gui.listBox(
            self.controlArea, self, 'grouping_select', "grouping",
            box="Grouping", callback=self.attr_changed)
        self.sorting_combo = gui.radioButtonsInBox(
            self.controlArea, self, 'display', box='Display',
            callback=self.display_changed,
            btnLabels=["Box plots", "Annotated boxes",
                       "Compare medians", "Compare means"])
        self.stretching_box = gui.checkBox(
            self.controlArea, self, 'stretched', "Stretch bars", box='Display',
            callback=self.display_changed).box
        gui.rubber(self.controlArea)

        gui.widgetBox(self.mainArea, addSpace=True)
        self.boxScene = QtGui.QGraphicsScene()
        self.boxView = QtGui.QGraphicsView(self.boxScene)
        self.boxView.setRenderHints(QtGui.QPainter.Antialiasing |
                                    QtGui.QPainter.TextAntialiasing |
                                    QtGui.QPainter.SmoothPixmapTransform)
        self.mainArea.layout().addWidget(self.boxView)
        self.posthoc_lines = []
        e = gui.widgetBox(self.mainArea, addSpace=False, orientation=0)
        self.infot1 = gui.widgetLabel(e, "<center>No test results.</center>")
        self.mainArea.setMinimumWidth(650)

        self.warning = gui.widgetBox(self.controlArea, "Warning:")
        self.warning_info = gui.widgetLabel(self.warning, "")
        self.warning.hide()

        self.stats = self.dist = self.conts = []
        self.is_continuous = False
        self.set_display_box()

        dlg = self.createColorDialog()
        self.discPalette = dlg.getDiscretePalette("discPalette")

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.layout_changed()

    def setColors(self):
        dlg = self.createColorDialog()
        if dlg.exec_():
            self.colorSettings = dlg.getColorSchemas()
            self.selectedSchemaIndex = dlg.selectedSchemaIndex
            self.discPalette = dlg.getDiscretePalette("discPalette")
            self.display_changed()

    def createColorDialog(self):
        c = colorpalette.ColorPaletteDlg(self, "Color Palette")
        c.createDiscretePalette("discPalette", "Discrete Palette")
        c.setColorSchemas(self.colorSettings, self.selectedSchemaIndex)
        return c

    # noinspection PyTypeChecker
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
        if dataset:
            self.openContext(self.ddataset)
            self.attributes = [(a.name, vartype(a)) for a in dataset.domain]
            self.grouping = ["None"] + [(a.name, vartype(a))
                                        for a in dataset.domain
                                        if isinstance(a, DiscreteVariable)]
            self.grouping_select = [0]
            self.attributes_select = [0]
            self.attr_changed()
        else:
            self.reset_all_data()

    def reset_all_data(self):
        self.attr_list_box.clear()
        self.attrCombo.clear()
        self.boxScene.clear()
        self.send("Basic statistic", None)
        self.send("Significant data", None)

    def attr_changed(self):
        self.compute_box_data()
        self.set_display_box()
        self.layout_changed()

    def compute_box_data(self):
        dataset = self.ddataset
        if dataset is None:
            self.stats = self.dist = self.conts = []
            return
        attr_ind = self.attributes_select[0]
        attr = dataset.domain[attr_ind]
        self.is_continuous = isinstance(attr, ContinuousVariable)
        group_by = self.grouping_select[0]
        if group_by:
            group_attr = self.grouping[group_by][0]
            group_ind = dataset.domain.index(group_attr)
            self.dist = []
            self.conts = datacaching.getCached(
                dataset, contingency.get_contingency,
                (dataset, attr_ind, group_ind))
            if self.is_continuous:
                self.stats = [BoxData(cont) for cont in self.conts]
            self.label_txts = dataset.domain[group_ind].values
        else:
            self.dist = datacaching.getCached(
                dataset, distribution.get_distribution, (dataset, attr_ind))
            self.conts = []
            if self.is_continuous:
                self.stats = [BoxData(self.dist)]
            self.label_txts = [""]
        self.stats = [stat for stat in self.stats if stat.N > 0]

    def set_display_box(self):
        if self.is_continuous:
            self.stretching_box.hide()
            self.sorting_combo.show()
            self.sorting_combo.setDisabled(len(self.stats) < 2)
        else:
            self.stretching_box.show()
            self.sorting_combo.hide()

    def clear_scene(self):
        self.boxScene.clear()
        self.posthoc_lines = []

    def layout_changed(self):
        self.clear_scene()
        if len(self.conts) == len(self.dist) == 0:
            return
        if not self.is_continuous:
            return self.display_changed_disc()

        attr = self.attributes[self.attributes_select[0]][0]
        attr = self.ddataset.domain[attr]

        self.mean_labels = [self.mean_label(stat, attr, lab)
                            for stat, lab in zip(self.stats, self.label_txts)]
        self.draw_axis()
        self.boxes = [self.box_group(stat) for stat in self.stats]
        self.labels = [self.label_group(stat, attr, mean_lab)
                       for stat, mean_lab in zip(self.stats, self.mean_labels)]
        self.attr_labels = [self.attr_label(lab) for lab in self.label_txts]
        for it in itertools.chain(self.labels, self.boxes, self.attr_labels):
            self.boxScene.addItem(it)
        self.display_changed()

    def display_changed(self):
        if not self.is_continuous:
            return self.display_changed_disc()

        self.order = list(range(len(self.stats)))
        criterion = self._sorting_criteria_attrs[self.display]
        if criterion:
            self.order.sort(key=lambda i: getattr(self.stats[i], criterion))
        heights = 90 if self.display == 1 else 60

        for row, box_index in enumerate(self.order):
            y = (-len(self.stats) + row) * heights + 10
            self.boxes[box_index].setY(y)
            labels = self.labels[box_index]
            if self.display == 1:
                labels.show()
                labels.setY(y)
            else:
                labels.hide()
            label = self.attr_labels[box_index]
            label.setY(y - 15 - label.boundingRect().height())
            if self.display == 1:
                label.hide()
            else:
                stat = self.stats[box_index]
                poss = (stat.q25, -1, stat.median + 5 / self.scale_x,
                        stat.mean + 5 / self.scale_x)
                label.show()
                label.setX(poss[self.display] * self.scale_x)

        r = QtCore.QRectF(self.scene_min_x, -30 - len(self.stats) * heights,
                          self.scene_width, len(self.stats) * heights + 90)
        self.boxScene.setSceneRect(r)
        self.boxView.centerOn(self.scene_min_x + self.scene_width / 2,
                              -30 - len(self.stats) * heights / 2 + 45)

        self.compute_tests()
        self.show_posthoc()

    def display_changed_disc(self):
        self.clear_scene()
        self.attr_labels = [self.attr_label(lab) for lab in self.label_txts]
        self.draw_axis_disc()
        if self.grouping_select[0]:
            self.discPalette.set_number_of_colors(len(self.conts[0]))
            self.boxes = [self.strudel(cont) for cont in self.conts]
        else:
            self.discPalette.set_number_of_colors(len(self.dist))
            self.boxes = [self.strudel(self.dist)]

        for row, box in enumerate(self.boxes):
            y = (-len(self.boxes) + row) * 40 + 10
            self.boxScene.addItem(box)
            box.setPos(0, y)
            label = self.attr_labels[row]
            b = label.boundingRect()
            label.setPos(-b.width() - 10, y - b.height() / 2)
            self.boxScene.addItem(label)
        self.boxScene.setSceneRect(-self.label_width - 5,
                                   -30 - len(self.boxes) * 40,
                                   self.scene_width, len(self.boxes * 40) + 90)
        self.boxView.centerOn(self.scene_width / 2,
                              -30 - len(self.boxes) * 40 / 2 + 45)

    def compute_tests(self):
        # The t-test and ANOVA are implemented here since they efficiently use
        # the widget-specific data in self.stats.
        # The non-parametric tests can't do this, so we use statistics.tests
        def stat_ttest():
            d1, d2 = self.stats
            pooled_var = d1.var / d1.N + d2.var / d2.N
            df = pooled_var ** 2 / \
                ((d1.var / d1.N) ** 2 / (d1.N - 1) +
                 (d2.var / d2.N) ** 2 / (d2.N - 1))
            t = abs(d1.mean - d2.mean) / math.sqrt(pooled_var)
            p = 2 * (1 - scipy.special.stdtr(df, t))
            return t, p

        # TODO: Check this function
        def stat_ANOVA():
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

        self.warning.hide()
        if self.display < 2 or len(self.stats) < 2:
            t = ""
        elif any(s.N <= 1 for s in self.stats):
            t = "At least one group has just one instance, " \
                "cannot compute significance"
        elif len(self.stats) == 2:
            if self.display == 2:
                z, self.p = tests.wilcoxon_rank_sum(
                    self.stats[0].dist, self.stats[1].dist)
                t = "Mann-Whitney's z: %.1f (p=%.3f)" % (z, self.p)
            else:
                t, self.p = stat_ttest()
                t = "Student's t: %.3f (p=%.3f)" % (t, self.p)
        else:
            if self.display == 2:
                U, self.p = -1, -1
                t = "Kruskal Wallis's U: %.1f (p=%.3f)" % (U, self.p)
            else:
                F, self.p = stat_ANOVA()
                t = "ANOVA: %.3f (p=%.3f)" % (F, self.p)
        self.infot1.setText("<center>%s</center>" % t)

    def attr_label(self, text):
        return QtGui.QGraphicsSimpleTextItem(text)

    def mean_label(self, stat, attr, val_name):
        label = QtGui.QGraphicsItemGroup()
        t = QtGui.QGraphicsSimpleTextItem(
            "%.*f" % (attr.number_of_decimals + 1, stat.mean), label)
        t.setFont(self._label_font)
        bbox = t.boundingRect()
        w2, h = bbox.width() / 2, bbox.height()
        t.setPos(-w2, -h)
        tpm = QtGui.QGraphicsSimpleTextItem(
            " \u00b1 " + "%.*f" % (attr.number_of_decimals + 1, stat.dev),
            label)
        tpm.setFont(self._label_font)
        tpm.setPos(w2, -h)
        if val_name:
            vnm = QtGui.QGraphicsSimpleTextItem(val_name + ": ", label)
            vnm.setFont(self._label_font)
            vnm.setBrush(self._attr_brush)
            vb = vnm.boundingRect()
            label.min_x = -w2 - vb.width()
            vnm.setPos(label.min_x, -h)
        else:
            label.min_x = -w2
        return label

    def draw_axis(self):
        """Draw the horizontal axis and sets self.scale_x"""
        bottom = min(stat.a_min for stat in self.stats)
        top = max(stat.a_max for stat in self.stats)

        first_val, step = compute_scale(bottom, top)
        while bottom < first_val:
            first_val -= step
        bottom = first_val
        no_ticks = math.ceil((top - first_val) / step) + 1
        top = max(top, first_val + (no_ticks - 1) * step)

        gbottom = min(bottom, min(stat.mean - stat.dev for stat in self.stats))
        gtop = max(top, max(stat.mean + stat.dev for stat in self.stats))

        bv = self.boxView
        viewRect = bv.viewport().rect().adjusted(15, 15, -15, -30)
        self.scale_x = scale_x = viewRect.width() / (gtop - gbottom)

        # In principle we should repeat this until convergence since the new
        # scaling is too conservative. (No chance am I doing this.)
        mlb = min(stat.mean + mean_lab.min_x / scale_x
                  for stat, mean_lab in zip(self.stats, self.mean_labels))
        if mlb < gbottom:
            gbottom = mlb
            self.scale_x = scale_x = viewRect.width() / (gtop - gbottom)

        self.scene_min_x = gbottom * scale_x
        self.scene_width = (gtop - gbottom) * scale_x

        val = first_val
        attr = self.attributes[self.attributes_select[0]][0]
        attr_desc = self.ddataset.domain[attr]
        while True:
            l = self.boxScene.addLine(val * scale_x, -1, val * scale_x, 1,
                                      self._pen_axis_tick)
            l.setZValue(100)
            t = self.boxScene.addSimpleText(
                attr_desc.repr_val(val), self._axis_font)
            t.setFlags(t.flags() |
                       QtGui.QGraphicsItem.ItemIgnoresTransformations)
            r = t.boundingRect()
            t.setPos(val * scale_x - r.width() / 2, 8)
            if val >= top:
                break
            val += step
        self.boxScene.addLine(bottom * scale_x - 4, 0,
                              top * scale_x + 4, 0, self._pen_axis)

    def draw_axis_disc(self):
        """
        Draw the horizontal axis and sets self.scale_x for discrete attributes
        """
        if self.stretched:
            step = steps = 10
        else:
            if self.grouping_select[0]:
                max_box = max(float(np.sum(dist)) for dist in self.conts)
            else:
                max_box = float(np.sum(self.dist))
            if max_box == 0:
                self.scale_x = 1
                return
            _, step = compute_scale(0, max_box)
            step = int(step)
            steps = int(math.ceil(max_box / step))
        max_box = step * steps

        bv = self.boxView
        viewRect = bv.viewport().rect().adjusted(15, 15, -15, -30)
        self.scene_width = viewRect.width()

        lab_width = max(lab.boundingRect().width() for lab in self.attr_labels)
        lab_width = max(lab_width, 40)
        lab_width = min(lab_width, self.scene_width / 3)
        self.label_width = lab_width
        self.scale_x = scale_x = (self.scene_width - lab_width - 10) / max_box

        self.boxScene.addLine(0, 0, max_box * scale_x, 0, self._pen_axis)
        for val in range(0, step * steps + 1, step):
            l = self.boxScene.addLine(val * scale_x, -1, val * scale_x, 1,
                                      self._pen_axis_tick)
            l.setZValue(100)
            t = self.boxScene.addSimpleText(str(val), self._axis_font)
            t.setPos(val * scale_x - t.boundingRect().width() / 2, 8)
        if self.stretched:
            self.scale_x *= 100

    def label_group(self, stat, attr, mean_lab):
        def centered_text(val, pos):
            t = QtGui.QGraphicsSimpleTextItem(
                "%.*f" % (attr.number_of_decimals + 1, val), labels)
            t.setFont(self._label_font)
            bbox = t.boundingRect()
            t.setPos(pos - bbox.width() / 2, 22)
            return t

        def line(x, down=1):
            QtGui.QGraphicsLineItem(x, 12 * down, x, 20 * down, labels)

        def move_label(label, frm, to):
            label.setX(to)
            to += t_box.width() / 2
            path = QtGui.QPainterPath()
            path.lineTo(0, 4)
            path.lineTo(to - frm, 4)
            path.lineTo(to - frm, 8)
            p = QtGui.QGraphicsPathItem(path)
            p.setPos(frm, 12)
            labels.addToGroup(p)

        labels = QtGui.QGraphicsItemGroup()

        labels.addToGroup(mean_lab)
        m = stat.mean * self.scale_x
        mean_lab.setPos(m, -22)
        line(m, -1)

        msc = stat.median * self.scale_x
        med_t = centered_text(stat.median, msc)
        med_box_width2 = med_t.boundingRect().width()
        line(msc)

        x = stat.q25 * self.scale_x
        t = centered_text(stat.q25, x)
        t_box = t.boundingRect()
        med_left = msc - med_box_width2
        if x + t_box.width() / 2 >= med_left - 5:
            move_label(t, x, med_left - t_box.width() - 5)
        else:
            line(x)

        x = stat.q75 * self.scale_x
        t = centered_text(stat.q75, x)
        t_box = t.boundingRect()
        med_right = msc + med_box_width2
        if x - t_box.width() / 2 <= med_right + 5:
            move_label(t, x, med_right + 5)
        else:
            line(x)

        return labels

    def box_group(self, stat, height=20):
        def line(x0, y0, x1, y1, *args, **kwargs):
            return QtGui.QGraphicsLineItem(x0 * scale_x, y0, x1 * scale_x, y1,
                                           *args, **kwargs)

        scale_x = self.scale_x
        box = QtGui.QGraphicsItemGroup()
        whisker1 = line(stat.a_min, -1.5, stat.a_min, 1.5, box)
        whisker2 = line(stat.a_max, -1.5, stat.a_max, 1.5, box)
        vert_line = line(stat.a_min, 0, stat.a_max, 0, box)
        mean_line = line(stat.mean, -height / 3, stat.mean, height / 3, box)
        for it in (whisker1, whisker2, mean_line):
            it.setPen(self._pen_paramet)
        vert_line.setPen(self._pen_dotted)
        var_line = line(stat.mean - stat.dev, 0, stat.mean + stat.dev, 0, box)
        var_line.setPen(self._pen_paramet)

        mbox = QtGui.QGraphicsRectItem(stat.q25 * scale_x, -height / 2,
                                       (stat.q75 - stat.q25) * scale_x, height,
                                       box)
        mbox.setBrush(self._box_brush)
        mbox.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        mbox.setZValue(-200)

        median_line = line(stat.median, -height / 2,
                           stat.median, height / 2, box)
        median_line.setPen(self._pen_median)
        median_line.setZValue(-150)

        return box

    def strudel(self, dist):
        ss = np.sum(dist)
        box = QtGui.QGraphicsItemGroup()
        if ss < 1e-6:
            QtGui.QGraphicsRectItem(0, -10, 1, 10, box)
        cum = 0
        get_color = self.discPalette.getRGB
        for i, v in enumerate(dist):
            if v < 1e-6:
                continue
            if self.stretched:
                v /= ss
            v *= self.scale_x
            rect = QtGui.QGraphicsRectItem(cum + 1, -6, v - 2, 12, box)
            rect.setBrush(QtGui.QBrush(QtGui.QColor(*get_color(i))))
            rect.setPen(QtGui.QPen(QtCore.Qt.NoPen))
            cum += v
        return box

    def show_posthoc(self):
        def line(y0, y1):
            it = self.boxScene.addLine(x, y0, x, y1, self._post_line_pen)
            it.setZValue(-100)
            self.posthoc_lines.append(it)

        while self.posthoc_lines:
            self.boxScene.removeItem(self.posthoc_lines.pop())
        if self.display < 2 or len(self.stats) < 2:
            return
        crit_line = self._sorting_criteria_attrs[self.display]
        xs = []
        y_up = -len(self.stats) * 60 + 10
        for pos, box_index in enumerate(self.order):
            stat = self.stats[box_index]
            x = getattr(stat, crit_line) * self.scale_x
            xs.append(x)
            by = y_up + pos * 60
            line(by + 12, 3)
            line(by - 12, by - 25)

        used_to = []
        last_to = 0
        for frm, frm_x in enumerate(xs[:-1]):
            for to in range(frm + 1, len(xs)):
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
            y = - 6 - rowi * 6
            it = self.boxScene.addLine(frm_x - 2, y, xs[to] + 2, y,
                                       self._post_grp_pen)
            self.posthoc_lines.append(it)
            last_to = to
