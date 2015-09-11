# -*- coding: utf-8 -*-
import sys
import math
import itertools
import numpy as np

from PyQt4 import QtCore
from PyQt4 import QtGui
from PyQt4.QtGui import QSizePolicy

import scipy.special

import Orange.data
from Orange.statistics import contingency, distribution

from Orange.widgets import widget, gui
from Orange.widgets.settings import (Setting, DomainContextHandler,
                                     ContextSetting)
from Orange.widgets.utils import datacaching, colorpalette, vartype
from Orange.widgets.io import FileFormats


def compute_scale(min_, max_):
    magnitude = int(3 * math.log10(abs(max_ - min_)) + 1)
    if magnitude % 3 == 0:
        first_place = 1
    elif magnitude % 3 == 1:
        first_place = 2
    else:
        first_place = 5
    magnitude = magnitude // 3 - 1
    step = first_place * pow(10, magnitude)
    first_val = math.ceil(min_ / step) * step
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
        thresholds = [N / 4, N / 2, N / 4 * 3]
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


# noinspection PyUnresolvedReferences
class OWBoxPlot(widget.OWWidget):
    """
    Here's how the widget's functions call each other:

    - `set_data` is a signal handler fills the list boxes and calls `attr_changed`.

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
    name = "Box Plot"
    description = "Visualize distribution of feature values in a box plots."
    icon = "icons/BoxPlot.svg"
    priority = 100
    author = "Amela Rakanović, Janez Demšar"
    inputs = [("Data", Orange.data.Table, "set_data")]

    #: Comparison types for continuous variables
    CompareNone, CompareMedians, CompareMeans = 0, 1, 2

    settingsHandler = DomainContextHandler()

    attributes_select = ContextSetting([0])
    grouping_select = ContextSetting([0])
    show_annotations = Setting(True)
    compare = Setting(CompareMedians)
    stattest = Setting(0)
    sig_threshold = Setting(0.05)
    stretched = Setting(True)

    _sorting_criteria_attrs = {
        CompareNone: "", CompareMedians: "median", CompareMeans: "mean"
    }

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

    want_graph = True

    def __init__(self):
        super().__init__()
        self.grouping = []
        self.attributes = []
        self.stats = []
        self.dataset = None
        self.posthoc_lines = []

        self.label_txts = self.mean_labels = self.boxes = self.labels = \
            self.label_txts_all = self.attr_labels = self.order = []
        self.p = -1.0
        self.scale_x = self.scene_min_x = self.scene_width = 0
        self.label_width = 0

        self.attr_list_box = gui.listBox(
            self.controlArea, self, "attributes_select", "attributes",
            box="Variable", callback=self.attr_changed,
            sizeHint=QtCore.QSize(200, 150))
        self.attr_list_box.setSizePolicy(QSizePolicy.Fixed,
                                         QSizePolicy.MinimumExpanding)

        box = gui.widgetBox(self.controlArea, "Grouping")
        self.group_list_box = gui.listBox(
            box, self, 'grouping_select', "grouping",
            callback=self.attr_changed,
            sizeHint=QtCore.QSize(200, 100))
        self.group_list_box.setSizePolicy(QSizePolicy.Fixed,
                                          QSizePolicy.MinimumExpanding)

        # TODO: move Compare median/mean to grouping box
        self.display_box = gui.widgetBox(self.controlArea, "Display")

        gui.checkBox(self.display_box, self, "show_annotations", "Annotate",
                     callback=self.display_changed)
        self.compare_rb = gui.radioButtonsInBox(
            self.display_box, self, 'compare',
            btnLabels=["No comparison", "Compare medians", "Compare means"],
            callback=self.display_changed)

        self.stretching_box = gui.checkBox(
            self.controlArea, self, 'stretched', "Stretch bars", box='Display',
            callback=self.display_changed).box

        gui.widgetBox(self.mainArea, addSpace=True)
        self.box_scene = QtGui.QGraphicsScene()
        self.box_view = QtGui.QGraphicsView(self.box_scene)
        self.box_view.setRenderHints(QtGui.QPainter.Antialiasing |
                                    QtGui.QPainter.TextAntialiasing |
                                    QtGui.QPainter.SmoothPixmapTransform)
        self.box_view.viewport().installEventFilter(self)

        self.mainArea.layout().addWidget(self.box_view)

        e = gui.widgetBox(self.mainArea, addSpace=False, orientation=0)
        self.infot1 = gui.widgetLabel(e, "<center>No test results.</center>")
        self.mainArea.setMinimumWidth(650)

        self.warning = gui.widgetBox(self.controlArea, "Warning:")
        self.warning_info = gui.widgetLabel(self.warning, "")
        self.warning.hide()

        self.stats = self.dist = self.conts = []
        self.is_continuous = False
        self.disc_palette = colorpalette.ColorPaletteGenerator()

        self.update_display_box()
        self.graphButton.clicked.connect(self.save_graph)

    def eventFilter(self, obj, event):
        if obj is self.box_view.viewport() and \
                event.type() == QtCore.QEvent.Resize:
            self.layout_changed()

        return super().eventFilter(obj, event)

    # noinspection PyTypeChecker
    def set_data(self, dataset):
        if dataset is not None and (
                not bool(dataset) or not len(dataset.domain)):
            dataset = None
        self.closeContext()
        self.dataset = dataset
        self.dist = self.stats = self.conts = []
        self.grouping_select = []
        self.attributes_select = []
        self.attr_list_box.clear()
        self.group_list_box.clear()
        if dataset:
            self.attributes = [(a.name, vartype(a)) for a in dataset.domain]
            self.grouping = ["None"] + [(a.name, vartype(a))
                                        for a in dataset.domain
                                        if a.is_discrete]
            self.grouping_select = [0]
            self.attributes_select = [0]

            self.openContext(self.dataset)

            self.attr_changed()
        else:
            self.reset_all_data()

    def reset_all_data(self):
        self.attr_list_box.clear()
        self.group_list_box.clear()
        self.clear_scene()
        self.infot1.setText("")

    def attr_changed(self):
        self.compute_box_data()
        self.update_display_box()
        self.layout_changed()

        if self.is_continuous:
            heights = 90 if self.show_annotations else 60
            self.box_view.centerOn(self.scene_min_x + self.scene_width / 2,
                                  -30 - len(self.stats) * heights / 2 + 45)
        else:
            self.box_view.centerOn(self.scene_width / 2,
                                  -30 - len(self.boxes) * 40 / 2 + 45)

    def compute_box_data(self):
        dataset = self.dataset
        if dataset is None:
            self.stats = self.dist = self.conts = []
            return
        attr_ind = self.attributes_select[0]
        attr = dataset.domain[attr_ind]
        self.is_continuous = attr.is_continuous
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
            self.label_txts_all = dataset.domain[group_ind].values
        else:
            self.dist = datacaching.getCached(
                dataset, distribution.get_distribution, (dataset, attr_ind))
            self.conts = []
            if self.is_continuous:
                self.stats = [BoxData(self.dist)]
            self.label_txts_all = [""]
        self.label_txts = [txts for stat, txts in zip(self.stats,
                                                      self.label_txts_all)
                           if stat.N > 0]
        self.stats = [stat for stat in self.stats if stat.N > 0]

    def update_display_box(self):
        if self.is_continuous:
            self.stretching_box.hide()
            self.display_box.show()
            group_by = self.grouping_select[0]
            self.compare_rb.setEnabled(group_by != 0)
        else:
            self.stretching_box.show()
            self.display_box.hide()

    def clear_scene(self):
        self.box_scene.clear()
        self.attr_labels = []
        self.labels = []
        self.boxes = []
        self.mean_labels = []
        self.posthoc_lines = []

    def layout_changed(self):
        self.clear_scene()
        if self.dataset is None or len(self.conts) == len(self.dist) == 0:
            return

        if not self.is_continuous:
            return self.display_changed_disc()

        attr = self.attributes[self.attributes_select[0]][0]
        attr = self.dataset.domain[attr]

        self.mean_labels = [self.mean_label(stat, attr, lab)
                            for stat, lab in zip(self.stats, self.label_txts)]
        self.draw_axis()
        self.boxes = [self.box_group(stat) for stat in self.stats]
        self.labels = [self.label_group(stat, attr, mean_lab)
                       for stat, mean_lab in zip(self.stats, self.mean_labels)]
        self.attr_labels = [QtGui.QGraphicsSimpleTextItem(lab)
                            for lab in self.label_txts]
        for it in itertools.chain(self.labels, self.boxes, self.attr_labels):
            self.box_scene.addItem(it)
        self.display_changed()

    def display_changed(self):
        if self.dataset is None:
            return

        if not self.is_continuous:
            return self.display_changed_disc()

        self.order = list(range(len(self.stats)))
        criterion = self._sorting_criteria_attrs[self.compare]
        if criterion:
            self.order = sorted(
                self.order, key=lambda i: getattr(self.stats[i], criterion))

        heights = 90 if self.show_annotations else 60

        for row, box_index in enumerate(self.order):
            y = (-len(self.stats) + row) * heights + 10
            self.boxes[box_index].setY(y)
            labels = self.labels[box_index]

            if self.show_annotations:
                labels.show()
                labels.setY(y)
            else:
                labels.hide()

            label = self.attr_labels[box_index]
            label.setY(y - 15 - label.boundingRect().height())
            if self.show_annotations:
                label.hide()
            else:
                stat = self.stats[box_index]

                if self.compare == OWBoxPlot.CompareMedians:
                    pos = stat.median + 5 / self.scale_x
                elif self.compare == OWBoxPlot.CompareMeans:
                    pos = stat.mean + 5 / self.scale_x
                else:
                    pos = stat.q25
                label.setX(pos * self.scale_x)
                label.show()

        r = QtCore.QRectF(self.scene_min_x, -30 - len(self.stats) * heights,
                          self.scene_width, len(self.stats) * heights + 90)
        self.box_scene.setSceneRect(r)

        self.compute_tests()
        self.show_posthoc()

    def display_changed_disc(self):
        self.clear_scene()
        self.attr_labels = [QtGui.QGraphicsSimpleTextItem(lab)
                            for lab in self.label_txts_all]

        if not self.stretched:
            if self.grouping_select[0]:
                self.labels = [QtGui.QGraphicsTextItem("{}".format(int(sum(cont))))
                               for cont in self.conts]
            else:
                self.labels = [QtGui.QGraphicsTextItem(str(int(sum(self.dist))))]

        self.draw_axis_disc()
        if self.grouping_select[0]:
            self.disc_palette.set_number_of_colors(len(self.conts[0]))
            self.boxes = [self.strudel(cont) for cont in self.conts]
        else:
            self.disc_palette.set_number_of_colors(len(self.dist))
            self.boxes = [self.strudel(self.dist)]

        for row, box in enumerate(self.boxes):
            y = (-len(self.boxes) + row) * 40 + 10
            self.box_scene.addItem(box)
            box.setPos(0, y)
            label = self.attr_labels[row]
            b = label.boundingRect()
            label.setPos(-b.width() - 10, y - b.height() / 2)
            self.box_scene.addItem(label)
            if not self.stretched:
                label = self.labels[row]
                b = label.boundingRect()
                if self.grouping_select[0]:
                    right = self.scale_x * sum(self.conts[row])
                else:
                    right = self.scale_x * sum(self.dist)
                label.setPos(right + 10, y - b.height() / 2)
                self.box_scene.addItem(label)

        self.box_scene.setSceneRect(-self.label_width - 5,
                                   -30 - len(self.boxes) * 40,
                                   self.scene_width, len(self.boxes * 40) + 90)
        self.infot1.setText("")

    # noinspection PyPep8Naming
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
        # noinspection PyPep8Naming
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
        if self.compare == OWBoxPlot.CompareNone or len(self.stats) < 2:
            t = ""
        elif any(s.N <= 1 for s in self.stats):
            t = "At least one group has just one instance, " \
                "cannot compute significance"
        elif len(self.stats) == 2:
            if self.compare == OWBoxPlot.CompareMedians:
                t = ""
                # z, self.p = tests.wilcoxon_rank_sum(
                #    self.stats[0].dist, self.stats[1].dist)
                # t = "Mann-Whitney's z: %.1f (p=%.3f)" % (z, self.p)
            else:
                t, self.p = stat_ttest()
                t = "Student's t: %.3f (p=%.3f)" % (t, self.p)
        else:
            if self.compare == OWBoxPlot.CompareMedians:
                t = ""
                # U, self.p = -1, -1
                # t = "Kruskal Wallis's U: %.1f (p=%.3f)" % (U, self.p)
            else:
                F, self.p = stat_ANOVA()
                t = "ANOVA: %.3f (p=%.3f)" % (F, self.p)
        self.infot1.setText("<center>%s</center>" % t)

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

        bv = self.box_view
        viewrect = bv.viewport().rect().adjusted(15, 15, -15, -30)
        self.scale_x = scale_x = viewrect.width() / (gtop - gbottom)

        # In principle we should repeat this until convergence since the new
        # scaling is too conservative. (No chance am I doing this.)
        mlb = min(stat.mean + mean_lab.min_x / scale_x
                  for stat, mean_lab in zip(self.stats, self.mean_labels))
        if mlb < gbottom:
            gbottom = mlb
            self.scale_x = scale_x = viewrect.width() / (gtop - gbottom)

        self.scene_min_x = gbottom * scale_x
        self.scene_width = (gtop - gbottom) * scale_x

        val = first_val
        attr = self.attributes[self.attributes_select[0]][0]
        attr_desc = self.dataset.domain[attr]
        while True:
            l = self.box_scene.addLine(val * scale_x, -1, val * scale_x, 1,
                                      self._pen_axis_tick)
            l.setZValue(100)
            t = self.box_scene.addSimpleText(
                attr_desc.repr_val(val), self._axis_font)
            t.setFlags(t.flags() |
                       QtGui.QGraphicsItem.ItemIgnoresTransformations)
            r = t.boundingRect()
            t.setPos(val * scale_x - r.width() / 2, 8)
            if val >= top:
                break
            val += step
        self.box_scene.addLine(bottom * scale_x - 4, 0,
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

        bv = self.box_view
        viewrect = bv.viewport().rect().adjusted(15, 15, -15, -30)
        self.scene_width = viewrect.width()

        lab_width = max(lab.boundingRect().width() for lab in self.attr_labels)
        lab_width = max(lab_width, 40)
        lab_width = min(lab_width, self.scene_width / 3)
        self.label_width = lab_width

        right_offset = 0  # offset for the right label
        if not self.stretched and self.labels:
            if self.grouping_select[0]:
                rows = list(zip(self.conts, self.labels))
            else:
                rows = [(self.dist, self.labels[0])]
            # available space left of the 'group labels'
            available = self.scene_width - lab_width - 10
            scale_x = (available - right_offset) / max_box
            max_right = max(sum(dist) * scale_x + 10 +
                            lbl.boundingRect().width()
                            for dist, lbl in rows)
            right_offset = max(0, max_right - max_box * scale_x)

        self.scale_x = scale_x = (self.scene_width - lab_width - 10 - right_offset) / max_box

        self.box_scene.addLine(0, 0, max_box * scale_x, 0, self._pen_axis)
        for val in range(0, step * steps + 1, step):
            l = self.box_scene.addLine(val * scale_x, -1, val * scale_x, 1,
                                      self._pen_axis_tick)
            l.setZValue(100)
            t = self.box_scene.addSimpleText(str(val), self._axis_font)
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
        def line(x0, y0, x1, y1, *args):
            return QtGui.QGraphicsLineItem(x0 * scale_x, y0, x1 * scale_x, y1,
                                           *args)

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
        attr_ind = self.attributes_select[0]
        attr = self.dataset.domain[attr_ind]

        ss = np.sum(dist)
        box = QtGui.QGraphicsItemGroup()
        if ss < 1e-6:
            QtGui.QGraphicsRectItem(0, -10, 1, 10, box)
        cum = 0
        get_color = self.disc_palette.getRGB
        for i, v in enumerate(dist):
            if v < 1e-6:
                continue
            if self.stretched:
                v /= ss
            v *= self.scale_x
            rect = QtGui.QGraphicsRectItem(cum + 1, -6, v - 2, 12, box)
            rect.setBrush(QtGui.QBrush(QtGui.QColor(*get_color(i))))
            rect.setPen(QtGui.QPen(QtCore.Qt.NoPen))
            if self.stretched:
                tooltip = "{}: {:.2f}%".format(attr.values[i],
                                               100 * dist[i] / sum(dist))
            else:
                tooltip = "{}: {}".format(attr.values[i], int(dist[i]))
            rect.setToolTip(tooltip)
            cum += v
        return box

    def show_posthoc(self):
        def line(y0, y1):
            it = self.box_scene.addLine(x, y0, x, y1, self._post_line_pen)
            it.setZValue(-100)
            self.posthoc_lines.append(it)

        while self.posthoc_lines:
            self.box_scene.removeItem(self.posthoc_lines.pop())

        if self.compare == OWBoxPlot.CompareNone or len(self.stats) < 2:
            return

        if self.compare == OWBoxPlot.CompareMedians:
            crit_line = "median"
        elif self.compare == OWBoxPlot.CompareMeans:
            crit_line = "mean"
        else:
            assert False

        xs = []

        height = 90 if self.show_annotations else 60

        y_up = -len(self.stats) * height + 10
        for pos, box_index in enumerate(self.order):
            stat = self.stats[box_index]
            x = getattr(stat, crit_line) * self.scale_x
            xs.append(x)
            by = y_up + pos * height
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
            it = self.box_scene.addLine(frm_x - 2, y, xs[to] + 2, y,
                                       self._post_grp_pen)
            self.posthoc_lines.append(it)
            last_to = to

    def save_graph(self):
        from Orange.widgets.data.owsave import OWSave

        save_img = OWSave(parent=self, data=self.box_scene,
                          file_formats=FileFormats.img_writers)
        save_img.exec_()


def main(argv=None):
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QtGui.QApplication(argv)
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "brown-selected"

    data = Orange.data.Table(filename)
    w = OWBoxPlot()
    w.show()
    w.raise_()
    w.set_data(data)
    w.handleNewSignals()
    rval = app.exec_()
    w.set_data(None)
    w.handleNewSignals()
    w.saveSettings()
    return rval

if __name__ == "__main__":
    sys.exit(main())
