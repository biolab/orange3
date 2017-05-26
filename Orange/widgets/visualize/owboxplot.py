# -*- coding: utf-8 -*-
import sys
import math
from itertools import chain
import numpy as np

from AnyQt.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsSimpleTextItem,
    QGraphicsTextItem, QGraphicsItemGroup, QGraphicsLineItem,
    QGraphicsPathItem, QGraphicsRectItem, QSizePolicy
)
from AnyQt.QtGui import QPen, QColor, QBrush, QPainterPath, QPainter, QFont
from AnyQt.QtCore import Qt, QEvent, QRectF, QSize

import scipy.special
from scipy.stats import f_oneway, chisquare

import Orange.data
from Orange.data.filter import FilterDiscrete, FilterContinuous, Values
from Orange.statistics import contingency, distribution

from Orange.widgets import widget, gui
from Orange.widgets.settings import (Setting, DomainContextHandler,
                                     ContextSetting)
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.widget import Input, Output


def compute_scale(min_, max_):
    if min_ == max_:
        return math.floor(min_), 1
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
    def __init__(self, dist, attr, group_val_index=None, group_var=None):
        self.dist = dist
        self.n = n = np.sum(dist[1])
        if n == 0:
            return
        self.a_min = float(dist[0, 0])
        self.a_max = float(dist[0, -1])
        self.mean = float(np.sum(dist[0] * dist[1]) / n)
        self.var = float(np.sum(dist[1] * (dist[0] - self.mean) ** 2) / n)
        self.dev = math.sqrt(self.var)
        s = 0
        thresholds = [n / 4, n / 2, n / 4 * 3]
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
                    self.q25, self.median, self.q75 = q
                    break
        else:
            self.q25 = self.q75 = None
            self.median = q[1] if len(q) == 2 else None
        self.conditions = [FilterContinuous(attr, FilterContinuous.Between,
                                            self.q25, self.q75)]
        if group_val_index is not None:
            self.conditions.append(FilterDiscrete(group_var, [group_val_index]))


class FilterGraphicsRectItem(QGraphicsRectItem):
    def __init__(self, conditions, *args):
        super().__init__(*args)
        self.filter = Values(conditions) if conditions else None
        self.setFlag(QGraphicsItem.ItemIsSelectable)


class OWBoxPlot(widget.OWWidget):
    """
    Here's how the widget's functions call each other:

    - `set_data` is a signal handler fills the list boxes and calls
    `grouping_changed`.

    - `grouping_changed` handles changes of grouping attribute: it enables or
    disables the box for ordering, orders attributes and calls `attr_changed`.

    - `attr_changed` handles changes of attribute. It recomputes box data by
    calling `compute_box_data`, shows the appropriate display box
    (discrete/continuous) and then calls`layout_changed`

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
    description = "Visualize the distribution of feature values in a box plot."
    icon = "icons/BoxPlot.svg"
    priority = 100

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        selected_data = Output("Selected Data", Orange.data.Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Orange.data.Table)

    #: Comparison types for continuous variables
    CompareNone, CompareMedians, CompareMeans = 0, 1, 2

    settingsHandler = DomainContextHandler()
    conditions = ContextSetting([])

    attribute = ContextSetting(None)
    order_by_importance = Setting(False)
    group_var = ContextSetting(None)
    show_annotations = Setting(True)
    compare = Setting(CompareMeans)
    stattest = Setting(0)
    sig_threshold = Setting(0.05)
    stretched = Setting(True)
    auto_commit = Setting(True)

    _sorting_criteria_attrs = {
        CompareNone: "", CompareMedians: "median", CompareMeans: "mean"
    }

    _pen_axis_tick = QPen(Qt.white, 5)
    _pen_axis = QPen(Qt.darkGray, 3)
    _pen_median = QPen(QBrush(QColor(0xff, 0xff, 0x00)), 2)
    _pen_paramet = QPen(QBrush(QColor(0x33, 0x00, 0xff)), 2)
    _pen_dotted = QPen(QBrush(QColor(0x33, 0x00, 0xff)), 1)
    _pen_dotted.setStyle(Qt.DotLine)
    _post_line_pen = QPen(Qt.lightGray, 2)
    _post_grp_pen = QPen(Qt.lightGray, 4)
    for pen in (_pen_paramet, _pen_median, _pen_dotted,
                _pen_axis, _pen_axis_tick, _post_line_pen, _post_grp_pen):
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
    _pen_axis_tick.setCapStyle(Qt.FlatCap)

    _box_brush = QBrush(QColor(0x33, 0x88, 0xff, 0xc0))

    _axis_font = QFont()
    _axis_font.setPixelSize(12)
    _label_font = QFont()
    _label_font.setPixelSize(11)
    _attr_brush = QBrush(QColor(0x33, 0x00, 0xff))

    graph_name = "box_scene"

    def __init__(self):
        super().__init__()
        self.stats = []
        self.dataset = None
        self.posthoc_lines = []

        self.label_txts = self.mean_labels = self.boxes = self.labels = \
            self.label_txts_all = self.attr_labels = self.order = []
        self.p = -1.0
        self.scale_x = self.scene_min_x = self.scene_width = 0
        self.label_width = 0

        self.attrs = VariableListModel()
        view = gui.listView(
            self.controlArea, self, "attribute", box="Variable",
            model=self.attrs, callback=self.attr_changed)
        view.setMinimumSize(QSize(30, 30))
        # Any other policy than Ignored will let the QListBox's scrollbar
        # set the minimal height (see the penultimate paragraph of
        # http://doc.qt.io/qt-4.8/qabstractscrollarea.html#addScrollBarWidget)
        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        gui.separator(view.box, 6, 6)
        self.cb_order = gui.checkBox(
            view.box, self, "order_by_importance",
            "Order by relevance",
            tooltip="Order by 𝜒² or ANOVA over the subgroups",
            callback=self.apply_sorting)
        self.group_vars = VariableListModel()
        view = gui.listView(
            self.controlArea, self, "group_var", box="Subgroups",
            model=self.group_vars, callback=self.grouping_changed)
        view.setMinimumSize(QSize(30, 30))
        # See the comment above
        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)

        # TODO: move Compare median/mean to grouping box
        # The vertical size policy is needed to let only the list views expand
        self.display_box = gui.vBox(
            self.controlArea, "Display",
            sizePolicy=(QSizePolicy.Minimum, QSizePolicy.Maximum))

        gui.checkBox(self.display_box, self, "show_annotations", "Annotate",
                     callback=self.display_changed)
        self.compare_rb = gui.radioButtonsInBox(
            self.display_box, self, 'compare',
            btnLabels=["No comparison", "Compare medians", "Compare means"],
            callback=self.layout_changed)

        # The vertical size policy is needed to let only the list views expand
        self.stretching_box = gui.checkBox(
            self.controlArea, self, 'stretched', "Stretch bars", box='Display',
            callback=self.display_changed,
            sizePolicy=(QSizePolicy.Minimum, QSizePolicy.Maximum)).box

        gui.auto_commit(self.controlArea, self, "auto_commit",
                        "Send Selection", "Send Automatically")

        gui.vBox(self.mainArea, addSpace=True)
        self.box_scene = QGraphicsScene()
        self.box_scene.selectionChanged.connect(self.commit)
        self.box_view = QGraphicsView(self.box_scene)
        self.box_view.setRenderHints(QPainter.Antialiasing |
                                     QPainter.TextAntialiasing |
                                     QPainter.SmoothPixmapTransform)
        self.box_view.viewport().installEventFilter(self)

        self.mainArea.layout().addWidget(self.box_view)

        e = gui.hBox(self.mainArea, addSpace=False)
        self.infot1 = gui.widgetLabel(e, "<center>No test results.</center>")
        self.mainArea.setMinimumWidth(600)

        self.stats = self.dist = self.conts = []
        self.is_continuous = False

        self.update_display_box()

    def sizeHint(self):
        return QSize(100, 500)  # Vertical size is regulated by mainArea

    def eventFilter(self, obj, event):
        if obj is self.box_view.viewport() and \
                event.type() == QEvent.Resize:
            self.layout_changed()

        return super().eventFilter(obj, event)

    # noinspection PyTypeChecker
    @Inputs.data
    def set_data(self, dataset):
        if dataset is not None and (
                not bool(dataset) or not len(dataset.domain)):
            dataset = None
        self.closeContext()
        self.dataset = dataset
        self.dist = self.stats = self.conts = []
        self.group_var = None
        self.attribute = None
        if dataset:
            domain = dataset.domain
            self.group_vars[:] = \
                [None] + \
                [a for a in chain(domain.variables, domain.metas)
                 if a.is_discrete]
            self.attrs[:] = chain(domain.variables,
                                  (a for a in domain.metas if a.is_primitive()))
            if self.attrs:
                self.attribute = self.attrs[0]
            if domain.class_var and domain.class_var.is_discrete:
                self.group_var = domain.class_var
            else:
                self.group_var = None  # Reset to trigger selection via callback
            self.openContext(self.dataset)
            self.grouping_changed()
        else:
            self.reset_all_data()
        self.commit()

    def apply_sorting(self):
        def compute_score(attr):
            if attr is group_var:
                return 3
            if attr.is_continuous:
                # One-way ANOVA
                col = data.get_column_view(attr)[0].astype(float)
                groups = (col[group_col == i] for i in range(n_groups))
                groups = (col[~np.isnan(col)] for col in groups)
                groups = [group for group in groups if len(group)]
                p = f_oneway(*groups)[1] if len(groups) > 1 else 2
            else:
                # Chi-square with the given distribution into groups
                # (see degrees of freedom in computation of the p-value)
                if not attr.values or not group_var.values:
                    return 2
                observed = np.array(
                    contingency.get_contingency(data, group_var, attr))
                observed = observed[observed.sum(axis=1) != 0, :]
                observed = observed[:, observed.sum(axis=0) != 0]
                if min(observed.shape) < 2:
                    return 2
                expected = \
                    np.outer(observed.sum(axis=1), observed.sum(axis=0)) / \
                    np.sum(observed)
                p = chisquare(observed.ravel(), f_exp=expected.ravel(),
                              ddof=n_groups - 1)[1]
            if math.isnan(p):
                return 2
            return p

        data = self.dataset
        if data is None:
            return
        domain = data.domain
        attribute = self.attribute
        group_var = self.group_var
        if self.order_by_importance and group_var is not None:
            n_groups = len(group_var.values)
            group_col = data.get_column_view(group_var)[0] if \
                domain.has_continuous_attributes(
                    include_class=True, include_metas=True) else None
            self.attrs.sort(key=compute_score)
        else:
            self.attrs[:] = chain(
                domain.variables,
                (a for a in data.domain.metas if a.is_primitive()))
        self.attribute = attribute

    def reset_all_data(self):
        self.clear_scene()
        self.infot1.setText("")
        self.attrs[:] = []
        self.group_vars[:] = []
        self.is_continuous = False
        self.update_display_box()

    def grouping_changed(self):
        self.cb_order.setEnabled(self.group_var is not None)
        self.apply_sorting()
        self.attr_changed()

    def select_box_items(self):
        temp_cond = self.conditions.copy()
        for box in self.box_scene.items():
            if isinstance(box, FilterGraphicsRectItem):
                box.setSelected(box.filter.conditions in
                                [c.conditions for c in temp_cond])

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
        attr = self.attribute
        if not attr:
            return
        dataset = self.dataset
        self.is_continuous = attr.is_continuous
        if dataset is None or not self.is_continuous and not attr.values or \
                        self.group_var and not self.group_var.values:
            self.stats = self.dist = self.conts = []
            return
        if self.group_var:
            self.dist = []
            self.conts = contingency.get_contingency(
                dataset, attr, self.group_var)
            if self.is_continuous:
                self.stats = [BoxData(cont, attr, i, self.group_var)
                              for i, cont in enumerate(self.conts)]
            self.label_txts_all = self.group_var.values
        else:
            self.dist = distribution.get_distribution(dataset, attr)
            self.conts = []
            if self.is_continuous:
                self.stats = [BoxData(self.dist, attr, None)]
            self.label_txts_all = [""]
        self.label_txts = [txts for stat, txts in zip(self.stats,
                                                      self.label_txts_all)
                           if stat.n > 0]
        self.stats = [stat for stat in self.stats if stat.n > 0]

    def update_display_box(self):
        if self.is_continuous:
            self.stretching_box.hide()
            self.display_box.show()
            self.compare_rb.setEnabled(self.group_var is not None)
        else:
            self.stretching_box.show()
            self.display_box.hide()

    def clear_scene(self):
        self.closeContext()
        self.box_scene.clearSelection()
        self.box_scene.clear()
        self.attr_labels = []
        self.labels = []
        self.boxes = []
        self.mean_labels = []
        self.posthoc_lines = []
        self.openContext(self.dataset)

    def layout_changed(self):
        attr = self.attribute
        if not attr:
            return
        self.clear_scene()
        if self.dataset is None or len(self.conts) == len(self.dist) == 0:
            return

        if not self.is_continuous:
            return self.display_changed_disc()

        self.mean_labels = [self.mean_label(stat, attr, lab)
                            for stat, lab in zip(self.stats, self.label_txts)]
        self.draw_axis()
        self.boxes = [self.box_group(stat) for stat in self.stats]
        self.labels = [self.label_group(stat, attr, mean_lab)
                       for stat, mean_lab in zip(self.stats, self.mean_labels)]
        self.attr_labels = [QGraphicsSimpleTextItem(lab)
                            for lab in self.label_txts]
        for it in chain(self.labels, self.attr_labels):
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
            vals = [getattr(stat, criterion) for stat in self.stats]
            overmax = max((val for val in vals if val is not None), default=0) \
                      + 1
            vals = [val if val is not None else overmax for val in vals]
            self.order = sorted(self.order, key=vals.__getitem__)

        heights = 90 if self.show_annotations else 60

        for row, box_index in enumerate(self.order):
            y = (-len(self.stats) + row) * heights + 10
            for item in self.boxes[box_index]:
                self.box_scene.addItem(item)
                item.setY(y)
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

                if self.compare == OWBoxPlot.CompareMedians and \
                        stat.median is not None:
                    pos = stat.median + 5 / self.scale_x
                elif self.compare == OWBoxPlot.CompareMeans or stat.q25 is None:
                    pos = stat.mean + 5 / self.scale_x
                else:
                    pos = stat.q25
                label.setX(pos * self.scale_x)
                label.show()

        r = QRectF(self.scene_min_x, -30 - len(self.stats) * heights,
                   self.scene_width, len(self.stats) * heights + 90)
        self.box_scene.setSceneRect(r)

        self.compute_tests()
        self.show_posthoc()
        self.select_box_items()

    def display_changed_disc(self):
        self.clear_scene()
        self.attr_labels = [QGraphicsSimpleTextItem(lab)
                            for lab in self.label_txts_all]

        if not self.stretched:
            if self.group_var:
                self.labels = [
                    QGraphicsTextItem("{}".format(int(sum(cont))))
                    for cont in self.conts]
            else:
                self.labels = [
                    QGraphicsTextItem(str(int(sum(self.dist))))]

        self.draw_axis_disc()
        if self.group_var:
            self.boxes = [self.strudel(cont, i) for i, cont in enumerate(self.conts)]
        else:
            self.boxes = [self.strudel(self.dist)]

        for row, box in enumerate(self.boxes):
            y = (-len(self.boxes) + row) * 40 + 10

            label = self.attr_labels[row]
            b = label.boundingRect()
            label.setPos(-b.width() - 10, y - b.height() / 2)
            self.box_scene.addItem(label)
            if not self.stretched:
                label = self.labels[row]
                b = label.boundingRect()
                if self.group_var:
                    right = self.scale_x * sum(self.conts[row])
                else:
                    right = self.scale_x * sum(self.dist)
                label.setPos(right + 10, y - b.height() / 2)
                self.box_scene.addItem(label)

            if self.attribute is not self.group_var:
                for text_item, bar_part in zip(box[1::2], box[::2]):
                    label = QGraphicsSimpleTextItem(
                        text_item.toPlainText())
                    label.setPos(bar_part.boundingRect().x(),
                                 y - label.boundingRect().height() - 8)
                    self.box_scene.addItem(label)
            for item in box:
                if isinstance(item, QGraphicsTextItem):
                    continue
                self.box_scene.addItem(item)
                item.setPos(0, y)
        self.box_scene.setSceneRect(-self.label_width - 5,
                                    -30 - len(self.boxes) * 40,
                                    self.scene_width, len(self.boxes * 40) + 90)
        self.infot1.setText("")
        self.select_box_items()

    # noinspection PyPep8Naming
    def compute_tests(self):
        # The t-test and ANOVA are implemented here since they efficiently use
        # the widget-specific data in self.stats.
        # The non-parametric tests can't do this, so we use statistics.tests
        def stat_ttest():
            d1, d2 = self.stats
            if d1.n == 0 or d2.n == 0:
                return np.nan, np.nan
            pooled_var = d1.var / d1.n + d2.var / d2.n
            df = pooled_var ** 2 / \
                ((d1.var / d1.n) ** 2 / (d1.n - 1) +
                 (d2.var / d2.n) ** 2 / (d2.n - 1))
            if pooled_var == 0:
                return np.nan, np.nan
            t = abs(d1.mean - d2.mean) / math.sqrt(pooled_var)
            p = 2 * (1 - scipy.special.stdtr(df, t))
            return t, p

        # TODO: Check this function
        # noinspection PyPep8Naming
        def stat_ANOVA():
            if any(stat.n == 0 for stat in self.stats):
                return np.nan, np.nan
            n = sum(stat.n for stat in self.stats)
            grand_avg = sum(stat.n * stat.mean for stat in self.stats) / n
            var_between = sum(stat.n * (stat.mean - grand_avg) ** 2
                              for stat in self.stats)
            df_between = len(self.stats) - 1

            var_within = sum(stat.n * stat.var for stat in self.stats)
            df_within = n - len(self.stats)
            F = (var_between / df_between) / (var_within / df_within)
            p = 1 - scipy.special.fdtr(df_between, df_within, F)
            return F, p

        if self.compare == OWBoxPlot.CompareNone or len(self.stats) < 2:
            t = ""
        elif any(s.n <= 1 for s in self.stats):
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
        label = QGraphicsItemGroup()
        t = QGraphicsSimpleTextItem(
            "%.*f" % (attr.number_of_decimals + 1, stat.mean), label)
        t.setFont(self._label_font)
        bbox = t.boundingRect()
        w2, h = bbox.width() / 2, bbox.height()
        t.setPos(-w2, -h)
        tpm = QGraphicsSimpleTextItem(
            " \u00b1 " + "%.*f" % (attr.number_of_decimals + 1, stat.dev),
            label)
        tpm.setFont(self._label_font)
        tpm.setPos(w2, -h)
        if val_name:
            vnm = QGraphicsSimpleTextItem(val_name + ": ", label)
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
        misssing_stats = not self.stats
        stats = self.stats or [BoxData(np.array([[0.], [1.]]), self.attribute)]
        mean_labels = self.mean_labels or [self.mean_label(stats[0], self.attribute, "")]
        bottom = min(stat.a_min for stat in stats)
        top = max(stat.a_max for stat in stats)

        first_val, step = compute_scale(bottom, top)
        while bottom <= first_val:
            first_val -= step
        bottom = first_val
        no_ticks = math.ceil((top - first_val) / step) + 1
        top = max(top, first_val + no_ticks * step)

        gbottom = min(bottom, min(stat.mean - stat.dev for stat in stats))
        gtop = max(top, max(stat.mean + stat.dev for stat in stats))

        bv = self.box_view
        viewrect = bv.viewport().rect().adjusted(15, 15, -15, -30)
        self.scale_x = scale_x = viewrect.width() / (gtop - gbottom)

        # In principle we should repeat this until convergence since the new
        # scaling is too conservative. (No chance am I doing this.)
        mlb = min(stat.mean + mean_lab.min_x / scale_x
                  for stat, mean_lab in zip(stats, mean_labels))
        if mlb < gbottom:
            gbottom = mlb
            self.scale_x = scale_x = viewrect.width() / (gtop - gbottom)

        self.scene_min_x = gbottom * scale_x
        self.scene_width = (gtop - gbottom) * scale_x

        val = first_val
        while True:
            l = self.box_scene.addLine(val * scale_x, -1, val * scale_x, 1,
                                       self._pen_axis_tick)
            l.setZValue(100)

            t = self.box_scene.addSimpleText(
                self.attribute.repr_val(val) if not misssing_stats else "?",
                self._axis_font)
            t.setFlags(
                t.flags() | QGraphicsItem.ItemIgnoresTransformations)
            r = t.boundingRect()
            t.setPos(val * scale_x - r.width() / 2, 8)
            if val >= top:
                break
            val += step
        self.box_scene.addLine(
            bottom * scale_x - 4, 0, top * scale_x + 4, 0, self._pen_axis)

    def draw_axis_disc(self):
        """
        Draw the horizontal axis and sets self.scale_x for discrete attributes
        """
        if self.stretched:
            step = steps = 10
        else:
            if self.group_var:
                max_box = max(float(np.sum(dist)) for dist in self.conts)
            else:
                max_box = float(np.sum(self.dist))
            if max_box == 0:
                self.scale_x = 1
                return
            _, step = compute_scale(0, max_box)
            step = int(step) if step > 1 else 1
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
            if self.group_var:
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

        self.scale_x = scale_x = \
            (self.scene_width - lab_width - 10 - right_offset) / max_box

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
            t = QGraphicsSimpleTextItem(
                "%.*f" % (attr.number_of_decimals + 1, val), labels)
            t.setFont(self._label_font)
            bbox = t.boundingRect()
            t.setPos(pos - bbox.width() / 2, 22)
            return t

        def line(x, down=1):
            QGraphicsLineItem(x, 12 * down, x, 20 * down, labels)

        def move_label(label, frm, to):
            label.setX(to)
            to += t_box.width() / 2
            path = QPainterPath()
            path.lineTo(0, 4)
            path.lineTo(to - frm, 4)
            path.lineTo(to - frm, 8)
            p = QGraphicsPathItem(path)
            p.setPos(frm, 12)
            labels.addToGroup(p)

        labels = QGraphicsItemGroup()

        labels.addToGroup(mean_lab)
        m = stat.mean * self.scale_x
        mean_lab.setPos(m, -22)
        line(m, -1)

        if stat.median is not None:
            msc = stat.median * self.scale_x
            med_t = centered_text(stat.median, msc)
            med_box_width2 = med_t.boundingRect().width()
            line(msc)

        if stat.q25 is not None:
            x = stat.q25 * self.scale_x
            t = centered_text(stat.q25, x)
            t_box = t.boundingRect()
            med_left = msc - med_box_width2
            if x + t_box.width() / 2 >= med_left - 5:
                move_label(t, x, med_left - t_box.width() - 5)
            else:
                line(x)

        if stat.q75 is not None:
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
            return QGraphicsLineItem(x0 * scale_x, y0, x1 * scale_x, y1, *args)

        scale_x = self.scale_x
        box = []
        whisker1 = line(stat.a_min, -1.5, stat.a_min, 1.5)
        whisker2 = line(stat.a_max, -1.5, stat.a_max, 1.5)
        vert_line = line(stat.a_min, 0, stat.a_max, 0)
        mean_line = line(stat.mean, -height / 3, stat.mean, height / 3)
        for it in (whisker1, whisker2, mean_line):
            it.setPen(self._pen_paramet)
        vert_line.setPen(self._pen_dotted)
        var_line = line(stat.mean - stat.dev, 0, stat.mean + stat.dev, 0)
        var_line.setPen(self._pen_paramet)
        box.extend([whisker1, whisker2, vert_line, mean_line, var_line])
        if stat.q25 is not None and stat.q75 is not None:
            mbox = FilterGraphicsRectItem(
                stat.conditions, stat.q25 * scale_x, -height / 2,
                (stat.q75 - stat.q25) * scale_x, height)
            mbox.setBrush(self._box_brush)
            mbox.setPen(QPen(Qt.NoPen))
            mbox.setZValue(-200)
            box.append(mbox)

        if stat.median is not None:
            median_line = line(stat.median, -height / 2,
                               stat.median, height / 2)
            median_line.setPen(self._pen_median)
            median_line.setZValue(-150)
            box.append(median_line)

        return box

    def strudel(self, dist, group_val_index=None):
        attr = self.attribute
        ss = np.sum(dist)
        box = []
        if ss < 1e-6:
            cond = [FilterDiscrete(attr, None)]
            if group_val_index is not None:
                cond.append(FilterDiscrete(self.group_var, [group_val_index]))
            box.append(FilterGraphicsRectItem(cond, 0, -10, 1, 10))
        cum = 0
        for i, v in enumerate(dist):
            if v < 1e-6:
                continue
            if self.stretched:
                v /= ss
            v *= self.scale_x
            cond = [FilterDiscrete(attr, [i])]
            if group_val_index is not None:
                cond.append(FilterDiscrete(self.group_var, [group_val_index]))
            rect = FilterGraphicsRectItem(cond, cum + 1, -6, v - 2, 12)
            rect.setBrush(QBrush(QColor(*attr.colors[i])))
            rect.setPen(QPen(Qt.NoPen))
            if self.stretched:
                tooltip = "{}: {:.2f}%".format(attr.values[i],
                                               100 * dist[i] / sum(dist))
            else:
                tooltip = "{}: {}".format(attr.values[i], int(dist[i]))
            rect.setToolTip(tooltip)
            text = QGraphicsTextItem(attr.values[i])
            box.append(rect)
            box.append(text)
            cum += v
        return box

    def commit(self):
        self.conditions = [item.filter for item in
                           self.box_scene.selectedItems() if item.filter]
        selected, selection = None, []
        if self.conditions:
            selected = Values(self.conditions, conjunction=False)(self.dataset)
            selection = [i for i, inst in enumerate(self.dataset)
                         if inst in selected]
        self.Outputs.selected_data.send(selected)
        self.Outputs.annotated_data.send(create_annotated_table(self.dataset, selection))

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
        else:
            crit_line = "mean"

        xs = []

        height = 90 if self.show_annotations else 60

        y_up = -len(self.stats) * height + 10
        for pos, box_index in enumerate(self.order):
            stat = self.stats[box_index]
            x = getattr(stat, crit_line)
            if x is None:
                continue
            x *= self.scale_x
            xs.append(x * self.scale_x)
            by = y_up + pos * height
            line(by + 12, 3)
            line(by - 12, by - 25)

        used_to = []
        last_to = to = 0
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

    def get_widget_name_extension(self):
        if self.attribute:
            return self.attribute.name

    def send_report(self):
        self.report_plot()
        text = ""
        if self.attribute:
            text += "Box plot for attribute '{}' ".format(self.attribute.name)
        if self.group_var:
            text += "grouped by '{}'".format(self.group_var.name)
        if text:
            self.report_caption(text)


def main(argv=None):
    from AnyQt.QtWidgets import QApplication
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QApplication(argv)
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "heart_disease"

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
