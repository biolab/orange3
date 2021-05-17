import math
from collections import namedtuple
from itertools import chain, count
import numpy as np

from AnyQt.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsSimpleTextItem,
    QGraphicsTextItem, QGraphicsItemGroup, QGraphicsLineItem,
    QGraphicsPathItem, QGraphicsRectItem, QSizePolicy
)
from AnyQt.QtGui import QPen, QColor, QBrush, QPainterPath, QPainter, QFont
from AnyQt.QtCore import Qt, QEvent, QRectF, QSize, QSortFilterProxyModel
from orangewidget.utils.listview import ListViewSearch

import scipy.special
from scipy.stats import f_oneway, chi2_contingency

import Orange.data
from Orange.data.filter import FilterDiscrete, FilterContinuous, Values, \
    IsDefined
from Orange.statistics import contingency, distribution

from Orange.widgets import widget, gui
from Orange.widgets.settings import (Setting, DomainContextHandler,
                                     ContextSetting)
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.utils.widgetpreview import WidgetPreview
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


def _quantiles(a, freq, q, interpolation="midpoint"):
    """
    Somewhat like np.quantiles, but with explicit sample frequencies.

    * Only 'higher', 'lower' and 'midpoint' interpolation.
    * `a` MUST be sorted.
    """
    a = np.asarray(a)
    freq = np.asarray(freq)
    assert a.size > 0 and a.size == freq.size
    cumdist = np.cumsum(freq)
    cumdist /= cumdist[-1]

    if interpolation == "midpoint":  # R quantile(..., type=2)
        left = np.searchsorted(cumdist, q, side="left")
        right = np.searchsorted(cumdist, q, side="right")
        # no mid point for the right most position
        np.clip(right, 0, a.size - 1, out=right)
        # right and left will be different only on the `q` boundaries
        # (excluding the right most sample)
        return (a[left] + a[right]) / 2
    elif interpolation == "higher":  # R quantile(... type=1)
        right = np.searchsorted(cumdist, q, side="right")
        np.clip(right, 0, a.size - 1, out=right)
        return a[right]
    elif interpolation == "lower":
        left = np.searchsorted(cumdist, q, side="left")
        return a[left]
    else:  # pragma: no cover
        raise ValueError("invalid interpolation: '{}'".format(interpolation))


ContDataRange = namedtuple("ContDataRange", ["low", "high", "group_value"])
DiscDataRange = namedtuple("DiscDataRange", ["value", "group_value"])


class BoxData:
    def __init__(self, dist, group_val=None):
        self.dist = dist
        self.n = n = np.sum(dist[1])
        if n == 0:
            return
        self.a_min = float(dist[0, 0])
        self.a_max = float(dist[0, -1])
        self.mean = float(np.sum(dist[0] * dist[1]) / n)
        self.var = float(np.sum(dist[1] * (dist[0] - self.mean) ** 2) / n)
        self.dev = math.sqrt(self.var)
        a, freq = np.asarray(dist)
        q25, median, q75 = _quantiles(a, freq, [0.25, 0.5, 0.75])
        self.median = median
        # The code below omits the q25 or q75 in the plot when they are None
        self.q25 = None if q25 == median else q25
        self.q75 = None if q75 == median else q75
        self.data_range = ContDataRange(q25, q75, group_val)


class FilterGraphicsRectItem(QGraphicsRectItem):
    def __init__(self, data_range, *args):
        super().__init__(*args)
        self.data_range = data_range
        self.setFlag(QGraphicsItem.ItemIsSelectable)


class SortProxyModel(QSortFilterProxyModel):
    def lessThan(self, left, right):
        role = self.sortRole()
        l_score = left.data(role)
        r_score = right.data(role)
        return r_score is not None and (l_score is None or l_score < r_score)

class OWBoxPlot(widget.OWWidget):
    name = "Box Plot"
    description = "Visualize the distribution of feature values in a box plot."
    icon = "icons/BoxPlot.svg"
    priority = 100
    keywords = ["whisker"]

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        selected_data = Output("Selected Data", Orange.data.Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Orange.data.Table)

    class Warning(widget.OWWidget.Warning):
        no_vars = widget.Msg(
            "Data contains no categorical or numeric variables")

    buttons_area_orientation = None

    #: Comparison types for continuous variables
    CompareNone, CompareMedians, CompareMeans = 0, 1, 2

    settingsHandler = DomainContextHandler()
    # If this was a list, context handler would try to match its elements to
    # variable names!
    selection = ContextSetting((), schema_only=True)

    attribute = ContextSetting(None)
    order_by_importance = Setting(False)
    order_grouping_by_importance = Setting(False)
    group_var = ContextSetting(None)
    show_annotations = Setting(True)
    compare = Setting(CompareMeans)
    stattest = Setting(0)
    sig_threshold = Setting(0.05)
    stretched = Setting(True)
    show_labels = Setting(True)
    sort_freqs = Setting(False)

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
    _attr_brush = QBrush(QColor(0x33, 0x00, 0xff))

    graph_name = "box_scene"

    def __init__(self):
        super().__init__()
        self._axis_font = QFont()
        self._axis_font.setPixelSize(12)
        self._label_font = QFont()
        self._label_font.setPixelSize(11)
        self.dataset = None
        self.stats = []
        self.dist = self.conts = None

        self.posthoc_lines = []

        self.label_txts = self.mean_labels = self.boxes = self.labels = \
            self.label_txts_all = self.attr_labels = self.order = []
        self.scale_x = 1
        self.scene_min_x = self.scene_max_x = self.scene_width = 0
        self.label_width = 0

        self.attrs = VariableListModel()
        sorted_model = SortProxyModel(sortRole=Qt.UserRole)
        sorted_model.setSourceModel(self.attrs)
        sorted_model.sort(0)
        box = gui.vBox(self.controlArea, "Variable")
        view = self.attr_list = ListViewSearch()
        view.setModel(sorted_model)
        view.setSelectionMode(view.SingleSelection)
        view.selectionModel().selectionChanged.connect(self.attr_changed)
        view.setMinimumSize(QSize(30, 30))
        # Any other policy than Ignored will let the QListBox's scrollbar
        # set the minimal height (see the penultimate paragraph of
        # http://doc.qt.io/qt-4.8/qabstractscrollarea.html#addScrollBarWidget)
        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        box.layout().addWidget(view)
        gui.checkBox(
            box, self, "order_by_importance",
            "Order by relevance to subgroups",
            tooltip="Order by 𝜒² or ANOVA over the subgroups",
            callback=self.apply_attr_sorting)

        self.group_vars = VariableListModel(placeholder="None")
        sorted_model = SortProxyModel(sortRole=Qt.UserRole)
        sorted_model.setSourceModel(self.group_vars)
        sorted_model.sort(0)

        box = gui.vBox(self.controlArea, "Subgroups")
        view = self.group_list = ListViewSearch()
        view.setModel(sorted_model)
        view.selectionModel().selectionChanged.connect(self.grouping_changed)
        view.setMinimumSize(QSize(30, 30))
        # See the comment above
        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        box.layout().addWidget(view)
        gui.checkBox(
            box, self, "order_grouping_by_importance",
            "Order by relevance to variable",
            tooltip="Order by 𝜒² or ANOVA over the variable values",
            callback=self.apply_group_sorting)

        # TODO: move Compare median/mean to grouping box
        # The vertical size policy is needed to let only the list views expand
        self.display_box = gui.vBox(
            self.controlArea, "Display",
            sizePolicy=(QSizePolicy.Minimum, QSizePolicy.Maximum))

        gui.checkBox(self.display_box, self, "show_annotations", "Annotate",
                     callback=self.update_graph)
        self.compare_rb = gui.radioButtonsInBox(
            self.display_box, self, 'compare',
            btnLabels=["No comparison", "Compare medians", "Compare means"],
            callback=self.update_graph)

        # The vertical size policy is needed to let only the list views expand
        self.stretching_box = box = gui.vBox(
            self.controlArea, box="Display",
            sizePolicy=(QSizePolicy.Minimum, QSizePolicy.Fixed))
        self.stretching_box.sizeHint = self.display_box.sizeHint
        gui.checkBox(
            box, self, 'stretched', "Stretch bars",
            callback=self.update_graph,
            stateWhenDisabled=False)
        gui.checkBox(
            box, self, 'show_labels', "Show box labels",
            callback=self.update_graph)
        self.sort_cb = gui.checkBox(
            box, self, 'sort_freqs', "Sort by subgroup frequencies",
            callback=self.update_graph,
            stateWhenDisabled=False)

        gui.vBox(self.mainArea)
        self.box_scene = QGraphicsScene(self)
        self.box_scene.selectionChanged.connect(self.on_selection_changed)
        self.box_view = QGraphicsView(self.box_scene)
        self.box_view.setRenderHints(QPainter.Antialiasing |
                                     QPainter.TextAntialiasing |
                                     QPainter.SmoothPixmapTransform)
        self.box_view.viewport().installEventFilter(self)

        self.mainArea.layout().addWidget(self.box_view)

        self.stat_test = ""
        self.mainArea.setMinimumWidth(300)
        self.update_box_visibilities()

    def sizeHint(self):
        return QSize(900, 500)

    def eventFilter(self, obj, event):
        if obj is self.box_view.viewport() and \
                event.type() == QEvent.Resize:
            self.update_graph()
        return super().eventFilter(obj, event)

    @property
    def show_stretched(self):
        return self.stretched and self.group_var is not self.attribute

    def reset_attrs(self):
        domain = self.dataset.domain
        self.attrs[:] = [
            var for var in chain(
                domain.class_vars, domain.metas, domain.attributes)
            if var.is_primitive() and not var.attributes.get("hidden", False)]

    def reset_groups(self):
        domain = self.dataset.domain
        self.group_vars[:] = [None] + [
            var for var in chain(
                domain.class_vars, domain.metas, domain.attributes)
            if var.is_discrete and not var.attributes.get("hidden", False)]

    @Inputs.data
    def set_data(self, dataset):
        self.closeContext()
        self._reset_all_data()
        if dataset and not (
                len(dataset.domain.variables)
                or any(var.is_primitive() for var in dataset.domain.metas)):
            self.Warning.no_vars()
            dataset = None

        self.dataset = dataset
        if dataset:
            self.reset_attrs()
            self.reset_groups()
            self._select_default_variables()
            self.openContext(self.dataset)
            self._set_list_view_selections()
            self.compute_box_data()
            self.apply_attr_sorting()
            self.apply_group_sorting()
            self.update_graph()
            self.select_box_items()

        self.update_box_visibilities()
        self.commit()

    def _reset_all_data(self):
        self.clear_scene()
        self.Warning.no_vars.clear()

        self.stats = []
        self.dist = self.conts = None
        self.group_var = None
        self.attribute = None
        self.stat_test = ""
        self.attrs[:] = []
        self.group_vars[:] = [None]
        self.selection = ()

    def _select_default_variables(self):
        # visualize first non-class variable, group by class (if present)
        domain = self.dataset.domain
        if len(self.attrs) > len(domain.class_vars):
            self.attribute = self.attrs[len(domain.class_vars)]
        elif self.attrs:
            self.attribute = self.attrs[0]

        if domain.class_var and domain.class_var.is_discrete:
            self.group_var = domain.class_var

    def _set_list_view_selections(self):
        for view, var, callback in (
                (self.attr_list, self.attribute, self.attr_changed),
                (self.group_list, self.group_var, self.grouping_changed)):
            src_model = view.model().sourceModel()
            if var not in src_model:
                continue
            sel_model = view.selectionModel()
            sel_model.selectionChanged.disconnect(callback)
            row = src_model.indexOf(var)
            index = view.model().index(row, 0)
            sel_model.select(index, sel_model.ClearAndSelect)
            self._ensure_selection_visible(view)
            sel_model.selectionChanged.connect(callback)

    def apply_attr_sorting(self):
        def compute_score(attr):
            # This function and the one in apply_group_sorting are similar, but
            # different in too many details, so they are kept as separate
            # functions.
            # If you discover a bug in this function, check the other one, too.
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
                p = self._chi_square(group_var, attr)[1]
            if math.isnan(p):
                return 2
            return p

        data = self.dataset
        if data is None:
            return
        domain = data.domain
        group_var = self.group_var
        if self.order_by_importance and group_var is not None:
            n_groups = len(group_var.values)
            group_col = data.get_column_view(group_var)[0] if \
                domain.has_continuous_attributes(
                    include_class=True, include_metas=True) else None
            self._sort_list(self.attrs, self.attr_list, compute_score)
        else:
            self._sort_list(self.attrs, self.attr_list, None)

    def apply_group_sorting(self):
        def compute_stat(group):
            # This function and the one in apply_attr_sorting are similar, but
            # different in too many details, so they are kept as separate
            # functions.
            # If you discover a bug in this function, check the other one, too.
            if group is attr:
                return 3
            if group is None:
                return -1
            if attr.is_continuous:
                group_col = data.get_column_view(group)[0].astype(float)
                groups = (attr_col[group_col == i]
                          for i in range(len(group.values)))
                groups = (col[~np.isnan(col)] for col in groups)
                groups = [group for group in groups if len(group)]
                p = f_oneway(*groups)[1] if len(groups) > 1 else 2
            else:
                p = self._chi_square(group, attr)[1]
            if math.isnan(p):
                return 2
            return p

        data = self.dataset
        if data is None:
            return
        attr = self.attribute
        if self.order_grouping_by_importance:
            if attr.is_continuous:
                attr_col = data.get_column_view(attr)[0].astype(float)
            self._sort_list(self.group_vars, self.group_list, compute_stat)
        else:
            self._sort_list(self.group_vars, self.group_list, None)

    def _sort_list(self, source_model, view, key=None):
        if key is None:
            c = count()
            def key(_):  # pylint: disable=function-redefined
                return next(c)

        for i, attr in enumerate(source_model):
            source_model.setData(source_model.index(i), key(attr), Qt.UserRole)
        self._ensure_selection_visible(view)

    @staticmethod
    def _ensure_selection_visible(view):
        selection = view.selectedIndexes()
        if len(selection) == 1:
            view.scrollTo(selection[0])

    def _chi_square(self, group_var, attr):
        # Chi-square with the given distribution into groups
        if not attr.values or not group_var.values:
            return 0, 2, 0
        observed = np.array(
            contingency.get_contingency(self.dataset, group_var, attr))
        observed = observed[observed.sum(axis=1) != 0, :]
        observed = observed[:, observed.sum(axis=0) != 0]
        if min(observed.shape) < 2:
            return 0, 2, 0
        return chi2_contingency(observed)[:3]

    def grouping_changed(self, selected):
        if not selected:
            return  # should never come here
        self.group_var = selected.indexes()[0].data(gui.TableVariable)
        self._variables_changed(self.apply_attr_sorting)

    def attr_changed(self, selected):
        if not selected:
            return  # should never come here
        self.attribute = selected.indexes()[0].data(gui.TableVariable)
        self._variables_changed(self.apply_group_sorting)

    def _variables_changed(self, sorting):
        self.selection = ()
        self.compute_box_data()
        sorting()
        self.update_graph()
        self.update_box_visibilities()
        self.commit()

    def update_graph(self):
        pending_selection = self.selection
        self.box_scene.selectionChanged.disconnect(self.on_selection_changed)
        try:  # not for exceptions, just to reconnect after all possible paths
            self.clear_scene()

            if self.dataset is None or self.attribute is None:
                return

            if self.attribute.is_continuous:
                self._display_changed_cont()
            else:
                self._display_changed_disc()
            self.selection = pending_selection
            self.draw_stat()
            self.select_box_items()

            if self.attribute.is_continuous:
                heights = 90 if self.show_annotations else 60
                self.box_view.centerOn(self.scene_min_x + self.scene_width / 2,
                                       -30 - len(self.stats) * heights / 2 + 45)
            else:
                self.box_view.centerOn(self.scene_width / 2,
                                       -30 - len(self.boxes) * 40 / 2 + 45)
        finally:
            self.box_scene.selectionChanged.connect(self.on_selection_changed)

    def select_box_items(self):
        selection = set(self.selection)
        for box in self.box_scene.items():
            if isinstance(box, FilterGraphicsRectItem):
                box.setSelected(box.data_range in selection)

    def compute_box_data(self):
        attr = self.attribute
        if not attr:
            return
        dataset = self.dataset
        if dataset is None \
                or not attr.is_continuous and not attr.values \
                or self.group_var and not self.group_var.values:
            self.stats = []
            self.dist = self.conts = None
            return
        if self.group_var:
            self.dist = None
            self.conts = contingency.get_contingency(
                dataset, attr, self.group_var)
            missing_val_str = f"missing '{self.group_var.name}'"
            group_var_labels = self.group_var.values + ("",)
            if self.attribute.is_continuous:
                stats, label_texts = [], []
                for cont, value in zip(self.conts.array_with_unknowns,
                                       group_var_labels):
                    if np.sum(cont[1]):
                        stats.append(BoxData(cont, value))
                        label_texts.append(value or missing_val_str)
                self.stats = stats
                self.label_txts_all = label_texts
            else:
                self.label_txts_all = [
                    v or missing_val_str for v, c in zip(
                        group_var_labels, self.conts.array_with_unknowns)
                    if np.sum(c) > 0]
        else:
            self.dist = distribution.get_distribution(dataset, attr)
            self.conts = None
            if self.attribute.is_continuous:
                self.stats = [BoxData(self.dist, None)]
            self.label_txts_all = [""]
        self.label_txts = [txts for stat, txts in zip(self.stats,
                                                      self.label_txts_all)
                           if stat.n > 0]
        self.stats = [stat for stat in self.stats if stat.n > 0]

    def update_box_visibilities(self):
        self.controls.stretched.setDisabled(self.group_var is self.attribute)

        if not self.attribute:
            self.stretching_box.hide()
            self.display_box.hide()
        elif self.attribute.is_continuous:
            self.stretching_box.hide()
            self.display_box.show()
            self.compare_rb.setEnabled(self.group_var is not None)
        else:
            self.stretching_box.show()
            self.display_box.hide()
            self.sort_cb.setEnabled(self.group_var is not None)

    def clear_scene(self):
        self.box_scene.clear()
        self.box_view.viewport().update()
        self.attr_labels = []
        self.labels = []
        self.boxes = []
        self.mean_labels = []
        self.posthoc_lines = []

    def _display_changed_cont(self):
        self.mean_labels = [self.mean_label(stat, self.attribute, lab)
                            for stat, lab in zip(self.stats, self.label_txts)]
        self.draw_axis()
        self.boxes = [self.box_group(stat) for stat in self.stats]
        self.labels = [self.label_group(stat, self.attribute, mean_lab)
                       for stat, mean_lab in zip(self.stats, self.mean_labels)]
        self.attr_labels = [QGraphicsSimpleTextItem(lab)
                            for lab in self.label_txts]
        for it in chain(self.labels, self.attr_labels):
            self.box_scene.addItem(it)

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

        self._compute_tests_cont()
        self._show_posthoc()

    def _display_changed_disc(self):
        self.clear_scene()
        self.attr_labels = [QGraphicsSimpleTextItem(lab)
                            for lab in self.label_txts_all]

        if not self.show_stretched:
            if self.group_var:
                self.labels = [
                    QGraphicsTextItem("{}".format(int(sum(cont))))
                    for cont in self.conts.array_with_unknowns
                    if np.sum(cont) > 0]
            else:
                self.labels = [
                    QGraphicsTextItem(str(int(sum(self.dist))))]

        self.order = list(range(len(self.attr_labels)))

        self.draw_axis_disc()
        if self.group_var:
            conts = self.conts.array_with_unknowns
            self.boxes = [
                self.strudel(cont, val)
                for cont, val in zip(conts, self.group_var.values + ("", ))
                if np.sum(cont) > 0
            ]
            sums_ = np.sum(conts, axis=1)
            sums_ = sums_[sums_ > 0]  # only bars with sum > 0 are shown

            if self.sort_freqs:
                # pylint: disable=invalid-unary-operand-type
                self.order = sorted(self.order, key=(-sums_).__getitem__)
        else:
            conts = self.dist.array_with_unknowns
            self.boxes = [self.strudel(conts)]
            sums_ = [np.sum(conts)]

        for row, box_index in enumerate(self.order):
            y = (-len(self.boxes) + row) * 40 + 10
            box = self.boxes[box_index]
            bars, labels = box[::2], box[1::2]

            self.__draw_group_labels(y, box_index)
            if not self.show_stretched:
                self.__draw_row_counts(
                    y, self.labels[box_index], sums_[box_index]
                )
            if self.show_labels and self.attribute is not self.group_var:
                self.__draw_bar_labels(y, bars, labels)
            self.__draw_bars(y, bars)

        self.box_scene.setSceneRect(-self.label_width - 5,
                                    -30 - len(self.boxes) * 40,
                                    self.scene_width, len(self.boxes * 40) + 90)
        self._compute_tests_disc()

    def __draw_group_labels(self, y, row):
        """Draw group labels

        Parameters
        ----------
        y: int
            vertical offset of bars
        row: int
            row index
        """
        label = self.attr_labels[row]
        b = label.boundingRect()
        label.setPos(-b.width() - 10, y - b.height() / 2)
        self.box_scene.addItem(label)

    def __draw_row_counts(self, y, label, row_sum_):
        """Draw row counts

        Parameters
        ----------
        y: int
            vertical offset of bars
        label: QGraphicsSimpleTextItem
            Label for group
        row_sum_: int
            Sum for the group
        """
        assert not self.attribute.is_continuous
        b = label.boundingRect()
        right = self.scale_x * row_sum_
        label.setPos(right + 10, y - b.height() / 2)
        self.box_scene.addItem(label)

    def __draw_bar_labels(self, y, bars, labels):
        """Draw bar labels

        Parameters
        ----------
        y: int
            vertical offset of bars
        bars: List[FilterGraphicsRectItem]
            list of bars being drawn
        labels: List[QGraphicsTextItem]
            list of labels for corresponding bars
        """
        for text_item, bar_part in zip(labels, bars):
            label = self.Label(
                text_item.toPlainText())
            label.setPos(bar_part.boundingRect().x(),
                         y - label.boundingRect().height() - 8)
            label.setMaxWidth(bar_part.boundingRect().width())
            self.box_scene.addItem(label)

    def __draw_bars(self, y, bars):
        """Draw bars

        Parameters
        ----------
        y: int
            vertical offset of bars

        bars: List[FilterGraphicsRectItem]
            list of bars to draw
        """
        for item in bars:
            item.setPos(0, y)
            self.box_scene.addItem(item)

    # noinspection PyPep8Naming
    def _compute_tests_cont(self):
        # The t-test and ANOVA are implemented here since they efficiently use
        # the widget-specific data in self.stats.
        # The non-parametric tests can't do this, so we use statistics.tests

        # pylint: disable=comparison-with-itself
        def stat_ttest():
            d1, d2 = self.stats
            if d1.n < 2 or d2.n < 2:
                return np.nan, np.nan
            pooled_var = d1.var / d1.n + d2.var / d2.n
            # pylint: disable=comparison-with-itself
            if pooled_var == 0 or np.isnan(pooled_var):
                return np.nan, np.nan
            df = pooled_var ** 2 / \
                ((d1.var / d1.n) ** 2 / (d1.n - 1) +
                 (d2.var / d2.n) ** 2 / (d2.n - 1))
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
            if var_within == 0 or df_within == 0 or df_between == 0:
                return np.nan, np.nan
            F = (var_between / df_between) / (var_within / df_within)
            p = 1 - scipy.special.fdtr(df_between, df_within, F)
            return F, p

        n = len(self.dataset)
        if self.compare == OWBoxPlot.CompareNone or len(self.stats) < 2:
            t = ""
        elif any(s.n <= 1 for s in self.stats):
            t = "At least one group has just one instance, " \
                "cannot compute significance"
        elif len(self.stats) == 2:
            if self.compare == OWBoxPlot.CompareMedians:
                t = ""
                # z, p = tests.wilcoxon_rank_sum(
                #    self.stats[0].dist, self.stats[1].dist)
                # t = "Mann-Whitney's z: %.1f (p=%.3f)" % (z, p)
            else:
                t, p = stat_ttest()
                t = "" if np.isnan(t) else f"Student's t: {t:.3f} (p={p:.3f}, N={n})"
        else:
            if self.compare == OWBoxPlot.CompareMedians:
                t = ""
                # U, p = -1, -1
                # t = "Kruskal Wallis's U: %.1f (p=%.3f)" % (U, p)
            else:
                F, p = stat_ANOVA()
                t = "" if np.isnan(F) else f"ANOVA: {F:.3f} (p={p:.3f}, N={n})"
        self.stat_test = t

    def _compute_tests_disc(self):
        if self.group_var is None or self.attribute is None:
            self.stat_test = ""
        else:
            chi, p, dof = self._chi_square(self.group_var, self.attribute)
            if np.isnan(p):
                self.stat_test = ""
            else:
                self.stat_test = f"χ²: {chi:.2f} (p={p:.3f}, dof={dof})"

    def mean_label(self, stat, attr, val_name):
        label = QGraphicsItemGroup()
        t = QGraphicsSimpleTextItem(attr.str_val(stat.mean), label)
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
        self.scene_max_x = gtop * scale_x
        self.scene_width = self.scene_max_x - self.scene_min_x

        val = first_val
        last_text = self.scene_min_x
        while True:
            l = self.box_scene.addLine(val * scale_x, -1, val * scale_x, 1,
                                       self._pen_axis_tick)
            l.setZValue(100)
            t = QGraphicsSimpleTextItem(
                self.attribute.str_val(val) if not misssing_stats else "?")
            t.setFont(self._axis_font)
            t.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            r = t.boundingRect()
            x_start = val * scale_x - r.width() / 2
            x_finish = x_start + r.width()
            if x_start > last_text + 10 and x_finish < self.scene_max_x:
                t.setPos(x_start, 8)
                self.box_scene.addItem(t)
                last_text = x_finish
            if val >= top:
                break
            val += step
        self.box_scene.addLine(
            bottom * scale_x - 4, 0, top * scale_x + 4, 0, self._pen_axis)

    def draw_stat(self):
        if self.stat_test:
            label = QGraphicsSimpleTextItem(self.stat_test)
            brect = self.box_scene.sceneRect()
            label.setPos(brect.center().x() - label.boundingRect().width()/2,
                         8 + self._axis_font.pixelSize()*2)
            label.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            self.box_scene.addItem(label)

    def draw_axis_disc(self):
        """
        Draw the horizontal axis and sets self.scale_x for discrete attributes
        """
        assert not self.attribute.is_continuous
        if self.show_stretched:
            if not self.attr_labels:
                return
            step = steps = 10
        else:
            if self.group_var:
                max_box = max(float(np.sum(dist))
                              for dist in self.conts.array_with_unknowns)
            else:
                max_box = float(np.sum(self.dist.array_with_unknowns))
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
        if not self.show_stretched and self.labels:
            if self.group_var:
                rows = list(zip(self.conts.array_with_unknowns, self.labels))
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
        if self.show_stretched:
            self.scale_x *= 100

    def label_group(self, stat, attr, mean_lab):
        def centered_text(val, pos):
            t = QGraphicsSimpleTextItem(attr.str_val(val), labels)
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
            med_box_width2 = med_t.boundingRect().width() / 2
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
        if stat.q25 is not None or stat.q75 is not None:
            # if any of them is None it means that its value is equal to median
            box_from = stat.median if stat.q25 is None else stat.q25
            box_to = stat.median if stat.q75 is None else stat.q75
            mbox = FilterGraphicsRectItem(
                stat.data_range, box_from * scale_x, -height / 2,
                (box_to - box_from) * scale_x, height)
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

    def strudel(self, dist, group_val=None):
        attr = self.attribute
        ss = np.sum(dist)
        box = []
        if ss < 1e-6:
            cond = DiscDataRange(None, group_val)
            box.append(FilterGraphicsRectItem(cond, 0, -10, 1, 10))
        cum = 0
        missing_val_str = f"missing '{attr.name}'"
        values = attr.values + ("",)
        colors = attr.palette.qcolors_w_nan
        total = sum(dist)
        for freq, value, color in zip(dist, values, colors):
            if freq < 1e-6:
                continue
            v = freq
            if self.show_stretched:
                v /= ss
            v *= self.scale_x
            cond = DiscDataRange(value, group_val)
            rect = FilterGraphicsRectItem(cond, cum + 1, -6, v - 2, 12)
            rect.setBrush(QBrush(color))
            rect.setPen(QPen(Qt.NoPen))
            value = value or missing_val_str
            if self.show_stretched:
                tooltip = f"{value}: {100 * freq / total:.2f}%"
            else:
                tooltip = f"{value}: ({int(freq)})"
            rect.setToolTip(tooltip)
            text = QGraphicsTextItem(value)
            box.append(rect)
            box.append(text)
            cum += v
        return box

    def on_selection_changed(self):
        self.selection = tuple(item.data_range
                               for item in self.box_scene.selectedItems()
                               if item.data_range)
        self.commit()

    def commit(self):
        conditions = self._gather_conditions()
        if conditions:
            selected = Values(conditions, conjunction=False)(self.dataset)
            selection = np.in1d(
                self.dataset.ids, selected.ids, assume_unique=True).nonzero()[0]
        else:
            selected, selection = None, []
        self.Outputs.selected_data.send(selected)
        self.Outputs.annotated_data.send(
            create_annotated_table(self.dataset, selection))

    def _gather_conditions(self):
        conditions = []
        attr = self.attribute
        group_attr = self.group_var
        for data_range in self.selection:
            if attr.is_discrete:
                # If some value was removed from the data (in case settings are
                # loaded from a scheme), do not include the corresponding
                # filter; this is appropriate since data with such value does
                # not exist anyway
                if not data_range.value:
                    condition = IsDefined([attr], negate=True)
                elif data_range.value not in attr.values:
                    continue
                else:
                    condition = FilterDiscrete(attr, [data_range.value])
            else:
                condition = FilterContinuous(attr, FilterContinuous.Between,
                                             data_range.low, data_range.high)
            if data_range.group_value:
                if not data_range.group_value:
                    grp_filter = IsDefined([group_attr], negate=True)
                elif data_range.group_value not in group_attr.values:
                    continue
                else:
                    grp_filter = FilterDiscrete(group_attr, [data_range.group_value])
                condition = Values([condition, grp_filter], conjunction=True)
            conditions.append(condition)
        return conditions

    def _show_posthoc(self):
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
            line(by + 12, 0)

        used_to = []
        last_to = to = 0
        for frm, frm_x in enumerate(xs[:-1]):
            for to in range(frm + 1, len(xs)):
                if xs[to] - frm_x > 1.5:
                    to -= 1
                    break
            if to in (last_to, frm):
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
        return self.attribute.name if self.attribute else None

    def send_report(self):
        self.report_plot()
        text = ""
        if self.attribute:
            text += "Box plot for attribute '{}' ".format(self.attribute.name)
        if self.group_var:
            text += "grouped by '{}'".format(self.group_var.name)
        if text:
            self.report_caption(text)

    class Label(QGraphicsSimpleTextItem):
        """Boxplot Label with settable maxWidth"""
        # Minimum width to display label text
        MIN_LABEL_WIDTH = 25

        # padding bellow the text
        PADDING = 3

        __max_width = None

        def maxWidth(self):
            return self.__max_width

        def setMaxWidth(self, max_width):
            self.__max_width = max_width

        def paint(self, painter, option, widget):
            """Overrides QGraphicsSimpleTextItem.paint

            If label text is too long, it is elided
            to fit into the allowed region
            """
            if self.__max_width is None:
                width = option.rect.width()
            else:
                width = self.__max_width

            if width < self.MIN_LABEL_WIDTH:
                # if space is too narrow, no label
                return

            fm = painter.fontMetrics()
            text = fm.elidedText(self.text(), Qt.ElideRight, width)
            painter.drawText(
                option.rect.x(),
                option.rect.y() + self.boundingRect().height() - self.PADDING,
                text)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWBoxPlot).run(Orange.data.Table("heart_disease.tab"))
