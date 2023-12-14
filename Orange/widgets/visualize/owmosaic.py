from collections import defaultdict
from functools import reduce
from itertools import product, chain, repeat, combinations
from math import sqrt, log
from operator import mul, attrgetter

import numpy as np
from scipy.stats import distributions
from scipy.special import comb
from AnyQt.QtCore import Qt, QSize, pyqtSignal as Signal
from AnyQt.QtGui import QColor, QPainter, QPen, QStandardItem
from AnyQt.QtWidgets import (
    QGraphicsScene, QGraphicsLineItem, QGraphicsItemGroup, QLabel, QComboBox)

from Orange.data import Table, filter, Variable, Domain, DiscreteVariable
from Orange.data.sql.table import SqlTable, LARGE_TABLE, DEFAULT_SAMPLE_TIME
from Orange.preprocess import Discretize
from Orange.preprocess.discretize import EqualFreq
from Orange.preprocess.score import ReliefF
from Orange.statistics.distribution import get_distribution, get_distributions
from Orange.widgets import gui, settings
from Orange.widgets.settings import (
    Setting, DomainContextHandler, ContextSetting)
from Orange.widgets.utils import to_html, get_variable_values_sorted
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.utils import (
    CanvasText, CanvasRectangle, ViewWithPress)
from Orange.widgets.visualize.utils import vizrank
from Orange.widgets.visualize.utils.vizrank import (
    VizRankDialogAttrs, VizRankMixin)
from Orange.widgets.visualize.utils.plotutils import wrap_legend_items
from Orange.widgets.widget import OWWidget, Msg, Input, Output


class MosaicVizRank(VizRankDialogAttrs):
    pairSelected = Signal(Variable, Variable, Variable, Variable)
    _AttrRole = next(gui.OrangeUserRole)

    def __init__(self, parent, data, attr_color, attr_range_index):
        self.attr_range_index = attr_range_index
        super().__init__(parent, data, attr_color=attr_color)
        self.marginal = {}

        box = gui.hBox(self)
        label = QLabel("Score Mosaics with ")

        combo = self.attrs_combo = QComboBox()
        combo.addItems(
            ["a single variable", "two variables", "three variables",
             "four variables", "at most two variables",
             "at most three variables", "at most four variables"])

        def disable_item(i):
            enabled = Qt.ItemIsSelectable | Qt.ItemIsEnabled
            item = combo.model().item(i)
            item.setFlags(item.flags() & ~enabled)
            if self.attr_range_index == i:
                # select one attribute less, or a pair instead of a single
                self.attr_range_index = i - 1 if i else 1

        if self.attr_color is None:
            disable_item(0)  # can't do a single attribute with Pearson
        if len(data.domain.attributes) < 4:
            disable_item(3)  # four attributes
        if len(data.domain.attributes) < 3:
            disable_item(2)  # three attributes

        combo.activated.connect(self.on_attrs_changed)
        combo.setCurrentIndex(self.attr_range_index)
        box.layout().addWidget(label)
        box.layout().addWidget(combo)
        gui.rubber(box)
        self.layout().addWidget(self.button)

    def on_attrs_changed(self):
        if self.run_state.state == vizrank.RunState.Running:
            self.pause_computation()

        new_attrs = self.attrs_combo.currentIndex()
        if new_attrs == self.attr_range():
            self.set_button_state()
        else:
            self.set_button_state(
                label="Restart with new settings",
                enabled=True)

    def prepare_run(self):
        super().prepare_run()
        data = self.data
        if self.attr_color is not None:
            self.marginal = get_distribution(data, self.attr_color)
            self.marginal.normalize()
        else:
            self.marginal = get_distributions(data)
            for dist in self.marginal:
                dist.normalize()

    def start_computation(self):
        if self.attr_range_index != self.attrs_combo.currentIndex():
            self.attr_range_index = self.attrs_combo.currentIndex()
            self.set_run_state(vizrank.RunState.Initialized)
        super().start_computation()

    def score_attributes(self):
        """
        Order attributes by Relief if there is a target variable. In case of
        ties or without target, order by name.

        Add the class variable at the beginning when not coloring by class
        distribution.

        If `self.attrs` is not `None`, keep the ordering and just add or remove
        the class as needed.
        """
        if self.attr_color is None:
            return sorted(self.data.domain, key=attrgetter("name"))
        data = self.data.transform(Domain(
            [attr for attr in self.attrs if attr is not self.attr_color],
            self.attr_color))
        weights = ReliefF(n_iterations=100, k_nearest=10)(data)
        attrs = sorted(zip(weights, data.domain.attributes),
                       key=lambda x: (-x[0], x[1].name))
        return [a for _, a in attrs]

    def attr_range(self):
        n_attrs = len(self.data.domain.attributes)
        mm = 2 if self.attr_color is None else 1
        min_attrs, max_attrs = [
            (mm, mm), (2, 2), (3, 3), (4, 4),
            (mm, 2), (mm, 3), (mm, 4)][self.attr_range_index]
        return min_attrs, 1 + min(n_attrs, max_attrs)

    def state_count(self):
        return sum(comb(len(self.attr_order), k, exact=True)
                   for k in range(*self.attr_range()))

    def state_generator(self):
        for n_attrs in range(*self.attr_range()):
            yield from combinations(list(range(len(self.attr_order))), n_attrs)

    def compute_score(self, state):
        """
        Compute score using chi-square test of independence.

        If mosaic colors by class distribution, chi-square is computed by
        comparing the expected (prior) and observed class distribution in
        each cell. Otherwise, compute the independence of the shown attributes.
        """
        data = self.data
        domain = data.domain
        attrlist = [self.attr_order[i] for i in state]
        cond_dist = get_conditional_distribution(data, attrlist)[0]
        n = cond_dist[""]
        ss = 0
        if self.attr_color is not None:
            class_values = self.attr_color.values
        else:
            class_values = None
            attr_indices = [domain.index(attr) for attr in attrlist]
        for indices in product(*(range(len(a.values)) for a in attrlist)):
            attr_vals = "-".join(attr.values[ind]
                                 for attr, ind in zip(attrlist, indices))
            total = cond_dist[attr_vals]
            if class_values:  # showing class distributions
                for i, class_val in enumerate(class_values):
                    expected = total * self.marginal[i]
                    if expected > 1e-6:
                        observed = cond_dist[attr_vals + '-' + class_val]
                        ss += (expected - observed) ** 2 / expected
            else:
                observed = cond_dist[attr_vals]
                expected = n * reduce(
                    mul,
                    (self.marginal[attr_idx][ind]
                     for attr_idx, ind in zip(attr_indices, indices)))
                if expected > 1e-6:
                    ss += (expected - observed) ** 2 / expected
        if class_values:
            dof = (len(class_values) - 1) * \
                  reduce(mul, (len(attr.values) for attr in attrlist))
        else:
            dof = reduce(mul, (len(attr.values) - 1 for attr in attrlist))
        return distributions.chi2.sf(ss, dof)

    def bar_length(self, score):
        return 1 if score == 0 else min(1, -log(score, 10) / 50)

    def on_selection_changed(self, selected, deselected):
        if not selected.isEmpty():
            attrs = selected.indexes()[0].data(self._AttrRole)
            self.selectionChanged.emit(attrs + (None, ) * (4 - len(attrs)))

    def auto_select(self, attrs):
        model = self.rank_model
        self.rank_table.selectionModel().clear()
        for row in range(model.rowCount()):
            row_attrs = model.data(model.index(row, 0), self._AttrRole)
            if row_attrs == tuple(attrs):
                self.rank_table.selectRow(row)
                return

    def row_for_state(self, score, state):
        # Override the inherited method to put the class (if present)
        # to the start of the list, so it's on the x-axis."""
        attrs = tuple(
            sorted((self.attr_order[x] for x in state),
                   key=lambda attr: (attr is not self.attr_color, attr.name)))
        item = QStandardItem(", ".join(a.name for a in attrs))
        item.setData(attrs, self._AttrRole)
        return [item]

    def emit_run_state_changed(self):
        self.runStateChanged.emit(self.run_state.state,
                                  {"attr_range_index": self.attr_range_index})

    def closeEvent(self, event):
        self.attrs_combo.setCurrentIndex(self.attr_range_index)
        super().closeEvent(event)


class OWMosaicDisplay(OWWidget, VizRankMixin(MosaicVizRank)):
    name = "Mosaic Display"
    description = "Display data in a mosaic plot."
    icon = "icons/MosaicDisplay.svg"
    priority = 220
    keywords = "mosaic display"

    class Inputs:
        data = Input("Data", Table, default=True)
        data_subset = Input("Data Subset", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    settingsHandler = DomainContextHandler()
    settings_version = 2
    use_boxes = Setting(True)
    variable1: Variable = ContextSetting(None)
    variable2: Variable = ContextSetting(None)
    variable3: Variable = ContextSetting(None)
    variable4: Variable = ContextSetting(None)
    variable_color: DiscreteVariable = ContextSetting(None)
    selection = Setting(set(), schema_only=True)
    vizrank_attr_range_index = Setting(6)

    BAR_WIDTH = 5
    SPACING = 4
    ATTR_NAME_OFFSET = 20
    ATTR_VAL_OFFSET = 3
    BLUE_COLORS = [QColor(255, 255, 255), QColor(210, 210, 255),
                   QColor(110, 110, 255), QColor(0, 0, 255)]
    RED_COLORS = [QColor(255, 255, 255), QColor(255, 200, 200),
                  QColor(255, 100, 100), QColor(255, 0, 0)]
    graph_name = "canvas"  # QGraphicsScene

    class Warning(OWWidget.Warning):
        incompatible_subset = Msg("Data subset is incompatible with Data")
        no_valid_data = Msg("No valid data")
        no_cont_selection_sql = \
            Msg("Selection of numeric features on SQL is not supported")

    def __init__(self):
        super().__init__()

        self.data = None
        self.discrete_data = None
        self.subset_data = None
        self.subset_indices = None
        self.__pending_selection = self.selection
        self.selection = set()

        self.color_data = None

        self.areas = []

        self.canvas = QGraphicsScene(self)
        self.canvas_view = ViewWithPress(
            self.canvas, handler=self.clear_selection)
        self.mainArea.layout().addWidget(self.canvas_view)
        self.canvas_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.canvas_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.canvas_view.setRenderHint(QPainter.Antialiasing)

        box = gui.vBox(self.controlArea, box=True)
        self.model_1 = DomainModel(
            order=DomainModel.MIXED, valid_types=DomainModel.PRIMITIVE)
        self.model_234 = DomainModel(
            order=DomainModel.MIXED, valid_types=DomainModel.PRIMITIVE,
            placeholder="(None)")
        self.attr_combos = [
            gui.comboBox(
                box, self, value="variable{}".format(i),
                orientation=Qt.Horizontal, contentsLength=12,
                searchable=True,
                callback=self.attr_changed,
                model=self.model_1 if i == 1 else self.model_234)
            for i in range(1, 5)]
        box.layout().addWidget(self.vizrank_button("Find Informative Mosaics"))
        self.vizrankSelectionChanged.connect(self.set_attr)
        self.vizrankRunStateChanged.connect(self.store_vizrank_attr_range)

        box2 = gui.vBox(self.controlArea, box="Interior Coloring")
        self.color_model = DomainModel(
            order=DomainModel.MIXED, valid_types=DomainModel.PRIMITIVE,
            placeholder="(Pearson residuals)")
        self.cb_attr_color = gui.comboBox(
            box2, self, value="variable_color",
            orientation=Qt.Horizontal, contentsLength=12, labelWidth=50,
            searchable=True,
            callback=self.set_color_data, model=self.color_model)
        self.bar_button = gui.checkBox(
            box2, self, 'use_boxes', label='Compare with total',
            callback=self.update_graph)
        gui.rubber(self.controlArea)

    def sizeHint(self):
        return QSize(720, 530)

    def _get_discrete_data(self, data):
        """
        Discretize continuous attributes.
        Return None when there is no data, no rows, or no primitive attributes.
        """
        if (data is None or
                not len(data) or
                not any(attr.is_discrete or attr.is_continuous
                        for attr in chain(data.domain.variables,
                                          data.domain.metas))):
            return None
        elif any(attr.is_continuous for attr in data.domain.variables):
            return Discretize(
                method=EqualFreq(n=4),
                remove_const=False,
                discretize_classes=True,
                discretize_metas=True)(data)
        else:
            return data

    def init_combos(self, data):
        def set_combos(value):
            self.model_1.set_domain(value)
            self.model_234.set_domain(value)
            self.color_model.set_domain(value)

        if data is None:
            set_combos(None)
            self.variable1 = self.variable2 = self.variable3 \
                = self.variable4 = self.variable_color = None
            return
        set_combos(self.data.domain)

        if len(self.model_1) > 0:
            self.variable1 = self.model_1[0]
            self.variable2 = self.model_1[min(1, len(self.model_1) - 1)]
        self.variable3 = self.variable4 = None
        self.variable_color = self.data.domain.class_var  # None is OK, too

    def get_disc_attr_list(self):
        return [self.discrete_data.domain[var.name]
                for var in (self.variable1, self.variable2,
                            self.variable3, self.variable4)
                if var]

    def set_attr(self, attrs):
        self.variable1, self.variable2, self.variable3, self.variable4 = [
            attr and self.data.domain[attr.name] for attr in attrs]
        self.reset_graph()

    def store_vizrank_attr_range(self, state, data):
        if state == vizrank.RunState.Running:
            self.vizrank_attr_range_index = data["attr_range_index"]

    def attr_changed(self):
        self.vizrankAutoSelect.emit(self.get_disc_attr_list())
        self.reset_graph()

    def resizeEvent(self, e):
        OWWidget.resizeEvent(self, e)
        self.update_graph()

    def showEvent(self, ev):
        OWWidget.showEvent(self, ev)
        self.update_graph()

    @Inputs.data
    def set_data(self, data):
        if isinstance(data, SqlTable) and data.approx_len() > LARGE_TABLE:
            data = data.sample_time(DEFAULT_SAMPLE_TIME)

        self.closeContext()
        self.data = data

        if self.data is None:
            self.discrete_data = None
            self.init_combos(None)
            return

        self.init_combos(self.data)
        self.openContext(self.data)

    def init_vizrank(self):
        if self.discrete_data is not None and len(self.discrete_data) > 1 \
                and len(self.discrete_data.domain.attributes) >= 2:
            attr_range_index = self.vizrank_attr_range_index
            if self.variable_color is None:
                if attr_range_index == 0:
                    attr_range_index = 1
                attr_color = None
            else:
                attr_color = self.discrete_data.domain[self.variable_color.name]
            super().init_vizrank(
                self.discrete_data, attr_color=attr_color,
                attr_range_index=attr_range_index)
        else:
            self.disable_vizrank("Not enough data")

    @Inputs.data_subset
    def set_subset_data(self, data):
        self.subset_data = data

    # this is called by widget after setData and setSubsetData are called.
    # this way the graph is updated only once
    def handleNewSignals(self):
        self.Warning.incompatible_subset.clear()
        self.subset_indices = None
        if self.data is not None and self.subset_data:
            transformed = self.subset_data.transform(self.data.domain)
            if np.all(np.isnan(transformed.X)) \
                    and np.all(np.isnan(transformed.Y)):
                self.Warning.incompatible_subset()
            else:
                indices = {e.id for e in transformed}
                self.subset_indices = [ex.id in indices for ex in self.data]
        if self.data is not None and self.__pending_selection is not None:
            self.selection = self.__pending_selection
            self.__pending_selection = None
        else:
            self.selection = set()
        self.set_color_data()
        self.update_graph()
        self.send_selection()

    def clear_selection(self):
        self.selection = set()
        self.update_selection_rects()
        self.send_selection()

    def reset_graph(self):
        self.clear_selection()
        self.update_graph()

    def set_color_data(self):
        if self.data is None:
            return
        self.bar_button.setEnabled(self.variable_color is not None)
        attrs = [v for v in self.model_1 if v and v is not self.variable_color]
        domain = Domain(attrs, self.variable_color, None)
        self.color_data = self.data.from_table(domain, self.data)
        self.discrete_data = self._get_discrete_data(self.color_data)
        self.init_vizrank()
        self.update_graph()

    def update_selection_rects(self):
        pens = (QPen(), QPen(Qt.black, 3, Qt.DotLine))
        for i, (_, _, area) in enumerate(self.areas):
            area.setPen(pens[i in self.selection])

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
            self.Outputs.selected_data.send(None)
            self.Outputs.annotated_data.send(
                create_annotated_table(self.data, []))
            return
        filters = []
        self.Warning.no_cont_selection_sql.clear()
        if self.discrete_data is not self.data:
            if isinstance(self.data, SqlTable):
                self.Warning.no_cont_selection_sql()
        for i in self.selection:
            cols, vals, _ = self.areas[i]
            filters.append(
                filter.Values(
                    filter.FilterDiscrete(col, [val])
                    for col, val in zip(cols, vals)))
        if len(filters) > 1:
            filters = filter.Values(filters, conjunction=False)
        else:
            filters = filters[0]
        selection = filters(self.discrete_data)
        idset = set(selection.ids)
        sel_idx = [i for i, id in enumerate(self.data.ids) if id in idset]
        if self.discrete_data is not self.data:
            selection = self.data[sel_idx]

        self.Outputs.selected_data.send(selection)
        self.Outputs.annotated_data.send(
            create_annotated_table(self.data, sel_idx))

    def send_report(self):
        self.report_plot()

    def update_graph(self):
        spacing = self.SPACING
        bar_width = self.BAR_WIDTH

        def get_counts(attr_vals, values):
            """Calculate rectangles' widths; if all are 0, they are set to 1."""
            if not attr_vals:
                counts = [conditionaldict[val] for val in values]
            else:
                counts = [conditionaldict[attr_vals + "-" + val]
                          for val in values]
            total = sum(counts)
            if total == 0:
                counts = [1] * len(values)
                total = sum(counts)
            return total, counts

        def draw_data(attr_list, x0_x1, y0_y1, side, condition,
                      total_attrs, used_attrs, used_vals, attr_vals=""):
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
            edge = len(attr_list) * spacing
            values = get_variable_values_sorted(attr)
            if side % 2:
                values = values[::-1]  # reverse names if necessary

            if side % 2 == 0:  # we are drawing on the x axis
                # remove the space needed for separating different attr. values
                whole = max(0, (x1 - x0) - edge * (len(values) - 1))
                if whole == 0:
                    edge = (x1 - x0) / float(len(values) - 1)
            else:  # we are drawing on the y axis
                whole = max(0, (y1 - y0) - edge * (len(values) - 1))
                if whole == 0:
                    edge = (y1 - y0) / float(len(values) - 1)

            total, counts = get_counts(attr_vals, values)

            # when visualizing the third attribute and the first attribute has
            # the last value, reverse the order in which the boxes are drawn;
            # otherwise, if the last cell, nearest to the labels of the fourth
            # attribute, is empty, we wouldn't be able to position the labels
            valrange = list(range(len(values)))
            if len(attr_list + used_attrs) == 4 and len(used_attrs) == 2:
                attr1values = get_variable_values_sorted(used_attrs[0])
                if used_vals[0] == attr1values[-1]:
                    valrange = valrange[::-1]

            for i in valrange:
                start = i * edge + whole * float(sum(counts[:i]) / total)
                end = i * edge + whole * float(sum(counts[:i + 1]) / total)
                val = values[i]
                htmlval = to_html(val)
                newattrvals = attr_vals + "-" + val if attr_vals else val

                tooltip = "{}&nbsp;&nbsp;&nbsp;&nbsp;{}: <b>{}</b><br/>".format(
                    condition, attr.name, htmlval)
                attrs = used_attrs + [attr]
                vals = used_vals + [val]
                args = attrs, vals, newattrvals
                if side % 2 == 0:  # if we are moving horizontally
                    if len(attr_list) == 1:
                        add_rect(x0 + start, x0 + end, y0, y1, tooltip, *args)
                    else:
                        draw_data(
                            attr_list[1:], (x0 + start, x0 + end), (y0, y1),
                            side + 1, tooltip, total_attrs, *args)
                else:
                    if len(attr_list) == 1:
                        add_rect(x0, x1, y0 + start, y0 + end, tooltip, *args)
                    else:
                        draw_data(
                            attr_list[1:], (x0, x1), (y0 + start, y0 + end),
                            side + 1, tooltip, total_attrs, *args)
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
                attr1values = get_variable_values_sorted(used_attrs[0])
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

            values = get_variable_values_sorted(attr)
            if side % 2:
                values = values[::-1]

            spaces = spacing * (total_attrs - side) * (len(values) - 1)
            width = x1 - x0 - spaces * (side % 2 == 0)
            height = y1 - y0 - spaces * (side % 2 == 1)

            # calculate position of first attribute
            currpos = 0
            total, counts = get_counts(attr_vals, values)
            aligns = [Qt.AlignTop | Qt.AlignHCenter,
                      Qt.AlignRight | Qt.AlignVCenter,
                      Qt.AlignBottom | Qt.AlignHCenter,
                      Qt.AlignLeft | Qt.AlignVCenter]
            align = aligns[side]
            for i, val in enumerate(values):
                if distributiondict[val] != 0:
                    perc = counts[i] / float(total)
                    rwidth = width * perc
                    xs = [x0 + currpos + rwidth / 2,
                          x0 - self.ATTR_VAL_OFFSET,
                          x0 + currpos + rwidth / 2,
                          x1 + self.ATTR_VAL_OFFSET]
                    ys = [y1 + self.ATTR_VAL_OFFSET,
                          y0 + currpos + height * 0.5 * perc,
                          y0 - self.ATTR_VAL_OFFSET,
                          y0 + currpos + height * 0.5 * perc]

                    CanvasText(
                        self.canvas, val, xs[side], ys[side], align,
                        max_width=rwidth if side == 0 else None)
                    space = height if side % 2 else width
                    currpos += perc * space + spacing * (total_attrs - side)

            xs = [x0 + (x1 - x0) / 2,
                  x0 - max_ylabel_w1 - self.ATTR_VAL_OFFSET,
                  x0 + (x1 - x0) / 2,
                  x1 + max_ylabel_w2 + self.ATTR_VAL_OFFSET]
            ys = [y1 + self.ATTR_VAL_OFFSET + self.ATTR_NAME_OFFSET,
                  y0 + (y1 - y0) / 2,
                  y0 - self.ATTR_VAL_OFFSET - self.ATTR_NAME_OFFSET,
                  y0 + (y1 - y0) / 2]
            CanvasText(
                self.canvas, attr.name, xs[side], ys[side], align, bold=True,
                vertical=side % 2)

        def add_rect(x0, x1, y0, y1, condition,
                     used_attrs, used_vals, attr_vals=""):
            area_index = len(self.areas)
            x1 += (x0 == x1)
            y1 += (y0 == y1)
            # rectangles of width and height 1 are not shown - increase
            y1 += (x1 - x0 + y1 - y0 == 2)
            colors = class_var and [QColor(*col) for col in class_var.colors]

            def select_area(_, ev):
                self.select_area(area_index, ev)

            def rect(x, y, w, h, z, pen_color=None, brush_color=None, **args):
                if pen_color is None:
                    return CanvasRectangle(
                        self.canvas, x, y, w, h, z=z, onclick=select_area,
                        **args)
                if brush_color is None:
                    brush_color = pen_color
                return CanvasRectangle(
                    self.canvas, x, y, w, h, pen_color, brush_color, z=z,
                    onclick=select_area, **args)

            def line(x1, y1, x2, y2):
                r = QGraphicsLineItem(x1, y1, x2, y2, None)
                self.canvas.addItem(r)
                r.setPen(QPen(Qt.white, 2))
                r.setZValue(30)

            outer_rect = rect(x0, y0, x1 - x0, y1 - y0, 30)
            self.areas.append((used_attrs, used_vals, outer_rect))
            if not conditionaldict[attr_vals]:
                return

            if self.variable_color is None:
                s = sum(apriori_dists[0])
                expected = s * reduce(
                    mul,
                    (apriori_dists[i][used_vals[i]] / float(s)
                     for i in range(len(used_vals))))
                actual = conditionaldict[attr_vals]
                pearson = float((actual - expected) / sqrt(expected))
                if pearson == 0:
                    ind = 0
                else:
                    ind = max(0, min(int(log(abs(pearson), 2)), 3))
                color = [self.RED_COLORS, self.BLUE_COLORS][pearson > 0][ind]
                rect(x0, y0, x1 - x0, y1 - y0, -20, color)
                outer_rect.setToolTip(
                    condition + "<hr/>" +
                    "Expected instances: %.1f<br>"
                    "Actual instances: %d<br>"
                    "Standardized (Pearson) residual: %.1f" %
                    (expected, conditionaldict[attr_vals], pearson))
            else:
                cls_values = get_variable_values_sorted(class_var)
                prior = get_distribution(data, class_var.name)
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
                        abs(x1 - x0) > bar_width and abs(y1 - y0) > bar_width:
                    total = 0
                    line(x0 + bar_width, y0, x0 + bar_width, y1)
                    n = sum(prior)
                    for i, (val, color) in enumerate(zip(prior, colors)):
                        if i == len(prior) - 1:
                            h = y1 - y0 - total
                        else:
                            h = (y1 - y0) * val / n
                        rect(x0, y0 + total, bar_width, h, 20, color)
                        total += h

                if conditionalsubsetdict:
                    if conditionalsubsetdict[attr_vals]:
                        if self.subset_indices is not None:
                            line(x1 - bar_width, y0, x1 - bar_width, y1)
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
                                    rect(x1 - bar_width, y0 + total,
                                         bar_width, v, 15, color)
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
                         apr / n_apriori * n_actual, 100.0 * apr / n_apriori)
                        for cls, act, apr in zip(cls_values, actual, apriori))
                else:
                    text = ""
                outer_rect.setToolTip(
                    "{}<hr>Instances: {}<br><br>{}".format(
                        condition, n_actual, text[:-4]))

        def create_legend():
            if self.variable_color is None:
                names = ["<-8", "-8:-4", "-4:-2", "-2:2", "2:4", "4:8", ">8",
                         "Residuals:"]
                colors = self.RED_COLORS[::-1] + self.BLUE_COLORS[1:]
                edges = repeat(Qt.black)
            else:
                names = get_variable_values_sorted(class_var)
                edges = colors = [QColor(*col) for col in class_var.colors]

            items = []
            size = 8
            for name, color, edgecolor in zip(names, colors, edges):
                item = QGraphicsItemGroup()
                item.addToGroup(
                    CanvasRectangle(None, -size / 2, -size / 2, size, size,
                                    edgecolor, color))
                item.addToGroup(
                    CanvasText(None, name, size, 0, Qt.AlignVCenter))
                items.append(item)
            return wrap_legend_items(
                items, hspacing=20, vspacing=16 + size,
                max_width=self.canvas_view.width() - xoff)

        self.canvas.clear()
        self.areas = []

        data = self.discrete_data
        if data is None:
            return
        attr_list = self.get_disc_attr_list()
        class_var = data.domain.class_var
        # TODO: check this
        # data = Preprocessor_dropMissing(data)

        unique = [v.name for v in set(attr_list + [class_var]) if v]
        if len(data[:, unique]) == 0:
            self.Warning.no_valid_data()
            return
        else:
            self.Warning.no_valid_data.clear()

        attrs = [attr for attr in attr_list if not attr.values]
        if attrs:
            CanvasText(self.canvas,
                       "Feature {} has no values".format(attrs[0]),
                       (self.canvas_view.width() - 120) / 2,
                       self.canvas_view.height() / 2)
            return
        if self.variable_color is None:
            apriori_dists = [get_distribution(data, attr) for attr
                             in attr_list]
        else:
            apriori_dists = []

        def get_max_label_width(attr):
            values = get_variable_values_sorted(attr)
            maxw = 0
            for val in values:
                t = CanvasText(self.canvas, val, 0, 0, bold=0, show=False)
                maxw = max(int(t.boundingRect().width()), maxw)
            return maxw

        xoff = 20

        # get the maximum width of rectangle
        width = 20
        max_ylabel_w1 = max_ylabel_w2 = 0
        if len(attr_list) > 1:
            text = CanvasText(self.canvas, attr_list[1].name, bold=1, show=0)
            max_ylabel_w1 = min(get_max_label_width(attr_list[1]), 150)
            width = 5 + text.boundingRect().height() + \
                self.ATTR_VAL_OFFSET + max_ylabel_w1
            xoff = width
            if len(attr_list) == 4:
                text = CanvasText(self.canvas, attr_list[3].name, bold=1, show=0)
                max_ylabel_w2 = min(get_max_label_width(attr_list[3]), 150)
                width += text.boundingRect().height() + \
                    self.ATTR_VAL_OFFSET + max_ylabel_w2 - 10

        legend = create_legend()

        # get the maximum height of rectangle
        yoff = 45
        legendoff = yoff + self.ATTR_NAME_OFFSET + self.ATTR_VAL_OFFSET + 35
        square_size = min(self.canvas_view.width() - width - 20,
                          self.canvas_view.height() - legendoff
                          - legend.boundingRect().height())

        if square_size < 0:
            return  # canvas is too small to draw rectangles
        self.canvas_view.setSceneRect(
            0, 0, self.canvas_view.width(), self.canvas_view.height())

        drawn_sides = set()
        draw_positions = {}

        conditionaldict, distributiondict = \
            get_conditional_distribution(data, attr_list)
        conditionalsubsetdict = None
        if self.subset_indices:
            conditionalsubsetdict, _ = get_conditional_distribution(
                self.discrete_data[self.subset_indices], attr_list)

        # draw rectangles
        draw_data(
            attr_list, (xoff, xoff + square_size), (yoff, yoff + square_size),
            0, "", len(attr_list), [], [])

        self.canvas.addItem(legend)
        legend.setPos(
            xoff - legend.boundingRect().x()
            + max(0, (square_size - legend.boundingRect().width()) / 2),
            legendoff + square_size)
        self.update_selection_rects()

    @classmethod
    def migrate_context(cls, context, version):
        if version < 2:
            settings.migrate_str_to_variable(context, none_placeholder="(None)")


def get_conditional_distribution(data, attrs):
    cond_dist = defaultdict(int)
    dist = defaultdict(int)
    cond_dist[""] = dist[""] = len(data)
    all_attrs = attrs[:]
    if data.domain.class_var is not None:
        all_attrs.append(data.domain.class_var)

    for i in range(1, len(all_attrs) + 1):
        attr = all_attrs[:i]
        if isinstance(data, SqlTable):
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


if __name__ == "__main__":  # pragma: no cover
    dataset = Table("zoo")
    WidgetPreview(OWMosaicDisplay).run(dataset, set_subset_data=dataset[::10])
