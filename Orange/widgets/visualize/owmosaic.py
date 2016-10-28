from collections import defaultdict
from functools import reduce
from itertools import product, chain
from math import sqrt, log
from operator import mul, attrgetter

from scipy.stats import distributions
from AnyQt.QtCore import Qt, QSize, pyqtSignal as Signal
from AnyQt.QtGui import QColor, QPainter, QPen, QStandardItem
from AnyQt.QtWidgets import QGraphicsScene, QGraphicsLineItem

from Orange.data import Table, filter, Variable
from Orange.data.sql.table import SqlTable, LARGE_TABLE, DEFAULT_SAMPLE_TIME
from Orange.preprocess import Discretize
from Orange.preprocess.discretize import EqualFreq
from Orange.preprocess.score import ReliefF
from Orange.statistics.distribution import get_distribution
from Orange.widgets import gui, widget
from Orange.widgets.gui import OWComponent
from Orange.widgets.settings import (
    Setting, DomainContextHandler, ContextSetting, SettingProvider)
from Orange.widgets.utils import to_html, get_variable_values_sorted
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.visualize.utils import (
    CanvasText, CanvasRectangle, ViewWithPress, VizRankDialog)
from Orange.widgets.widget import OWWidget, Default, Msg


class MosaicVizRank(VizRankDialog, OWComponent):
    captionTitle = "Mosaic Ranking"
    K = 10  # for ReliefF
    max_attrs = Setting(3)

    pairSelected = Signal(Variable, Variable, Variable, Variable)
    _AttrRole = next(gui.OrangeUserRole)

    def __init__(self, master):
        VizRankDialog.__init__(self, master)
        OWComponent.__init__(self, master)

        box = gui.vBox(self)
        self.max_attr_spin = gui.spin(
            box, self, "max_attrs", 2, 4,
            label="Limit the number of attributes to: ",
            controlWidth=50, alignment=Qt.AlignRight,
            callback=self.max_attr_changed)
        self.layout().addWidget(self.button)
        self.resize(320, 512)
        self.attrs = []
        self.marginal = {}
        self.last_run_max_attr = None

    def sizeHint(self):
        """Assuming two columns in the table, return `QSize(320, 512)` as
        a reasonable default size."""
        return QSize(320, 512)

    def run(self):
        if self.max_attrs != self.last_run_max_attr:
            self.saved_state = None
            self.saved_progress = 0
        if self.saved_state is None:
            self.scores = []
            self.rank_model.clear()
        self.last_run_max_attr = self.max_attrs
        self.max_attr_spin.setDisabled(True)
        try:
            super().run()
        finally:
            self.max_attr_spin.setDisabled(False)

    def max_attr_changed(self):
        self.button.setText("Start")
        self.button.setEnabled(self.check_preconditions())

    def check_preconditions(self):
        self.Information.add_message(
            "class_required",
            "Data with a discrete class variable is required.")
        self.Information.add_message(
            "no_attributes", "Mosaic requires at least one variable.")
        self.Information.class_required.clear()
        self.Information.no_attributes.clear()
        if not super().check_preconditions():
            return False
        domain = self.master.discrete_data.domain
        if not domain.class_var:
            self.Information.class_required()
            return False
        if not domain.attributes:
            self.Information.no_attributes()
            return False
        return True

    def score_heuristic(self):
        data = self.master.discrete_data
        weights = ReliefF(n_iterations=100, k_nearest=self.K)(data)
        attrs = sorted(zip(weights, data.domain.attributes),
                       key=lambda x: (-x[0], x[1].name))
        return [a for _, a in attrs]

    def state_count(self):
        n_attrs = len(self.master.discrete_data.domain.attributes)
        total = 0
        count = 1
        for i in range(min(n_attrs, self.max_attrs)):
            count *= (n_attrs - i)
            count //= (i + 1)  # (n_attrs - i) may not be divisible by (i + 1)
            total += count
        return total

    def iterate_states(self, state):
        # If we put initialization of `self.attrs` to `initialize`,
        # `score_heuristic` would be run on every call to `set_data`.
        data = self.master.discrete_data
        if state is None:  # on the first call, compute order
            self.attrs = self.score_heuristic()
            self.marginal = get_distribution(data, data.domain.class_var)
            self.marginal.normalize()
            state = [0]
        n_attrs = len(data.domain.attributes)
        while True:
            yield state
            for up in range(len(state)):
                state[up] += 1
                if up + 1 == len(state) or state[up] < state[up + 1]:
                    break
                state[up] = up
            if state[-1] == len(self.attrs):
                if len(state) < min(self.max_attrs, n_attrs):
                    state = list(range(len(state) + 1))
                else:
                    break

    def compute_score(self, state):
        data = self.master.discrete_data
        attrlist = [self.attrs[i] for i in state]
        cond_dist = get_conditional_distribution(data, attrlist)[0]
        ss = 0
        class_values = data.domain.class_var.values
        for indices in product(*(range(len(a.values)) for a in attrlist)):
            attr_vals = "-".join(attrlist[k].values[ind]
                                 for k, ind in enumerate(indices))
            total = cond_dist[attr_vals]
            for i, class_val in enumerate(class_values):
                expected = total * self.marginal[i]
                if expected > 1e-6:
                    observed = cond_dist[attr_vals + '-' + class_val]
                    ss += (expected - observed) ** 2 / expected
        dof = reduce(mul,
                     (len(attr.values) for attr in attrlist),
                     len(class_values) - 1)
        return distributions.chi2.sf(ss, dof)

    def on_selection_changed(self, selected, deselected):
        attrs = selected.indexes()[0].data(self._AttrRole)
        self.selectionChanged.emit(attrs + (None, ) * (4 - len(attrs)))

    def row_for_state(self, score, state):
        attrs = tuple(sorted((self.attrs[x] for x in state),
                             key=attrgetter("name")))
        item = QStandardItem(", ".join(a.name for a in attrs))
        item.setData(attrs, self._AttrRole)
        return [item]


class OWMosaicDisplay(OWWidget):
    name = "Mosaic Display"
    description = "Display data in a mosaic plot."
    icon = "icons/MosaicDisplay.svg"
    priority = 220

    inputs = [("Data", Table, "set_data", Default),
              ("Data Subset", Table, "set_subset_data")]
    outputs = [("Selected Data", Table, widget.Default),
               (ANNOTATED_DATA_SIGNAL_NAME, Table)]

    settingsHandler = DomainContextHandler()
    use_boxes = Setting(True)
    variable1 = ContextSetting("", exclude_metas=False)
    variable2 = ContextSetting("", exclude_metas=False)
    variable3 = ContextSetting("", exclude_metas=False)
    variable4 = ContextSetting("", exclude_metas=False)
    selection = ContextSetting(set())
    # interior_coloring is context setting to properly reset it
    # if the widget switches to regression and back (set setData)
    interior_coloring = ContextSetting(1)

    PEARSON, CLASS_DISTRIBUTION = 0, 1
    interior_coloring_opts = ["Pearson residuals",
                              "Class distribution"]
    BAR_WIDTH = 5
    SPACING = 4
    ATTR_NAME_OFFSET = 20
    ATTR_VAL_OFFSET = 3
    BLUE_COLORS = [QColor(255, 255, 255), QColor(210, 210, 255),
                   QColor(110, 110, 255), QColor(0, 0, 255)]
    RED_COLORS = [QColor(255, 255, 255), QColor(255, 200, 200),
                  QColor(255, 100, 100), QColor(255, 0, 0)]

    vizrank = SettingProvider(MosaicVizRank)

    graph_name = "canvas"

    class Warning(OWWidget.Warning):
        incompatible_subset = Msg("Data subset is incompatible with Data")
        no_valid_data = Msg("No valid data")
        no_cont_selection_sql = \
            Msg("Selection of continuous variables on SQL is not supported")

    def __init__(self):
        super().__init__()

        self.data = None
        self.discrete_data = None
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
                orientation=Qt.Horizontal, contentsLength=12,
                callback=self.reset_graph,
                sendSelectedValue=True, valueType=str)
            for i in range(1, 5)]
        self.vizrank, self.vizrank_button = MosaicVizRank.add_vizrank(
            box, self, "Find Informative Mosaics", self.set_attr)

        self.rb_colors = gui.radioButtonsInBox(
            self.controlArea, self, "interior_coloring",
            self.interior_coloring_opts, box="Interior Coloring",
            callback=self.update_graph)
        self.bar_button = gui.checkBox(
            gui.indentedBox(self.rb_colors),
            self, 'use_boxes', label='Compare with total',
            callback=self._compare_with_total)
        gui.rubber(self.controlArea)

    def sizeHint(self):
        return QSize(530, 720)

    def _compare_with_total(self):
        if self.data and self.data.domain.has_discrete_class:
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
            if attr.is_discrete or attr.is_continuous:
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

    def set_attr(self, *attrs):
        self.variable1, self.variable2, self.variable3, self.variable4 = \
            [a and a.name for a in attrs]
        self.reset_graph()

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
        if not self.data:
            self.discrete_data = None
            return
        if any(attr.is_continuous for attr in data.domain):
            self.discrete_data = Discretize(method=EqualFreq(n=4))(data)
        else:
            self.discrete_data = self.data

        if self.data.domain.class_var is None:
            self.rb_colors.setDisabled(True)
            disc_class = False
        else:
            self.rb_colors.setDisabled(False)
            disc_class = self.data.domain.has_discrete_class
            self.rb_colors.group.button(2).setDisabled(not disc_class)
            self.bar_button.setDisabled(not disc_class)
        self.interior_coloring = bool(disc_class)
        self.vizrank.initialize()
        self.vizrank_button.setEnabled(
            self.data is not None and self.data.domain.class_var is not None
            and len(self.data.domain.attributes) >= 1 and len(self.data) > 1)

        self.openContext(self.data)

        # if we first received subset we now call setSubsetData to process it
        if self.unprocessed_subset_data:
            self.set_subset_data(self.unprocessed_subset_data)
            self.unprocessed_subset_data = None

    def set_subset_data(self, data):
        self.Warning.incompatible_subset.clear()
        if self.data is None:
            self.unprocessed_subset_data = data
            return
        try:
            self.subset_data = data.from_table(self.data.domain, data)
        except:
            self.subset_data = None
            self.Warning.incompatible_subset(shown=data is not None)

    # this is called by widget after setData and setSubsetData are called.
    # this way the graph is updated only once
    def handleNewSignals(self):
        self.reset_graph()

    def clear_selection(self):
        self.selection = set()
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
            self.send(ANNOTATED_DATA_SIGNAL_NAME,
                      create_annotated_table(self.data, []))
            return
        filters = []
        self.Warning.no_cont_selection_sql.clear()
        if self.discrete_data is not self.data:
            if isinstance(self.data, SqlTable):
                self.Warning.no_cont_selection_sql()
        for i in self.selection:
            cols, vals, area = self.areas[i]
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
        self.send("Selected Data", selection)
        self.send(ANNOTATED_DATA_SIGNAL_NAME,
                  create_annotated_table(self.data, sel_idx))

    def send_report(self):
        self.report_plot(self.canvas)

    def update_graph(self):
        spacing = self.SPACING
        bar_width = self.BAR_WIDTH

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
            edge = len(attr_list) * spacing
            values = get_variable_values_sorted(data.domain[attr])
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
                    data.domain[used_attrs[0]])
                if used_vals[0] == attr1values[-1]:
                    valrange = valrange[::-1]

            for i in valrange:
                start = i * edge + whole * float(sum(counts[:i]) / total)
                end = i * edge + whole * float(sum(counts[:i + 1]) / total)
                val = values[i]
                htmlval = to_html(val)
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
                    get_variable_values_sorted(data.domain[used_attrs[0]])
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

            values = get_variable_values_sorted(data.domain[attr])
            if side % 2:
                values = values[::-1]

            spaces = spacing * (total_attrs - side) * (len(values) - 1)
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
            for i, val in enumerate(values):
                perc = counts[i] / float(total)
                if distributiondict[val] != 0:
                    if side == 0:
                        CanvasText(self.canvas, str(val),
                                   x0 + currpos + width * 0.5 * perc,
                                   y1 + self.ATTR_VAL_OFFSET, align)
                    elif side == 1:
                        CanvasText(self.canvas, str(val),
                                   x0 - self.ATTR_VAL_OFFSET,
                                   y0 + currpos + height * 0.5 * perc, align)
                    elif side == 2:
                        CanvasText(self.canvas, str(val),
                                   x0 + currpos + width * perc * 0.5,
                                   y0 - self.ATTR_VAL_OFFSET, align)
                    else:
                        CanvasText(self.canvas, str(val),
                                   x1 + self.ATTR_VAL_OFFSET,
                                   y0 + currpos + height * 0.5 * perc, align)

                if side % 2 == 0:
                    currpos += perc * width + spacing * (total_attrs - side)
                else:
                    currpos += perc * height + spacing * (total_attrs - side)

            if side == 0:
                CanvasText(
                    self.canvas, attr,
                    x0 + (x1 - x0) / 2,
                    y1 + self.ATTR_VAL_OFFSET + self.ATTR_NAME_OFFSET,
                    align, bold=1)
            elif side == 1:
                CanvasText(
                    self.canvas, attr,
                    x0 - max_ylabel_w1 - self.ATTR_VAL_OFFSET,
                    y0 + (y1 - y0) / 2,
                    align, bold=1, vertical=True)
            elif side == 2:
                CanvasText(
                    self.canvas, attr,
                    x0 + (x1 - x0) / 2,
                    y0 - self.ATTR_VAL_OFFSET - self.ATTR_NAME_OFFSET,
                    align, bold=1)
            else:
                CanvasText(
                    self.canvas, attr,
                    x1 + max_ylabel_w2 + self.ATTR_VAL_OFFSET,
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

            if class_var and class_var.is_discrete:
                colors = [QColor(*col) for col in class_var.colors]
            else:
                colors = None

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

            if self.interior_coloring == self.PEARSON:
                s = sum(apriori_dists[0])
                expected = s * reduce(
                    mul,
                    (apriori_dists[i][used_vals[i]] / float(s)
                     for i in range(len(used_vals))))
                actual = conditionaldict[attr_vals]
                pearson = (actual - expected) / sqrt(expected)
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
                        abs(x1 - x0) > bar_width and \
                        abs(y1 - y0) > bar_width:
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
                        counts = [conditionalsubsetdict[attr_vals + "-" + val]
                                  for val in cls_values]
                        if sum(counts) == 1:
                            rect(x0 - 2, y0 - 2, x1 - x0 + 5, y1 - y0 + 5, -550,
                                 colors[counts.index(1)], Qt.white,
                                 penWidth=2, penStyle=Qt.DashLine)
                        if self.subset_data is not None:
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
                         apr / n_apriori * n_actual, 100.0 * apr / n_apriori
                         )
                        for cls, act, apr in zip(cls_values, actual, apriori))
                else:
                    text = ""
                outer_rect.setToolTip(
                    "{}<hr>Instances: {}<br><br>{}".format(
                    condition, n_actual, text[:-4]))

        def draw_legend(x0_x1, y0_y1):
            x0, x1 = x0_x1
            y0, y1 = y0_y1
            if self.interior_coloring == self.PEARSON:
                names = ["<-8", "-8:-4", "-4:-2", "-2:2", "2:4", "4:8", ">8",
                         "Residuals:"]
                colors = self.RED_COLORS[::-1] + self.BLUE_COLORS[1:]
            else:
                names = get_variable_values_sorted(class_var) + \
                        [class_var.name + ":"]
                colors = [QColor(*col) for col in class_var.colors]

            names = [CanvasText(self.canvas, name, alignment=Qt.AlignVCenter)
                     for name in names]
            totalwidth = sum(text.boundingRect().width() for text in names)

            # compute the x position of the center of the legend
            y = y1 + self.ATTR_NAME_OFFSET + self.ATTR_VAL_OFFSET + 35
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

                CanvasRectangle(self.canvas, startx + xoffset, y - size / 2,
                                size, size, edgecolor, colors[i])
                names[i].setPos(startx + xoffset + 10, y)
                xoffset += distance + names[i].boundingRect().width()

        self.canvas.clear()
        self.areas = []

        data = self.discrete_data
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
            self.Warning.no_valid_data()
            return
        else:
            self.Warning.no_valid_data.clear()

        if self.interior_coloring == self.PEARSON:
            apriori_dists = [get_distribution(data, attr) for attr in attr_list]
        else:
            apriori_dists = []

        def get_max_label_width(attr):
            values = get_variable_values_sorted(data.domain[attr])
            maxw = 0
            for val in values:
                t = CanvasText(self.canvas, val, 0, 0, bold=0, show=False)
                maxw = max(int(t.boundingRect().width()), maxw)
            return maxw

        # get the maximum width of rectangle
        xoff = 20
        width = 20
        if len(attr_list) > 1:
            text = CanvasText(self.canvas, attr_list[1], bold=1, show=0)
            max_ylabel_w1 = min(get_max_label_width(attr_list[1]), 150)
            width = 5 + text.boundingRect().height() + \
                self.ATTR_VAL_OFFSET + max_ylabel_w1
            xoff = width
            if len(attr_list) == 4:
                text = CanvasText(self.canvas, attr_list[3], bold=1, show=0)
                max_ylabel_w2 = min(get_max_label_width(attr_list[3]), 150)
                width += text.boundingRect().height() + \
                    self.ATTR_VAL_OFFSET + max_ylabel_w2 - 10

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
        draw_legend((xoff, xoff + square_size), (yoff, yoff + square_size))
        self.update_selection_rects()


def get_conditional_distribution(data, attrs):
    cond_dist = defaultdict(int)
    dist = defaultdict(int)
    cond_dist[""] = dist[""] = len(data)
    all_attrs = [data.domain[a] for a in attrs]
    if data.domain.has_discrete_class:
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


# test widget appearance
if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication
    a = QApplication(sys.argv)
    ow = OWMosaicDisplay()
    ow.show()
    data = Table("zoo.tab")
    ow.set_data(data)
    ow.handleNewSignals()
    a.exec_()
