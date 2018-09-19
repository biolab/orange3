from itertools import chain
from xml.sax.saxutils import escape

import numpy as np
from scipy.stats import linregress
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score

from AnyQt.QtCore import Qt, QTimer, QPointF
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import QApplication

import pyqtgraph as pg

from Orange.data import Table, Domain, DiscreteVariable, Variable
from Orange.data.sql.table import SqlTable, AUTO_DL_LIMIT
from Orange.preprocess.score import ReliefF, RReliefF

from Orange.widgets import gui, report
from Orange.widgets.io import MatplotlibFormat, MatplotlibPDFFormat
from Orange.widgets.settings import (
    DomainContextHandler, Setting, ContextSetting, SettingProvider
)
from Orange.widgets.utils import get_variable_values_sorted
from Orange.widgets.utils.annotated_data import (
    create_annotated_table, create_groups_table, ANNOTATED_DATA_SIGNAL_NAME
)
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.visualize.owscatterplotgraph import (
    OWScatterPlotBase, OWProjectionWidget
)
from Orange.widgets.visualize.utils import VizRankDialogAttrPair
from Orange.widgets.widget import AttributeList, Msg, Input, Output


class ScatterPlotVizRank(VizRankDialogAttrPair):
    captionTitle = "Score Plots"
    minK = 10
    attr_color = None

    def __init__(self, master):
        super().__init__(master)
        self.attr_color = self.master.attr_color

    def initialize(self):
        self.attr_color = self.master.attr_color
        super().initialize()

    def check_preconditions(self):
        self.Information.add_message(
            "color_required", "Color variable must be selected")
        self.Information.color_required.clear()
        if not super().check_preconditions():
            return False
        if not self.attr_color:
            self.Information.color_required()
            return False
        return True

    def iterate_states(self, initial_state):
        # If we put initialization of `self.attrs` to `initialize`,
        # `score_heuristic` would be run on every call to `set_data`.
        if initial_state is None:  # on the first call, compute order
            self.attrs = self.score_heuristic()
        yield from super().iterate_states(initial_state)

    def compute_score(self, state):
        attrs = [self.attrs[i] for i in state]
        data = self.master.data
        data = data.transform(Domain(attrs, self.attr_color))
        data = data[~np.isnan(data.X).any(axis=1) & ~np.isnan(data.Y).T]
        if len(data) < self.minK:
            return
        n_neighbors = min(self.minK, len(data) - 1)
        knn = NearestNeighbors(n_neighbors=n_neighbors).fit(data.X)
        ind = knn.kneighbors(return_distance=False)
        if data.domain.has_discrete_class:
            return -np.sum(data.Y[ind] == data.Y.reshape(-1, 1)) / \
                   n_neighbors / len(data.Y)
        else:
            return -r2_score(data.Y, np.mean(data.Y[ind], axis=1)) * \
                   (len(data.Y) / len(self.master.data))

    def bar_length(self, score):
        return max(0, -score)

    def score_heuristic(self):
        assert self.attr_color is not None
        master_domain = self.master.data.domain
        vars = [v for v in chain(master_domain.variables, master_domain.metas)
                if v is not self.attr_color]
        domain = Domain(attributes=vars, class_vars=self.attr_color)
        data = self.master.data.transform(domain)
        relief = ReliefF if isinstance(domain.class_var, DiscreteVariable) \
            else RReliefF
        weights = relief(n_iterations=100, k_nearest=self.minK)(data)
        attrs = sorted(zip(weights, domain.attributes),
                       key=lambda x: (-x[0], x[1].name))
        return [a for _, a in attrs]


class OWScatterPlotGraph(OWScatterPlotBase):
    show_reg_line = Setting(False)
    jitter_size = Setting(10)
    jitter_continuous = Setting(False)

    def __init__(self, scatter_widget, parent):
        super().__init__(scatter_widget, parent)
        self.reg_line_item = None

    def clear(self):
        super().clear()
        self.reg_line_item = None

    def set_axis_labels(self, axis, labels):
        axis = self.plot_widget.getAxis(axis)
        if labels:
            axis.setTicks([list(enumerate(labels))])
        else:
            axis.setTicks(None)

    def set_axis_title(self, axis, title):
        self.plot_widget.setLabel(axis=axis, text=title)

    def update_coordinates(self):
        super().update_coordinates()
        self.update_regression_line()

    def jitter_coordinates(self, x, y):
        def get_span(attr):
            if attr.is_discrete:
                # Assuming the maximal jitter size is 10, a span of 4 will
                # jitter by 4 * 10 / 100 = 0.4, so there will be no overlap
                return 4
            elif self.jitter_continuous:
                return None  # Let _jitter_data determine the span
            else:
                return 0  # No jittering
        span_x = get_span(self.master.attr_x)
        span_y = get_span(self.master.attr_y)
        if self.jitter_size == 0 or (span_x == 0 and span_y == 0):
            return x, y
        return self._jitter_data(x, y, span_x, span_y)

    def update_regression_line(self):
        if self.reg_line_item is not None:
            self.plot_widget.removeItem(self.reg_line_item)
            self.reg_line_item = None
        if not (self.show_reg_line
                and self.master.can_draw_regresssion_line()):
            return
        x, y = self.master.get_coordinates_data()
        if x is None:
            return
        min_x, max_x = np.min(x), np.max(x)
        slope, intercept, rvalue, _, _ = linregress(x, y)
        start_y = min_x * slope + intercept
        end_y = max_x * slope + intercept
        angle = np.degrees(np.arctan((end_y - start_y) / (max_x - min_x)))
        rotate = ((angle + 45) % 180) - 45 > 90
        color = QColor("#505050")
        l_opts = dict(color=color, position=abs(int(rotate) - 0.85),
                      rotateAxis=(1, 0), movable=True)
        self.reg_line_item = pg.InfiniteLine(
            pos=QPointF(min_x, start_y), angle=angle,
            pen=pg.mkPen(color=color, width=1),
            label="r = {:.2f}".format(rvalue), labelOpts=l_opts
        )
        if rotate:
            self.reg_line_item.label.angle = 180
            self.reg_line_item.label.updateTransform()
        self.plot_widget.addItem(self.reg_line_item)


class OWScatterPlot(OWProjectionWidget):
    """Scatterplot visualization with explorative analysis and intelligent
    data visualization enhancements."""

    name = 'Scatter Plot'
    description = "Interactive scatter plot visualization with " \
                  "intelligent data visualization enhancements."
    icon = "icons/ScatterPlot.svg"
    priority = 140
    keywords = []

    class Inputs:
        data = Input("Data", Table, default=True)
        data_subset = Input("Data Subset", Table)
        features = Input("Features", AttributeList)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)
        features = Output("Features", AttributeList, dynamic=False)

    settings_version = 3
    settingsHandler = DomainContextHandler()

    auto_send_selection = Setting(True)
    auto_sample = Setting(True)

    attr_x = ContextSetting(None)
    attr_y = ContextSetting(None)
    tooltip_shows_all = Setting(True)

    #: Serialized selection state to be restored
    selection_group = Setting(None, schema_only=True)

    graph = SettingProvider(OWScatterPlotGraph)
    graph_name = "graph.plot_widget.plotItem"

    class Warning(OWProjectionWidget.Warning):
        missing_coords = Msg(
            "Plot cannot be displayed because '{}' or '{}' "
            "is missing for all data points")

    class Information(OWProjectionWidget.Information):
        sampled_sql = Msg("Large SQL table; showing a sample.")
        missing_coords = Msg(
            "Points with missing '{}' or '{}' are not displayed")

    def __init__(self):
        super().__init__()

        box = gui.vBox(self.mainArea, True, margin=0)
        self.graph = OWScatterPlotGraph(self, box)
        box.layout().addWidget(self.graph.plot_widget)

        self.subset_data = None  # Orange.data.Table
        self.subset_indices = None
        self.sql_data = None  # Orange.data.sql.table.SqlTable
        self.attribute_selection_list = None  # list of Orange.data.Variable
        self.__timer = QTimer(self, interval=1200)
        self.__timer.timeout.connect(self.add_data)
        #: Remember the saved state to restore
        self.__pending_selection_restore = self.selection_group
        self.selection_group = None

        common_options = dict(
            labelWidth=50, orientation=Qt.Horizontal, sendSelectedValue=True,
            valueType=str, contentsLength=14
        )
        box = gui.vBox(self.controlArea, True)
        dmod = DomainModel
        self.xy_model = DomainModel(dmod.MIXED, valid_types=dmod.PRIMITIVE)
        self.cb_attr_x = gui.comboBox(
            box, self, "attr_x", label="Axis x:", callback=self.attr_changed,
            model=self.xy_model, **common_options)
        self.cb_attr_y = gui.comboBox(
            box, self, "attr_y", label="Axis y:", callback=self.attr_changed,
            model=self.xy_model, **common_options)

        vizrank_box = gui.hBox(box)
        #gui.separator(vizrank_box, width=common_options["labelWidth"])
        self.vizrank, self.vizrank_button = ScatterPlotVizRank.add_vizrank(
            vizrank_box, self, "Find Informative Projections", self.set_attr)

        g = self.graph.gui

        self.sampling = gui.auto_commit(
            self.controlArea, self, "auto_sample", "Sample", box="Sampling",
            callback=self.switch_sampling, commit=lambda: self.add_data(1))
        self.sampling.setVisible(False)

        g.point_properties_box(self.controlArea)

        box = g.effects_box(self.controlArea)
        g.add_widget(g.JitterNumericValues, box)

        box_plot_prop = g.plot_properties_box(self.controlArea)
        g.add_widgets([
            g.ShowGridLines,
            g.ToolTipShowsAll,
            g.RegressionLine], box_plot_prop)

        self.controlArea.layout().addStretch(100)
        self.graph.box_zoom_select(self.controlArea)
        gui.auto_commit(self.controlArea, self, "auto_send_selection",
                        "Send Selection", "Send Automatically")

        # manually register Matplotlib file writers
        self.graph_writers = self.graph_writers.copy()
        for w in [MatplotlibFormat, MatplotlibPDFFormat]:
            for ext in w.EXTENSIONS:
                self.graph_writers[ext] = w

    def _vizrank_color_change(self):
        self.vizrank.initialize()
        is_enabled = self.data is not None and not self.data.is_sparse() and \
            len(self.xy_model) > 2 and len(self.data[self.valid_data]) > 1 \
            and np.all(np.nan_to_num(np.nanstd(self.data.X, 0)) != 0)
        self.vizrank_button.setEnabled(
            is_enabled and self.attr_color is not None and
            not np.isnan(self.data.get_column_view(
                self.attr_color)[0].astype(float)).all())
        text = "Color variable has to be selected." \
            if is_enabled and self.attr_color is None else ""
        self.vizrank_button.setToolTip(text)

    @Inputs.data
    def set_data(self, data):
        self.clear_messages()
        self.Information.sampled_sql.clear()
        self.__timer.stop()
        self.sampling.setVisible(False)
        self.sql_data = None
        if isinstance(data, SqlTable):
            if data.approx_len() < 4000:
                data = Table(data)
            else:
                self.Information.sampled_sql()
                self.sql_data = data
                data_sample = data.sample_time(0.8, no_cache=True)
                data_sample.download_data(2000, partial=True)
                data = Table(data_sample)
                self.sampling.setVisible(True)
                if self.auto_sample:
                    self.__timer.start()

        if data is not None and (len(data) == 0 or len(data.domain) == 0):
            data = None
        if self.data and data and self.data.checksum() == data.checksum():
            return

        self.closeContext()
        same_domain = (self.data and data and
                       data.domain.checksum() == self.data.domain.checksum())
        self.data = data

        if not same_domain:
            self.init_attr_values()
        self.openContext(self.data)

        def findvar(name, iterable):
            """Find a Orange.data.Variable in `iterable` by name"""
            for el in iterable:
                if isinstance(el, Variable) and el.name == name:
                    return el
            return None

        # handle restored settings from  < 3.3.9 when attr_* were stored
        # by name
        if isinstance(self.attr_x, str):
            self.attr_x = findvar(self.attr_x, self.xy_model)
        if isinstance(self.attr_y, str):
            self.attr_y = findvar(self.attr_y, self.xy_model)
        if isinstance(self.attr_label, str):
            self.attr_label = findvar(
                self.attr_label, self.graph.gui.label_model)
        if isinstance(self.attr_color, str):
            self.attr_color = findvar(
                self.attr_color, self.graph.gui.color_model)
        if isinstance(self.attr_shape, str):
            self.attr_shape = findvar(
                self.attr_shape, self.graph.gui.shape_model)
        if isinstance(self.attr_size, str):
            self.attr_size = findvar(
                self.attr_size, self.graph.gui.size_model)

    def get_coordinates_data(self):
        self.Warning.missing_coords.clear()
        self.Information.missing_coords.clear()

        x_data = self.get_column(self.attr_x, filter_valid=False)
        y_data = self.get_column(self.attr_y, filter_valid=False)
        if x_data is None or y_data is None:
            self.valid_data = None
            return None, None
        self.valid_data = np.isfinite(x_data) & np.isfinite(y_data)
        if not np.all(self.valid_data):
            msg = self.Information if np.any(self.valid_data) else self.Warning
            msg.missing_coords(self.attr_x.name, self.attr_y.name)
        return x_data[self.valid_data], y_data[self.valid_data]

    def get_subset_mask(self):
        if self.subset_indices:
            return np.array([ex.id in self.subset_indices
                             for ex in self.data[self.valid_data]])

    # Tooltip
    def _point_tooltip(self, point_id, skip_attrs=()):
        point_data = self.data[point_id]
        xy_attrs = (self.attr_x, self.attr_y)
        text = "<br/>".join(
            escape('{} = {}'.format(var.name, point_data[var]))
            for var in xy_attrs)
        if self.tooltip_shows_all:
            others =  super()._point_tooltip(point_id, skip_attrs=xy_attrs)
            if others:
                text = "<b>{}</b><br/><br/>{}".format(text, others)
        return text

    def can_draw_regresssion_line(self):
        return self.data is not None and\
               self.data.domain is not None and \
               self.attr_x.is_continuous and \
               self.attr_y.is_continuous

    def add_data(self, time=0.4):
        if self.data and len(self.data) > 2000:
            return self.__timer.stop()
        data_sample = self.sql_data.sample_time(time, no_cache=True)
        if data_sample:
            data_sample.download_data(2000, partial=True)
            data = Table(data_sample)
            self.data = Table.concatenate((self.data, data), axis=0)
            self.handleNewSignals()

    def init_attr_values(self):
        super().init_attr_values()
        data = self.data
        domain = data.domain if data and len(data) else None
        self.xy_model.set_domain(domain)
        self.attr_x = self.xy_model[0] if self.xy_model else None
        self.attr_y = self.xy_model[1] if len(self.xy_model) >= 2 \
            else self.attr_x

    def switch_sampling(self):
        self.__timer.stop()
        if self.auto_sample and self.sql_data:
            self.add_data()
            self.__timer.start()

    @Inputs.data_subset
    def set_subset_data(self, subset_data):
        self.warning()
        if isinstance(subset_data, SqlTable):
            if subset_data.approx_len() < AUTO_DL_LIMIT:
                subset_data = Table(subset_data)
            else:
                self.warning("Data subset does not support large Sql tables")
                subset_data = None
        self.subset_data = subset_data
        self.subset_indices = {e.id for e in subset_data} \
            if subset_data is not None else {}
        self.controls.graph.alpha_value.setEnabled(subset_data is None)

    # called when all signals are received, so the graph is updated only once
    def handleNewSignals(self):
        if self.attribute_selection_list and self.data is not None and \
                self.data.domain is not None and \
                all(attr in self.data.domain for attr
                    in self.attribute_selection_list):
            self.attr_x = self.attribute_selection_list[0]
            self.attr_y = self.attribute_selection_list[1]
        self.attribute_selection_list = None
        self.attr_changed()
        self._vizrank_color_change()
        self.cb_class_density.setEnabled(self.can_draw_density())
        self.cb_reg_line.setEnabled(self.can_draw_regresssion_line())
        if self.data is not None and self.__pending_selection_restore is not None:
            self.apply_selection(self.__pending_selection_restore)
            self.__pending_selection_restore = None
        self.unconditional_commit()

    def apply_selection(self, selection):
        """Apply `selection` to the current plot."""
        if self.data is not None:
            self.graph.selection = np.zeros(self.graph.n_points, dtype=np.uint8)
            self.selection_group = [x for x in selection if x[0] < len(self.data)]
            selection_array = np.array(self.selection_group).T
            self.graph.selection[selection_array[0]] = selection_array[1]
            self.graph.update_selection_colors()

    @Inputs.features
    def set_shown_attributes(self, attributes):
        if attributes and len(attributes) >= 2:
            self.attribute_selection_list = attributes[:2]
        else:
            self.attribute_selection_list = None

    def set_attr(self, attr_x, attr_y):
        self.attr_x, self.attr_y = attr_x, attr_y
        self.attr_changed()

    def attr_changed(self):
        self.graph.reset_graph()
        for axis, var in (("bottom", self.attr_x), ("left", self.attr_y)):
            self.graph.set_axis_title(axis, var)
            if var and var.is_discrete:
                self.graph.set_axis_labels(axis, get_variable_values_sorted(var))
            else:
                self.graph.set_axis_labels(axis, None)

        self.cb_class_density.setEnabled(self.can_draw_density())
        self.cb_reg_line.setEnabled(self.can_draw_regresssion_line())
        self.send_features()

    # The color combo's callback calls self.graph.update_colors AND this method
    def update_colors(self):
        self._vizrank_color_change()
        self.cb_class_density.setEnabled(self.can_draw_density())

    def selection_changed(self):
        # Store current selection in a setting that is stored in workflow
        if isinstance(self.data, SqlTable):
            selection = None
        elif self.data is not None:
            selection = self.graph.get_selection()
        else:
            selection = None
        if selection is not None and len(selection):
            self.selection_group = list(zip(selection, self.graph.selection[selection]))
        else:
            self.selection_group = None

        self.commit()

    def send_data(self):
        # TODO: Implement selection for sql data
        def _get_selected():
            if not len(selection):
                return None
            return create_groups_table(data, group_selection, False, "Group")

        def _get_annotated():
            if graph.selection is not None and np.max(graph.selection) > 1:
                return create_groups_table(data, group_selection)
            else:
                return create_annotated_table(data, selection)

        graph = self.graph
        data = self.data
        selection = graph.get_selection()
        if graph.selection is not None:
            group_selection = np.zeros(len(self.data), dtype=int)
            group_selection[self.valid_data] = graph.selection
        self.Outputs.annotated_data.send(_get_annotated())
        self.Outputs.selected_data.send(_get_selected())

    def send_features(self):
        features = [attr for attr in [self.attr_x, self.attr_y] if attr]
        self.Outputs.features.send(features or None)

    def commit(self):
        self.send_data()
        self.send_features()

    def get_widget_name_extension(self):
        if self.data is not None:
            return "{} vs {}".format(self.attr_x.name, self.attr_y.name)

    def send_report(self):
        if self.data is None:
            return

        def name(var):
            return var and var.name

        caption = report.render_items_vert((
            ("Color", name(self.attr_color)),
            ("Label", name(self.attr_label)),
            ("Shape", name(self.attr_shape)),
            ("Size", name(self.attr_size)),
            ("Jittering", (self.attr_x.is_discrete or
                           self.attr_y.is_discrete or
                           self.graph.jitter_continuous) and
             self.graph.jitter_size)))
        self.report_plot()
        if caption:
            self.report_caption(caption)

    def onDeleteWidget(self):
        super().onDeleteWidget()
        self.graph.plot_widget.getViewBox().deleteLater()
        self.graph.plot_widget.clear()

    @classmethod
    def migrate_settings(cls, settings, version):
        if version < 2 and "selection" in settings and settings["selection"]:
            settings["selection_group"] = [(a, 1) for a in settings["selection"]]

    @classmethod
    def migrate_context(cls, context, version):
        if version < 3:
            values = context.values
            values["attr_color"] = values["graph"]["attr_color"]
            values["attr_size"] = values["graph"]["attr_size"]
            values["attr_shape"] = values["graph"]["attr_shape"]
            values["attr_label"] = values["graph"]["attr_label"]


def main(argv=None):
    import sys
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    a = QApplication(argv)
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "heart_disease"

    ow = OWScatterPlot()
    ow.show()
    ow.raise_()
    data = Table(filename)
    ow.set_data(data)
    ow.set_subset_data(data[:30])
    ow.handleNewSignals()

    rval = a.exec()

    ow.set_data(None)
    ow.set_subset_data(None)
    ow.handleNewSignals()
    ow.saveSettings()
    ow.onDeleteWidget()

    return rval

if __name__ == "__main__":
    main()
