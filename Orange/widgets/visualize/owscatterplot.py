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
    Setting, ContextSetting, SettingProvider
)
from Orange.widgets.utils import get_variable_values_sorted
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase
from Orange.widgets.visualize.utils import VizRankDialogAttrPair
from Orange.widgets.visualize.utils.widget import OWDataProjectionWidget
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
            return None
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
        self.update_tooltip()

    def _get_jittering_tooltip(self):
        def is_discrete(attr):
            return attr and attr.is_discrete

        if self.jitter_continuous or is_discrete(self.master.attr_x) or \
                is_discrete(self.master.attr_y):
            return super()._get_jittering_tooltip()
        return ""

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


class OWScatterPlot(OWDataProjectionWidget):
    """Scatterplot visualization with explorative analysis and intelligent
    data visualization enhancements."""

    name = 'Scatter Plot'
    description = "Interactive scatter plot visualization with " \
                  "intelligent data visualization enhancements."
    icon = "icons/ScatterPlot.svg"
    priority = 140
    keywords = []

    class Inputs(OWDataProjectionWidget.Inputs):
        features = Input("Features", AttributeList)

    class Outputs(OWDataProjectionWidget.Outputs):
        features = Output("Features", AttributeList, dynamic=False)

    settings_version = 3
    auto_sample = Setting(True)
    attr_x = ContextSetting(None)
    attr_y = ContextSetting(None)
    tooltip_shows_all = Setting(True)

    GRAPH_CLASS = OWScatterPlotGraph
    graph = SettingProvider(OWScatterPlotGraph)
    embedding_variables_names = None

    class Warning(OWDataProjectionWidget.Warning):
        missing_coords = Msg(
            "Plot cannot be displayed because '{}' or '{}' "
            "is missing for all data points")

    class Information(OWDataProjectionWidget.Information):
        sampled_sql = Msg("Large SQL table; showing a sample.")
        missing_coords = Msg(
            "Points with missing '{}' or '{}' are not displayed")

    def __init__(self):
        self.sql_data = None  # Orange.data.sql.table.SqlTable
        self.attribute_selection_list = None  # list of Orange.data.Variable
        self.__timer = QTimer(self, interval=1200)
        self.__timer.timeout.connect(self.add_data)
        super().__init__()

        # manually register Matplotlib file writers
        self.graph_writers = self.graph_writers.copy()
        for w in [MatplotlibFormat, MatplotlibPDFFormat]:
            for ext in w.EXTENSIONS:
                self.graph_writers[ext] = w

    def _add_controls(self):
        self._add_controls_axis()
        self._add_controls_sampling()
        super()._add_controls()
        self.graph.gui.add_widget(self.graph.gui.JitterNumericValues,
                                  self._effects_box)
        self.graph.gui.add_widgets([self.graph.gui.ShowGridLines,
                                    self.graph.gui.ToolTipShowsAll,
                                    self.graph.gui.RegressionLine],
                                   self._plot_box)

    def _add_controls_axis(self):
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
        self.vizrank, self.vizrank_button = ScatterPlotVizRank.add_vizrank(
            vizrank_box, self, "Find Informative Projections", self.set_attr)

    def _add_controls_sampling(self):
        self.sampling = gui.auto_commit(
            self.controlArea, self, "auto_sample", "Sample", box="Sampling",
            callback=self.switch_sampling, commit=lambda: self.add_data(1))
        self.sampling.setVisible(False)

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

    def set_data(self, data):
        if self.data and data and self.data.checksum() == data.checksum():
            return
        super().set_data(data)

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

    def check_data(self):
        self.clear_messages()
        self.__timer.stop()
        self.sampling.setVisible(False)
        self.sql_data = None
        if isinstance(self.data, SqlTable):
            if self.data.approx_len() < 4000:
                self.data = Table(self.data)
            else:
                self.Information.sampled_sql()
                self.sql_data = self.data
                data_sample = self.data.sample_time(0.8, no_cache=True)
                data_sample.download_data(2000, partial=True)
                self.data = Table(data_sample)
                self.sampling.setVisible(True)
                if self.auto_sample:
                    self.__timer.start()

        if self.data is not None and (len(self.data) == 0 or
                                      len(self.data.domain) == 0):
            self.data = None

    def get_embedding(self):
        self.valid_data = None
        if self.data is None:
            return None

        x_data = self.get_column(self.attr_x, filter_valid=False)
        y_data = self.get_column(self.attr_y, filter_valid=False)
        if x_data is None or y_data is None:
            return None

        self.Warning.missing_coords.clear()
        self.Information.missing_coords.clear()
        self.valid_data = np.isfinite(x_data) & np.isfinite(y_data)
        if self.valid_data is not None and not np.all(self.valid_data):
            msg = self.Information if np.any(self.valid_data) else self.Warning
            msg.missing_coords(self.attr_x.name, self.attr_y.name)
        return np.vstack((x_data, y_data)).T

    # Tooltip
    def _point_tooltip(self, point_id, skip_attrs=()):
        point_data = self.data[point_id]
        xy_attrs = (self.attr_x, self.attr_y)
        text = "<br/>".join(
            escape('{} = {}'.format(var.name, point_data[var]))
            for var in xy_attrs)
        if self.tooltip_shows_all:
            others = super()._point_tooltip(point_id, skip_attrs=xy_attrs)
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
            self.__timer.stop()
            return
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

    def set_subset_data(self, subset_data):
        self.warning()
        if isinstance(subset_data, SqlTable):
            if subset_data.approx_len() < AUTO_DL_LIMIT:
                subset_data = Table(subset_data)
            else:
                self.warning("Data subset does not support large Sql tables")
                subset_data = None
        super().set_subset_data(subset_data)

    # called when all signals are received, so the graph is updated only once
    def handleNewSignals(self):
        if self.attribute_selection_list and self.data is not None and \
                self.data.domain is not None and \
                all(attr in self.data.domain for attr
                        in self.attribute_selection_list):
            self.attr_x = self.attribute_selection_list[0]
            self.attr_y = self.attribute_selection_list[1]
        self.attribute_selection_list = None
        super().handleNewSignals()
        self._vizrank_color_change()
        self.cb_reg_line.setEnabled(self.can_draw_regresssion_line())

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
        self.cb_reg_line.setEnabled(self.can_draw_regresssion_line())
        self.setup_plot()
        self.commit()

    def setup_plot(self):
        super().setup_plot()
        for axis, var in (("bottom", self.attr_x), ("left", self.attr_y)):
            self.graph.set_axis_title(axis, var)
            if var and var.is_discrete:
                self.graph.set_axis_labels(axis,
                                           get_variable_values_sorted(var))
            else:
                self.graph.set_axis_labels(axis, None)

    def colors_changed(self):
        super().colors_changed()
        self._vizrank_color_change()

    def commit(self):
        super().commit()
        self.send_features()

    def send_features(self):
        features = [attr for attr in [self.attr_x, self.attr_y] if attr]
        self.Outputs.features.send(features or None)

    def get_widget_name_extension(self):
        if self.data is not None:
            return "{} vs {}".format(self.attr_x.name, self.attr_y.name)
        return None

    def _get_send_report_caption(self):
        return report.render_items_vert((
            ("Color", self._get_caption_var_name(self.attr_color)),
            ("Label", self._get_caption_var_name(self.attr_label)),
            ("Shape", self._get_caption_var_name(self.attr_shape)),
            ("Size", self._get_caption_var_name(self.attr_size)),
            ("Jittering", (self.attr_x.is_discrete or
                           self.attr_y.is_discrete or
                           self.graph.jitter_continuous) and
             self.graph.jitter_size)))

    @classmethod
    def migrate_settings(cls, settings, version):
        if version < 2 and "selection" in settings and settings["selection"]:
            settings["selection_group"] = [(a, 1) for a in settings["selection"]]
        if version < 3:
            if "auto_send_selection" in settings:
                settings["auto_commit"] = settings["auto_send_selection"]
            if "selection_group" in settings:
                settings["selection"] = settings["selection_group"]

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
        filename = "iris"

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
