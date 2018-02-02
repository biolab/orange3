from itertools import chain

import numpy as np

from AnyQt.QtCore import Qt, QTimer
from AnyQt.QtGui import QPen, QPalette
from AnyQt.QtWidgets import QApplication

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score

import Orange
from Orange.data import Table, Domain, DiscreteVariable
from Orange.canvas import report
from Orange.data.sql.table import SqlTable, AUTO_DL_LIMIT
from Orange.preprocess.score import ReliefF, RReliefF
from Orange.widgets import gui
from Orange.widgets.settings import \
    DomainContextHandler, Setting, ContextSetting, SettingProvider
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotGraph
from Orange.widgets.visualize.utils import VizRankDialogAttrPair
from Orange.widgets.widget import OWWidget, AttributeList, Msg, Input, Output
from Orange.widgets.utils.annotated_data import (
    create_annotated_table, create_groups_table, ANNOTATED_DATA_SIGNAL_NAME)


class ScatterPlotVizRank(VizRankDialogAttrPair):
    captionTitle = "Score Plots"
    minK = 10
    attr_color = None

    def __init__(self, master):
        super().__init__(master)
        self.attr_color = self.master.graph.attr_color

    def initialize(self):
        self.attr_color = self.master.graph.attr_color
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
        data = self.master.graph.scaled_data
        data = data.transform(Domain(attrs, self.attr_color))
        data = data[~np.isnan(data.X).any(axis=1) & ~np.isnan(data.Y).T]
        if len(data) < self.minK:
            return
        n_neighbors = min(self.minK, len(data) - 1)
        knn = NearestNeighbors(n_neighbors=n_neighbors).fit(data.X)
        ind = knn.kneighbors(return_distance=False)
        if data.domain.has_discrete_class:
            return -np.sum(data.Y[ind] == data.Y.reshape(-1, 1)) / n_neighbors / len(data.Y)
        else:
            return -r2_score(data.Y, np.mean(data.Y[ind], axis=1)) * \
                   (len(data.Y) / len(self.master.data))

    def bar_length(self, score):
        return max(0, -score)

    def score_heuristic(self):
        assert self.attr_color is not None
        master_domain = self.master.graph.scaled_data.domain
        vars = [v for v in chain(master_domain.variables, master_domain.metas)
                if v is not self.attr_color]
        domain = Domain(attributes=vars, class_vars=self.attr_color)
        data = self.master.graph.scaled_data.transform(domain)
        relief = ReliefF if isinstance(domain.class_var, DiscreteVariable) else RReliefF
        weights = relief(n_iterations=100, k_nearest=self.minK)(data)
        attrs = sorted(zip(weights, domain.attributes), key=lambda x: (-x[0], x[1].name))
        return [a for _, a in attrs]



class OWScatterPlot(OWWidget):
    """Scatterplot visualization with explorative analysis and intelligent
    data visualization enhancements."""

    name = 'Scatter Plot'
    description = "Interactive scatter plot visualization with " \
                  "intelligent data visualization enhancements."
    icon = "icons/ScatterPlot.svg"
    priority = 140

    class Inputs:
        data = Input("Data", Table, default=True)
        data_subset = Input("Data Subset", Table)
        features = Input("Features", AttributeList)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)
        features = Output("Features", AttributeList, dynamic=False)

    settings_version = 2
    settingsHandler = DomainContextHandler()

    auto_send_selection = Setting(True)
    auto_sample = Setting(True)
    toolbar_selection = Setting(0)

    attr_x = ContextSetting(None)
    attr_y = ContextSetting(None)

    #: Serialized selection state to be restored
    selection_group = Setting(None, schema_only=True)

    graph = SettingProvider(OWScatterPlotGraph)

    jitter_sizes = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10]

    graph_name = "graph.plot_widget.plotItem"

    class Information(OWWidget.Information):
        sampled_sql = Msg("Large SQL table; showing a sample.")

    def __init__(self):
        super().__init__()

        box = gui.vBox(self.mainArea, True, margin=0)
        self.graph = OWScatterPlotGraph(self, box, "ScatterPlot")
        box.layout().addWidget(self.graph.plot_widget)
        plot = self.graph.plot_widget

        axispen = QPen(self.palette().color(QPalette.Text))
        axis = plot.getAxis("bottom")
        axis.setPen(axispen)

        axis = plot.getAxis("left")
        axis.setPen(axispen)

        self.data = None  # Orange.data.Table
        self.subset_data = None  # Orange.data.Table
        self.sql_data = None  # Orange.data.sql.table.SqlTable
        self.attribute_selection_list = None  # list of Orange.data.Variable
        self.__timer = QTimer(self, interval=1200)
        self.__timer.timeout.connect(self.add_data)
        #: Remember the saved state to restore
        self.__pending_selection_restore = self.selection_group
        self.selection_group = None

        common_options = dict(
            labelWidth=50, orientation=Qt.Horizontal, sendSelectedValue=True,
            valueType=str)
        box = gui.vBox(self.controlArea, "Axis Data")
        dmod = DomainModel
        self.xy_model = DomainModel(dmod.MIXED, valid_types=dmod.PRIMITIVE)
        self.cb_attr_x = gui.comboBox(
            box, self, "attr_x", label="Axis x:", callback=self.update_attr,
            model=self.xy_model, **common_options)
        self.cb_attr_y = gui.comboBox(
            box, self, "attr_y", label="Axis y:", callback=self.update_attr,
            model=self.xy_model, **common_options)

        vizrank_box = gui.hBox(box)
        gui.separator(vizrank_box, width=common_options["labelWidth"])
        self.vizrank, self.vizrank_button = ScatterPlotVizRank.add_vizrank(
            vizrank_box, self, "Find Informative Projections", self.set_attr)

        gui.separator(box)

        g = self.graph.gui
        g.add_widgets([g.JitterSizeSlider,
                       g.JitterNumericValues], box)

        self.sampling = gui.auto_commit(
            self.controlArea, self, "auto_sample", "Sample", box="Sampling",
            callback=self.switch_sampling, commit=lambda: self.add_data(1))
        self.sampling.setVisible(False)

        g.point_properties_box(self.controlArea)
        self.models = [self.xy_model] + g.points_models

        box_plot_prop = gui.vBox(self.controlArea, "Plot Properties")
        g.add_widgets([g.ShowLegend,
                       g.ShowGridLines,
                       g.ToolTipShowsAll,
                       g.ClassDensity,
                       g.RegressionLine,
                       g.LabelOnlySelected], box_plot_prop)

        self.graph.box_zoom_select(self.controlArea)

        self.controlArea.layout().addStretch(100)
        self.icons = gui.attributeIconDict

        p = self.graph.plot_widget.palette()
        self.graph.set_palette(p)

        gui.auto_commit(self.controlArea, self, "auto_send_selection",
                        "Send Selection", "Send Automatically")

        self.graph.zoom_actions(self)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        self.graph.update_tooltip(event.modifiers())

    def keyReleaseEvent(self, event):
        super().keyReleaseEvent(event)
        self.graph.update_tooltip(event.modifiers())

    def reset_graph_data(self, *_):
        if self.data is not None:
            self.graph.rescale_data()
            self.update_graph()

    def _vizrank_color_change(self):
        self.vizrank.initialize()
        is_enabled = self.data is not None and not self.data.is_sparse() and \
                     len([v for v in chain(self.data.domain.variables, self.data.domain.metas)
                          if v.is_primitive]) > 2\
                     and len(self.data) > 1
        self.vizrank_button.setEnabled(
            is_enabled and self.graph.attr_color is not None and
            not np.isnan(self.data.get_column_view(self.graph.attr_color)[0].astype(float)).all())
        if is_enabled and self.graph.attr_color is None:
            self.vizrank_button.setToolTip("Color variable has to be selected.")
        else:
            self.vizrank_button.setToolTip("")

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
        self._vizrank_color_change()

        def findvar(name, iterable):
            """Find a Orange.data.Variable in `iterable` by name"""
            for el in iterable:
                if isinstance(el, Orange.data.Variable) and el.name == name:
                    return el
            return None

        # handle restored settings from  < 3.3.9 when attr_* were stored
        # by name
        if isinstance(self.attr_x, str):
            self.attr_x = findvar(self.attr_x, self.xy_model)
        if isinstance(self.attr_y, str):
            self.attr_y = findvar(self.attr_y, self.xy_model)
        if isinstance(self.graph.attr_label, str):
            self.graph.attr_label = findvar(
                self.graph.attr_label, self.graph.gui.label_model)
        if isinstance(self.graph.attr_color, str):
            self.graph.attr_color = findvar(
                self.graph.attr_color, self.graph.gui.color_model)
        if isinstance(self.graph.attr_shape, str):
            self.graph.attr_shape = findvar(
                self.graph.attr_shape, self.graph.gui.shape_model)
        if isinstance(self.graph.attr_size, str):
            self.graph.attr_size = findvar(
                self.graph.attr_size, self.graph.gui.size_model)

    def add_data(self, time=0.4):
        if self.data and len(self.data) > 2000:
            return self.__timer.stop()
        data_sample = self.sql_data.sample_time(time, no_cache=True)
        if data_sample:
            data_sample.download_data(2000, partial=True)
            data = Table(data_sample)
            self.data = Table.concatenate((self.data, data), axis=0)
            self.handleNewSignals()

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
        self.controls.graph.alpha_value.setEnabled(subset_data is None)

    # called when all signals are received, so the graph is updated only once
    def handleNewSignals(self):
        self.graph.new_data(self.data, self.subset_data)
        if self.attribute_selection_list and self.graph.domain is not None and \
                all(attr in self.graph.domain
                        for attr in self.attribute_selection_list):
            self.attr_x = self.attribute_selection_list[0]
            self.attr_y = self.attribute_selection_list[1]
        self.attribute_selection_list = None
        self.update_graph()
        self.cb_class_density.setEnabled(self.graph.can_draw_density())
        self.cb_reg_line.setEnabled(self.graph.can_draw_regresssion_line())
        if self.data is not None and self.__pending_selection_restore is not None:
            self.apply_selection(self.__pending_selection_restore)
            self.__pending_selection_restore = None
        self.unconditional_commit()

    def apply_selection(self, selection):
        """Apply `selection` to the current plot."""
        if self.data is not None:
            self.graph.selection = np.zeros(len(self.data), dtype=np.uint8)
            self.selection_group = [x for x in selection if x[0] < len(self.data)]
            selection_array = np.array(self.selection_group).T
            self.graph.selection[selection_array[0]] = selection_array[1]
            self.graph.update_colors(keep_colors=True)

    @Inputs.features
    def set_shown_attributes(self, attributes):
        if attributes and len(attributes) >= 2:
            self.attribute_selection_list = attributes[:2]
        else:
            self.attribute_selection_list = None

    def init_attr_values(self):
        data = self.data
        domain = data.domain if data and len(data) else None
        self.xy_model.set_domain(domain)
        self.attr_x = self.xy_model[0] if self.xy_model else None
        self.attr_y = self.xy_model[1] if len(self.xy_model) >= 2 \
            else self.attr_x
        self.graph.set_domain(data)

    def set_attr(self, attr_x, attr_y):
        self.attr_x, self.attr_y = attr_x, attr_y
        self.update_attr()

    def update_attr(self):
        self.update_graph()
        self.cb_class_density.setEnabled(self.graph.can_draw_density())
        self.cb_reg_line.setEnabled(self.graph.can_draw_regresssion_line())
        self.send_features()

    def update_colors(self):
        self._vizrank_color_change()
        self.cb_class_density.setEnabled(self.graph.can_draw_density())

    def update_density(self):
        self.update_graph(reset_view=False)

    def update_regression_line(self):
        self.update_graph(reset_view=False)

    def update_graph(self, reset_view=True, **_):
        self.graph.zoomStack = []
        if self.graph.data is None:
            return
        self.graph.update_data(self.attr_x, self.attr_y, reset_view)

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
            return create_groups_table(data, graph.selection, False, "Group")

        def _get_annotated():
            if graph.selection is not None and np.max(graph.selection) > 1:
                return create_groups_table(data, graph.selection)
            else:
                return create_annotated_table(data, selection)

        graph = self.graph
        data = self.data
        selection = graph.get_selection()
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
            ("Color", name(self.graph.attr_color)),
            ("Label", name(self.graph.attr_label)),
            ("Shape", name(self.graph.attr_shape)),
            ("Size", name(self.graph.attr_size)),
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
    data = Orange.data.Table(filename)
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
