import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import Qt, QTimer
from AnyQt.QtGui import (
    QPen, QFont, QFontInfo, QPalette, QKeySequence,
)
from AnyQt.QtWidgets import QApplication, QAction

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score

import Orange
from Orange.data import Table, Domain, StringVariable, ContinuousVariable, \
    DiscreteVariable
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
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME,
                                                 get_next_name)


def font_resize(font, factor, minsize=None, maxsize=None):
    font = QFont(font)
    fontinfo = QFontInfo(font)
    size = fontinfo.pointSizeF() * factor

    if minsize is not None:
        size = max(size, minsize)
    if maxsize is not None:
        size = min(size, maxsize)

    font.setPointSizeF(size)
    return font


class ScatterPlotVizRank(VizRankDialogAttrPair):
    captionTitle = "Score Plots"
    minK = 10

    def check_preconditions(self):
        self.Information.add_message(
            "class_required", "Data with a class variable is required.")
        self.Information.class_required.clear()
        if not super().check_preconditions():
            return False
        if not self.master.data.domain.class_var:
            self.Information.class_required()
            return False
        return True

    def iterate_states(self, initial_state):
        # If we put initialization of `self.attrs` to `initialize`,
        # `score_heuristic` would be run on every call to `set_data`.
        if initial_state is None:  # on the first call, compute order
            self.attrs = self.score_heuristic()
        yield from super().iterate_states(initial_state)

    def compute_score(self, state):
        graph = self.master.graph
        ind12 = [graph.domain.index(self.attrs[x]) for x in state]
        valid = graph.get_valid_list(ind12)
        X = graph.jittered_data[ind12, :][:, valid].T
        Y = self.master.data.Y[valid]
        if X.shape[0] < self.minK:
            return
        n_neighbors = min(self.minK, len(X) - 1)
        knn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
        ind = knn.kneighbors(return_distance=False)
        if self.master.data.domain.has_discrete_class:
            return -np.sum(Y[ind] == Y.reshape(-1, 1)) / n_neighbors / len(Y)
        else:
            return -r2_score(Y, np.mean(Y[ind], axis=1)) * \
                   (len(Y) / len(self.master.data))

    def bar_length(self, score):
        return max(0, -score)

    def score_heuristic(self):
        X = self.master.graph.jittered_data.T
        Y = self.master.data.Y
        mdomain = self.master.data.domain
        dom = Domain([ContinuousVariable(str(i)) for i in range(X.shape[1])],
                     mdomain.class_vars)
        data = Table(dom, X, Y)
        relief = ReliefF if isinstance(dom.class_var, DiscreteVariable) \
            else RReliefF
        weights = relief(n_iterations=100, k_nearest=self.minK)(data)
        attrs = sorted(zip(weights, mdomain.attributes),
                       key=lambda x: (-x[0], x[1].name))
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
        features = Output("Features", Table, dynamic=False)

    settingsHandler = DomainContextHandler()

    auto_send_selection = Setting(True)
    auto_sample = Setting(True)
    toolbar_selection = Setting(0)

    attr_x = ContextSetting(None)
    attr_y = ContextSetting(None)
    selection = Setting(None, schema_only=True)

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
        self.data_metas_X = None  # self.data, where primitive metas are moved to X
        self.sql_data = None  # Orange.data.sql.table.SqlTable
        self.attribute_selection_list = None  # list of Orange.data.Variable
        self.__timer = QTimer(self, interval=1200)
        self.__timer.timeout.connect(self.add_data)

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

        gui.valueSlider(
            box, self, value='graph.jitter_size', label='Jittering: ',
            values=self.jitter_sizes, callback=self.reset_graph_data,
            labelFormat=lambda x:
            "None" if x == 0 else ("%.1f %%" if x < 1 else "%d %%") % x)
        gui.checkBox(
            gui.indentedBox(box), self, 'graph.jitter_continuous',
            'Jitter continuous values', callback=self.reset_graph_data)

        self.sampling = gui.auto_commit(
            self.controlArea, self, "auto_sample", "Sample", box="Sampling",
            callback=self.switch_sampling, commit=lambda: self.add_data(1))
        self.sampling.setVisible(False)

        g = self.graph.gui
        g.point_properties_box(self.controlArea)
        self.models = [self.xy_model] + g.points_models

        box = gui.vBox(self.controlArea, "Plot Properties")
        g.add_widgets([g.ShowLegend, g.ShowGridLines], box)
        gui.checkBox(
            box, self, value='graph.tooltip_shows_all',
            label='Show all data on mouse hover')
        self.cb_class_density = gui.checkBox(
            box, self, value='graph.class_density', label='Show class density',
            callback=self.update_density)
        self.cb_reg_line = gui.checkBox(
            box, self, value='graph.show_reg_line',
            label='Show regression line', callback=self.update_regression_line)
        gui.checkBox(
            box, self, 'graph.label_only_selected',
            'Label only selected points', callback=self.graph.update_labels)

        self.zoom_select_toolbar = g.zoom_select_toolbar(
            gui.vBox(self.controlArea, "Zoom/Select"), nomargin=True,
            buttons=[g.StateButtonsBegin, g.SimpleSelect, g.Pan, g.Zoom,
                     g.StateButtonsEnd, g.ZoomReset]
        )
        buttons = self.zoom_select_toolbar.buttons
        buttons[g.Zoom].clicked.connect(self.graph.zoom_button_clicked)
        buttons[g.Pan].clicked.connect(self.graph.pan_button_clicked)
        buttons[g.SimpleSelect].clicked.connect(self.graph.select_button_clicked)
        buttons[g.ZoomReset].clicked.connect(self.graph.reset_button_clicked)
        self.controlArea.layout().addStretch(100)
        self.icons = gui.attributeIconDict

        p = self.graph.plot_widget.palette()
        self.graph.set_palette(p)

        gui.auto_commit(self.controlArea, self, "auto_send_selection",
                        "Send Selection", "Send Automatically")

        def zoom(s):
            """Zoom in/out by factor `s`."""
            viewbox = plot.getViewBox()
            # scaleBy scales the view's bounds (the axis range)
            viewbox.scaleBy((1 / s, 1 / s))

        def fit_to_view():
            viewbox = plot.getViewBox()
            viewbox.autoRange()

        zoom_in = QAction(
            "Zoom in", self, triggered=lambda: zoom(1.25)
        )
        zoom_in.setShortcuts([QKeySequence(QKeySequence.ZoomIn),
                              QKeySequence(self.tr("Ctrl+="))])
        zoom_out = QAction(
            "Zoom out", self, shortcut=QKeySequence.ZoomOut,
            triggered=lambda: zoom(1 / 1.25)
        )
        zoom_fit = QAction(
            "Fit in view", self,
            shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_0),
            triggered=fit_to_view
        )
        self.addActions([zoom_in, zoom_out, zoom_fit])

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        self.graph.update_tooltip(event.modifiers())

    def keyReleaseEvent(self, event):
        super().keyReleaseEvent(event)
        self.graph.update_tooltip(event.modifiers())

    # def settingsFromWidgetCallback(self, handler, context):
    #     context.selectionPolygons = []
    #     for curve in self.graph.selectionCurveList:
    #         xs = [curve.x(i) for i in range(curve.dataSize())]
    #         ys = [curve.y(i) for i in range(curve.dataSize())]
    #         context.selectionPolygons.append((xs, ys))

    # def settingsToWidgetCallback(self, handler, context):
    #     selections = getattr(context, "selectionPolygons", [])
    #     for (xs, ys) in selections:
    #         c = SelectionCurve("")
    #         c.setData(xs,ys)
    #         c.attach(self.graph)
    #         self.graph.selectionCurveList.append(c)

    def reset_graph_data(self, *_):
        if self.data is not None:
            self.graph.rescale_data()
            self.update_graph()

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
        self.data_metas_X = self.move_primitive_metas_to_X(data)

        if not same_domain:
            self.init_attr_values()
        self.vizrank.initialize()
        self.vizrank.attrs = self.data.domain.attributes if self.data is not None else []
        self.vizrank_button.setEnabled(
            self.data is not None and self.data.domain.class_var is not None
            and len(self.data.domain.attributes) > 1 and len(self.data) > 1)
        if self.data is not None and self.data.domain.class_var is None \
            and len(self.data.domain.attributes) > 1 and len(self.data) > 1:
            self.vizrank_button.setToolTip(
                "Data with a class variable is required.")
        else:
            self.vizrank_button.setToolTip("")
        self.openContext(self.data)

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
            self.data_metas_X = self.move_primitive_metas_to_X(self.data)
            self.handleNewSignals()

    def switch_sampling(self):
        self.__timer.stop()
        if self.auto_sample and self.sql_data:
            self.add_data()
            self.__timer.start()

    def move_primitive_metas_to_X(self, data):
        if data is not None:
            new_attrs = [a for a in data.domain.attributes + data.domain.metas
                         if a.is_primitive()]
            new_metas = [m for m in data.domain.metas if not m.is_primitive()]
            new_domain = Domain(new_attrs, data.domain.class_vars, new_metas)
            data = data.transform(new_domain)
        return data

    @Inputs.data_subset
    def set_subset_data(self, subset_data):
        self.warning()
        if isinstance(subset_data, SqlTable):
            if subset_data.approx_len() < AUTO_DL_LIMIT:
                subset_data = Table(subset_data)
            else:
                self.warning("Data subset does not support large Sql tables")
                subset_data = None
        self.subset_data = self.move_primitive_metas_to_X(subset_data)
        self.controls.graph.alpha_value.setEnabled(subset_data is None)

    # called when all signals are received, so the graph is updated only once
    def handleNewSignals(self):
        self.graph.new_data(self.sparse_to_dense(self.data_metas_X),
                            self.sparse_to_dense(self.subset_data))
        if self.attribute_selection_list and self.graph.domain and \
                all(attr in self.graph.domain
                        for attr in self.attribute_selection_list):
            self.attr_x = self.attribute_selection_list[0]
            self.attr_y = self.attribute_selection_list[1]
        self.attribute_selection_list = None
        self.update_graph()
        self.cb_class_density.setEnabled(self.graph.can_draw_density())
        self.cb_reg_line.setEnabled(self.graph.can_draw_regresssion_line())
        self.apply_selection()
        self.unconditional_commit()

    def prepare_data(self):
        """
        Only when dealing with sparse matrices.
        GH-2152
        """
        self.graph.new_data(self.sparse_to_dense(self.data_metas_X),
                            self.sparse_to_dense(self.subset_data),
                            new=False)

    def sparse_to_dense(self, input_data=None):
        self.vizrank_button.setEnabled(not (self.data and self.data.is_sparse()))
        if input_data is None or not input_data.is_sparse():
            return input_data
        keys = []
        attrs = {self.attr_x,
                 self.attr_y,
                 self.graph.attr_color,
                 self.graph.attr_shape,
                 self.graph.attr_size,
                 self.graph.attr_label}
        for i, attr in enumerate(input_data.domain):
            if attr in attrs:
                keys.append(i)
        new_domain = input_data.domain.select_columns(keys)
        dmx = input_data.transform(new_domain)
        dmx.X = dmx.X.toarray()
        # TODO: remove once we make sure Y is always dense.
        if sp.issparse(dmx.Y):
            dmx.Y = dmx.Y.toarray()
        return dmx

    def apply_selection(self):
        """Apply selection saved in workflow."""
        if self.data is not None and self.selection is not None:
            self.graph.selection = np.zeros(len(self.data), dtype=np.uint8)
            self.selection = [x for x in self.selection if x < len(self.data)]
            self.graph.selection[self.selection] = 1
            self.graph.update_colors(keep_colors=True)

    @Inputs.features
    def set_shown_attributes(self, attributes):
        if attributes and len(attributes) >= 2:
            self.attribute_selection_list = attributes[:2]
        else:
            self.attribute_selection_list = None

    def get_shown_attributes(self):
        return self.attr_x, self.attr_y

    def init_attr_values(self):
        domain = self.data and self.data.domain
        for model in self.models:
            model.set_domain(domain)
        self.attr_x = self.xy_model[0] if self.xy_model else None
        self.attr_y = self.xy_model[1] if len(self.xy_model) >= 2 \
            else self.attr_x
        self.graph.attr_color = domain and self.data.domain.class_var or None
        self.graph.attr_shape = None
        self.graph.attr_size = None
        self.graph.attr_label = None

    def set_attr(self, attr_x, attr_y):
        self.attr_x, self.attr_y = attr_x, attr_y
        self.update_attr()

    def update_attr(self):
        self.prepare_data()
        self.update_graph()
        self.cb_class_density.setEnabled(self.graph.can_draw_density())
        self.cb_reg_line.setEnabled(self.graph.can_draw_regresssion_line())
        self.send_features()

    def update_colors(self):
        self.prepare_data()
        self.cb_class_density.setEnabled(self.graph.can_draw_density())

    def update_density(self):
        self.update_graph(reset_view=False)

    def update_regression_line(self):
        self.update_graph(reset_view=False)

    def update_graph(self, reset_view=True, **_):
        axis = self.graph.plot_widget.getAxis("left")
        axis.textWidth = 0
        self.graph.zoomStack = []
        if self.graph.data is None:
            return
        self.graph.update_data(self.attr_x, self.attr_y, reset_view)

    def selection_changed(self):
        self.send_data()

    @staticmethod
    def create_groups_table(data, selection):
        if data is None:
            return None
        names = [var.name for var in data.domain.variables + data.domain.metas]
        name = get_next_name(names, "Selection group")
        metas = data.domain.metas + (
            DiscreteVariable(
                name,
                ["Unselected"] + ["G{}".format(i + 1)
                                  for i in range(np.max(selection))]),
        )
        domain = Domain(data.domain.attributes, data.domain.class_vars, metas)
        table = data.transform(domain)
        table.metas[:, len(data.domain.metas):] = \
            selection.reshape(len(data), 1)
        return table

    def send_data(self):
        selected = None
        selection = None
        # TODO: Implement selection for sql data
        graph = self.graph
        if isinstance(self.data, SqlTable):
            selected = self.data
        elif self.data is not None:
            selection = graph.get_selection()
            if len(selection) > 0:
                selected = self.data[selection]
        if graph.selection is not None and np.max(graph.selection) > 1:
            annotated = self.create_groups_table(self.data, graph.selection)
        else:
            annotated = create_annotated_table(self.data, selection)
        self.Outputs.selected_data.send(selected)
        self.Outputs.annotated_data.send(annotated)

        # Store current selection in a setting that is stored in workflow
        if self.selection is not None and len(selection):
            self.selection = list(selection)

    def send_features(self):
        features = None
        if self.attr_x or self.attr_y:
            dom = Domain([], metas=(StringVariable(name="feature"),))
            features = Table(dom, [[self.attr_x], [self.attr_y]])
            features.name = "Features"
        self.Outputs.features.send(features)

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

    def closeContext(self):
        if self.current_context is not None:
            # When dataset changes, forget selection
            self.selection = None
        super().closeContext()

    def onDeleteWidget(self):
        super().onDeleteWidget()
        self.graph.plot_widget.getViewBox().deleteLater()
        self.graph.plot_widget.clear()


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
