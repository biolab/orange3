import numpy as np
from PyQt4.QtCore import Qt, QTimer
from PyQt4 import QtGui
from PyQt4.QtGui import QApplication
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
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotGraph
from Orange.widgets.visualize.utils import VizRankDialogAttrPair
from Orange.widgets.widget import OWWidget, Default, AttributeList, Msg


def font_resize(font, factor, minsize=None, maxsize=None):
    font = QtGui.QFont(font)
    fontinfo = QtGui.QFontInfo(font)
    size = fontinfo.pointSizeF() * factor

    if minsize is not None:
        size = max(size, minsize)
    if maxsize is not None:
        size = min(size, maxsize)

    font.setPointSizeF(size)
    return font


class ScatterPlotVizRank(VizRankDialogAttrPair):
    captionTitle = "Score Plots"
    K = 10

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
        ind12 = [graph.data_domain.index(self.attrs[x]) for x in state]
        valid = graph.get_valid_list(ind12)
        X = graph.scaled_data[ind12, :][:, valid].T
        Y = self.master.data.Y[valid]
        if X.shape[0] < self.K:
            return
        n_neighbors = min(self.K, len(X) - 1)
        knn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
        ind = knn.kneighbors(return_distance=False)
        if self.master.data.domain.has_discrete_class:
            return -np.sum(Y[ind] == Y.reshape(-1, 1))
        else:
            return -r2_score(Y, np.mean(Y[ind], axis=1)) * \
                   (len(Y) / len(self.master.data))

    def score_heuristic(self):
        X = self.master.graph.scaled_data.T
        Y = self.master.data.Y
        mdomain = self.master.data.domain
        dom = Domain([ContinuousVariable(str(i)) for i in range(X.shape[1])],
                     mdomain.class_vars)
        data = Table(dom, X, Y)
        relief = ReliefF if isinstance(dom.class_var, DiscreteVariable) \
            else RReliefF
        weights = relief(n_iterations=100, k_nearest=self.K)(data)
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

    inputs = [("Data", Table, "set_data", Default),
              ("Data Subset", Table, "set_subset_data"),
              ("Features", AttributeList, "set_shown_attributes")]

    outputs = [("Selected Data", Table, Default),
               ("Other Data", Table),
               ("Features", Table)]

    settingsHandler = DomainContextHandler()

    auto_send_selection = Setting(True)
    auto_sample = Setting(True)
    toolbar_selection = Setting(0)

    attr_x = ContextSetting("")
    attr_y = ContextSetting("")

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

        axispen = QtGui.QPen(self.palette().color(QtGui.QPalette.Text))
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
        self.cb_attr_x = gui.comboBox(box, self, "attr_x", label="Axis x:",
                                      callback=self.update_attr,
                                      **common_options)
        self.cb_attr_y = gui.comboBox(box, self, "attr_y", label="Axis y:",
                                      callback=self.update_attr,
                                      **common_options)

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

        box = gui.vBox(self.controlArea, "Points")
        self.cb_attr_color = gui.comboBox(
            box, self, "graph.attr_color", label="Color:",
            emptyString="(Same color)", callback=self.update_colors,
            **common_options)
        self.cb_attr_label = gui.comboBox(
            box, self, "graph.attr_label", label="Label:",
            emptyString="(No labels)", callback=self.graph.update_labels,
            **common_options)
        self.cb_attr_shape = gui.comboBox(
            box, self, "graph.attr_shape", label="Shape:",
            emptyString="(Same shape)", callback=self.graph.update_shapes,
            **common_options)
        self.cb_attr_size = gui.comboBox(
            box, self, "graph.attr_size", label="Size:",
            emptyString="(Same size)", callback=self.graph.update_sizes,
            **common_options)

        g = self.graph.gui
        g.point_properties_box(self.controlArea, box)
        box = gui.vBox(self.controlArea, "Plot Properties")
        g.add_widgets([g.ShowLegend, g.ShowGridLines], box)
        gui.checkBox(
            box, self, value='graph.tooltip_shows_all',
            label='Show all data on mouse hover')
        self.cb_class_density = gui.checkBox(
            box, self, value='graph.class_density', label='Show class density',
            callback=self.update_density)
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

        zoom_in = QtGui.QAction(
            "Zoom in", self, triggered=lambda: zoom(1.25)
        )
        zoom_in.setShortcuts([QtGui.QKeySequence(QtGui.QKeySequence.ZoomIn),
                              QtGui.QKeySequence(self.tr("Ctrl+="))])
        zoom_out = QtGui.QAction(
            "Zoom out", self, shortcut=QtGui.QKeySequence.ZoomOut,
            triggered=lambda: zoom(1 / 1.25)
        )
        zoom_fit = QtGui.QAction(
            "Fit in view", self,
            shortcut=QtGui.QKeySequence(Qt.ControlModifier | Qt.Key_0),
            triggered=fit_to_view
        )
        self.addActions([zoom_in, zoom_out, zoom_fit])

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
        self.graph.rescale_data()
        self.update_graph()

    def set_data(self, data):
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
            data = Table.from_table(Domain(new_attrs, data.domain.class_vars,
                                           new_metas), data)
        return data

    def set_subset_data(self, subset_data):
        self.warning()
        if isinstance(subset_data, SqlTable):
            if subset_data.approx_len() < AUTO_DL_LIMIT:
                subset_data = Table(subset_data)
            else:
                self.warning("Data subset does not support large Sql tables")
                subset_data = None
        self.subset_data = self.move_primitive_metas_to_X(subset_data)

    # called when all signals are received, so the graph is updated only once
    def handleNewSignals(self):
        self.graph.new_data(self.data_metas_X, self.subset_data)
        if self.attribute_selection_list and \
                all(attr in self.graph.data_domain
                    for attr in self.attribute_selection_list):
            self.attr_x = self.attribute_selection_list[0].name
            self.attr_y = self.attribute_selection_list[1].name
        self.attribute_selection_list = None
        self.update_graph()
        self.cb_class_density.setEnabled(self.graph.can_draw_density())
        self.unconditional_commit()

    def set_shown_attributes(self, attributes):
        if attributes and len(attributes) >= 2:
            self.attribute_selection_list = attributes[:2]
        else:
            self.attribute_selection_list = None

    def get_shown_attributes(self):
        return self.attr_x, self.attr_y

    def init_attr_values(self):
        self.cb_attr_x.clear()
        self.attr_x = None
        self.cb_attr_y.clear()
        self.attr_y = None
        self.cb_attr_color.clear()
        self.cb_attr_color.addItem("(Same color)")
        self.graph.attr_color = None
        self.cb_attr_label.clear()
        self.cb_attr_label.addItem("(No labels)")
        self.graph.attr_label = None
        self.cb_attr_shape.clear()
        self.cb_attr_shape.addItem("(Same shape)")
        self.graph.attr_shape = None
        self.cb_attr_size.clear()
        self.cb_attr_size.addItem("(Same size)")
        self.graph.attr_size = None
        if not self.data:
            return

        for var in self.data.domain.metas:
            if not var.is_primitive():
                self.cb_attr_label.addItem(self.icons[var], var.name)
        for attr in self.data.domain.variables:
            self.cb_attr_x.addItem(self.icons[attr], attr.name)
            self.cb_attr_y.addItem(self.icons[attr], attr.name)
            self.cb_attr_color.addItem(self.icons[attr], attr.name)
            if attr.is_discrete:
                self.cb_attr_shape.addItem(self.icons[attr], attr.name)
            else:
                self.cb_attr_size.addItem(self.icons[attr], attr.name)
            self.cb_attr_label.addItem(self.icons[attr], attr.name)
        for var in self.data.domain.metas:
            if var.is_primitive():
                self.cb_attr_x.addItem(self.icons[var], var.name)
                self.cb_attr_y.addItem(self.icons[var], var.name)
                self.cb_attr_color.addItem(self.icons[var], var.name)
                if var.is_discrete:
                    self.cb_attr_shape.addItem(self.icons[var], var.name)
                else:
                    self.cb_attr_size.addItem(self.icons[var], var.name)
                self.cb_attr_label.addItem(self.icons[var], var.name)

        self.attr_x = self.cb_attr_x.itemText(0)
        if self.cb_attr_y.count() > 1:
            self.attr_y = self.cb_attr_y.itemText(1)
        else:
            self.attr_y = self.cb_attr_y.itemText(0)

        if self.data.domain.class_var:
            self.graph.attr_color = self.data.domain.class_var.name
        else:
            self.graph.attr_color = ""
        self.graph.attr_shape = ""
        self.graph.attr_size = ""
        self.graph.attr_label = ""

    def set_attr(self, attr_x, attr_y):
        self.attr_x, self.attr_y = attr_x.name, attr_y.name
        self.update_attr()

    def update_attr(self):
        self.update_graph()
        self.cb_class_density.setEnabled(self.graph.can_draw_density())
        self.send_features()

    def update_colors(self):
        self.graph.update_colors()
        self.cb_class_density.setEnabled(self.graph.can_draw_density())

    def update_density(self):
        self.update_graph(reset_view=False)

    def update_graph(self, reset_view=True, **_):
        self.graph.zoomStack = []
        if not self.graph.have_data:
            return
        self.graph.update_data(self.attr_x, self.attr_y, reset_view)

    def selection_changed(self):
        self.send_data()

    def send_data(self):
        selected = unselected = None
        # TODO: Implement selection for sql data
        if isinstance(self.data, SqlTable):
            selected = unselected = self.data
        elif self.data is not None:
            selection = self.graph.get_selection()
            if len(selection) == 0:
                self.send("Selected Data", None)
                self.send("Other Data", self.data)
                return
            selected = self.data[selection]
            unselection = np.full(len(self.data), True, dtype=bool)
            unselection[selection] = False
            unselected = self.data[unselection]
        self.send("Selected Data", selected)
        if unselected is None or len(unselected) == 0:
            self.send("Other Data", None)
        else:
            self.send("Other Data", unselected)

    def send_features(self):
        features = None
        if self.attr_x or self.attr_y:
            dom = Domain([], metas=(StringVariable(name="feature"),))
            features = Table(dom, [[self.attr_x], [self.attr_y]])
            features.name = "Features"
        self.send("Features", features)

    def commit(self):
        self.send_data()
        self.send_features()

    def get_widget_name_extension(self):
        if self.data is not None:
            return "{} vs {}".format(self.combo_value(self.cb_attr_x),
                                     self.combo_value(self.cb_attr_y))

    def send_report(self):
        disc_attr = False
        if self.data:
            domain = self.data.domain
            disc_attr = domain[self.attr_x].is_discrete or \
                        domain[self.attr_y].is_discrete
        caption = report.render_items_vert((
            ("Color", self.combo_value(self.cb_attr_color)),
            ("Label", self.combo_value(self.cb_attr_label)),
            ("Shape", self.combo_value(self.cb_attr_shape)),
            ("Size", self.combo_value(self.cb_attr_size)),
            ("Jittering", (self.graph.jitter_continuous or disc_attr) and
             self.graph.jitter_size)))
        self.report_plot()
        if caption:
            self.report_caption(caption)

    def onDeleteWidget(self):
        super().onDeleteWidget()
        self.graph.plot_widget.getViewBox().deleteLater()
        self.graph.plot_widget.clear()


def test_main(argv=None):
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
    test_main()
