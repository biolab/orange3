from itertools import islice, permutations, chain
from math import factorial
from types import SimpleNamespace as namespace
from xml.sax.saxutils import escape
import warnings

import numpy as np
from scipy.spatial import distance

from AnyQt.QtGui import QStandardItem, QColor, QFontMetrics
from AnyQt.QtCore import Qt, QEvent, QSize, QRectF, QPoint
from AnyQt.QtCore import pyqtSignal as Signal
from AnyQt.QtWidgets import qApp, QSizePolicy, QApplication, QToolTip, QGraphicsSceneMouseEvent, \
    QGraphicsEllipseItem

import pyqtgraph as pg
from pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.data.sql.table import SqlTable
from Orange.preprocess.score import ReliefF, RReliefF
from Orange.projection import radviz
from Orange.widgets import widget, gui, settings
from Orange.widgets.gui import OWComponent
from Orange.widgets.settings import Setting
from Orange.widgets.utils.annotated_data import (
    create_annotated_table, ANNOTATED_DATA_SIGNAL_NAME, create_groups_table)
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.plot import VariablesSelection
from Orange.widgets.visualize.utils import VizRankDialog
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotGraph, InteractiveViewBox, \
    HelpEventDelegate
from Orange.widgets.visualize.utils.plotutils import TextItem
from Orange.widgets.widget import Input, Output
from Orange.canvas import report


class RadvizVizRank(VizRankDialog, OWComponent):
    captionTitle = "Score Plots"
    n_attrs = settings.Setting(3)
    minK = 10

    attrsSelected = Signal([])
    _AttrRole = next(gui.OrangeUserRole)

    percent_data_used = Setting(100)

    def __init__(self, master):
        """Add the spin box for maximal number of attributes"""
        VizRankDialog.__init__(self, master)
        OWComponent.__init__(self, master)

        self.master = master
        self.n_neighbors = 10
        max_n_attrs = len(master.model_selected) + len(master.model_other) - 1

        box = gui.hBox(self)
        self.n_attrs_spin = gui.spin(
            box, self, "n_attrs", 3, max_n_attrs, label="Maximum number of variables: ",
            controlWidth=50, alignment=Qt.AlignRight, callback=self._n_attrs_changed)
        gui.rubber(box)
        self.last_run_n_attrs = None
        self.attr_color = master.graph.attr_color
        self.attr_ordering = None
        self.data = None
        self.valid_data = None

    def initialize(self):
        super().initialize()
        self.attr_color = self.master.graph.attr_color

    def _compute_attr_order(self):
        """
        used by VizRank to evaluate attributes
        """
        master = self.master
        attrs = [v for v in chain(master.model_selected[:], master.model_other[:])
                 if v is not self.attr_color]
        data = self.master.data.transform(Domain(attributes=attrs, class_vars=self.attr_color))
        self.data = data
        self.valid_data = np.hstack((~np.isnan(data.X), ~np.isnan(data.Y.reshape(len(data.Y), 1))))
        relief = ReliefF if self.attr_color.is_discrete else RReliefF
        weights = relief(n_iterations=100, k_nearest=self.minK)(data)
        attrs = sorted(zip(weights, attrs), key=lambda x: (-x[0], x[1].name))
        self.attr_ordering = attr_ordering = [a for _, a in attrs]
        return attr_ordering

    def _evaluate_projection(self, x, y):
        """
        kNNEvaluate - evaluate class separation in the given projection using a k-NN method
        Parameters
        ----------
        x - variables to evaluate
        y - class

        Returns
        -------
        scores
        """
        if self.percent_data_used != 100:
            rand = np.random.choice(len(x), int(len(x) * self.percent_data_used / 100),
                                    replace=False)
            x = x[rand]
            y = y[rand]
        neigh = KNeighborsClassifier(n_neighbors=3) if self.attr_color.is_discrete else \
            KNeighborsRegressor(n_neighbors=3)
        assert ~(np.isnan(x).any(axis=None) | np.isnan(x).any(axis=None))
        neigh.fit(x, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            scores = cross_val_score(neigh, x, y, cv=3)
        return scores.mean()

    def _n_attrs_changed(self):
        """
        Change the button label when the number of attributes changes. The method does not reset
        anything so the user can still see the results until actually restarting the search.
        """
        if self.n_attrs != self.last_run_n_attrs or self.saved_state is None:
            self.button.setText("Start")
        else:
            self.button.setText("Continue")
        self.button.setEnabled(self.check_preconditions())

    def progressBarSet(self, value, processEvents=None):
        self.setWindowTitle(self.captionTitle + " Evaluated {} permutations".format(value))
        if processEvents is not None and processEvents is not False:
            qApp.processEvents(processEvents)

    def check_preconditions(self):
        master = self.master
        if not super().check_preconditions():
            return False
        elif not master.btn_vizrank.isEnabled():
            return False
        self.n_attrs_spin.setMaximum(20)  # all primitive vars except color one
        return True

    def on_selection_changed(self, selected, deselected):
        attrs = selected.indexes()[0].data(self._AttrRole)
        self.selectionChanged.emit([attrs])

    def iterate_states(self, state):
        if state is None:  # on the first call, compute order
            self.attrs = self._compute_attr_order()
            state = list(range(3))
        else:
            state = list(state)

        def combinations(n, s):
            while True:
                yield s
                for up, _ in enumerate(s):
                    s[up] += 1
                    if up + 1 == len(s) or s[up] < s[up + 1]:
                        break
                    s[up] = up
                if s[-1] == n:
                    if len(s) < self.n_attrs:
                        s = list(range(len(s) + 1))
                    else:
                        break

        for c in combinations(len(self.attrs), state):
            for p in islice(permutations(c[1:]), factorial(len(c) - 1) // 2):
                yield (c[0],) + p

    def compute_score(self, state):
        attrs = [self.attrs[i] for i in state]
        domain = Domain(attributes=attrs, class_vars=[self.attr_color])
        data = self.data.transform(domain)
        radviz_xy, _, mask = radviz(data, attrs)
        y = data.Y[mask]
        return -self._evaluate_projection(radviz_xy, y)

    def bar_length(self, score):
        return -score

    def row_for_state(self, score, state):
        attrs = [self.attrs[s] for s in state]
        item = QStandardItem("[{:0.6f}] ".format(-score) + ", ".join(a.name for a in attrs))
        item.setData(attrs, self._AttrRole)
        return [item]

    def _update_progress(self):
        self.progressBarSet(int(self.saved_progress))

    def before_running(self):
        """
        Disable the spin for number of attributes before running and
        enable afterwards. Also, if the number of attributes is different than
        in the last run, reset the saved state (if it was paused).
        """
        if self.n_attrs != self.last_run_n_attrs:
            self.saved_state = None
            self.saved_progress = 0
        if self.saved_state is None:
            self.scores = []
            self.rank_model.clear()
        self.last_run_n_attrs = self.n_attrs
        self.n_attrs_spin.setDisabled(True)

    def stopped(self):
        self.n_attrs_spin.setDisabled(False)


class RadvizInteractiveViewBox(InteractiveViewBox):
    def __init__(self, graph, enable_menu=False):
        self.mouse_state = 0
        self.point_i = None
        super().__init__(graph, enable_menu)

    def _dragtip_pos(self):
        return 10, 10

    def mouseDragEvent(self, ev, axis=None):
        master = self.graph.master
        if master.data is None or master.graph.data is None:
            super().mouseDragEvent(ev, axis)
            return

        pos = self.childGroup.mapFromParent(ev.pos())
        points = master.plotdata.points
        np_pos = np.array([[pos.x(), pos.y()]])
        distances = distance.cdist(np_pos, points[:, :2])
        is_near = np.min(distances) < 0.1

        if ev.button() != Qt.LeftButton or (ev.start and not is_near):
            self.mouse_state = 2
        if self.mouse_state == 2:
            if ev.finish:
                self.mouse_state = 0
            super().mouseDragEvent(ev, axis)
            return

        ev.accept()
        if ev.start:
            self.setCursor(Qt.ClosedHandCursor)
            self.mouse_state = 1
            self.point_i = np.argmin(distances)
            master.randomize_indices()
        if self.mouse_state == 1:
            if ev.finish:
                self.setCursor(Qt.ArrowCursor)
                self.mouse_state = 0
            angle = np.arctan2(pos.y(), pos.x())
            QToolTip.showText(
                QPoint(ev.screenPos().x(), ev.screenPos().y()), "{:.2f}".format(np.rad2deg(angle)))
            points[self.point_i][0] = np.cos(angle)
            points[self.point_i][1] = np.sin(angle)
            if ev.finish:
                master.setup_plot()
                master.commit()
            else:
                master.manual_move()
            self.graph.show_arc_arrow(pos.x(), pos.y())


class EventDelegate(HelpEventDelegate):
    def __init__(self, delegate, delegate2, parent=None):
        self.delegate2 = delegate2
        super().__init__(delegate, parent=parent)

    def eventFilter(self, obj, ev):
        if isinstance(ev, QGraphicsSceneMouseEvent):
            self.delegate2(ev)
        return super().eventFilter(obj, ev)



SELECTION_WIDTH = 5

class OWRadvizGraph(OWScatterPlotGraph):
    jitter_size = settings.Setting(0)

    def __init__(self, scatter_widget, parent=None, name="None", view_box=None):
        super().__init__(scatter_widget, parent=parent, _=name, view_box=view_box)
        self._tooltip_delegate = EventDelegate(self.help_event, self._show_arc)
        self.plot_widget.scene().installEventFilter(self._tooltip_delegate)
        self.scatterplot_points = ScatterPlotItem(x=[], y=[])

    def hide_axes(self):
        for axis in ["left", "bottom"]:
            self.plot_widget.hideAxis(axis)

    def update_data(self, attr_x, attr_y, reset_view=True):
        if reset_view:
            self.view_box.setRange(RANGE, padding=0.025)
            self.view_box.setAspectLocked(True, 1)
        super().update_data(attr_x, attr_y, reset_view=False)
        self.hide_axes()

    def show_arc_arrow(self, x=None, y=None, point_i=None):
        def remove_arc_arrows():
            for arcarrow in self.master.plotdata.arcarrows:
                self.plot_widget.removeItem(arcarrow)
            self.master.plotdata.arcarrows = []
        def add_arc_arrows(x, y, col):
            func = self.view_box.childGroup.mapToDevice
            dx = (func(QPoint(1, 0)) - func(QPoint(-1, 0))).x()
            dangle = 6000 / dx
            arc = add_arc(np.arctan2(y, x), col, dangle)
            for a in arc:
                self.plot_widget.addItem(a)
            self.master.plotdata.arcarrows += arc

        remove_arc_arrows()
        if self.view_box.mouse_state == 0 and point_i is not None:
            point = self.master.plotdata.points[point_i, :]
            add_arc_arrows(point[0], point[1], 1)
        if self.view_box.mouse_state == 1 and x is not None:
            add_arc_arrows(x, y, 0)

    def _show_arc(self, ev):
        if self.scatterplot_item is None:
            return False
        if self.view_box.mouse_state == 1:
            return True

        for arcarrow in self.master.plotdata.arcarrows:
            self.plot_widget.removeItem(arcarrow)
        self.master.plotdata.arcarrows = []

        pos = self.scatterplot_item.mapFromScene(ev.scenePos())
        x = pos.x()
        y = pos.y()
        points = self.master.plotdata.points

        np_pos = np.array([[x, y]])
        distances = distance.cdist(np_pos, points[:, :2])[0]
        if len(distances) and np.min(distances) < 0.08:
            self.view_box.setCursor(Qt.OpenHandCursor)
            self.show_arc_arrow(point_i=np.argmin(distances))
        else:
            self.view_box.setCursor(Qt.ArrowCursor)
        return True

    def help_event(self, event):
        if self.scatterplot_item is None:
            return False

        act_pos = self.scatterplot_item.mapFromScene(event.scenePos())
        points = self.scatterplot_item.pointsAt(act_pos)
        text = ""
        vars = self.master.model_selected
        if len(points):
            for i, p in enumerate(points):
                index = p.data()
                text += "Attributes:\n"
                text += "".join(
                    "   {} = {}\n".format(attr.name, self.data[index][attr])
                    for attr in vars)
                if len(vars[:]) > 10:
                    text += "   ... and {} others\n\n".format(len(vars[:]) - 12)
                # class_var is always:
                text += "Class:\n   {} = {}\n".format(self.domain.class_var.name,
                                                      self.data[index][self.data.domain.class_var])
                if i < len(points) - 1:
                    text += '------------------\n'
            text = ('<span style="white-space:pre">{}</span>'.format(escape(text)))

            QToolTip.showText(event.screenPos(), text, widget=self.plot_widget)
            return True
        return False


RANGE = QRectF(-1.2, -1.05, 2.4, 2.1)
MAX_POINTS = 100

class OWRadviz(widget.OWWidget):
    name = "Radviz"
    description = "Radviz"

    icon = "icons/Radviz.svg"
    priority = 240

    class Inputs:
        data = Input("Data", Table, default=True)
        data_subset = Input("Data Subset", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)
        components = Output("Components", Table)

    settings_version = 1
    settingsHandler = settings.DomainContextHandler()

    variable_state = settings.ContextSetting({})

    auto_commit = settings.Setting(True)
    graph = settings.SettingProvider(OWRadvizGraph)
    vizrank = settings.SettingProvider(RadvizVizRank)

    jitter_sizes = [0, 0.1, 0.5, 1.0, 2.0]

    ReplotRequest = QEvent.registerEventType()

    graph_name = "graph.plot_widget.plotItem"

    class Information(widget.OWWidget.Information):
        sql_sampled_data = widget.Msg("Data has been sampled")

    class Warning(widget.OWWidget.Warning):
        no_features = widget.Msg("At least 2 features have to be chosen")

    class Error(widget.OWWidget.Error):
        sparse_data = widget.Msg("Sparse data is not supported")
        no_features = widget.Msg("At least 3 numeric or categorical variables are required")
        no_instances = widget.Msg("At least 2 data instances are required")

    def __init__(self):
        super().__init__()

        self.data = None
        self.subset_data = None
        self._subset_mask = None
        self._selection = None  # np.array
        self.__replot_requested = False
        self._new_plotdata()

        self.variable_x = ContinuousVariable("radviz-x")
        self.variable_y = ContinuousVariable("radviz-y")

        box = gui.vBox(self.mainArea, True, margin=0)
        self.graph = OWRadvizGraph(self, box, "Plot", view_box=RadvizInteractiveViewBox)
        self.graph.hide_axes()

        box.layout().addWidget(self.graph.plot_widget)
        plot = self.graph.plot_widget

        SIZE_POLICY = (QSizePolicy.Minimum, QSizePolicy.Maximum)

        self.variables_selection = VariablesSelection()
        self.model_selected = VariableListModel(enable_dnd=True)
        self.model_other = VariableListModel(enable_dnd=True)
        self.variables_selection(self, self.model_selected, self.model_other)

        self.vizrank, self.btn_vizrank = RadvizVizRank.add_vizrank(
            self.controlArea, self, "Suggest features", self.vizrank_set_attrs)
        self.btn_vizrank.setSizePolicy(*SIZE_POLICY)
        self.variables_selection.add_remove.layout().addWidget(self.btn_vizrank)

        self.viewbox = plot.getViewBox()
        self.replot = None

        g = self.graph.gui
        pp_box = g.point_properties_box(self.controlArea)
        pp_box.setSizePolicy(*SIZE_POLICY)
        self.models = g.points_models

        box = gui.vBox(self.controlArea, "Plot Properties")
        box.setSizePolicy(*SIZE_POLICY)
        g.add_widget(g.JitterSizeSlider, box)

        g.add_widgets([g.ShowLegend, g.ClassDensity, g.LabelOnlySelected], box)

        zoom_select = self.graph.box_zoom_select(self.controlArea)
        zoom_select.setSizePolicy(*SIZE_POLICY)

        self.icons = gui.attributeIconDict

        p = self.graph.plot_widget.palette()
        self.graph.set_palette(p)

        gui.auto_commit(self.controlArea, self, "auto_commit", "Send Selection",
                        auto_label="Send Automatically")

        self.graph.zoom_actions(self)

        self._circle = QGraphicsEllipseItem()
        self._circle.setRect(QRectF(-1., -1., 2., 2.))
        self._circle.setPen(pg.mkPen(QColor(0, 0, 0), width=2))

    def resizeEvent(self, event):
        self._update_points_labels()

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        self.graph.update_tooltip(event.modifiers())

    def keyReleaseEvent(self, event):
        super().keyReleaseEvent(event)
        self.graph.update_tooltip(event.modifiers())

    def vizrank_set_attrs(self, attrs):
        if not attrs:
            return
        self.variables_selection.display_none()
        self.model_selected[:] = attrs[:]
        self.model_other[:] = [v for v in self.model_other if v not in attrs]

    def _new_plotdata(self):
        self.plotdata = namespace(
            valid_mask=None,
            embedding_coords=None,
            points=None,
            arcarrows=[],
            point_labels=[],
            rand=None,
            data=None,
        )

    def update_colors(self):
        self._vizrank_color_change()
        self.cb_class_density.setEnabled(self.graph.can_draw_density())

    def sizeHint(self):
        return QSize(800, 500)

    def clear(self):
        """
        Clear/reset the widget state
        """
        self.data = None
        self.model_selected.clear()
        self.model_other.clear()
        self._clear_plot()

    def _clear_plot(self):
        self._new_plotdata()
        self.graph.plot_widget.clear()

    def invalidate_plot(self):
        """
        Schedule a delayed replot.
        """
        if not self.__replot_requested:
            self.__replot_requested = True
            QApplication.postEvent(self, QEvent(self.ReplotRequest), Qt.LowEventPriority - 10)

    def init_attr_values(self):
        self.graph.set_domain(self.data)

    def _vizrank_color_change(self):
        attr_color = self.graph.attr_color
        is_enabled = self.data is not None and not self.data.is_sparse() and \
                     (len(self.model_other) + len(self.model_selected)) > 3 and len(self.data) > 1
        self.btn_vizrank.setEnabled(
            is_enabled and attr_color is not None
            and not np.isnan(self.data.get_column_view(attr_color)[0].astype(float)).all())
        self.vizrank.initialize()

    @Inputs.data
    def set_data(self, data):
        """
        Set the input dataset and check if data is valid.

        Args:
            data (Orange.data.table): data instances
        """
        def sql(data):
            self.Information.sql_sampled_data.clear()
            if isinstance(data, SqlTable):
                if data.approx_len() < 4000:
                    data = Table(data)
                else:
                    self.Information.sql_sampled_data()
                    data_sample = data.sample_time(1, no_cache=True)
                    data_sample.download_data(2000, partial=True)
                    data = Table(data_sample)
            return data

        def settings(data):
            # get the default encoded state, replacing the position with Inf
            state = VariablesSelection.encode_var_state(
                [list(self.model_selected), list(self.model_other)]
            )
            state = {key: (source_ind, np.inf) for key, (source_ind, _) in state.items()}

            self.openContext(data.domain)
            selected_keys = [key
                             for key, (sind, _) in self.variable_state.items()
                             if sind == 0]

            if set(selected_keys).issubset(set(state.keys())):
                pass

            # update the defaults state (the encoded state must contain
            # all variables in the input domain)
            state.update(self.variable_state)
            # ... and restore it with saved positions taking precedence over
            # the defaults
            selected, other = VariablesSelection.decode_var_state(
                state, [list(self.model_selected), list(self.model_other)])
            return selected, other

        def is_sparse(data):
            if data.is_sparse():
                self.Error.sparse_data()
                data = None
            return data

        def are_features(data):
            domain = data.domain
            vars = [var for var in chain(domain.class_vars, domain.metas, domain.attributes)
                    if var.is_primitive()]
            if len(vars) < 3:
                self.Error.no_features()
                data = None
            return data

        def are_instances(data):
            if len(data) < 2:
                self.Error.no_instances()
                data = None
            return data

        self.clear_messages()
        self.btn_vizrank.setEnabled(False)
        self.closeContext()
        self.clear()
        self.information()
        self.Error.clear()
        for f in [sql, is_sparse, are_features, are_instances]:
            if data is None:
                break
            data = f(data)

        if data is not None:
            self.data = data
            self.init_attr_values()
            domain = data.domain
            vars = [v for v in chain(domain.metas, domain.attributes)
                    if v.is_primitive()]
            self.model_selected[:] = vars[:5]
            self.model_other[:] = vars[5:] + list(domain.class_vars)
            self.model_selected[:], self.model_other[:] = settings(data)
            self._selection = np.zeros(len(data), dtype=np.uint8)
            self.invalidate_plot()
        else:
            self.data = None

    @Inputs.data_subset
    def set_subset_data(self, subset):
        """
        Set the supplementary input subset dataset.

        Args:
            subset (Orange.data.table): subset of data instances
        """
        self.subset_data = subset
        self._subset_mask = None
        self.controls.graph.alpha_value.setEnabled(subset is None)

    def handleNewSignals(self):
        if self.data is not None:
            self._clear_plot()
            if self.subset_data is not None and self._subset_mask is None:
                dataids = self.data.ids.ravel()
                subsetids = np.unique(self.subset_data.ids)
                self._subset_mask = np.in1d(
                    dataids, subsetids, assume_unique=True)
            self.setup_plot(reset_view=True)
            self.cb_class_density.setEnabled(self.graph.can_draw_density())
        else:
            self.init_attr_values()
            self.graph.new_data(None)
        self._vizrank_color_change()
        self.commit()

    def customEvent(self, event):
        if event.type() == OWRadviz.ReplotRequest:
            self.__replot_requested = False
            self._clear_plot()
            self.setup_plot(reset_view=True)
        else:
            super().customEvent(event)

    def closeContext(self):
        self.variable_state = VariablesSelection.encode_var_state(
            [list(self.model_selected), list(self.model_other)]
        )
        super().closeContext()

    def prepare_radviz_data(self, variables):
        ec, points, valid_mask = radviz(self.data, variables, self.plotdata.points)
        self.plotdata.embedding_coords = ec
        self.plotdata.points = points
        self.plotdata.valid_mask = valid_mask

    def setup_plot(self, reset_view=True):
        if self.data is None:
            return
        self.graph.jitter_continuous = True
        self.__replot_requested = False

        variables = list(self.model_selected)
        if len(variables) < 2:
            self.Warning.no_features()
            self.graph.new_data(None)
            return

        self.Warning.clear()
        self.prepare_radviz_data(variables)

        if self.plotdata.embedding_coords is None:
            return

        domain = self.data.domain
        new_metas = domain.metas + (self.variable_x, self.variable_y)
        domain = Domain(attributes=domain.attributes,
                        class_vars=domain.class_vars,
                        metas=new_metas)
        mask = self.plotdata.valid_mask
        array = np.zeros((len(self.data), 2), dtype=np.float)
        array[mask] = self.plotdata.embedding_coords
        data = self.data.transform(domain)
        data[:, self.variable_x] = array[:, 0].reshape(-1, 1)
        data[:, self.variable_y] = array[:, 1].reshape(-1, 1)
        subset_data = data[self._subset_mask & mask]\
            if self._subset_mask is not None and len(self._subset_mask) else None
        self.plotdata.data = data
        self.graph.new_data(data[mask], subset_data)
        if self._selection is not None:
            self.graph.selection = self._selection[self.plotdata.valid_mask]
        self.graph.update_data(self.variable_x, self.variable_y, reset_view=reset_view)
        self.graph.plot_widget.addItem(self._circle)
        self.graph.scatterplot_points = ScatterPlotItem(
            x=self.plotdata.points[:, 0],
            y=self.plotdata.points[:, 1]
        )
        self._update_points_labels()
        self.graph.plot_widget.addItem(self.graph.scatterplot_points)

    def randomize_indices(self):
        ec = self.plotdata.embedding_coords
        self.plotdata.rand = np.random.choice(len(ec), MAX_POINTS, replace=False) \
            if len(ec) > MAX_POINTS else None

    def manual_move(self):
        self.__replot_requested = False

        if self.plotdata.rand is not None:
            rand = self.plotdata.rand
            valid_mask = self.plotdata.valid_mask
            data = self.data[valid_mask]
            selection = self._selection[valid_mask]
            selection = selection[rand]
            ec, _, valid_mask = radviz(data, list(self.model_selected), self.plotdata.points)
            assert sum(valid_mask) == len(data)
            data = data[rand]
            ec = ec[rand]
            data_x = data.X
            data_y = data.Y
            data_metas = data.metas
        else:
            self.prepare_radviz_data(list(self.model_selected))
            ec = self.plotdata.embedding_coords
            valid_mask = self.plotdata.valid_mask
            data_x = self.data.X[valid_mask]
            data_y = self.data.Y[valid_mask]
            data_metas = self.data.metas[valid_mask]
            selection = self._selection[valid_mask]

        attributes = (self.variable_x, self.variable_y) + self.data.domain.attributes
        domain = Domain(attributes=attributes,
                        class_vars=self.data.domain.class_vars,
                        metas=self.data.domain.metas)
        data = Table.from_numpy(domain, X=np.hstack((ec, data_x)), Y=data_y, metas=data_metas)
        self.graph.new_data(data, None)
        self.graph.selection = selection
        self.graph.update_data(self.variable_x, self.variable_y, reset_view=True)
        self.graph.plot_widget.addItem(self._circle)
        self.graph.scatterplot_points = ScatterPlotItem(
            x=self.plotdata.points[:, 0], y=self.plotdata.points[:, 1])
        self._update_points_labels()
        self.graph.plot_widget.addItem(self.graph.scatterplot_points)

    def _update_points_labels(self):
        if self.plotdata.points is None:
            return
        for point_label in self.plotdata.point_labels:
            self.graph.plot_widget.removeItem(point_label)
        self.plotdata.point_labels = []
        sx, sy = self.graph.view_box.viewPixelSize()

        for row in self.plotdata.points:
            ti = TextItem()
            metrics = QFontMetrics(ti.textItem.font())
            text_width = ((RANGE.width())/2. - np.abs(row[0])) / sx
            name = row[2].name
            ti.setText(name)
            ti.setTextWidth(text_width)
            ti.setColor(QColor(0, 0, 0))
            br = ti.boundingRect()
            width = metrics.width(name) if metrics.width(name) < br.width() else br.width()
            width = sx * (width + 5)
            height = sy * br.height()
            ti.setPos(row[0] - (row[0] < 0) * width, row[1] + (row[1] > 0) * height)
            self.plotdata.point_labels.append(ti)
            self.graph.plot_widget.addItem(ti)

    def _update_jitter(self):
        self.invalidate_plot()

    def reset_graph_data(self, *_):
        if self.data is not None:
            self.graph.rescale_data()
            self._update_graph()

    def _update_graph(self, reset_view=True, **_):
        self.graph.zoomStack = []
        if self.graph.data is None:
            return
        self.graph.update_data(self.variable_x, self.variable_y, reset_view=reset_view)

    def update_density(self):
        self._update_graph(reset_view=True)

    def selection_changed(self):
        if self.graph.selection is not None:
            self._selection[self.plotdata.valid_mask] = self.graph.selection
        self.commit()

    def prepare_data(self):
        pass

    def commit(self):
        selected = annotated = components = None
        graph = self.graph
        if self.plotdata.data is not None:
            name = self.data.name
            data = self.plotdata.data
            mask = self.plotdata.valid_mask.astype(int)
            mask[mask == 1] = graph.selection if graph.selection is not None \
                else [False * len(mask)]
            selection = np.array([], dtype=np.uint8) if mask is None else np.flatnonzero(mask)
            if len(selection):
                selected = data[selection]
                selected.name = name + ": selected"
                selected.attributes = self.data.attributes
            if graph.selection is not None and np.max(graph.selection) > 1:
                annotated = create_groups_table(data, mask)
            else:
                annotated = create_annotated_table(data, selection)
            annotated.attributes = self.data.attributes
            annotated.name = name + ": annotated"

            comp_domain = Domain(
                self.plotdata.points[:, 2],
                metas=[StringVariable(name='component')])

            metas = np.array([["RX"], ["RY"], ["angle"]])
            angle = np.arctan2(np.array(self.plotdata.points[:, 1].T, dtype=float),
                               np.array(self.plotdata.points[:, 0].T, dtype=float))
            components = Table.from_numpy(
                comp_domain,
                X=np.row_stack((self.plotdata.points[:, :2].T, angle)),
                metas=metas)
            components.name = name + ": components"

        self.Outputs.selected_data.send(selected)
        self.Outputs.annotated_data.send(annotated)
        self.Outputs.components.send(components)

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
            ("Jittering", self.graph.jitter_size != 0 and "{} %".format(self.graph.jitter_size))))
        self.report_plot()
        if caption:
            self.report_caption(caption)


def add_arc(angle, col, dangle=5):
    if col:
        color = QColor(128, 128, 128)  # gray
    else:
        color = QColor(0, 0, 0)  # black
    angle_d = np.rad2deg(angle)
    angle_2 = 90 - angle_d - dangle
    angle_1 = 270 - angle_d + dangle
    dangle = np.deg2rad(dangle)
    arrow1 = pg.ArrowItem(parent=None, angle=angle_1, brush=color, pen=pg.mkPen(color, width=1))
    arrow1.setPos(np.cos(angle - dangle), np.sin(angle - dangle))
    arrow2 = pg.ArrowItem(parent=None, angle=angle_2, brush=color, pen=pg.mkPen(color, width=1))
    arrow2.setPos(np.cos(angle + dangle), np.sin(angle + dangle))
    arc_x = np.fromfunction(lambda i: np.cos((angle - dangle) + (2 * dangle) * i / 120.), (121,),
                            dtype=int)
    arc_y = np.fromfunction(lambda i: np.sin((angle - dangle) + (2 * dangle) * i / 120.), (121,),
                            dtype=int)
    arc = pg.PlotCurveItem(
        x=arc_x, y=arc_y,
        pen=pg.mkPen(color, width=1),
        antialias=False
    )
    return [arc, arrow1, arrow2]


def main(argv=None):
    import sys
    import sip

    argv = sys.argv[1:] if argv is None else argv
    if argv:
        filename = argv[0]
    else:
        filename = "heart_disease"

    data = Table(filename)

    app = QApplication([])
    w = OWRadviz()
    w.set_data(data)
    w.set_subset_data(data[::10])
    w.handleNewSignals()
    w.show()
    w.raise_()
    r = app.exec()
    w.set_data(None)
    w.saveSettings()
    sip.delete(w)
    del w
    return r


if __name__ == "__main__":
    import sys
    sys.exit(main())
