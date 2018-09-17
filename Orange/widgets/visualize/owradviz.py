from itertools import islice, permutations, chain
from math import factorial
import warnings

import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from AnyQt.QtGui import QStandardItem, QColor
from AnyQt.QtCore import (
    Qt, QEvent, QRectF, QPoint, pyqtSignal as Signal
)
from AnyQt.QtWidgets import (
    qApp, QApplication, QToolTip, QGraphicsEllipseItem
)

import pyqtgraph as pg
from pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.data.sql.table import SqlTable
from Orange.preprocess.score import ReliefF, RReliefF
from Orange.projection import radviz
from Orange.widgets import widget, gui, settings, report
from Orange.widgets.gui import OWComponent
from Orange.widgets.settings import Setting
from Orange.widgets.utils.annotated_data import (
    create_annotated_table, ANNOTATED_DATA_SIGNAL_NAME, create_groups_table
)
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.plot import VariablesSelection
from Orange.widgets.visualize.owscatterplotgraph import OWProjectionWidget
from Orange.widgets.visualize.utils import VizRankDialog
from Orange.widgets.visualize.utils.component import OWVizGraph
from Orange.widgets.visualize.utils.plotutils import (
    TextItem, VizInteractiveViewBox
)
from Orange.widgets.widget import Input, Output


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
        self.attr_color = master.attr_color
        self.attr_ordering = None
        self.data = None
        self.valid_data = None

    def initialize(self):
        super().initialize()
        self.attr_color = self.master.attr_color

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


class RadvizInteractiveViewBox(VizInteractiveViewBox):
    def mouseDragEvent(self, ev, axis=None):
        super().mouseDragEvent(ev, axis)
        if ev.finish:
            self.setCursor(Qt.ArrowCursor)
            self.graph.show_indicator(None)

    def _show_tooltip(self, ev):
        pos = self.childGroup.mapFromParent(ev.pos())
        angle = np.arctan2(pos.y(), pos.x())
        point = QPoint(ev.screenPos().x(), ev.screenPos().y())
        QToolTip.showText(point, "{:.2f}".format(np.rad2deg(angle)))


class OWRadvizGraph(OWVizGraph):
    def __init__(self, scatter_widget, parent):
        super().__init__(scatter_widget, parent, RadvizInteractiveViewBox)
        self._text_items = []

    def set_point(self, i, x, y):
        angle = np.arctan2(y, x)
        super().set_point(i, np.cos(angle), np.sin(angle))

    def set_view_box_range(self):
        self.view_box.setRange(RANGE, padding=0.025)

    def can_show_indicator(self, pos):
        if self._points is None:
            return False, None

        np_pos = np.array([[pos.x(), pos.y()]])
        distances = distance.cdist(np_pos, self._points[:, :2])[0]
        if len(distances) and np.min(distances) < self.DISTANCE_DIFF:
            return True, np.argmin(distances)
        return False, None

    def update_items(self):
        super().update_items()
        self._update_text_items()

    def _update_text_items(self):
        self._remove_text_items()
        self._add_text_items()

    def _remove_text_items(self):
        for item in self._text_items:
            self.plot_widget.removeItem(item)
        self._text_items = []

    def _add_text_items(self):
        if self._points is None:
            return
        for point in self._points:
            ti = TextItem()
            ti.setText(point[2].name)
            ti.setColor(QColor(0, 0, 0))
            ti.setPos(point[0], point[1])
            self._text_items.append(ti)
            self.plot_widget.addItem(ti)

    def _add_point_items(self):
        if self._points is None:
            return
        x, y = self._points[:, 0], self._points[:, 1]
        self._point_items = ScatterPlotItem(x=x, y=y)
        self.plot_widget.addItem(self._point_items)

    def _add_circle_item(self):
        if self._points is None:
            return
        self._circle_item = QGraphicsEllipseItem()
        self._circle_item.setRect(QRectF(-1., -1., 2., 2.))
        self._circle_item.setPen(pg.mkPen(QColor(0, 0, 0), width=2))
        self.plot_widget.addItem(self._circle_item)

    def _add_indicator_item(self, point_i):
        if point_i is None:
            return
        x, y = self._points[point_i][:2]
        col = self.view_box.mouse_state
        dx = (self.view_box.childGroup.mapToDevice(QPoint(1, 0)) -
              self.view_box.childGroup.mapToDevice(QPoint(-1, 0))).x()
        self._indicator_item = MoveIndicator(np.arctan2(y, x), col, 6000 / dx)
        self.plot_widget.addItem(self._indicator_item)


RANGE = QRectF(-1.2, -1.05, 2.4, 2.1)
MAX_POINTS = 100


class OWRadviz(OWProjectionWidget):
    name = "Radviz"
    description = "Display Radviz projection"
    icon = "icons/Radviz.svg"
    priority = 241
    keywords = ["viz"]

    class Inputs:
        data = Input("Data", Table, default=True)
        data_subset = Input("Data Subset", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)
        components = Output("Components", Table)

    settings_version = 2
    settingsHandler = settings.DomainContextHandler()

    variable_state = settings.ContextSetting({})
    auto_commit = settings.Setting(True)

    vizrank = settings.SettingProvider(RadvizVizRank)
    graph = settings.SettingProvider(OWRadvizGraph)
    graph_name = "graph.plot_widget.plotItem"

    ReplotRequest = QEvent.registerEventType()

    class Information(OWProjectionWidget.Information):
        sql_sampled_data = widget.Msg("Data has been sampled")

    class Warning(OWProjectionWidget.Warning):
        no_features = widget.Msg("At least 2 features have to be chosen")
        invalid_embedding = widget.Msg("No projection for selected features")

    class Error(OWProjectionWidget.Error):
        sparse_data = widget.Msg("Sparse data is not supported")
        no_features = widget.Msg(
            "At least 3 numeric or categorical variables are required"
        )
        no_instances = widget.Msg("At least 2 data instances are required")

    def __init__(self):
        super().__init__()

        self.data = None
        self.subset_data = None
        self.subset_indices = None
        self._embedding_coords = None
        self._rand_indices = None

        self.__replot_requested = False

        self.variable_x = ContinuousVariable("radviz-x")
        self.variable_y = ContinuousVariable("radviz-y")

        box = gui.vBox(self.mainArea, True, margin=0)
        self.graph = OWRadvizGraph(self, box)
        box.layout().addWidget(self.graph.plot_widget)

        self.variables_selection = VariablesSelection()
        self.model_selected = selected = VariableListModel(enable_dnd=True)
        self.model_other = other = VariableListModel(enable_dnd=True)
        self.variables_selection(self, selected, other, self.controlArea)

        self.vizrank, self.btn_vizrank = RadvizVizRank.add_vizrank(
            None, self, "Suggest features", self.vizrank_set_attrs)
        # Todo: this button introduces some margin at the bottom?!
        self.variables_selection.add_remove.layout().addWidget(self.btn_vizrank)

        g = self.graph.gui
        g.point_properties_box(self.controlArea)
        g.effects_box(self.controlArea)
        g.plot_properties_box(self.controlArea)

        self.graph.box_zoom_select(self.controlArea)

        gui.auto_commit(self.controlArea, self, "auto_commit",
                        "Send Selection", "Send Automatically")

        self.graph.view_box.started.connect(self._randomize_indices)
        self.graph.view_box.moved.connect(self._manual_move)
        self.graph.view_box.finished.connect(self._finish_manual_move)

    def vizrank_set_attrs(self, attrs):
        if not attrs:
            return
        self.variables_selection.display_none()
        self.model_selected[:] = attrs[:]
        self.model_other[:] = [v for v in self.model_other if v not in attrs]

    def update_colors(self):
        self._vizrank_color_change()
        self.cb_class_density.setEnabled(self.can_draw_density())

    def invalidate_plot(self):
        """
        Schedule a delayed replot.
        """
        if not self.__replot_requested:
            self.__replot_requested = True
            QApplication.postEvent(self, QEvent(self.ReplotRequest), Qt.LowEventPriority - 10)

    def _vizrank_color_change(self):
        is_enabled = self.data is not None and not self.data.is_sparse() and \
            len(self.model_other) + len(self.model_selected) > 3 and \
            len(self.data[self.valid_data]) > 1 and \
            np.all(np.nan_to_num(np.nanstd(self.data.X, 0)) != 0)
        self.btn_vizrank.setEnabled(
            is_enabled and self.attr_color is not None
            and not np.isnan(self.data.get_column_view(
                self.attr_color)[0].astype(float)).all())
        self.vizrank.initialize()

    def clear(self):
        self.data = None
        self.valid_data = None
        self._embedding_coords = None
        self._rand_indices = None
        self.model_selected.clear()
        self.model_other.clear()

        self.graph.set_attributes(())
        self.graph.set_points(None)
        self.graph.update_coordinates()
        self.graph.clear()

    @Inputs.data
    def set_data(self, data):
        self.clear_messages()
        self.btn_vizrank.setEnabled(False)
        self.closeContext()
        self.clear()
        self.data = data
        self._check_data()
        self.init_attr_values()
        self.openContext(self.data)
        if self.data is not None:
            self.model_selected[:], self.model_other[:] = self._load_settings()

    def _check_data(self):
        if self.data is not None:
            domain = self.data.domain
            if self.data.is_sparse():
                self.Error.sparse_data()
                self.data = None
            elif isinstance(self.data, SqlTable):
                if self.data.approx_len() < 4000:
                    self.data = Table(self.data)
                else:
                    self.Information.sql_sampled_data()
                    data_sample = self.data.sample_time(1, no_cache=True)
                    data_sample.download_data(2000, partial=True)
                    self.data = Table(data_sample)
            elif len(self.data) < 2:
                self.Error.no_instances()
                self.data = None
            elif len([v for v in domain.variables +
                     domain.metas if v.is_primitive()]) < 3:
                self.Error.no_features()
                self.data = None

    def _load_settings(self):
        domain = self.data.domain
        variables = [v for v in domain.attributes + domain.metas
                     if v.is_primitive()]
        self.model_selected[:] = variables[:5]
        self.model_other[:] = variables[5:] + list(domain.class_vars)

        state = VariablesSelection.encode_var_state(
            [list(self.model_selected), list(self.model_other)]
        )
        state = {key: (ind, np.inf) for key, (ind, _) in state.items()}
        state.update(self.variable_state)
        return VariablesSelection.decode_var_state(
            state, [list(self.model_selected), list(self.model_other)])

    @Inputs.data_subset
    def set_subset_data(self, subset):
        self.subset_data = subset
        self.subset_indices = {e.id for e in subset} \
            if subset is not None else {}
        self.controls.graph.alpha_value.setEnabled(subset is None)

    def handleNewSignals(self):
        self.setup_plot()
        self._vizrank_color_change()
        self.commit()

    def get_coordinates_data(self):
        ec = self._embedding_coords
        if ec is None or np.any(np.isnan(ec)):
            return None, None
        return ec[:, 0], ec[:, 1]

    def get_subset_mask(self):
        if self.subset_indices:
            return np.array([ex.id in self.subset_indices
                             for ex in self.data[self.valid_data]])

    def customEvent(self, event):
        if event.type() == OWRadviz.ReplotRequest:
            self.__replot_requested = False
            self.setup_plot()
        else:
            super().customEvent(event)

    def closeContext(self):
        self.variable_state = VariablesSelection.encode_var_state(
            [list(self.model_selected), list(self.model_other)]
        )
        super().closeContext()

    def setup_plot(self):
        if self.data is None:
            return
        self.__replot_requested = False

        self.clear_messages()
        if len(self.model_selected) < 2:
            self.Warning.no_features()
            self.graph.clear()
            return

        r = radviz(self.data, self.model_selected)
        self._embedding_coords = r[0]
        self.graph.set_points(r[1])
        self.valid_data = r[2]
        if self._embedding_coords is None or \
                np.any(np.isnan(self._embedding_coords)):
            self.Warning.invalid_embedding()
        self.graph.reset_graph()

    def _randomize_indices(self):
        n = len(self._embedding_coords)
        if n > MAX_POINTS:
            self._rand_indices = np.random.choice(n, MAX_POINTS, replace=False)
            self._rand_indices = sorted(self._rand_indices)

    def _manual_move(self):
        self.__replot_requested = False

        res = radviz(self.data, self.model_selected, self.graph.get_points())
        self._embedding_coords = res[0]
        if self._rand_indices is not None:
            # save widget state
            selection = self.graph.selection
            valid_data = self.valid_data.copy()
            data = self.data.copy()
            ec = self._embedding_coords.copy()

            # plot subset
            self.__plot_random_subset(selection)

            # restore widget state
            self.graph.selection = selection
            self.valid_data = valid_data
            self.data = data
            self._embedding_coords = ec
        else:
            self.graph.update_coordinates()

    def __plot_random_subset(self, selection):
        self._embedding_coords = self._embedding_coords[self._rand_indices]
        self.data = self.data[self._rand_indices]
        self.valid_data = self.valid_data[self._rand_indices]
        self.graph.reset_graph()
        if selection is not None:
            self.graph.selection = selection[self._rand_indices]
            self.graph.update_selection_colors()

    def _finish_manual_move(self):
        if self._rand_indices is not None:
            selection = self.graph.selection
            self.graph.reset_graph()
            if selection is not None:
                self.graph.selection = selection
                self.graph.select_by_index(self.graph.get_selection())

    def selection_changed(self):
        self.commit()

    def commit(self):
        selected = annotated = components = None
        if self.data is not None and np.sum(self.valid_data):
            name = self.data.name
            domain = self.data.domain
            metas = domain.metas + (self.variable_x, self.variable_y)
            domain = Domain(domain.attributes, domain.class_vars, metas)
            embedding_coords = np.zeros((len(self.data), 2), dtype=np.float)
            embedding_coords[self.valid_data] = self._embedding_coords

            data = self.data.transform(domain)
            data[:, self.variable_x] = embedding_coords[:, 0][:, None]
            data[:, self.variable_y] = embedding_coords[:, 1][:, None]

            selection = self.graph.get_selection()
            if len(selection):
                selected = data[selection]
                selected.name = name + ": selected"
                selected.attributes = self.data.attributes
            if self.graph.selection is not None and \
                    np.max(self.graph.selection) > 1:
                annotated = create_groups_table(data, self.graph.selection)
            else:
                annotated = create_annotated_table(data, selection)
            annotated.attributes = self.data.attributes
            annotated.name = name + ": annotated"

            points = self.graph.get_points()
            comp_domain = Domain(
                points[:, 2],
                metas=[StringVariable(name='component')])

            metas = np.array([["RX"], ["RY"], ["angle"]])
            angle = np.arctan2(np.array(points[:, 1].T, dtype=float),
                               np.array(points[:, 0].T, dtype=float))
            components = Table.from_numpy(
                comp_domain,
                X=np.row_stack((points[:, :2].T, angle)),
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
            ("Color", name(self.attr_color)),
            ("Label", name(self.attr_label)),
            ("Shape", name(self.attr_shape)),
            ("Size", name(self.attr_size)),
            ("Jittering", self.graph.jitter_size != 0 and
             "{} %".format(self.graph.jitter_size))))
        self.report_plot()
        if caption:
            self.report_caption(caption)

    @classmethod
    def migrate_context(cls, context, version):
        if version < 3:
            values = context.values
            values["attr_color"] = values["graph"]["attr_color"]
            values["attr_size"] = values["graph"]["attr_size"]
            values["attr_shape"] = values["graph"]["attr_shape"]
            values["attr_label"] = values["graph"]["attr_label"]


class MoveIndicator(pg.GraphicsObject):
    def __init__(self, angle, col, dangle=5, parent=None):
        super().__init__(parent)
        color = QColor(0, 0, 0) if col else QColor(128, 128, 128)
        angle_d = np.rad2deg(angle)
        angle_2 = 90 - angle_d - dangle
        angle_1 = 270 - angle_d + dangle
        dangle = np.deg2rad(dangle)
        arrow1 = pg.ArrowItem(
            parent=self, angle=angle_1, brush=color, pen=pg.mkPen(color)
        )
        arrow1.setPos(np.cos(angle - dangle), np.sin(angle - dangle))
        arrow2 = pg.ArrowItem(
            parent=self, angle=angle_2, brush=color, pen=pg.mkPen(color)
        )
        arrow2.setPos(np.cos(angle + dangle), np.sin(angle + dangle))
        arc_x = np.fromfunction(
            lambda i: np.cos((angle - dangle) + (2 * dangle) * i / 120.),
            (121,), dtype=int
        )
        arc_y = np.fromfunction(
            lambda i: np.sin((angle - dangle) + (2 * dangle) * i / 120.),
            (121,), dtype=int
        )
        pg.PlotCurveItem(
            parent=self, x=arc_x, y=arc_y, pen=pg.mkPen(color), antialias=False
        )

    def paint(self, painter, option, widget):
        pass

    def boundingRect(self):
        return QRectF()


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
