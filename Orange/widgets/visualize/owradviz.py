from itertools import islice, permutations, chain
from math import factorial
import warnings

import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from AnyQt.QtGui import QStandardItem, QColor
from AnyQt.QtCore import Qt, QRectF, QPoint, pyqtSignal as Signal
from AnyQt.QtWidgets import qApp

import pyqtgraph as pg
from pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem

from Orange.data import Table, Domain
from Orange.preprocess.score import ReliefF, RReliefF
from Orange.projection import RadViz
from Orange.widgets import widget, gui
from Orange.widgets.gui import OWComponent
from Orange.widgets.settings import Setting, ContextSetting, SettingProvider
from Orange.widgets.utils import vartype
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.plot import VariablesSelection
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.utils import VizRankDialog
from Orange.widgets.visualize.utils.component import OWGraphWithAnchors
from Orange.widgets.visualize.utils.plotutils import TextItem
from Orange.widgets.visualize.utils.widget import OWAnchorProjectionWidget


class RadvizVizRank(VizRankDialog, OWComponent):
    captionTitle = "Score Plots"
    n_attrs = Setting(3)
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
        projector = RadViz()
        projection = projector(data)
        radviz_xy = projection(data)
        y = projector.preprocess(data).Y
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


class OWRadvizGraph(OWGraphWithAnchors):
    def __init__(self, scatter_widget, parent):
        super().__init__(scatter_widget, parent)
        self.anchors_scatter_item = None

    def clear(self):
        super().clear()
        self.anchors_scatter_item = None

    def set_view_box_range(self):
        self.view_box.setRange(QRectF(-1.2, -1.05, 2.4, 2.1), padding=0.025)

    def closest_draggable_item(self, pos):
        points, _ = self.master.get_anchors()
        if points is None:
            return None
        np_pos = np.array([[pos.x(), pos.y()]])
        distances = distance.cdist(np_pos, points[:, :2])[0]
        if len(distances) and np.min(distances) < self.DISTANCE_DIFF:
            return np.argmin(distances)
        return None

    def update_anchors(self):
        points, labels = self.master.get_anchors()
        if points is None:
            return
        if self.anchor_items is None:
            self.anchor_items = []
            for point, label in zip(points, labels):
                anchor = TextItem()
                anchor.setText(label)
                anchor.setColor(QColor(0, 0, 0))
                anchor.setPos(*point)
                self.plot_widget.addItem(anchor)
                self.anchor_items.append(anchor)
        else:
            for anchor, point in zip(self.anchor_items, points):
                anchor.setPos(*point)
        self._update_anchors_scatter_item(points)

    def _update_anchors_scatter_item(self, points):
        if self.anchors_scatter_item is not None:
            self.plot_widget.removeItem(self.anchors_scatter_item)
            self.anchors_scatter_item = None
        self.anchors_scatter_item = ScatterPlotItem(x=points[:, 0],
                                                    y=points[:, 1])
        self.plot_widget.addItem(self.anchors_scatter_item)

    def _add_indicator_item(self, anchor_idx):
        if anchor_idx is None:
            return
        x, y = self.anchor_items[anchor_idx].get_xy()
        col = self.view_box.mouse_state
        dx = (self.view_box.childGroup.mapToDevice(QPoint(1, 0)) -
              self.view_box.childGroup.mapToDevice(QPoint(-1, 0))).x()
        self.indicator_item = MoveIndicator(np.arctan2(y, x), col, 6000 / dx)
        self.plot_widget.addItem(self.indicator_item)


class OWRadviz(OWAnchorProjectionWidget):
    name = "Radviz"
    description = "Display Radviz projection"
    icon = "icons/Radviz.svg"
    priority = 241
    keywords = ["viz"]

    settings_version = 2

    selected_vars = ContextSetting([])
    vizrank = SettingProvider(RadvizVizRank)
    GRAPH_CLASS = OWRadvizGraph
    graph = SettingProvider(OWRadvizGraph)

    class Warning(OWAnchorProjectionWidget.Warning):
        invalid_embedding = widget.Msg("No projection for selected features")
        removed_vars = widget.Msg("Categorical variables with more than"
                                  " two values are not shown.")

    def __init__(self):
        self.model_selected = VariableListModel(enable_dnd=True)
        self.model_selected.removed.connect(self.__model_selected_changed)
        self.model_other = VariableListModel(enable_dnd=True)

        self.vizrank, self.btn_vizrank = RadvizVizRank.add_vizrank(
            None, self, "Suggest features", self.__vizrank_set_attrs
        )
        super().__init__()

    def _add_controls(self):
        self.variables_selection = VariablesSelection(
            self, self.model_selected, self.model_other, self.controlArea
        )
        self.variables_selection.added.connect(self.__model_selected_changed)
        self.variables_selection.removed.connect(self.__model_selected_changed)
        self.variables_selection.add_remove.layout().addWidget(
            self.btn_vizrank
        )
        super()._add_controls()
        self.controlArea.layout().removeWidget(self.control_area_stretch)
        self.control_area_stretch.setParent(None)

    @property
    def primitive_variables(self):
        if self.data is None or self.data.domain is None:
            return []
        dom = self.data.domain
        return [v for v in chain(dom.variables, dom.metas)
                if v.is_continuous or v.is_discrete and len(v.values) == 2]

    @property
    def effective_variables(self):
        return self.model_selected[:]

    def __vizrank_set_attrs(self, attrs):
        if not attrs:
            return
        self.model_selected[:] = attrs[:]
        self.model_other[:] = [var for var in self.primitive_variables
                               if var not in attrs]
        self.__model_selected_changed()

    def __model_selected_changed(self):
        self.selected_vars = [(var.name, vartype(var)) for var
                              in self.model_selected]
        self.init_projection()
        self.setup_plot()
        self.commit()

    def colors_changed(self):
        super().colors_changed()
        self._init_vizrank()

    def set_data(self, data):
        super().set_data(data)
        self._init_vizrank()
        self.init_projection()

    def use_context(self):
        self.model_selected.clear()
        self.model_other.clear()
        if self.data is not None and len(self.selected_vars):
            d, selected = self.data.domain, [v[0] for v in self.selected_vars]
            self.model_selected[:] = [d[name] for name in selected]
            self.model_other[:] = [d[attr.name] for attr in
                                   self.primitive_variables
                                   if attr.name not in selected]
        elif self.data is not None:
            d, variables = self.data.domain, self.primitive_variables
            class_var = [variables.pop(variables.index(d.class_var))] \
                if d.class_var in variables else []
            self.model_selected[:] = variables[:5]
            self.model_other[:] = variables[5:] + class_var

    def _init_vizrank(self):
        is_enabled = self.data is not None and \
            len(self.primitive_variables) > 3 and \
            self.attr_color is not None and \
            not np.isnan(self.data.get_column_view(
                self.attr_color)[0].astype(float)).all() and \
            len(self.data[self.valid_data]) > 1 and \
            np.all(np.nan_to_num(np.nanstd(self.data.X, 0)) != 0)
        self.btn_vizrank.setEnabled(is_enabled)
        if is_enabled:
            self.vizrank.initialize()

    def check_data(self):
        super().check_data()
        if self.data is not None:
            domain = self.data.domain
            vars_ = chain(domain.variables, domain.metas)
            n_vars = sum(v.is_primitive() for v in vars_)
            if len(self.primitive_variables) < n_vars:
                self.Warning.removed_vars()

    def init_attr_values(self):
        super().init_attr_values()
        self.selected_vars = []

    def _manual_move(self, anchor_idx, x, y):
        angle = np.arctan2(y, x)
        super()._manual_move(anchor_idx, np.cos(angle), np.sin(angle))

    def _send_components_x(self):
        components_ = super()._send_components_x()
        angle = np.arctan2(*components_[::-1])
        return np.row_stack((components_, angle))

    def _send_components_metas(self):
        return np.vstack((super()._send_components_metas(), ["angle"]))

    def clear(self):
        super().clear()
        self.projector = RadViz()

    @classmethod
    def migrate_context(cls, context, version):
        if version < 2:
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


if __name__ == "__main__":  # pragma: no cover
    data = Table("heart_disease")
    WidgetPreview(OWRadviz).run(set_data=data, set_subset_data=data[::10])
