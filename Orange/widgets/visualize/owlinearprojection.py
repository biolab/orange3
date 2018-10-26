"""
Linear Projection widget
------------------------
"""

from itertools import islice, permutations, chain
from math import factorial

import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score

from AnyQt.QtWidgets import QApplication, QSizePolicy
from AnyQt.QtGui import QStandardItem, QColor
from AnyQt.QtCore import Qt, QRectF, QLineF, pyqtSignal as Signal

import pyqtgraph as pg

from Orange.data import Table, Domain, StringVariable
from Orange.preprocess import Normalize
from Orange.preprocess.score import ReliefF, RReliefF
from Orange.projection import PCA
from Orange.util import Enum
from Orange.widgets import gui, report
from Orange.widgets.gui import OWComponent
from Orange.widgets.settings import Setting, ContextSetting, SettingProvider
from Orange.widgets.utils import vartype
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.plot import VariablesSelection
from Orange.widgets.visualize.utils import VizRankDialog
from Orange.widgets.visualize.utils.component import OWGraphWithAnchors
from Orange.widgets.visualize.utils.plotutils import AnchorItem
from Orange.widgets.visualize.utils.widget import OWAnchorProjectionWidget
from Orange.widgets.widget import Input, Msg


class LinearProjectionVizRank(VizRankDialog, OWComponent):
    captionTitle = "Score Plots"
    n_attrs = Setting(3)
    minK = 10

    attrsSelected = Signal([])
    _AttrRole = next(gui.OrangeUserRole)

    def __init__(self, master):
        # Add the spin box for a number of attributes to take into account.
        VizRankDialog.__init__(self, master)
        OWComponent.__init__(self, master)

        box = gui.hBox(self)
        max_n_attrs = len(master.model_selected) + len(master.model_other)
        self.n_attrs_spin = gui.spin(
            box, self, "n_attrs", 3, max_n_attrs, label="Number of variables: ",
            controlWidth=50, alignment=Qt.AlignRight, callback=self._n_attrs_changed)
        gui.rubber(box)
        self.last_run_n_attrs = None
        self.attr_color = master.attr_color

    def initialize(self):
        super().initialize()
        self.attr_color = self.master.attr_color

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

    def check_preconditions(self):
        master = self.master
        if not super().check_preconditions():
            return False
        elif not master.btn_vizrank.isEnabled():
            return False
        n_cont_var = len([v for v in master.continuous_variables
                          if v is not master.attr_color])
        self.n_attrs_spin.setMaximum(n_cont_var)
        return True

    def state_count(self):
        n_all_attrs = len(self.attrs)
        n_attrs = self.n_attrs
        return factorial(n_all_attrs) // (2 * factorial(n_all_attrs - n_attrs) * n_attrs)

    def iterate_states(self, state):
        if state is None:  # on the first call, compute order
            self.attrs = self._score_heuristic()
            state = list(range(self.n_attrs))
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
                    break

        for c in combinations(len(self.attrs), state):
            for p in islice(permutations(c[1:]), factorial(len(c) - 1) // 2):
                yield (c[0], ) + p

    def compute_score(self, state):
        master = self.master
        _, ec, _ = master.prepare_projection_data(
            [self.attrs[i] for i in state])
        y = column_data(master.data, self.attr_color, dtype=float)
        if ec.shape[0] < self.minK:
            return None
        n_neighbors = min(self.minK, len(ec) - 1)
        knn = NearestNeighbors(n_neighbors=n_neighbors).fit(ec)
        ind = knn.kneighbors(return_distance=False)
        if self.attr_color.is_discrete:
            return -np.sum(y[ind] == y.reshape(-1, 1)) / n_neighbors / len(y)
        return -r2_score(y, np.mean(y[ind], axis=1)) * (len(y) / len(master.data))

    def bar_length(self, score):
        return max(0, -score)

    def _score_heuristic(self):
        def normalized(col):
            if not col.size:
                return col.copy()
            col_min, col_max = np.nanmin(col), np.nanmax(col)
            if np.isnan(col_min):
                return col.copy()
            span = col_max - col_min
            return (col - col_min) / (span or 1)
        domain = self.master.data.domain
        attr_color = self.master.attr_color
        domain = Domain(
            attributes=[v for v in chain(domain.variables, domain.metas)
                        if v.is_continuous and v is not attr_color],
            class_vars=attr_color
        )
        data = self.master.data.transform(domain)
        for i, col in enumerate(data.X.T):
            data.X.T[i] = normalized(col)
        relief = ReliefF if attr_color.is_discrete else RReliefF
        weights = relief(n_iterations=100, k_nearest=self.minK)(data)
        results = sorted(zip(weights, domain.attributes), key=lambda x: (-x[0], x[1].name))
        return [attr for _, attr in results]

    def row_for_state(self, score, state):
        attrs = [self.attrs[i] for i in state]
        item = QStandardItem(", ".join(a.name for a in attrs))
        item.setData(attrs, self._AttrRole)
        return [item]

    def on_selection_changed(self, selected, deselected):
        attrs = selected.indexes()[0].data(self._AttrRole)
        self.selectionChanged.emit([attrs])

    def _n_attrs_changed(self):
        if self.n_attrs != self.last_run_n_attrs or self.saved_state is None:
            self.button.setText("Start")
        else:
            self.button.setText("Continue")
        self.button.setEnabled(self.check_preconditions())


class OWLinProjGraph(OWGraphWithAnchors):
    hide_radius = Setting(0)

    @property
    def always_show_axes(self):
        return self.master.placement == self.master.Placement.Circular

    @property
    def scaled_radius(self):
        return self.hide_radius / 100 + 1e-5

    def update_radius(self):
        self.update_circle()
        self.update_anchors()

    def set_view_box_range(self):
        def min_max(a, b):
            return (min(np.amin(a), np.amin(b), -1.05),
                    max(np.amax(a), np.amax(b), 1.05))

        points, _ = self.master.get_anchors()
        coords = self.master.get_coordinates_data()
        if points is None or coords is None:
            return

        min_x, max_x = min_max(points[:, 0], coords[0])
        min_y, max_y = min_max(points[:, 1], coords[1])
        rect = QRectF(min_x, min_y, max_x - min_x, max_y - min_y)
        self.view_box.setRange(rect, padding=0.025)

    def update_anchors(self):
        points, labels = self.master.get_anchors()
        if points is None:
            return
        r = self.scaled_radius * np.max(np.linalg.norm(points, axis=1))
        if self.anchor_items is None:
            self.anchor_items = []
            for point, label in zip(points, labels):
                anchor = AnchorItem(line=QLineF(0, 0, *point), text=label)
                visible = self.always_show_axes or np.linalg.norm(point) > r
                anchor.setVisible(visible)
                anchor.setPen(pg.mkPen((100, 100, 100)))
                self.plot_widget.addItem(anchor)
                self.anchor_items.append(anchor)
        else:
            for anchor, point, label in zip(self.anchor_items, points, labels):
                anchor.setLine(QLineF(0, 0, *point))
                visible = self.always_show_axes or np.linalg.norm(point) > r
                anchor.setVisible(visible)

    def update_circle(self):
        super().update_circle()

        if self.always_show_axes:
            self.plot_widget.removeItem(self.circle_item)
            self.circle_item = None

        if self.circle_item is not None:
            points, _ = self.master.get_anchors()
            if points is None:
                return

            r = self.scaled_radius * np.max(np.linalg.norm(points, axis=1))
            self.circle_item.setRect(QRectF(-r, -r, 2 * r, 2 * r))
            pen = pg.mkPen(QColor(Qt.lightGray), width=1, cosmetic=True)
            self.circle_item.setPen(pen)


class OWLinearProjection(OWAnchorProjectionWidget):
    name = "Linear Projection"
    description = "A multi-axis projection of data onto " \
                  "a two-dimensional plane."
    icon = "icons/LinearProjection.svg"
    priority = 240
    keywords = []

    class Inputs(OWAnchorProjectionWidget.Inputs):
        projection_input = Input("Projection", Table)

    Placement = Enum("Placement", dict(Circular=0, LDA=1, PCA=2, Projection=3),
                     type=int, qualname="OWLinearProjection.Placement")

    Component_name = {Placement.Circular: "C", Placement.LDA: "LD",
                      Placement.PCA: "PC"}
    Variable_name = {Placement.Circular: "circular",
                     Placement.LDA: "lda",
                     Placement.PCA: "pca",
                     Placement.Projection: "projection"}
    Projection_name = {Placement.Circular: "Circular Placement",
                       Placement.LDA: "Linear Discriminant Analysis",
                       Placement.PCA: "Principal Component Analysis",
                       Placement.Projection: "Use input projection"}

    settings_version = 4

    placement = Setting(Placement.Circular)
    selected_vars = ContextSetting([])
    vizrank = SettingProvider(LinearProjectionVizRank)
    GRAPH_CLASS = OWLinProjGraph
    graph = SettingProvider(OWLinProjGraph)

    class Warning(OWAnchorProjectionWidget.Warning):
        not_enough_comp = Msg("Input projection has less than two components")
        trivial_components = Msg(
            "All components of the PCA are trivial (explain zero variance). "
            "Input data is constant (or near constant).")

    class Error(OWAnchorProjectionWidget.Error):
        no_cont_features = Msg("Plotting requires numeric features")
        proj_and_domain_match = Msg("Projection and Data domains do not match")

    def __init__(self):
        self.model_selected = VariableListModel(enable_dnd=True)
        self.model_selected.rowsInserted.connect(self.__model_selected_changed)
        self.model_selected.rowsRemoved.connect(self.__model_selected_changed)
        self.model_other = VariableListModel(enable_dnd=True)

        self.vizrank, self.btn_vizrank = LinearProjectionVizRank.add_vizrank(
            None, self, "Suggest Features", self.__vizrank_set_attrs)

        super().__init__()
        self.projection_input = None
        self.variables = None

    def _add_controls(self):
        self._add_controls_variables()
        self._add_controls_placement()
        super()._add_controls()
        self.graph.gui.add_control(
            self._effects_box, gui.hSlider, "Hide radius:", master=self.graph,
            value="hide_radius", minValue=0, maxValue=100, step=10,
            createLabel=False, callback=self.__radius_slider_changed
        )
        self.controlArea.layout().removeWidget(self.control_area_stretch)
        self.control_area_stretch.setParent(None)

    def _add_controls_variables(self):
        self.variables_selection = VariablesSelection(
            self, self.model_selected, self.model_other, self.controlArea
        )
        self.variables_selection.add_remove.layout().addWidget(
            self.btn_vizrank
        )

    def _add_controls_placement(self):
        box = gui.widgetBox(
            self.controlArea, True,
            sizePolicy=(QSizePolicy.Minimum, QSizePolicy.Maximum)
        )
        self.radio_placement = gui.radioButtonsInBox(
            box, self, "placement",
            btnLabels=[self.Projection_name[x] for x in self.Placement],
            callback=self.__placement_radio_changed
        )

    @property
    def continuous_variables(self):
        if self.data is None or self.data.domain is None:
            return []
        dom = self.data.domain
        return [v for v in chain(dom.variables, dom.metas) if v.is_continuous]

    def __vizrank_set_attrs(self, attrs):
        if not attrs:
            return
        self.model_selected[:] = attrs[:]
        self.model_other[:] = [var for var in self.continuous_variables
                               if var not in attrs]

    def __model_selected_changed(self):
        self.selected_vars = [(var.name, vartype(var)) for var
                              in self.model_selected]
        self.projection = None
        self.variables = None
        self._check_options()
        self.setup_plot()
        self.commit()

    def __placement_radio_changed(self):
        self.variables_selection.set_enabled(
            self.placement in [self.Placement.Circular, self.Placement.LDA])
        self.controls.graph.hide_radius.setEnabled(
            self.placement != self.Placement.Circular)
        self.projection = None
        self.variables = None
        self._init_vizrank()
        self.setup_plot()
        self.commit()

    def __radius_slider_changed(self):
        self.graph.update_radius()

    def colors_changed(self):
        super().colors_changed()
        self._init_vizrank()

    def set_data(self, data):
        super().set_data(data)
        if self.data is not None and len(self.selected_vars):
            d, selected = self.data.domain, [v[0] for v in self.selected_vars]
            self.model_selected[:] = [d[attr] for attr in selected]
            self.model_other[:] = [d[attr.name] for attr in
                                   self.continuous_variables
                                   if attr.name not in selected]
        elif self.data is not None:
            self.model_selected[:] = self.continuous_variables[:3]
            self.model_other[:] = self.continuous_variables[3:]

        self._check_options()
        self._init_vizrank()

    def _check_options(self):
        buttons = self.radio_placement.buttons
        for btn in buttons:
            btn.setEnabled(True)
        if self.data is not None:
            has_discrete_class = self.data.domain.has_discrete_class
            if not has_discrete_class or len(np.unique(self.data.Y)) < 2:
                buttons[self.Placement.LDA].setEnabled(False)
                if self.placement == self.Placement.LDA:
                    self.placement = self.Placement.Circular
            if not self.projection_input:
                buttons[self.Placement.Projection].setEnabled(False)
                if self.placement == self.Placement.Projection:
                    self.placement = self.Placement.Circular

        self.variables_selection.set_enabled(
            self.placement in [self.Placement.Circular, self.Placement.LDA])
        self.controls.graph.hide_radius.setEnabled(
            self.placement != self.Placement.Circular)

    def _init_vizrank(self):
        is_enabled, msg = False, ""
        if self.data is None:
            msg = "There is no data."
        elif self.placement not in [self.Placement.Circular,
                                    self.Placement.LDA]:
            msg = "Suggest Features works only for Circular and " \
                  "Linear Discriminant Analysis Projection"
        elif self.attr_color is None:
            msg = "Color variable has to be selected"
        elif self.attr_color.is_continuous and \
                self.placement == self.Placement.LDA:
            msg = "Suggest Features does not work for Linear " \
                  "Discriminant Analysis Projection when " \
                  "continuous color variable is selected."
        elif len([v for v in self.continuous_variables
                  if v is not self.attr_color]) < 3:
            msg = "Not enough available continuous variables"
        elif len(self.data[self.valid_data]) < 2:
            msg = "Not enough valid data instances"
        else:
            is_enabled = not np.isnan(self.data.get_column_view(
                self.attr_color)[0].astype(float)).all()
        self.btn_vizrank.setToolTip(msg)
        self.btn_vizrank.setEnabled(is_enabled)
        if is_enabled:
            self.vizrank.initialize()

    def check_data(self):
        def error(err):
            err()
            self.data = None

        super().check_data()
        if self.data is not None:
            if not len(self.continuous_variables):
                error(self.Error.no_cont_features)

    def init_attr_values(self):
        super().init_attr_values()
        self.selected_vars = []

    @Inputs.projection_input
    def set_projection(self, projection):
        self.Warning.not_enough_comp.clear()
        if projection and len(projection) < 2:
            self.Warning.not_enough_comp()
            projection = None
        if projection is not None:
            self.placement = self.Placement.Projection
        self.projection_input = projection
        self._check_options()

    def get_embedding(self):
        self.valid_data = None
        if self.data is None or not self.variables:
            return None

        if self.placement == self.Placement.PCA:
            self.valid_data, ec, self.projection = self._get_pca()
            self.variables = self._pca.orig_domain.attributes
        else:
            self.valid_data, ec, self.projection = \
                self.prepare_projection_data(self.variables)

        self.Error.no_valid_data.clear()
        if self.valid_data is None or not sum(self.valid_data) or \
                self.projection is None or ec is None:
            self.Error.no_valid_data()
            return None

        embedding = np.zeros((len(self.data), 2), dtype=np.float)
        embedding[self.valid_data] = ec
        return embedding

    def prepare_projection_data(self, variables):
        def projection(_vars):
            attrs = self.projection_input.domain.attributes
            if set(attrs).issuperset(_vars):
                return self.projection_input[:2, _vars].X
            elif set(f.name for f in attrs).issuperset(f.name for f in _vars):
                return self.projection_input[:2, [f.name for f in _vars]].X
            else:
                self.Error.proj_and_domain_match()
                return None

        def get_axes(_vars):
            self.Error.proj_and_domain_match.clear()
            if self.placement == self.Placement.Circular:
                return LinProj.defaultaxes(len(_vars))
            elif self.placement == self.Placement.LDA:
                return self._get_lda(self.data, _vars)
            elif self.placement == self.Placement.Projection and \
                    self.projection_input is not None:
                return projection(_vars)
            else:
                return None

        coords = np.vstack(column_data(self.data, v, float) for v in variables)
        axes = get_axes(variables)
        if axes is None:
            return None, None, None

        valid_mask = ~np.isnan(coords).any(axis=0)
        X, Y = np.dot(axes, coords[:, valid_mask])
        if X.size and Y.size:
            X = normalized(X)
            Y = normalized(Y)
        return valid_mask, np.stack((X, Y), axis=1), axes.T

    def get_anchors(self):
        if self.projection is None:
            return None, None
        return self.projection, [v.name for v in self.variables]

    def setup_plot(self):
        self.init_projection_variables()
        super().setup_plot()

    def init_projection_variables(self):
        self.variables = None
        if self.data is None:
            return

        if self.placement in [self.Placement.Circular, self.Placement.LDA]:
            self.variables = self.model_selected[:]
        elif self.placement == self.Placement.Projection:
            self.variables = self.model_selected[:] + self.model_other[:]
        elif self.placement == self.Placement.PCA:
            self.variables = [var for var in self.data.domain.attributes
                              if var.is_continuous]

    def _get_lda(self, data, variables):
        data = data.transform(Domain(variables, data.domain.class_vars))
        lda = LinearDiscriminantAnalysis(solver='eigen', n_components=2)
        lda.fit(data.X, data.Y)
        scalings = lda.scalings_[:, :2].T
        if scalings.shape == (1, 1):
            scalings = np.array([[1.], [0.]])
        return scalings

    def _get_pca(self):
        pca_projector = PCA(n_components=2)
        pca_projector.component = 2
        pca_projector.preprocessors = PCA.preprocessors + [Normalize()]

        pca = pca_projector(self.data)
        variance_ratio = pca.explained_variance_ratio_
        cumulative = np.cumsum(variance_ratio)

        self._pca = pca
        if not np.isfinite(cumulative[-1]):
            self.Warning.trivial_components()

        coords = pca(self.data).X
        valid_mask = ~np.isnan(coords).any(axis=1)
        # scale axes
        max_radius = np.min([np.abs(np.min(coords, axis=0)),
                             np.max(coords, axis=0)])
        axes = pca.components_.T.copy()
        axes *= max_radius / np.max(np.linalg.norm(axes, axis=1))
        return valid_mask, coords, axes

    def send_components(self):
        components = None
        if self.data is not None and self.valid_data is not None and \
                self.projection is not None:
            if self.placement in [self.Placement.Circular, self.Placement.LDA]:
                axes = self.projection
                attrs = self.model_selected
            elif self.placement == self.Placement.PCA:
                axes = self._pca.components_.T
                attrs = self._pca.orig_domain.attributes
            if self.placement != self.Placement.Projection:
                meta_attrs = [StringVariable(name='component')]
                metas = np.array(
                    [["{}{}".format(self.Component_name[self.placement], i + 1)
                      for i in range(axes.shape[1])]], dtype=object).T
                components = Table(Domain(attrs, metas=meta_attrs),
                                   axes.T, metas=metas)
                components.name = self.data.name
            else:
                components = self.projection_input
        self.Outputs.components.send(components)

    def _get_projection_variables(self):
        pn = self.Variable_name[self.placement]
        self.embedding_variables_names = ("{}-x".format(pn), "{}-y".format(pn))
        return super()._get_projection_variables()

    def _get_send_report_caption(self):
        def projection_name():
            return self.Projection_name[self.placement]

        return report.render_items_vert((
            ("Projection", projection_name()),
            ("Color", self._get_caption_var_name(self.attr_color)),
            ("Label", self._get_caption_var_name(self.attr_label)),
            ("Shape", self._get_caption_var_name(self.attr_shape)),
            ("Size", self._get_caption_var_name(self.attr_size)),
            ("Jittering", self.graph.jitter_size != 0 and
             "{} %".format(self.graph.jitter_size))))

    def clear(self):
        self.variables = None
        if self.model_selected:
            self.model_selected.clear()
        if self.model_other:
            self.model_other.clear()
        super().clear()

    @classmethod
    def migrate_settings(cls, settings_, version):
        if version < 2:
            settings_["point_width"] = settings_["point_size"]
        if version < 3:
            settings_graph = {}
            settings_graph["jitter_size"] = settings_["jitter_value"]
            settings_graph["point_width"] = settings_["point_width"]
            settings_graph["alpha_value"] = settings_["alpha_value"]
            settings_graph["class_density"] = settings_["class_density"]
            settings_["graph"] = settings_graph
        if version < 4:
            if "radius" in settings_:
                settings_["graph"]["hide_radius"] = settings_["radius"]
            if "selection_indices" in settings_ and \
                    settings_["selection_indices"] is not None:
                selection = settings_["selection_indices"]
                settings_["selection"] = [(i, 1) for i, selected in
                                          enumerate(selection) if selected]

    @classmethod
    def migrate_context(cls, context, version):
        if version < 2:
            domain = context.ordered_domain
            c_domain = [t for t in context.ordered_domain if t[1] == 2]
            d_domain = [t for t in context.ordered_domain if t[1] == 1]
            for d, old_val, new_val in ((domain, "color_index", "attr_color"),
                                        (d_domain, "shape_index", "attr_shape"),
                                        (c_domain, "size_index", "attr_size")):
                index = context.values[old_val][0] - 1
                context.values[new_val] = (d[index][0], d[index][1] + 100) \
                    if 0 <= index < len(d) else None
        if version < 3:
            context.values["graph"] = {
                "attr_color": context.values["attr_color"],
                "attr_shape": context.values["attr_shape"],
                "attr_size": context.values["attr_size"]
            }
        if version == 3:
            values = context.values
            values["attr_color"] = values["graph"]["attr_color"]
            values["attr_size"] = values["graph"]["attr_size"]
            values["attr_shape"] = values["graph"]["attr_shape"]
            values["attr_label"] = values["graph"]["attr_label"]


def column_data(table, var, dtype):
    dtype = np.dtype(dtype)
    col, copy = table.get_column_view(var)
    if not isinstance(col.dtype.type, np.inexact):
        col = col.astype(float)
        copy = True
    if dtype != col.dtype:
        col = col.astype(dtype)
        copy = True

    if not copy:
        col = col.copy()
    return col


class LinProj:
    @staticmethod
    def defaultaxes(naxes):
        # Return circular axes for linear projection
        assert naxes > 0

        if naxes == 1:
            axes_angle = [0]
        elif naxes == 2:
            axes_angle = [0, np.pi / 2]
        else:
            axes_angle = np.linspace(0, 2 * np.pi, naxes, endpoint=False)

        axes = np.vstack(
            (np.cos(axes_angle),
             np.sin(axes_angle))
        )
        return axes

    @staticmethod
    def project(axes, X):
        return np.dot(axes, X)


def normalized(a):
    if not a.size:
        return a.copy()
    amin, amax = np.nanmin(a), np.nanmax(a)
    if np.isnan(amin):
        return a.copy()
    span = amax - amin
    mean = np.nanmean(a)
    return (a - mean) / (span or 1)


def main(argv=None):
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if argv:
        filename = argv[0]
    else:
        filename = "iris"

    data = Table(filename)

    app = QApplication([])
    w = OWLinearProjection()
    w.set_data(data)
    w.set_subset_data(data[::10])
    w.handleNewSignals()
    w.show()
    w.raise_()
    app.exec()
    w.set_data(None)
    w.saveSettings()
    del w


if __name__ == "__main__":
    import sys
    sys.exit(main())
