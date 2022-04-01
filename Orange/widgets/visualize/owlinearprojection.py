"""
Linear Projection widget
------------------------
"""

from itertools import islice, permutations, chain
from math import factorial

import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score

from AnyQt.QtGui import QStandardItem, QPalette
from AnyQt.QtCore import Qt, QRectF, QLineF, pyqtSignal as Signal

import pyqtgraph as pg

from Orange.data import Table, Domain
from Orange.preprocess import Normalize
from Orange.preprocess.score import ReliefF, RReliefF
from Orange.projection import PCA, LDA, LinearProjector
from Orange.util import Enum
from Orange.widgets import gui, report
from Orange.widgets.gui import OWComponent
from Orange.widgets.settings import Setting, ContextSetting, SettingProvider
from Orange.widgets.utils.plot import variables_selection
from Orange.widgets.utils.plot.owplotgui import VariableSelectionModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.utils import VizRankDialog
from Orange.widgets.visualize.utils.component import OWGraphWithAnchors
from Orange.widgets.visualize.utils.plotutils import AnchorItem
from Orange.widgets.visualize.utils.widget import OWAnchorProjectionWidget
from Orange.widgets.widget import Msg


MAX_LABEL_LEN = 20


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
        max_n_attrs = len(master.model_selected)
        self.n_attrs_spin = gui.spin(
            box, self, "n_attrs", 3, max_n_attrs, label="Number of variables: ",
            controlWidth=50, alignment=Qt.AlignRight, callback=self._n_attrs_changed)
        gui.rubber(box)
        self.last_run_n_attrs = None
        self.attr_color = master.attr_color
        self.attrs = []

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
        if not n_all_attrs:
            return 0
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
        data = master.data
        domain = Domain([self.attrs[i] for i in state], data.domain.class_vars)
        projection = master.projector(data.transform(domain))
        ec = projection(data).X
        y = column_data(data, self.attr_color, dtype=float)
        if ec.shape[0] < self.minK:
            return None
        n_neighbors = min(self.minK, len(ec) - 1)
        knn = NearestNeighbors(n_neighbors=n_neighbors).fit(ec)
        ind = knn.kneighbors(return_distance=False)
        # pylint: disable=invalid-unary-operand-type
        if self.attr_color.is_discrete:
            return -np.sum(y[ind] == y.reshape(-1, 1)) / n_neighbors / len(y)
        return -r2_score(y, np.mean(y[ind], axis=1)) * (len(y) / len(data))

    def bar_length(self, score):
        return max(0, -score)

    def _score_heuristic(self):
        def normalized(a):
            span = np.max(a, axis=0) - np.min(a, axis=0)
            span[span == 0] = 1
            return (a - np.mean(a, axis=0)) / span

        domain = self.master.data.domain
        attr_color = self.master.attr_color
        domain = Domain(
            attributes=[v for v in chain(domain.variables, domain.metas)
                        if v.is_continuous and v is not attr_color],
            class_vars=attr_color
        )
        data = self.master.data.transform(domain).copy()
        with data.unlocked():
            data.X = normalized(data.X)
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
        if not selected.indexes():
            return
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
        return self.master.placement == Placement.Circular

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
                anchor = AnchorItem(line=QLineF(0, 0, *point))
                anchor._label.setToolTip(f"<b>{label}</b>")
                label = label[:MAX_LABEL_LEN - 3] + "..." if len(label) > MAX_LABEL_LEN else label
                anchor.setText(label)
                anchor.setFont(self.parameter_setter.anchor_font)

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
                anchor.setFont(self.parameter_setter.anchor_font)

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
            color = self.plot_widget.palette().color(QPalette.Disabled, QPalette.Text)
            pen = pg.mkPen(color, width=1, cosmetic=True)
            self.circle_item.setPen(pen)


Placement = Enum("Placement", dict(Circular=0, LDA=1, PCA=2), type=int,
                 qualname="Placement")


class OWLinearProjection(OWAnchorProjectionWidget):
    name = "Linear Projection"
    description = "A multi-axis projection of data onto " \
                  "a two-dimensional plane."
    icon = "icons/LinearProjection.svg"
    priority = 240
    keywords = []

    Projection_name = {Placement.Circular: "Circular Placement",
                       Placement.LDA: "Linear Discriminant Analysis",
                       Placement.PCA: "Principal Component Analysis"}

    settings_version = 6

    placement = Setting(Placement.Circular)
    selected_vars = ContextSetting([])
    vizrank = SettingProvider(LinearProjectionVizRank)
    GRAPH_CLASS = OWLinProjGraph
    graph = SettingProvider(OWLinProjGraph)

    class Error(OWAnchorProjectionWidget.Error):
        no_cont_features = Msg("Plotting requires numeric features")

    def _add_controls(self):
        box = gui.vBox(self.controlArea, box="Features")
        self._add_controls_variables(box)
        self._add_controls_placement(box)
        super()._add_controls()
        self.gui.add_control(
            self._effects_box, gui.hSlider, "Hide radius:", master=self.graph,
            value="hide_radius", minValue=0, maxValue=100, step=10,
            createLabel=False, callback=self.__radius_slider_changed
        )

    def _add_controls_variables(self, box):
        self.model_selected = VariableSelectionModel(self.selected_vars)
        variables_selection(box, self, self.model_selected)
        self.model_selected.selection_changed.connect(
            self.__model_selected_changed)
        self.vizrank, self.btn_vizrank = LinearProjectionVizRank.add_vizrank(
            None, self, "Suggest Features", self.__vizrank_set_attrs)
        box.layout().addWidget(self.btn_vizrank)

    def _add_controls_placement(self, box):
        self.radio_placement = gui.radioButtonsInBox(
            box, self, "placement",
            btnLabels=[self.Projection_name[x] for x in Placement],
            callback=self.__placement_radio_changed
        )

    def _add_buttons(self):
        self.gui.box_zoom_select(self.buttonsArea)
        gui.auto_send(self.buttonsArea, self, "auto_commit")

    @property
    def continuous_variables(self):
        if self.data is None or self.data.domain is None:
            return []
        dom = self.data.domain
        return [v for v in chain(dom.variables, dom.metas) if v.is_continuous]

    @property
    def effective_variables(self):
        return self.selected_vars

    @property
    def effective_data(self):
        cvs = None
        if self.placement == Placement.LDA:
            cvs = self.data.domain.class_vars
        return self.data.transform(Domain(self.effective_variables, cvs))

    def __vizrank_set_attrs(self, attrs):
        if not attrs:
            return
        self.selected_vars[:] = attrs
        # Ugly, but the alternative is to have yet another signal to which
        # the view will have to connect
        self.model_selected.selection_changed.emit()

    def __model_selected_changed(self):
        self.projection = None
        self._check_options()
        self.init_projection()
        self.setup_plot()
        self.commit.deferred()

    def __placement_radio_changed(self):
        self.controls.graph.hide_radius.setEnabled(
            self.placement != Placement.Circular)
        self.projection = self.projector = None
        self._init_vizrank()
        self.init_projection()
        self.setup_plot()
        self.commit.deferred()

    def __radius_slider_changed(self):
        self.graph.update_radius()

    def colors_changed(self):
        super().colors_changed()
        self._init_vizrank()

    @OWAnchorProjectionWidget.Inputs.data
    def set_data(self, data):
        super().set_data(data)
        self._check_options()
        self._init_vizrank()
        self.init_projection()

    def _check_options(self):
        buttons = self.radio_placement.buttons
        for btn in buttons:
            btn.setEnabled(True)

        if self.data is not None:
            has_discrete_class = self.data.domain.has_discrete_class
            if not has_discrete_class or len(np.unique(self.data.Y)) < 3:
                buttons[Placement.LDA].setEnabled(False)
                if self.placement == Placement.LDA:
                    self.placement = Placement.Circular

        self.controls.graph.hide_radius.setEnabled(
            self.placement != Placement.Circular)

    def _init_vizrank(self):
        is_enabled, msg = False, ""
        if self.data is None:
            msg = "There is no data."
        elif self.attr_color is None:
            msg = "Color variable has to be selected"
        elif self.attr_color.is_continuous and \
                self.placement == Placement.LDA:
            msg = "Suggest Features does not work for Linear " \
                  "Discriminant Analysis Projection when " \
                  "continuous color variable is selected."
        elif len([v for v in self.continuous_variables
                  if v is not self.attr_color]) < 3:
            msg = "Not enough available continuous variables"
        elif np.sum(np.all(np.isfinite(self.data.X), axis=1)) < 2:
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
        self.selected_vars[:] = self.continuous_variables[:3]
        self.model_selected[:] = self.continuous_variables

    def init_projection(self):
        if self.placement == Placement.Circular:
            self.projector = CircularPlacement()
        elif self.placement == Placement.LDA:
            self.projector = LDA(solver="eigen", n_components=2)
        elif self.placement == Placement.PCA:
            self.projector = PCA(n_components=2)
            self.projector.component = 2
            self.projector.preprocessors = PCA.preprocessors + [Normalize()]

        super().init_projection()

    def get_coordinates_data(self):
        def normalized(a):
            span = np.max(a, axis=0) - np.min(a, axis=0)
            span[span == 0] = 1
            return (a - np.mean(a, axis=0)) / span

        embedding = self.get_embedding()
        if embedding is None:
            return None, None
        norm_emb = normalized(embedding[self.valid_data])
        return (norm_emb.ravel(), np.zeros(len(norm_emb), dtype=float)) \
            if embedding.shape[1] == 1 else norm_emb.T

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
        if version < 5:
            if "placement" in settings_ and \
                    settings_["placement"] not in Placement:
                settings_["placement"] = Placement.Circular

    @classmethod
    def migrate_context(cls, context, version):
        values = context.values
        if version < 2:
            domain = context.ordered_domain
            c_domain = [t for t in context.ordered_domain if t[1] == 2]
            d_domain = [t for t in context.ordered_domain if t[1] == 1]
            for d, old_val, new_val in ((domain, "color_index", "attr_color"),
                                        (d_domain, "shape_index", "attr_shape"),
                                        (c_domain, "size_index", "attr_size")):
                index = context.values[old_val][0] - 1
                values[new_val] = (d[index][0], d[index][1] + 100) \
                    if 0 <= index < len(d) else None
        if version < 3:
            values["graph"] = {
                "attr_color": values["attr_color"],
                "attr_shape": values["attr_shape"],
                "attr_size": values["attr_size"]
            }
        if version == 3:
            values["attr_color"] = values["graph"]["attr_color"]
            values["attr_size"] = values["graph"]["attr_size"]
            values["attr_shape"] = values["graph"]["attr_shape"]
            values["attr_label"] = values["graph"]["attr_label"]
        if version < 6 and "selected_vars" in values:
            values["selected_vars"] = (values["selected_vars"], -3)

    # for backward compatibility with settings < 6, pull the enum from global
    # namespace into class
    Placement = Placement


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


class CircularPlacement(LinearProjector):
    def get_components(self, X, Y):
        # Return circular axes for linear projection
        n_axes = X.shape[1]
        if n_axes == 1:
            axes_angle = [0]
        elif n_axes == 2:
            axes_angle = [0, np.pi / 2]
        else:
            axes_angle = np.linspace(0, 2 * np.pi, n_axes,
                                     endpoint=False)
        return np.vstack((np.cos(axes_angle), np.sin(axes_angle)))


if __name__ == "__main__":  # pragma: no cover
    iris = Table("iris")
    WidgetPreview(OWLinearProjection).run(set_data=iris,
                                          set_subset_data=iris[::10])
