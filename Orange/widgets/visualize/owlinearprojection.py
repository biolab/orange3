"""
Linear Projection widget
------------------------
"""

from itertools import islice, permutations, chain
from math import factorial
from types import SimpleNamespace as namespace

import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score

from AnyQt.QtWidgets import QGraphicsEllipseItem, QApplication, QSizePolicy
from AnyQt.QtGui import QPen, QStandardItem
from AnyQt.QtCore import Qt, QEvent, QRectF, QLineF
from AnyQt.QtCore import pyqtSignal as Signal

import pyqtgraph as pg

from Orange.data import Table, Domain, StringVariable, ContinuousVariable
from Orange.data.sql.table import SqlTable
from Orange.preprocess import Normalize
from Orange.projection import PCA
from Orange.util import Enum
from Orange.widgets import widget, gui, settings
from Orange.widgets.gui import OWComponent
from Orange.widgets.utils.annotated_data import (
    create_annotated_table, ANNOTATED_DATA_SIGNAL_NAME, create_groups_table, get_unique_names)
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.plot import VariablesSelection
from Orange.widgets.visualize.utils import VizRankDialog
from Orange.widgets.visualize.utils.plotutils import AnchorItem
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotGraph, InteractiveViewBox
from Orange.widgets.widget import Input, Output
from Orange.canvas import report
from Orange.preprocess.score import ReliefF, RReliefF


class LinearProjectionVizRank(VizRankDialog, OWComponent):
    captionTitle = "Score Plots"
    n_attrs = settings.Setting(3)
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
        self.attr_color = master.graph.attr_color

    def initialize(self):
        super().initialize()
        self.attr_color = self.master.graph.attr_color

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
        self.n_attrs_spin.setMaximum(self.master.n_cont_var)
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
        _, ec, _ = master.prepare_plot_data([self.attrs[i] for i in state])
        y = column_data(master.data, self.attr_color, dtype=float)
        if ec.shape[0] < self.minK:
            return
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
        attr_color = self.master.graph.attr_color
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


class LinProjInteractiveViewBox(InteractiveViewBox):
    def _dragtip_pos(self):
        return 10, 10


class OWLinProjGraph(OWScatterPlotGraph):
    jitter_size = settings.Setting(0)

    def hide_axes(self):
        for axis in ["left", "bottom"]:
            self.plot_widget.hideAxis(axis)

    def update_data(self, attr_x, attr_y, reset_view=True):
        axes = self.master.plotdata.axes
        axes_x, axes_y = axes[:, 0], axes[:, 1]
        x_data, y_data = self.get_xy_data_positions(attr_x, attr_y, self.valid_data)
        f = lambda a, b: (min(np.nanmin(a), np.nanmin(b)), max(np.nanmax(a), np.nanmax(b)))
        min_x, max_x = f(axes_x, x_data)
        min_y, max_y = f(axes_y, y_data)
        self.view_box.setRange(QRectF(min_x, min_y, max_x - min_x, max_y - min_y), padding=0.025)
        self.view_box.setAspectLocked(True, 1)

        super().update_data(attr_x, attr_y, reset_view=False)
        self.hide_axes()

    def update_labels(self):
        if self.master.model_selected[:]:
            super().update_labels()

    def update_shapes(self):
        if self.master.model_selected[:]:
            super().update_shapes()


class OWLinearProjection(widget.OWWidget):
    name = "Linear Projection"
    description = "A multi-axis projection of data onto " \
                  "a two-dimensional plane."
    icon = "icons/LinearProjection.svg"
    priority = 240

    selection_indices = settings.Setting(None, schema_only=True)

    class Inputs:
        data = Input("Data", Table, default=True)
        data_subset = Input("Data Subset", Table)
        projection = Input("Projection", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)
        components = Output("Components", Table)

    Placement = Enum("Placement",
                     dict(Circular=0,
                          LDA=1,
                          PCA=2,
                          Projection=3),
                     type=int,
                     qualname="OWLinearProjection.Placement")

    Component_name = {Placement.Circular: "C", Placement.LDA: "LD", Placement.PCA: "PC"}
    Variable_name = {Placement.Circular: "circular",
                     Placement.LDA: "lda",
                     Placement.PCA: "pca",
                     Placement.Projection: "projection"}

    jitter_sizes = [0, 0.1, 0.5, 1.0, 2.0]

    settings_version = 3
    settingsHandler = settings.DomainContextHandler()

    variable_state = settings.ContextSetting({})
    placement = settings.Setting(Placement.Circular)
    radius = settings.Setting(0)
    auto_commit = settings.Setting(True)

    resolution = 256

    graph = settings.SettingProvider(OWLinProjGraph)
    ReplotRequest = QEvent.registerEventType()
    vizrank = settings.SettingProvider(LinearProjectionVizRank)
    graph_name = "graph.plot_widget.plotItem"

    class Warning(widget.OWWidget.Warning):
        no_cont_features = widget.Msg("Plotting requires numeric features")
        not_enough_components = widget.Msg("Input projection has less than 2 components")
        trivial_components = widget.Msg(
            "All components of the PCA are trivial (explain 0 variance). "
            "Input data is constant (or near constant).")

    class Error(widget.OWWidget.Error):
        proj_and_domain_match = widget.Msg("Projection and Data domains do not match")
        no_valid_data = widget.Msg("No projection due to invalid data")

    def __init__(self):
        super().__init__()

        self.data = None
        self.projection = None
        self.subset_data = None
        self._subset_mask = None
        self._selection = None
        self.__replot_requested = False
        self.n_cont_var = 0
        #: Remember the saved state to restore
        self.__pending_selection_restore = self.selection_indices
        self.selection_indices = None

        self.variable_x = None
        self.variable_y = None

        box = gui.vBox(self.mainArea, True, margin=0)
        self.graph = OWLinProjGraph(self, box, "Plot", view_box=LinProjInteractiveViewBox)
        box.layout().addWidget(self.graph.plot_widget)
        plot = self.graph.plot_widget

        SIZE_POLICY = (QSizePolicy.Minimum, QSizePolicy.Maximum)

        self.variables_selection = VariablesSelection()
        self.model_selected = VariableListModel(enable_dnd=True)
        self.model_other = VariableListModel(enable_dnd=True)
        self.variables_selection(self, self.model_selected, self.model_other)

        self.vizrank, self.btn_vizrank = LinearProjectionVizRank.add_vizrank(
            self.controlArea, self, "Suggest Features", self._vizrank)
        self.variables_selection.add_remove.layout().addWidget(self.btn_vizrank)

        box = gui.widgetBox(
            self.controlArea, "Placement", sizePolicy=SIZE_POLICY)
        self.radio_placement = gui.radioButtonsInBox(
            box, self, "placement",
            btnLabels=["Circular Placement",
                       "Linear Discriminant Analysis",
                       "Principal Component Analysis",
                       "Use input projection"],
            callback=self._change_placement
        )

        self.viewbox = plot.getViewBox()
        self.replot = None

        g = self.graph.gui
        box = g.point_properties_box(self.controlArea)
        self.models = g.points_models
        g.add_widget(g.JitterSizeSlider, box)
        box.setSizePolicy(*SIZE_POLICY)

        box = gui.widgetBox(self.controlArea, "Hide axes", sizePolicy=SIZE_POLICY)
        self.rslider = gui.hSlider(
            box, self, "radius", minValue=0, maxValue=100,
            step=5, label="Radius", createLabel=False, ticks=True,
            callback=self.update_radius)
        self.rslider.setTickInterval(0)
        self.rslider.setPageStep(10)

        box = gui.vBox(self.controlArea, "Plot Properties")
        box.setSizePolicy(*SIZE_POLICY)

        g.add_widgets([g.ShowLegend,
                       g.ToolTipShowsAll,
                       g.ClassDensity,
                       g.LabelOnlySelected], box)

        box = self.graph.box_zoom_select(self.controlArea)
        box.setSizePolicy(*SIZE_POLICY)

        self.icons = gui.attributeIconDict

        p = self.graph.plot_widget.palette()
        self.graph.set_palette(p)
        gui.auto_commit(self.controlArea, self, "auto_commit", "Send Selection",
                        auto_label="Send Automatically")
        self.graph.zoom_actions(self)

        self._new_plotdata()
        self._change_placement()
        self.graph.jitter_continuous = True

    def reset_graph_data(self):
        if self.data is not None:
            self.graph.rescale_data()
            self._update_graph(reset_view=True)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        self.graph.update_tooltip(event.modifiers())

    def keyReleaseEvent(self, event):
        super().keyReleaseEvent(event)
        self.graph.update_tooltip(event.modifiers())

    def _vizrank(self, attrs):
        self.variables_selection.display_none()
        self.model_selected[:] = attrs[:]
        self.model_other[:] = [var for var in self.model_other if var not in attrs]

    def _change_placement(self):
        placement = self.placement
        p_Circular = self.Placement.Circular
        p_LDA = self.Placement.LDA
        self.variables_selection.set_enabled(placement in [p_Circular, p_LDA])
        self._vizrank_color_change()
        self.rslider.setEnabled(placement != p_Circular)
        self._setup_plot()
        self.commit()

    def _get_min_radius(self):
        return self.radius * np.max(np.linalg.norm(self.plotdata.axes, axis=1)) / 100 + 1e-5

    def update_radius(self):
        # Update the anchor/axes visibility
        pd = self.plotdata
        assert pd is not None
        if pd.hidecircle is None:
            return
        min_radius = self._get_min_radius()
        for anchor, item in zip(pd.axes, pd.axisitems):
            item.setVisible(np.linalg.norm(anchor) > min_radius)
        pd.hidecircle.setRect(QRectF(-min_radius, -min_radius, 2 * min_radius, 2 * min_radius))

    def _new_plotdata(self):
        self.plotdata = namespace(
            valid_mask=None,
            embedding_coords=None,
            axisitems=[],
            axes=[],
            variables=[],
            data=None,
            hidecircle=None
        )

    def _anchor_circle(self, variables):
        # minimum visible anchor radius (radius)
        min_radius = self._get_min_radius()
        axisitems = []
        for anchor, var in zip(self.plotdata.axes, variables[:]):
            axitem = AnchorItem(line=QLineF(0, 0, *anchor), text=var.name,)
            axitem.setVisible(np.linalg.norm(anchor) > min_radius)
            axitem.setPen(pg.mkPen((100, 100, 100)))
            axitem.setArrowVisible(True)
            self.viewbox.addItem(axitem)
            axisitems.append(axitem)

        self.plotdata.axisitems = axisitems
        if self.placement == self.Placement.Circular:
            return

        hidecircle = QGraphicsEllipseItem()
        hidecircle.setRect(QRectF(-min_radius, -min_radius, 2 * min_radius, 2 * min_radius))

        _pen = QPen(Qt.lightGray, 1)
        _pen.setCosmetic(True)
        hidecircle.setPen(_pen)

        self.viewbox.addItem(hidecircle)
        self.plotdata.hidecircle = hidecircle

    def update_colors(self):
        self._vizrank_color_change()

    def clear(self):
        # Clear/reset the widget state
        self.data = None
        self.model_selected.clear()
        self.model_other.clear()
        self._clear_plot()
        self.selection_indices = None

    def _clear_plot(self):
        self.Warning.trivial_components.clear()
        for axisitem in self.plotdata.axisitems:
            self.viewbox.removeItem(axisitem)
        if self.plotdata.hidecircle:
            self.viewbox.removeItem(self.plotdata.hidecircle)
        self._new_plotdata()
        self.graph.hide_axes()

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
        is_enabled = False
        if self.data is None:
            self.btn_vizrank.setToolTip("There is no data.")
            return
        vars = [v for v in chain(self.data.domain.variables, self.data.domain.metas) if
                v.is_primitive and v is not self.graph.attr_color]
        self.n_cont_var = len(vars)
        if self.placement not in [self.Placement.Circular, self.Placement.LDA]:
            msg = "Suggest Features works only for Circular and " \
                  "Linear Discriminant Analysis Projection"
        elif self.graph.attr_color is None:
            msg = "Color variable has to be selected"
        elif self.graph.attr_color.is_continuous and self.placement == self.Placement.LDA:
            msg = "Suggest Features does not work for Linear Discriminant Analysis Projection " \
                  "when continuous color variable is selected."
        elif len(vars) < 3:
            msg = "Not enough available continuous variables"
        else:
            is_enabled = True
            msg = ""
        self.btn_vizrank.setToolTip(msg)
        self.btn_vizrank.setEnabled(is_enabled)
        self.vizrank.stop_and_reset(is_enabled)

    @Inputs.projection
    def set_projection(self, projection):
        self.Warning.not_enough_components.clear()
        if projection and len(projection) < 2:
            self.Warning.not_enough_components()
            projection = None
        if projection is not None:
            self.placement = self.Placement.Projection
        self.projection = projection

    @Inputs.data
    def set_data(self, data):
        """
        Set the input dataset.

        Args:
            data (Orange.data.table): data instances
        """
        def sql(data):
            if isinstance(data, SqlTable):
                if data.approx_len() < 4000:
                    data = Table(data)
                else:
                    self.information("Data has been sampled")
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
            selected_keys = [key for key, (sind, _) in self.variable_state.items() if sind == 0]

            if set(selected_keys).issubset(set(state.keys())):
                pass

            if self.__pending_selection_restore is not None:
                self._selection = np.array(self.__pending_selection_restore, dtype=int)
                self.__pending_selection_restore = None

            # update the defaults state (the encoded state must contain
            # all variables in the input domain)
            state.update(self.variable_state)
            # ... and restore it with saved positions taking precedence over
            # the defaults
            selected, other = VariablesSelection.decode_var_state(
                state, [list(self.model_selected), list(self.model_other)])
            return selected, other

        self.closeContext()
        self.clear()
        self.Warning.no_cont_features.clear()
        self.information()
        data = sql(data)
        if data is not None:
            domain = data.domain
            vars = [var for var in chain(domain.variables, domain.metas) if var.is_continuous]
            if not len(vars):
                self.Warning.no_cont_features()
                data = None
        self.data = data
        self.init_attr_values()
        if data is not None and len(data):
            self._initialize(data)
            self.model_selected[:], self.model_other[:] = settings(data)
            self.vizrank.stop_and_reset()
            self.vizrank.attrs = self.data.domain.attributes if self.data is not None else []

    def _check_possible_opt(self):
        def set_enabled(is_enabled):
            for btn in self.radio_placement.buttons:
                btn.setEnabled(is_enabled)
            self.variables_selection.set_enabled(is_enabled)

        p_Circular = self.Placement.Circular
        p_LDA = self.Placement.LDA
        p_Input = self.Placement.Projection
        if self.data:
            set_enabled(True)
            domain = self.data.domain
            if not domain.has_discrete_class or len(domain.class_var.values) < 2:
                self.radio_placement.buttons[p_LDA].setEnabled(False)
                if self.placement == p_LDA:
                    self.placement = p_Circular
            if not self.projection:
                self.radio_placement.buttons[p_Input].setEnabled(False)
                if self.placement == p_Input:
                    self.placement = p_Circular
            self._setup_plot()
        else:
            self.graph.new_data(None)
            self.rslider.setEnabled(False)
            set_enabled(False)
        self.commit()

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
        if self.data is not None and self.subset_data is not None:
            # Update the plot's highlight items
            dataids = self.data.ids.ravel()
            subsetids = np.unique(self.subset_data.ids)
            self._subset_mask = np.in1d(dataids, subsetids, assume_unique=True)
        self._check_possible_opt()
        self._change_placement()
        self.commit()

    def customEvent(self, event):
        if event.type() == OWLinearProjection.ReplotRequest:
            self.__replot_requested = False
            self._setup_plot()
            self.commit()
        else:
            super().customEvent(event)

    def closeContext(self):
        self.variable_state = VariablesSelection.encode_var_state(
            [list(self.model_selected), list(self.model_other)]
        )
        super().closeContext()

    def _initialize(self, data):
        # Initialize the GUI controls from data's domain.
        vars = [v for v in chain(data.domain.metas, data.domain.attributes) if v.is_continuous]
        self.model_other[:] = vars[3:]
        self.model_selected[:] = vars[:3]

    def prepare_plot_data(self, variables):
        def projection(variables):
            if set(self.projection.domain.attributes).issuperset(variables):
                axes = self.projection[:2, variables].X
            elif set(f.name for f in
                     self.projection.domain.attributes).issuperset(f.name for f in variables):
                axes = self.projection[:2, [f.name for f in variables]].X
            else:
                self.Error.proj_and_domain_match()
                axes = None
            return axes

        def get_axes(variables):
            self.Error.proj_and_domain_match.clear()
            axes = None
            if self.placement == self.Placement.Circular:
                axes = LinProj.defaultaxes(len(variables))
            elif self.placement == self.Placement.LDA:
                axes = self._get_lda(self.data, variables)
            elif self.placement == self.Placement.Projection and self.projection:
                axes = projection(variables)
            return axes

        coords = [column_data(self.data, var, dtype=float) for var in variables]
        coords = np.vstack(coords)
        p, N = coords.shape
        assert N == len(self.data), p == len(variables)

        axes = get_axes(variables)
        if axes is None:
            return None, None, None
        assert axes.shape == (2, p)

        valid_mask = ~np.isnan(coords).any(axis=0)
        coords = coords[:, valid_mask]

        X, Y = np.dot(axes, coords)
        if X.size and Y.size:
            X = normalized(X)
            Y = normalized(Y)

        return valid_mask, np.stack((X, Y), axis=1), axes.T

    def _setup_plot(self):
        self._clear_plot()
        if self.data is None:
            return
        self.__replot_requested = False
        names = get_unique_names([v.name for v in chain(self.data.domain.variables,
                                                        self.data.domain.metas)],
                                 ["{}-x".format(self.Variable_name[self.placement]),
                                  "{}-y".format(self.Variable_name[self.placement])])
        self.variable_x = ContinuousVariable(names[0])
        self.variable_y = ContinuousVariable(names[1])
        if self.placement in [self.Placement.Circular, self.Placement.LDA]:
            variables = list(self.model_selected)
        elif self.placement == self.Placement.Projection:
            variables = self.model_selected[:] + self.model_other[:]
        elif self.placement == self.Placement.PCA:
            variables = [var for var in self.data.domain.attributes if var.is_continuous]
        if not variables:
            self.graph.new_data(None)
            return
        if self.placement == self.Placement.PCA:
            valid_mask, ec, axes = self._get_pca()
            variables = self._pca.orig_domain.attributes
        else:
            valid_mask, ec, axes = self.prepare_plot_data(variables)

        self.plotdata.variables = variables
        self.plotdata.valid_mask = valid_mask
        self.plotdata.embedding_coords = ec
        self.plotdata.axes = axes
        if any(e is None for e in (valid_mask, ec, axes)):
            return

        if not sum(valid_mask):
            self.Error.no_valid_data()
            self.graph.new_data(None, None)
            return
        self.Error.no_valid_data.clear()

        self._anchor_circle(variables=variables)
        self._plot()

    def _plot(self):
        domain = self.data.domain
        new_metas = domain.metas + (self.variable_x, self.variable_y)
        domain = Domain(attributes=domain.attributes, class_vars=domain.class_vars, metas=new_metas)
        valid_mask = self.plotdata.valid_mask
        array = np.zeros((len(self.data), 2), dtype=np.float)
        array[valid_mask] = self.plotdata.embedding_coords
        self.plotdata.data = data = self.data.transform(domain)
        data[:, self.variable_x] = array[:, 0].reshape(-1, 1)
        data[:, self.variable_y] = array[:, 1].reshape(-1, 1)
        subset_data = data[self._subset_mask & valid_mask]\
            if self._subset_mask is not None and len(self._subset_mask) else None
        self.plotdata.data = data
        self.graph.new_data(data[valid_mask], subset_data)
        if self._selection is not None:
            self.graph.selection = self._selection[valid_mask]
        self.graph.update_data(self.variable_x, self.variable_y, False)

    def _get_lda(self, data, variables):
        domain = Domain(attributes=variables, class_vars=data.domain.class_vars)
        data = data.transform(domain)
        lda = LinearDiscriminantAnalysis(solver='eigen', n_components=2)
        lda.fit(data.X, data.Y)
        scalings = lda.scalings_[:, :2].T
        if scalings.shape == (1, 1):
            scalings = np.array([[1.], [0.]])
        return scalings

    def _get_pca(self):
        data = self.data
        MAX_COMPONENTS = 2
        ncomponents = 2
        DECOMPOSITIONS = [PCA]  # TruncatedSVD
        cls = DECOMPOSITIONS[0]
        pca_projector = cls(n_components=MAX_COMPONENTS)
        pca_projector.component = ncomponents
        pca_projector.preprocessors = cls.preprocessors + [Normalize()]

        pca = pca_projector(data)
        variance_ratio = pca.explained_variance_ratio_
        cumulative = np.cumsum(variance_ratio)

        self._pca = pca
        if not np.isfinite(cumulative[-1]):
            self.Warning.trivial_components()

        coords = pca(data).X
        valid_mask = ~np.isnan(coords).any(axis=1)
        # scale axes
        max_radius = np.min([np.abs(np.min(coords, axis=0)), np.max(coords, axis=0)])
        axes = pca.components_.T.copy()
        axes *= max_radius / np.max(np.linalg.norm(axes, axis=1))
        return valid_mask, coords, axes

    def _update_graph(self, reset_view=False):
        self.graph.zoomStack = []
        if self.graph.data is None:
            return
        self.graph.update_data(self.variable_x, self.variable_y, reset_view)

    def update_density(self):
        self._update_graph(reset_view=False)

    def selection_changed(self):
        if self.graph.selection is not None:
            self._selection = np.zeros(len(self.data), dtype=np.uint8)
            self._selection[self.plotdata.valid_mask] = self.graph.selection
            self.selection_indices = self._selection.tolist()
        else:
            self._selection = self.selection_indices = None
        self.commit()

    def prepare_data(self):
        pass

    def commit(self):
        def prepare_components():
            if self.placement in [self.Placement.Circular, self.Placement.LDA]:
                attrs = [a for a in self.model_selected[:]]
                axes = self.plotdata.axes
            elif self.placement == self.Placement.PCA:
                axes = self._pca.components_.T
                attrs = [a for a in self._pca.orig_domain.attributes]
            if self.placement != self.Placement.Projection:
                domain = Domain([ContinuousVariable(a.name, compute_value=lambda _: None)
                                 for a in attrs],
                                metas=[StringVariable(name='component')])
                metas = np.array([["{}{}".format(self.Component_name[self.placement], i + 1)
                                   for i in range(axes.shape[1])]],
                                 dtype=object).T
                components = Table(domain, axes.T, metas=metas)
                components.name = 'components'
            else:
                components = self.projection
            return components

        selected = annotated = components = None
        if self.data is not None and self.plotdata.data is not None:
            components = prepare_components()

            graph = self.graph
            mask = self.plotdata.valid_mask.astype(int)
            mask[mask == 1] = graph.selection if graph.selection is not None \
            else [False * len(mask)]

            selection = np.array([], dtype=np.uint8) if mask is None else np.flatnonzero(mask)
            name = self.data.name
            data = self.plotdata.data
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

        self.Outputs.selected_data.send(selected)
        self.Outputs.annotated_data.send(annotated)
        self.Outputs.components.send(components)

    def send_report(self):
        if self.data is None:
            return

        def name(var):
            return var and var.name

        def projection_name():
            name = ("Circular Placement",
                    "Linear Discriminant Analysis",
                    "Principal Component Analysis",
                    "Input projection")
            return name[self.placement]

        caption = report.render_items_vert((
            ("Projection", projection_name()),
            ("Color", name(self.graph.attr_color)),
            ("Label", name(self.graph.attr_label)),
            ("Shape", name(self.graph.attr_shape)),
            ("Size", name(self.graph.attr_size)),
            ("Jittering", self.graph.jitter_size != 0 and "{} %".format(self.graph.jitter_size))))
        self.report_plot()
        if caption:
            self.report_caption(caption)

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
    import sip

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
    r = app.exec()
    w.set_data(None)
    w.saveSettings()
    sip.delete(w)
    del w
    return r


if __name__ == "__main__":
    import sys
    sys.exit(main())
