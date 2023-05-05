# pylint: disable=too-many-ancestors
import time
from types import SimpleNamespace as namespace

import numpy as np
import scipy.spatial.distance

from AnyQt.QtGui import QIcon
from AnyQt.QtWidgets import QSizePolicy, QGridLayout, QLabel, QPushButton

import pyqtgraph as pg

from orangecanvas.gui.svgiconengine import SvgIconEngine

from Orange.data import ContinuousVariable, Domain, Table, StringVariable
from Orange.data.util import array_equal
from Orange.distance import Euclidean
from Orange.misc import DistMatrix
from Orange.projection.manifold import torgerson, MDS

from Orange.widgets import gui
from Orange.widgets.settings import SettingProvider, Setting
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase
from Orange.widgets.visualize.utils.widget import OWDataProjectionWidget
from Orange.widgets.widget import Msg, Input


class Result(namespace):
    embedding = None  # type: np.ndarray


def run_mds(matrix: DistMatrix, max_iter: int, step_size: int, init_type: int,
            embedding: np.ndarray, state: TaskState):
    res = Result(embedding=embedding)

    iterations_done = 0
    init = embedding
    state.set_status("Running...")
    oldstress = np.finfo(float).max

    while True:
        loop_start = time.time()
        step_iter = min(max_iter - iterations_done, step_size)
        mds = MDS(
            dissimilarity="precomputed", n_components=2,
            n_init=1, max_iter=step_iter,
            init_type=init_type, init_data=init
        )

        mdsfit = mds(matrix)
        iterations_done += step_iter

        embedding, stress = mdsfit.embedding_, mdsfit.stress_
        emb_norm = np.sqrt(np.sum(embedding ** 2, axis=1)).sum()
        if emb_norm > 0:
            stress /= emb_norm

        res.embedding = embedding
        state.set_partial_result(res)
        state.set_progress_value(100 * iterations_done / max_iter)
        if iterations_done >= max_iter or stress == 0 or \
                (oldstress - stress) < mds.params["eps"]:
            return res
        init = embedding
        oldstress = stress
        if state.is_interruption_requested():
            return res
        if (wait := 0.1 - (time.time() - loop_start)) > 0:
            time.sleep(wait)


#: Maximum number of displayed closest pairs.
MAX_N_PAIRS = 10000


class OWMDSGraph(OWScatterPlotBase):
    #: Percentage of all pairs displayed (ranges from 0 to 20)
    connected_pairs = Setting(5)

    def __init__(self, scatter_widget, parent):
        super().__init__(scatter_widget, parent)
        self.pairs_curve = None
        self._similar_pairs = None
        self.effective_matrix = None

    def set_effective_matrix(self, effective_matrix):
        self.effective_matrix = effective_matrix
        self._similar_pairs = None

    def update_coordinates(self):
        super().update_coordinates()
        self.update_pairs()

    def update_jittering(self):
        super().update_jittering()
        self.update_pairs()

    def update_pairs(self):
        if self.pairs_curve:
            self.plot_widget.removeItem(self.pairs_curve)
        if self.connected_pairs == 0 \
                or self.effective_matrix is None \
                or self.scatterplot_item is None:
            return
        emb_x, emb_y = self.scatterplot_item.getData()
        if self._similar_pairs is None:
            # This code requires storing lower triangle of X (n x n / 2
            # doubles), n x n / 2 * 2 indices to X, n x n / 2 indices for
            # argsort result. If this becomes an issue, it can be reduced to
            # n x n argsort indices by argsorting the entire X. Then we
            # take the first n + 2 * p indices. We compute their coordinates
            # i, j in the original matrix. We keep those for which i < j.
            # n + 2 * p will suffice to exclude the diagonal (i = j). If the
            # number of those for which i < j is smaller than p, we instead
            # take i > j. Among those that remain, we take the first p.
            # Assuming that MDS can't show so many points that memory could
            # become an issue, I preferred using simpler code.
            m = self.effective_matrix
            n = len(m)
            p = min(n * (n - 1) // 2 * self.connected_pairs // 100,
                    MAX_N_PAIRS * self.connected_pairs // 20)
            indcs = np.triu_indices(n, 1)
            sorted_ind = np.argsort(m[indcs])[:p]
            self._similar_pairs = fpairs = np.empty(2 * p, dtype=int)
            fpairs[::2] = indcs[0][sorted_ind]
            fpairs[1::2] = indcs[1][sorted_ind]
        emb_x_pairs = emb_x[self._similar_pairs].reshape((-1, 2))
        emb_y_pairs = emb_y[self._similar_pairs].reshape((-1, 2))

        # Filter out zero distance lines (in embedding coords).
        # Null (zero length) line causes bad rendering artifacts
        # in Qt when using the raster graphics system (see gh-issue: 1668).
        (x1, x2), (y1, y2) = (emb_x_pairs.T, emb_y_pairs.T)
        pairs_mask = ~(np.isclose(x1, x2) & np.isclose(y1, y2))
        emb_x_pairs = emb_x_pairs[pairs_mask, :]
        emb_y_pairs = emb_y_pairs[pairs_mask, :]
        self.pairs_curve = pg.PlotCurveItem(
            emb_x_pairs.ravel(), emb_y_pairs.ravel(),
            pen=pg.mkPen(0.8, width=1, cosmetic=True),
            connect="pairs", antialias=True)
        self.pairs_curve.setSegmentedLineMode("on")
        self.pairs_curve.setZValue(-1)
        self.plot_widget.addItem(self.pairs_curve)


class OWMDS(OWDataProjectionWidget, ConcurrentWidgetMixin):
    name = "MDS"
    description = "Two-dimensional data projection by multidimensional " \
                  "scaling constructed from a distance matrix."
    icon = "icons/MDS.svg"
    keywords = "mds, multidimensional scaling, multi dimensional scaling"

    class Inputs(OWDataProjectionWidget.Inputs):
        distances = Input("Distances", DistMatrix)

    settings_version = 3

    #: Initialization type
    PCA, Random, Jitter = 0, 1, 2

    #: Refresh rate
    RefreshRate = [
        ("Every iteration", 1),
        ("Every 5 steps", 5),
        ("Every 10 steps", 10),
        ("Every 25 steps", 25),
        ("Every 50 steps", 50),
        ("None", -1)
    ]

    max_iter = Setting(300)
    initialization = Setting(PCA)
    refresh_rate: int = Setting(3)

    GRAPH_CLASS = OWMDSGraph
    graph = SettingProvider(OWMDSGraph)
    embedding_variables_names = ("mds-x", "mds-y")

    class Error(OWDataProjectionWidget.Error):
        not_enough_rows = Msg("Input data needs at least 2 rows")
        matrix_not_symmetric = Msg("Distance matrix is not symmetric")
        matrix_too_small = Msg("Input matrix must be at least 2x2")
        no_attributes = Msg("Data has no attributes")
        mismatching_dimensions = \
            Msg("Data and distances dimensions do not match.")
        out_of_memory = Msg("Out of memory")
        optimization_error = Msg("Error during optimization\n{}")

    def __init__(self):
        OWDataProjectionWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        #: Input dissimilarity matrix
        self.matrix = None  # type: Optional[DistMatrix]
        #: Data table from the `self.matrix.row_items` (if present)
        self.matrix_data = None  # type: Optional[Table]
        #: Input data table
        self.signal_data = None

        self.embedding = None  # type: Optional[np.ndarray]
        self.effective_matrix = None  # type: Optional[DistMatrix]
        self.stress = None

        self.size_model = self.gui.points_models[2]
        self.size_model.order = \
            self.gui.points_models[2].order[:1] \
            + ("Stress", ) + \
            self.gui.points_models[2].order[1:]

    def _add_controls(self):
        self._add_controls_optimization()
        super()._add_controls()
        self.gui.add_control(
            self._effects_box, gui.hSlider, "Show similar pairs:",
            master=self.graph, value="connected_pairs", minValue=0,
            maxValue=20, createLabel=False, callback=self._on_connected_changed
        )

    def _add_controls_optimization(self):
        # This is a part of init
        # pylint: disable=attribute-defined-outside-init
        box = gui.vBox(self.controlArea, box="Optimize", spacing=0)
        hbox = gui.hBox(box, margin=0)
        gui.button(hbox, self, "PCA", callback=self.do_PCA, autoDefault=False)
        gui.button(hbox, self, "Randomize", callback=self.do_random, autoDefault=False)
        gui.button(hbox, self, "Jitter", callback=self.do_jitter, autoDefault=False)
        gui.separator(box, height=18)
        grid = QGridLayout()
        gui.widgetBox(box, orientation=grid)
        self.run_button = gui.button(None, self, "Start", self._toggle_run)
        self.step_button = QPushButton(QIcon(SvgIconEngine(_playpause_icon)), "")
        self.step_button.pressed.connect(self._step)
        self.step_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        grid.addWidget(self.run_button, 0, 0, 1, 2)
        grid.addWidget(self.step_button, 0, 2)
        grid.addWidget(QLabel("Refresh:"), 1, 0)
        grid.addWidget(
            gui.comboBox(
                None, self, "refresh_rate",
                items=[t for t, _ in OWMDS.RefreshRate],
                sizePolicy=(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed),
                callback=self.__refresh_rate_combo_changed),
            1, 1)
        self.stress_label = QLabel("Kruskal Stress: -")
        grid.addWidget(self.stress_label, 2, 0, 1, 3)

    def __refresh_rate_combo_changed(self):
        if self.task is not None:
            self._run()

    @Inputs.data
    def set_data(self, data):
        """Set the input dataset.

        Parameters
        ----------
        data : Optional[Table]
        """
        if data is not None and len(data) < 2:
            self.Error.not_enough_rows()
            data = None
        else:
            self.Error.not_enough_rows.clear()

        self.signal_data = data

    @Inputs.distances
    def set_disimilarity(self, matrix):
        """Set the dissimilarity (distance) matrix.

        Parameters
        ----------
        matrix : Optional[Orange.misc.DistMatrix]
        """

        self.Error.matrix_too_small.clear()
        self.Error.matrix_not_symmetric.clear()
        self.matrix = None
        if matrix is not None:
            if not matrix.is_symmetric():
                self.Error.matrix_not_symmetric()
            elif len(matrix) < 2:
                self.Error.matrix_too_small()
            else:
                self.matrix = matrix

        self.matrix_data = matrix.row_items if matrix is not None else None

    def clear(self):
        super().clear()
        self.cancel()
        self.embedding = None
        self.graph.set_effective_matrix(None)

    def _initialize(self):
        matrix_existed = self.effective_matrix is not None
        effective_matrix = self.effective_matrix
        self._invalidated = True
        self.data = None
        self.effective_matrix = None
        self.closeContext()

        self.Error.no_attributes.clear()
        self.Error.mismatching_dimensions.clear()
        self.Error.out_of_memory.clear()
        self.Error.optimization_error.clear()

        # if no data nor matrix is present reset plot
        if self.signal_data is None and self.matrix is None:
            self.clear()
            self.init_attr_values()
            return

        if self.signal_data is not None and self.matrix is not None and \
                len(self.signal_data) != len(self.matrix):
            self.Error.mismatching_dimensions()
            self.clear()
            self.init_attr_values()
            return

        if self.signal_data is not None:
            self.data = self.signal_data
        elif self.matrix_data is not None:
            self.data = self.matrix_data

        if self.matrix is not None:
            self.effective_matrix = self.matrix
            if self.matrix.axis == 0 and self.data is not None \
                    and self.data is self.matrix_data:
                names = [[attr.name] for attr in self.data.domain.attributes]
                domain = Domain([], metas=[StringVariable("labels")])
                self.data = Table.from_list(domain, names)
        elif self.data.domain.attributes:
            preprocessed_data = MDS().preprocess(self.data)
            self.effective_matrix = Euclidean(preprocessed_data)
        else:
            self.Error.no_attributes()
            self.clear()
            self.init_attr_values()
            return

        self.init_attr_values()
        self.openContext(self.data)
        self._invalidated = not (matrix_existed and
                                 self.effective_matrix is not None and
                                 array_equal(effective_matrix, self.effective_matrix))
        if self._invalidated:
            self.clear()
        self.graph.set_effective_matrix(self.effective_matrix)

    def init_attr_values(self):
        super().init_attr_values()
        if self.matrix is not None and self.matrix.axis == 0 and \
                self.data is not None and len(self.data):
            self.attr_label = self.data.domain["labels"]

    def _toggle_run(self):
        if self.task is not None:
            self.cancel()
            self.run_button.setText("Resume")
            self.step_button.setEnabled(True)
            self.commit.deferred()
        else:
            self._run()

    def _step(self):
        self._run(1)

    def _run(self, steps=None):
        if self.effective_matrix is None \
                or np.allclose(self.effective_matrix, 0):
            return
        self.run_button.setText("Stop")
        self.step_button.setEnabled(False)
        # false positive, pylint: disable=invalid-sequence-index
        _, step_size = OWMDS.RefreshRate[self.refresh_rate]
        if step_size == -1:
            step_size = self.max_iter
        init_type = "PCA" if self.initialization == OWMDS.PCA else "random"
        self.start(run_mds, self.effective_matrix, steps or self.max_iter,
                   step_size, init_type, self.embedding)

    # ConcurrentWidgetMixin
    def on_partial_result(self, result: Result):
        assert isinstance(result.embedding, np.ndarray)
        assert len(result.embedding) == len(self.effective_matrix)
        first_result = self.embedding is None
        new_embedding = result.embedding
        need_update = new_embedding is not self.embedding
        self.embedding = new_embedding
        if first_result:
            self.setup_plot()
        else:
            if need_update:
                self.graph.update_coordinates()
                self.graph.update_density()
                self.update_stress()

    def on_done(self, result: Result):
        assert isinstance(result.embedding, np.ndarray)
        assert len(result.embedding) == len(self.effective_matrix)
        # embedding, graph and stress are already updated in on_partial_result
        self.run_button.setText("Start")
        self.step_button.setEnabled(True)
        self.commit.deferred()

    def update_stress(self):
        self.stress = self._compute_stress()
        stress_val = "-" if self.stress is None else f"{self.stress:.3f}"
        self.stress_label.setText(f"Kruskal Stress: {stress_val}")

    def _compute_stress(self):
        if self.embedding is None or self.effective_matrix is None:
            return None
        point_stress = self.get_stress(self.embedding, self.effective_matrix)
        return np.sqrt(2 * np.sum(point_stress)
                       / (np.sum(self.effective_matrix ** 2) or 1))

    def on_exception(self, ex: Exception):
        if isinstance(ex, MemoryError):
            self.Error.out_of_memory()
        else:
            self.Error.optimization_error(str(ex))
        self.run_button.setText("Start")
        self.step_button.setEnabled(True)

    def do_PCA(self):
        self.do_initialization(self.PCA)

    def do_random(self):
        self.do_initialization(self.Random)

    def do_jitter(self):
        self.do_initialization(self.Jitter)

    def do_initialization(self, init_type: int):
        self.run_button.setText("Start")
        self.step_button.setEnabled(True)
        self.__invalidate_embedding(init_type)
        self.graph.update_coordinates()
        self.commit.deferred()

    def __invalidate_embedding(self, initialization=PCA):
        def jitter_coord(part):
            span = np.max(part) - np.min(part)
            part += np.random.uniform(-span / 20, span / 20, len(part))

        # reset/invalidate the MDS embedding, to the default initialization
        # (Random or PCA), restarting the optimization if necessary.
        if self.effective_matrix is None:
            self.graph.reset_graph()
            self.update_stress()
            return

        X = self.effective_matrix

        if initialization == OWMDS.PCA:
            self.embedding = torgerson(X)
        elif initialization == OWMDS.Random:
            self.embedding = np.random.rand(len(X), 2)
        else:
            jitter_coord(self.embedding[:, 0])
            jitter_coord(self.embedding[:, 1])

        # restart the optimization if it was interrupted.
        if self.task is not None:
            self._run()
        else:
            self.update_stress()

    def handleNewSignals(self):
        self._initialize()
        self.input_changed.emit(self.data)
        if self._invalidated:
            self.__invalidate_embedding()
            self.enable_controls()
            if self.effective_matrix is not None:
                self._run()
        super().handleNewSignals()

    def _on_connected_changed(self):
        self.graph.set_effective_matrix(self.effective_matrix)
        self.graph.update_pairs()

    def setup_plot(self):
        super().setup_plot()
        if self.embedding is not None:
            self.graph.update_pairs()

    def get_size_data(self):
        if self.attr_size == "Stress":
            return self.get_stress(self.embedding, self.effective_matrix)
        else:
            return super().get_size_data()

    @staticmethod
    def get_stress(X, distD):
        assert X.shape[0] == distD.shape[0] == distD.shape[1]
        D1_c = scipy.spatial.distance.pdist(X, metric="euclidean")
        D1 = scipy.spatial.distance.squareform(D1_c, checks=False)
        delta = D1 - distD
        delta_sq = np.square(delta, out=delta)
        return delta_sq.sum(axis=0) / 2

    def get_embedding(self):
        self.valid_data = np.ones(len(self.embedding), dtype=bool) \
            if self.embedding is not None else None
        return self.embedding

    def _get_projection_data(self):
        if self.embedding is None:
            return None

        if self.data is None:
            x_name, y_name = self.embedding_variables_names
            variables = ContinuousVariable(x_name), ContinuousVariable(y_name)
            return Table(Domain(variables), self.embedding)
        return super()._get_projection_data()

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    @classmethod
    def migrate_settings(cls, settings, version):
        if version < 2:
            settings_graph = {}
            for old, new in (("label_only_selected", "label_only_selected"),
                             ("symbol_opacity", "alpha_value"),
                             ("symbol_size", "point_width"),
                             ("jitter", "jitter_size")):
                settings_graph[new] = settings[old]
            settings["graph"] = settings_graph
            settings["auto_commit"] = settings["autocommit"]

        if version < 3:
            if "connected_pairs" in settings:
                connected_pairs = settings["connected_pairs"]
                settings["graph"]["connected_pairs"] = connected_pairs

    @classmethod
    def migrate_context(cls, context, version):
        if version < 2:
            domain = context.ordered_domain
            n_domain = [t for t in context.ordered_domain if t[1] == 2]
            c_domain = [t for t in context.ordered_domain if t[1] == 1]
            context_values = {}
            for _, old_val, new_val in ((domain, "color_value", "attr_color"),
                                        (c_domain, "shape_value", "attr_shape"),
                                        (n_domain, "size_value", "attr_size"),
                                        (domain, "label_value", "attr_label")):
                tmp = context.values[old_val]
                if tmp[1] >= 0:
                    context_values[new_val] = (tmp[0], tmp[1] + 100)
                elif tmp[0] != "Stress":
                    context_values[new_val] = None
                else:
                    context_values[new_val] = tmp
            context.values = context_values

        if version < 3 and "graph" in context.values:
            values = context.values
            values["attr_color"] = values["graph"]["attr_color"]
            values["attr_size"] = values["graph"]["attr_size"]
            values["attr_shape"] = values["graph"]["attr_shape"]
            values["attr_label"] = values["graph"]["attr_label"]


_playpause_icon = b"""<?xml version="1.0" encoding="utf-8"?>
<svg version="1.1"
	 id="svg3025" xmlns:cc="http://creativecommons.org/ns#" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd" xmlns:svg="http://www.w3.org/2000/svg"
	 xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 1347.7 1178.5"
	 style="enable-background:new 0 0 1347.7 1178.5;" xml:space="preserve">
<sodipodi:namedview  bordercolor="#666666" borderopacity="1" gridtolerance="10" guidetolerance="10" id="namedview3031" inkscape:current-layer="svg3025" inkscape:cx="896" inkscape:cy="896" inkscape:pageopacity="0" inkscape:pageshadow="2" inkscape:window-height="480" inkscape:window-maximized="0" inkscape:window-width="640" inkscape:window-x="0" inkscape:window-y="25" inkscape:zoom="0.13169643" objecttolerance="10" pagecolor="#ffffff" showgrid="false">
	</sodipodi:namedview>
<g id="g3027" transform="matrix(1,0,0,-1,37.966102,1320.6441)">
	<path inkscape:connector-curvature="0" d="M1003,628.1c-1-4.2-4.5-7.6-7.6-11.1L581,202.5c-7.4-7.4-13.6-9.9-18.7-7.6
		c-5.1,2.3-7.6,8.6-7.6,18.7v414.5c-1.9-3.9-4.5-7.6-7.6-11.1L132.6,202.5c-7.4-7.4-13.6-9.9-18.7-7.6c-5.1,2.3-7.6,8.6-7.6,18.7
		v859.3c0,10.1,2.5,16.3,7.6,18.7c5.1,2.3,11.3-0.2,18.7-7.6l414.5-414.5c3.1-3.1,5.6-6.8,7.6-11.1v414.5c0,10.1,2.5,16.3,7.6,18.7
		c5.1,2.3,11.3-0.2,18.7-7.6l414.5-414.5c3.1-3.1,6.4-6.5,7.6-11.1C1004.9,651.1,1004.8,635.4,1003,628.1z"/>
	<path inkscape:connector-curvature="0" d="M1038,650v404.2c0,10.1,3.7,18.9,11.1,26.3c7.4,7.4,16.2,11.1,26.3,11.1h74.7
		c10.1,0,18.9-3.7,26.3-11.1c7.4-7.4,11.1-16.2,11.1-26.3v-822c0-10.1-3.7-18.9-11.1-26.3c-7.4-7.4-16.2-11.1-26.3-11.1h-74.7
		c-10.1,0-18.9,3.7-26.3,11.1c-7.4,7.4-11.1,16.2-11.1,26.3V650"/>
	<path d="M918.9,404.1"/>
</g>
</svg>
"""

if __name__ == "__main__":  # pragma: no cover
    table = Table("iris")
    WidgetPreview(OWMDS).run(set_data=table, set_subset_data=table[:30])
