import numpy as np
import scipy.spatial.distance

from AnyQt.QtCore import Qt, QTimer

import pyqtgraph as pg

from Orange.data import ContinuousVariable, Domain, Table
from Orange.distance import Euclidean
from Orange.misc import DistMatrix
from Orange.projection.manifold import torgerson, MDS

from Orange.widgets import gui, settings
from Orange.widgets.settings import SettingProvider
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase
from Orange.widgets.visualize.utils.widget import OWDataProjectionWidget
from Orange.widgets.widget import Msg, Input


def stress(X, distD):
    assert X.shape[0] == distD.shape[0] == distD.shape[1]
    D1_c = scipy.spatial.distance.pdist(X, metric="euclidean")
    D1 = scipy.spatial.distance.squareform(D1_c, checks=False)
    delta = D1 - distD
    delta_sq = np.square(delta, out=delta)
    return delta_sq.sum(axis=0) / 2


#: Maximum number of displayed closest pairs.
MAX_N_PAIRS = 10000


class OWMDSGraph(OWScatterPlotBase):
    #: Percentage of all pairs displayed (ranges from 0 to 20)
    connected_pairs = settings.Setting(5)

    def __init__(self, scatter_widget, parent):
        super().__init__(scatter_widget, parent)
        self.pairs_curve = None
        self.draw_pairs = True
        self._similar_pairs = None
        self.effective_matrix = None

    def set_effective_matrix(self, effective_matrix):
        self.effective_matrix = effective_matrix

    def pause_drawing_pairs(self):
        self.draw_pairs = False

    def resume_drawing_pairs(self):
        self.draw_pairs = True
        self.update_pairs(True)

    def update_coordinates(self):
        super().update_coordinates()
        self.update_pairs(reconnect=False)

    def update_pairs(self, reconnect):
        if self.pairs_curve:
            self.plot_widget.removeItem(self.pairs_curve)
        if not self.draw_pairs or self.connected_pairs == 0 \
                or self.effective_matrix is None \
                or self.scatterplot_item is None:
            return
        emb_x, emb_y = self.scatterplot_item.getData()
        if self._similar_pairs is None or reconnect:
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
            sorted = np.argsort(m[indcs])[:p]
            self._similar_pairs = fpairs = np.empty(2 * p, dtype=int)
            fpairs[::2] = indcs[0][sorted]
            fpairs[1::2] = indcs[1][sorted]
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
            pen=pg.mkPen(0.8, width=2, cosmetic=True),
            connect="pairs", antialias=True)
        self.plot_widget.addItem(self.pairs_curve)


class OWMDS(OWDataProjectionWidget):
    name = "MDS"
    description = "Two-dimensional data projection by multidimensional " \
                  "scaling constructed from a distance matrix."
    icon = "icons/MDS.svg"
    keywords = ["multidimensional scaling", "multi dimensional scaling"]

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

    #: Runtime state
    Running, Finished, Waiting = 1, 2, 3

    max_iter = settings.Setting(300)
    initialization = settings.Setting(PCA)
    refresh_rate = settings.Setting(3)

    GRAPH_CLASS = OWMDSGraph
    graph = SettingProvider(OWMDSGraph)
    embedding_variables_names = ("mds-x", "mds-y")

    class Error(OWDataProjectionWidget.Error):
        not_enough_rows = Msg("Input data needs at least 2 rows")
        matrix_too_small = Msg("Input matrix must be at least 2x2")
        no_attributes = Msg("Data has no attributes")
        mismatching_dimensions = \
            Msg("Data and distances dimensions do not match.")
        out_of_memory = Msg("Out of memory")
        optimization_error = Msg("Error during optimization\n{}")

    def __init__(self):
        super().__init__()
        #: Input dissimilarity matrix
        self.matrix = None  # type: Optional[DistMatrix]
        #: Data table from the `self.matrix.row_items` (if present)
        self.matrix_data = None  # type: Optional[Table]
        #: Input data table
        self.signal_data = None

        self.__invalidated = True
        self.embedding = None
        self.effective_matrix = None

        self.__update_loop = None
        # timer for scheduling updates
        self.__timer = QTimer(self, singleShot=True, interval=0)
        self.__timer.timeout.connect(self.__next_step)
        self.__state = OWMDS.Waiting
        self.__in_next_step = False

        self.graph.pause_drawing_pairs()

        self.size_model = self.gui.points_models[2]
        self.size_model.order = \
            self.gui.points_models[2].order[:1] \
            + ("Stress", ) + \
            self.gui.points_models[2].order[1:]
        # self._initialize()

    def _add_controls(self):
        self._add_controls_optimization()
        super()._add_controls()
        self.gui.add_control(
            self._effects_box, gui.hSlider, "Show similar pairs:",
            master=self.graph, value="connected_pairs", minValue=0,
            maxValue=20, createLabel=False, callback=self._on_connected_changed
        )

    def _add_controls_optimization(self):
        box = gui.vBox(self.controlArea, box=True)
        self.runbutton = gui.button(box, self, "Run optimization",
                                    callback=self._toggle_run)
        gui.comboBox(box, self, "refresh_rate", label="Refresh: ",
                     orientation=Qt.Horizontal,
                     items=[t for t, _ in OWMDS.RefreshRate],
                     callback=self.__invalidate_refresh)
        hbox = gui.hBox(box, margin=0)
        gui.button(hbox, self, "PCA", callback=self.do_PCA)
        gui.button(hbox, self, "Randomize", callback=self.do_random)
        gui.button(hbox, self, "Jitter", callback=self.do_jitter)

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

        if matrix is not None and len(matrix) < 2:
            self.Error.matrix_too_small()
            matrix = None
        else:
            self.Error.matrix_too_small.clear()

        self.matrix = matrix
        self.matrix_data = matrix.row_items if matrix is not None else None

    def clear(self):
        super().clear()
        self.embedding = None
        self.graph.set_effective_matrix(None)
        self.__set_update_loop(None)
        self.__state = OWMDS.Waiting

    def _initialize(self):
        matrix_existed = self.effective_matrix is not None
        effective_matrix = self.effective_matrix
        self.__invalidated = True
        self.data = None
        self.effective_matrix = None
        self.closeContext()
        self.clear_messages()

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
            if self.matrix.axis == 0 and self.data is self.matrix_data:
                self.data = None
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
        self.__invalidated = not (matrix_existed and
                                  self.effective_matrix is not None and
                                  np.array_equal(effective_matrix,
                                                 self.effective_matrix))
        if self.__invalidated:
            self.clear()
        self.graph.set_effective_matrix(self.effective_matrix)

    def _toggle_run(self):
        if self.__state == OWMDS.Running:
            self.stop()
            self._invalidate_output()
        else:
            self.start()

    def start(self):
        if self.__state == OWMDS.Running:
            return
        elif self.__state == OWMDS.Finished:
            # Resume/continue from a previous run
            self.__start()
        elif self.__state == OWMDS.Waiting and \
                self.effective_matrix is not None:
            self.__start()

    def stop(self):
        if self.__state == OWMDS.Running:
            self.__set_update_loop(None)

    def __start(self):
        self.graph.pause_drawing_pairs()
        X = self.effective_matrix
        init = self.embedding

        # number of iterations per single GUI update step
        _, step_size = OWMDS.RefreshRate[self.refresh_rate]
        if step_size == -1:
            step_size = self.max_iter

        def update_loop(X, max_iter, step, init):
            """
            return an iterator over successive improved MDS point embeddings.
            """
            # NOTE: this code MUST NOT call into QApplication.processEvents
            done = False
            iterations_done = 0
            oldstress = np.finfo(np.float).max
            init_type = "PCA" if self.initialization == OWMDS.PCA else "random"

            while not done:
                step_iter = min(max_iter - iterations_done, step)
                mds = MDS(
                    dissimilarity="precomputed", n_components=2,
                    n_init=1, max_iter=step_iter,
                    init_type=init_type, init_data=init
                )

                mdsfit = mds(X)
                iterations_done += step_iter

                embedding, stress = mdsfit.embedding_, mdsfit.stress_
                stress /= np.sqrt(np.sum(embedding ** 2, axis=1)).sum()

                if iterations_done >= max_iter:
                    done = True
                elif (oldstress - stress) < mds.params["eps"]:
                    done = True
                init = embedding
                oldstress = stress

                yield embedding, mdsfit.stress_, iterations_done / max_iter

        self.__set_update_loop(update_loop(X, self.max_iter, step_size, init))
        self.progressBarInit(processEvents=None)

    def __set_update_loop(self, loop):
        """
        Set the update `loop` coroutine.

        The `loop` is a generator yielding `(embedding, stress, progress)`
        tuples where `embedding` is a `(N, 2) ndarray` of current updated
        MDS points, `stress` is the current stress and `progress` a float
        ratio (0 <= progress <= 1)

        If an existing update coroutine loop is already in place it is
        interrupted (i.e. closed).

        .. note::
            The `loop` must not explicitly yield control flow to the event
            loop (i.e. call `QApplication.processEvents`)

        """
        if self.__update_loop is not None:
            self.__update_loop.close()
            self.__update_loop = None
            self.progressBarFinished(processEvents=None)

        self.__update_loop = loop

        if loop is not None:
            self.setBlocking(True)
            self.progressBarInit(processEvents=None)
            self.setStatusMessage("Running")
            self.runbutton.setText("Stop")
            self.__state = OWMDS.Running
            self.__timer.start()
        else:
            self.setBlocking(False)
            self.setStatusMessage("")
            self.runbutton.setText("Start")
            self.__state = OWMDS.Finished
            self.__timer.stop()

    def __next_step(self):
        if self.__update_loop is None:
            return

        assert not self.__in_next_step
        self.__in_next_step = True

        loop = self.__update_loop
        self.Error.out_of_memory.clear()
        try:
            embedding, _, progress = next(self.__update_loop)
            assert self.__update_loop is loop
        except StopIteration:
            self.__set_update_loop(None)
            self.unconditional_commit()
            self.graph.resume_drawing_pairs()
        except MemoryError:
            self.Error.out_of_memory()
            self.__set_update_loop(None)
            self.graph.resume_drawing_pairs()
        except Exception as exc:
            self.Error.optimization_error(str(exc))
            self.__set_update_loop(None)
            self.graph.resume_drawing_pairs()
        else:
            self.progressBarSet(100.0 * progress, processEvents=None)
            self.embedding = embedding
            self.graph.update_coordinates()
            # schedule next update
            self.__timer.start()

        self.__in_next_step = False

    def do_PCA(self):
        self.__invalidate_embedding(self.PCA)

    def do_random(self):
        self.__invalidate_embedding(self.Random)

    def do_jitter(self):
        self.__invalidate_embedding(self.Jitter)

    def __invalidate_embedding(self, initialization=PCA):
        def jitter_coord(part):
            span = np.max(part) - np.min(part)
            part += np.random.uniform(-span / 20, span / 20, len(part))

        # reset/invalidate the MDS embedding, to the default initialization
        # (Random or PCA), restarting the optimization if necessary.
        state = self.__state
        if self.__update_loop is not None:
            self.__set_update_loop(None)

        if self.effective_matrix is None:
            self.graph.reset_graph()
            return

        X = self.effective_matrix

        if initialization == OWMDS.PCA:
            self.embedding = torgerson(X)
        elif initialization == OWMDS.Random:
            self.embedding = np.random.rand(len(X), 2)
        else:
            jitter_coord(self.embedding[:, 0])
            jitter_coord(self.embedding[:, 1])

        self.setup_plot()

        # restart the optimization if it was interrupted.
        if state == OWMDS.Running:
            self.__start()

    def __invalidate_refresh(self):
        state = self.__state

        if self.__update_loop is not None:
            self.__set_update_loop(None)

        # restart the optimization if it was interrupted.
        # TODO: decrease the max iteration count by the already
        # completed iterations count.
        if state == OWMDS.Running:
            self.__start()

    def handleNewSignals(self):
        self._initialize()
        if self.__invalidated:
            self.graph.pause_drawing_pairs()
            self.__invalidated = False
            self.__invalidate_embedding()
            self.cb_class_density.setEnabled(self.can_draw_density())
            self.start()
        else:
            self.graph.update_point_props()
        self.commit()

    def _invalidate_output(self):
        self.commit()

    def _on_connected_changed(self):
        self.graph.set_effective_matrix(self.effective_matrix)
        self.graph.update_pairs(reconnect=True)

    def setup_plot(self):
        super().setup_plot()
        if self.embedding is not None:
            self.graph.update_pairs(reconnect=True)

    def get_size_data(self):
        if self.attr_size == "Stress":
            return stress(self.embedding, self.effective_matrix)
        else:
            return super().get_size_data()

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

    @classmethod
    def migrate_settings(cls, settings_, version):
        if version < 2:
            settings_graph = {}
            for old, new in (("label_only_selected", "label_only_selected"),
                             ("symbol_opacity", "alpha_value"),
                             ("symbol_size", "point_width"),
                             ("jitter", "jitter_size")):
                settings_graph[new] = settings_[old]
            settings_["graph"] = settings_graph
            settings_["auto_commit"] = settings_["autocommit"]

        if version < 3:
            if "connected_pairs" in settings_:
                connected_pairs = settings_["connected_pairs"]
                settings_["graph"]["connected_pairs"] = connected_pairs

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


if __name__ == "__main__":  # pragma: no cover
    data = Table("iris")
    WidgetPreview(OWMDS).run(set_data=data, set_subset_data=data[:30])
