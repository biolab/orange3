import sys

import numpy as np
import scipy.spatial.distance

from AnyQt.QtWidgets import QFormLayout, QApplication
from AnyQt.QtGui import QPainter
from AnyQt.QtCore import Qt, QTimer

import pyqtgraph as pg

import Orange.data
from Orange.data import Domain, Table, ContinuousVariable
from Orange.data.util import hstack
import Orange.projection
from Orange.projection.manifold import torgerson
import Orange.distance
import Orange.misc
from Orange.widgets import gui, settings
from Orange.widgets.settings import SettingProvider
from Orange.widgets.utils.sql import check_sql_input
from Orange.canvas import report
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotGraph, InteractiveViewBox
from Orange.widgets.widget import Msg, OWWidget, Input, Output
from Orange.widgets.utils.annotated_data import (
    ANNOTATED_DATA_SIGNAL_NAME, create_annotated_table, create_groups_table,
    get_unique_names)


def stress(X, distD):
    assert X.shape[0] == distD.shape[0] == distD.shape[1]
    D1_c = scipy.spatial.distance.pdist(X, metric="euclidean")
    D1 = scipy.spatial.distance.squareform(D1_c, checks=False)
    delta = D1 - distD
    delta_sq = np.square(delta, out=delta)
    return delta_sq.sum(axis=0) / 2


class MDSInteractiveViewBox(InteractiveViewBox):
    def _dragtip_pos(self):
        return 10, 10


class OWMDSGraph(OWScatterPlotGraph):
    jitter_size = settings.Setting(0)

    def __init__(self, scatter_widget, parent=None, name="None", view_box=None):
        super().__init__(scatter_widget, parent=parent, _=name, view_box=view_box)
        for axis_loc in ["left", "bottom"]:
            self.plot_widget.hideAxis(axis_loc)

    def update_data(self, attr_x, attr_y, reset_view=True):
        super().update_data(attr_x, attr_y, reset_view=reset_view)
        for axis in ["left", "bottom"]:
            self.plot_widget.hideAxis(axis)
        self.plot_widget.setAspectLocked(True, 1)

    def compute_sizes(self):
        def scale(a):
            dmin, dmax = np.nanmin(a), np.nanmax(a)
            if dmax - dmin > 0:
                return (a - dmin) / (dmax - dmin)
            else:
                return np.zeros_like(a)

        self.master.Information.missing_size.clear()
        if self.attr_size is None:
            size_data = np.full((self.n_points,), self.point_width,
                                dtype=float)
        elif self.attr_size == "Stress":
            size_data = scale(stress(self.master.embedding, self.master.effective_matrix))
            size_data = self.MinShapeSize + size_data * self.point_width
        else:
            size_data = \
                self.MinShapeSize + \
                self.scaled_data.get_column_view(self.attr_size)[0][self.valid_data] * \
                self.point_width
        nans = np.isnan(size_data)
        if np.any(nans):
            size_data[nans] = self.MinShapeSize - 2
            self.master.Information.missing_size(self.attr_size)
        return size_data


#: Maximum number of displayed closest pairs.
MAX_N_PAIRS = 10000

class OWMDS(OWWidget):
    name = "MDS"
    description = "Two-dimensional data projection by multidimensional " \
                  "scaling constructed from a distance matrix."
    icon = "icons/MDS.svg"

    class Inputs:
        data = Input("Data", Orange.data.Table, default=True)
        distances = Input("Distances", Orange.misc.DistMatrix)
        data_subset = Input("Data Subset", Orange.data.Table)

    class Outputs:
        selected_data = Output("Selected Data", Orange.data.Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Orange.data.Table)

    settings_version = 2

    #: Initialization type
    PCA, Random = 0, 1

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

    settingsHandler = settings.DomainContextHandler()

    max_iter = settings.Setting(300)
    initialization = settings.Setting(PCA)
    refresh_rate = settings.Setting(3)

    # output embedding role.
    NoRole, AttrRole, AddAttrRole, MetaRole = 0, 1, 2, 3

    auto_commit = settings.Setting(True)

    selection_indices = settings.Setting(None, schema_only=True)

    #: Percentage of all pairs displayed (ranges from 0 to 20)
    connected_pairs = settings.Setting(5)

    legend_anchor = settings.Setting(((1, 0), (1, 0)))

    graph = SettingProvider(OWMDSGraph)

    jitter_sizes = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10]

    graph_name = "graph.plot_widget.plotItem"

    class Error(OWWidget.Error):
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
        self.matrix = None  # type: Optional[Orange.misc.DistMatrix]
        #: Effective data used for plot styling/annotations. Can be from the
        #: input signal (`self.signal_data`) or the input matrix
        #: (`self.matrix.data`)
        self.data = None  # type: Optional[Orange.data.Table]
        #: Input subset data table
        self.subset_data = None  # type: Optional[Orange.data.Table]
        #: Data table from the `self.matrix.row_items` (if present)
        self.matrix_data = None  # type: Optional[Orange.data.Table]
        #: Input data table
        self.signal_data = None

        self._similar_pairs = None
        self._subset_mask = None  # type: Optional[np.ndarray]
        self._invalidated = False
        self.effective_matrix = None
        self._curve = None

        self.variable_x = ContinuousVariable("mds-x")
        self.variable_y = ContinuousVariable("mds-y")

        self.__update_loop = None
        # timer for scheduling updates
        self.__timer = QTimer(self, singleShot=True, interval=0)
        self.__timer.timeout.connect(self.__next_step)
        self.__state = OWMDS.Waiting
        self.__in_next_step = False
        self.__draw_similar_pairs = False

        box = gui.vBox(self.controlArea, "MDS Optimization")
        form = QFormLayout(
            labelAlignment=Qt.AlignLeft,
            formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
            verticalSpacing=10
        )

        form.addRow(
            "Max iterations:",
            gui.spin(box, self, "max_iter", 10, 10 ** 4, step=1))

        form.addRow(
            "Initialization:",
            gui.radioButtons(box, self, "initialization", btnLabels=("PCA (Torgerson)", "Random"),
                             callback=self.__invalidate_embedding))

        box.layout().addLayout(form)
        form.addRow(
            "Refresh:",
            gui.comboBox(box, self, "refresh_rate", items=[t for t, _ in OWMDS.RefreshRate],
                         callback=self.__invalidate_refresh))
        gui.separator(box, 10)
        self.runbutton = gui.button(box, self, "Run", callback=self._toggle_run)

        box = gui.vBox(self.mainArea, True, margin=0)
        self.graph = OWMDSGraph(self, box, "MDSGraph", view_box=MDSInteractiveViewBox)
        box.layout().addWidget(self.graph.plot_widget)
        self.plot = self.graph.plot_widget

        g = self.graph.gui
        box = g.point_properties_box(self.controlArea)
        self.models = g.points_models
        self.size_model = self.models[2]
        self.label_model = self.models[3]
        self.size_model.order = \
            self.size_model.order[:1] + ("Stress", ) + self.models[2].order[1:]

        gui.hSlider(box, self, "connected_pairs", label="Show similar pairs:", minValue=0,
                    maxValue=20, createLabel=False, callback=self._on_connected_changed)
        g.add_widgets(ids=[g.JitterSizeSlider], widget=box)

        box = gui.vBox(self.controlArea, "Plot Properties")
        g.add_widgets([g.ShowLegend,
                       g.ToolTipShowsAll,
                       g.ClassDensity,
                       g.LabelOnlySelected], box)

        self.controlArea.layout().addStretch(100)
        self.icons = gui.attributeIconDict

        palette = self.graph.plot_widget.palette()
        self.graph.set_palette(palette)

        gui.rubber(self.controlArea)

        self.graph.box_zoom_select(self.controlArea)

        gui.auto_commit(box, self, "auto_commit", "Send Selected",
                        checkbox_label="Send selected automatically",
                        box=None)

        self.plot.getPlotItem().hideButtons()
        self.plot.setRenderHint(QPainter.Antialiasing)

        self.graph.jitter_continuous = True
        self._initialize()

    def reset_graph_data(self, *_):
        if self.data is not None:
            self.graph.rescale_data()
            self.update_graph()
        self.connect_pairs()

    def update_colors(self):
        pass

    def update_density(self):
        self.update_graph(reset_view=False)

    def update_regression_line(self):
        self.update_graph(reset_view=False)

    def init_attr_values(self):
        self.graph.set_domain(self.data)

    def prepare_data(self):
        pass

    def update_graph(self, reset_view=True, **_):
        self.graph.zoomStack = []
        if self.graph.data is None:
            return
        self.graph.update_data(self.variable_x, self.variable_y, True)

    def selection_changed(self):
        self.commit()

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        """Set the input dataset.

        Parameters
        ----------
        data : Optional[Orange.data.Table]
        """
        if data is not None and len(data) < 2:
            self.Error.not_enough_rows()
            data = None
        else:
            self.Error.not_enough_rows.clear()

        self.signal_data = data

        if self.matrix is not None and data is not None and len(self.matrix) == len(data):
            self.closeContext()
            self.data = data
            self.init_attr_values()
            self.openContext(data)
        else:
            self._invalidated = True

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
        self._invalidated = True

    @Inputs.data_subset
    def set_subset_data(self, subset_data):
        """Set a subset of `data` input to highlight in the plot.

        Parameters
        ----------
        subset_data: Optional[Orange.data.Table]
        """
        self.subset_data = subset_data
        # invalidate the pen/brush when the subset is changed
        self._subset_mask = None  # type: Optional[np.ndarray]
        self.controls.graph.alpha_value.setEnabled(subset_data is None)

    def _clear(self):
        self._similar_pairs = None

        self.__set_update_loop(None)
        self.__state = OWMDS.Waiting

    def _clear_plot(self):
        self.graph.plot_widget.clear()

    def _initialize(self):
        # clear everything
        self.closeContext()
        self._clear()
        self.Error.clear()
        self.data = None
        self.effective_matrix = None
        self.embedding = None
        self.init_attr_values()

        # if no data nor matrix is present reset plot
        if self.signal_data is None and self.matrix is None:
            return

        if self.signal_data is not None and self.matrix is not None and \
                len(self.signal_data) != len(self.matrix):
            self.Error.mismatching_dimensions()
            self._update_plot()
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
            preprocessed_data = Orange.projection.MDS().preprocess(self.data)
            self.effective_matrix = Orange.distance.Euclidean(preprocessed_data)
        else:
            self.Error.no_attributes()
            return

        self.init_attr_values()
        self.openContext(self.data)

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
        self.__draw_similar_pairs = False
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
                mds = Orange.projection.MDS(
                    dissimilarity="precomputed", n_components=2,
                    n_init=1, max_iter=step_iter,
                    init_type=init_type, init_data=init)

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
            self.__draw_similar_pairs = True
            self._update_plot()
        except MemoryError:
            self.Error.out_of_memory()
            self.__set_update_loop(None)
            self.__draw_similar_pairs = True
        except Exception as exc:
            self.Error.optimization_error(str(exc))
            self.__set_update_loop(None)
            self.__draw_similar_pairs = True
        else:
            self.progressBarSet(100.0 * progress, processEvents=None)
            self.embedding = embedding
            self._update_plot()
            # schedule next update
            self.__timer.start()

        self.__in_next_step = False

    def __invalidate_embedding(self):
        # reset/invalidate the MDS embedding, to the default initialization
        # (Random or PCA), restarting the optimization if necessary.
        if self.embedding is None:
            return
        state = self.__state
        if self.__update_loop is not None:
            self.__set_update_loop(None)

        X = self.effective_matrix

        if self.initialization == OWMDS.PCA:
            self.embedding = torgerson(X)
        else:
            self.embedding = np.random.rand(len(X), 2)

        self._update_plot()

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
        if self._invalidated:
            self.__draw_similar_pairs = False
            self._invalidated = False
            self._initialize()
            self.start()

        if self._subset_mask is None and self.subset_data is not None and \
                self.data is not None:
            self._subset_mask = np.in1d(self.data.ids, self.subset_data.ids)

        self._update_plot(new=True)
        self.unconditional_commit()

    def _invalidate_output(self):
        self.commit()

    def _on_connected_changed(self):
        self._similar_pairs = None
        self.connect_pairs()

    def _update_plot(self, new=False):
        self._clear_plot()

        if self.embedding is not None:
            self._setup_plot(new=new)
        else:
            self.graph.new_data(None)

    def connect_pairs(self):
        if self._curve:
            self.graph.plot_widget.removeItem(self._curve)
        if not (self.connected_pairs and self.__draw_similar_pairs):
            return
        emb_x, emb_y = self.graph.get_xy_data_positions(
            self.variable_x, self.variable_y, self.graph.valid_data)
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
        self._curve = pg.PlotCurveItem(
            emb_x_pairs.ravel(), emb_y_pairs.ravel(),
            pen=pg.mkPen(0.8, width=2, cosmetic=True),
            connect="pairs", antialias=True)
        self.graph.plot_widget.addItem(self._curve)

    def _setup_plot(self, new=False):
        emb_x, emb_y = self.embedding[:, 0], self.embedding[:, 1]
        coords = np.vstack((emb_x, emb_y)).T

        data = self.data
        attributes = data.domain.attributes + (self.variable_x, self.variable_y)
        domain = Domain(attributes=attributes,
                        class_vars=data.domain.class_vars,
                        metas=data.domain.metas)
        data = Table.from_numpy(domain, X=hstack((data.X, coords)),
                                Y=data.Y, metas=data.metas)
        subset_data = data[self._subset_mask] if self._subset_mask is not None else None
        self.graph.new_data(data, subset_data=subset_data, new=new)
        self.graph.update_data(self.variable_x, self.variable_y, True)
        self.connect_pairs()

    def commit(self):
        if self.embedding is not None:
            names = get_unique_names([v.name for v in self.data.domain.variables],
                                     ["mds-x", "mds-y"])
            output = embedding = Orange.data.Table.from_numpy(
                Orange.data.Domain([ContinuousVariable(names[0]), ContinuousVariable(names[1])]),
                self.embedding
            )
        else:
            output = embedding = None

        if self.embedding is not None and self.data is not None:
            domain = self.data.domain
            domain = Orange.data.Domain(domain.attributes,
                                        domain.class_vars,
                                        domain.metas + embedding.domain.attributes)
            output = self.data.transform(domain)
            output.metas[:, -2:] = embedding.X

        selection = self.graph.get_selection()
        if output is not None and len(selection) > 0:
            selected = output[selection]
        else:
            selected = None
        if self.graph.selection is not None and np.max(self.graph.selection) > 1:
            annotated = create_groups_table(output, self.graph.selection)
        else:
            annotated = create_annotated_table(output, selection)
        self.Outputs.selected_data.send(selected)
        self.Outputs.annotated_data.send(annotated)

    def onDeleteWidget(self):
        super().onDeleteWidget()
        self._clear_plot()
        self._clear()

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


    @classmethod
    def migrate_context(cls, context, version):
        if version < 2:
            domain = context.ordered_domain
            n_domain = [t for t in context.ordered_domain if t[1] == 2]
            c_domain = [t for t in context.ordered_domain if t[1] == 1]
            context_values_graph = {}
            for _, old_val, new_val in ((domain, "color_value", "attr_color"),
                                        (c_domain, "shape_value", "attr_shape"),
                                        (n_domain, "size_value", "attr_size"),
                                        (domain, "label_value", "attr_label")):
                tmp = context.values[old_val]
                if tmp[1] >= 0:
                    context_values_graph[new_val] = (tmp[0], tmp[1] + 100)
                elif tmp[0] != "Stress":
                    context_values_graph[new_val] = None
                else:
                    context_values_graph[new_val] = tmp
            context.values["graph"] = context_values_graph


def main(argv=None):
    if argv is None:
        argv = sys.argv
    import gc
    app = QApplication(list(argv))
    argv = app.arguments()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "iris"

    data = Orange.data.Table(filename)
    w = OWMDS()
    w.set_data(data)
    w.set_subset_data(data[np.random.choice(len(data), 10)])
    w.handleNewSignals()

    w.show()
    w.raise_()
    rval = app.exec_()

    w.set_subset_data(None)
    w.set_data(None)
    w.handleNewSignals()

    w.saveSettings()
    w.onDeleteWidget()
    w.deleteLater()
    del w
    gc.collect()
    app.processEvents()
    return rval

if __name__ == "__main__":
    sys.exit(main())
