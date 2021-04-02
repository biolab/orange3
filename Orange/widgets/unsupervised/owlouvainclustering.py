from functools import partial

from concurrent import futures
from concurrent.futures import Future

from types import SimpleNamespace as namespace
from typing import Optional, Callable, Tuple, Any

import numpy as np
import scipy.sparse as sp
import networkx as nx

from AnyQt.QtCore import (
    Qt, QObject, QTimer, pyqtSignal as Signal, pyqtSlot as Slot
)
from AnyQt.QtWidgets import QSlider, QCheckBox, QWidget, QLabel

from Orange.clustering.louvain import matrix_to_knn_graph, Louvain
from Orange.data import Table, DiscreteVariable
from Orange.data.util import get_unique_names, array_equal
from Orange import preprocess
from Orange.projection import PCA
from Orange.widgets import widget, gui, report
from Orange.widgets.settings import Setting
from Orange.widgets.utils.annotated_data import add_columns, \
    ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.utils.concurrent import FutureWatcher
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg

try:
    from orangecontrib.network.network import Network
except ImportError:
    Network = None


_MAX_PCA_COMPONENTS = 50
_DEFAULT_PCA_COMPONENTS = 25
_MAX_K_NEIGBOURS = 200
_DEFAULT_K_NEIGHBORS = 30


METRICS = [("Euclidean", "l2"), ("Manhattan", "l1"), ("Cosine", "cosine")]


class OWLouvainClustering(widget.OWWidget):
    name = "Louvain Clustering"
    description = "Detects communities in a network of nearest neighbors."
    icon = "icons/LouvainClustering.svg"
    priority = 2110

    settings_version = 2

    want_main_area = False
    resizing_enabled = False

    class Inputs:
        data = Input("Data", Table, default=True)

    class Outputs:
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table, default=True)
        if Network is not None:
            graph = Output("Network", Network)

    apply_pca = Setting(True)
    pca_components = Setting(_DEFAULT_PCA_COMPONENTS)
    normalize = Setting(True)
    metric_idx = Setting(0)
    k_neighbors = Setting(_DEFAULT_K_NEIGHBORS)
    resolution = Setting(1.)
    auto_commit = Setting(False)

    class Information(widget.OWWidget.Information):
        modified = Msg("Press commit to recompute clusters and send new data")

    class Error(widget.OWWidget.Error):
        empty_dataset = Msg("No features in data")

    def __init__(self):
        super().__init__()

        self.data = None  # type: Optional[Table]
        self.preprocessed_data = None  # type: Optional[Table]
        self.pca_projection = None  # type: Optional[Table]
        self.graph = None  # type: Optional[nx.Graph]
        self.partition = None  # type: Optional[np.array]
        # Use a executor with a single worker, to limit CPU overcommitment for
        # cancelled tasks. The method does not have a fine cancellation
        # granularity so we assure that there are not N - 1 jobs executing
        # for no reason only to be thrown away. It would be better to use the
        # global pool but implement a limit on jobs from this source.
        self.__executor = futures.ThreadPoolExecutor(max_workers=1)
        self.__task = None   # type: Optional[TaskState]
        self.__invalidated = False
        # coalescing commit timer
        self.__commit_timer = QTimer(self, singleShot=True)
        self.__commit_timer.timeout.connect(self.commit)

        # Set up UI
        info_box = gui.vBox(self.controlArea, "Info")
        self.info_label = gui.widgetLabel(info_box, "No data on input.")  # type: QLabel

        preprocessing_box = gui.vBox(self.controlArea, "Preprocessing")
        self.normalize_cbx = gui.checkBox(
            preprocessing_box, self, "normalize", label="Normalize data",
            callback=self._invalidate_preprocessed_data, attribute=Qt.WA_LayoutUsesWidgetRect
        )  # type: QCheckBox
        self.apply_pca_cbx = gui.checkBox(
            preprocessing_box, self, "apply_pca", label="Apply PCA preprocessing",
            callback=self._apply_pca_changed, attribute=Qt.WA_LayoutUsesWidgetRect
        )  # type: QCheckBox
        self.pca_components_slider = gui.hSlider(
            preprocessing_box, self, "pca_components", label="PCA Components: ", minValue=2,
            maxValue=_MAX_PCA_COMPONENTS,
            callback=self._invalidate_pca_projection, tracking=False
        )  # type: QSlider

        graph_box = gui.vBox(self.controlArea, "Graph parameters")
        self.metric_combo = gui.comboBox(
            graph_box, self, "metric_idx", label="Distance metric",
            items=[m[0] for m in METRICS], callback=self._invalidate_graph,
            orientation=Qt.Horizontal,
        )
        self.k_neighbors_spin = gui.spin(
            graph_box, self, "k_neighbors", minv=1, maxv=_MAX_K_NEIGBOURS,
            label="k neighbors", controlWidth=80, alignment=Qt.AlignRight,
            callback=self._invalidate_graph,
        )
        self.resolution_spin = gui.hSlider(
            graph_box, self, "resolution", minValue=0, maxValue=5., step=1e-1,
            label="Resolution", intOnly=False, labelFormat="%.1f",
            callback=self._invalidate_partition, tracking=False,
        )  # type: QSlider
        self.resolution_spin.parent().setToolTip(
            "The resolution parameter affects the number of clusters to find. "
            "Smaller values tend to produce more clusters and larger values "
            "retrieve less clusters."
        )
        self.apply_button = gui.auto_apply(
            self.buttonsArea, self, "auto_commit",
            commit=lambda: self.commit(), callback=lambda: self._on_auto_commit_changed()
        )  # type: QWidget

    def _preprocess_data(self):
        if self.preprocessed_data is None:
            if self.normalize:
                normalizer = preprocess.Normalize(center=False)
                self.preprocessed_data = normalizer(self.data)
            else:
                self.preprocessed_data = self.data

    def _apply_pca_changed(self):
        self.controls.pca_components.setEnabled(self.apply_pca)
        self._invalidate_graph()

    def _invalidate_preprocessed_data(self):
        self.preprocessed_data = None
        self._invalidate_pca_projection()
        # If we don't apply PCA, this still invalidates the graph, otherwise
        # this change won't be propagated further
        if not self.apply_pca:
            self._invalidate_graph()

    def _invalidate_pca_projection(self):
        self.pca_projection = None
        if not self.apply_pca:
            return

        self._invalidate_graph()
        self._set_modified(True)

    def _invalidate_graph(self):
        self.graph = None
        self._invalidate_partition()
        self._set_modified(True)

    def _invalidate_partition(self):
        self.partition = None
        self._invalidate_output()
        self.Information.modified()
        self._set_modified(True)

    def _invalidate_output(self):
        self.__invalidated = True
        if self.__task is not None:
            self.__cancel_task(wait=False)

        if self.auto_commit:
            self.__commit_timer.start()
        else:
            self.__set_state_ready()

    def _set_modified(self, state):
        """
        Mark the widget (GUI) as containing modified state.
        """
        if self.data is None:
            # does not apply when we have no data
            state = False
        elif self.auto_commit:
            # does not apply when auto commit is on
            state = False
        self.Information.modified(shown=state)

    def _on_auto_commit_changed(self):
        if self.auto_commit and self.__invalidated:
            self.commit()

    def cancel(self):
        """Cancel any running jobs."""
        self.__cancel_task(wait=False)
        self.__set_state_ready()

    def commit(self):
        self.__commit_timer.stop()
        self.__invalidated = False
        self._set_modified(False)

        # Cancel current running task
        self.__cancel_task(wait=False)

        if self.data is None:
            self.__set_state_ready()
            return

        self.Error.clear()

        if self.partition is not None:
            self.__set_state_ready()
            self._send_data()
            return

        self._preprocess_data()

        state = TaskState(self)

        # Prepare/assemble the task(s) to run; reuse partial results
        if self.apply_pca:
            if self.pca_projection is not None:
                data = self.pca_projection
                pca_components = None
            else:
                data = self.preprocessed_data
                pca_components = self.pca_components
        else:
            data = self.preprocessed_data
            pca_components = None

        if self.graph is not None:
            # run on graph only; no need to do PCA and k-nn search ...
            graph = self.graph
            k_neighbors = metric = None
        else:
            k_neighbors, metric = self.k_neighbors, METRICS[self.metric_idx][1]
            graph = None

        if graph is None:
            task = partial(
                run_on_data, data, pca_components=pca_components,
                normalize=self.normalize, k_neighbors=k_neighbors,
                metric=metric, resolution=self.resolution, state=state,
            )
        else:
            task = partial(
                run_on_graph, graph, resolution=self.resolution, state=state
            )

        self.info_label.setText("Running...")
        self.__set_state_busy()
        self.__start_task(task, state)

    @Slot(object)
    def __set_partial_results(self, result):
        # type: (Tuple[str, Any]) -> None
        which, res = result
        if which == "pca_projection":
            assert isinstance(res, Table) and len(res) == len(self.data)
            self.pca_projection = res
        elif which == "graph":
            assert isinstance(res, nx.Graph)
            self.graph = res
        elif which == "partition":
            assert isinstance(res, np.ndarray)
            self.partition = res
        else:
            assert False, which

    @Slot(object)
    def __on_done(self, future):
        # type: (Future["Results"]) -> None
        assert future.done()
        assert self.__task is not None
        assert self.__task.future is future
        assert self.__task.watcher.future() is future
        self.__task, task = None, self.__task
        task.deleteLater()

        self.__set_state_ready()

        result = future.result()
        self.__set_results(result)

    @Slot(str)
    def setStatusMessage(self, text):
        super().setStatusMessage(text)

    @Slot(float)
    def progressBarSet(self, value, *a, **kw):
        super().progressBarSet(value, *a, **kw)

    def __set_state_ready(self):
        self.progressBarFinished()
        self.setInvalidated(False)
        self.setStatusMessage("")

    def __set_state_busy(self):
        self.progressBarInit()
        self.setInvalidated(True)

    def __start_task(self, task, state):
        # type: (Callable[[], Any], TaskState) -> None
        assert self.__task is None
        state.status_changed.connect(self.setStatusMessage)
        state.progress_changed.connect(self.progressBarSet)
        state.partial_result_ready.connect(self.__set_partial_results)
        state.watcher.done.connect(self.__on_done)
        state.start(self.__executor, task)
        state.setParent(self)
        self.__task = state

    def __cancel_task(self, wait=True):
        # Cancel and dispose of the current task
        if self.__task is not None:
            state, self.__task = self.__task, None
            state.cancel()
            state.partial_result_ready.disconnect(self.__set_partial_results)
            state.status_changed.disconnect(self.setStatusMessage)
            state.progress_changed.disconnect(self.progressBarSet)
            state.watcher.done.disconnect(self.__on_done)
            if wait:
                futures.wait([state.future])
                state.deleteLater()
            else:
                w = FutureWatcher(state.future, parent=state)
                w.done.connect(state.deleteLater)

    def __set_results(self, results):
        # type: ("Results") -> None
        # NOTE: All of these have already been set by __set_partial_results,
        # we double check that they are aliases
        if results.pca_projection is not None:
            assert self.pca_components == results.pca_components
            assert self.pca_projection is results.pca_projection
            self.pca_projection = results.pca_projection
        if results.graph is not None:
            assert results.metric == METRICS[self.metric_idx][1]
            assert results.k_neighbors == self.k_neighbors
            assert self.graph is results.graph
            self.graph = results.graph
        if results.partition is not None:
            assert results.resolution == self.resolution
            assert self.partition is results.partition
            self.partition = results.partition

        # Display the number of found clusters in the UI
        num_clusters = len(np.unique(self.partition))
        self.info_label.setText("%d clusters found." % num_clusters)

        self._send_data()

    def _send_data(self):
        if self.partition is None or self.data is None:
            return
        domain = self.data.domain
        # Compute the frequency of each cluster index
        counts = np.bincount(self.partition)
        indices = np.argsort(counts)[::-1]
        index_map = {n: o for n, o in zip(indices, range(len(indices)))}
        new_partition = list(map(index_map.get, self.partition))

        cluster_var = DiscreteVariable(
            get_unique_names(domain, "Cluster"),
            values=["C%d" % (i + 1) for i, _ in enumerate(np.unique(new_partition))]
        )

        new_domain = add_columns(domain, metas=[cluster_var])
        new_table = self.data.transform(new_domain)
        with new_table.unlocked(new_table.metas):
            new_table.get_column_view(cluster_var)[0][:] = new_partition

        self.Outputs.annotated_data.send(new_table)

        if Network is not None:
            n_edges = self.graph.number_of_edges()
            edges = sp.coo_matrix(
                (np.ones(n_edges), np.array(self.graph.edges()).T),
                shape=(n_edges, n_edges))
            graph = Network(new_table, edges)
            self.Outputs.graph.send(graph)

    @Inputs.data
    def set_data(self, data):
        self.Error.clear()

        prev_data, self.data = self.data, data
        # Make sure to properly enable/disable slider based on `apply_pca` setting
        self.controls.pca_components.setEnabled(self.apply_pca)

        if prev_data and self.data and array_equal(prev_data.X, self.data.X):
            if self.auto_commit and not self.isInvalidated():
                self._send_data()
            return

        self.cancel()
        # Clear the outputs
        self.Outputs.annotated_data.send(None)
        if Network is not None:
            self.Outputs.graph.send(None)

        # Clear internal state
        self.clear()
        self._invalidate_pca_projection()

        # Make sure the dataset is ok
        if self.data is not None and len(self.data.domain.attributes) < 1:
            self.Error.empty_dataset()
            self.data = None

        if self.data is None:
            return

        # Can't have more PCA components than the number of attributes
        n_attrs = len(data.domain.attributes)
        self.pca_components_slider.setMaximum(min(_MAX_PCA_COMPONENTS, n_attrs))
        # Can't have more k neighbors than there are data points
        self.k_neighbors_spin.setMaximum(min(_MAX_K_NEIGBOURS, len(data) - 1))

        self.info_label.setText("Clustering not yet run.")

        self.commit()

    def clear(self):
        self.__cancel_task(wait=False)
        self.preprocessed_data = None
        self.pca_projection = None
        self.graph = None
        self.partition = None
        self.Error.clear()
        self.Information.modified.clear()
        self.info_label.setText("No data on input.")

    def onDeleteWidget(self):
        self.__cancel_task(wait=True)
        self.__executor.shutdown(True)
        self.clear()
        self.data = None
        super().onDeleteWidget()

    def send_report(self):
        pca = report.bool_str(self.apply_pca)
        if self.apply_pca:
            pca += report.plural(", {number} component{s}", self.pca_components)

        self.report_items((
            ("Normalize data", report.bool_str(self.normalize)),
            ("PCA preprocessing", pca),
            ("Metric", METRICS[self.metric_idx][0]),
            ("k neighbors", self.k_neighbors),
            ("Resolution", self.resolution),
        ))

    @classmethod
    def migrate_settings(cls, settings, version):
        if version < 2 and "context_settings" in settings:
            try:
                current_context = settings["context_settings"][0]
                for n in ['apply_pca', 'k_neighbors', 'metric_idx',
                          'normalize', 'pca_components', 'resolution']:
                    if n in current_context.values:
                        settings[n] = current_context.values[n][0]
            except:  # pylint: disable=bare-except
                pass
            finally:
                del settings["context_settings"]


class TaskState(QObject):

    status_changed = Signal(str)
    _p_status_changed = Signal(str)

    progress_changed = Signal(float)
    _p_progress_changed = Signal(float)

    partial_result_ready = Signal(object)
    _p_partial_result_ready = Signal(object)

    def __init__(self, *args):
        super().__init__(*args)
        self.__future = None
        self.watcher = FutureWatcher()
        self.__interuption_requested = False
        self.__progress = 0
        # Helpers to route the signal emits via a this object's queue.
        # This ensures 'atomic' disconnect from signals for targets/slots
        # in the same thread. Requires that the event loop is running in this
        # object's thread.
        self._p_status_changed.connect(
            self.status_changed, Qt.QueuedConnection)
        self._p_progress_changed.connect(
            self.progress_changed, Qt.QueuedConnection)
        self._p_partial_result_ready.connect(
            self.partial_result_ready, Qt.QueuedConnection)

    @property
    def future(self):
        # type: () -> Future
        return self.__future

    def set_status(self, text):
        self._p_status_changed.emit(text)

    def set_progress_value(self, value):
        if round(value, 1) > round(self.__progress, 1):
            # Only emit progress when it has changed sufficiently
            self._p_progress_changed.emit(value)
            self.__progress = value

    def set_partial_results(self, value):
        self._p_partial_result_ready.emit(value)

    def is_interuption_requested(self):
        return self.__interuption_requested

    def start(self, executor, func=None):
        # type: (futures.Executor, Callable[[], Any]) -> Future
        assert self.future is None
        assert not self.__interuption_requested
        self.__future = executor.submit(func)
        self.watcher.setFuture(self.future)
        return self.future

    def cancel(self):
        assert not self.__interuption_requested
        self.__interuption_requested = True
        if self.future is not None:
            rval = self.future.cancel()
        else:
            # not even scheduled yet
            rval = True
        return rval


class InteruptRequested(BaseException):
    pass


class Results(namespace):
    pca_projection = None    # type: Optional[Table]
    pca_components = None    # type: Optional[int]
    normalize = None         # type: Optional[bool]
    k_neighbors = None       # type: Optional[int]
    metric = None            # type: Optional[str]
    graph = None             # type: Optional[nx.Graph]
    resolution = None        # type: Optional[float]
    partition = None         # type: Optional[np.ndarray]


def run_on_data(data, normalize, pca_components, k_neighbors, metric, resolution, state):
    # type: (Table, Optional[int], int, str, float, bool, TaskState) -> Results
    """
    Run the louvain clustering on `data`.

    state is used to report progress and partial results. Returns early if
    `task.is_interuption_requested()` returns true.

    Parameters
    ----------
    data : Table
        Data table
    normalize : bool
        If `True`, the data is first normalized before computing PCA.
    pca_components : Optional[int]
        If not `None` then the data is first projected onto first
        `pca_components` principal components.
    k_neighbors : int
        Passed to `table_to_knn_graph`
    metric : str
        Passed to `table_to_knn_graph`
    resolution : float
        Passed to `Louvain`
    state : TaskState

    Returns
    -------
    res : Results
    """
    state = state  # type: TaskState
    res = Results(
        normalize=normalize, pca_components=pca_components,
        k_neighbors=k_neighbors, metric=metric, resolution=resolution,
    )
    step = 0
    if state.is_interuption_requested():
        return res

    if pca_components is not None:
        steps = 3
        state.set_status("Computing PCA...")
        pca = PCA(n_components=pca_components, random_state=0)

        data = res.pca_projection = pca(data)(data)
        assert isinstance(data, Table)
        state.set_partial_results(("pca_projection", res.pca_projection))
        step += 1
    else:
        steps = 2

    if state.is_interuption_requested():
        return res

    state.set_progress_value(100. * step / steps)
    state.set_status("Building graph...")

    # Apply Louvain preprocessing before converting the table into a graph
    louvain = Louvain(resolution=resolution, random_state=0)
    data = louvain.preprocess(data)

    if state.is_interuption_requested():
        return res

    def pcallback(val):
        state.set_progress_value((100. * step + 100 * val) / steps)
        if state.is_interuption_requested():
            raise InteruptRequested()

    try:
        res.graph = graph = matrix_to_knn_graph(
            data.X, k_neighbors=k_neighbors, metric=metric,
            progress_callback=pcallback)
    except InteruptRequested:
        return res

    state.set_partial_results(("graph", res.graph))

    step += 1
    state.set_progress_value(100 * step / steps)
    state.set_status("Detecting communities...")
    if state.is_interuption_requested():
        return res

    res.partition = louvain(graph)
    state.set_partial_results(("partition", res.partition))
    return res


def run_on_graph(graph, resolution, state):
    # type: (nx.Graph, float, TaskState) -> Results
    """
    Run the louvain clustering on `graph`.
    """
    state = state  # type: TaskState
    res = Results(resolution=resolution)
    louvain = Louvain(resolution=resolution, random_state=0)
    state.set_status("Detecting communities...")
    if state.is_interuption_requested():
        return res
    partition = louvain(graph)
    res.partition = partition
    state.set_partial_results(("partition", res.partition))
    return res


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWLouvainClustering).run(Table("iris"))
