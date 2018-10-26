from functools import partial

from concurrent import futures
from concurrent.futures import Future

from types import SimpleNamespace as namespace
from typing import Optional, Callable, Tuple, Any

import numpy as np
import networkx as nx

from AnyQt.QtCore import (
    Qt, QObject, QTimer, pyqtSignal as Signal, pyqtSlot as Slot
)
from AnyQt.QtWidgets import QSlider, QCheckBox, QWidget

from Orange.clustering.louvain import table_to_knn_graph, Louvain
from Orange.data import Table, DiscreteVariable
from Orange.projection import PCA
from Orange.widgets import widget, gui, report
from Orange.widgets.settings import DomainContextHandler, ContextSetting, \
    Setting
from Orange.widgets.utils.annotated_data import get_next_name, add_columns, \
    ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.utils.concurrent import FutureWatcher
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.widget import Msg

try:
    from orangecontrib.network.network import Graph
except ImportError:
    Graph = None


_MAX_PCA_COMPONENTS = 50
_DEFAULT_PCA_COMPONENTS = 25
_MAX_K_NEIGBOURS = 200
_DEFAULT_K_NEIGHBORS = 30


METRICS = [('Euclidean', 'l2'), ('Manhattan', 'l1')]


class OWLouvainClustering(widget.OWWidget):
    name = 'Louvain Clustering'
    description = 'Detects communities in a network of nearest neighbors.'
    icon = 'icons/LouvainClustering.svg'
    priority = 2110

    want_main_area = False

    settingsHandler = DomainContextHandler()

    class Inputs:
        data = Input('Data', Table, default=True)

    if Graph is not None:
        class Outputs:
            annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table, default=True)
            graph = Output('Network', Graph)
    else:
        class Outputs:
            annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table, default=True)

    apply_pca = ContextSetting(True)
    pca_components = ContextSetting(_DEFAULT_PCA_COMPONENTS)
    metric_idx = ContextSetting(0)
    k_neighbors = ContextSetting(_DEFAULT_K_NEIGHBORS)
    resolution = ContextSetting(1.)
    auto_commit = Setting(False)

    class Information(widget.OWWidget.Information):
        modified = Msg("Press commit to recompute clusters and send new data")

    class Error(widget.OWWidget.Error):
        empty_dataset = Msg('No features in data')
        general_error = Msg('Error occured during clustering\n{}')

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

        pca_box = gui.vBox(self.controlArea, 'PCA Preprocessing')
        self.apply_pca_cbx = gui.checkBox(
            pca_box, self, 'apply_pca', label='Apply PCA preprocessing',
            callback=self._invalidate_graph,
        )  # type: QCheckBox
        self.pca_components_slider = gui.hSlider(
            pca_box, self, 'pca_components', label='Components: ', minValue=2,
            maxValue=_MAX_PCA_COMPONENTS,
            callback=self._invalidate_pca_projection, tracking=False
        )  # type: QSlider

        graph_box = gui.vBox(self.controlArea, 'Graph parameters')
        self.metric_combo = gui.comboBox(
            graph_box, self, 'metric_idx', label='Distance metric',
            items=[m[0] for m in METRICS], callback=self._invalidate_graph,
            orientation=Qt.Horizontal,
        )  # type: gui.OrangeComboBox
        self.k_neighbors_spin = gui.spin(
            graph_box, self, 'k_neighbors', minv=1, maxv=_MAX_K_NEIGBOURS,
            label='k neighbors', controlWidth=80, alignment=Qt.AlignRight,
            callback=self._invalidate_graph,
        )  # type: gui.SpinBoxWFocusOut
        self.resolution_spin = gui.hSlider(
            graph_box, self, 'resolution', minValue=0, maxValue=5., step=1e-1,
            label='Resolution', intOnly=False, labelFormat='%.1f',
            callback=self._invalidate_partition, tracking=False,
        )  # type: QSlider
        self.resolution_spin.parent().setToolTip(
            'The resolution parameter affects the number of clusters to find. '
            'Smaller values tend to produce more clusters and larger values '
            'retrieve less clusters.'
        )
        self.apply_button = gui.auto_commit(
            self.controlArea, self, 'auto_commit', 'Apply', box=None,
            commit=lambda: self.commit(),
            callback=lambda: self._on_auto_commit_changed(),
        )  # type: QWidget

    def _invalidate_pca_projection(self):
        self.pca_projection = None
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
        self.Error.clear()

        # Cancel current running task
        self.__cancel_task(wait=False)

        if self.data is None:
            self.__set_state_ready()
            return

        # Make sure the dataset is ok
        if len(self.data.domain.attributes) < 1:
            self.Error.empty_dataset()
            self.__set_state_ready()
            return

        if self.partition is not None:
            self.__set_state_ready()
            self._send_data()
            return

        # Preprocess the dataset
        if self.preprocessed_data is None:
            louvain = Louvain()
            self.preprocessed_data = louvain.preprocess(self.data)

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
                k_neighbors=k_neighbors, metric=metric,
                resolution=self.resolution, state=state
            )
        else:
            task = partial(
                run_on_graph, graph, resolution=self.resolution, state=state
            )

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
        # type: (Future['Results']) -> None
        assert future.done()
        assert self.__task is not None
        assert self.__task.future is future
        assert self.__task.watcher.future() is future
        self.__task, task = None, self.__task
        task.deleteLater()

        self.__set_state_ready()
        try:
            result = future.result()
        except Exception as err:  # pylint: disable=broad-except
            self.Error.general_error(str(err), exc_info=True)
        else:
            self.__set_results(result)

    @Slot(str)
    def setStatusMessage(self, text):
        super().setStatusMessage(text)

    @Slot(float)
    def progressBarSet(self, value, *a, **kw):
        super().progressBarSet(value, *a, **kw)

    def __set_state_ready(self):
        self.progressBarFinished()
        self.setBlocking(False)
        self.setStatusMessage("")

    def __set_state_busy(self):
        self.progressBarInit()
        self.setBlocking(True)

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
        # type: ('Results') -> None
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
            get_next_name(domain, 'Cluster'),
            values=['C%d' % (i + 1) for i, _ in enumerate(np.unique(new_partition))]
        )

        new_domain = add_columns(domain, metas=[cluster_var])
        new_table = self.data.transform(new_domain)
        new_table.get_column_view(cluster_var)[0][:] = new_partition
        self.Outputs.annotated_data.send(new_table)

        if Graph is not None:
            graph = Graph(self.graph)
            graph.set_items(new_table)
            self.Outputs.graph.send(graph)

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.Error.clear()

        prev_data, self.data = self.data, data
        self.openContext(self.data)

        # If X hasn't changed, there's no reason to recompute clusters
        if prev_data and self.data and np.array_equal(self.data.X, prev_data.X):
            if self.auto_commit:
                self._send_data()
            return

        # Clear the outputs
        self.Outputs.annotated_data.send(None)
        if Graph is not None:
            self.Outputs.graph.send(None)

        # Clear internal state
        self.clear()
        self._invalidate_pca_projection()
        if self.data is None:
            return

        # Can't have more PCA components than the number of attributes
        n_attrs = len(data.domain.attributes)
        self.pca_components_slider.setMaximum(min(_MAX_PCA_COMPONENTS, n_attrs))
        self.pca_components_slider.setValue(min(_DEFAULT_PCA_COMPONENTS, n_attrs))
        # Can't have more k neighbors than there are data points
        self.k_neighbors_spin.setMaximum(min(_MAX_K_NEIGBOURS, len(data) - 1))
        self.k_neighbors_spin.setValue(min(_DEFAULT_K_NEIGHBORS, len(data) - 1))

        self.commit()

    def clear(self):
        self.__cancel_task(wait=False)
        self.preprocessed_data = None
        self.pca_projection = None
        self.graph = None
        self.partition = None
        self.Error.clear()
        self.Information.modified.clear()

    def onDeleteWidget(self):
        self.__cancel_task(wait=True)
        self.__executor.shutdown(True)
        self.clear()
        self.data = None
        super().onDeleteWidget()

    def send_report(self):
        pca = report.bool_str(self.apply_pca)
        if self.apply_pca:
            pca += report.plural(', {number} component{s}', self.pca_components)

        self.report_items((
            ('PCA preprocessing', pca),
            ('Metric', METRICS[self.metric_idx][0]),
            ('k neighbors', self.k_neighbors),
            ('Resolution', self.resolution),
        ))


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
    k_neighbors = None       # type: Optional[int]
    metric = None            # type: Optional[str]
    graph = None             # type: Optional[nx.Graph]
    resolution = None        # type: Optional[float]
    partition = None         # type: Optional[np.ndarray]


def run_on_data(data, pca_components, k_neighbors, metric, resolution, state):
    # type: (Table, Optional[int], int, str, float, TaskState) -> Results
    """
    Run the louvain clustering on `data`.

    state is used to report progress and partial results. Returns early if
    `task.is_interuption_requested()` returns true.

    Parameters
    ----------
    data : Table
        Data table
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
        pca_components=pca_components, k_neighbors=k_neighbors, metric=metric,
        resolution=resolution,
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

    def pcallback(val):
        state.set_progress_value((100. * step + 100 * val) / steps)
        if state.is_interuption_requested():
            raise InteruptRequested()

    try:
        res.graph = graph = table_to_knn_graph(
            data, k_neighbors=k_neighbors, metric=metric,
            progress_callback=pcallback
        )
    except InteruptRequested:
        return res

    state.set_partial_results(("graph", res.graph))

    step += 1
    state.set_progress_value(100 * step / steps)
    state.set_status("Detecting communities...")
    if state.is_interuption_requested():
        return res

    louvain = Louvain(resolution=resolution)
    res.partition = louvain.fit_predict(graph)
    state.set_partial_results(("partition", res.partition))
    return res


def run_on_graph(graph, resolution, state):
    # type: (nx.Graph, float, TaskState) -> Results
    """
    Run the louvain clustering on `graph`.
    """
    state = state  # type: TaskState
    res = Results(resolution=resolution)
    louvain = Louvain(resolution=resolution)
    state.set_status("Detecting communities...")
    if state.is_interuption_requested():
        return res
    partition = louvain.fit_predict(graph)
    res.partition = partition
    state.set_partial_results(("partition", res.partition))
    return res


if __name__ == '__main__':
    from AnyQt.QtWidgets import QApplication  # pylint: disable=ungrouped-imports
    import sys

    app = QApplication(sys.argv)
    ow = OWLouvainClustering()
    ow.resetSettings()

    ow.set_data(Table(sys.argv[1] if len(sys.argv) > 1 else 'iris'))
    ow.show()
    app.exec_()
