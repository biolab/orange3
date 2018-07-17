from collections import deque
from concurrent.futures import Future  # pylint: disable=unused-import
from enum import Enum

import networkx as nx  # pylint: disable=unused-import
import numpy as np
from AnyQt.QtCore import Qt, pyqtSignal as Signal, QObject
from AnyQt.QtWidgets import QSlider, QCheckBox, QWidget  # pylint: disable=unused-import
from types import SimpleNamespace as namespace
from typing import Optional  # pylint: disable=unused-import

from Orange.clustering.louvain import table_to_knn_graph, Louvain
from Orange.data import Table, DiscreteVariable
from Orange.projection import PCA
from Orange.widgets import widget, gui
from Orange.widgets.settings import DomainContextHandler, ContextSetting, \
    Setting
from Orange.widgets.utils.annotated_data import get_next_name, add_columns, \
    ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.utils.concurrent import ThreadExecutor
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.widget import Msg

try:
    from orangecontrib.network.network import Graph
except:
    Graph = None


_MAX_PCA_COMPONENTS = 50
_DEFAULT_PCA_COMPONENTS = 25
_MAX_K_NEIGBOURS = 200
_DEFAULT_K_NEIGHBORS = 30


METRICS = [('Euclidean', 'l2'), ('Manhattan', 'l1')]


class TaskQueue(QObject):
    """Not really a task queue `per-se`. Running start will run the tasks in
    the current list and cannot handle adding other tasks while running."""
    on_exception = Signal(Exception)
    on_complete = Signal()
    on_progress = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.__tasks = deque()
        self.__progress = 0

    def push(self, task):
        self.__tasks.append(task)

    def __set_progress(self, progress):
        # Only emit progress signal when the progress has changed sufficiently
        if int(progress * 100) > int(self.__progress * 100):
            self.on_progress.emit(progress)
        self.__progress = progress

    def start(self):
        num_tasks = len(self.__tasks)

        for idx, task_spec in enumerate(self.__tasks):

            def __task_progress(percentage):
                current_progress = idx / num_tasks
                # How much progress can each task contribute to the total
                # work to be done
                task_percentage = len(self.__tasks) ** -1
                # Convert the progress done by the task into the total
                # progress to the task
                relative_progress = task_percentage * percentage
                self.__set_progress(current_progress + relative_progress)

            try:
                if getattr(task_spec, 'progress_callback', False):
                    task_spec.task(progress_callback=__task_progress)
                else:
                    task_spec.task()
                self.__set_progress((idx + 1) / num_tasks)

            except Exception as e:
                self.on_exception.emit(e)
                break

        self.on_complete.emit()


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
    auto_commit = Setting(True)

    class Error(widget.OWWidget.Error):
        empty_dataset = Msg('No features in data')
        general_error = Msg('Error occured during clustering\n{}')

    class State(Enum):
        Pending, Running = range(2)

    def __init__(self):
        super().__init__()

        self.data = None  # type: Optional[Table]
        self.preprocessed_data = None  # type: Optional[Table]
        self.graph = None  # type: Optional[nx.Graph]
        self.partition = None  # type: Optional[np.array]

        self.__executor = ThreadExecutor(parent=self)
        self.__future = None  # type: Optional[Future]
        self.__state = self.State.Pending

        pca_box = gui.vBox(self.controlArea, 'PCA Preprocessing')
        self.apply_pca_cbx = gui.checkBox(
            pca_box, self, 'apply_pca', label='Apply PCA preprocessing',
            callback=self._update_graph,
        )  # type: QCheckBox
        self.pca_components_slider = gui.hSlider(
            pca_box, self, 'pca_components', label='Components: ', minValue=2,
            maxValue=_MAX_PCA_COMPONENTS,
        )  # type: QSlider
        self.pca_components_slider.sliderReleased.connect(self._update_pca_components)

        graph_box = gui.vBox(self.controlArea, 'Graph parameters')
        self.metric_combo = gui.comboBox(
            graph_box, self, 'metric_idx', label='Distance metric',
            items=[m[0] for m in METRICS], callback=self._update_graph,
            orientation=Qt.Horizontal,
        )  # type: gui.OrangeComboBox
        self.k_neighbors_spin = gui.spin(
            graph_box, self, 'k_neighbors', minv=1, maxv=_MAX_K_NEIGBOURS,
            label='k neighbors', controlWidth=80, alignment=Qt.AlignRight,
            callback=self._update_graph,
        )  # type: gui.SpinBoxWFocusOut
        self.resolution_spin = gui.hSlider(
            graph_box, self, 'resolution', minValue=0, maxValue=5., step=1e-1,
            label='Resolution', intOnly=False, labelFormat='%.1f',
        )  # type: QSlider
        self.resolution_spin.sliderReleased.connect(self._update_resolution)
        self.resolution_spin.parent().setToolTip(
            'The resolution parameter affects the number of clusters to find. '
            'Smaller values tend to produce more clusters and larger values '
            'retrieve less clusters.'
        )

        self.apply_button = gui.auto_commit(
            self.controlArea, self, 'auto_commit', 'Apply', box=None,
            commit=self.commit,
        )  # type: QWidget

    def _update_graph(self):
        self._invalidate_graph()
        self.commit()

    def _update_pca_components(self):
        self._invalidate_pca_projection()
        self.commit()

    def _update_resolution(self):
        self._invalidate_partition()
        self.commit()

    def _compute_pca_projection(self):
        if self.pca_projection is None and self.apply_pca:
            self.setStatusMessage('Computing PCA...')

            pca = PCA(n_components=self.pca_components, random_state=0)
            model = pca(self.preprocessed_data)
            self.pca_projection = model(self.preprocessed_data)

    def _compute_graph(self, progress_callback=None):
        if self.graph is None:
            self.setStatusMessage('Building graph...')

            data = self.pca_projection if self.apply_pca else self.preprocessed_data

            self.graph = table_to_knn_graph(
                data, k_neighbors=self.k_neighbors,
                metric=METRICS[self.metric_idx][1],
                progress_callback=progress_callback,
            )

    def _compute_partition(self):
        if self.partition is None:
            self.setStatusMessage('Detecting communities...')
            self.setBlocking(True)

            louvain = Louvain(resolution=self.resolution)
            self.partition = louvain.fit_predict(self.graph)

    def _processing_complete(self):
        self.setStatusMessage('')
        self.setBlocking(False)
        self.progressBarFinished()

    def _handle_exceptions(self, ex):
        self.Error.general_error(str(ex))

    def cancel(self):
        """Cancel any running jobs."""
        if self.__state == self.State.Running:
            assert self.__future is not None
            self.__future.cancel()
            self.__future = None

        self.__state = self.State.Pending

    def commit(self):
        self.Error.clear()
        # Kill any running jobs
        self.cancel()
        assert self.__state == self.State.Pending

        if self.data is None:
            return

        # Make sure the dataset is ok
        if len(self.data.domain.attributes) < 1:
            self.Error.empty_dataset()
            return

        # Preprocess the dataset
        if self.preprocessed_data is None:
            louvain = Louvain()
            self.preprocessed_data = louvain.preprocess(self.data)

        # Prepare the tasks to run
        queue = TaskQueue(parent=self)

        if self.pca_projection is None and self.apply_pca:
            queue.push(namespace(task=self._compute_pca_projection))

        if self.graph is None:
            queue.push(namespace(task=self._compute_graph, progress_callback=True))

        if self.partition is None:
            queue.push(namespace(task=self._compute_partition))

        # Prepare callbacks
        queue.on_progress.connect(lambda val: self.progressBarSet(100 * val))
        queue.on_complete.connect(self._processing_complete)
        queue.on_complete.connect(self._send_data)
        queue.on_exception.connect(self._handle_exceptions)

        # Run the task queue
        self.progressBarInit()
        self.setBlocking(True)
        self.__future = self.__executor.submit(queue.start)
        self.__state = self.State.Running

    def _send_data(self):
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

    def _invalidate_pca_projection(self):
        self.pca_projection = None
        self._invalidate_graph()

    def _invalidate_graph(self):
        self.graph = None
        self._invalidate_partition()

    def _invalidate_partition(self):
        self.partition = None

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
        self.preprocessed_data = None
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

    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()


if __name__ == '__main__':
    from AnyQt.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    ow = OWLouvainClustering()
    ow.resetSettings()

    ow.set_data(Table(sys.argv[1] if len(sys.argv) > 1 else 'iris'))
    ow.show()
    app.exec_()
