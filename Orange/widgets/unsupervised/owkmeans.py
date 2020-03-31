from concurrent.futures import Future
from typing import Optional, List, Dict

import numpy as np
from AnyQt.QtCore import Qt, QTimer, QAbstractTableModel, QModelIndex, QThread, \
    pyqtSlot as Slot
from AnyQt.QtGui import QIntValidator
from AnyQt.QtWidgets import QGridLayout, QTableView
from sklearn.metrics import silhouette_samples, silhouette_score

from Orange.clustering import KMeans
from Orange.clustering.kmeans import KMeansModel
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.data.util import get_unique_names, array_equal
from Orange.preprocess import Normalize
from Orange.preprocess.impute import ReplaceUnknowns
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.annotated_data import \
    ANNOTATED_DATA_SIGNAL_NAME, add_columns
from Orange.widgets.utils.concurrent import ThreadExecutor, FutureSetWatcher
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output


RANDOM_STATE = 0
SILHOUETTE_MAX_SAMPLES = 5000


class ClusterTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scores = []
        self.start_k = 0

    def rowCount(self, index=QModelIndex()):
        return 0 if index.isValid() else len(self.scores)

    @staticmethod
    def columnCount(_index=QModelIndex()):
        return 1

    def flags(self, index):
        if isinstance(self.scores[index.row()], str):
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def set_scores(self, scores, start_k):
        self.modelAboutToBeReset.emit()
        self.scores = scores
        self.start_k = start_k
        self.modelReset.emit()

    def clear_scores(self):
        self.modelAboutToBeReset.emit()
        self.scores = []
        self.start_k = 0
        self.modelReset.emit()

    def data(self, index, role=Qt.DisplayRole):
        score = self.scores[index.row()]
        valid = not isinstance(score, str)
        if role == Qt.DisplayRole:
            return "{:.3f}".format(score) if valid else "NA"
        elif role == Qt.TextAlignmentRole:
            return Qt.AlignVCenter | Qt.AlignLeft
        elif role == Qt.ToolTipRole and not valid:
            return score
        elif role == gui.BarRatioRole and valid:
            return score
        return None

    def headerData(self, row, _orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return str(row + self.start_k)
        return None


class Task:
    futures = []    # type: List[Future]
    watcher = ...   # type: FutureSetWatcher
    cancelled = False

    def __init__(self, futures, watcher):
        self.futures = futures
        self.watcher = watcher

    def cancel(self):
        self.cancelled = True
        for f in self.futures:
            f.cancel()


class NotEnoughData(ValueError):
    pass


class OWKMeans(widget.OWWidget):
    name = "k-Means"
    description = "k-Means clustering algorithm with silhouette-based " \
                  "quality estimation."
    icon = "icons/KMeans.svg"
    priority = 2100
    keywords = ["kmeans", "clustering"]

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        annotated_data = Output(
            ANNOTATED_DATA_SIGNAL_NAME, Table, default=True,
            replaces=["Annotated Data"]
        )
        centroids = Output("Centroids", Table)

    class Error(widget.OWWidget.Error):
        failed = widget.Msg("Clustering failed\nError: {}")
        not_enough_data = widget.Msg(
            "Too few ({}) unique data instances for {} clusters"
        )
        no_attributes = widget.Msg("Data is missing features.")

    class Warning(widget.OWWidget.Warning):
        no_silhouettes = widget.Msg(
            "Silhouette scores are not computed for >{} samples".format(
                SILHOUETTE_MAX_SAMPLES)
        )
        not_enough_data = widget.Msg(
            "Too few ({}) unique data instances for {} clusters"
        )

    INIT_METHODS = (("Initialize with KMeans++", "k-means++"),
                    ("Random initialization", "random"))

    resizing_enabled = False
    buttons_area_orientation = Qt.Vertical

    k = Setting(3)
    k_from = Setting(2)
    k_to = Setting(8)
    optimize_k = Setting(False)
    max_iterations = Setting(300)
    n_init = Setting(10)
    smart_init = Setting(0)  # KMeans++
    selection = Setting(None, schema_only=True)  # type: Optional[int]
    auto_commit = Setting(True)
    normalize = Setting(True)

    settings_version = 2

    @classmethod
    def migrate_settings(cls, settings, version):
        # type: (Dict, int) -> None
        if version < 2:
            if 'auto_apply' in settings:
                settings['auto_commit'] = settings.get('auto_apply', True)
                settings.pop('auto_apply', None)

    def __init__(self):
        super().__init__()

        self.data = None  # type: Optional[Table]
        self.__pending_selection = self.selection  # type: Optional[int]
        self.clusterings = {}

        self.__executor = ThreadExecutor(parent=self)
        self.__task = None  # type: Optional[Task]

        layout = QGridLayout()
        bg = gui.radioButtonsInBox(
            self.controlArea, self, "optimize_k", orientation=layout,
            box="Number of Clusters", callback=self.update_method,
        )

        layout.addWidget(
            gui.appendRadioButton(bg, "Fixed:", addToLayout=False), 1, 1)
        sb = gui.hBox(None, margin=0)
        gui.spin(
            sb, self, "k", minv=2, maxv=30,
            controlWidth=60, alignment=Qt.AlignRight, callback=self.update_k)
        gui.rubber(sb)
        layout.addWidget(sb, 1, 2)

        layout.addWidget(
            gui.appendRadioButton(bg, "From", addToLayout=False), 2, 1)
        ftobox = gui.hBox(None)
        ftobox.layout().setContentsMargins(0, 0, 0, 0)
        layout.addWidget(ftobox, 2, 2)
        gui.spin(
            ftobox, self, "k_from", minv=2, maxv=29,
            controlWidth=60, alignment=Qt.AlignRight,
            callback=self.update_from)
        gui.widgetLabel(ftobox, "to")
        gui.spin(
            ftobox, self, "k_to", minv=3, maxv=30,
            controlWidth=60, alignment=Qt.AlignRight,
            callback=self.update_to)
        gui.rubber(ftobox)

        box = gui.vBox(self.controlArea, "Preprocessing")
        gui.checkBox(box, self, "normalize", "Normalize columns",
                     callback=self.invalidate)

        box = gui.vBox(self.controlArea, "Initialization")
        gui.comboBox(
            box, self, "smart_init", items=[m[0] for m in self.INIT_METHODS],
            callback=self.invalidate)

        layout = QGridLayout()
        gui.widgetBox(box, orientation=layout)
        layout.addWidget(gui.widgetLabel(None, "Re-runs: "), 0, 0, Qt.AlignLeft)
        sb = gui.hBox(None, margin=0)
        layout.addWidget(sb, 0, 1)
        gui.lineEdit(
            sb, self, "n_init", controlWidth=60,
            valueType=int, validator=QIntValidator(), callback=self.invalidate)
        layout.addWidget(
            gui.widgetLabel(None, "Maximum iterations: "), 1, 0, Qt.AlignLeft)
        sb = gui.hBox(None, margin=0)
        layout.addWidget(sb, 1, 1)
        gui.lineEdit(
            sb, self, "max_iterations", controlWidth=60, valueType=int,
            validator=QIntValidator(), callback=self.invalidate)

        self.apply_button = gui.auto_apply(self.buttonsArea, self, "auto_commit", box=None,
                                           commit=self.commit)
        gui.rubber(self.controlArea)

        box = gui.vBox(self.mainArea, box="Silhouette Scores")
        self.mainArea.setVisible(self.optimize_k)
        self.table_model = ClusterTableModel(self)
        table = self.table_view = QTableView(self.mainArea)
        table.setModel(self.table_model)
        table.setSelectionMode(QTableView.SingleSelection)
        table.setSelectionBehavior(QTableView.SelectRows)
        table.setItemDelegate(gui.ColoredBarItemDelegate(self, color=Qt.cyan))
        table.selectionModel().selectionChanged.connect(self.select_row)
        table.setMaximumWidth(200)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().hide()
        table.setShowGrid(False)
        box.layout().addWidget(table)

    def adjustSize(self):
        self.ensurePolished()
        s = self.sizeHint()
        self.resize(s)

    def update_method(self):
        self.table_model.clear_scores()
        self.commit()

    def update_k(self):
        self.optimize_k = False
        self.table_model.clear_scores()
        self.commit()

    def update_from(self):
        self.k_to = max(self.k_from + 1, self.k_to)
        self.optimize_k = True
        self.table_model.clear_scores()
        self.commit()

    def update_to(self):
        self.k_from = min(self.k_from, self.k_to - 1)
        self.optimize_k = True
        self.table_model.clear_scores()
        self.commit()

    def enough_data_instances(self, k):
        """k cannot be larger than the number of data instances."""
        return len(self.data) >= k

    @property
    def has_attributes(self):
        return len(self.data.domain.attributes)

    @staticmethod
    def _compute_clustering(data, k, init, n_init, max_iter, random_state):
        # type: (Table, int, str, int, int, bool) -> KMeansModel
        if k > len(data):
            raise NotEnoughData()

        model = KMeans(
            n_clusters=k, init=init, n_init=n_init, max_iter=max_iter,
            random_state=random_state, preprocessors=[]
        ).get_model(data)

        if data.X.shape[0] <= SILHOUETTE_MAX_SAMPLES:
            model.silhouette_samples = silhouette_samples(data.X, model.labels)
            model.silhouette = np.mean(model.silhouette_samples)
        else:
            model.silhouette_samples = None
            model.silhouette = \
                silhouette_score(data.X, model.labels,
                                 sample_size=SILHOUETTE_MAX_SAMPLES,
                                 random_state=RANDOM_STATE)

        return model

    @Slot(int, int)
    def __progress_changed(self, n, d):
        assert QThread.currentThread() is self.thread()
        assert self.__task is not None
        self.progressBarSet(100 * n / d)

    @Slot(int, Exception)
    def __on_exception(self, idx, ex):
        assert QThread.currentThread() is self.thread()
        assert self.__task is not None

        if isinstance(ex, NotEnoughData):
            self.Error.not_enough_data(len(self.data), self.k_from + idx)

        # Only show failed message if there is only 1 k to compute
        elif not self.optimize_k:
            self.Error.failed(str(ex))

        self.clusterings[self.k_from + idx] = str(ex)

    @Slot(int, object)
    def __clustering_complete(self, _, result):
        # type: (int, KMeansModel) -> None
        assert QThread.currentThread() is self.thread()
        assert self.__task is not None

        self.clusterings[result.k] = result

    @Slot()
    def __commit_finished(self):
        assert QThread.currentThread() is self.thread()
        assert self.__task is not None
        assert self.data is not None

        self.__task = None
        self.setInvalidated(False)
        self.progressBarFinished()

        if self.optimize_k:
            self.update_results()

        if self.optimize_k and all(isinstance(self.clusterings[i], str)
                                   for i in range(self.k_from, self.k_to + 1)):
            # Show the error of the last clustering
            self.Error.failed(self.clusterings[self.k_to])

        self.send_data()

    def __launch_tasks(self, ks):
        # type: (List[int]) -> None
        """Execute clustering in separate threads for all given ks."""
        preprocessed_data = self.preproces(self.data)
        futures = [self.__executor.submit(
            self._compute_clustering,
            data=preprocessed_data,
            k=k,
            init=self.INIT_METHODS[self.smart_init][1],
            n_init=self.n_init,
            max_iter=self.max_iterations,
            random_state=RANDOM_STATE,
        ) for k in ks]
        watcher = FutureSetWatcher(futures)
        watcher.resultReadyAt.connect(self.__clustering_complete)
        watcher.progressChanged.connect(self.__progress_changed)
        watcher.exceptionReadyAt.connect(self.__on_exception)
        watcher.doneAll.connect(self.__commit_finished)

        self.__task = Task(futures, watcher)
        self.progressBarInit()
        self.setInvalidated(True)

    def cancel(self):
        if self.__task is not None:
            task, self.__task = self.__task, None
            task.cancel()

            task.watcher.resultReadyAt.disconnect(self.__clustering_complete)
            task.watcher.progressChanged.disconnect(self.__progress_changed)
            task.watcher.exceptionReadyAt.disconnect(self.__on_exception)
            task.watcher.doneAll.disconnect(self.__commit_finished)

            self.progressBarFinished()
            self.setInvalidated(False)

    def run_optimization(self):
        if not self.enough_data_instances(self.k_from):
            self.Error.not_enough_data(len(self.data), self.k_from)
            return

        if not self.enough_data_instances(self.k_to):
            self.Warning.not_enough_data(len(self.data), self.k_to)
            return

        needed_ks = [k for k in range(self.k_from, self.k_to + 1)
                     if k not in self.clusterings]

        if needed_ks:
            self.__launch_tasks(needed_ks)
        else:
            # If we don't need to recompute anything, just set the results to
            # what they were before
            self.update_results()

    def cluster(self):
        # Check if the k already has a computed clustering
        if self.k in self.clusterings:
            self.send_data()
            return

        # Check if there is enough data
        if not self.enough_data_instances(self.k):
            self.Error.not_enough_data(len(self.data), self.k)
            return

        self.__launch_tasks([self.k])

    def commit(self):
        self.cancel()
        self.clear_messages()

        # Some time may pass before the new scores are computed, so clear the
        # old scores to avoid potential confusion. Hiding the mainArea could
        # cause flickering when the clusters are computed quickly, so this is
        # the better alternative
        self.table_model.clear_scores()
        self.mainArea.setVisible(self.optimize_k and self.data is not None and
                                 self.has_attributes)

        if self.data is None:
            self.send_data()
            return

        if not self.has_attributes:
            self.Error.no_attributes()
            self.send_data()
            return

        if self.optimize_k:
            self.run_optimization()
        else:
            self.cluster()

        QTimer.singleShot(100, self.adjustSize)

    def invalidate(self, unconditional=False):
        self.cancel()
        self.Error.clear()
        self.Warning.clear()
        self.clusterings = {}
        self.table_model.clear_scores()

        if unconditional:
            self.unconditional_commit()
        else:
            self.commit()

    def update_results(self):
        scores = [mk if isinstance(mk, str) else mk.silhouette for mk in
                  (self.clusterings[k] for k in range(self.k_from, self.k_to + 1))]
        best_row = max(
            range(len(scores)), default=0,
            key=lambda x: 0 if isinstance(scores[x], str) else scores[x]
        )
        self.table_model.set_scores(scores, self.k_from)
        self.apply_selection(best_row)
        self.table_view.setFocus(Qt.OtherFocusReason)
        self.table_view.resizeRowsToContents()

    def apply_selection(self, best_row):
        pending = best_row
        if self.__pending_selection is not None:
            pending = self.__pending_selection
            self.__pending_selection = None
        self.table_view.selectRow(pending)

    def selected_row(self):
        indices = self.table_view.selectedIndexes()
        if not indices:
            return None
        return indices[0].row()

    def select_row(self):
        self.selection = self.selected_row()
        self.send_data()

    def preproces(self, data):
        if self.normalize:
            data = Normalize()(data)
        for preprocessor in KMeans.preprocessors:  # use same preprocessors than
            data = preprocessor(data)
        return data

    def send_data(self):
        if self.optimize_k:
            row = self.selected_row()
            k = self.k_from + row if row is not None else None
        else:
            k = self.k

        km = self.clusterings.get(k)
        if self.data is None or km is None or isinstance(km, str):
            self.Outputs.annotated_data.send(None)
            self.Outputs.centroids.send(None)
            return

        domain = self.data.domain
        cluster_var = DiscreteVariable(
            get_unique_names(domain, "Cluster"),
            values=["C%d" % (x + 1) for x in range(km.k)]
        )
        clust_ids = km.labels
        silhouette_var = ContinuousVariable(
            get_unique_names(domain, "Silhouette"))
        if km.silhouette_samples is not None:
            self.Warning.no_silhouettes.clear()
            scores = np.arctan(km.silhouette_samples) / np.pi + 0.5
            clust_scores = []
            for i in range(km.k):
                in_clust = clust_ids == i
                if in_clust.any():
                    clust_scores.append(np.mean(scores[in_clust]))
                else:
                    clust_scores.append(0.)
            clust_scores = np.atleast_2d(clust_scores).T
        else:
            self.Warning.no_silhouettes()
            scores = np.nan
            clust_scores = np.full((km.k, 1), np.nan)

        new_domain = add_columns(domain, metas=[cluster_var, silhouette_var])
        new_table = self.data.transform(new_domain)
        new_table.get_column_view(cluster_var)[0][:] = clust_ids
        new_table.get_column_view(silhouette_var)[0][:] = scores

        domain_attributes = set(domain.attributes)
        centroid_attributes = [
            attr.compute_value.variable
            if isinstance(attr.compute_value, ReplaceUnknowns)
            and attr.compute_value.variable in domain_attributes
            else attr
            for attr in km.domain.attributes]
        centroid_domain = add_columns(
            Domain(centroid_attributes, [], domain.metas),
            metas=[cluster_var, silhouette_var])
        centroids = Table(
            centroid_domain, km.centroids, None,
            np.hstack((np.full((km.k, len(domain.metas)), np.nan),
                       np.arange(km.k).reshape(km.k, 1),
                       clust_scores))
        )
        if self.data.name == Table.name:
            centroids.name = "centroids"
        else:
            centroids.name = f"{self.data.name} centroids"

        self.Outputs.annotated_data.send(new_table)
        self.Outputs.centroids.send(centroids)

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        self.data, old_data = data, self.data
        self.selection = None

        # Do not needlessly recluster the data if X hasn't changed
        if old_data and self.data and array_equal(self.data.X, old_data.X):
            if self.auto_commit:
                self.send_data()
        else:
            self.invalidate(unconditional=True)

    def send_report(self):
        # False positives (Setting is not recognized as int)
        # pylint: disable=invalid-sequence-index
        if self.optimize_k and self.selected_row() is not None:
            k_clusters = self.k_from + self.selected_row()
        else:
            k_clusters = self.k
        init_method = self.INIT_METHODS[self.smart_init][0]
        init_method = init_method[0].lower() + init_method[1:]
        self.report_items((
            ("Number of clusters", k_clusters),
            ("Optimization", "{}, {} re-runs limited to {} steps".format(
                init_method, self.n_init, self.max_iterations))))
        if self.data is not None:
            self.report_data("Data", self.data)
            if self.optimize_k:
                self.report_table(
                    "Silhouette scores for different numbers of clusters",
                    self.table_view)

    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWKMeans).run(Table("heart_disease"))
