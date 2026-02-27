"""
Correlations widget
"""
import warnings
from enum import IntEnum
from operator import attrgetter
from types import SimpleNamespace
from itertools import combinations, groupby, chain

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans

from AnyQt.QtCore import Qt, QItemSelectionModel, QItemSelection, \
    QSize, pyqtSignal as Signal
from AnyQt.QtGui import QStandardItem
from AnyQt.QtWidgets import QHeaderView

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.preprocess import SklImpute, Normalize, Remove
from Orange.statistics.util import FDR
from Orange.widgets import gui
from Orange.widgets.settings import Setting, ContextSetting, \
    DomainContextHandler
from Orange.widgets.utils import vartype
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.utils import VizRankDialogAttrPair
from Orange.widgets.widget import OWWidget, AttributeList, Msg

NAN = 2
SIZE_LIMIT = 1000000


class CorrelationType(IntEnum):
    """
    Correlation type enumerator. Possible correlations: Pearson, Spearman.
    """
    PEARSON, SPEARMAN = 0, 1

    @staticmethod
    def items():
        """
        Texts for correlation types. Can be used in gui controls (eg. combobox).
        """
        return ["Pearson correlation", "Spearman correlation"]


class Cluster(SimpleNamespace):
    instances = None  # type: Optional[List]
    centroid = None  # type: Optional[np.ndarray]


class KMeansCorrelationHeuristic:
    """
    Heuristic to obtain the most promising attribute pairs, when there are too
    many attributes to calculate correlations for all possible pairs.
    """
    def __init__(self, data):
        self.n_attributes = len(data.domain.attributes)
        self.data = data
        self.clusters = None
        self.n_clusters = int(np.sqrt(self.n_attributes))

    def get_clusters_of_attributes(self):
        """
        Generates groups of attribute IDs, grouped by cluster. Clusters are
        obtained by KMeans algorithm.

        :return: generator of attributes grouped by cluster
        """
        data = Normalize()(self.data).X.T
        if data.base is not None:
            data = data.copy()
        self._impute_means(data)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=1).fit(data)
        labels_attrs = sorted([(l, i) for i, l in enumerate(kmeans.labels_)])
        return [Cluster(instances=list(pair[1] for pair in group),
                        centroid=kmeans.cluster_centers_[l])
                for l, group in groupby(labels_attrs, key=lambda x: x[0])]

    @staticmethod
    def _impute_means(arr):
        nans = np.isnan(arr)
        if np.any(nans):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                means = np.nanmean(arr, axis=1)
            means = np.nan_to_num(means)
            inds = np.where(nans)
            arr[inds] = means[inds[0]]

    def get_states(self, initial_state):
        """
        Generates states (attribute pairs) - the most promising first, i.e.
        states within clusters, following by states among clusters.

        :param initial_state: initial state; None if this is the first call
        :return: generator of tuples of states
        """
        if self.clusters is None:
            self.clusters = self.get_clusters_of_attributes()
        clusters = self.clusters

        # combinations within clusters
        states0 = chain.from_iterable(combinations(cluster.instances, 2)
                                      for cluster in clusters)
        if self.n_clusters == 1:
            return states0

        # combinations among clusters - closest clusters first
        centroids = [c.centroid for c in clusters]
        centroids_combs = np.array(list(combinations(centroids, 2)))
        distances = np.linalg.norm((centroids_combs[:, 0] -
                                    centroids_combs[:, 1]), axis=1)
        cluster_combs = list(combinations(range(len(clusters)), 2))
        states = ((min((c1, c2)), max((c1, c2))) for i in np.argsort(distances)
                  for c1 in clusters[cluster_combs[i][0]].instances
                  for c2 in clusters[cluster_combs[i][1]].instances)
        states = chain(states0, states)

        if initial_state is not None:
            while next(states) != initial_state:
                pass
            return chain([initial_state], states)
        return states


class CorrelationRank(VizRankDialogAttrPair):
    """
    Correlations rank widget.
    """
    threadStopped = Signal()
    PValRole = next(gui.OrangeUserRole)
    CorrRole = next(gui.OrangeUserRole)

    def __init__(self, *args):
        super().__init__(*args)
        self.heuristic = None
        self.use_heuristic = False
        self.sel_feature_index = None

    def initialize(self):
        super().initialize()
        data = self.master.actual_data
        self.attrs = data and data.domain.attributes
        self.model_proxy.setFilterKeyColumn(-1)
        self.heuristic = None
        self.use_heuristic = False
        if self.master.feature is not None:
            self.sel_feature_index = data.domain.index(self.master.feature)
        else:
            self.sel_feature_index = None
        if data:
            # use heuristic if data is too big
            self.use_heuristic = len(data) * len(self.attrs) ** 2 > SIZE_LIMIT \
                and self.sel_feature_index is None
            if self.use_heuristic:
                self.heuristic = KMeansCorrelationHeuristic(data)

    def compute_score(self, state):
        (attr1, attr2), corr_type = state, self.master.correlation_type
        data = self.master.actual_data.X
        col1, col2 = data[:, attr1], data[:, attr2]
        mask = ~np.isnan(col1) & ~np.isnan(col2)
        if np.sum(mask) < 2:
            return np.inf, np.nan, np.nan # no valid data
        col1, col2 = col1[mask], col2[mask]
        corr = pearsonr if corr_type == CorrelationType.PEARSON else spearmanr
        r, p_value = corr(col1, col2)
        return -abs(r) if not np.isnan(r) else np.inf, r, p_value

    def row_for_state(self, score, state):
        attrs = sorted((self.attrs[x] for x in state), key=attrgetter("name"))
        attr_items = []
        for attr, halign in zip(attrs, (Qt.AlignRight, Qt.AlignLeft)):
            item = QStandardItem(attr.name)
            item.setData(attrs, self._AttrRole)
            item.setData(halign + Qt.AlignVCenter, Qt.TextAlignmentRole)
            item.setToolTip(attr.name)
            attr_items.append(item)
            if halign is Qt.AlignRight:
                colon = QStandardItem(":")
                colon.setData(Qt.AlignCenter, Qt.TextAlignmentRole)
                attr_items.append(colon)
        if np.isnan(score[1]):
            correlation_item = QStandardItem("N/A")
        else:
            correlation_item = QStandardItem(f"{score[1]:+.3f}")
            correlation_item.setData(
                self.NEGATIVE_COLOR if score[1] < 0 else self.POSITIVE_COLOR,
                gui.TableBarItem.BarColorRole)
        correlation_item.setData(score[1], self.CorrRole)
        correlation_item.setData(score[2], self.PValRole)
        correlation_item.setData(attrs, self._AttrRole)
        return [correlation_item] + attr_items

    def check_preconditions(self):
        return self.master.actual_data is not None

    def iterate_states(self, initial_state):
        if self.sel_feature_index is not None:
            return self.iterate_states_by_feature()
        elif self.use_heuristic:
            return self.heuristic.get_states(initial_state)
        else:
            return super().iterate_states(initial_state)

    def iterate_states_by_feature(self):
        for j in range(len(self.attrs)):
            if j != self.sel_feature_index:
                yield self.sel_feature_index, j

    def state_count(self):
        n = len(self.attrs)
        return n * (n - 1) / 2 if self.sel_feature_index is None else n - 1

    @staticmethod
    def bar_length(score):
        return abs(score[1])

    def stopped(self):
        self.threadStopped.emit()
        header = self.rank_table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

    def start(self, task, *args, **kwargs):
        self._set_empty_status()
        super().start(task, *args, **kwargs)
        self.__set_state_busy()

    def cancel(self):
        super().cancel()
        self.__set_state_ready()

    def _connect_signals(self, state):
        super()._connect_signals(state)
        state.progress_changed.connect(self.master.progressBarSet)
        state.status_changed.connect(self.master.setStatusMessage)

    def _disconnect_signals(self, state):
        super()._disconnect_signals(state)
        state.progress_changed.disconnect(self.master.progressBarSet)
        state.status_changed.disconnect(self.master.setStatusMessage)

    def _on_task_done(self, future):
        super()._on_task_done(future)
        self.__set_state_ready()

    def __set_state_ready(self):
        self._set_empty_status()
        self.master.setBlocking(False)

    def __set_state_busy(self):
        self.master.progressBarInit()
        self.master.setBlocking(True)

    def _set_empty_status(self):
        self.master.progressBarFinished()
        self.master.setStatusMessage("")


class OWCorrelations(OWWidget):
    name = "Correlations"
    description = "Compute all pairwise attribute correlations."
    icon = "icons/Correlations.svg"
    priority = 1106
    category = "Unsupervised"
    keywords = "pearson, spearman"

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)
        features = Output("Features", AttributeList)
        correlations = Output("Correlations", Table, dynamic=False)

    want_main_area = False
    want_control_area = True

    correlation_type: int

    settings_version = 3
    settingsHandler = DomainContextHandler()
    selection = ContextSetting([])
    feature = ContextSetting(None)
    correlation_type = Setting(0)
    impute_missing = Setting(True)

    class Information(OWWidget.Information):
        removed_cons_feat = Msg("Constant features have been removed.")

    class Error(OWWidget.Error):
        not_enough_vars = Msg("At least two numeric features are needed.")
        not_enough_inst = Msg("At least two instances are needed.")

    def __init__(self):
        super().__init__()
        self.data = None  # type: Table
        self.cont_data = None  # type: Table
        self.actual_data = None  # type: Table

        # GUI
        box = gui.vBox(self.controlArea)
        self.correlation_combo = gui.comboBox(
            box, self, "correlation_type", items=CorrelationType.items(),
            orientation=Qt.Horizontal, callback=self._correlation_combo_changed
        )

        self.feature_model = DomainModel(
            order=DomainModel.ATTRIBUTES, separators=False,
            placeholder="(All combinations)", valid_types=ContinuousVariable)
        gui.comboBox(
            box, self, "feature", callback=self._feature_combo_changed,
            model=self.feature_model, searchable=True
        )

        gui.checkBox(
            box, self, "impute_missing", "Impute missing values",
            toolTip="Replace missing values with means;\n"
                    "if disabled, rows with missing values for the corre"
                    "sponding variables are ignored",
            callback=self._impute_missing_changed
        )

        self.vizrank, _ = CorrelationRank.add_vizrank(
            None, self, None, self._vizrank_selection_changed)
        self.vizrank.button.setEnabled(False)
        self.vizrank.threadStopped.connect(self._vizrank_stopped)

        gui.separator(box)
        box.layout().addWidget(self.vizrank.filter)
        box.layout().addWidget(self.vizrank.rank_table)

        button_box = gui.hBox(self.buttonsArea)
        button_box.layout().addWidget(self.vizrank.button)

    @staticmethod
    def sizeHint():
        return QSize(350, 400)

    def _correlation_combo_changed(self):
        self.apply()

    def _feature_combo_changed(self):
        self.apply()

    def _impute_missing_changed(self):
        self.set_actual_data()
        self.apply()

    def _vizrank_selection_changed(self, *args):
        self.selection = list(args)
        self.commit()

    def _vizrank_stopped(self):
        self._vizrank_select()

    def _vizrank_select(self):
        model = self.vizrank.rank_table.model()
        if not model.rowCount():
            return
        selection = QItemSelection()

        # This flag is needed because data in the model could be
        # filtered by a feature and therefore selection could not be found
        selection_in_model = False
        if self.selection:
            sel_names = sorted(var.name for var in self.selection)
            for i in range(model.rowCount()):
                # pylint: disable=protected-access
                names = sorted(x.name for x in model.data(
                    model.index(i, 0), CorrelationRank._AttrRole))
                if names == sel_names:
                    selection.select(model.index(i, 0),
                                     model.index(i, model.columnCount() - 1))
                    selection_in_model = True
                    break
        if not selection_in_model:
            selection.select(model.index(0, 0),
                             model.index(0, model.columnCount() - 1))
        self.vizrank.rank_table.selectionModel().select(
            selection, QItemSelectionModel.ClearAndSelect)

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.clear_messages()
        self.data = data
        self.cont_data = None
        self.actual_data = None
        self.selection = []
        if data is not None:
            if len(data) < 2:
                self.Error.not_enough_inst()
            else:
                domain = data.domain
                cont_vars = [a for a in domain.class_vars + domain.metas +
                             domain.attributes if a.is_continuous]
                cont_data = Table.from_table(Domain(cont_vars), data)
                remover = Remove(Remove.RemoveConstant)
                cont_data = remover(cont_data)
                if remover.attr_results["removed"]:
                    self.Information.removed_cons_feat()
                if len(cont_data.domain.attributes) < 2:
                    self.Error.not_enough_vars()
                else:
                    self.cont_data = cont_data
        self.set_actual_data()

        if self.actual_data and data.domain.has_continuous_class:
            self.feature = self.actual_data.domain[data.domain.class_var.name]
        else:
            self.feature = None
        self.openContext(self.actual_data)
        self.apply()
        self.vizrank.button.setEnabled(self.actual_data is not None)

    def set_actual_data(self):
        if self.cont_data is None:
            self.actual_data = None
            self.feature_model.set_domain(None)
            self.feature = None
            self.vizrank.setEnabled(False)
            return

        if self.impute_missing and self.cont_data.has_missing_attribute():
            imputer = SklImpute(strategy="mean")
            self.actual_data = imputer(self.cont_data)
        else:
            self.actual_data = self.cont_data

        feature_name = self.feature and self.feature.name
        self.feature_model.set_domain(self.actual_data.domain)
        if feature_name and feature_name in self.actual_data.domain:
            self.feature = self.actual_data.domain[feature_name]
        else:
            self.feature = None

    def apply(self):
        self.vizrank.initialize()
        if self.actual_data is not None:
            # this triggers self.commit() by changing vizrank selection
            self.vizrank.toggle()
        else:
            self.commit()

    def commit(self):
        self.Outputs.data.send(self.data)

        if self.data is None or self.cont_data is None:
            self.Outputs.features.send(None)
            self.Outputs.correlations.send(None)
            return

        attrs = [ContinuousVariable("Correlation"),
                 ContinuousVariable("uncorrected p"),
                 ContinuousVariable("FDR")]
        metas = [StringVariable("Feature 1"), StringVariable("Feature 2")]
        domain = Domain(attrs, metas=metas)
        model = self.vizrank.rank_model
        count = model.rowCount()
        index = model.index
        corr_p = np.array([
            [d(CorrelationRank.CorrRole), d(CorrelationRank.PValRole)]
            for d in (index(row, 0).data for row in range(count))
        ])
        fdr = FDR(corr_p[:, 1])
        x = np.hstack((corr_p, fdr[:, np.newaxis]))
        # pylint: disable=protected-access
        m = np.array([[a.name
                       for a in index(row, 0).data(CorrelationRank._AttrRole)]
                      for row in range(count)], dtype=object)
        corr_table = Table(domain, x, metas=m)
        corr_table.name = "Correlations"

        # data has been imputed; send original attributes
        self.Outputs.features.send(AttributeList(
            [self.data.domain[var.name] for var in self.selection]))
        self.Outputs.correlations.send(corr_table)

    def send_report(self):
        self.report_table(CorrelationType.items()[self.correlation_type],
                          self.vizrank.rank_table)

    @classmethod
    def migrate_context(cls, context, version):
        if version < 2:
            sel = context.values["selection"]
            context.values["selection"] = [(var.name, vartype(var))
                                           for var in sel[0]]
        if version < 3:
            sel = context.values["selection"]
            context.values["selection"] = ([(name, vtype + 100)
                                            for name, vtype in sel], -3)


def mock_data():
    # pylint: disable=import-outside-toplevel
    from Orange.data import DiscreteVariable
    domain = Domain([DiscreteVariable("a", values="abc")]
                    + [ContinuousVariable(x) for x in "defghij"])
    n = np.nan
    s = 1 / 2
    return Table.from_numpy(
        domain,
        np.array([[0, 0, 0, 0, 1, 0],  # a
                  [1, 0, 0, 1, 0, 0],  # d 0
                  [0, 1, 1, 0, 1, 1],  # e 1
                  [1, 0, 0, 1, 0, 0],  # f 2
                  [1, 0, s, 1, 0, s],  # g 3
                  [1, 0, n, 1, 0, n],  # h 4
                  [n, 0, n, 1, 0, n],  # i 5
                  [0, n, n, n, n, 1]]  # j 6
                  ).T
    )


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWCorrelations).run(
        Table("iris")
        # mock_data()
    )
