"""
Correlations widget
"""
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
        Generates groupes of attribute IDs, grouped by cluster. Clusters are
        obtained by KMeans algorithm.

        :return: generator of attributes grouped by cluster
        """
        data = Normalize()(self.data).X.T
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(data)
        labels_attrs = sorted([(l, i) for i, l in enumerate(kmeans.labels_)])
        return [Cluster(instances=list(pair[1] for pair in group),
                        centroid=kmeans.cluster_centers_[l])
                for l, group in groupby(labels_attrs, key=lambda x: x[0])]

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

    def __init__(self, *args):
        super().__init__(*args)
        self.heuristic = None
        self.use_heuristic = False
        self.sel_feature_index = None

    def initialize(self):
        super().initialize()
        data = self.master.cont_data
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
        data = self.master.cont_data.X
        corr = pearsonr if corr_type == CorrelationType.PEARSON else spearmanr
        r, p_value = corr(data[:, attr1], data[:, attr2])
        return -abs(r) if not np.isnan(r) else NAN, r, p_value

    def row_for_state(self, score, state):
        attrs = sorted((self.attrs[x] for x in state), key=attrgetter("name"))
        attr_items = []
        for attr in attrs:
            item = QStandardItem(attr.name)
            item.setData(attrs, self._AttrRole)
            item.setData(Qt.AlignLeft + Qt.AlignTop, Qt.TextAlignmentRole)
            item.setToolTip(attr.name)
            attr_items.append(item)
        correlation_item = QStandardItem("{:+.3f}".format(score[1]))
        correlation_item.setData(score[2], self.PValRole)
        correlation_item.setData(attrs, self._AttrRole)
        correlation_item.setData(
            self.NEGATIVE_COLOR if score[1] < 0 else self.POSITIVE_COLOR,
            gui.TableBarItem.BarColorRole)
        return [correlation_item] + attr_items

    def check_preconditions(self):
        return self.master.cont_data is not None

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

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)
        features = Output("Features", AttributeList)
        correlations = Output("Correlations", Table)

    want_main_area = False
    want_control_area = True

    correlation_type: int

    settings_version = 3
    settingsHandler = DomainContextHandler()
    selection = ContextSetting([])
    feature = ContextSetting(None)
    correlation_type = Setting(0)

    class Information(OWWidget.Information):
        removed_cons_feat = Msg("Constant features have been removed.")

    class Warning(OWWidget.Warning):
        not_enough_vars = Msg("At least two numeric features are needed.")
        not_enough_inst = Msg("At least two instances are needed.")

    def __init__(self):
        super().__init__()
        self.data = None  # type: Table
        self.cont_data = None  # type: Table

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
            model=self.feature_model
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
        self.selection = []
        if data is not None:
            if len(data) < 2:
                self.Warning.not_enough_inst()
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
                    self.Warning.not_enough_vars()
                else:
                    self.cont_data = SklImpute()(cont_data)
        self.set_feature_model()
        self.openContext(self.cont_data)
        self.apply()
        self.vizrank.button.setEnabled(self.cont_data is not None)

    def set_feature_model(self):
        self.feature_model.set_domain(
            self.cont_data.domain if self.cont_data else None)
        data = self.data
        if self.cont_data and data.domain.has_continuous_class:
            self.feature = self.cont_data.domain[data.domain.class_var.name]
        else:
            self.feature = None

    def apply(self):
        self.vizrank.initialize()
        if self.cont_data is not None:
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

        attrs = [ContinuousVariable("Correlation"), ContinuousVariable("FDR")]
        metas = [StringVariable("Feature 1"), StringVariable("Feature 2")]
        domain = Domain(attrs, metas=metas)
        model = self.vizrank.rank_model
        x = np.array([[float(model.data(model.index(row, 0), role))
                       for role in (Qt.DisplayRole, CorrelationRank.PValRole)]
                      for row in range(model.rowCount())])
        x[:, 1] = FDR(list(x[:, 1]))
        # pylint: disable=protected-access
        m = np.array([[a.name for a in model.data(model.index(row, 0),
                                                  CorrelationRank._AttrRole)]
                      for row in range(model.rowCount())], dtype=object)
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


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWCorrelations).run(Table("iris"))
