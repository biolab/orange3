"""
Correlations widget
"""
from enum import IntEnum
from operator import attrgetter
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


class KMeansCorrelationHeuristic:
    """
    Heuristic to obtain the most promising attribute pairs, when there are to
    many attributes to calculate correlations for all possible pairs.
    """
    n_clusters = 10

    def __init__(self, data):
        self.n_attributes = len(data.domain.attributes)
        self.data = data
        self.states = None

    def get_clusters_of_attributes(self):
        """
        Generates groupes of attribute IDs, grouped by cluster. Clusters are
        obtained by KMeans algorithm.

        :return: generator of attributes grouped by cluster
        """
        data = Normalize()(self.data).X.T
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(data)
        labels_attrs = sorted([(l, i) for i, l in enumerate(kmeans.labels_)])
        for _, group in groupby(labels_attrs, key=lambda x: x[0]):
            group = list(group)
            if len(group) > 1:
                yield list(pair[1] for pair in group)

    def get_states(self, initial_state):
        """
        Generates the most promising states (attribute pairs).

        :param initial_state: initial state; None if this is the first call
        :return: generator of tuples of states
        """
        if self.states is not None:
            return chain([initial_state], self.states)
        self.states = chain.from_iterable(combinations(inds, 2) for inds in
                                          self.get_clusters_of_attributes())
        return self.states


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
            n_attrs = len(self.attrs)
            use_heuristic = n_attrs > KMeansCorrelationHeuristic.n_clusters
            self.use_heuristic = use_heuristic and \
                len(data) * n_attrs ** 2 > SIZE_LIMIT and \
                self.sel_feature_index is None
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
        if self.sel_feature_index is not None:
            return len(self.attrs) - 1
        elif self.use_heuristic:
            n_clusters = KMeansCorrelationHeuristic.n_clusters
            n_avg_attrs = len(self.attrs) / n_clusters
            return n_clusters * n_avg_attrs * (n_avg_attrs - 1) / 2
        else:
            n_attrs = len(self.attrs)
            return n_attrs * (n_attrs - 1) / 2

    @staticmethod
    def bar_length(score):
        return abs(score[1])

    def stopped(self):
        self.threadStopped.emit()
        header = self.rank_table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.Stretch)


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

    want_control_area = False

    correlation_type: int

    settings_version = 2
    settingsHandler = DomainContextHandler()
    selection = ContextSetting(())
    feature = ContextSetting(None)
    correlation_type = Setting(0)

    class Information(OWWidget.Information):
        removed_cons_feat = Msg("Constant features have been removed.")

    class Warning(OWWidget.Warning):
        not_enough_vars = Msg("Need at least two continuous features.")
        not_enough_inst = Msg("Need at least two instances.")

    def __init__(self):
        super().__init__()
        self.data = None
        self.cont_data = None

        # GUI
        box = gui.vBox(self.mainArea)
        self.correlation_combo = gui.comboBox(
            box, self, "correlation_type", items=CorrelationType.items(),
            orientation=Qt.Horizontal, callback=self._correlation_combo_changed
        )

        self.feature_model = DomainModel(
            separators=False, placeholder="(All combinations)",
            valid_types=ContinuousVariable,
        )
        gui.comboBox(
            box, self, "feature", callback=self._feature_combo_changed,
            model=self.feature_model
        )

        self.vizrank, _ = CorrelationRank.add_vizrank(
            None, self, None, self._vizrank_selection_changed)
        self.vizrank.progressBar = self.progressBar
        self.vizrank.button.setEnabled(False)
        self.vizrank.threadStopped.connect(self._vizrank_stopped)

        gui.separator(box)
        box.layout().addWidget(self.vizrank.filter)
        box.layout().addWidget(self.vizrank.rank_table)

        button_box = gui.hBox(self.mainArea)
        button_box.layout().addWidget(self.vizrank.button)

    @staticmethod
    def sizeHint():
        return QSize(350, 400)

    def _correlation_combo_changed(self):
        self.apply()

    def _feature_combo_changed(self):
        self.apply()

    def _vizrank_selection_changed(self, *args):
        self.selection = [(var.name, vartype(var)) for var in args]
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
            sel_names = sorted(name for name, _ in self.selection)
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
        self.selection = ()
        if data is not None:
            if len(data) < 2:
                self.Warning.not_enough_inst()
            else:
                domain = data.domain
                cont_attrs = [a for a in domain.attributes if a.is_continuous]
                cont_dom = Domain(cont_attrs, domain.class_vars, domain.metas)
                cont_data = Table.from_table(cont_dom, data)
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
        self.feature_model.set_domain(self.cont_data and self.cont_data.domain)
        self.feature = None

    def apply(self):
        self.vizrank.initialize()
        if self.cont_data is not None:
            # this triggers self.commit() by changing vizrank selection
            self.vizrank.toggle()
        else:
            self.commit()

    def commit(self):
        if self.data is None or self.cont_data is None:
            self.Outputs.data.send(self.data)
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

        self.Outputs.data.send(self.data)
        # data has been imputed; send original attributes
        self.Outputs.features.send(AttributeList(
            [self.data.domain[name] for name, _ in self.selection]))
        self.Outputs.correlations.send(corr_table)

    def send_report(self):
        self.report_table(CorrelationType.items()[self.correlation_type],
                          self.vizrank.rank_table)

    @classmethod
    def migrate_context(cls, context, version):
        if version < 2:
            sel = context.values["selection"]
            context.values["selection"] = ([(var.name, vartype(var))
                                            for var in sel[0]], sel[1])


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWCorrelations).run(Table("iris"))
