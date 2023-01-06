import warnings
from operator import attrgetter
from typing import Union, Dict, List

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning

from AnyQt.QtWidgets import QHeaderView, QStyledItemDelegate, QMenu, \
    QApplication, QToolButton
from AnyQt.QtGui import QStandardItemModel, QStandardItem, QClipboard, QColor
from AnyQt.QtCore import Qt, QSize, QObject, pyqtSignal as Signal, \
    QSortFilterProxyModel

from orangewidget.gui import OrangeUserRole

from Orange.data import Domain, Variable
from Orange.evaluation import scoring
from Orange.evaluation.scoring import Score
from Orange.widgets import gui
from Orange.widgets.utils.tableview import table_selection_to_mime_data
from Orange.widgets.gui import OWComponent
from Orange.widgets.settings import Setting


def check_results_adequacy(results, error_group, check_nan=True):
    error_group.add_message("invalid_results")
    error_group.invalid_results.clear()

    def anynan(a):
        return np.any(np.isnan(a))

    if results is None:
        return None
    if results.data is None:
        error_group.invalid_results(
            "Results do not include information on test data.")
    elif not results.data.domain.has_discrete_class:
        error_group.invalid_results(
            "Categorical target variable is required.")
    elif not results.actual.size:
        error_group.invalid_results(
            "Empty result on input. Nothing to display.")
    elif check_nan and (anynan(results.actual) or
                        anynan(results.predicted) or
                        (results.probabilities is not None and
                         anynan(results.probabilities))):
        error_group.invalid_results(
            "Results contain invalid values.")
    else:
        return results


def results_for_preview(data_name=""):
    from Orange.data import Table
    from Orange.evaluation import CrossValidation
    from Orange.classification import \
        LogisticRegressionLearner, SVMLearner, NuSVMLearner

    data = Table(data_name or "heart_disease")
    results = CrossValidation(
        data,
        [LogisticRegressionLearner(penalty="l2"),
         LogisticRegressionLearner(penalty="l1"),
         SVMLearner(probability=True),
         NuSVMLearner(probability=True)
        ],
        store_data=True
    )
    results.learner_names = ["LR l2", "LR l1", "SVM", "Nu SVM"]
    return results


def learner_name(learner):
    """Return the value of `learner.name` if it exists, or the learner's type
    name otherwise"""
    return getattr(learner, "name", type(learner).__name__)


def usable_scorers(domain_or_var: Union[Variable, Domain]):
    if domain_or_var is None:
        return []

    # 'abstract' is retrieved from __dict__ to avoid inheriting
    candidates = [
        scorer for scorer in scoring.Score.registry.values()
        if scorer.is_scalar and not scorer.__dict__.get("abstract")
        and scorer.is_compatible(domain_or_var) and scorer.class_types]
    return sorted(candidates, key=attrgetter("priority"))


def scorer_caller(scorer, ovr_results, target=None):
    def thunked():
        with warnings.catch_warnings():
            # F-score and Precision return 0 for labels with no predicted
            # samples. We're OK with that.
            warnings.filterwarnings(
                "ignore", "((F-score|Precision)) is ill-defined.*",
                UndefinedMetricWarning)
            if scorer.is_binary:
                return scorer(ovr_results, target=target, average='weighted')
            else:
                return scorer(ovr_results)

    return thunked


class ScoreModel(QSortFilterProxyModel):
    def lessThan(self, left, right):
        def is_bad(x):
            return not isinstance(x, (int, float, str)) \
                   or isinstance(x, float) and np.isnan(x)

        left = left.data()
        right = right.data()
        is_ascending = self.sortOrder() == Qt.AscendingOrder

        # bad entries go below; if both are bad, left remains above
        if is_bad(left) or is_bad(right):
            return is_bad(right) == is_ascending

        # for data of different types, numbers are at the top
        if type(left) is not type(right):
            return isinstance(left, float) == is_ascending

        # case insensitive comparison for strings
        if isinstance(left, str):
            return left.upper() < right.upper()

        # otherwise, compare numbers
        return left < right


DEFAULT_HINTS = {"Model_": True, "Train_": False, "Test_": False}


class PersistentMenu(QMenu):
    def mouseReleaseEvent(self, e):
        action = self.activeAction()
        if action:
            action.setEnabled(False)
            super().mouseReleaseEvent(e)
            action.setEnabled(True)
            action.trigger()
        else:
            super().mouseReleaseEvent(e)


class SelectableColumnsHeader(QHeaderView):
    SelectMenuRole = next(OrangeUserRole)
    ShownHintRole = next(OrangeUserRole)
    sectionVisibleChanged = Signal(int, bool)

    def __init__(self, shown_columns_hints, *args, **kwargs):
        super().__init__(Qt.Horizontal, *args, **kwargs)
        self.show_column_hints = shown_columns_hints
        self.button = QToolButton(self)
        self.button.setArrowType(Qt.DownArrow)
        self.button.setFixedSize(24, 12)
        col = self.button.palette().color(self.button.backgroundRole())
        self.button.setStyleSheet(
            f"border: none; background-color: {col.name(QColor.NameFormat.HexRgb)}")
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_column_chooser)
        self.button.clicked.connect(self._on_button_clicked)

    def showEvent(self, e):
        self._set_pos()
        self.button.show()
        super().showEvent(e)

    def resizeEvent(self, e):
        self._set_pos()
        super().resizeEvent(e)

    def _set_pos(self):
        w, h = self.button.width(), self.button.height()
        vw, vh = self.viewport().width(), self.viewport().height()
        self.button.setGeometry(vw - w, (vh - h) // 2, w, h)

    def __data(self, section, role):
        return self.model().headerData(section, Qt.Horizontal, role)

    def show_column_chooser(self, pos):
        # pylint: disable=unsubscriptable-object, unsupported-assignment-operation
        menu = PersistentMenu()
        for section in range(self.count()):
            name, enabled = self.__data(section, self.SelectMenuRole)
            hint_id = self.__data(section, self.ShownHintRole)
            action = menu.addAction(name)
            action.setDisabled(not enabled)
            action.setCheckable(True)
            action.setChecked(self.show_column_hints[hint_id])

            @action.triggered.connect  # pylint: disable=cell-var-from-loop
            def update(checked, q=hint_id, section=section):
                self.show_column_hints[q] = checked
                self.setSectionHidden(section, not checked)
                self.sectionVisibleChanged.emit(section, checked)
                self.resizeSections(self.ResizeToContents)

        pos.setY(self.viewport().height())
        menu.exec(self.mapToGlobal(pos))

    def _on_button_clicked(self):
        self.show_column_chooser(self.button.pos())

    def update_shown_columns(self):
        for section in range(self.count()):
            hint_id = self.__data(section, self.ShownHintRole)
            self.setSectionHidden(section, not self.show_column_hints[hint_id])


class ScoreTable(OWComponent, QObject):
    show_score_hints: Dict[str, bool] = Setting(DEFAULT_HINTS)
    shownScoresChanged = Signal()

    # backwards compatibility
    @property
    def shown_scores(self):
        # pylint: disable=unsubscriptable-object
        column_names = {
            self.model.horizontalHeaderItem(col).data(Qt.DisplayRole)
            for col in range(1, self.model.columnCount())}
        return column_names & {score.name for score in Score.registry.values()
                               if self.show_score_hints[score.__name__]}

    class ItemDelegate(QStyledItemDelegate):
        def sizeHint(self, *args):
            size = super().sizeHint(*args)
            return QSize(size.width(), size.height() + 6)

        def displayText(self, value, locale):
            if isinstance(value, float):
                return f"{value:.3f}"
            else:
                return super().displayText(value, locale)

    def __init__(self, master):
        QObject.__init__(self)
        OWComponent.__init__(self, master)

        self.view = gui.TableView(
            wordWrap=True, editTriggers=gui.TableView.NoEditTriggers
        )
        header = self.view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setDefaultAlignment(Qt.AlignCenter)
        header.setStretchLastSection(False)

        for score in Score.registry.values():
            self.show_score_hints.setdefault(score.__name__, score.default_visible)

        self.model = QStandardItemModel(master)
        header = SelectableColumnsHeader(self.show_score_hints)
        header.setSectionsClickable(True)
        self.view.setHorizontalHeader(header)
        self.sorted_model = ScoreModel()
        self.sorted_model.setSourceModel(self.model)
        self.view.setModel(self.sorted_model)
        self.view.setItemDelegate(self.ItemDelegate())
        header.sectionVisibleChanged.connect(self.shownScoresChanged.emit)
        self.sorted_model.dataChanged.connect(self.view.resizeColumnsToContents)

    def update_header(self, scorers: List[Score]):
        self.model.setColumnCount(3 + len(scorers))
        SelectMenuRole = SelectableColumnsHeader.SelectMenuRole
        ShownHintRole = SelectableColumnsHeader.ShownHintRole
        for i, name, long_name, id_, in ((0, "Model", "Model", "Model_"),
                                         (1, "Train", "Train time [s]", "Train_"),
                                         (2, "Test", "Test time [s]", "Test_")):
            item = QStandardItem(name)
            item.setData((long_name, i != 0), SelectMenuRole)
            item.setData(id_, ShownHintRole)
            item.setToolTip(long_name)
            self.model.setHorizontalHeaderItem(i, item)
        for col, score in enumerate(scorers, start=3):
            item = QStandardItem(score.name)
            name = score.long_name
            if name != score.name:
                name += f" ({score.name})"
            item.setData((name, True), SelectMenuRole)
            item.setData(score.__name__, ShownHintRole)
            item.setToolTip(score.long_name)
            self.model.setHorizontalHeaderItem(col, item)

        self.view.horizontalHeader().update_shown_columns()
        self.view.resizeColumnsToContents()

    def copy_selection_to_clipboard(self):
        mime = table_selection_to_mime_data(self.view)
        QApplication.clipboard().setMimeData(
            mime, QClipboard.Clipboard
        )

    @staticmethod
    def migrate_to_show_scores_hints(settings):
        # Migration cannot disable anything because it can't know which score
        # have been present when the setting was created.
        settings["show_score_hints"] = DEFAULT_HINTS.copy()
        settings["show_score_hints"].update(
            dict.fromkeys(settings["shown_scores"], True))
