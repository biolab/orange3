import warnings
from functools import partial
from itertools import chain

import numpy as np

from AnyQt.QtWidgets import QHeaderView, QStyledItemDelegate, QMenu, \
    QApplication
from AnyQt.QtGui import QStandardItemModel, QStandardItem, QClipboard
from AnyQt.QtCore import Qt, QSize, QObject, pyqtSignal as Signal, \
    QSortFilterProxyModel
from sklearn.exceptions import UndefinedMetricWarning

from Orange.data import DiscreteVariable, ContinuousVariable, Domain
from Orange.evaluation import scoring
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


BUILTIN_SCORERS_ORDER = {
    DiscreteVariable: ("AUC", "CA", "F1", "Precision", "Recall"),
    ContinuousVariable: ("MSE", "RMSE", "MAE", "R2")}


def learner_name(learner):
    """Return the value of `learner.name` if it exists, or the learner's type
    name otherwise"""
    return getattr(learner, "name", type(learner).__name__)


def usable_scorers(domain: Domain):
    if domain is None:
        return []

    order = {name: i
             for i, name in enumerate(chain.from_iterable(BUILTIN_SCORERS_ORDER.values()))}

    # 'abstract' is retrieved from __dict__ to avoid inheriting
    scorer_candidates = [cls for cls in scoring.Score.registry.values()
                         if cls.is_scalar and not cls.__dict__.get("abstract")]

    usable = [scorer for scorer in scorer_candidates if
              scorer.is_compatible(domain) and scorer.class_types]
    return sorted(usable, key=lambda cls: order.get(cls.name, 99))



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


class ScoreTable(OWComponent, QObject):
    shown_scores = Setting(set(chain(*BUILTIN_SCORERS_ORDER.values())))
    shownScoresChanged = Signal()

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
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self.show_column_chooser)

        # Currently, this component will never show scoring methods
        # defined in add-ons by default. To support them properly, the
        # "shown_scores" settings will need to be reworked.
        # The following is a temporary solution to show the scoring method
        # for survival data (it does not influence other problem types).
        # It is added here so that the "C-Index" method
        # will show up even if the users already have the setting defined.
        # This temporary fix is here due to a paper deadline needing the feature.
        # When removing, also remove TestScoreTable.test_column_settings_reminder
        if isinstance(self.shown_scores, set):  # TestScoreTable does not initialize settings
            self.shown_scores.add("C-Index")

        self.model = QStandardItemModel(master)
        self.model.setHorizontalHeaderLabels(["Method"])
        self.sorted_model = ScoreModel()
        self.sorted_model.setSourceModel(self.model)
        self.view.setModel(self.sorted_model)
        self.view.setItemDelegate(self.ItemDelegate())

    def _column_names(self):
        return (self.model.horizontalHeaderItem(section).data(Qt.DisplayRole)
                for section in range(1, self.model.columnCount()))

    def show_column_chooser(self, pos):
        # pylint doesn't know that self.shown_scores is a set, not a Setting
        # pylint: disable=unsupported-membership-test
        def update(col_name, checked):
            if checked:
                self.shown_scores.add(col_name)
            else:
                self.shown_scores.remove(col_name)
            self._update_shown_columns()

        menu = QMenu()
        header = self.view.horizontalHeader()
        for col_name in self._column_names():
            action = menu.addAction(col_name)
            action.setCheckable(True)
            action.setChecked(col_name in self.shown_scores)
            action.triggered.connect(partial(update, col_name))
        menu.exec(header.mapToGlobal(pos))

    def _update_shown_columns(self):
        # pylint doesn't know that self.shown_scores is a set, not a Setting
        # pylint: disable=unsupported-membership-test
        self.view.resizeColumnsToContents()
        header = self.view.horizontalHeader()
        for section, col_name in enumerate(self._column_names(), start=1):
            header.setSectionHidden(section, col_name not in self.shown_scores)
        self.shownScoresChanged.emit()

    def update_header(self, scorers):
        # Set the correct horizontal header labels on the results_model.
        self.model.setColumnCount(3 + len(scorers))
        self.model.setHorizontalHeaderItem(0, QStandardItem("Model"))
        self.model.setHorizontalHeaderItem(1, QStandardItem("Train time [s]"))
        self.model.setHorizontalHeaderItem(2, QStandardItem("Test time [s]"))
        for col, score in enumerate(scorers, start=3):
            item = QStandardItem(score.name)
            item.setToolTip(score.long_name)
            self.model.setHorizontalHeaderItem(col, item)
        self._update_shown_columns()

    def copy_selection_to_clipboard(self):
        mime = table_selection_to_mime_data(self.view)
        QApplication.clipboard().setMimeData(
            mime, QClipboard.Clipboard
        )
