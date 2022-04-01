# pylint doesn't understand the Settings magic
# pylint: disable=invalid-sequence-index
# pylint: disable=too-many-lines,too-many-instance-attributes
import abc
import enum
import logging
import traceback
import copy
from functools import partial, reduce

from concurrent.futures import Future
from collections import OrderedDict
from itertools import count
from typing import (
    Any, Optional, List, Dict, Callable, Sequence, NamedTuple, Tuple
)

import numpy as np
import baycomp

from AnyQt import QtGui
from AnyQt.QtCore import Qt, QSize, QThread
from AnyQt.QtCore import pyqtSlot as Slot
from AnyQt.QtGui import QStandardItem, QDoubleValidator
from AnyQt.QtWidgets import \
    QHeaderView, QTableWidget, QLabel, QComboBox, QSizePolicy

from Orange.base import Learner
import Orange.classification
from Orange.data import Table, DiscreteVariable
from Orange.data.table import DomainTransformationError
from Orange.data.filter import HasClass
from Orange.data.sql.table import SqlTable, AUTO_DL_LIMIT
import Orange.evaluation
from Orange.evaluation import Results
from Orange.preprocess.preprocess import Preprocess
import Orange.regression
from Orange.statistics.util import unique
from Orange.widgets import gui, settings, widget
from Orange.widgets.evaluate.utils import \
    usable_scorers, ScoreTable, learner_name, scorer_caller
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.concurrent import ThreadExecutor, TaskState
from Orange.widgets.widget import OWWidget, Msg, Input, MultiInput, Output

log = logging.getLogger(__name__)


class InputLearner(NamedTuple):
    learner:  Orange.base.Learner
    results: Optional['Try[Orange.evaluation.Results]']
    stats: Optional[Sequence['Try[float]']]
    key: Any


class Try(abc.ABC):
    """Try to walk in a Turing tar pit."""

    class Success:
        """Data type for instance constructed on success"""
        __slots__ = ("__value",)
#         __bool__ = lambda self: True
        success = property(lambda self: True)
        value = property(lambda self: self.__value)

        def __init__(self, value):
            self.__value = value

        def __getnewargs__(self):
            return (self.value,)

        def __repr__(self):
            return "{}({!r})".format(self.__class__.__qualname__,
                                     self.value)

        def map(self, fn):
            return Try(lambda: fn(self.value))

    class Fail:
        """Data type for instance constructed on fail"""
        __slots__ = ("__exception", )
#         __bool__ = lambda self: False
        success = property(lambda self: False)
        exception = property(lambda self: self.__exception)

        def __init__(self, exception):
            self.__exception = exception

        def __getnewargs__(self):
            return (self.exception, )

        def __repr__(self):
            return "{}({!r})".format(self.__class__.__qualname__,
                                     self.exception)

        def map(self, _fn):
            return self

    def __new__(cls, f, *args, **kwargs):
        try:
            rval = f(*args, **kwargs)
        except BaseException as ex:
            # IMPORANT: Will capture traceback in ex.__traceback__
            return Try.Fail(ex)
        else:
            return Try.Success(rval)

Try.register(Try.Success)
Try.register(Try.Fail)


def raise_(exc):
    raise exc

Try.register = lambda cls: raise_(TypeError())


class State(enum.Enum):
    """
    OWTestAndScore's runtime state.
    """
    #: No or insufficient input (i.e. no data or no learners)
    Waiting = "Waiting"
    #: Executing/running the evaluations
    Running = "Running"
    #: Finished running evaluations (success or error)
    Done = "Done"
    #: Evaluation cancelled
    Cancelled = "Cancelled"


class OWTestAndScore(OWWidget):
    name = "Test and Score"
    description = "Cross-validation accuracy estimation."
    icon = "icons/TestLearners1.svg"
    priority = 100
    keywords = ['Cross Validation', 'CV']
    replaces = ["Orange.widgets.evaluate.owtestlearners.OWTestLearners"]

    class Inputs:
        train_data = Input("Data", Table, default=True)
        test_data = Input("Test Data", Table)
        learner = MultiInput(
            "Learner", Learner, filter_none=True
        )
        preprocessor = Input("Preprocessor", Preprocess)

    class Outputs:
        predictions = Output("Predictions", Table)
        evaluations_results = Output("Evaluation Results", Results)

    settings_version = 3
    buttons_area_orientation = None
    UserAdviceMessages = [
        widget.Message(
            "Click on the table header to select shown columns",
            "click_header")]

    settingsHandler = settings.PerfectDomainContextHandler()
    score_table = settings.SettingProvider(ScoreTable)

    #: Resampling/testing types
    KFold, FeatureFold, ShuffleSplit, LeaveOneOut, TestOnTrain, TestOnTest \
        = 0, 1, 2, 3, 4, 5
    #: Numbers of folds
    NFolds = [2, 3, 5, 10, 20]
    #: Number of repetitions
    NRepeats = [2, 3, 5, 10, 20, 50, 100]
    #: Sample sizes
    SampleSizes = [5, 10, 20, 25, 30, 33, 40, 50, 60, 66, 70, 75, 80, 90, 95]

    #: Selected resampling type
    resampling = settings.Setting(0)
    #: Number of folds for K-fold cross validation
    n_folds = settings.Setting(2)
    #: Stratified sampling for K-fold
    cv_stratified = settings.Setting(True)
    #: Number of repeats for ShuffleSplit sampling
    n_repeats = settings.Setting(3)
    #: ShuffleSplit sample size
    sample_size = settings.Setting(9)
    #: Stratified sampling for Random Sampling
    shuffle_stratified = settings.Setting(True)
    # CV where nr. of feature values determines nr. of folds
    fold_feature = settings.ContextSetting(None)
    fold_feature_selected = settings.ContextSetting(False)

    use_rope = settings.Setting(False)
    rope = settings.Setting(0.1)
    comparison_criterion = settings.Setting(0, schema_only=True)

    TARGET_AVERAGE = "(None, show average over classes)"
    class_selection = settings.ContextSetting(TARGET_AVERAGE)

    class Error(OWWidget.Error):
        test_data_empty = Msg("Test dataset is empty.")
        class_required_test = Msg("Test data input requires a target variable.")
        too_many_folds = Msg("Number of folds exceeds the data size")
        class_inconsistent = Msg("Test and train datasets "
                                 "have different target variables.")
        memory_error = Msg("Not enough memory.")
        test_data_incompatible = Msg(
            "Test data may be incompatible with train data.")
        train_data_error = Msg("{}")

    class Warning(OWWidget.Warning):
        missing_data = \
            Msg("Instances with unknown target values were removed from{}data.")
        test_data_missing = Msg("Missing separate test data input.")
        scores_not_computed = Msg("Some scores could not be computed.")
        test_data_unused = Msg("Test data is present but unused. "
                               "Select 'Test on test data' to use it.")
        cant_stratify = \
            Msg("Can't run stratified {}-fold cross validation; "
                "the least common class has only {} instances.")

    class Information(OWWidget.Information):
        data_sampled = Msg("Train data has been sampled")
        test_data_sampled = Msg("Test data has been sampled")
        test_data_transformed = Msg(
            "Test data has been transformed to match the train data.")
        cant_stratify_numeric = Msg("Stratification is ignored for regression")
        cant_stratify_multitarget = Msg("Stratification is ignored when there are"
                                        " multiple target variables.")

    def __init__(self):
        super().__init__()

        self.data = None
        self.test_data = None
        self.preprocessor = None
        self.train_data_missing_vals = False
        self.test_data_missing_vals = False
        self.scorers = []
        self.__pending_comparison_criterion = self.comparison_criterion
        self.__id_gen = count()
        self._learner_inputs = []  # type: List[Tuple[Any, Learner]]
        #: An Ordered dictionary with current inputs and their testing results
        #: (keyed by ids generated by __id_gen).
        self.learners = OrderedDict()  # type: Dict[Any, Input]

        self.__state = State.Waiting
        # Do we need to [re]test any learners, set by _invalidate and
        # cleared by __update
        self.__needupdate = False
        self.__task = None  # type: Optional[TaskState]
        self.__executor = ThreadExecutor()

        sbox = gui.vBox(self.controlArea, box=True)
        rbox = gui.radioButtons(
            sbox, self, "resampling", callback=self._param_changed)

        gui.appendRadioButton(rbox, "Cross validation")
        ibox = gui.indentedBox(rbox)
        gui.comboBox(
            ibox, self, "n_folds", label="Number of folds: ",
            items=[str(x) for x in self.NFolds],
            orientation=Qt.Horizontal, callback=self.kfold_changed)
        gui.checkBox(
            ibox, self, "cv_stratified", "Stratified",
            callback=self.kfold_changed)
        gui.appendRadioButton(rbox, "Cross validation by feature")
        ibox = gui.indentedBox(rbox)
        self.feature_model = DomainModel(
            order=DomainModel.METAS, valid_types=DiscreteVariable)
        self.features_combo = gui.comboBox(
            ibox, self, "fold_feature", model=self.feature_model,
            orientation=Qt.Horizontal, searchable=True,
            callback=self.fold_feature_changed)

        gui.appendRadioButton(rbox, "Random sampling")
        ibox = gui.indentedBox(rbox)
        gui.comboBox(
            ibox, self, "n_repeats", label="Repeat train/test: ",
            items=[str(x) for x in self.NRepeats], orientation=Qt.Horizontal,
            callback=self.shuffle_split_changed
        )
        gui.comboBox(
            ibox, self, "sample_size", label="Training set size: ",
            items=["{} %".format(x) for x in self.SampleSizes],
            orientation=Qt.Horizontal, callback=self.shuffle_split_changed
        )
        gui.checkBox(
            ibox, self, "shuffle_stratified", "Stratified",
            callback=self.shuffle_split_changed)

        gui.appendRadioButton(rbox, "Leave one out")

        gui.appendRadioButton(rbox, "Test on train data")
        gui.appendRadioButton(rbox, "Test on test data")

        gui.rubber(self.controlArea)

        self.score_table = ScoreTable(self)
        self.score_table.shownScoresChanged.connect(self.update_stats_model)
        view = self.score_table.view
        view.setSizeAdjustPolicy(view.AdjustToContents)

        self.results_box = gui.vBox(self.mainArea, box=True)
        self.cbox = gui.hBox(self.results_box)
        self.class_selection_combo = gui.comboBox(
            self.cbox, self, "class_selection", items=[],
            label="Evaluation results for target", orientation=Qt.Horizontal,
            sendSelectedValue=True, searchable=True, contentsLength=25,
            callback=self._on_target_class_changed
        )
        self.cbox.layout().addStretch(100)
        self.class_selection_combo.setMaximumContentsLength(30)
        self.results_box.layout().addWidget(self.score_table.view)

        gui.separator(self.mainArea, 16)
        self.compbox = box = gui.vBox(self.mainArea, box=True)
        cbox = gui.comboBox(
            box, self, "comparison_criterion", label="Compare models by:",
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon,
            sizePolicy=(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed),
            orientation=Qt.Horizontal, callback=self.update_comparison_table).box

        gui.separator(cbox, 8)
        gui.checkBox(cbox, self, "use_rope", "Negligible diff.: ",
                     callback=self._on_use_rope_changed)
        gui.lineEdit(cbox, self, "rope", validator=QDoubleValidator(),
                     controlWidth=50, callback=self.update_comparison_table,
                     alignment=Qt.AlignRight)
        self.controls.rope.setEnabled(self.use_rope)

        table = self.comparison_table = QTableWidget(
            wordWrap=False, editTriggers=QTableWidget.NoEditTriggers,
            selectionMode=QTableWidget.NoSelection)
        table.setSizeAdjustPolicy(table.AdjustToContents)
        header = table.verticalHeader()
        header.setSectionResizeMode(QHeaderView.Fixed)
        header.setSectionsClickable(False)

        header = table.horizontalHeader()
        header.setTextElideMode(Qt.ElideRight)
        header.setDefaultAlignment(Qt.AlignCenter)
        header.setSectionsClickable(False)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        avg_width = self.fontMetrics().averageCharWidth()
        header.setMinimumSectionSize(8 * avg_width)
        header.setMaximumSectionSize(15 * avg_width)
        header.setDefaultSectionSize(15 * avg_width)
        box.layout().addWidget(table)
        box.layout().addWidget(QLabel(
            "<small>Table shows probabilities that the score for the model in "
            "the row is higher than that of the model in the column. "
            "Small numbers show the probability that the difference is "
            "negligible.</small>", wordWrap=True))

    def sizeHint(self):
        sh = super().sizeHint()
        return QSize(780, sh.height())

    def _update_controls(self):
        self.fold_feature = None
        self.feature_model.set_domain(None)
        if self.data:
            self.feature_model.set_domain(self.data.domain)
            if self.fold_feature is None and self.feature_model:
                self.fold_feature = self.feature_model[0]
        enabled = bool(self.feature_model)
        self.controls.resampling.buttons[
            OWTestAndScore.FeatureFold].setEnabled(enabled)
        self.features_combo.setEnabled(enabled)
        if self.resampling == OWTestAndScore.FeatureFold and not enabled:
            self.resampling = OWTestAndScore.KFold

    @Inputs.learner
    def set_learner(self, index: int, learner: Learner):
        """
        Set the input `learner` at `index`.

        Parameters
        ----------
        index: int
        learner: Orange.base.Learner
        """
        key, _ = self._learner_inputs[index]
        slot = self.learners[key]
        self.learners[key] = slot._replace(learner=learner, results=None)
        self._invalidate([key])

    @Inputs.learner.insert
    def insert_learner(self, index: int, learner: Learner):
        key = next(self.__id_gen)
        self._learner_inputs.insert(index, (key, learner))
        self.learners[key] = InputLearner(learner, None, None, key)
        self.learners = {key: self.learners[key] for key, _ in self._learner_inputs}
        self._invalidate([key])

    @Inputs.learner.remove
    def remove_learner(self, index: int):
        key, _ = self._learner_inputs[index]
        self._invalidate([key])
        self._learner_inputs.pop(index)
        self.learners.pop(key)

    @Inputs.train_data
    def set_train_data(self, data):
        """
        Set the input training dataset.

        Parameters
        ----------
        data : Optional[Orange.data.Table]
        """
        self.cancel()
        self.Information.data_sampled.clear()
        self.Error.train_data_error.clear()

        if data is not None:
            data_errors = [
                ("Train dataset is empty.", len(data) == 0),
                (
                    "Train data input requires a target variable.",
                    not data.domain.class_vars
                ),
                ("Target variable has no values.", np.isnan(data.Y).all()),
                (
                    "Target variable has only one value.",
                    data.domain.has_discrete_class and len(unique(data.Y)) < 2
                ),
                ("Data has no features to learn from.", data.X.shape[1] == 0),
            ]

            for error_msg, cond in data_errors:
                if cond:
                    self.Error.train_data_error(error_msg)
                    data = None
                    break

        if isinstance(data, SqlTable):
            if data.approx_len() < AUTO_DL_LIMIT:
                data = Table(data)
            else:
                self.Information.data_sampled()
                data_sample = data.sample_time(1, no_cache=True)
                data_sample.download_data(AUTO_DL_LIMIT, partial=True)
                data = Table(data_sample)

        self.train_data_missing_vals = \
            data is not None and np.isnan(data.Y).any()
        if self.train_data_missing_vals or self.test_data_missing_vals:
            self.Warning.missing_data(self._which_missing_data())
            if data:
                data = HasClass()(data)
        else:
            self.Warning.missing_data.clear()

        self.data = data
        self.closeContext()
        self._update_scorers()
        self._update_controls()
        if data is not None:
            self._update_class_selection()
            self.openContext(data.domain)
            if self.fold_feature_selected and bool(self.feature_model):
                self.resampling = OWTestAndScore.FeatureFold
        self._invalidate()

    @Inputs.test_data
    def set_test_data(self, data):
        # type: (Orange.data.Table) -> None
        """
        Set the input separate testing dataset.

        Parameters
        ----------
        data : Optional[Orange.data.Table]
        """
        self.Information.test_data_sampled.clear()
        self.Error.test_data_empty.clear()
        if data is not None and not data:
            self.Error.test_data_empty()
            data = None

        if data and not data.domain.class_vars:
            self.Error.class_required_test()
            data = None
        else:
            self.Error.class_required_test.clear()

        if isinstance(data, SqlTable):
            if data.approx_len() < AUTO_DL_LIMIT:
                data = Table(data)
            else:
                self.Information.test_data_sampled()
                data_sample = data.sample_time(1, no_cache=True)
                data_sample.download_data(AUTO_DL_LIMIT, partial=True)
                data = Table(data_sample)

        self.test_data_missing_vals = \
            data is not None and np.isnan(data.Y).any()
        if self.train_data_missing_vals or self.test_data_missing_vals:
            self.Warning.missing_data(self._which_missing_data())
            if data:
                data = HasClass()(data)
        else:
            self.Warning.missing_data.clear()

        self.test_data = data
        if self.resampling == OWTestAndScore.TestOnTest:
            self._invalidate()

    def _which_missing_data(self):
        return {(True, True): " ",  # both, don't specify
                (True, False): " train ",
                (False, True): " test "}[(self.train_data_missing_vals,
                                          self.test_data_missing_vals)]

    # List of scorers shouldn't be retrieved globally, when the module is
    # loading since add-ons could have registered additional scorers.
    # It could have been cached but
    # - we don't gain much with it
    # - it complicates the unit tests
    def _update_scorers(self):
        new_scorers = []
        if self.data:
            new_scorers = usable_scorers(self.data.domain)

        # Don't unnecessarily reset the combo because this would always reset
        # comparison_criterion; we also set it explicitly, though, for clarity
        if new_scorers != self.scorers:
            self.scorers = new_scorers
            combo = self.controls.comparison_criterion
            combo.clear()
            combo.addItems([scorer.long_name or scorer.name
                            for scorer in self.scorers])
            if self.scorers:
                self.comparison_criterion = 0
        if self.__pending_comparison_criterion is not None:
            # Check for the unlikely case that some scorers have been removed
            # from modules
            if self.__pending_comparison_criterion < len(self.scorers):
                self.comparison_criterion = self.__pending_comparison_criterion
            self.__pending_comparison_criterion = None

    @Inputs.preprocessor
    def set_preprocessor(self, preproc):
        """
        Set the input preprocessor to apply on the training data.
        """
        self.preprocessor = preproc
        self._invalidate()

    def handleNewSignals(self):
        """Reimplemented from OWWidget.handleNewSignals."""
        self._update_class_selection()
        self.score_table.update_header(self.scorers)
        self._update_view_enabled()
        self.update_stats_model()
        if self.__needupdate:
            self.__update()

    def kfold_changed(self):
        self.resampling = OWTestAndScore.KFold
        self._param_changed()

    def fold_feature_changed(self):
        self.resampling = OWTestAndScore.FeatureFold
        self._param_changed()

    def shuffle_split_changed(self):
        self.resampling = OWTestAndScore.ShuffleSplit
        self._param_changed()

    def _param_changed(self):
        self._update_view_enabled()
        self._invalidate()
        self.__update()

    def _update_view_enabled(self):
        self.compbox.setEnabled(
            self.resampling == OWTestAndScore.KFold
            and len(self.learners) > 1
            and self.data is not None)
        self.score_table.view.setEnabled(
            self.data is not None)

    def update_stats_model(self):
        # Update the results_model with up to date scores.
        # Note: The target class specific scores (if requested) are
        # computed as needed in this method.
        model = self.score_table.model
        # clear the table model, but preserving the header labels
        for r in reversed(range(model.rowCount())):
            model.takeRow(r)

        target_index = None
        if self.data is not None:
            class_var = self.data.domain.class_var
            if self.data.domain.has_discrete_class and \
                            self.class_selection != self.TARGET_AVERAGE:
                target_index = class_var.values.index(self.class_selection)
        else:
            class_var = None

        errors = []
        has_missing_scores = False

        names = []
        for key, slot in self.learners.items():
            name = learner_name(slot.learner)
            names.append(name)
            head = QStandardItem(name)
            head.setData(key, Qt.UserRole)
            results = slot.results
            if results is not None and results.success:
                train = QStandardItem("{:.3f}".format(results.value.train_time))
                train.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                train.setData(key, Qt.UserRole)
                test = QStandardItem("{:.3f}".format(results.value.test_time))
                test.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                test.setData(key, Qt.UserRole)
                row = [head, train, test]
            else:
                row = [head]
            if isinstance(results, Try.Fail):
                head.setToolTip(str(results.exception))
                head.setText("{} (error)".format(name))
                head.setForeground(QtGui.QBrush(Qt.red))
                if isinstance(results.exception, DomainTransformationError) \
                        and self.resampling == self.TestOnTest:
                    self.Error.test_data_incompatible()
                    self.Information.test_data_transformed.clear()
                else:
                    errors.append("{name} failed with error:\n"
                                  "{exc.__class__.__name__}: {exc!s}"
                                  .format(name=name, exc=slot.results.exception)
                                  )

            if class_var is not None and class_var.is_discrete and \
                    target_index is not None:
                if slot.results is not None and slot.results.success:
                    ovr_results = results_one_vs_rest(
                        slot.results.value, target_index)

                    # Cell variable is used immediatelly, it's not stored
                    # pylint: disable=cell-var-from-loop
                    stats = [Try(scorer_caller(scorer, ovr_results, target=1))
                             for scorer in self.scorers]
                else:
                    stats = None
            else:
                stats = slot.stats

            if stats is not None:
                for stat, scorer in zip(stats, self.scorers):
                    item = QStandardItem()
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    if stat.success:
                        item.setData(float(stat.value[0]), Qt.DisplayRole)
                    else:
                        item.setToolTip(str(stat.exception))
                        if scorer.name in self.score_table.shown_scores:
                            has_missing_scores = True
                    row.append(item)

            model.appendRow(row)

        # Resort rows based on current sorting
        header = self.score_table.view.horizontalHeader()
        model.sort(
            header.sortIndicatorSection(),
            header.sortIndicatorOrder()
        )
        self._set_comparison_headers(names)

        self.error("\n".join(errors), shown=bool(errors))
        self.Warning.scores_not_computed(shown=has_missing_scores)

    def _on_use_rope_changed(self):
        self.controls.rope.setEnabled(self.use_rope)
        self.update_comparison_table()

    def update_comparison_table(self):
        self.comparison_table.clearContents()
        slots = self._successful_slots()
        if not (slots and self.scorers):
            return
        names = [learner_name(slot.learner) for slot in slots]
        self._set_comparison_headers(names)
        if self.resampling == OWTestAndScore.KFold:
            scores = self._scores_by_folds(slots)
            self._fill_table(names, scores)

    def _successful_slots(self):
        model = self.score_table.model
        proxy = self.score_table.sorted_model

        keys = (model.data(proxy.mapToSource(proxy.index(row, 0)), Qt.UserRole)
                for row in range(proxy.rowCount()))
        slots = [slot for slot in (self.learners[key] for key in keys)
                 if slot.results is not None and slot.results.success]
        return slots

    def _set_comparison_headers(self, names):
        table = self.comparison_table
        try:
            # Prevent glitching during update
            table.setUpdatesEnabled(False)
            header = table.horizontalHeader()
            if len(names) > 2:
                header.setSectionResizeMode(QHeaderView.Stretch)
            else:
                header.setSectionResizeMode(QHeaderView.Fixed)
            table.setRowCount(len(names))
            table.setColumnCount(len(names))
            table.setVerticalHeaderLabels(names)
            table.setHorizontalHeaderLabels(names)
        finally:
            table.setUpdatesEnabled(True)

    def _scores_by_folds(self, slots):
        scorer = self.scorers[self.comparison_criterion]()
        if scorer.is_binary:
            if self.class_selection != self.TARGET_AVERAGE:
                class_var = self.data.domain.class_var
                target_index = class_var.values.index(self.class_selection)
                kw = dict(target=target_index)
            else:
                kw = dict(average='weighted')
        else:
            kw = {}

        def call_scorer(results):
            def thunked():
                return scorer.scores_by_folds(results.value, **kw).flatten()

            return thunked

        scores = [Try(call_scorer(slot.results)) for slot in slots]
        scores = [score.value if score.success else None for score in scores]
        # `None in scores doesn't work -- these are np.arrays)
        if any(score is None for score in scores):
            self.Warning.scores_not_computed()
        return scores

    def _fill_table(self, names, scores):
        table = self.comparison_table
        for row, row_name, row_scores in zip(count(), names, scores):
            for col, col_name, col_scores in zip(range(row), names, scores):
                if row_scores is None or col_scores is None:
                    continue
                if self.use_rope and self.rope:
                    p0, rope, p1 = baycomp.two_on_single(
                        row_scores, col_scores, self.rope)
                    if np.isnan(p0) or np.isnan(rope) or np.isnan(p1):
                        self._set_cells_na(table, row, col)
                        continue
                    self._set_cell(table, row, col,
                                   f"{p0:.3f}<br/><small>{rope:.3f}</small>",
                                   f"p({row_name} > {col_name}) = {p0:.3f}\n"
                                   f"p({row_name} = {col_name}) = {rope:.3f}")
                    self._set_cell(table, col, row,
                                   f"{p1:.3f}<br/><small>{rope:.3f}</small>",
                                   f"p({col_name} > {row_name}) = {p1:.3f}\n"
                                   f"p({col_name} = {row_name}) = {rope:.3f}")
                else:
                    p0, p1 = baycomp.two_on_single(row_scores, col_scores)
                    if np.isnan(p0) or np.isnan(p1):
                        self._set_cells_na(table, row, col)
                        continue
                    self._set_cell(table, row, col,
                                   f"{p0:.3f}",
                                   f"p({row_name} > {col_name}) = {p0:.3f}")
                    self._set_cell(table, col, row,
                                   f"{p1:.3f}",
                                   f"p({col_name} > {row_name}) = {p1:.3f}")

    @classmethod
    def _set_cells_na(cls, table, row, col):
        cls._set_cell(table, row, col, "NA", "comparison cannot be computed")
        cls._set_cell(table, col, row, "NA", "comparison cannot be computed")

    @staticmethod
    def _set_cell(table, row, col, label, tooltip):
        item = QLabel(label)
        item.setToolTip(tooltip)
        item.setAlignment(Qt.AlignCenter)
        table.setCellWidget(row, col, item)

    def _update_class_selection(self):
        self.class_selection_combo.setCurrentIndex(-1)
        self.class_selection_combo.clear()
        if not self.data:
            return

        if self.data.domain.has_discrete_class:
            self.cbox.setVisible(True)
            class_var = self.data.domain.class_var
            items = (self.TARGET_AVERAGE, ) + class_var.values
            self.class_selection_combo.addItems(items)

            class_index = 0
            if self.class_selection in class_var.values:
                class_index = class_var.values.index(self.class_selection) + 1

            self.class_selection_combo.setCurrentIndex(class_index)
            self.class_selection = items[class_index]
        else:
            self.cbox.setVisible(False)

    def _on_target_class_changed(self):
        self.update_stats_model()
        self.update_comparison_table()

    def _invalidate(self, which=None):
        self.cancel()
        self.fold_feature_selected = \
            self.resampling == OWTestAndScore.FeatureFold
        # Invalidate learner results for `which` input keys
        # (if None then all learner results are invalidated)
        if which is None:
            which = self.learners.keys()

        model = self.score_table.model
        statmodelkeys = [model.item(row, 0).data(Qt.UserRole)
                         for row in range(model.rowCount())]

        for key in which:
            self.learners[key] = \
                self.learners[key]._replace(results=None, stats=None)

            if key in statmodelkeys:
                row = statmodelkeys.index(key)
                for c in range(1, model.columnCount()):
                    item = model.item(row, c)
                    if item is not None:
                        item.setData(None, Qt.DisplayRole)
                        item.setData(None, Qt.ToolTipRole)

        self.comparison_table.clearContents()

        self.__needupdate = True

    def commit(self):
        """
        Commit the results to output.
        """
        self.Error.memory_error.clear()
        valid = [slot for slot in self.learners.values()
                 if slot.results is not None and slot.results.success]
        combined = None
        predictions = None
        if valid:
            # Evaluation results
            combined = results_merge([slot.results.value for slot in valid])
            combined.learner_names = [learner_name(slot.learner)
                                      for slot in valid]

            # Predictions & Probabilities
            try:
                predictions = combined.get_augmented_data(combined.learner_names)
            except MemoryError:
                self.Error.memory_error()

        self.Outputs.evaluations_results.send(combined)
        self.Outputs.predictions.send(predictions)

    def send_report(self):
        """Report on the testing schema and results"""
        if not self.data or not self.learners:
            return
        if self.resampling == self.KFold:
            stratified = 'Stratified ' if self.cv_stratified else ''
            items = [("Sampling type", "{}{}-fold Cross validation".
                      format(stratified, self.NFolds[self.n_folds]))]
        elif self.resampling == self.LeaveOneOut:
            items = [("Sampling type", "Leave one out")]
        elif self.resampling == self.ShuffleSplit:
            stratified = 'Stratified ' if self.shuffle_stratified else ''
            items = [("Sampling type",
                      "{}Shuffle split, {} random samples with {}% data "
                      .format(stratified, self.NRepeats[self.n_repeats],
                              self.SampleSizes[self.sample_size]))]
        elif self.resampling == self.TestOnTrain:
            items = [("Sampling type", "No sampling, test on training data")]
        elif self.resampling == self.TestOnTest:
            items = [("Sampling type", "No sampling, test on testing data")]
        else:
            items = []
        if self.data.domain.has_discrete_class:
            items += [("Target class", self.class_selection.strip("()"))]
        if items:
            self.report_items("Settings", items)
        self.report_table("Scores", self.score_table.view)

    @classmethod
    def migrate_settings(cls, settings_, version):
        if version < 2:
            if settings_["resampling"] > 0:
                settings_["resampling"] += 1
        if version < 3:
            # Older version used an incompatible context handler
            settings_["context_settings"] = [
                c for c in settings_.get("context_settings", ())
                if not hasattr(c, 'classes')]

    @Slot(float)
    def setProgressValue(self, value):
        self.progressBarSet(value)

    def __update(self):
        self.__needupdate = False

        assert self.__task is None or self.__state == State.Running
        if self.__state == State.Running:
            self.cancel()

        self.Warning.test_data_unused.clear()
        self.Error.test_data_incompatible.clear()
        self.Warning.test_data_missing.clear()
        self.Warning.cant_stratify.clear()
        self.Information.cant_stratify_numeric.clear()
        self.Information.cant_stratify_multitarget.clear()
        self.Information.test_data_transformed(
            shown=self.resampling == self.TestOnTest
            and self.data is not None
            and self.test_data is not None
            and self.data.domain.attributes != self.test_data.domain.attributes)
        self.warning()
        self.Error.class_inconsistent.clear()
        self.Error.too_many_folds.clear()
        self.error()

        # check preconditions and return early or show warnings
        if self.data is None:
            self.__state = State.Waiting
            self.commit()
            return
        if not self.learners:
            self.__state = State.Waiting
            self.commit()
            return
        if self.resampling == OWTestAndScore.KFold:
            k = self.NFolds[self.n_folds]
            if len(self.data) < k:
                self.Error.too_many_folds()
                self.__state = State.Waiting
                self.commit()
                return
            do_stratify = self.cv_stratified
            if do_stratify:
                if len(self.data.domain.class_vars) > 1:
                    self.Information.cant_stratify_multitarget()
                    do_stratify = False
                elif self.data.domain.class_var.is_discrete:
                    least = min(filter(None,
                                       np.bincount(self.data.Y.astype(int))))
                    if least < k:
                        self.Warning.cant_stratify(k, least)
                        do_stratify = False
                else:
                    self.Information.cant_stratify_numeric()
                    do_stratify = False

        elif self.resampling == OWTestAndScore.TestOnTest:
            if self.test_data is None:
                if not self.Error.test_data_empty.is_shown():
                    self.Warning.test_data_missing()
                self.__state = State.Waiting
                self.commit()
                return
            elif self.test_data.domain.class_var != self.data.domain.class_var:
                self.Error.class_inconsistent()
                self.__state = State.Waiting
                self.commit()
                return

        elif self.test_data is not None:
            self.Warning.test_data_unused()

        rstate = 42
        # items in need of an update
        items = [(key, slot) for key, slot in self.learners.items()
                 if slot.results is None]
        learners = [slot.learner for _, slot in items]

        # deepcopy all learners as they are not thread safe (by virtue of
        # the base API). These will be the effective learner objects tested
        # but will be replaced with the originals on return (see restore
        # learners bellow)
        learners_c = [copy.deepcopy(learner) for learner in learners]

        if self.resampling == OWTestAndScore.TestOnTest:
            test_f = partial(
                Orange.evaluation.TestOnTestData(
                    store_data=True, store_models=True),
                self.data, self.test_data, learners_c, self.preprocessor
            )
        else:
            if self.resampling == OWTestAndScore.KFold:
                sampler = Orange.evaluation.CrossValidation(
                    k=self.NFolds[self.n_folds],
                    random_state=rstate,
                    stratified=do_stratify)
            elif self.resampling == OWTestAndScore.FeatureFold:
                sampler = Orange.evaluation.CrossValidationFeature(
                    feature=self.fold_feature)
            elif self.resampling == OWTestAndScore.LeaveOneOut:
                sampler = Orange.evaluation.LeaveOneOut()
            elif self.resampling == OWTestAndScore.ShuffleSplit:
                sampler = Orange.evaluation.ShuffleSplit(
                    n_resamples=self.NRepeats[self.n_repeats],
                    train_size=self.SampleSizes[self.sample_size] / 100,
                    test_size=None,
                    stratified=self.shuffle_stratified,
                    random_state=rstate)
            elif self.resampling == OWTestAndScore.TestOnTrain:
                sampler = Orange.evaluation.TestOnTrainingData(
                    store_models=True)
            else:
                assert False, "self.resampling %s" % self.resampling

            sampler.store_data = True
            test_f = partial(
                sampler, self.data, learners_c, self.preprocessor)

        def replace_learners(evalfunc, *args, **kwargs):
            res = evalfunc(*args, **kwargs)
            assert all(lc is lo for lc, lo in zip(learners_c, res.learners))
            res.learners[:] = learners
            return res

        test_f = partial(replace_learners, test_f)

        self.__submit(test_f)

    def __submit(self, testfunc):
        # type: (Callable[[Callable[[float], None]], Results]) -> None
        """
        Submit a testing function for evaluation

        MUST not be called if an evaluation is already pending/running.
        Cancel the existing task first.

        Parameters
        ----------
        testfunc : Callable[[Callable[float]], Results])
            Must be a callable taking a single `callback` argument and
            returning a Results instance
        """
        assert self.__state != State.Running
        # Setup the task
        task = TaskState()

        def progress_callback(finished):
            if task.is_interruption_requested():
                raise UserInterrupt()
            task.set_progress_value(100 * finished)

        testfunc = partial(testfunc, callback=progress_callback)
        task.start(self.__executor, testfunc)

        task.progress_changed.connect(self.setProgressValue)
        task.watcher.finished.connect(self.__task_complete)

        self.Outputs.evaluations_results.invalidate()
        self.Outputs.predictions.invalidate()
        self.progressBarInit()
        self.setStatusMessage("Running")

        self.__state = State.Running
        self.__task = task

    @Slot(object)
    def __task_complete(self, f: 'Future[Results]'):
        # handle a completed task
        assert self.thread() is QThread.currentThread()
        assert self.__task is not None and self.__task.future is f
        self.progressBarFinished()
        self.setStatusMessage("")
        assert f.done()
        self.__task = None
        self.__state = State.Done
        try:
            results = f.result()    # type: Results
            learners = results.learners  # type: List[Learner]
        except Exception as er:  # pylint: disable=broad-except
            log.exception("testing error (in __task_complete):",
                          exc_info=True)
            self.error("\n".join(traceback.format_exception_only(type(er), er)))
            return

        learner_key = {slot.learner: key for key, slot in
                       self.learners.items()}
        assert all(learner in learner_key for learner in learners)

        # Update the results for individual learners
        for learner, result in zip(learners, results.split_by_model()):
            ex = result.failed[0]
            if ex:
                stats = [Try.Fail(ex)] * len(self.scorers)
                result = Try.Fail(ex)
            else:
                stats = [Try(scorer_caller(scorer, result))
                         for scorer in self.scorers]
                result = Try.Success(result)
            key = learner_key.get(learner)
            self.learners[key] = \
                self.learners[key]._replace(results=result, stats=stats)

        self.score_table.update_header(self.scorers)
        self.update_stats_model()
        self.update_comparison_table()

        self.commit()

    def cancel(self):
        """
        Cancel the current/pending evaluation (if any).
        """
        if self.__task is not None:
            assert self.__state == State.Running
            self.__state = State.Cancelled
            task, self.__task = self.__task, None
            task.cancel()
            task.progress_changed.disconnect(self.setProgressValue)
            task.watcher.finished.disconnect(self.__task_complete)

            self.progressBarFinished()
            self.setStatusMessage("")

    def onDeleteWidget(self):
        self.cancel()
        self.__executor.shutdown(wait=False)
        super().onDeleteWidget()

    def copy_to_clipboard(self):
        self.score_table.copy_selection_to_clipboard()


class UserInterrupt(BaseException):
    """
    A BaseException subclass used for cooperative task/thread cancellation
    """


def results_add_by_model(x, y):
    def is_empty(res):
        return (getattr(res, "models", None) is None
                and getattr(res, "row_indices", None) is None)

    if is_empty(x):
        return y
    elif is_empty(y):
        return x

    assert (x.row_indices == y.row_indices).all()
    assert (x.actual == y.actual).all()

    res = Orange.evaluation.Results()
    res.data = x.data
    res.domain = x.domain
    res.row_indices = x.row_indices
    res.folds = x.folds
    res.actual = x.actual
    res.predicted = np.vstack((x.predicted, y.predicted))
    if getattr(x, "probabilities", None) is not None \
            and getattr(y, "probabilities") is not None:
        res.probabilities = np.vstack((x.probabilities, y.probabilities))

    if x.models is not None:
        res.models = np.hstack((x.models, y.models))
    return res


def results_merge(results):
    return reduce(results_add_by_model, results, Orange.evaluation.Results())


def results_one_vs_rest(results, pos_index):
    from Orange.preprocess.transformation import Indicator
    actual = results.actual == pos_index
    predicted = results.predicted == pos_index
    if results.probabilities is not None:
        c = results.probabilities.shape[2]
        assert c >= 2
        neg_indices = [i for i in range(c) if i != pos_index]
        pos_prob = results.probabilities[:, :, [pos_index]]
        neg_prob = np.sum(results.probabilities[:, :, neg_indices],
                          axis=2, keepdims=True)
        probabilities = np.dstack((neg_prob, pos_prob))
    else:
        probabilities = None

    res = Orange.evaluation.Results()
    res.actual = actual
    res.predicted = predicted
    res.folds = results.folds
    res.row_indices = results.row_indices
    res.probabilities = probabilities

    value = results.domain.class_var.values[pos_index]
    class_var = Orange.data.DiscreteVariable(
        "I({}=={})".format(results.domain.class_var.name, value),
        values=("False", "True"),
        compute_value=Indicator(results.domain.class_var, pos_index)
    )
    domain = Orange.data.Domain(
        results.domain.attributes,
        [class_var],
        results.domain.metas
    )
    res.data = None
    res.domain = domain
    return res


if __name__ == "__main__":  # pragma: no cover
    filename = "iris"
    preview_data = Table(filename)
    if preview_data.domain.class_var.is_discrete:
        prev_learners = [lambda data: 1 / 0,
                         Orange.classification.LogisticRegressionLearner(),
                         Orange.classification.MajorityLearner(),
                         Orange.classification.NaiveBayesLearner()]
    else:
        prev_learners = [lambda data: 1 / 0,
                         Orange.regression.MeanLearner(),
                         Orange.regression.KNNRegressionLearner(),
                         Orange.regression.RidgeRegressionLearner()]

    WidgetPreview(OWTestAndScore).run(
        set_train_data=preview_data,
        set_test_data=preview_data,
        insert_learner=list(enumerate(prev_learners))
    )
