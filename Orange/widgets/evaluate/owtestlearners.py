# pylint doesn't understand the Settings magic
# pylint: disable=invalid-sequence-index

import sys
from itertools import chain
import abc
import enum
import logging
import traceback
import copy
from functools import partial, reduce

import concurrent.futures
from concurrent.futures import Future
from collections import OrderedDict, namedtuple

try:
    # only used in type hinting
    # pylint: disable=unused-import
    from typing import Any, Optional, List, Tuple, Dict, Callable
except ImportError:
    pass

import numpy as np

from AnyQt import QtGui
from AnyQt.QtWidgets import QHeaderView, QStyledItemDelegate, QMenu
from AnyQt.QtGui import QStandardItemModel, QStandardItem
from AnyQt.QtCore import Qt, QSize, QThread, QMetaObject, Q_ARG
from AnyQt.QtCore import pyqtSlot as Slot

from Orange.base import Learner
import Orange.classification
from Orange.data import Table, DiscreteVariable, ContinuousVariable
from Orange.data.filter import HasClass
from Orange.data.sql.table import SqlTable, AUTO_DL_LIMIT
import Orange.evaluation
from Orange.evaluation import scoring, Results
from Orange.preprocess.preprocess import Preprocess
import Orange.regression
from Orange.widgets import gui, settings, widget
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets.utils.concurrent import ThreadExecutor


log = logging.getLogger(__name__)

InputLearner = namedtuple(
    "InputLearner",
    ["learner",  # :: Orange.base.Learner
     "results",  # :: Option[Try[Orange.evaluation.Results]]
     "stats"]    # :: Option[Sequence[Try[float]]]
)


class ItemDelegate(QStyledItemDelegate):
    def sizeHint(self, *args):
        size = super().sizeHint(*args)
        return QSize(size.width(), size.height() + 6)


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

        def map(self, fn):
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
    OWTestLearner's runtime state.
    """
    #: No or insufficient input (i.e. no data or no learners)
    Waiting = "Waiting"
    #: Executing/running the evaluations
    Running = "Running"
    #: Finished running evaluations (success or error)
    Done = "Done"
    #: Evaluation cancelled
    Cancelled = "Cancelled"


class OWTestLearners(OWWidget):
    name = "Test & Score"
    description = "Cross-validation accuracy estimation."
    icon = "icons/TestLearners1.svg"
    priority = 100

    class Inputs:
        train_data = Input("Data", Table, default=True)
        test_data = Input("Test Data", Table)
        learner = Input("Learner", Learner, multiple=True)
        preprocessor = Input("Preprocessor", Preprocess)

    class Outputs:
        predictions = Output("Predictions", Table)
        evaluations_results = Output("Evaluation Results", Results)

    settings_version = 3
    UserAdviceMessages = [
        widget.Message(
            "Click on the table header to select shown columns",
            "click_header")]

    settingsHandler = settings.PerfectDomainContextHandler()

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
    n_folds = settings.Setting(3)
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

    TARGET_AVERAGE = "(Average over classes)"
    class_selection = settings.ContextSetting(TARGET_AVERAGE)

    BUILTIN_ORDER = {
        DiscreteVariable: ("AUC", "CA", "F1", "Precision", "Recall"),
        ContinuousVariable: ("MSE", "RMSE", "MAE", "R2")}

    shown_scores = \
        settings.Setting(set(chain(*BUILTIN_ORDER.values())))

    class Error(OWWidget.Error):
        train_data_empty = Msg("Train dataset is empty.")
        test_data_empty = Msg("Test dataset is empty.")
        class_required = Msg("Train data input requires a target variable.")
        too_many_classes = Msg("Too many target variables.")
        class_required_test = Msg("Test data input requires a target variable.")
        too_many_folds = Msg("Number of folds exceeds the data size")
        class_inconsistent = Msg("Test and train datasets "
                                 "have different target variables.")
        memory_error = Msg("Not enough memory.")
        no_class_values = Msg("Target variable has no values.")
        only_one_class_var_value = Msg("Target variable has only one value.")

    class Warning(OWWidget.Warning):
        missing_data = \
            Msg("Instances with unknown target values were removed from{}data.")
        test_data_missing = Msg("Missing separate test data input.")
        scores_not_computed = Msg("Some scores could not be computed.")
        test_data_unused = Msg("Test data is present but unused. "
                               "Select 'Test on test data' to use it.")

    class Information(OWWidget.Information):
        data_sampled = Msg("Train data has been sampled")
        test_data_sampled = Msg("Test data has been sampled")

    def __init__(self):
        super().__init__()

        self.data = None
        self.test_data = None
        self.preprocessor = None
        self.train_data_missing_vals = False
        self.test_data_missing_vals = False
        self.scorers = []

        #: An Ordered dictionary with current inputs and their testing results.
        self.learners = OrderedDict()  # type: Dict[Any, Input]

        self.__state = State.Waiting
        # Do we need to [re]test any learners, set by _invalidate and
        # cleared by __update
        self.__needupdate = False
        self.__task = None  # type: Optional[Task]
        self.__executor = ThreadExecutor()

        sbox = gui.vBox(self.controlArea, "Sampling")
        rbox = gui.radioButtons(
            sbox, self, "resampling", callback=self._param_changed)

        gui.appendRadioButton(rbox, "Cross validation")
        ibox = gui.indentedBox(rbox)
        gui.comboBox(
            ibox, self, "n_folds", label="Number of folds: ",
            items=[str(x) for x in self.NFolds], maximumContentsLength=3,
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
            orientation=Qt.Horizontal, callback=self.fold_feature_changed)

        gui.appendRadioButton(rbox, "Random sampling")
        ibox = gui.indentedBox(rbox)
        gui.comboBox(
            ibox, self, "n_repeats", label="Repeat train/test: ",
            items=[str(x) for x in self.NRepeats], maximumContentsLength=3,
            orientation=Qt.Horizontal, callback=self.shuffle_split_changed)
        gui.comboBox(
            ibox, self, "sample_size", label="Training set size: ",
            items=["{} %".format(x) for x in self.SampleSizes],
            maximumContentsLength=5, orientation=Qt.Horizontal,
            callback=self.shuffle_split_changed)
        gui.checkBox(
            ibox, self, "shuffle_stratified", "Stratified",
            callback=self.shuffle_split_changed)

        gui.appendRadioButton(rbox, "Leave one out")

        gui.appendRadioButton(rbox, "Test on train data")
        gui.appendRadioButton(rbox, "Test on test data")

        self.cbox = gui.vBox(self.controlArea, "Target Class")
        self.class_selection_combo = gui.comboBox(
            self.cbox, self, "class_selection", items=[],
            sendSelectedValue=True, valueType=str,
            callback=self._on_target_class_changed,
            contentsLength=8)

        gui.rubber(self.controlArea)

        self.view = gui.TableView(
            wordWrap=True,
            editTriggers=gui.TableView.NoEditTriggers
        )
        header = self.view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setDefaultAlignment(Qt.AlignCenter)
        header.setStretchLastSection(False)
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self.show_column_chooser)

        self.result_model = QStandardItemModel(self)
        self.result_model.setHorizontalHeaderLabels(["Method"])
        self.view.setModel(self.result_model)
        self.view.setItemDelegate(ItemDelegate())

        box = gui.vBox(self.mainArea, "Evaluation Results")
        box.layout().addWidget(self.view)

    def sizeHint(self):
        return QSize(780, 1)

    def _update_controls(self):
        self.fold_feature = None
        self.feature_model.set_domain(None)
        if self.data:
            self.feature_model.set_domain(self.data.domain)
            if self.fold_feature is None and self.feature_model:
                self.fold_feature = self.feature_model[0]
        enabled = bool(self.feature_model)
        self.controls.resampling.buttons[
            OWTestLearners.FeatureFold].setEnabled(enabled)
        self.features_combo.setEnabled(enabled)
        if self.resampling == OWTestLearners.FeatureFold and not enabled:
            self.resampling = OWTestLearners.KFold

    @Inputs.learner
    def set_learner(self, learner, key):
        """
        Set the input `learner` for `key`.

        Parameters
        ----------
        learner : Optional[Orange.base.Learner]
        key : Any
        """
        if key in self.learners and learner is None:
            # Removed
            self._invalidate([key])
            del self.learners[key]
        else:
            self.learners[key] = InputLearner(learner, None, None)
            self._invalidate([key])

    @Inputs.train_data
    def set_train_data(self, data):
        """
        Set the input training dataset.

        Parameters
        ----------
        data : Optional[Orange.data.Table]
        """
        self.Information.data_sampled.clear()
        self.Error.train_data_empty.clear()
        self.Error.class_required.clear()
        self.Error.too_many_classes.clear()
        self.Error.no_class_values.clear()
        self.Error.only_one_class_var_value.clear()
        if data is not None and not len(data):
            self.Error.train_data_empty()
            data = None
        if data:
            conds = [not data.domain.class_vars,
                     len(data.domain.class_vars) > 1,
                     np.isnan(data.Y).all(),
                     data.domain.has_discrete_class and len(data.domain.class_var.values) == 1]
            errors = [self.Error.class_required,
                      self.Error.too_many_classes,
                      self.Error.no_class_values,
                      self.Error.only_one_class_var_value]
            for cond, error in zip(conds, errors):
                if cond:
                    error()
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
                self.resampling = OWTestLearners.FeatureFold
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
        if data is not None and not len(data):
            self.Error.test_data_empty()
            data = None
        if data and not data.domain.class_var:
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
        if self.resampling == OWTestLearners.TestOnTest:
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
        if self.data is None or self.data.domain.class_var is None:
            self.scorers = []
            return
        class_var = self.data and self.data.domain.class_var
        order = {name: i
                 for i, name in enumerate(self.BUILTIN_ORDER[type(class_var)])}
        # 'abstract' is retrieved from __dict__ to avoid inheriting
        usable = (cls for cls in scoring.Score.registry.values()
                  if cls.is_scalar and not cls.__dict__.get("abstract")
                  and isinstance(class_var, cls.class_types))
        self.scorers = sorted(usable, key=lambda cls: order.get(cls.name, 99))

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
        self._update_header()
        self._update_stats_model()
        if self.__needupdate:
            self.__update()

    def kfold_changed(self):
        self.resampling = OWTestLearners.KFold
        self._param_changed()

    def fold_feature_changed(self):
        self.resampling = OWTestLearners.FeatureFold
        self._param_changed()

    def shuffle_split_changed(self):
        self.resampling = OWTestLearners.ShuffleSplit
        self._param_changed()

    def _param_changed(self):
        self._invalidate()
        self.__update()

    def _update_header(self):
        # Set the correct horizontal header labels on the results_model.
        model = self.result_model
        model.setColumnCount(1 + len(self.scorers))
        for col, score in enumerate(self.scorers):
            item = QStandardItem(score.name)
            item.setToolTip(score.long_name)
            model.setHorizontalHeaderItem(col + 1, item)
        self._update_shown_columns()

    def _update_shown_columns(self):
        # pylint doesn't know that self.shown_scores is a set, not a Setting
        # pylint: disable=unsupported-membership-test
        model = self.result_model
        header = self.view.horizontalHeader()
        for section in range(1, model.columnCount()):
            col_name = model.horizontalHeaderItem(section).data(Qt.DisplayRole)
            header.setSectionHidden(section, col_name not in self.shown_scores)

    def _update_stats_model(self):
        # Update the results_model with up to date scores.
        # Note: The target class specific scores (if requested) are
        # computed as needed in this method.
        model = self.view.model()
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

        for key, slot in self.learners.items():
            name = learner_name(slot.learner)
            head = QStandardItem(name)
            head.setData(key, Qt.UserRole)
            if isinstance(slot.results, Try.Fail):
                head.setToolTip(str(slot.results.exception))
                head.setText("{} (error)".format(name))
                head.setForeground(QtGui.QBrush(Qt.red))
                errors.append("{name} failed with error:\n"
                              "{exc.__class__.__name__}: {exc!s}"
                              .format(name=name, exc=slot.results.exception))

            row = [head]

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
                for stat in stats:
                    item = QStandardItem()
                    if stat.success:
                        item.setText("{:.3f}".format(stat.value[0]))
                    else:
                        item.setToolTip(str(stat.exception))
                        has_missing_scores = True
                    row.append(item)

            model.appendRow(row)

        # Resort rows based on current sorting
        header = self.view.horizontalHeader()
        model.sort(
            header.sortIndicatorSection(),
            header.sortIndicatorOrder()
        )

        self.error("\n".join(errors), shown=bool(errors))
        self.Warning.scores_not_computed(shown=has_missing_scores)

    def _update_class_selection(self):
        self.class_selection_combo.setCurrentIndex(-1)
        self.class_selection_combo.clear()
        if not self.data:
            return

        if self.data.domain.has_discrete_class:
            self.cbox.setVisible(True)
            class_var = self.data.domain.class_var
            items = [self.TARGET_AVERAGE] + class_var.values
            self.class_selection_combo.addItems(items)

            class_index = 0
            if self.class_selection in class_var.values:
                class_index = class_var.values.index(self.class_selection) + 1

            self.class_selection_combo.setCurrentIndex(class_index)
            self.class_selection = items[class_index]
        else:
            self.cbox.setVisible(False)

    def _on_target_class_changed(self):
        self._update_stats_model()

    def _invalidate(self, which=None):
        self.fold_feature_selected = \
            self.resampling == OWTestLearners.FeatureFold
        # Invalidate learner results for `which` input keys
        # (if None then all learner results are invalidated)
        if which is None:
            which = self.learners.keys()

        model = self.view.model()
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

        self.__needupdate = True

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
        model = self.result_model
        header = self.view.horizontalHeader()
        for section in range(1, model.columnCount()):
            col_name = model.horizontalHeaderItem(section).data(Qt.DisplayRole)
            action = menu.addAction(col_name)
            action.setCheckable(True)
            action.setChecked(col_name in self.shown_scores)
            action.triggered.connect(partial(update, col_name))
        menu.exec(header.mapToGlobal(pos))

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
        self.report_table("Scores", self.view)

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
        self.progressBarSet(value, processEvents=False)

    def __update(self):
        self.__needupdate = False

        assert self.__task is None or self.__state == State.Running
        if self.__state == State.Running:
            self.cancel()

        self.Warning.test_data_unused.clear()
        self.Warning.test_data_missing.clear()
        self.warning()
        self.Error.class_inconsistent.clear()
        self.Error.too_many_folds.clear()
        self.error()

        # check preconditions and return early
        if self.data is None:
            self.__state = State.Waiting
            self.commit()
            return
        if not self.learners:
            self.__state = State.Waiting
            self.commit()
            return
        if self.resampling == OWTestLearners.KFold and \
                len(self.data) < self.NFolds[self.n_folds]:
            self.Error.too_many_folds()
            self.__state = State.Waiting
            self.commit()
            return

        elif self.resampling == OWTestLearners.TestOnTest:
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
        common_args = dict(
            store_data=True,
            preprocessor=self.preprocessor,
        )
        # items in need of an update
        items = [(key, slot) for key, slot in self.learners.items()
                 if slot.results is None]
        learners = [slot.learner for _, slot in items]

        # deepcopy all learners as they are not thread safe (by virtue of
        # the base API). These will be the effective learner objects tested
        # but will be replaced with the originals on return (see restore
        # learners bellow)
        learners_c = [copy.deepcopy(learner) for learner in learners]

        if self.resampling == OWTestLearners.KFold:
            folds = self.NFolds[self.n_folds]
            test_f = partial(
                Orange.evaluation.CrossValidation,
                self.data, learners_c, k=folds,
                random_state=rstate, **common_args)
        elif self.resampling == OWTestLearners.FeatureFold:
            test_f = partial(
                Orange.evaluation.CrossValidationFeature,
                self.data, learners_c, self.fold_feature,
                **common_args
            )
        elif self.resampling == OWTestLearners.LeaveOneOut:
            test_f = partial(
                Orange.evaluation.LeaveOneOut,
                self.data, learners_c, **common_args
            )
        elif self.resampling == OWTestLearners.ShuffleSplit:
            train_size = self.SampleSizes[self.sample_size] / 100
            test_f = partial(
                Orange.evaluation.ShuffleSplit,
                self.data, learners_c,
                n_resamples=self.NRepeats[self.n_repeats],
                train_size=train_size, test_size=None,
                stratified=self.shuffle_stratified,
                random_state=rstate, **common_args
            )
        elif self.resampling == OWTestLearners.TestOnTrain:
            test_f = partial(
                Orange.evaluation.TestOnTrainingData,
                self.data, learners_c, **common_args
            )
        elif self.resampling == OWTestLearners.TestOnTest:
            test_f = partial(
                Orange.evaluation.TestOnTestData,
                self.data, self.test_data, learners_c, **common_args
            )
        else:
            assert False, "self.resampling %s" % self.resampling

        def replace_learners(evalfunc, *args, **kwargs):
            res = evalfunc(*args, **kwargs)
            assert all(lc is lo for lc, lo in zip(learners_c, res.learners))
            res.learners[:] = learners
            return res

        test_f = partial(replace_learners, test_f)

        self.__submit(test_f)

    def __submit(self, testfunc):
        # type: (Callable[[Callable[float]], Results]) -> None
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
        task = Task()

        def progress_callback(finished):
            if task.cancelled:
                raise UserInterrupt()
            QMetaObject.invokeMethod(
                self, "setProgressValue", Qt.QueuedConnection,
                Q_ARG(float, 100 * finished)
            )

        def ondone(_):
            QMetaObject.invokeMethod(
                self, "__task_complete", Qt.QueuedConnection,
                Q_ARG(object, task))

        testfunc = partial(testfunc, callback=progress_callback)
        task.future = self.__executor.submit(testfunc)
        task.future.add_done_callback(ondone)

        self.progressBarInit(processEvents=None)
        self.setBlocking(True)
        self.setStatusMessage("Running")

        self.__state = State.Running
        self.__task = task

    @Slot(object)
    def __task_complete(self, task):
        # handle a completed task
        assert self.thread() is QThread.currentThread()
        if self.__task is not task:
            assert task.cancelled
            log.debug("Reaping cancelled task: %r", "<>")
            return

        self.setBlocking(False)
        self.progressBarFinished(processEvents=None)
        self.setStatusMessage("")
        result = task.future
        assert result.done()
        self.__task = None
        try:
            results = result.result()    # type: Results
            learners = results.learners  # type: List[Learner]
        except Exception as er:
            log.exception("testing error (in __task_complete):",
                          exc_info=True)
            self.error("\n".join(traceback.format_exception_only(type(er), er)))
            self.__state = State.Done
            return

        self.__state = State.Done

        learner_key = {slot.learner: key for key, slot in
                       self.learners.items()}
        assert all(learner in learner_key for learner in learners)

        # Update the results for individual learners
        class_var = results.domain.class_var
        for learner, result in zip(learners, results.split_by_model()):
            stats = None
            if class_var.is_primitive():
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

        self._update_header()
        self._update_stats_model()

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
            assert task.future.done()

    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()


def scorer_caller(scorer, ovr_results, target=None):
    if scorer.is_binary:
        return lambda: scorer(ovr_results, target=target, average='weighted')
    else:
        return lambda: scorer(ovr_results)


class UserInterrupt(BaseException):
    """
    A BaseException subclass used for cooperative task/thread cancellation
    """
    pass


def learner_name(learner):
    """Return the value of `learner.name` if it exists, or the learner's type
    name otherwise"""
    return getattr(learner, "name", type(learner).__name__)


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
        res.models = [xm + ym for xm, ym in zip(x.models, y.models)]
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
        values=["False", "True"],
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


class Task:
    """
    A simple task state.
    """
    #: A future holding the results. This field is set by the client.
    future = ...        # type: Future
    #: True if the task was cancelled
    cancelled = False   # type: bool
    #: A function to call. Filled by the client.
    func = ...          # type: Callable[Callable[float], Results]

    def cancel(self):
        """
        Cancel the task.

        Set the `cancelled` field to True and block until the future is done.
        """
        log.debug("cancel task")
        self.cancelled = True
        cancelled = self.future.cancel()
        if cancelled:
            log.debug("Task cancelled before starting")
        else:
            log.debug("Attempting cooperative cancellation for task")
        concurrent.futures.wait([self.future])


def main(argv=None):
    """Show and test the widget"""
    from AnyQt.QtWidgets import QApplication
    logging.basicConfig(level=logging.DEBUG)
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QApplication(argv)
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "iris"

    data = Table(filename)
    class_var = data.domain.class_var

    if class_var is None:
        return 1
    elif class_var.is_discrete:
        learners = [lambda data: 1 / 0,
                    Orange.classification.LogisticRegressionLearner(),
                    Orange.classification.MajorityLearner(),
                    Orange.classification.NaiveBayesLearner()]
    else:
        learners = [lambda data: 1 / 0,
                    Orange.regression.MeanLearner(),
                    Orange.regression.KNNRegressionLearner(),
                    Orange.regression.RidgeRegressionLearner()]

    w = OWTestLearners()
    w.show()
    w.set_train_data(data)
    w.set_test_data(data)

    for i, learner in enumerate(learners):
        w.set_learner(learner, i)

    w.handleNewSignals()
    rval = app.exec_()

    for i in range(len(learners)):
        w.set_learner(None, i)
    w.handleNewSignals()
    w.saveSettings()
    return rval

if __name__ == "__main__":
    sys.exit(main())
