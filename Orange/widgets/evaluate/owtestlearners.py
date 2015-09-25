import sys
import abc
import functools

from collections import OrderedDict, namedtuple

import numpy as np

from PyQt4 import QtGui
from PyQt4.QtGui import (
    QTreeView, QStandardItemModel, QStandardItem, QHeaderView,
    QStyledItemDelegate
)
from PyQt4.QtCore import Qt, QSize

import Orange.data
import Orange.evaluation
import Orange.classification
import Orange.regression

from Orange.base import Learner
from Orange.evaluation import scoring
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets import widget, gui, settings


Input = namedtuple(
    "Input",
    ["learner",  # :: Orange.base.Learner
     "results",  # :: Option[Try[Orange.evaluation.Results]]
     "stats"]    # :: Option[Sequence[Try[float]]]
)


def classification_stats(results):
    return tuple(score(results) for score in classification_stats.scores)

classification_stats.headers, classification_stats.scores = zip(*(
    ("AUC", scoring.AUC),
    ("CA", scoring.CA),
    ("F1", scoring.F1),
    ("Precision", scoring.Precision),
    ("Recall", scoring.Recall),
))


def regression_stats(results):
    return tuple(score(results) for score in regression_stats.scores)

regression_stats.headers, regression_stats.scores = zip(*(
    ("MSE", scoring.MSE),
    ("RMSE", scoring.RMSE),
    ("MAE", scoring.MAE),
    ("R2", scoring.R2),
))


class ItemDelegate(QStyledItemDelegate):
    def sizeHint(self, *args):
        size = super().sizeHint(*args)
        return QSize(size.width(), size.height() + 6)


class Try(abc.ABC):
    # Try to walk in a Turing tar pit.

    class Success:
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


class OWTestLearners(widget.OWWidget):
    name = "Test & Score"
    description = "Cross-validation accuracy estimation."
    icon = "icons/TestLearners1.svg"
    priority = 100

    inputs = [("Learner", Learner,
               "set_learner", widget.Multiple),
              ("Data", Orange.data.Table, "set_train_data", widget.Default),
              ("Test Data", Orange.data.Table, "set_test_data"),
              ("Preprocessor", Preprocess, "set_preprocessor")]

    outputs = [("Evaluation Results", Orange.evaluation.Results)]

    settingsHandler = settings.ClassValuesContextHandler()

    #: Resampling/testing types
    KFold, LeaveOneOut, ShuffleSplit, TestOnTrain, TestOnTest = 0, 1, 2, 3, 4

    #: Selected resampling type
    resampling = settings.Setting(0)
    #: Number of folds for K-fold cross validation
    k_folds = settings.Setting(10)
    #: Number of repeats for ShuffleSplit sampling
    n_repeat = settings.Setting(10)
    #: ShuffleSplit sampling p
    sample_p = settings.Setting(75)

    TARGET_AVERAGE = "(Average over classes)"
    class_selection = settings.ContextSetting(TARGET_AVERAGE)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None
        self.test_data = None
        self.preprocessor = None

        #: An Ordered dictionary with current inputs and their testing
        #: results.
        self.learners = OrderedDict()

        sbox = gui.widgetBox(self.controlArea, "Sampling")
        rbox = gui.radioButtons(
            sbox, self, "resampling", callback=self._param_changed
        )
        gui.appendRadioButton(rbox, "Cross validation")
        ibox = gui.indentedBox(rbox)
        gui.spin(ibox, self, "k_folds", 2, 50, label="Number of folds:",
                 callback=self.kfold_changed)
        gui.appendRadioButton(rbox, "Leave one out")
        gui.appendRadioButton(rbox, "Random sampling")
        ibox = gui.indentedBox(rbox)
        gui.spin(ibox, self, "n_repeat", 2, 50, label="Repeat train/test",
                 callback=self.shuffle_split_changed)
        gui.widgetLabel(ibox, "Relative training set size:")
        gui.hSlider(ibox, self, "sample_p", minValue=1, maxValue=99,
                    ticks=20, vertical=False, labelFormat="%d %%",
                    callback=self.shuffle_split_changed)

        gui.appendRadioButton(rbox, "Test on train data")
        gui.appendRadioButton(rbox, "Test on test data")

        rbox.layout().addSpacing(5)
        self.apply_button = gui.button(
            rbox, self, "Apply", callback=self.apply, default=True)

        self.cbox = gui.widgetBox(self.controlArea, "Target class")
        self.class_selection_combo = gui.comboBox(
            self.cbox, self, "class_selection", items=[],
            sendSelectedValue=True, valueType=str,
            callback=self._on_target_class_changed,
            contentsLength=8)

        gui.rubber(self.controlArea)

        self.view = QTreeView(
            rootIsDecorated=False,
            uniformRowHeights=True,
            wordWrap=True,
            editTriggers=QTreeView.NoEditTriggers
        )
        header = self.view.header()
        header.setResizeMode(QHeaderView.ResizeToContents)
        header.setDefaultAlignment(Qt.AlignCenter)
        header.setStretchLastSection(False)

        self.result_model = QStandardItemModel(self)
        self.result_model.setHorizontalHeaderLabels(["Method"])
        self.view.setModel(self.result_model)
        self.view.setItemDelegate(ItemDelegate())

        box = gui.widgetBox(self.mainArea, "Evaluation Results")
        box.layout().addWidget(self.view)

    def sizeHint(self):
        return QSize(780, 1)

    def set_learner(self, learner, key):
        """
        Set the input `learner` for `key`.
        """
        if key in self.learners and learner is None:
            # Removed
            del self.learners[key]
        else:
            self.learners[key] = Input(learner, None, None)
            self._invalidate([key])

    def set_train_data(self, data):
        """
        Set the input training dataset.
        """
        self.error(0)
        if data and not data.domain.class_var:
            self.error(0, "Train data input requires a class variable")
            data = None

        self.data = data
        self.closeContext()
        if data is not None:
            self.openContext(data.domain.class_var)
        self._invalidate()

    def set_test_data(self, data):
        """
        Set the input separate testing dataset.
        """
        self.error(1)
        if data and not data.domain.class_var:
            self.error(1, "Test data input requires a class variable")
            data = None

        self.test_data = data
        if self.resampling == OWTestLearners.TestOnTest:
            self._invalidate()

    def set_preprocessor(self, preproc):
        """
        Set the input preprocessor to apply on the training data.
        """
        self.preprocessor = preproc
        self._invalidate()

    def handleNewSignals(self):
        """Reimplemented from OWWidget.handleNewSignals."""
        self._update_class_selection()
        self.apply()

    def kfold_changed(self):
        self.resampling = OWTestLearners.KFold
        self._param_changed()

    def shuffle_split_changed(self):
        self.resampling = OWTestLearners.ShuffleSplit
        self._param_changed()

    def _param_changed(self):
        self._invalidate()

    def _update_results(self):
        """
        Run/evaluate the learners.
        """
        self.warning([1, 2])
        self.error(2)

        if self.data is None:
            return

        class_var = self.data.domain.class_var

        if self.resampling == OWTestLearners.TestOnTest:
            if self.test_data is None:
                self.warning(2, "Missing separate test data input")
                return
            elif self.test_data.domain.class_var != class_var:
                self.error(2, ("Inconsistent class variable between test " +
                               "and train data sets"))
                return

        # items in need of an update
        items = [(key, slot) for key, slot in self.learners.items()
                 if slot.results is None]
        learners = [slot.learner for _, slot in items]

        if self.test_data is not None and \
                self.resampling != OWTestLearners.TestOnTest:
            self.warning(1, "Test data is present but unused. "
                            "Select 'Test on test data' to use it.")

        rstate = 42
        def update_progress(finished):
            self.progressBarSet(100 * finished)
        common_args = dict(
            store_data=True,
            preprocessor=self.preprocessor,
            callback=update_progress)
        self.setStatusMessage("Running")
        self.progressBarInit()
        if self.resampling == OWTestLearners.KFold:
            results = Orange.evaluation.CrossValidation(
                self.data, learners, k=self.k_folds, random_state=rstate,
                **common_args)
        elif self.resampling == OWTestLearners.LeaveOneOut:
            results = Orange.evaluation.LeaveOneOut(
                self.data, learners, **common_args)
        elif self.resampling == OWTestLearners.ShuffleSplit:
            train_size = self.sample_p / 100
            results = Orange.evaluation.ShuffleSplit(
                self.data, learners, n_resamples=self.n_repeat,
                train_size=train_size, test_size=None,
                random_state=rstate, **common_args)
        elif self.resampling == OWTestLearners.TestOnTrain:
            results = Orange.evaluation.TestOnTrainingData(
                self.data, learners, **common_args)
        elif self.resampling == OWTestLearners.TestOnTest:
            results = Orange.evaluation.TestOnTestData(
                self.data, self.test_data, learners, **common_args)
        else:
            assert False

        learner_key = {slot.learner: key for key, slot in self.learners.items()}
        for learner, result in zip(learners, split_by_model(results)):
            stats = None
            if class_var.is_discrete:
                scorers = classification_stats.scores
            elif class_var.is_continuous:
                scorers = regression_stats.scores
            else:
                scorers = None
            if scorers:
                ex = result.failed[0]
                if ex:
                    stats = [Try.Fail(ex)] * len(scorers)
                    result = Try.Fail(ex)
                else:
                    stats = [Try(lambda: score(result)) for score in scorers]
                    result = Try.Success(result)
            key = learner_key[learner]
            self.learners[key] = \
                self.learners[key]._replace(results=result, stats=stats)

        self.setStatusMessage("")
        self.progressBarFinished()

    def _update_header(self):
        # Set the correct horizontal header labels on the results_model.
        headers = ["Method"]
        if self.data is not None:
            if self.data.domain.has_discrete_class:
                headers.extend(classification_stats.headers)
            else:
                headers.extend(regression_stats.headers)

        # remove possible extra columns from the model.
        for i in reversed(range(len(headers),
                                self.result_model.columnCount())):
            self.result_model.takeColumn(i)

        self.result_model.setHorizontalHeaderLabels(headers)

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
            if class_var.is_discrete and \
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

                    stats = [Try(lambda: score(ovr_results))
                             for score in classification_stats.scores]
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

        if errors:
            self.error(3, "\n".join(errors))
        else:
            self.error(3)

        if has_missing_scores:
            self.warning(3, "Some scores could not be computed")
        else:
            self.warning(3)

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

        self.apply_button.setEnabled(True)

    def apply(self):
        self.apply_button.setEnabled(False)
        self._update_header()
        # Update the view to display the model names
        self._update_stats_model()
        self._update_results()
        self._update_stats_model()
        self.commit()

    def commit(self):
        valid = [slot for slot in self.learners.values()
                 if slot.results is not None and slot.results.success]
        if valid:
            combined = results_merge([slot.results.value for slot in valid])
            combined.learner_names = [learner_name(slot.learner)
                                      for slot in valid]
        else:
            combined = None
        self.send("Evaluation Results", combined)



def learner_name(learner):
    return getattr(learner, "name", type(learner).__name__)


def split_by_model(results):
    """
    Split evaluation results by models
    """
    data = results.data
    nmethods = len(results.predicted)
    for i in range(nmethods):
        res = Orange.evaluation.Results()
        res.data = data
        res.domain = results.domain
        res.row_indices = results.row_indices
        res.actual = results.actual
        res.predicted = results.predicted[(i,), :]

        if getattr(results, "probabilities", None) is not None:
            res.probabilities = results.probabilities[(i,), :, :]

        if results.models:
            res.models = [mf[i] for mf in results.models]

        if results.folds:
            res.folds = results.folds

        res.failed = [results.failed[i]]
        yield res


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
    return functools.reduce(results_add_by_model, results,
                            Orange.evaluation.Results())


def results_one_vs_rest(results, pos_index):
    actual = results.actual == pos_index
    predicted = results.predicted == pos_index
    return Orange.evaluation.Results(
        nmethods=1, domain=results.domain,
        actual=actual, predicted=predicted)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QtGui.QApplication(argv)
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "iris"

    data = Orange.data.Table(filename)
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
    return rval

if __name__ == "__main__":
    sys.exit(main())
