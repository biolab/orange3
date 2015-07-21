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
from PyQt4.QtCore import Qt, QSize, QObject, QEvent, QCoreApplication
from PyQt4.QtCore import pyqtSignal as Signal

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


class AsyncUpdateLoop(QObject):
    Next = QEvent.registerEventType()

    #: State flags
    Idle, Running, Cancelled, Finished = 0, 1, 2, 3
    yielded = Signal(object)
    finished = Signal()

    returned = Signal(object)
    raised = Signal(object)
    cancelled = Signal()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__update_loop = None
        self.__next_pending = False
        self.__in_next = False
        self.__state = AsyncUpdateLoop.Idle

    def set_update_loop(self, loop):
        if self.__update_loop is not None:
            self.__update_loop.close()
            self.__update_loop = None
            self.__state = AsyncUpdateLoop.Cancelled

            self.cancelled.emit()
            self.finished.emit()

        if loop is not None:
            self.__update_loop = loop
            self.__state = AsyncUpdateLoop.Running
            self.__schedule_next()

    def cancel(self):
        self.set_update_loop(None)

    def state(self):
        return self.__state

    def __schedule_next(self):
        if not self.__next_pending:
            self.__next_pending = True
            QCoreApplication.postEvent(
                self, QEvent(AsyncUpdateLoop.Next), Qt.LowEventPriority)

    def __next(self):
        if self.__update_loop is not None:
            try:
                rval = next(self.__update_loop)
            except StopIteration as stop:
                self.__state = AsyncUpdateLoop.Finished
                self.returned.emit(stop.value)
                self.finished.emit()
                self.__update_loop = None
            except BaseException as er:
                self.__state = AsyncUpdateLoop.Finished
                self.raised.emit(er)
                self.finished.emit()
                self.__update_loop = None
            else:
                self.yielded.emit(rval)
                self.__schedule_next()

    def customEvent(self, event):
        if event.type() == AsyncUpdateLoop.Next:
            assert self.__next_pending
            self.__next_pending = False
            if not self.__in_next:
                self.__in_next = True
                try:
                    self.__next()
                finally:
                    self.__in_next = False
            else:
                # warn
                self.__schedule_next()


class Try(abc.ABC):
    # Try to walk in a Turing tar pit.

    class Success(object):
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

    class Fail(object):
        __slots__ = ("__exception")
#         __bool__ = lambda self: False
        success = property(lambda self: False)
        exception = property(lambda self: self.__exception)

        def __init__(self, exception):
            self.__exception = exception

        def __getnewargs__(self):
            return (self.exception,)

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
    name = "Test Learners"
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
    KFold, LeaveOneOut, Bootstrap, TestOnTrain, TestOnTest = 0, 1, 2, 3, 4

    #: Selected resampling type
    resampling = settings.Setting(0)
    #: Number of folds for K-fold cross validation
    k_folds = settings.Setting(10)
    #: Number of repeats for bootstrap sampling
    n_repeat = settings.Setting(10)
    #: Bootstrap sampling p
    sample_p = settings.Setting(75)

    class_selection = settings.ContextSetting("(None)")

    def __init__(self, parent=None):
        super().__init__(parent)

        self.orig_train_data = None
        self.train_data = None
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
                 callback=self.bootstrap_changed)
        gui.widgetLabel(ibox, "Relative training set size:")
        gui.hSlider(ibox, self, "sample_p", minValue=1, maxValue=100,
                    ticks=20, vertical=False, labelFormat="%d %%",
                    callback=self.bootstrap_changed)

        gui.appendRadioButton(rbox, "Test on train data")
        gui.appendRadioButton(rbox, "Test on test data")

        rbox.layout().addSpacing(5)
        gui.button(rbox, self, "Apply", callback=self.apply)

        self.cbox = gui.widgetBox(self.controlArea, "Target class")
        self.class_selection_combo = gui.comboBox(
            self.cbox, self, "class_selection", items=[],
            sendSelectedValue=True, valueType=str,
            callback=self._on_target_class_changed)

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
        self.result_model.setHorizontalHeaderLabels(["Model"])
        self.view.setModel(self.result_model)
        self.view.setItemDelegate(ItemDelegate())

        box = gui.widgetBox(self.mainArea, "Evaluation Results")
        box.layout().addWidget(self.view)

        self.__loop = AsyncUpdateLoop(parent=self)
        self.__loop.yielded.connect(self.__add_result)
        self.__loop.finished.connect(self.__on_finished)
        self.__loop.returned.connect(self.__on_returned)

    def set_learner(self, learner, key):
        """
        Set the input `learner` for `key`.
        """
        if key in self.learners and learner is None:
            # Removed
            del self.learners[key]
        else:
            self.learners[key] = Input(learner, None, None)

        self._update_stats_model()

    def set_train_data(self, data):
        """
        Set the input training dataset.
        """
        self.error(0)
        if data and not data.domain.class_var:
            self.error(0, "Train data input requires a class variable")
            data = None

        self.orig_train_data = data
        self.train_data = None
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
        self.train_data = None
        self._invalidate()

    def handleNewSignals(self):
        """Reimplemented from OWWidget.handleNewSignals."""
        if self.train_data is None:
            if self.preprocessor and self.orig_train_data:
                self.train_data = self.preprocessor(self.orig_train_data)
            else:
                self.train_data = self.orig_train_data

            self._update_class_selection()

        self._update_header()
        self.apply()

    def kfold_changed(self):
        self.resampling = OWTestLearners.KFold
        self._param_changed()

    def bootstrap_changed(self):
        self.resampling = OWTestLearners.Bootstrap
        self._param_changed()

    def _param_changed(self):
        self._invalidate()

    def _start_update(self):
        """
        Run/evaluate the learners.
        """
        self.warning([1, 2])
        self.error(2)

        if self.train_data is None:
            return

        if self.resampling == OWTestLearners.TestOnTest:
            if self.test_data is None:
                self.warning(2, "Missing separate test data input")
                return

            elif self.test_data.domain.class_var != \
                    self.train_data.domain.class_var:
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
        if self.resampling == OWTestLearners.KFold:
            def evaluate(learner, data=self.train_data, k=self.k_folds):
                return Orange.evaluation.CrossValidation(
                    data, [learner], k=k, random_state=rstate, store_data=True)
        elif self.resampling == OWTestLearners.LeaveOneOut:
            def evaluate(learner, data=self.train_data):
                return Orange.evaluation.LeaveOneOut(
                    data, [learner], store_data=True)
        elif self.resampling == OWTestLearners.Bootstrap:
            def evaluate(learner, data=self.train_data,
                         n_resamples=self.n_repeat, p=self.sample_p / 100.0):
                return Orange.evaluation.Bootstrap(
                    data, [learner], n_resamples=n_resamples, p=p,
                    random_state=rstate, store_data=True)
        elif self.resampling == OWTestLearners.TestOnTrain:
            def evaluate(learner, data=self.train_data):
                return Orange.evaluation.TestOnTrainingData(
                    data, [learner], store_data=True)
        elif self.resampling == OWTestLearners.TestOnTest:
            def evaluate(learner, train=self.train_data, test=self.test_data):
                return Orange.evaluation.TestOnTestData(
                    train, test, [learner], store_data=True)
        else:
            assert False

        def update_loop(learners):
            for i, learner in enumerate(learners):
                yield (i, learner, Try(lambda: evaluate(learner)))

        self.setStatusMessage("Running")
        self.progressBarInit(processEvents=None)
        self.__loop.set_update_loop(update_loop(learners))

    def __add_result(self, r):
        # add evaluation results for a single learner.
        # :type r: (int, Learner, Try[Results])
        step, learner, result = r

        if not result.success:
            sys.excepthook(type(result.exception), result.exception, None)
            # Strip the captured traceback from the exception.
            result = Try.Fail(type(result.exception)(*result.exception.args))

        class_var = self.train_data.domain.class_var

        if result.success and class_var.is_discrete:
            stats = [Try(lambda: score(result.value))
                     for score in classification_stats.scores]
        elif result.success:
            assert class_var.is_continuous
            stats = [Try(lambda: score(result.value))
                     for score in regression_stats.scores]
        else:
            stats = None

        for key, slot in list(self.learners.items()):
            if slot.learner is learner:
                self.learners[key] = slot._replace(results=result, stats=stats)

        self.progressBarSet(100 * step / len(self.learners),
                            processEvents=False)

    def __on_finished(self):
        # The update coroutine has finished normally, by
        # error or by interruption.
        self.setStatusMessage("")
        self.progressBarFinished(processEvents=None)

    def __on_returned(self, rval):
        # The update coroutine has returned in a normal fashion.
        self._update_stats_model()
        self.commit()

    def _update_header(self):
        # Set the correct horizontal header labels on the results_model.
        headers = ["Method"]
        if self.train_data is not None:
            if self.train_data.domain.has_discrete_class:
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

        if self.train_data is None:
            return

        target_index = None
        class_var = self.train_data.domain.class_var
        if class_var.is_discrete and self.class_selection != "(None)":
            target_index = class_var.values.index(self.class_selection)

        for slot in self.learners.values():
            name = learner_name(slot.learner)
            head = QStandardItem(name)

            if isinstance(slot.results, Try.Fail):
                head.setToolTip(str(slot.results.exception))
                head.setText("{} (error)".format(name))
                head.setForeground(QtGui.QBrush(Qt.red))

            row = [head]

            if class_var.is_discrete and target_index is not None:
                if slot.results.success:
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
                    row.append(item)

            model.appendRow(row)

    def _update_class_selection(self):
        self.class_selection_combo.setCurrentIndex(-1)
        self.class_selection_combo.clear()
        if not self.train_data:
            return

        if self.train_data.domain.has_discrete_class:
            self.cbox.setVisible(True)
            class_var = self.train_data.domain.class_var
            items = ["(None)"] + class_var.values
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
        if which is None:
            which = self.learners.keys()

        self.__loop.set_update_loop(None)

        all_keys = list(self.learners.keys())
        model = self.view.model()

        for key in which:
            self.learners[key] = \
                self.learners[key]._replace(results=None, stats=None)

            if key in self.learners:
                row = all_keys.index(key)
                for c in range(1, model.columnCount()):
                    item = model.item(row, c)
                    if item is not None:
                        item.setData(None, Qt.DisplayRole)
                        item.setData(None, Qt.ToolTipRole)

    def apply(self):
        self._update_header()
        if self.train_data is not None:
            self._start_update()
        else:
            # Clear the output
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

    def onDeleteWidget(self):
        super().onDeleteWidget()
        self.__loop.set_update_loop(None)


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
