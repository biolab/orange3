from collections import OrderedDict, namedtuple
import functools

import numpy as np

from PyQt4 import QtGui
from PyQt4.QtGui import QTreeView, QStandardItemModel, QStandardItem, \
    QHeaderView, QItemDelegate
from PyQt4.QtCore import Qt, QSize

import Orange
from Orange.evaluation import *
from Orange.widgets import widget, gui, settings
from Orange.data import Domain


Input = namedtuple("Input", ["learner", "results", "stats"])


def classification_stats(results):
    return (AUC(results),
            CA(results),
            F1(results),
            Precision(results),
            Recall(results))

classification_stats.headers = ["AUC", "CA", "F1", "Precision", "Recall"]


def regression_stats(results):
    return (MSE(results),
            RMSE(results),
            MAE(results),
            R2(results))

regression_stats.headers = ["MSE", "RMSE", "MAE", "R2"]


class ItemDelegate(QItemDelegate):
    def sizeHint(self, *args):
        size = super().sizeHint(*args)
        return QSize(size.width(), size.height() + 6)


class OWTestLearners(widget.OWWidget):
    name = "Test Learners"
    description = "Cross-validation accuracy estimation."
    icon = "icons/TestLearners1.svg"
    priority = 100

    inputs = [("Learner", Orange.classification.Learner,
               "set_learner", widget.Multiple),
              ("Data", Orange.data.Table, "set_train_data", widget.Default),
              ("Test Data", Orange.data.Table, "set_test_data")]

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

        self.train_data = None
        self.test_data = None

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
        self.class_selection_combo = gui.comboBox(self.cbox, self, "class_selection",
             items=[],
             callback=self._select_class,
             sendSelectedValue=True, valueType=str)

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

        self.result_model = QStandardItemModel()
        self.view.setModel(self.result_model)
        self.view.setItemDelegate(ItemDelegate())
        self._update_header()
        box = gui.widgetBox(self.mainArea, "Evaluation Results")
        box.layout().addWidget(self.view)

    def set_learner(self, learner, key):
        if key in self.learners and learner is None:
            del self.learners[key]
        else:
            self.learners[key] = Input(learner, None, ())
        self._update_stats_model()

    def set_train_data(self, data):
        self.error(0)
        if data and not data.domain.class_var:
            self.error(0, "Train data input requires a class variable")
            data = None

        self.train_data = data
        self.closeContext()
        if data is not None:
            self.openContext(data.domain.class_var)
            self._update_header()
        self._update_class_selection()
        self._invalidate()

    def set_test_data(self, data):
        self.error(1)
        if data and not data.domain.class_var:
            self.error(1, "Test data input requires a class variable")
            data = None

        self.test_data = data
        if self.resampling == OWTestLearners.TestOnTest:
            self._invalidate()

    def handleNewSignals(self):
        self.update_results()
        self.commit()

    def kfold_changed(self):
        self.resampling = OWTestLearners.KFold
        self._param_changed()

    def bootstrap_changed(self):
        self.resampling = OWTestLearners.Bootstrap
        self._param_changed()

    def _param_changed(self):
        self._invalidate()

    def update_results(self):
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
        items = [(key, input) for key, input in self.learners.items()
                 if input.results is None]
        learners = [input.learner for _, input in items]

        self.setStatusMessage("Running")
        if self.test_data is not None and \
                self.resampling != OWTestLearners.TestOnTest:
            self.warning(1, "Test data is present but unused. "
                            "Select 'Test on test data' to use it.")

        # TODO: Test each learner individually
        try:
            if self.resampling == OWTestLearners.KFold:
                results = Orange.evaluation.CrossValidation(
                    self.train_data, learners, k=self.k_folds, store_data=True
                )
            elif self.resampling == OWTestLearners.LeaveOneOut:
                results = Orange.evaluation.LeaveOneOut(
                    self.train_data, learners, store_data=True
                )
            elif self.resampling == OWTestLearners.Bootstrap:
                p = self.sample_p / 100.0
                results = Orange.evaluation.Bootstrap(
                    self.train_data, learners, n_resamples=self.n_repeat, p=p,
                    store_data=True
                )
            elif self.resampling == OWTestLearners.TestOnTrain:
                results = Orange.evaluation.TestOnTrainingData(
                    self.train_data, learners, store_data=True
                )
            elif self.resampling == OWTestLearners.TestOnTest:
                if self.test_data is None:
                    return
                results = Orange.evaluation.TestOnTestData(
                    self.train_data, self.test_data, learners, store_data=True
                )
            else:
                assert False
        except Exception as e:
            self.error(2, str(e))
            return

        self.results = results
        results = list(split_by_model(results))

        class_var = self.train_data.domain.class_var

        if class_var.is_discrete:
            stats = [classification_stats(self.one_vs_rest(res)) for res in results]
        else:
            stats = [regression_stats(res) for res in results]

        self._update_header()

        for (key, input), res, stat in zip(items, results, stats):
            self.learners[key] = input._replace(results=res, stats=stat)

        self.setStatusMessage("")

        self._update_stats_model()


    def _update_header(self):
        headers = ["Method"]
        if self.train_data is not None:
            if self.train_data.domain.class_var.is_discrete:
                headers.extend(classification_stats.headers)
            else:
                headers.extend(regression_stats.headers)
        for i in reversed(range(len(headers),
                                self.result_model.columnCount())):
            self.result_model.takeColumn(i)

        self.result_model.setHorizontalHeaderLabels(headers)

    def _update_stats_model(self):
        model = self.view.model()

        for r in reversed(range(model.rowCount())):
            model.takeRow(r)

        for input in self.learners.values():
            name = learner_name(input.learner)
            row = []
            head = QStandardItem()
            head.setData(name, Qt.DisplayRole)
            row.append(head)
            for stat in input.stats:
                item = QStandardItem()
                item.setData(" {:.3f} ".format(stat[0]), Qt.DisplayRole)
                row.append(item)
            model.appendRow(row)

    def _update_class_selection(self):
        self.class_selection_combo.clear()
        if not self.train_data:
            return
        if self.train_data.domain.class_var.is_discrete:
            self.cbox.setVisible(True)
            values = self.train_data.domain.class_var.values
            self.class_selection_combo.addItem("(None)")
            self.class_selection_combo.addItems(values)

            class_index = 0
            if self.class_selection in self.train_data.domain.class_var.values:
                    class_index = self.train_data.domain.class_var.values.index(self.class_selection)+1
            else:
                self.class_selection = '(None)'

            self.class_selection_combo.setCurrentIndex(class_index)
            self.previous_class_selection = "(None)"
        else:
            self.cbox.setVisible(False)

    def one_vs_rest(self, res):
        if self.class_selection != '(None)' and self.class_selection != 0:
            class_ = self.train_data.domain.class_var.values.index(self.class_selection)
            actual = res.actual == class_
            predicted = res.predicted == class_
            return Results(
                nmethods=1, domain=self.train_data.domain,
                actual=actual, predicted=predicted)
        else:
            return res

    def _select_class(self):
        if self.previous_class_selection == self.class_selection:
            return

        results = list(split_by_model(self.results))

        items = [(key, input) for key, input in self.learners.items()]
        learners = [input.learner for _, input in items]

        class_var = self.train_data.domain.class_var
        if class_var.is_discrete:
            stats = [classification_stats(self.one_vs_rest(res)) for res in results]
        else:
            stats = [regression_stats(res) for res in results]

        for (key, input), res, stat in zip(items, results, stats):
            self.learners[key] = input._replace(results=res, stats=stat)

        self.setStatusMessage("")

        self._update_stats_model()
        self.previous_class_selection = self.class_selection

    def _invalidate(self, which=None):
        if which is None:
            which = self.learners.keys()

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

    def apply(self):
        self.update_results()
        self.commit()

    def commit(self):
        results = [val.results for val in self.learners.values()
                   if val.results is not None]
        if results:
            combined = results_merge(results)
            combined.learner_names = [learner_name(val.learner)
                                     for val in self.learners.values()]
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


def main():
    app = QtGui.QApplication([])
    data = Orange.data.Table("iris")
    w = OWTestLearners()
    w.show()
    w.set_train_data(data)
    w.set_test_data(data)
    w.set_learner(Orange.classification.LogisticRegressionLearner(), 1)
    w.set_learner(Orange.classification.MajorityLearner(), 2)
    w.handleNewSignals()
    return app.exec_()

if __name__ == "__main__":
    import sys
    sys.exit(main())
