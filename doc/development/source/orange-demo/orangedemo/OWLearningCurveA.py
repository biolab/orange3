import sys
from collections import OrderedDict

import numpy
from AnyQt.QtWidgets import QTableWidget, QTableWidgetItem

import Orange.data
import Orange.classification
import Orange.evaluation

from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input
from Orange.evaluation.testing import Results


class OWLearningCurveA(OWWidget):
    name = "Learning Curve (A)"
    description = ("Takes a dataset and a set of learners and shows a "
                   "learning curve in a table")
    icon = "icons/LearningCurve.svg"
    priority = 1000

# [start-snippet-1]
    class Inputs:
        data = Input("Data", Orange.data.Table)
        learner = Input("Learner", Orange.classification.Learner,
                        multiple=True)
# [end-snippet-1]

    #: cross validation folds
    folds = settings.Setting(5)
    #: points in the learning curve
    steps = settings.Setting(10)
    #: index of the selected scoring function
    scoringF = settings.Setting(0)
    #: compute curve on any change of parameters
    commitOnChange = settings.Setting(True)

    def __init__(self):
        super().__init__()

        # sets self.curvePoints, self.steps equidistant points from
        # 1/self.steps to 1
        self.updateCurvePoints()

# [start-snippet-2]
        self.scoring = [
            ("Classification Accuracy", Orange.evaluation.scoring.CA),
            ("AUC", Orange.evaluation.scoring.AUC),
            ("Precision", Orange.evaluation.scoring.Precision),
            ("Recall", Orange.evaluation.scoring.Recall)
        ]
# [end-snippet-2]
        #: input data on which to construct the learning curve
        self.data = None
        #: A {input_id: Learner} mapping of current learners from input channel
        self.learners = OrderedDict()
        #: A {input_id: List[Results]} mapping of input id to evaluation
        #: results list, one for each curve point
        self.results = OrderedDict()
        #: A {input_id: List[float]} mapping of input id to learning curve
        #: point scores
        self.curves = OrderedDict()

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.infoa = gui.widgetLabel(box, 'No data on input.')
        self.infob = gui.widgetLabel(box, 'No learners.')

        gui.separator(self.controlArea)

        box = gui.widgetBox(self.controlArea, "Evaluation Scores")
        gui.comboBox(box, self, "scoringF",
                     items=[x[0] for x in self.scoring],
                     callback=self._invalidate_curves)

        gui.separator(self.controlArea)

        box = gui.widgetBox(self.controlArea, "Options")
        gui.spin(box, self, 'folds', 2, 100, step=1,
                 label='Cross validation folds:  ', keyboardTracking=False,
                 callback=lambda:
                    self._invalidate_results() if self.commitOnChange else None
        )
        gui.spin(box, self, 'steps', 2, 100, step=1,
                 label='Learning curve points:  ', keyboardTracking=False,
                 callback=[self.updateCurvePoints,
                           lambda: self._invalidate_results() if self.commitOnChange else None])
        gui.checkBox(box, self, 'commitOnChange', 'Apply setting on any change')
        self.commitBtn = gui.button(box, self, "Apply Setting",
                                    callback=self._invalidate_results,
                                    disabled=True)

        gui.rubber(self.controlArea)

        # table widget
        self.table = gui.table(self.mainArea,
                               selectionMode=QTableWidget.NoSelection)

    ##########################################################################
    # slots: handle input signals

    @Inputs.data
    def set_dataset(self, data):
        """Set the input train dataset."""
        # Clear all results/scores
        for id in list(self.results):
            self.results[id] = None
        for id in list(self.curves):
            self.curves[id] = None

        self.data = data

        if data is not None:
            self.infoa.setText('%d instances in input dataset' % len(data))
        else:
            self.infoa.setText('No data on input.')

        self.commitBtn.setEnabled(self.data is not None)

    @Inputs.learner
    def set_learner(self, learner, id):
        """Set the input learner for channel id."""
        if id in self.learners:
            if learner is None:
                # remove a learner and corresponding results
                del self.learners[id]
                del self.results[id]
                del self.curves[id]
            else:
                # update/replace a learner on a previously connected link
                self.learners[id] = learner
                # invalidate the cross-validation results and curve scores
                # (will be computed/updated in `_update`)
                self.results[id] = None
                self.curves[id] = None
        else:
            if learner is not None:
                self.learners[id] = learner
                # initialize the cross-validation results and curve scores
                # (will be computed/updated in `_update`)
                self.results[id] = None
                self.curves[id] = None

        if len(self.learners):
            self.infob.setText("%d learners on input." % len(self.learners))
        else:
            self.infob.setText("No learners.")

        self.commitBtn.setEnabled(len(self.learners))

    def handleNewSignals(self):
        if self.data is not None:
            self._update()
            self._update_curve_points()
        self._update_table()

    def _invalidate_curves(self):
        if self.data is not None:
            self._update_curve_points()
        self._update_table()

    def _invalidate_results(self):
        for id in self.learners:
            self.curves[id] = None
            self.results[id] = None

        if self.data is not None:
            self._update()
            self._update_curve_points()
        self._update_table()

    def _update(self):
        assert self.data is not None
        # collect all learners for which results have not yet been computed
        need_update = [(id, learner) for id, learner in self.learners.items()
                       if self.results[id] is None]
        if not need_update:
            return

        learners = [learner for _, learner in need_update]

        # compute the learning curve result for all learners in one go
        results = learning_curve(
            learners, self.data, folds=self.folds,
            proportions=self.curvePoints,
        )
        # split the combined result into per learner/model results
        results = [list(Results.split_by_model(p_results))
                   for p_results in results]

        for i, (id, learner) in enumerate(need_update):
            self.results[id] = [p_results[i] for p_results in results]

    def _update_curve_points(self):
        for id in self.learners:
            curve = [self.scoring[self.scoringF][1](x)[0]
                     for x in self.results[id]]
            self.curves[id] = curve

    def _update_table(self):
        self.table.setRowCount(0)
        self.table.setRowCount(len(self.curvePoints))
        self.table.setColumnCount(len(self.learners))

        self.table.setHorizontalHeaderLabels(
            [learner.name for _, learner in self.learners.items()])
        self.table.setVerticalHeaderLabels(
            ["{:.2f}".format(p) for p in self.curvePoints])

        if self.data is None:
            return

        for column, curve in enumerate(self.curves.values()):
            for row, point in enumerate(curve):
                self.table.setItem(
                    row, column, QTableWidgetItem("{:.5f}".format(point)))

        for i in range(len(self.learners)):
            sh = self.table.sizeHintForColumn(i)
            cwidth = self.table.columnWidth(i)
            self.table.setColumnWidth(i, max(sh, cwidth))

    def updateCurvePoints(self):
        self.curvePoints = [(x + 1.)/self.steps for x in range(self.steps)]


def learning_curve(learners, data, folds=10, proportions=None,
                   random_state=None, callback=None):

    if proportions is None:
        proportions = numpy.linspace(0.0, 1.0, 10 + 1, endpoint=True)[1:]

    def select_proportion_preproc(data, p, rstate=None):
        assert 0 < p <= 1
        rstate = numpy.random.RandomState(None) if rstate is None else rstate
        indices = rstate.permutation(len(data))
        n = int(numpy.ceil(len(data) * p))
        return data[indices[:n]]

    if callback is not None:
        parts_count = len(proportions)
        callback_wrapped = lambda part: \
            lambda value: callback(value / parts_count + part / parts_count)
    else:
        callback_wrapped = lambda part: None

    results = [
        Orange.evaluation.CrossValidation(
            data, learners, k=folds,
            preprocessor=lambda data, p=p:
                select_proportion_preproc(data, p),
            callback=callback_wrapped(i)
        )
        for i, p in enumerate(proportions)
    ]
    return results


def main(argv=None):
    from AnyQt.QtWidgets import QApplication
    app = QApplication(list(argv) if argv else [])
    argv = app.arguments()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "iris"

    data = Orange.data.Table(filename)

    ow = OWLearningCurveA()
    ow.show()
    ow.raise_()

    l1 = Orange.classification.NaiveBayesLearner()
    l1.name = 'Naive Bayes'
    ow.set_learner(l1, 1)
    ow.set_dataset(data)

    l2 = Orange.classification.LogisticRegressionLearner()
    l2.name = 'Logistic Regression'
    ow.set_learner(l2, 2)

    l4 = Orange.classification.SklTreeLearner()
    l4.name = "Decision Tree"
    ow.set_learner(l4, 3)

    ow.handleNewSignals()

    app.exec_()

    ow.set_dataset(None)

    ow.set_learner(None, 1)
    ow.set_learner(None, 2)
    ow.set_learner(None, 3)
    ow.handleNewSignals()
    ow.onDeleteWidget()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
