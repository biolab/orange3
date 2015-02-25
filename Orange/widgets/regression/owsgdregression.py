# -*- coding: utf-8 -*-

from PyQt4 import QtGui
from PyQt4.QtGui import QGridLayout, QLabel
from PyQt4.QtCore import Qt

import Orange.data
from Orange.regression import linear
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets import widget, settings, gui


class OWSGDRegression(widget.OWWidget):
    name = "Stochastic Gradient Descent"
    description = "Stochastic Gradient Descent Regression."
    icon = "icons/SGDRegression.svg"

    inputs = [{"name": "Data",
               "type": Orange.data.Table,
               "handler": "set_data"},
              {"name": "Preprocessor",
               "type": Preprocess,
               "handler": "set_preprocessor"}]
    outputs = [{"name": "Learner",
                "type": linear.SGDRegressionLearner},
               {"name": "Predictor",
                "type": linear.LinearModel}]

    learner_name = settings.Setting("SGD Regression")

    alpha = settings.Setting(0.0001)
    #: epsilon parameter for Epsilon SVR
    epsilon = settings.Setting(0.1)
    eta0 = settings.Setting(0.01)
    l1_ratio = settings.Setting(0.15)
    power_t = settings.Setting(0.25)
    n_iter = settings.Setting(5)

    #: Loss of function to be used
    SqLoss, Huber, Epsilon_i, SqEpsilon_i = 0, 1, 2, 3
    L1, L2, ElasticNet = 0, 1, 2
    #: Selected loss of function
    loss_function = settings.Setting(SqLoss)
    penalty_type = settings.Setting(L2)
    InvScaling, Constant = 0, 1
    learning_rate = settings.Setting(InvScaling)

    want_main_area = False

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None
        self.preprocessors = None

        box = gui.widgetBox(self.controlArea, self.tr("Name"))
        gui.lineEdit(box, self, "learner_name")

        form = QGridLayout()
        typebox = gui.radioButtonsInBox(
            self.controlArea, self, "lossfunc", [],
            orientation=form,
        )

        # Loss function control
        box = gui.widgetBox(self.controlArea, self.tr("Loss function to be used"))
        buttonbox = gui.radioButtonsInBox(
            box, self, "loss_function",
            btnLabels=["Squared loss",
                       "Huber",
                       "Epsilon insensitive",
                       "Squared Epsilon insensitive"],
            callback=self._on_func_changed
        )

        parambox = gui.widgetBox(box, orientation="horizontal")

        box = gui.widgetBox(self.controlArea, self.tr("Penalty"))
        buttonbox = gui.radioButtonsInBox(
            box, self, "penalty_type",
            btnLabels=["Absolute norm (L1)",
                       "Euclidean norm (L2)",
                       "Elastic Net (both)"],
            callback=self._on_penalty_changed
        )

        parambox = gui.widgetBox(box, orientation="horizontal")

        box = gui.widgetBox(self.controlArea, self.tr("Learning rate"))
        buttonbox = gui.radioButtonsInBox(
            box, self, "learning_rate",
            btnLabels=["Inverse scaling",
                       "Constant"],
            callback=self._on_lrate_changed
        )



        box = gui.widgetBox(self.controlArea, self.tr("Constants"))

        form = QtGui.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)

        box.layout().addLayout(form)

        alpha = gui.doubleSpin(box, self, "alpha", 0.0, 10.0, step=0.0001)
        form.addRow("Alpha:", alpha)

        spin = gui.doubleSpin(box, self, "eta0", 0.0, 10, step=0.01)
        form.addRow("Eta0:", spin)

        epsilon = gui.doubleSpin(box, self, "epsilon", 0.0, 10.0, step=0.01)
        form.addRow("Epsilon:", epsilon)

        l1_ratio = gui.doubleSpin(box, self, "l1_ratio", 0.0, 10.0, step=0.01)
        form.addRow("L1 ratio:", l1_ratio)

        power_t = gui.doubleSpin(box, self, "power_t", 0.0, 10.0, step=0.01)
        form.addRow("Power t:", power_t)


        # Number of iterations control
        box = gui.widgetBox(self.controlArea, "Number of iterations")
        gui.doubleSpin(box, self, "n_iter", 0, 1e+6, step=1)

        self._func_params = [epsilon]
        self._penalty_params = [l1_ratio]
        self._lrate_params = [power_t]

        gui.button(self.controlArea, self, "&Apply",
                   callback=self.apply, default=True)

        self.setSizePolicy(
            QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed,
                              QtGui.QSizePolicy.Fixed)
        )

        self.setMinimumWidth(300)

        self._on_func_changed()

        self.apply()

    def set_data(self, data):
        """Set the input train data set."""
        self.warning(0)

        if data is not None:
            if not isinstance(data.domain.class_var,
                              Orange.data.ContinuousVariable):
                data = None
                self.warning(0, "Data does not have a continuous class var")

        self.data = data
        if data is not None:
            self.apply()

    def set_preprocessor(self, preproc):
        if preproc is None:
            self.preprocessors = None
        else:
            self.preprocessors = (preproc,)
        self.apply()

    def apply(self):
        loss = ["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"][self.loss_function]
        penalty = ["l1", "l2", "elasticnet"][self.penalty_type]
        learning_rate = ["invscaling", "constant"][self.learning_rate]
        common_args = dict(
            loss=loss,
            alpha=self.alpha,
            epsilon=self.epsilon,
            eta0=self.eta0,
            l1_ratio=self.l1_ratio,
            power_t=self.power_t,
            penalty=penalty,
            learning_rate=learning_rate,
            n_iter=self.n_iter,
        )

        learner = linear.SGDRegressionLearner(
            preprocessors=self.preprocessors, **common_args)
        learner.name = self.learner_name

        predictor = None
        if self.data is not None:
            predictor = learner(self.data)
            predictor.name = self.learner_name

        self.send("Learner", learner)
        self.send("Predictor", predictor)

    def _on_func_changed(self):
        enabled = [[False],  # squared loss
                   [True],   # huber
                   [True],   # epsilon insensitive
                   [True]]   # squared epsilon insensitive

        mask = enabled[self.loss_function]
        for spin, enabled in zip(self._func_params, mask):
            spin.setEnabled(enabled)

    def _on_penalty_changed(self):
        enabled = [[False],  # l1
                   [False],  # l2
                   [True]]   # elasticnet

        mask = enabled[self.penalty_type]
        for spin, enabled in zip(self._penalty_params, mask):
            spin.setEnabled(enabled)

    def _on_lrate_changed(self):
        enabled = [[True],  # invscaling
                   [False]]     # constant

        mask = enabled[self.learning_rate]
        for spin, enabled in zip(self._lrate_params, mask):
            spin.setEnabled(enabled)



def main():
    app = QtGui.QApplication([])
    w = OWSGDRegression()
    w.set_data(Orange.data.Table("housing"))
    w.show()
    return app.exec_()


if __name__ == "__main__":
    import sys
    sys.exit(main())
