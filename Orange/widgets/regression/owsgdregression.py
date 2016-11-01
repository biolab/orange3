# -*- coding: utf-8 -*-
from collections import OrderedDict

from AnyQt.QtWidgets import QFormLayout
from AnyQt.QtCore import  Qt


from Orange.data import Table
from Orange.regression.linear import SGDRegressionLearner
from Orange.widgets import settings, gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWSGDRegression(OWBaseLearner):
    name = "Stochastic Gradient Descent"
    description = "Minimize an objective function using a stochastic " \
                  "approximation of gradient descent. "
    icon = "icons/SGDRegression.svg"
    priority = 90
    LEARNER = SGDRegressionLearner

    learner_name = settings.Setting("SGD Regression")

    alpha = settings.Setting(0.0001)
    epsilon = settings.Setting(0.1)
    eta0 = settings.Setting(0.01)
    l1_ratio = settings.Setting(0.15)
    power_t = settings.Setting(0.25)
    n_iter = settings.Setting(5)
    SqLoss, Huber, Epsilon_i, SqEpsilon_i = 0, 1, 2, 3
    L1, L2, ElasticNet = 0, 1, 2
    loss_function = settings.Setting(SqLoss)
    penalty_type = settings.Setting(L2)
    Constant, InvScaling = 0, 1
    learning_rate = settings.Setting(InvScaling)

    LOSS_FUNCTIONS = ["Squared loss",
                      "Huber",
                      "Epsilon insensitive",
                      "Squared epsilon insensitive"]
    PENALTIES = ["Absolute norm (L1)",
                 "Euclidean norm (L2)",
                 "Elastic net (L1 and L2)"]
    LEARNING_RATES = ["Constant", "Inverse scaling"]

    def setup_layout(self):
        super().setup_layout()

        # enable/disable parameters based on selected loss/penalty/rate
        self._on_func_changed()
        self._on_penalty_changed()
        self._on_lrate_changed()

    def add_main_layout(self):
        def add_form(box):
            gui.separator(box)
            box2 = gui.hBox(box)
            gui.rubber(box2)
            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            box2.layout().addLayout(form)
            return form

        box = gui.radioButtons(
            self.controlArea, self, "loss_function", box="Loss Function",
            btnLabels=self.LOSS_FUNCTIONS, callback=self._on_func_changed,
            orientation=Qt.Vertical)
        form = add_form(box)
        epsilon = gui.doubleSpin(
            box, self, "epsilon", 0.0, 10.0, 0.01, controlWidth=70)
        form.addRow("ε:", epsilon)
        self._func_params = [epsilon]

        box = gui.radioButtons(
            self.controlArea, self, "penalty_type", box="Penalty",
            btnLabels=self.PENALTIES, callback=self._on_penalty_changed,
            orientation=Qt.Vertical)
        form = add_form(box)
        alpha = gui.doubleSpin(
            box, self, "alpha", 0.0, 10.0, 0.0001, controlWidth=80)
        form.addRow("α:", alpha)
        l1_ratio = gui.doubleSpin(
            box, self, "l1_ratio", 0.0, 10.0, 0.01, controlWidth=80)
        form.addRow("L1 ratio:", l1_ratio)
        self._penalty_params = [l1_ratio]

        box = gui.radioButtons(
            self.controlArea, self, "learning_rate", box="Learning Rate",
            btnLabels=self.LEARNING_RATES, callback=self._on_lrate_changed,
            orientation=Qt.Vertical)
        form = add_form(box)
        spin = gui.doubleSpin(
            box, self, "eta0", 0.0, 10, 0.01, controlWidth=70)
        form.addRow("η<sub>0</sub>:", spin)
        power_t = gui.doubleSpin(
            box, self, "power_t", 0.0, 10.0, 0.01, controlWidth=70)
        form.addRow("Power t:", power_t)
        gui.separator(box, height=8)
        niterations = gui.doubleSpin(
            box, self, "n_iter", 1, 1e+6, 1, controlWidth=70)
        form.addRow("Number of iterations:", niterations)
        self._lrate_params = [power_t]

    def create_learner(self):
        loss = ["squared_loss", "huber", "epsilon_insensitive",
                "squared_epsilon_insensitive"][self.loss_function]
        penalty = ["l1", "l2", "elasticnet"][self.penalty_type]
        learning_rate = ["constant", "invscaling"][self.learning_rate]
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
        return self.LEARNER(preprocessors=self.preprocessors, **common_args)

    def _on_func_changed(self):
        enabled = [[False],  # squared loss
                   [True],   # huber
                   [True],   # epsilon insensitive
                   [True]]   # squared epsilon insensitive

        mask = enabled[self.loss_function]
        for spin, enabled in zip(self._func_params, mask):
            spin.setEnabled(enabled)
        self.settings_changed()

    def _on_penalty_changed(self):
        enabled = [[False],  # l1
                   [False],  # l2
                   [True]]   # elasticnet

        mask = enabled[self.penalty_type]
        for spin, enabled in zip(self._penalty_params, mask):
            spin.setEnabled(enabled)
        self.settings_changed()

    def _on_lrate_changed(self):
        enabled = [[True],   # invscaling
                   [False]]  # constant

        mask = enabled[self.learning_rate]
        for spin, enabled in zip(self._lrate_params, mask):
            spin.setEnabled(enabled)
        self.settings_changed()

    def get_learner_parameters(self):
        items = OrderedDict()
        items['Loss function'] = self.LOSS_FUNCTIONS[self.loss_function]
        if self.loss_function != self.SqLoss:
            items['Loss function'] += ", ε={}".format(self.epsilon)
        items['Penalty'] = self.PENALTIES[self.penalty_type]
        if self.penalty_type == self.ElasticNet:
            items['Penalty'] += ": L1 : L2 = {} : {}".format(
                                self.l1_ratio, 1.0 - self.l1_ratio)
        items['Penalty'] = items['Penalty'] + ', α={}'.format(self.alpha)
        items['Learning rate'] = self.LEARNING_RATES[self.learning_rate]
        items['Learning rate'] += ", η<sub>0</sub>={}".format(self.eta0)
        if self.learning_rate == self.InvScaling:
            items['Learning rate'] += ", power_t={}".format(self.power_t)
        items['Number of iterations'] = self.n_iter
        return items


if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    ow = OWSGDRegression()
    d = Table('housing')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
