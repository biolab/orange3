# -*- coding: utf-8 -*-

from PyQt4 import QtGui
from PyQt4.QtGui import QGridLayout, QLabel
from PyQt4.QtCore import Qt

import Orange.data
from Orange.classification import svm, SklModel
from Orange.widgets import widget, settings, gui


class OWSVMRegression(widget.OWWidget):
    name = "SVM Regression"
    description = "Support Vector Machine Regression."
    icon = "icons/SVMRegression.svg"

    inputs = [{"name": "Data",
               "type": Orange.data.Table,
               "handler": "set_data"}]
    outputs = [{"name": "Learner",
                "type": svm.SVRLearner},
               {"name": "Predictor",
                "type": SklModel}]

    learner_name = settings.Setting("SVM Regression")

    #: SVR types
    Epsilon_SVR, Nu_SVR = 0, 1
    #: Selected SVR type
    svrtype = settings.Setting(Epsilon_SVR)
    #: C parameter for Epsilon SVR
    epsilon_C = settings.Setting(1.0)
    #: epsilon parameter for Epsilon SVR
    epsilon = settings.Setting(0.1)
    #: C parameter for Nu SVR
    nu_C = settings.Setting(1.0)
    #: Nu pareter for Nu SVR
    nu = settings.Setting(0.5)

    #: Kernel types
    Linear, Poly, RBF, Sigmoid = 0, 1, 2, 3
    #: Selected kernel type
    kernel_type = settings.Setting(RBF)
    #: kernel degree
    degree = settings.Setting(3)
    #: gamma
    gamma = settings.Setting(0.0)
    #: coef0 (adative constant)
    coef0 = settings.Setting(0.0)

    #: numerical tolerance
    tol = settings.Setting(0.001)

    want_main_area = False

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None

        box = gui.widgetBox(self.controlArea, self.tr("Name"))
        gui.lineEdit(box, self, "learner_name")

        form = QGridLayout()
        typebox = gui.radioButtonsInBox(
            self.controlArea, self, "svrtype", [],
            box=self.tr("SVM Regression Type"),
            orientation=form,
        )

        eps_svr = gui.appendRadioButton(typebox, "ε-SVR", addToLayout=False)
        form.addWidget(eps_svr, 0, 0, Qt.AlignLeft)

        form.addWidget(QtGui.QLabel(self.tr("Cost (C)")), 0, 1, Qt.AlignRight)
        c_spin = gui.doubleSpin(
            typebox, self, "epsilon_C", 0.1, 512.0, 0.1,
            decimals=2, addToLayout=False
        )
        form.addWidget(c_spin, 0, 2)

        form.addWidget(QLabel("Loss Epsilon (ε)"), 1, 1, Qt.AlignRight)
        eps_spin = gui.doubleSpin(
            typebox, self, "epsilon",  0.1, 512.0, 0.1,
            decimals=2, addToLayout=False
        )
        form.addWidget(eps_spin, 1, 2)

        nu_svr = gui.appendRadioButton(typebox, "ν-SVR", addToLayout=False)
        form.addWidget(nu_svr, 2, 0, Qt.AlignLeft)

        form.addWidget(QLabel(self.tr("Cost (C)")), 2, 1, Qt.AlignRight)
        c_spin = gui.doubleSpin(
            typebox, self, "nu_C", 0.1, 512.0, 0.1,
            decimals=2, addToLayout=False
        )
        form.addWidget(c_spin, 2, 2)

        form.addWidget(QLabel("Complexity bound (ν)"),
                       3, 1, Qt.AlignRight)
        nu_spin = gui.doubleSpin(
            typebox, self, "nu", 0.05, 1.0, 0.05,
            decimals=2, addToLayout=False
        )
        form.addWidget(nu_spin, 3, 2)

        # Kernel control
        box = gui.widgetBox(self.controlArea, self.tr("Kernel"))
        buttonbox = gui.radioButtonsInBox(
            box, self, "kernel_type",
            btnLabels=["Linear,   x∙y",
                       "Polynomial,   (g x∙y + c)^d",
                       "RBF,   exp(-g|x-y|²)",
                       "Sigmoid,   tanh(g x∙y + c)"],
            callback=self._on_kernel_changed
        )
        parambox = gui.widgetBox(box, orientation="horizontal")
        gamma = gui.doubleSpin(
            parambox, self, "gamma", 0.0, 10.0, 0.0001,
            label=" g: ", orientation="horizontal",
            alignment=Qt.AlignRight
        )
        coef0 = gui.doubleSpin(
            parambox, self, "coef0", 0.0, 10.0, 0.0001,
            label=" c: ", orientation="horizontal",
            alignment=Qt.AlignRight
        )
        degree = gui.doubleSpin(
            parambox, self, "degree", 0.0, 10.0, 0.5,
            label=" d: ", orientation="horizontal",
            alignment=Qt.AlignRight
        )
        self._kernel_params = [gamma, coef0, degree]

        # Numerical tolerance control
        box = gui.widgetBox(self.controlArea, "Numerical Tolerance")
        gui.doubleSpin(box, self, "tol", 1e-7, 1e-3, 5e-7)

        gui.button(self.controlArea, self, "&Apply",
                   callback=self.apply, default=True)

        self.setSizePolicy(
            QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed,
                              QtGui.QSizePolicy.Fixed)
        )

        self.setMinimumWidth(300)

        self._on_kernel_changed()

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

    def apply(self):
        kernel = ["linear", "poly", "rbf", "sigmoid"][self.kernel_type]
        common_args = dict(
            kernel=kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            tol=self.tol,
        )
        if self.svrtype == OWSVMRegression.Epsilon_SVR:
            learner = svm.SVRLearner(
                C=self.epsilon_C, epsilon=self.epsilon, **common_args
            )
        else:
            learner = svm.NuSVRLearner(C=self.nu_C, nu=self.nu, **common_args)
        learner.name = self.learner_name

        predictor = None
        if self.data is not None:
            predictor = learner(self.data)
            predictor.name = self.learner_name

        self.send("Learner", learner)
        self.send("Predictor", predictor)

    def _on_kernel_changed(self):
        enabled = [[False, False, False],  # linear
                   [True, True, True],     # poly
                   [True, False, False],   # rbf
                   [True, True, False]]    # sigmoid

        mask = enabled[self.kernel_type]
        for spin, enabled in zip(self._kernel_params, mask):
            spin.setEnabled(enabled)


def main():
    app = QtGui.QApplication([])
    w = OWSVMRegression()
    w.set_data(Orange.data.Table("housing"))
    w.show()
    return app.exec_()


if __name__ == "__main__":
    import sys
    sys.exit(main())
