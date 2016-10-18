from collections import OrderedDict

from PyQt4 import QtGui
from PyQt4.QtGui import QLabel
from PyQt4.QtCore import Qt

from Orange.data import Table
from Orange.regression import SVRLearner, NuSVRLearner
from Orange.widgets import widget, settings, gui
from Orange.widgets.classify.owsvmclassification import OWBaseSVM


class OWSVMRegression(OWBaseSVM):
    name = "SVM Regression"
    description = "Support Vector Machines map inputs to higher-dimensional " \
                  "feature spaces that best map instances to a linear function.  "
    icon = "icons/SVMRegression.svg"
    priority = 50

    LEARNER = SVRLearner

    outputs = [("Support vectors", Table, widget.Explicit)]

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

    def _add_type_box(self):
        form = QtGui.QGridLayout()
        self.type_box = box = gui.radioButtonsInBox(
                self.controlArea, self, "svrtype", [], box="SVR Type",
                orientation=form)

        self.epsilon_radio = gui.appendRadioButton(box, "ε-SVR",
                                                   addToLayout=False)
        self.epsilon_C_spin = gui.doubleSpin(box, self, "epsilon_C", 0.1, 512.0,
                                             0.1, decimals=2, addToLayout=False)
        self.epsilon_spin = gui.doubleSpin(box, self, "epsilon", 0.1, 512.0,
                                           0.1, decimals=2, addToLayout=False)
        form.addWidget(self.epsilon_radio, 0, 0, Qt.AlignLeft)
        form.addWidget(QtGui.QLabel("Cost (C):"), 0, 1, Qt.AlignRight)
        form.addWidget(self.epsilon_C_spin, 0, 2)
        form.addWidget(QLabel("Loss epsilon (ε):"), 1, 1, Qt.AlignRight)
        form.addWidget(self.epsilon_spin, 1, 2)

        self.nu_radio = gui.appendRadioButton(box, "ν-SVR", addToLayout=False)
        self.nu_C_spin = gui.doubleSpin(box, self, "nu_C", 0.1, 512.0, 0.1,
                                        decimals=2, addToLayout=False)
        self.nu_spin = gui.doubleSpin(box, self, "nu", 0.05, 1.0, 0.05,
                                      decimals=2, addToLayout=False)
        form.addWidget(self.nu_radio, 2, 0, Qt.AlignLeft)
        form.addWidget(QLabel("Cost (C):"), 2, 1, Qt.AlignRight)
        form.addWidget(self.nu_C_spin, 2, 2)
        form.addWidget(QLabel("Complexity bound (ν):"), 3, 1, Qt.AlignRight)
        form.addWidget(self.nu_spin, 3, 2)

    def create_learner(self):
        kernel = ["linear", "poly", "rbf", "sigmoid"][self.kernel_type]
        common_args = dict(
            kernel=kernel,
            degree=self.degree,
            gamma=self.gamma if self.gamma else self._default_gamma,
            coef0=self.coef0,
            tol=self.tol,
            preprocessors=self.preprocessors
        )
        if self.svrtype == OWSVMRegression.Epsilon_SVR:
            return SVRLearner(
                C=self.epsilon_C, epsilon=self.epsilon, **common_args
            )
        else:
            return NuSVRLearner(C=self.nu_C, nu=self.nu, **common_args)

    def get_learner_parameters(self):
        items = OrderedDict()
        if self.svrtype == 0:
            items["SVM type"] = \
                "ε-SVR, C={}, ε={}".format(self.epsilon_C, self.epsilon)
        else:
            items["SVM type"] = "ν-SVR, C={}, ν={}".format(self.nu_C, self.nu)
        self._report_kernel_parameters(items)
        items["Numerical tolerance"] = "{:.6}".format(self.tol)
        return items


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication

    a = QApplication(sys.argv)
    ow = OWSVMRegression()
    d = Table('housing')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
