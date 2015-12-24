from collections import OrderedDict

from PyQt4 import QtGui
from PyQt4.QtGui import QLabel
from PyQt4.QtCore import Qt

from Orange.data import Table
from Orange.regression import SVRLearner, NuSVRLearner, SklModel
from Orange.widgets import widget, settings, gui
from Orange.widgets.utils.owlearnerwidget import OWProvidesLearner
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.classify.owsvmclassification import SVMBaseMixin


class OWSVMRegression(SVMBaseMixin, widget.OWWidget):
    name = "SVM Regression"
    description = "Support vector machine regression algorithm."
    icon = "icons/SVMRegression.svg"

    LEARNER = SVRLearner

    inputs = [("Data", Table, "set_data")] + OWProvidesLearner.inputs
    outputs = [("Learner", LEARNER, widget.Default),
               ("Predictor", SklModel),
               ("Support vectors", Table)]

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

    def __init__(self):
        super().__init__()
        self.data = None
        self.preprocessors = None

        self._setup_layout()
        self._on_kernel_changed()
        self.apply()

    def _add_type_box(self):
        form = QtGui.QGridLayout()
        self.type_box = box = gui.radioButtonsInBox(
                self.controlArea, self, "svrtype", [], box="SVR Type",
                orientation=form)

        form.addWidget(gui.appendRadioButton(box, "ε-SVR", addToLayout=False),
                       0, 0, Qt.AlignLeft)
        form.addWidget(QtGui.QLabel("Cost (C)"),
                       0, 1, Qt.AlignRight)
        form.addWidget(gui.doubleSpin(box, self, "epsilon_C", 0.1, 512.0, 0.1,
                                      decimals=2, addToLayout=False),
                       0, 2)
        form.addWidget(QLabel("Loss Epsilon (ε)"),
                       1, 1, Qt.AlignRight)
        form.addWidget(gui.doubleSpin(box, self, "epsilon", 0.1, 512.0, 0.1,
                                      decimals=2, addToLayout=False),
                       1, 2)

        form.addWidget(gui.appendRadioButton(box, "ν-SVR", addToLayout=False),
                       2, 0, Qt.AlignLeft)
        form.addWidget(QLabel("Cost (C)"),
                       2, 1, Qt.AlignRight)
        form.addWidget(gui.doubleSpin(box, self, "nu_C", 0.1, 512.0, 0.1,
                                      decimals=2, addToLayout=False),
                       2, 2)
        form.addWidget(QLabel("Complexity bound (ν)"),
                       3, 1, Qt.AlignRight)
        form.addWidget(gui.doubleSpin(box, self, "nu", 0.05, 1.0, 0.05,
                                      decimals=2, addToLayout=False),
                       3, 2)

    @check_sql_input
    def set_data(self, data):
        """Set the input train data set."""
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
            preprocessors=self.preprocessors
        )
        if self.svrtype == OWSVMRegression.Epsilon_SVR:
            learner = SVRLearner(
                C=self.epsilon_C, epsilon=self.epsilon, **common_args
            )
        else:
            learner = NuSVRLearner(C=self.nu_C, nu=self.nu, **common_args)
        learner.name = self.learner_name
        predictor = None

        sv = None
        if self.data is not None:
            self.error(0)
            if not learner.check_learner_adequacy(self.data.domain):
                self.error(0, learner.learner_adequacy_err_msg)
            else:
                predictor = learner(self.data)
                predictor.name = self.learner_name
                sv = self.data[predictor.skl_model.support_]

        self.send("Learner", learner)
        self.send("Predictor", predictor)
        self.send("Support vectors", sv)

    def send_report(self):
        self.report_items((("Name", self.learner_name),))

        items = OrderedDict()
        if self.svrtype == 0:
            items["SVM type"] = \
                "ε-SVR, C={}, ε={}".format(self.epsilon_C, self.epsilon)
        else:
            items["SVM type"] = "ν-SVR, C={}, ν={}".format(self.nu_C, self.nu)
        self._report_kernel_parameters(items)
        items["Numerical tolerance"] = "{:.6}".format(self.tol)
        self.report_items("Model parameters", items)

        if self.data:
            self.report_data("Data", self.data)


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
