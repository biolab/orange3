# -*- coding: utf-8 -*-

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt

from Orange.data import Table
from Orange.classification.svm import SVMLearner, SVMClassifier, NuSVMLearner
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets import widget, settings, gui


class OWSVMClassification(widget.OWWidget):
    name = "SVM"
    description = "Support vector machines classifier with standard " \
                  "selection of kernels."
    icon = "icons/SVM.svg"

    inputs = [("Data", Table, "set_data"),
              ("Preprocessor", Preprocess, "set_preprocessor")]
    outputs = [("Learner", SVMLearner),
               ("Classifier", SVMClassifier),
               ("Support vectors", Table)]

    want_main_area = False
    resizing_enabled = False

    learner_name = settings.Setting("SVM Learner")

    # 0: c_svc, 1: nu_svc
    svmtype = settings.Setting(0)
    C = settings.Setting(1.0)
    nu = settings.Setting(0.5)
    # 0: Linear, 1: Poly, 2: RBF, 3: Sigmoid
    kernel_type = settings.Setting(0)
    degree = settings.Setting(3)
    gamma = settings.Setting(0.0)
    coef0 = settings.Setting(0.0)
    shrinking = settings.Setting(True),
    probability = settings.Setting(False)
    tol = settings.Setting(0.001)
    max_iter = settings.Setting(100)
    limit_iter = settings.Setting(True)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None
        self.preprocessors = None

        box = gui.widgetBox(self.controlArea, self.tr("Name"))
        gui.lineEdit(box, self, "learner_name")

        form = QtGui.QGridLayout()
        typebox = gui.radioButtonsInBox(
            self.controlArea, self, "svmtype", [],
            box=self.tr("SVM Type"),
            orientation=form,
        )

        c_svm = gui.appendRadioButton(typebox, "C-SVM", addToLayout=False)
        form.addWidget(c_svm, 0, 0, Qt.AlignLeft)
        form.addWidget(QtGui.QLabel(self.tr("Cost (C)")), 0, 1, Qt.AlignRight)
        c_spin = gui.doubleSpin(
            typebox, self, "C", 1e-3, 1000.0, 0.1,
            decimals=3, addToLayout=False
        )

        form.addWidget(c_spin, 0, 2)

        nu_svm = gui.appendRadioButton(typebox, "ν-SVM", addToLayout=False)
        form.addWidget(nu_svm, 1, 0, Qt.AlignLeft)

        form.addWidget(
            QtGui.QLabel(self.trUtf8("Complexity bound (\u03bd)")),
            1, 1, Qt.AlignRight
        )

        nu_spin = gui.doubleSpin(
            typebox, self, "nu", 0.05, 1.0, 0.05,
            decimals=2, addToLayout=False
        )
        form.addWidget(nu_spin, 1, 2)

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
        box = gui.widgetBox(self.controlArea, "Optimization parameters")
        gui.doubleSpin(box, self, "tol", 1e-7, 1.0, 5e-7,
        label="Numerical Tolerance")
        gui.spin(box, self, "max_iter", 0, 1e6, 100,
        label="Iteration Limit", checked="limit_iter")

        gui.button(self.controlArea, self, "&Apply",
                   callback=self.apply, default=True)

        self._on_kernel_changed()

        self.apply()

    def set_data(self, data):
        """Set the input train data set."""
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
        kernel = ["linear", "poly", "rbf", "sigmoid"][self.kernel_type]
        common_args = dict(
            kernel=kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            tol=self.tol,
            max_iter=self.max_iter if self.limit_iter else -1,
            probability=True,
            preprocessors=self.preprocessors
        )
        if self.svmtype == 0:
            learner = SVMLearner(C=self.C, **common_args)
        else:
            learner = NuSVMLearner(nu=self.nu, **common_args)
        learner.name = self.learner_name
        classifier = None
        sv = None
        if self.data is not None:
            self.error(0)
            if not learner.check_learner_adequacy(self.data.domain):
                self.error(0, learner.learner_adequacy_err_msg)
            else:
                classifier = learner(self.data)
                classifier.name = self.learner_name
                sv = self.data[classifier.skl_model.support_]

        self.send("Learner", learner)
        self.send("Classifier", classifier)
        self.send("Support vectors", sv)

    def _on_kernel_changed(self):
        enabled = [[False, False, False],  # linear
                   [True, True, True],     # poly
                   [True, False, False],   # rbf
                   [True, True, False]]    # sigmoid

        mask = enabled[self.kernel_type]
        for spin, enabled in zip(self._kernel_params, mask):
            spin.setEnabled(enabled)


if __name__ == "__main__":
    app = QtGui.QApplication([])
    w = OWSVMClassification()
    w.set_data(Table("iris"))
    w.show()
    app.exec_()
