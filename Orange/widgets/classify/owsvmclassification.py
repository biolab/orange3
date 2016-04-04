import numpy as np
from collections import OrderedDict

from PyQt4 import QtGui
from PyQt4.QtCore import Qt

from Orange.data import Table
from Orange.classification.svm import SVMLearner, SVMClassifier, NuSVMLearner
from Orange.widgets import widget, settings, gui
from Orange.widgets.utils.owlearnerwidget import OWProvidesLearner
from Orange.widgets.utils.sql import check_sql_input


class SVMBaseMixin(OWProvidesLearner):
    #: Kernel types
    Linear, Poly, RBF, Sigmoid = 0, 1, 2, 3
    #: Selected kernel type
    kernel_type = settings.Setting(RBF)
    #: kernel degree
    degree = settings.Setting(3)
    #: gamma
    gamma = settings.Setting(1.0)
    #: coef0 (adative constant)
    coef0 = settings.Setting(0.0)

    #: numerical tolerance
    tol = settings.Setting(0.001)

    want_main_area = False
    resizing_enabled = False

    kernels = (("Linear", "x⋅y"),
               ("Polynomial", "(g x⋅y + c)<sup>d</sup>"),
               ("RBF", "exp(-g|x-y|²)"),
               ("Sigmoid", "tanh(g x⋅y + c)"))

    def _add_kernel_box(self):
        # Initialize with the widest label to measure max width
        self.kernel_eq = self.kernels[-1][1]

        self.kernel_box = box = gui.widgetBox(
            self.controlArea, "Kernel", orientation="horizontal")

        buttonbox = gui.radioButtonsInBox(
            box, self, "kernel_type", btnLabels=[k[0] for k in self.kernels],
            callback=self._on_kernel_changed, addSpace=20)
        buttonbox.layout().setSpacing(10)
        gui.rubber(buttonbox)

        parambox = gui.widgetBox(box)
        gui.label(parambox, self, "Kernel: %(kernel_eq)s")
        common = dict(orientation="horizontal",
                      alignment=Qt.AlignRight, controlWidth=80)
        spbox = gui.widgetBox(parambox, orientation="horizontal")
        gui.rubber(spbox)
        inbox = gui.widgetBox(spbox)
        gamma = gui.doubleSpin(
            inbox, self, "gamma", 0.0, 10.0, 0.01, label=" g: ", **common)
        coef0 = gui.doubleSpin(
            inbox, self, "coef0", 0.0, 10.0, 0.01, label=" c: ", **common)
        degree = gui.doubleSpin(
            inbox, self, "degree", 0.0, 10.0, 0.5, label=" d: ", **common)
        self._kernel_params = [gamma, coef0, degree]
        gui.rubber(parambox)

        # This is the maximal height (all double spins are visible)
        # and the maximal width (the label is initialized to the widest one)
        box.layout().activate()
        box.setFixedHeight(box.sizeHint().height())
        box.setMinimumWidth(box.sizeHint().width())

    def _add_optimization_box(self):
        self.optimization_box = gui.widgetBox(
                self.controlArea, "Optimization parameters")

        gui.doubleSpin(
                self.optimization_box, self, "tol", 1e-6, 1.0, 1e-5,
                label="Numerical Tolerance",
                decimals=6, alignment=Qt.AlignRight, controlWidth=100
        )

    def _setup_layout(self):
        gui.lineEdit(self.controlArea, self, "learner_name", box="Name")

        self._add_type_box()
        self._add_kernel_box()
        self._add_optimization_box()

        box = gui.widgetBox(self.controlArea, True, orientation="horizontal")
        box.layout().addWidget(self.report_button)
        gui.separator(box, 20)
        gui.button(box, self, "&Apply", callback=self.apply, default=True)

    def _on_kernel_changed(self):
        enabled = [[False, False, False],  # linear
                   [True, True, True],  # poly
                   [True, False, False],  # rbf
                   [True, True, False]]  # sigmoid

        self.kernel_eq = self.kernels[self.kernel_type][1]
        mask = enabled[self.kernel_type]
        for spin, enabled in zip(self._kernel_params, mask):
            [spin.box.hide, spin.box.show][enabled]()

    def _report_kernel_parameters(self, items):
        if self.kernel_type == 0:
            items["Kernel"] = "Linear"
        elif self.kernel_type == 1:
            items["Kernel"] = \
                "Polynomial, ({g:.4} x⋅y + {c:.4})<sup>{d}</sup>".format(
                g=self.gamma, c=self.coef0, d=self.degree)
        elif self.kernel_type == 2:
            items["Kernel"] = "RBF, exp(-{:.4}|x-y|²)".format(self.gamma)
        else:
            items["Kernel"] = "Sigmoid, tanh({g:.4} x⋅y + {c:.4})".format(
                    g=self.gamma, c=self.coef0)


class OWSVMClassification(SVMBaseMixin, widget.OWWidget):
    name = "SVM"
    description = "Support vector machines classifier with standard " \
                  "selection of kernels."
    icon = "icons/SVM.svg"

    LEARNER = SVMLearner

    inputs = [("Data", Table, "set_data")] + OWProvidesLearner.inputs
    outputs = [("Learner", LEARNER, widget.Default),
               ("Classifier", SVMClassifier),
               ("Support vectors", Table)]

    learner_name = settings.Setting("SVM Learner")

    # 0: c_svc, 1: nu_svc
    svmtype = settings.Setting(0)
    C = settings.Setting(1.0)
    nu = settings.Setting(0.5)
    shrinking = settings.Setting(True),
    probability = settings.Setting(False)
    max_iter = settings.Setting(100)
    limit_iter = settings.Setting(True)

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
                self.controlArea, self, "svmtype", [], box="SVM Type",
                orientation=form)

        form.addWidget(gui.appendRadioButton(box, "C-SVM", addToLayout=False),
                       0, 0, Qt.AlignLeft)
        form.addWidget(QtGui.QLabel("Cost (C)"),
                       0, 1, Qt.AlignRight)
        form.addWidget(gui.doubleSpin(box, self, "C", 1e-3, 1000.0, 0.1,
                                      decimals=3, alignment=Qt.AlignRight,
                                      controlWidth=80, addToLayout=False),
                       0, 2)

        form.addWidget(gui.appendRadioButton(box, "ν-SVM", addToLayout=False),
                       1, 0, Qt.AlignLeft)
        form.addWidget(QtGui.QLabel(self.trUtf8("Complexity (\u03bd)")),
                       1, 1, Qt.AlignRight)
        form.addWidget(gui.doubleSpin(box, self, "nu", 0.05, 1.0, 0.05,
                                      decimals=2, alignment=Qt.AlignRight,
                                      controlWidth=80, addToLayout=False),
                       1, 2)

    def _add_optimization_box(self):
        super()._add_optimization_box()
        gui.spin(self.optimization_box, self, "max_iter", 50, 1e6, 50,
                 label="Iteration Limit", checked="limit_iter",
                 alignment=Qt.AlignRight, controlWidth=100)

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
            self.error([0, 1])
            if not learner.check_learner_adequacy(self.data.domain):
                self.error(0, learner.learner_adequacy_err_msg)
            elif len(np.unique(self.data.Y)) < 2:
                self.error(1, "Data contains only one target value.")
            else:
                classifier = learner(self.data)
                classifier.name = self.learner_name
                sv = self.data[classifier.skl_model.support_]

        self.send("Learner", learner)
        self.send("Classifier", classifier)
        self.send("Support vectors", sv)

    def send_report(self):
        self.report_items((("Name", self.learner_name),))

        items = OrderedDict()
        if self.svmtype == 0:
            items["SVM type"] = "C-SVM, C={}".format(self.C)
        else:
            items["SVM type"] = "ν-SVM, ν={}".format(self.nu)
        self._report_kernel_parameters(items)
        items["Numerical tolerance"] = "{:.6}".format(self.tol)
        items["Iteration limt"] = \
            self.max_iter if self.limit_iter else "unlimited"
        self.report_items("Model parameters", items)

        if self.data:
            self.report_data("Data", self.data)

if __name__ == "__main__":
    app = QtGui.QApplication([])
    w = OWSVMClassification()
    w.set_data(Table("iris")[:50])
    w.show()
    app.exec_()
