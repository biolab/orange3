from collections import OrderedDict

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QLabel, QGridLayout

from Orange.data import Table
from Orange.modelling import SVMLearner, NuSVMLearner
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWSVM(OWBaseLearner):
    name = 'SVM'
    description = "Support Vector Machines map inputs to higher-dimensional " \
                  "feature spaces."
    icon = "icons/SVM.svg"
    replaces = [
        "Orange.widgets.classify.owsvmclassification.OWSVMClassification",
        "Orange.widgets.regression.owsvmregression.OWSVMRegression",
    ]
    priority = 50

    LEARNER = SVMLearner

    outputs = [("Support vectors", Table, widget.Explicit)]

    #: Different types of SVMs
    SVM, Nu_SVM = range(2)
    #: SVM type
    svm_type = Setting(SVM)

    C = Setting(1.)
    epsilon = Setting(.1)
    nu_C = Setting(1.)
    nu = Setting(.5)

    #: Kernel types
    Linear, Poly, RBF, Sigmoid = range(4)
    #: Selected kernel type
    kernel_type = Setting(RBF)
    #: kernel degree
    degree = Setting(3)
    #: gamma
    gamma = Setting(0.0)
    #: coef0 (adative constant)
    coef0 = Setting(0.0)

    #: numerical tolerance
    tol = Setting(0.001)
    #: whether or not to limit number of iterations
    limit_iter = Setting(True)
    #: maximum number of iterations
    max_iter = Setting(100)

    _default_gamma = "auto"
    kernels = (("Linear", "x⋅y"),
               ("Polynomial", "(g x⋅y + c)<sup>d</sup>"),
               ("RBF", "exp(-g|x-y|²)"),
               ("Sigmoid", "tanh(g x⋅y + c)"))

    def add_main_layout(self):
        self._add_type_box()
        self._add_kernel_box()
        self._add_optimization_box()
        self._show_right_kernel()

    def _add_type_box(self):
        form = QGridLayout()
        self.type_box = box = gui.radioButtonsInBox(
            self.controlArea, self, "svm_type", [], box="SVM Type",
            orientation=form, callback=self._update_type)

        self.epsilon_radio = gui.appendRadioButton(
            box, "SVM", addToLayout=False)
        self.C_spin = gui.doubleSpin(
            box, self, "C", 0.1, 512.0, 0.1, decimals=2,
            alignment=Qt.AlignRight, addToLayout=False,
            callback=self.settings_changed)
        self.epsilon_spin = gui.doubleSpin(
            box, self, "epsilon", 0.1, 512.0, 0.1, decimals=2,
            alignment=Qt.AlignRight, addToLayout=False,
            callback=self.settings_changed)
        form.addWidget(self.epsilon_radio, 0, 0, Qt.AlignLeft)
        form.addWidget(QLabel("Cost (C):"), 0, 1, Qt.AlignRight)
        form.addWidget(self.C_spin, 0, 2)
        form.addWidget(QLabel(
            "Regression loss epsilon (ε):"), 1, 1, Qt.AlignRight)
        form.addWidget(self.epsilon_spin, 1, 2)

        self.nu_radio = gui.appendRadioButton(box, "ν-SVM", addToLayout=False)
        self.nu_C_spin = gui.doubleSpin(
            box, self, "nu_C", 0.1, 512.0, 0.1, decimals=2,
            alignment=Qt.AlignRight, addToLayout=False,
            callback=self.settings_changed)
        self.nu_spin = gui.doubleSpin(
            box, self, "nu", 0.05, 1.0, 0.05, decimals=2,
            alignment=Qt.AlignRight, addToLayout=False,
            callback=self.settings_changed)
        form.addWidget(self.nu_radio, 2, 0, Qt.AlignLeft)
        form.addWidget(QLabel("Regression cost (C):"), 2, 1, Qt.AlignRight)
        form.addWidget(self.nu_C_spin, 2, 2)
        form.addWidget(QLabel("Complexity bound (ν):"), 3, 1, Qt.AlignRight)
        form.addWidget(self.nu_spin, 3, 2)

        # Correctly enable/disable the appropriate boxes
        self._update_type()

    def _update_type(self):
        # Enable/disable SVM type parameters depending on selected SVM type
        if self.svm_type == self.SVM:
            self.C_spin.setEnabled(True)
            self.epsilon_spin.setEnabled(True)
            self.nu_C_spin.setEnabled(False)
            self.nu_spin.setEnabled(False)
        else:
            self.C_spin.setEnabled(False)
            self.epsilon_spin.setEnabled(False)
            self.nu_C_spin.setEnabled(True)
            self.nu_spin.setEnabled(True)
        self.settings_changed()

    def _add_kernel_box(self):
        # Initialize with the widest label to measure max width
        self.kernel_eq = self.kernels[-1][1]

        box = gui.hBox(self.controlArea, "Kernel")

        self.kernel_box = buttonbox = gui.radioButtonsInBox(
            box, self, "kernel_type", btnLabels=[k[0] for k in self.kernels],
            callback=self._on_kernel_changed, addSpace=20)
        buttonbox.layout().setSpacing(10)
        gui.rubber(buttonbox)

        parambox = gui.vBox(box)
        gui.label(parambox, self, "Kernel: %(kernel_eq)s")
        common = dict(orientation=Qt.Horizontal, callback=self.settings_changed,
                      alignment=Qt.AlignRight, controlWidth=80)
        spbox = gui.hBox(parambox)
        gui.rubber(spbox)
        inbox = gui.vBox(spbox)
        gamma = gui.doubleSpin(
            inbox, self, "gamma", 0.0, 10.0, 0.01, label=" g: ", **common)
        gamma.setSpecialValueText(self._default_gamma)
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
        self.optimization_box = gui.vBox(
            self.controlArea, "Optimization Parameters")
        self.tol_spin = gui.doubleSpin(
            self.optimization_box, self, "tol", 1e-4, 1.0, 1e-4,
            label="Numerical tolerance: ",
            alignment=Qt.AlignRight, controlWidth=100,
            callback=self.settings_changed)
        self.max_iter_spin = gui.spin(
            self.optimization_box, self, "max_iter", 5, 1e6, 50,
            label="Iteration limit: ", checked="limit_iter",
            alignment=Qt.AlignRight, controlWidth=100,
            callback=self.settings_changed,
            checkCallback=self.settings_changed)

    def _show_right_kernel(self):
        enabled = [[False, False, False],  # linear
                   [True, True, True],  # poly
                   [True, False, False],  # rbf
                   [True, True, False]]  # sigmoid

        self.kernel_eq = self.kernels[self.kernel_type][1]
        mask = enabled[self.kernel_type]
        for spin, enabled in zip(self._kernel_params, mask):
            [spin.box.hide, spin.box.show][enabled]()

    def update_model(self):
        super().update_model()
        sv = None
        if self.model is not None:
            sv = self.data[self.model.skl_model.support_]
        self.send("Support vectors", sv)

    def _on_kernel_changed(self):
        self._show_right_kernel()
        self.settings_changed()

    def create_learner(self):
        kernel = ["linear", "poly", "rbf", "sigmoid"][self.kernel_type]
        common_args = {
            'kernel': kernel,
            'degree': self.degree,
            'gamma': self.gamma or self._default_gamma,
            'coef0': self.coef0,
            'probability': True,
            'tol': self.tol,
            'max_iter': self.max_iter if self.limit_iter else -1,
            'preprocessors': self.preprocessors
        }
        if self.svm_type == self.SVM:
            return SVMLearner(C=self.C, epsilon=self.epsilon, **common_args)
        else:
            return NuSVMLearner(nu=self.nu, C=self.nu_C, **common_args)

    def get_learner_parameters(self):
        items = OrderedDict()
        if self.svm_type == self.SVM:
            items["SVM type"] = "SVM, C={}, ε={}".format(self.C, self.epsilon)
        else:
            items["SVM type"] = "ν-SVM, ν={}, C={}".format(self.nu, self.nu_C)
        self._report_kernel_parameters(items)
        items["Numerical tolerance"] = "{:.6}".format(self.tol)
        items["Iteration limt"] = self.max_iter if self.limit_iter else "unlimited"
        return items

    def _report_kernel_parameters(self, items):
        gamma = self.gamma or self._default_gamma
        if self.kernel_type == 0:
            items["Kernel"] = "Linear"
        elif self.kernel_type == 1:
            items["Kernel"] = \
                "Polynomial, ({g:.4} x⋅y + {c:.4})<sup>{d}</sup>".format(
                    g=gamma, c=self.coef0, d=self.degree)
        elif self.kernel_type == 2:
            items["Kernel"] = "RBF, exp(-{:.4}|x-y|²)".format(gamma)
        else:
            items["Kernel"] = "Sigmoid, tanh({g:.4} x⋅y + {c:.4})".format(
                g=gamma, c=self.coef0)


if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    ow = OWSVM()
    ow.resetSettings()
    d = Table(sys.argv[1] if len(sys.argv) > 1 else 'iris')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()
