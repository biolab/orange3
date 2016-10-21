from collections import OrderedDict

from AnyQt import QtWidgets
from AnyQt.QtWidgets import QLabel
from AnyQt.QtCore import Qt

from Orange.data import Table
from Orange.classification.svm import SVMLearner, NuSVMLearner
from Orange.widgets import widget, settings, gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWBaseSVM(OWBaseLearner):
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

    _default_gamma = "auto"
    kernels = (("Linear", "x⋅y"),
               ("Polynomial", "(g x⋅y + c)<sup>d</sup>"),
               ("RBF", "exp(-g|x-y|²)"),
               ("Sigmoid", "tanh(g x⋅y + c)"))

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
            self.optimization_box, self, "tol", 1e-6, 1.0, 1e-5,
            label="Numerical tolerance:",
            decimals=6, alignment=Qt.AlignRight, controlWidth=100,
            callback=self.settings_changed)

    def add_main_layout(self):
        self._add_type_box()
        self._add_kernel_box()
        self._add_optimization_box()
        self._show_right_kernel()

    def _show_right_kernel(self):
        enabled = [[False, False, False],  # linear
                   [True, True, True],  # poly
                   [True, False, False],  # rbf
                   [True, True, False]]  # sigmoid

        self.kernel_eq = self.kernels[self.kernel_type][1]
        mask = enabled[self.kernel_type]
        for spin, enabled in zip(self._kernel_params, mask):
            [spin.box.hide, spin.box.show][enabled]()

    def _on_kernel_changed(self):
        self._show_right_kernel()
        self.settings_changed()

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

    def update_model(self):
        super().update_model()

        sv = None
        if self.valid_data:
            sv = self.data[self.model.skl_model.support_]
        self.send("Support vectors", sv)


class OWSVMClassification(OWBaseSVM):
    name = "SVM"
    description = "Support Vector Machines map inputs to higher-dimensional " \
                  "feature spaces that best separate different classes. "
    icon = "icons/SVM.svg"
    priority = 50

    LEARNER = SVMLearner

    outputs = [("Support vectors", Table, widget.Explicit)]

    C_SVC, Nu_SVC = 0, 1
    svmtype = settings.Setting(0)
    C = settings.Setting(1.0)
    nu = settings.Setting(0.5)
    shrinking = settings.Setting(True),
    probability = settings.Setting(False)
    max_iter = settings.Setting(100)
    limit_iter = settings.Setting(True)

    def _add_type_box(self):
        form = QtWidgets.QGridLayout()
        self.type_box = box = gui.radioButtonsInBox(
            self.controlArea, self, "svmtype", [], box="SVM Type",
            orientation=form, callback=self.settings_changed)

        self.c_radio = gui.appendRadioButton(box, "C-SVM", addToLayout=False)
        self.nu_radio = gui.appendRadioButton(box, "ν-SVM", addToLayout=False)
        self.c_spin = gui.doubleSpin(
            box, self, "C", 1e-3, 1000.0, 0.1, decimals=3,
            alignment=Qt.AlignRight, controlWidth=80, addToLayout=False,
            callback=self.settings_changed)
        self.nu_spin = gui.doubleSpin(
            box, self, "nu", 0.05, 1.0, 0.05, decimals=2,
            alignment=Qt.AlignRight, controlWidth=80, addToLayout=False,
            callback=self.settings_changed)
        form.addWidget(self.c_radio, 0, 0, Qt.AlignLeft)
        form.addWidget(QLabel("Cost (C):"), 0, 1, Qt.AlignRight)
        form.addWidget(self.c_spin, 0, 2)
        form.addWidget(self.nu_radio, 1, 0, Qt.AlignLeft)
        form.addWidget(QLabel("Complexity (ν):"), 1, 1, Qt.AlignRight)
        form.addWidget(self.nu_spin, 1, 2)

    def _add_optimization_box(self):
        super()._add_optimization_box()
        self.max_iter_spin = gui.spin(
            self.optimization_box, self, "max_iter", 50, 1e6, 50,
            label="Iteration limit:", checked="limit_iter",
            alignment=Qt.AlignRight, controlWidth=100,
            callback=self.settings_changed)

    def create_learner(self):
        kernel = ["linear", "poly", "rbf", "sigmoid"][self.kernel_type]
        common_args = dict(
            kernel=kernel,
            degree=self.degree,
            gamma=self.gamma or self._default_gamma,
            coef0=self.coef0,
            tol=self.tol,
            max_iter=self.max_iter if self.limit_iter else -1,
            probability=True,
            preprocessors=self.preprocessors
        )
        if self.svmtype == OWSVMClassification.C_SVC:
            return SVMLearner(C=self.C, **common_args)
        else:
            return NuSVMLearner(nu=self.nu, **common_args)

    def get_learner_parameters(self):
        items = OrderedDict()
        if self.svmtype == OWSVMClassification.C_SVC:
            items["SVM type"] = "C-SVM, C={}".format(self.C)
        else:
            items["SVM type"] = "ν-SVM, ν={}".format(self.nu)
        self._report_kernel_parameters(items)
        items["Numerical tolerance"] = "{:.6}".format(self.tol)
        items["Iteration limt"] = self.max_iter if self.limit_iter else "unlimited"
        return items


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    w = OWSVMClassification()
    w.set_data(Table("iris")[:50])
    w.show()
    app.exec_()
