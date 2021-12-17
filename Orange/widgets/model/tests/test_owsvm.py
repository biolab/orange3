# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from scipy.sparse import csr_matrix

from Orange.widgets.model.owsvm import OWSVM
from Orange.widgets.tests.base import (
    WidgetTest,
    DefaultParameterMapping,
    ParameterMapping,
    WidgetLearnerTestMixin
)
from Orange.data import Table


class TestOWSVMClassification(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(
            OWSVM, stored_settings={"auto_apply": False})
        self.init()

        # All this nonsense is necessary because we add an `auto` option to the
        # gamma spin box
        gamma_spin = self.widget._kernel_params[0]
        values = [self.widget._default_gamma, gamma_spin.maximum()]

        def getter():
            value = gamma_spin.value()
            return gamma_spin.specialValueText() \
                if value == gamma_spin.minimum() else value

        def setter(value):
            if value == gamma_spin.specialValueText():
                gamma_spin.setValue(gamma_spin.minimum())
            else:
                gamma_spin.setValue(value)

        self.parameters = [
            ParameterMapping("C", self.widget.c_spin),
            ParameterMapping("gamma", self.widget._kernel_params[0],
                             values=values, setter=setter, getter=getter),
            ParameterMapping("coef0", self.widget._kernel_params[1]),
            ParameterMapping("degree", self.widget._kernel_params[2]),
            ParameterMapping("tol", self.widget.tol_spin),
            ParameterMapping("max_iter", self.widget.max_iter_spin[1])]

    def test_parameters_unchecked(self):
        """Check learner and model for various values of all parameters
        when Iteration limit is not checked
        """
        self.widget.max_iter_spin[0].setCheckState(False)
        self.parameters[-1] = DefaultParameterMapping("max_iter", -1)
        self.test_parameters()

    def test_parameters_svm_type(self):
        """Check learner and model for various values of all parameters
        when NuSVM is chosen
        """
        self.assertEqual(self.widget.svm_type, OWSVM.SVM)
        # setChecked(True) does not trigger callback event
        self.widget.nu_radio.click()
        self.assertEqual(self.widget.svm_type, OWSVM.Nu_SVM)
        self.parameters[0] = ParameterMapping("nu", self.widget.nu_spin)
        self.test_parameters()

    def test_kernel_equation(self):
        """Check if the right equation is written according to kernel """
        for i in range(4):
            if self.widget.kernel_box.buttons[i].isChecked():
                self.assertEqual(self.widget.kernel_eq,
                                 self.widget.kernels[i][1])
                break
        for i in range(4):
            self.widget.kernel_box.buttons[i].click()
            self.assertEqual(self.widget.kernel_eq, self.widget.kernels[i][1])

    def test_kernel_spins(self):
        """Check if the right spins are visible according to kernel """
        for i, hidden in enumerate([[True, True, True],
                                    [False, False, False],
                                    [False, True, True],
                                    [False, False, True]]):
            if self.widget.kernel_box.buttons[i].isChecked():
                self.assertEqual([self.widget._kernel_params[j].box.isHidden()
                                  for j in range(3)], hidden)
                break
        for i, hidden in enumerate([[True, True, True],
                                    [False, False, False],
                                    [False, True, True],
                                    [False, False, True]]):
            self.widget.kernel_box.buttons[i].click()
            self.assertEqual([self.widget._kernel_params[j].box.isHidden()
                              for j in range(3)], hidden)

    def test_sparse_warning(self):
        """Check if the user is warned about sparse input"""
        data = Table("iris")
        self.send_signal("Data", data)
        self.assertFalse(self.widget.Warning.sparse_data.is_shown())

        with data.unlocked():
            data.X = csr_matrix(data.X)
        self.send_signal("Data", data)
        self.assertTrue(self.widget.Warning.sparse_data.is_shown())
