# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table
from Orange.widgets.classify.owsvmclassification import OWSVMClassification
from Orange.widgets.tests.base import WidgetTest
from PyQt4 import QtGui


class TestOWSVMClassification(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSVMClassification)
        self.widget.spin_boxes = self.widget.findChildren(QtGui.QDoubleSpinBox)
        self.widget.spin_boxes.append(self.widget.findChildren(QtGui.QSpinBox)[0])  # for max iter spin
        self.widget.max_iter_check_box = self.widget.findChildren(QtGui.QCheckBox)[0]  # for max iter checkbox
        self.spin_boxes = self.widget.spin_boxes
        self.event_data = None

    def test_kernel_equation_run(self):
        """ Check if right text is written for specific kernel """
        for i in range(0, 4):
            if self.widget.kernel_box.buttons[i].isChecked():
                self.assertEqual(self.widget.kernel_eq, self.widget.kernels[i][1])

    def test_kernel_equation(self):
        """ Check if right text is written for specific kernel after click """
        for index in range(0, 4):
            self.widget.kernel_box.buttons[index].click()
            self.assertEqual(self.widget.kernel_eq, self.widget.kernels[index][1])

    def test_kernel_display_run(self):
        """ Check if right spinner box for selected kernel are visible after widget start """
        for button_pos, value in ((0, [False, False, False]),
                                  (1, [True, True, True]),
                                  (2, [True, False, False]),
                                  (3, [True, True, False])):
            if self.widget.kernel_box.buttons[button_pos].isChecked():
                self.assertEqual([not self.spin_boxes[i].box.isHidden() for i in range(2, 5)], value)
                break

    def test_kernel_display(self):
        """ Check if right spinner box for selected kernel are visible after we select kernel """
        for button_pos, value in ((0, [False, False, False]),
                                  (1, [True, True, True]),
                                  (2, [True, False, False]),
                                  (3, [True, True, False])):
            self.widget.kernel_box.buttons[button_pos].click()
            self.widget.kernel_box.buttons[button_pos].isChecked()
            self.assertEqual([not self.spin_boxes[i].box.isHidden() for i in range(2, 5)], value)

    def test_optimization_box_visible(self):
        """ Check if both spinner box is visible after starting widget """
        self.assertEqual(self.spin_boxes[5].box.isHidden(), False)
        self.assertEqual(self.spin_boxes[6].box.isHidden(), False)

    def test_optimization_box_checked(self):
        """ Check if spinner box for iteration limit is enabled or disabled """
        for value in (True, False):
            self.widget.max_iter_check_box.setChecked(value)
            self.assertEqual(self.widget.max_iter_check_box.isChecked(), value)
            self.assertEqual(self.spin_boxes[6].isEnabled(), value)

    def test_type_button_checked(self):
        """ Check if SVM type is selected after click """
        self.widget.type_box.buttons[0].click()
        self.assertEqual(self.widget.type_box.buttons[0].isChecked(), True)
        self.widget.type_box.buttons[1].click()
        self.assertEqual(self.widget.type_box.buttons[1].isChecked(), True)

    def test_type_button_properties_visible(self):
        """ Check if spinner box in SVM type are visible """
        self.assertEqual(not self.spin_boxes[0].isHidden(), True)
        self.assertEqual(not self.spin_boxes[1].isHidden(), True)

    def test_data_before_apply(self):
        """ Check if data are set """
        self.widget.set_data(Table("iris")[:100])
        self.widget.apply()
        self.assertEqual(len(self.widget.data), 100)

    def test_output_signal_classifier(self):
        """ Check if we have classifier on output """
        output_classifier = False
        for signal in self.widget.outputs:
            if signal.name == "Classifier":
                output_classifier = True
                break
        self.assertEqual(output_classifier, True)
        self.assertEqual(self.widget.learner, None)

    def test_output_signal_learner(self):
        """ Check if we have on output learner """
        self.widget.kernel_box.buttons[0].click()
        self.widget.set_data(Table("iris")[:100])
        self.widget.apply()
        self.assertNotEqual(self.widget.learner, None)

    def test_output_params(self):
        """ Check ouput params """
        self.widget.kernel_box.buttons[0].click()
        self.widget.set_data(Table("iris")[:100])
        self.widget.max_iter_check_box.setChecked(True)
        self.widget.apply()
        self.widget.type_box.buttons[0].click()
        params = self.widget.learner.params
        self.assertEqual(params.get('C'), self.spin_boxes[0].value())
        self.widget.type_box.buttons[1].click()
        params = self.widget.learner.params
        self.assertEqual(params.get('nu'), self.spin_boxes[1].value())
        self.assertEqual(params.get('gamma'), self.spin_boxes[2].value())
        self.assertEqual(params.get('coef0'), self.spin_boxes[3].value())
        self.assertEqual(params.get('degree'), self.spin_boxes[4].value())
        self.assertEqual(params.get('tol'), self.spin_boxes[5].value())
        self.assertEqual(params.get('max_iter'), self.spin_boxes[6].value())
