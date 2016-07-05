# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from PyQt4 import QtGui

from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.classify.owknn import OWKNNLearner


class TestOwKnnClassification(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWKNNLearner)
        self.combo_box = self.widget.findChildren(QtGui.QComboBox)
        self.spinner = self.widget.findChildren(QtGui.QSpinBox)

    def test_boxes_visible(self):
        """ Check if all boxes visible """
        self.assertEqual(self.combo_box[0].isHidden(), False)
        self.assertEqual(self.combo_box[1].isHidden(), False)
        self.assertEqual(self.spinner[0].isHidden(), False)

    def test_values_on_output(self):
        """ Check if all values right on output """
        self.widget.apply()
        learner = self.widget.learner.params
        self.assertEqual(learner.get("n_neighbors"), self.spinner[0].value())
        self.assertEqual(learner.get("metric").capitalize(), self.combo_box[0].currentText())
        self.assertEqual(learner.get("weights").capitalize(), self.combo_box[1].currentText())

    def test_selected_values_metrics(self):
        """ Check right value of combobox metric is right on output """
        for index, metric in enumerate(self.widget.metrics):
            self.combo_box[0].activated.emit(index)
            self.combo_box[0].setCurrentIndex(index)
            self.assertEqual(self.combo_box[0].currentText().capitalize(), metric.capitalize())
            self.widget.apply()
            self.assertEqual(self.widget.learner.params.get("metric").capitalize(),
                             self.combo_box[0].currentText().capitalize())

    def test_selected_values_weights(self):
        """ Check right value of combobox metric is right on output """
        for index, metric in enumerate(self.widget.weights):
            self.combo_box[1].activated.emit(index)
            self.combo_box[1].setCurrentIndex(index)
            self.assertEqual(self.combo_box[1].currentText().capitalize(), metric.capitalize())
            self.widget.apply()
            self.assertEqual(self.widget.learner.params.get("weights").capitalize(),
                             self.combo_box[1].currentText().capitalize())

    def test_learner_on_output(self):
        """ Check if learner is on output after create widget  and apply """
        self.widget.apply()
        self.assertNotEqual(self.widget.learner, None)

