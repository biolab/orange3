from Orange.widgets.tests.base import GuiTest
from Orange.widgets.classify.owknn import OWKNNLearner
from PyQt4 import QtGui


class TestOwKnnClassification(GuiTest):
    def setUp(self):
        self.widget = OWKNNLearner()
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
        learner = self.widget.learner._params
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
            self.assertEqual(self.widget.learner._params.get("metric").capitalize(), self.combo_box[0].currentText().capitalize())

    def test_selected_values_weights(self):
        """ Check right value of combobox metric is right on output """
        for index, metric in enumerate(self.widget.weights):
            self.combo_box[1].activated.emit(index)
            self.combo_box[1].setCurrentIndex(index)
            self.assertEqual(self.combo_box[1].currentText().capitalize(), metric.capitalize())
            self.widget.apply()
            self.assertEqual(self.widget.learner._params.get("weights").capitalize(),
                             self.combo_box[1].currentText().capitalize())

    def test_learner_on_output(self):
        """ Check if learner is on output after create widget  and apply """
        self.widget.apply()
        self.assertNotEqual(self.widget.learner, None)

    def test_output_signal_classifier(self):
        """ Check if classifier out """
        output_classifier = False
        for signal in self.widget.outputs:
            if signal.name == "Classifier":
                output_classifier = True
                break
        self.assertEqual(output_classifier, True)
        self.assertEqual(self.widget.learner, None)
