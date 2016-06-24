from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.classify.owadaboost import OWAdaBoostClassification
from PyQt4 import QtGui


class TestOWAdaBoostClassification(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWAdaBoostClassification)
        self.spinners = []
        self.spinners.append(self.widget.findChildren(QtGui.QSpinBox)[0])
        self.spinners.append(self.widget.findChildren(QtGui.QDoubleSpinBox)[0])
        self.combobox_algorithm = self.widget.findChildren(QtGui.QComboBox)[0]

    def test_visible_boxes(self):
        """ Check if boxes are visible """
        self.assertEqual(self.spinners[0].isHidden(), False)
        self.assertEqual(self.spinners[1].isHidden(), False)
        self.assertEqual(self.combobox_algorithm.isHidden(), False)

    def test_parameters_on_output(self):
        """ Check right paramaters  on output """
        self.widget.apply()
        learner_params = self.widget.learner._params
        self.assertEqual(learner_params.get("n_estimators"), self.spinners[0].value())
        self.assertEqual(learner_params.get("learning_rate"), self.spinners[1].value())
        self.assertEqual(learner_params.get('algorithm'), self.combobox_algorithm.currentText())


    def test_output_algorithm(self):
        """ Check if right learning algorithm is on output when we change algorithm """
        for index, algorithmName in enumerate(self.widget.losses):
            self.combobox_algorithm.setCurrentIndex(index)
            self.combobox_algorithm.activated.emit(index)
            self.assertEqual(self.combobox_algorithm.currentText(), algorithmName)
            self.widget.apply()
            self.assertEqual(self.widget.learner._params.get("algorithm").capitalize(),
                             self.combobox_algorithm.currentText().capitalize())

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
