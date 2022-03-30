from unittest.mock import Mock

from Orange.classification import ThresholdLearner, CalibratedLearner, \
    NaiveBayesLearner, ThresholdClassifier, CalibratedClassifier
from Orange.classification.base_classification import ModelClassification, \
    LearnerClassification
from Orange.classification.naive_bayes import NaiveBayesModel
from Orange.data import Table
from Orange.widgets.model.owcalibratedlearner import OWCalibratedLearner
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin, \
    datasets
from Orange.widgets.tests.utils import qbuttongroup_emit_clicked


class TestOWCalibratedLearner(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(
            OWCalibratedLearner, stored_settings={"auto_apply": False})
        self.send_signal(self.widget.Inputs.base_learner, NaiveBayesLearner())

        self.data = Table("heart_disease")
        self.valid_datasets = (self.data,)
        self.inadequate_dataset = (Table(datasets.path("testing_dataset_reg")),)
        self.learner_class = LearnerClassification
        self.model_class = ModelClassification
        self.model_name = 'Calibrated classifier'
        self.parameters = []

    def test_output_learner(self):
        """Check if learner is on output after apply"""
        # Overridden to change the output type in the last test
        initial = self.get_output("Learner")
        self.assertIsNotNone(initial, "Does not initialize the learner output")
        self.widget.apply_button.button.click()
        newlearner = self.get_output("Learner")
        self.assertIsNot(initial, newlearner,
                         "Does not send a new learner instance on `Apply`.")
        self.assertIsNotNone(newlearner)
        self.assertIsInstance(
            newlearner,
            (CalibratedLearner, ThresholdLearner, NaiveBayesLearner))

    def test_output_model(self):
        """Check if model is on output after sending data and apply"""
        # Overridden to change the output type in the last two test
        self.assertIsNone(self.get_output(self.widget.Outputs.model))
        self.widget.apply_button.button.click()
        self.assertIsNone(self.get_output(self.widget.Outputs.model))
        self.send_signal('Data', self.data)
        self.widget.apply_button.button.click()
        self.wait_until_stop_blocking()
        model = self.get_output(self.widget.Outputs.model)
        self.assertIsNotNone(model)
        self.assertIsInstance(
            model, (CalibratedClassifier, ThresholdClassifier, NaiveBayesModel))

    def test_create_learner(self):
        widget = self.widget  #: OWCalibratedLearner
        self.widget.base_learner = Mock()

        widget.calibration = widget.SigmoidCalibration
        widget.threshold = widget.OptimizeF1
        learner = self.widget.create_learner()
        self.assertIsInstance(learner, ThresholdLearner)
        self.assertEqual(learner.threshold_criterion, learner.OptimizeF1)
        cal_learner = learner.base_learner
        self.assertIsInstance(cal_learner, CalibratedLearner)
        self.assertEqual(cal_learner.calibration_method, cal_learner.Sigmoid)
        self.assertIs(cal_learner.base_learner, self.widget.base_learner)

        widget.calibration = widget.IsotonicCalibration
        widget.threshold = widget.OptimizeCA
        learner = self.widget.create_learner()
        self.assertIsInstance(learner, ThresholdLearner)
        self.assertEqual(learner.threshold_criterion, learner.OptimizeCA)
        cal_learner = learner.base_learner
        self.assertIsInstance(cal_learner, CalibratedLearner)
        self.assertEqual(cal_learner.calibration_method, cal_learner.Isotonic)
        self.assertIs(cal_learner.base_learner, self.widget.base_learner)

        widget.calibration = widget.NoCalibration
        widget.threshold = widget.OptimizeCA
        learner = self.widget.create_learner()
        self.assertIsInstance(learner, ThresholdLearner)
        self.assertEqual(learner.threshold_criterion, learner.OptimizeCA)
        self.assertIs(learner.base_learner, self.widget.base_learner)

        widget.calibration = widget.IsotonicCalibration
        widget.threshold = widget.NoThresholdOptimization
        learner = self.widget.create_learner()
        self.assertIsInstance(learner, CalibratedLearner)
        self.assertEqual(learner.calibration_method, cal_learner.Isotonic)
        self.assertIs(learner.base_learner, self.widget.base_learner)

        widget.calibration = widget.NoCalibration
        widget.threshold = widget.NoThresholdOptimization
        learner = self.widget.create_learner()
        self.assertIs(learner, self.widget.base_learner)

        widget.calibration = widget.SigmoidCalibration
        widget.threshold = widget.OptimizeF1
        widget.base_learner = None
        learner = self.widget.create_learner()
        self.assertIsNone(learner)

    def test_preprocessors(self):
        widget = self.widget  #: OWCalibratedLearner
        self.widget.base_learner = Mock()
        self.widget.base_learner.preprocessors = ()

        widget.calibration = widget.SigmoidCalibration
        widget.threshold = widget.OptimizeF1
        widget.preprocessors = Mock()
        learner = self.widget.create_learner()
        self.assertEqual(learner.preprocessors, (widget.preprocessors, ))
        self.assertEqual(learner.base_learner.preprocessors, ())
        self.assertEqual(learner.base_learner.base_learner.preprocessors, ())

        widget.calibration = widget.NoCalibration
        widget.threshold = widget.NoThresholdOptimization
        learner = self.widget.create_learner()
        self.assertIsNot(learner, self.widget.base_learner)
        self.assertFalse(
            isinstance(learner, (CalibratedLearner, ThresholdLearner)))
        self.assertEqual(learner.preprocessors, (widget.preprocessors, ))

    def test_set_learner_calls_unconditional_apply(self):
        widget = self.widget
        self.assertIsNotNone(self.get_output(widget.Outputs.learner))

        widget.auto_apply = False
        self.send_signal(widget.Inputs.base_learner, None)
        self.assertIsNone(self.get_output(widget.Outputs.learner))

    def test_name_changes(self):
        widget = self.widget
        widget.auto_apply = True
        learner = NaiveBayesLearner()
        learner.name = "foo"
        self.send_signal(widget.Inputs.base_learner, learner)

        widget.calibration = widget.IsotonicCalibration
        widget.threshold = widget.OptimizeCA
        qbuttongroup_emit_clicked(widget.controls.calibration.group,
                                  widget.IsotonicCalibration)

        learner = self.get_output(widget.Outputs.learner)
        self.assertEqual(learner.name, "Foo + Isotonic + CA")

        widget.calibration = widget.NoCalibration
        widget.threshold = widget.OptimizeCA
        qbuttongroup_emit_clicked(widget.controls.calibration.group,
                                  widget.NoCalibration)
        learner = self.get_output(widget.Outputs.learner)
        self.assertEqual(learner.name, "Foo + CA")

        self.send_signal(widget.Inputs.base_learner, None)
        self.assertEqual(widget.controls.learner_name.placeholderText(),
                         "Calibrated Learner")
