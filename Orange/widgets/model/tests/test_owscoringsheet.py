import unittest

from orangewidget.tests.base import WidgetTest

from Orange.data import Table
from Orange.preprocess import Impute

from Orange.classification.scoringsheet import ScoringSheetLearner
from Orange.widgets.model.owscoringsheet import OWScoringSheet


class TestOWScoringSheet(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.heart = Table("heart_disease")
        cls.housing = Table("housing")
        cls.scoring_sheet_learner = ScoringSheetLearner(20, 5, 5, None)
        cls.scoring_sheet_model = cls.scoring_sheet_learner(cls.heart)

    def setUp(self):
        self.widget = self.create_widget(OWScoringSheet)

    def test_no_data_input(self):
        self.assertIsNotNone(self.get_output(self.widget.Outputs.learner))
        self.assertIsNone(self.get_output(self.widget.Outputs.model))

    def test_numerical_target_attribute(self):
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.fitting_failed.is_shown())

    def test_settings_in_learner(self):
        self.widget.num_attr_after_selection = 20
        self.widget.num_decision_params = 7
        self.widget.max_points_per_param = 8
        self.widget.custom_features_checkbox = True
        self.widget.num_input_features = 4

        self.widget.apply()

        self.send_signal(self.widget.Inputs.data, self.heart)
        learner = self.get_output(self.widget.Outputs.learner)

        self.assertEqual(learner.num_decision_params, 7)
        self.assertEqual(learner.max_points_per_param, 8)
        self.assertEqual(learner.num_input_features, 4)

    def test_settings_in_model(self):
        self.widget.num_attr_after_selection = 20
        self.widget.num_decision_params = 7
        self.widget.max_points_per_param = 8
        self.widget.custom_features_checkbox = True
        self.widget.num_input_features = 4

        self.widget.apply()

        self.send_signal(self.widget.Inputs.data, self.heart)
        self.wait_until_finished()
        model = self.get_output(self.widget.Outputs.model)

        coefficients = model.model.coefficients
        non_zero_coefficients = [coef for coef in coefficients if coef != 0]

        self.assertEqual(len(coefficients), self.widget.num_attr_after_selection)

        # most often equal, but in some cases the optimizer finds fewer parameters
        self.assertLessEqual(len(non_zero_coefficients), self.widget.num_decision_params)

        self.assertLessEqual(
            max(non_zero_coefficients, key=lambda x: abs(x)),
            self.widget.max_points_per_param,
        )

    def test_custom_number_input_features_information(self):
        self.widget.custom_features_checkbox = True
        self.widget.custom_input_features()
        self.assertTrue(self.widget.Information.custom_num_of_input_features.is_shown())

        self.widget.custom_features_checkbox = False
        self.widget.custom_input_features()
        self.assertFalse(
            self.widget.Information.custom_num_of_input_features.is_shown()
        )

    def test_custom_preprocessors_information(self):
        preprocessor = Impute()
        self.send_signal(self.widget.Inputs.preprocessor, preprocessor)
        self.assertTrue(self.widget.Information.ignored_preprocessors.is_shown())

        self.send_signal(self.widget.Inputs.preprocessor, None)
        self.assertFalse(self.widget.Information.ignored_preprocessors.is_shown())

    def test_custom_preprocessors_spin_disabled(self):
        preprocessor = Impute()
        self.send_signal(self.widget.Inputs.preprocessor, preprocessor)
        self.assertFalse(self.widget.num_attr_after_selection_spin.isEnabled())

    def test_default_preprocessors_are_used(self):
        learner = self.get_output(self.widget.Outputs.learner)

        self.assertIsNotNone(learner.preprocessors)
        self.assertEqual(len(learner.preprocessors), 5)

    def test_custom_preprocessors_are_used(self):
        preprocessor = Impute()
        self.send_signal(self.widget.Inputs.preprocessor, preprocessor)
        learner = self.get_output(self.widget.Outputs.learner)

        self.assertIsNotNone(learner.preprocessors)
        self.assertEqual(len(learner.preprocessors), 1)
        self.assertEqual(learner.preprocessors[0], preprocessor)


if __name__ == "__main__":
    unittest.main()
