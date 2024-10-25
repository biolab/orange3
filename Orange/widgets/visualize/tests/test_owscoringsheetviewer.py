import unittest

from AnyQt.QtCore import Qt

from orangewidget.tests.base import WidgetTest

from Orange.data import Table
from Orange.classification import LogisticRegressionLearner, ScoringSheetLearner
from Orange.widgets.widget import AttributeList
from Orange.widgets.visualize.owscoringsheetviewer import OWScoringSheetViewer


class TestOWScoringSheetViewer(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.heart = Table("heart_disease")
        cls.scoring_sheet_learner = ScoringSheetLearner(20, 5, 5, None)
        cls.scoring_sheet_model = cls.scoring_sheet_learner(cls.heart)
        cls.logistic_regression_learner = LogisticRegressionLearner(tol=1)
        cls.logistic_regression_model = cls.logistic_regression_learner(cls.heart[:10])

    def setUp(self):
        self.widget = self.create_widget(OWScoringSheetViewer)

    def test_no_classifier_input(self):
        coef_table = self.widget.coefficient_table
        risk_slider = self.widget.risk_slider
        class_combo = self.widget.class_combo

        self.assertEqual(coef_table.rowCount(), 0)
        self.assertEqual(risk_slider.slider.value(), 0)
        self.assertEqual(class_combo.count(), 0)

    def test_no_classifier_output(self):
        self.assertIsNone(self.get_output(self.widget.Outputs.features))
        self.send_signal(self.widget.Inputs.classifier, self.scoring_sheet_model)
        self.send_signal(self.widget.Inputs.classifier, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.features))

    def test_classifier_output(self):
        self.send_signal(self.widget.Inputs.classifier, self.scoring_sheet_model)
        output = self.get_output(self.widget.Outputs.features)
        self.assertIsInstance(output, AttributeList)
        self.assertEqual(len(output), self.scoring_sheet_learner.num_decision_params)

    def test_table_population_on_model_input(self):
        self.send_signal(self.widget.Inputs.classifier, self.scoring_sheet_model)
        table = self.widget.coefficient_table
        self.assertEqual(
            table.rowCount(), self.scoring_sheet_learner.num_decision_params
        )

        for column in range(table.columnCount()):
            for row in range(table.rowCount()):
                self.assertIsNotNone(table.item(row, column))
                if column == 2:
                    self.assertEqual(table.item(row, column).checkState(), Qt.Unchecked)

    def test_slider_population_on_model_input(self):
        self.send_signal(self.widget.Inputs.classifier, self.scoring_sheet_model)
        slider = self.widget.risk_slider
        self.assertIsNotNone(slider.points)
        self.assertIsNotNone(slider.probabilities)
        self.assertEqual(len(slider.points), len(slider.probabilities))

    def test_slider_update_on_checkbox_toggle(self):
        self.send_signal(self.widget.Inputs.classifier, self.scoring_sheet_model)

        coef_table = self.widget.coefficient_table
        risk_slider = self.widget.risk_slider
        risk_slider_points = risk_slider.points

        # Get the items in the first row of the table
        checkbox_item = coef_table.item(0, 2)
        attribute_points_item = coef_table.item(0, 1)

        # Check if the slider value is "0" before changing the checkbox
        self.assertEqual(risk_slider.slider.value(), risk_slider_points.index(0))

        # Directly change the checkbox state to Checked
        checkbox_item.setCheckState(Qt.Checked)

        # Re-fetch the items after change
        attribute_points_item = coef_table.item(0, 1)

        # Check if the slider value is now the same as the attribute's coefficient
        self.assertEqual(
            risk_slider.slider.value(),
            risk_slider_points.index(float(attribute_points_item.text())),
        )

        # Directly change the checkbox state to Unchecked
        checkbox_item.setCheckState(Qt.Unchecked)

        # Check if the slider value is "0" again
        self.assertEqual(risk_slider.slider.value(), risk_slider_points.index(0))

    def test_target_class_change(self):
        self.send_signal(self.widget.Inputs.classifier, self.scoring_sheet_model)
        self.class_combo = self.widget.class_combo

        # Check if the values of the combobox "match" the domain
        self.assertEqual(
            self.class_combo.count(),
            len(self.scoring_sheet_model.domain.class_var.values),
        )
        for i in range(self.class_combo.count()):
            self.assertEqual(
                self.class_combo.itemText(i),
                self.scoring_sheet_model.domain.class_var.values[i],
            )

        old_coefficients = self.widget.coefficients.copy()
        old_all_scores = self.widget.all_scores.copy()
        old_all_risks = self.widget.all_risks.copy()

        # Change the target class to the second class
        self.class_combo.setCurrentIndex(1)
        self.widget._class_combo_changed()

        # Check if the coefficients, scores, and risks have changed
        self.assertNotEqual(old_coefficients, self.widget.coefficients)
        self.assertNotEqual(old_all_scores, self.widget.all_scores)
        self.assertNotEqual(old_all_risks, self.widget.all_risks)

    def test_invalid_classifier_error(self):
        self.send_signal(self.widget.Inputs.classifier, self.logistic_regression_model)
        self.assertTrue(self.widget.Error.invalid_classifier.is_shown())
        self.send_signal(self.widget.Inputs.classifier, self.scoring_sheet_model)
        self.assertFalse(self.widget.Error.invalid_classifier.is_shown())

    def test_multiple_instances_information(self):
        self.send_signal(self.widget.Inputs.data, self.heart[:2])
        self.assertTrue(self.widget.Information.multiple_instances.is_shown())
        self.send_signal(self.widget.Inputs.data, self.heart[:1])
        self.assertFalse(self.widget.Information.multiple_instances.is_shown())

    def _get_checkbox_states(self, coef_table):
        for row in range(coef_table.rowCount()):
            if self.widget.instance_points[row] == 1:
                self.assertEqual(coef_table.item(row, 2).checkState(), Qt.Checked)
            else:
                self.assertEqual(coef_table.item(row, 2).checkState(), Qt.Unchecked)

    def test_checkbox_after_instance_input(self):
        self.send_signal(self.widget.Inputs.classifier, self.scoring_sheet_model)
        self.send_signal(self.widget.Inputs.data, self.heart[:1])
        coef_table = self.widget.coefficient_table
        self._get_checkbox_states(coef_table)
        self.send_signal(self.widget.Inputs.data, self.heart[1:2])
        self._get_checkbox_states(coef_table)

    def test_no_classifier_UI(self):
        coef_table = self.widget.coefficient_table
        risk_slider = self.widget.risk_slider
        class_combo = self.widget.class_combo

        self.assertEqual(coef_table.rowCount(), 0)
        self.assertEqual(risk_slider.points, [])
        self.assertEqual(class_combo.count(), 0)

        self.send_signal(self.widget.Inputs.classifier, self.scoring_sheet_model)

        self.assertEqual(
            coef_table.rowCount(), self.scoring_sheet_learner.num_decision_params
        )
        self.assertIsNotNone(risk_slider.points)
        self.assertEqual(
            class_combo.count(), len(self.scoring_sheet_model.domain.class_var.values)
        )

        self.send_signal(self.widget.Inputs.classifier, None)

        self.assertEqual(coef_table.rowCount(), 0)
        self.assertEqual(risk_slider.points, [])
        self.assertEqual(class_combo.count(), 0)


if __name__ == "__main__":
    unittest.main()
