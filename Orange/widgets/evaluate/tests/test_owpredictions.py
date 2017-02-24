"""Tests for OWPredictions"""

import numpy as np

from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.evaluate.owpredictions import OWPredictions

from Orange.data import Table
from Orange.classification import MajorityLearner
from Orange.evaluation import Results


class TestOWPredictions(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWPredictions)  # type: OWPredictions
        self.iris = Table("iris")

    def test_rowCount_from_model(self):
        """Don't crash if the bottom row is visible"""
        self.send_signal("Data", self.iris[:5])
        self.widget.dataview.sizeHintForColumn(0)

    def test_nan_target_input(self):
        data = self.iris[::10].copy()
        data.Y[1] = np.nan
        yvec, _ = data.get_column_view(data.domain.class_var)
        nanmask = np.isnan(yvec)
        self.send_signal("Data", data)
        self.send_signal("Predictors", MajorityLearner()(data), 1)
        pred = self.get_output("Predictions", )
        self.assertIsInstance(pred, Table)
        np.testing.assert_array_equal(
            yvec, pred.get_column_view(data.domain.class_var)[0])

        evres = self.get_output("Evaluation Results")
        self.assertIsInstance(evres, Results)
        self.assertIsInstance(evres.data, Table)
        ev_yvec, _ = evres.data.get_column_view(data.domain.class_var)

        self.assertTrue(np.all(~np.isnan(ev_yvec)))
        self.assertTrue(np.all(~np.isnan(evres.actual)))

        data.Y[:] = np.nan
        self.send_signal("Data", data)
        evres = self.get_output("Evaluation Results")
        self.assertEqual(len(evres.data), 0)

    def test_mismatching_targets(self):
        titanic = Table("titanic")
        majority_titanic = MajorityLearner()(titanic)
        majority_iris = MajorityLearner()(self.iris)

        self.send_signal("Data", self.iris)
        self.send_signal("Predictors", majority_iris, 1)
        self.send_signal("Predictors", majority_titanic, 2)
        self.assertTrue(self.widget.Error.predictor_failed.is_shown())
        output = self.get_output("Predictions")
        self.assertEqual(len(output.domain.metas), 4)

        self.send_signal("Predictors", None, 1)
        self.assertTrue(self.widget.Error.predictor_failed.is_shown())
        self.assertIsNone(self.get_output("Predictions"))

        self.send_signal("Data", None)
        self.assertFalse(self.widget.Error.predictor_failed.is_shown())
        self.assertIsNone(self.get_output("Predictions"))

        self.send_signal("Predictors", None, 2)
        self.assertFalse(self.widget.Error.predictor_failed.is_shown())
        self.assertIsNone(self.get_output("Predictions"))

        self.send_signal("Predictors", majority_titanic, 2)
        self.assertFalse(self.widget.Error.predictor_failed.is_shown())
        self.assertIsNone(self.get_output("Predictions"))

        self.send_signal("Data", self.iris)
        self.assertTrue(self.widget.Error.predictor_failed.is_shown())
        self.assertIsNone(self.get_output("Predictions"))

        self.send_signal("Predictors", majority_iris, 2)
        self.assertFalse(self.widget.Error.predictor_failed.is_shown())
        self.assertEqual(len(output.domain.metas), 4)

        self.send_signal("Predictors", majority_iris, 1)
        self.send_signal("Predictors", majority_titanic, 3)
        output = self.get_output("Predictions")
        self.assertEqual(len(output.domain.metas), 8)
