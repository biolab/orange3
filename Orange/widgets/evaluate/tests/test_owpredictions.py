"""Tests for OWPredictions"""
import io
import numpy as np

from Orange.data.io import TabReader
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.evaluate.owpredictions import OWPredictions

from Orange.data import Table, Domain
from Orange.classification import MajorityLearner, TreeLearner
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
        error = self.widget.Error

        titanic = Table("titanic")
        majority_titanic = MajorityLearner()(titanic)
        majority_iris = MajorityLearner()(self.iris)

        self.send_signal("Data", self.iris)
        self.send_signal("Predictors", majority_iris, 1)
        self.send_signal("Predictors", majority_titanic, 2)
        self.assertTrue(error.predictors_target_mismatch.is_shown())
        self.assertIsNone(self.get_output("Predictions"))

        self.send_signal("Predictors", None, 1)
        self.assertFalse(error.predictors_target_mismatch.is_shown())
        self.assertTrue(error.data_target_mismatch.is_shown())
        self.assertIsNone(self.get_output("Predictions"))

        self.send_signal("Data", None)
        self.assertFalse(error.predictors_target_mismatch.is_shown())
        self.assertFalse(error.data_target_mismatch.is_shown())
        self.assertIsNone(self.get_output("Predictions"))

        self.send_signal("Predictors", None, 2)
        self.assertFalse(error.predictors_target_mismatch.is_shown())
        self.assertFalse(error.data_target_mismatch.is_shown())
        self.assertIsNone(self.get_output("Predictions"))

        self.send_signal("Predictors", majority_titanic, 2)
        self.assertFalse(error.predictors_target_mismatch.is_shown())
        self.assertFalse(error.data_target_mismatch.is_shown())
        self.assertIsNone(self.get_output("Predictions"))

        self.send_signal("Data", self.iris)
        self.assertFalse(error.predictors_target_mismatch.is_shown())
        self.assertTrue(error.data_target_mismatch.is_shown())
        self.assertIsNone(self.get_output("Predictions"))

        self.send_signal("Predictors", majority_iris, 2)
        self.assertFalse(error.predictors_target_mismatch.is_shown())
        self.assertFalse(error.data_target_mismatch.is_shown())
        output = self.get_output("Predictions")
        self.assertEqual(len(output.domain.metas), 4)

        self.send_signal("Predictors", majority_iris, 1)
        self.send_signal("Predictors", majority_titanic, 3)
        self.assertTrue(error.predictors_target_mismatch.is_shown())
        self.assertFalse(error.data_target_mismatch.is_shown())
        self.assertIsNone(self.get_output("Predictions"))

    def test_no_class_on_test(self):
        """Allow test data with no class"""
        error = self.widget.Error

        titanic = Table("titanic")
        majority_titanic = MajorityLearner()(titanic)
        majority_iris = MajorityLearner()(self.iris)

        no_class = Table(Domain(titanic.domain.attributes, None), titanic)
        self.send_signal("Predictors", majority_titanic, 1)
        self.send_signal("Data", no_class)
        out = self.get_output("Predictions")
        np.testing.assert_allclose(out.get_column_view("majority")[0], 0)

        self.send_signal("Predictors", majority_iris, 2)
        self.assertTrue(error.predictors_target_mismatch.is_shown())
        self.assertFalse(error.data_target_mismatch.is_shown())
        self.assertIsNone(self.get_output("Predictions"))

        self.send_signal("Predictors", None, 2)
        self.send_signal("Data", titanic)
        out = self.get_output("Predictions")
        np.testing.assert_allclose(out.get_column_view("majority")[0], 0)

    def test_bad_data(self):
        """
        Firstly it creates predictions with TreeLearner. Then sends predictions and
        different data with different domain to Predictions widget. Those different
        data and domain are similar to original data and domain but they have three
        different target values instead of two.
        GH-2129
        """
        filestr1 = """\
        age\tsex\tsurvived
        d\td\td
        \t\tclass
        adult\tmale\tyes
        adult\tfemale\tno
        child\tmale\tyes
        child\tfemale\tyes
        """
        file1 = io.StringIO(filestr1)
        table = TabReader(file1).read()

        learner = TreeLearner()
        tree = learner(table)

        filestr2 = """\
        age\tsex\tsurvived
        d\td\td
        \t\tclass
        adult\tmale\tyes
        adult\tfemale\tno
        child\tmale\tyes
        child\tfemale\tunknown
        """
        file2 = io.StringIO(filestr2)
        bad_table = TabReader(file2).read()

        self.send_signal("Predictors", tree, 1)
        self.send_signal("Data", bad_table)
