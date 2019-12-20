"""Tests for OWPredictions"""
import io
from unittest.mock import Mock

import numpy as np

from Orange.data.io import TabReader
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.evaluate.owpredictions import OWPredictions
from Orange.widgets.evaluate.owcalibrationplot import OWCalibrationPlot
from Orange.widgets.evaluate.owconfusionmatrix import OWConfusionMatrix
from Orange.widgets.evaluate.owliftcurve import OWLiftCurve
from Orange.widgets.evaluate.owrocanalysis import OWROCAnalysis

from Orange.data import Table, Domain, DiscreteVariable
from Orange.modelling import ConstantLearner, TreeLearner
from Orange.evaluation import Results
from Orange.widgets.tests.utils import excepthook_catch


class TestOWPredictions(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWPredictions)  # type: OWPredictions
        self.iris = Table("iris")

    def test_rowCount_from_model(self):
        """Don't crash if the bottom row is visible"""
        self.send_signal(self.widget.Inputs.data, self.iris[:5])
        self.widget.dataview.sizeHintForColumn(0)

    def test_nan_target_input(self):
        data = self.iris[::10].copy()
        data.Y[1] = np.nan
        yvec, _ = data.get_column_view(data.domain.class_var)
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.predictors, ConstantLearner()(data), 1)
        pred = self.get_output(self.widget.Outputs.predictions)
        self.assertIsInstance(pred, Table)
        np.testing.assert_array_equal(
            yvec, pred.get_column_view(data.domain.class_var)[0])

        evres = self.get_output(self.widget.Outputs.evaluation_results)
        self.assertIsInstance(evres, Results)
        self.assertIsInstance(evres.data, Table)
        ev_yvec, _ = evres.data.get_column_view(data.domain.class_var)

        self.assertTrue(np.all(~np.isnan(ev_yvec)))
        self.assertTrue(np.all(~np.isnan(evres.actual)))

        data.Y[:] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        evres = self.get_output(self.widget.Outputs.evaluation_results)
        self.assertEqual(len(evres.data), 0)

    def test_no_values_target(self):
        train = Table("titanic")
        model = ConstantLearner()(train)
        self.send_signal(self.widget.Inputs.predictors, model)
        domain = Domain([DiscreteVariable("status", values=["first", "third"]),
                         DiscreteVariable("age", values=["adult", "child"]),
                         DiscreteVariable("sex", values=["female", "male"])],
                        [DiscreteVariable("survived", values=[])])
        test = Table(domain, np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
                     np.full((3, 1), np.nan))
        self.send_signal(self.widget.Inputs.data, test)
        pred = self.get_output(self.widget.Outputs.predictions)
        self.assertEqual(len(pred), len(test))

        results = self.get_output(self.widget.Outputs.evaluation_results)

        cm_widget = self.create_widget(OWConfusionMatrix)
        self.send_signal(cm_widget.Inputs.evaluation_results, results,
                         widget=cm_widget)

        ra_widget = self.create_widget(OWROCAnalysis)
        self.send_signal(ra_widget.Inputs.evaluation_results, results,
                         widget=ra_widget)

        lc_widget = self.create_widget(OWLiftCurve)
        self.send_signal(lc_widget.Inputs.evaluation_results, results,
                         widget=lc_widget)

        cp_widget = self.create_widget(OWCalibrationPlot)
        self.send_signal(cp_widget.Inputs.evaluation_results, results,
                         widget=cp_widget)

    def test_mismatching_targets(self):
        warning = self.widget.Warning

        maj_iris = ConstantLearner()(self.iris)
        dom = self.iris.domain
        iris3 = self.iris.transform(Domain(dom[:3], dom[3]))
        maj_iris3 = ConstantLearner()(iris3)

        self.send_signal(self.widget.Inputs.predictors, maj_iris, 1)
        self.send_signal(self.widget.Inputs.predictors, maj_iris3, 2)
        self.assertFalse(warning.wrong_targets.is_shown())

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertTrue(warning.wrong_targets.is_shown())

        self.send_signal(self.widget.Inputs.predictors, None, 2)
        self.assertFalse(warning.wrong_targets.is_shown())

        self.send_signal(self.widget.Inputs.predictors, maj_iris3, 2)
        self.assertTrue(warning.wrong_targets.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(warning.wrong_targets.is_shown())

    def test_no_class_on_test(self):
        """Allow test data with no class"""
        titanic = Table("titanic")
        majority_titanic = ConstantLearner()(titanic)

        no_class = titanic.transform(Domain(titanic.domain.attributes, None))
        self.send_signal(self.widget.Inputs.predictors, majority_titanic, 1)
        self.send_signal(self.widget.Inputs.data, no_class)
        out = self.get_output(self.widget.Outputs.predictions)
        np.testing.assert_allclose(out.get_column_view("constant")[0], 0)

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

        self.send_signal(self.widget.Inputs.predictors, tree, 1)

        with excepthook_catch():
            self.send_signal(self.widget.Inputs.data, bad_table)

    def test_continuous_class(self):
        data = Table("housing")
        cl_data = ConstantLearner()(data)
        self.send_signal(self.widget.Inputs.predictors, cl_data, 1)
        self.send_signal(self.widget.Inputs.data, data)

    def test_predictor_fails(self):
        titanic = Table("titanic")
        failing_model = ConstantLearner()(titanic)
        failing_model.predict = Mock(side_effect=ValueError("foo"))
        self.send_signal(self.widget.Inputs.predictors, failing_model, 1)
        self.send_signal(self.widget.Inputs.data, titanic)

        model2 = ConstantLearner()(titanic)
        self.send_signal(self.widget.Inputs.predictors, model2, 2)


if __name__ == "__main__":
    import unittest
    unittest.main()
