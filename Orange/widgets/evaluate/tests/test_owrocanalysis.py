import unittest
import numpy

import Orange.data
import Orange.evaluation
import Orange.classification

from Orange.widgets.evaluate import owrocanalysis


class TestROC(unittest.TestCase):
    def test_ROCData_from_results(self):
        data = Orange.data.Table("iris")
        learners = [
            Orange.classification.MajorityLearner(),
            Orange.classification.LogisticRegressionLearner(),
            Orange.classification.TreeLearner()
        ]
        res = Orange.evaluation.CrossValidation(data, learners, k=10)

        for i, _ in enumerate(learners):
            for c in range(len(data.domain.class_var.values)):
                rocdata = owrocanalysis.ROCData_from_results(res, i, target=c)
                self.assertTrue(rocdata.merged.is_valid)
                self.assertEqual(len(rocdata.folds), 10)
                self.assertTrue(all(c.is_valid for c in rocdata.folds))
                self.assertTrue(rocdata.avg_vertical.is_valid)
                self.assertTrue(rocdata.avg_threshold.is_valid)

        data = data[numpy.random.choice(len(data), size=20)]
        res = Orange.evaluation.LeaveOneOut(data, learners)

        for i, _ in enumerate(learners):
            for c in range(len(data.domain.class_var.values)):
                rocdata = owrocanalysis.ROCData_from_results(res, i, target=c)
                self.assertTrue(rocdata.merged.is_valid)
                self.assertEqual(len(rocdata.folds), 20)
                # all individual fold curves and averaged curve data
                # should be invalid
                self.assertTrue(all(not c.is_valid for c in rocdata.folds))
                self.assertFalse(rocdata.avg_vertical.is_valid)
                self.assertFalse(rocdata.avg_threshold.is_valid)

        # equivalent test to the LeaveOneOut but from a slightly different
        # constructed Orange.evaluation.Results
        res = Orange.evaluation.CrossValidation(data, learners, k=20)

        for i, _ in enumerate(learners):
            for c in range(len(data.domain.class_var.values)):
                rocdata = owrocanalysis.ROCData_from_results(res, i, target=c)
                self.assertTrue(rocdata.merged.is_valid)
                self.assertEqual(len(rocdata.folds), 20)
                # all individual fold curves and averaged curve data
                # should be invalid
                self.assertTrue(all(not c.is_valid for c in rocdata.folds))
                self.assertFalse(rocdata.avg_vertical.is_valid)
                self.assertFalse(rocdata.avg_threshold.is_valid)
