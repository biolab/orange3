# pylint: disable=missing-docstring
import numpy as np

import unittest

from Orange.data import Table, Domain
from Orange.classification import MajorityLearner
from Orange.regression import MeanLearner
from Orange.modelling import ConstantLearner

from Orange.evaluation import Results, TestOnTestData
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.evaluate.owtestlearners import OWTestLearners
from Orange.widgets.evaluate import owtestlearners


class TestOWTestLearners(WidgetTest):
    def setUp(self):
        super().setUp()
        self.widget = self.create_widget(OWTestLearners)  # type: OWTestLearners

    def test_basic(self):
        data = Table("iris")[::3]
        self.send_signal("Data", data)
        self.send_signal("Learner", MajorityLearner(), 0)
        res = self.get_output("Evaluation Results")
        self.assertIsInstance(res, Results)
        self.assertIsNotNone(res.domain)
        self.assertIsNotNone(res.data)
        self.assertIsNotNone(res.probabilities)

        self.send_signal("Learner", None, 0)

        data = Table("housing")[::10]
        self.send_signal("Data", data)
        self.send_signal("Learner", MeanLearner(), 0)
        res = self.get_output("Evaluation Results")
        self.assertIsInstance(res, Results)
        self.assertIsNotNone(res.domain)
        self.assertIsNotNone(res.data)

    def test_testOnTest(self):
        data = Table("iris")
        self.send_signal("Data", data)
        self.widget.resampling = OWTestLearners.TestOnTest
        self.send_signal("Test Data", data)

    def test_CrossValidationByFeature(self):
        data = Table("iris")
        attrs = data.domain.attributes
        domain = Domain(attrs[:-1], attrs[-1], data.domain.class_vars)
        data_with_disc_metas = Table.from_table(domain, data)
        rb = self.widget.controls.resampling.buttons[OWTestLearners.FeatureFold]

        self.send_signal("Learner", ConstantLearner(), 0)
        self.send_signal("Data", data)
        self.assertFalse(rb.isEnabled())
        self.assertFalse(self.widget.features_combo.isEnabled())

        self.send_signal("Data", data_with_disc_metas)
        self.assertTrue(rb.isEnabled())
        rb.click()
        self.assertEqual(self.widget.resampling, OWTestLearners.FeatureFold)
        self.assertTrue(self.widget.features_combo.isEnabled())
        self.assertEqual(self.widget.features_combo.currentText(), "iris")
        self.assertEqual(len(self.widget.features_combo.model()), 1)

        self.send_signal("Data", None)
        self.assertFalse(rb.isEnabled())
        self.assertEqual(self.widget.resampling, OWTestLearners.KFold)
        self.assertFalse(self.widget.features_combo.isEnabled())


class TestHelpers(unittest.TestCase):
    def test_results_one_vs_rest(self):
        data = Table("lenses")
        learners = [MajorityLearner()]
        res = TestOnTestData(data[1::2], data[::2], learners=learners)
        r1 = owtestlearners.results_one_vs_rest(res, pos_index=0)
        r2 = owtestlearners.results_one_vs_rest(res, pos_index=1)
        r3 = owtestlearners.results_one_vs_rest(res, pos_index=2)

        np.testing.assert_almost_equal(np.sum(r1.probabilities, axis=2), 1.0)
        np.testing.assert_almost_equal(np.sum(r2.probabilities, axis=2), 1.0)
        np.testing.assert_almost_equal(np.sum(r3.probabilities, axis=2), 1.0)

        np.testing.assert_almost_equal(
            r1.probabilities[:, :, 1] +
            r2.probabilities[:, :, 1] +
            r3.probabilities[:, :, 1],
            1.0
        )
        self.assertEqual(r1.folds, res.folds)
        self.assertEqual(r2.folds, res.folds)
        self.assertEqual(r3.folds, res.folds)

        np.testing.assert_equal(r1.row_indices, res.row_indices)
        np.testing.assert_equal(r2.row_indices, res.row_indices)
        np.testing.assert_equal(r3.row_indices, res.row_indices)
