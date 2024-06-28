# pylint: disable=missing-docstring,protected-access
import unittest

from Orange.classification import LogisticRegressionLearner, NaiveBayesLearner
from Orange.data import Table, Domain
from Orange.ensembles import StackedFitter
from Orange.regression import LinearRegressionLearner
from Orange.widgets.evaluate.owpermutationplot import OWPermutationPlot
from Orange.widgets.tests.base import WidgetTest


class TestOWPermutationPlot(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.heart = Table("heart_disease")
        cls.housing = Table("housing")
        cls.naive_bayes = NaiveBayesLearner()
        cls.lin_reg = LinearRegressionLearner()

    def setUp(self):
        self.widget = self.create_widget(OWPermutationPlot,
                                         stored_settings={"n_permutations": 3})

    def test_input_disc_target(self):
        self.send_signal(self.widget.Inputs.data, self.heart)
        self.send_signal(self.widget.Inputs.learner, self.naive_bayes)
        self.wait_until_finished()

        lin_reg = LinearRegressionLearner()
        self.send_signal(self.widget.Inputs.learner, lin_reg)
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.incompatible_learner.is_shown())

        self.send_signal(self.widget.Inputs.learner, None)
        self.assertFalse(self.widget.Error.incompatible_learner.is_shown())

    def test_input_cont_target(self):
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.send_signal(self.widget.Inputs.learner, self.lin_reg)
        self.wait_until_finished()

        log_reg = LogisticRegressionLearner()
        self.send_signal(self.widget.Inputs.learner, log_reg)
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.incompatible_learner.is_shown())

        self.send_signal(self.widget.Inputs.learner, None)
        self.assertFalse(self.widget.Error.unknown_err.is_shown())

    def test_input_cont_target_ensemble(self):
        self.send_signal(self.widget.Inputs.data, self.housing)

        learner = StackedFitter([LinearRegressionLearner()])
        self.send_signal(self.widget.Inputs.learner, learner)
        self.wait_until_finished()

        learner = StackedFitter([LogisticRegressionLearner()])
        self.send_signal(self.widget.Inputs.learner, learner)
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.unknown_err.is_shown())

    def test_input_no_target(self):
        domain = Domain(self.housing.domain.attributes)
        data = self.housing.transform(domain)
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.learner, self.lin_reg)
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.incompatible_learner.is_shown())

    def test_input_multi_target(self):
        domain = Domain(self.housing.domain.attributes[:-2],
                        self.housing.domain.attributes[-2:])
        data = self.housing.transform(domain)
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.learner, self.lin_reg)
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.multiple_targets_data.is_shown())

    def test_sample_data(self):
        self.send_signal(self.widget.Inputs.learner, self.naive_bayes)
        self.send_signal(self.widget.Inputs.data, self.heart[:6])
        self.assertTrue(self.widget.Error.not_enough_data.is_shown())
        self.send_signal(self.widget.Inputs.data, self.heart[:7])
        self.assertFalse(self.widget.Error.not_enough_data.is_shown())
        self.wait_until_finished()

    def test_info(self):
        self.send_signal(self.widget.Inputs.learner, self.naive_bayes)
        self.send_signal(self.widget.Inputs.data, self.heart)
        self.wait_until_finished()
        self.assertIn('<th style="padding: 2px 4px" align=right>CV</th>',
                      self.widget._info.text())
        self.assertIn('<th style="padding: 2px 4px" align=right>Train</th>',
                      self.widget._info.text())

        text = """<th style="padding: 2px 4px" align=right>Train</th>
        <td style="padding: 2px 4px" align=right>0.6686</td>
        <td style="padding: 2px 4px" align=right>0.9200</td>"""
        self.assertIn(text, self.widget._info.text())

        text = """<th style="padding: 2px 4px" align=right>CV</th>
        <td style="padding: 2px 4px" align=right>0.5292</td>
        <td style="padding: 2px 4px" align=right>0.9076</td>"""
        self.assertIn(text, self.widget._info.text())

        self.send_signal(self.widget.Inputs.learner, None)
        self.assertEqual(self.widget._info.text(), "No data available.")

    def test_send_report(self):
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, self.heart[:10])
        self.send_signal(self.widget.Inputs.learner, self.naive_bayes)
        self.wait_until_finished()
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, self.housing[:10])
        self.send_signal(self.widget.Inputs.learner, self.lin_reg)
        self.wait_until_finished()
        self.send_signal(self.widget.Inputs.data, None)
        self.send_signal(self.widget.Inputs.learner, self.lin_reg)
        self.widget.send_report()


if __name__ == "__main__":
    unittest.main()
