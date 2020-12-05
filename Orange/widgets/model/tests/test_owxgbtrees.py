import unittest

from Orange.preprocess.score import Scorer
from Orange.widgets.model.owxgbtrees import OWXGBTrees
from Orange.widgets.tests.base import WidgetTest, ParameterMapping, \
    WidgetLearnerTestMixin, datasets


class TestOWXGBTrees(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWXGBTrees,
                                         stored_settings={"auto_apply": False})
        self.init()

        controls = self.widget.controls
        self.parameters = [
            ParameterMapping("n_estimators", controls.n_estimators, [100, 10]),
            ParameterMapping("max_depth", controls.max_depth),
            ParameterMapping("learning_rate", controls.learning_rate),
            ParameterMapping("gamma", controls.gamma),
            ParameterMapping("subsample", controls.subsample),
            ParameterMapping("colsample_bytree", controls.colsample_bytree),
            ParameterMapping("colsample_bylevel", controls.colsample_bylevel),
            ParameterMapping("colsample_bynode", controls.colsample_bynode),
            ParameterMapping("reg_alpha", controls.reg_alpha),
            ParameterMapping("reg_lambda", controls.reg_lambda),
        ]

    def test_scorer(self):
        self.assertIsInstance(
            self.get_output(self.widget.Outputs.learner), Scorer
        )

    def test_datasets(self):
        for ds in datasets.datasets():
            self.send_signal(self.widget.Inputs.data, ds)


if __name__ == "__main__":
    unittest.main()
