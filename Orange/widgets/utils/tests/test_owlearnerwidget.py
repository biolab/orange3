# pylint: disable=missing-docstring
from Orange.base import Learner, Model
from Orange.classification import KNNLearner
from Orange.data import Table, Domain
from Orange.modelling import TreeLearner
from Orange.regression import MeanLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.signals import Output


class TestOWBaseLearner(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")

    def test_error_on_learning(self):
        """Check that widget shows error message when learner fails"""

        class FailingLearner(Learner):
            """A learner that fails when given data"""
            __returns__ = Model

            def __call__(self, data, *_):
                if data is not None:
                    raise ValueError("boom")
                return Model(Domain([]))

        class OWFailingLearner(OWBaseLearner):
            """Widget for the above learner"""
            name = learner_name = "foo"
            LEARNER = FailingLearner
            auto_apply = True

        self.widget = self.create_widget(OWFailingLearner)
        self.send_signal("Data", self.iris)
        self.assertTrue(self.widget.Error.fitting_failed.is_shown())
        self.send_signal("Data", None)
        self.assertFalse(self.widget.Error.fitting_failed.is_shown())

    def test_subclasses_do_not_share_outputs(self):
        class WidgetA(OWBaseLearner, openclass=True):
            name = "A"
            LEARNER = KNNLearner

        class WidgetB(OWBaseLearner):
            name = "B"
            LEARNER = MeanLearner

        self.assertEqual(WidgetA.Outputs.learner.type, KNNLearner)
        self.assertEqual(WidgetB.Outputs.learner.type, MeanLearner)

        class WidgetC(WidgetA):
            name = "C"
            LEARNER = TreeLearner

            class Outputs(WidgetA.Outputs):
                test = Output("test", str)

        self.assertEqual(WidgetC.Outputs.learner.type, TreeLearner)
        self.assertEqual(WidgetC.Outputs.test.name, "test")
        self.assertEqual(WidgetA.Outputs.learner.type, KNNLearner)
        self.assertFalse(hasattr(WidgetA.Outputs, "test"))

    def test_send_backward_compatibility(self):
        class WidgetA(OWBaseLearner):
            name = "A"
            LEARNER = KNNLearner

        w = self.create_widget(WidgetA)
        w.send(w.OUTPUT_MODEL_NAME, "Foo")
        self.assertEqual(self.get_output(w.OUTPUT_MODEL_NAME, w), "Foo")

        # Old old signal name
        w.send("Predictor", "Bar")
        self.assertEqual(self.get_output(w.OUTPUT_MODEL_NAME, w), "Bar")

    def test_old_style_signals_on_subclass_backward_compatibility(self):
        class WidgetA(OWBaseLearner):
            name = "A"
            LEARNER = KNNLearner

            inputs = [("A", None, "set_data")]
            outputs = [("A", None)]

        desc = WidgetA.get_widget_description()
        inputs = [i.name for i in desc["inputs"]]
        outputs = [o.name for o in desc["outputs"]]

        self.assertIn(WidgetA.Outputs.learner.name, outputs)
        self.assertIn(WidgetA.Outputs.model.name, outputs)
        self.assertIn("A", outputs)

        self.assertIn(WidgetA.Inputs.data.name, inputs)
        self.assertIn(WidgetA.Inputs.preprocessor.name, inputs)
        self.assertIn("A", inputs)

    def test_persists_learner_name_in_settings(self):
        class WidgetA(OWBaseLearner):
            name = "A"
            LEARNER = KNNLearner

        w1 = self.create_widget(WidgetA)
        self.assertEqual(w1.learner_name, "A")
        w1.learner_name = "MyWidget"

        settings = w1.settingsHandler.pack_data(w1)
        w2 = self.create_widget(WidgetA, settings)
        self.assertEqual(w2.learner_name, w1.learner_name)
