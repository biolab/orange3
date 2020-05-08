import scipy.sparse as sp

# pylint: disable=missing-docstring, protected-access
from Orange.base import Learner, Model
from Orange.classification import KNNLearner
from Orange.data import Table, Domain
from Orange.modelling import TreeLearner
from Orange.preprocess import continuize
from Orange.regression import MeanLearner, LinearRegressionLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.signals import Output
from Orange.widgets.utils.state_summary import format_summary_details


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
        w1.learner_name = "MyWidget"

        settings = w1.settingsHandler.pack_data(w1)
        w2 = self.create_widget(WidgetA, settings)
        self.assertEqual(w2.learner_name, w1.learner_name)

    def test_converts_sparse_targets_to_dense(self):
        class WidgetLR(OWBaseLearner):
            name = "lr"
            LEARNER = LinearRegressionLearner

        w = self.create_widget(WidgetLR)

        # Orange will want do do one-hot encoding when continuizing discrete variable
        pp = continuize.DomainContinuizer(
            multinomial_treatment=continuize.Continuize.AsOrdinal,
            transform_class=True,
        )
        data = self.iris.transform(pp(self.iris))
        data.Y = sp.csr_matrix(data.Y)

        self.send_signal(w.Inputs.data, data, widget=w)
        self.assertFalse(any(w.Error.active))

        model = self.get_output(w.Outputs.model, widget=w)
        self.assertIsNotNone(model)

    def test_summary(self):
        """Check if the status bar is updated when data is received"""
        class WidgetA(OWBaseLearner):
            name = "A"
            LEARNER = KNNLearner
        widget = self.create_widget(WidgetA)
        info = widget.info
        no_input = "No data on input"

        self.send_signal(widget.Inputs.data, self.iris)
        summary, details = "150", format_summary_details(self.iris)
        self.assertEqual(info._StateInfo__input_summary.brief, summary)
        self.assertEqual(info._StateInfo__input_summary.details, details)
        self.send_signal(widget.Inputs.data, None)
        self.assertEqual(info._StateInfo__input_summary.brief, "")
        self.assertEqual(info._StateInfo__input_summary.details, no_input)
