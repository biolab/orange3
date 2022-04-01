from unittest.mock import Mock, patch

import scipy.sparse as sp

# pylint: disable=missing-docstring, protected-access
from Orange.base import Learner, Model
from Orange.classification import KNNLearner
from Orange.data import Table, Domain
from Orange.modelling import TreeLearner, Fitter
from Orange.preprocess import continuize
from Orange.regression import MeanLearner, LinearRegressionLearner
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
        data = self.iris.transform(pp(self.iris)).copy()
        with data.unlocked():
            data.Y = sp.csr_matrix(data.Y)

        self.send_signal(w.Inputs.data, data, widget=w)
        self.assertFalse(any(w.Error.active))

        model = self.get_output(w.Outputs.model, widget=w)
        self.assertIsNotNone(model)

    def test_invalid_number_of_targets(self):
        class MockLearner(Fitter):
            name = 'mock'
            __fits__ = {'classification': Mock()}
            __returns__ = Mock()

        class WidgetLR(OWBaseLearner):
            name = "lr"
            LEARNER = MockLearner

        w = self.create_widget(WidgetLR)
        error = w.Error.data_error
        heart = Table("heart_disease")
        domain = heart.domain

        no_target = heart.transform(
            Domain(domain.attributes,
                   []))
        two_targets = heart.transform(
            Domain([domain["age"]],
                   [domain["gender"], domain["chest pain"]]))

        self.send_signal(w.Inputs.data, heart)
        self.assertFalse(error.is_shown())
        self.assertIs(w.data, heart)

        self.send_signal(w.Inputs.data, no_target)
        self.assertTrue(error.is_shown())
        self.assertIsNone(w.data)
        err_no_target = str(error)
        self.assertIn("target", err_no_target)

        self.send_signal(w.Inputs.data, two_targets)
        self.assertTrue(error.is_shown())
        self.assertIsNone(w.data)
        err_two_targets = str(error)
        self.assertIn("target", err_no_target)
        self.assertNotEqual(err_no_target, err_two_targets)

        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(w.data)
        self.assertFalse(error.is_shown())

        self.send_signal(w.Inputs.data, two_targets)
        self.assertTrue(error.is_shown())

        self.send_signal(w.Inputs.data, None)
        self.assertFalse(error.is_shown())

    def test_default_name(self):
        class TestLearner(Fitter):
            name = "Test"
            __returns__ = Mock()

        class TestWidget(OWBaseLearner):
            name = "Test"
            LEARNER = TestLearner

        def check_name(name):
            self.assertEqual(name, w.effective_learner_name())
            self.assertEqual(name, self.get_output(w.Outputs.learner, widget=w).name)

        w = self.create_widget(TestWidget)

        check_name("Test")
        w.setCaption("Foo")
        check_name("Foo")
        w.set_default_learner_name("Bar")
        check_name("Bar")
        w.setCaption("Frob")
        check_name("Bar")
        w.learner_name = "This is not a test"
        w.learner_name_changed()
        check_name("This is not a test")
        w.set_default_learner_name("Bar")
        check_name("This is not a test")
        w.setCaption("Blarg")
        check_name("This is not a test")
        w.learner_name = ""
        w.learner_name_changed()
        check_name("Bar")
        w.set_default_learner_name("")
        check_name("Blarg")

    def test_preprocessor_warning(self):
        class TestLearnerNoPreprocess(Learner):
            name = "Test"
            __returns__ = Mock()

        class TestWidgetNoPreprocess(OWBaseLearner):
            name = "Test"
            LEARNER = TestLearnerNoPreprocess

        class TestLearnerPreprocess(Learner):
            name = "Test"
            preprocessors = [Mock()]
            __returns__ = Mock()

        class TestWidgetPreprocess(OWBaseLearner):
            name = "Test"
            LEARNER = TestLearnerPreprocess

        class TestFitterPreprocess(Fitter):
            name = "Test"
            preprocessors = [Mock()]
            __returns__ = Mock()

        class TestWidgetPreprocessFit(OWBaseLearner):
            name = "Test"
            LEARNER = TestFitterPreprocess

        wno = self.create_widget(TestWidgetNoPreprocess)
        wyes = self.create_widget(TestWidgetPreprocess)
        wfit = self.create_widget(TestWidgetPreprocessFit)

        self.assertFalse(wno.Information.ignored_preprocessors.is_shown())
        self.assertFalse(wyes.Information.ignored_preprocessors.is_shown())
        self.assertFalse(wfit.Information.ignored_preprocessors.is_shown())

        pp = continuize.Continuize()
        self.send_signal(wno.Inputs.preprocessor, pp)
        self.send_signal(wyes.Inputs.preprocessor, pp)
        self.send_signal(wfit.Inputs.preprocessor, pp)

        self.assertFalse(wno.Information.ignored_preprocessors.is_shown())
        self.assertTrue(wyes.Information.ignored_preprocessors.is_shown())
        self.assertFalse(wfit.Information.ignored_preprocessors.is_shown())

        self.send_signal(wno.Inputs.preprocessor, None)
        self.send_signal(wyes.Inputs.preprocessor, None)
        self.send_signal(wfit.Inputs.preprocessor, None)

        self.assertFalse(wno.Information.ignored_preprocessors.is_shown())
        self.assertFalse(wyes.Information.ignored_preprocessors.is_shown())
        self.assertFalse(wfit.Information.ignored_preprocessors.is_shown())

    def test_multiple_sends(self):
        class TestLearner(Learner):
            name = "Test"
            __returns__ = Mock()

        class TestWidget(OWBaseLearner):
            name = "Test"
            LEARNER = TestLearner

        widget = self.create_widget(TestWidget)
        pp = continuize.Continuize()
        with patch.object(widget.Outputs.learner, "send") as model_send, \
                patch.object(widget.Outputs.model, "send") as learner_send:
            self.send_signals([(widget.Inputs.data, self.iris),
                               (widget.Inputs.preprocessor, pp)])
            learner_send.assert_called_once()
            model_send.assert_called_once()
