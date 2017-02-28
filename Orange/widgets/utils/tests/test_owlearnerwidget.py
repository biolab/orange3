# pylint: disable=missing-docstring
from Orange.base import Learner, Model
from Orange.data import Table, Domain
from Orange.preprocess import Normalize
from Orange.widgets.model.owlogisticregression import OWLogisticRegression
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.tests.base import WidgetTest


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

    def test_set_preprocessor(self):
        w = self.create_widget(OWLogisticRegression)  # type: OWLogisticRegression
        assert isinstance(w, OWBaseLearner)

        for preprocessor in (Normalize(),
                             [Normalize()]):
            with self.subTest(preprocessor=preprocessor):
                w.set_preprocessor(preprocessor)
                learner = w.create_learner()
                learner.fit_storage(self.iris)
