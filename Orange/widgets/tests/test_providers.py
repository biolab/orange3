import unittest

from Orange.classification import KNNLearner
from Orange.modelling import TreeLearner
from Orange.regression import MeanLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.signals import Output


class TestProviderMetaClass(unittest.TestCase):

    def assertChannelsEqual(self, first, second, msg=None):
        self.assertEqual(len(first), len(second), msg)
        self.assertEqual(set(i.name for i in first),
                         set(i[0] for i in second), msg)

    def test_widgets_do_not_share_outputs(self):
        class WidgetA(OWBaseLearner):
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
