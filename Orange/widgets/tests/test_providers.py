import unittest

from Orange.classification import KNNLearner
from Orange.data import Table, TableBase
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class TestProviderMetaClass(unittest.TestCase):

    def assertChannelsEqual(self, first, second, msg=None):
        self.assertEqual(len(first), len(second), msg)
        self.assertEqual(set(i.name for i in first),
                         set(i[0] for i in second), msg)

    def test_inputs(self):
        inputs = [("Data", TableBase, "set_data"),
                  ("Preprocessor", Preprocess, "set_preprocessor")]

        class OWTestProvider(OWBaseLearner):
            LEARNER = KNNLearner
            name = "test widget"

        self.assertChannelsEqual(OWTestProvider.inputs, inputs)

    def test_outputs(self):
        expected_outputs = [
            ("Learner", KNNLearner),
            ("Classifier", KNNLearner.__returns__),
            ("Test", Table)
        ]

        class OWTestProvider(OWBaseLearner):
            LEARNER = KNNLearner
            name = "test widget"

            outputs = [("Test", Table)]

        self.assertChannelsEqual(OWTestProvider.outputs, expected_outputs)

    def test_class_without_learner(self):
        with self.assertRaises(AttributeError):
            class OWTestProvider(OWBaseLearner):
                name = 'test'
