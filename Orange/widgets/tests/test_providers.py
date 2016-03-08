import unittest

from Orange.classification import KNNLearner
from Orange.data import Table
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class TestProviderMetaClass(unittest.TestCase):

    def assertChannelsEqual(self, first, second, msg=None):
        self.assertEqual(len(first), len(second), msg)
        self.assertEqual(set(i.name for i in first),
                         set(i[0] for i in second), msg)

    def test_inputs(self):
        inputs = [("Data", Table, "set_data"),
                  ("Preprocessor", Preprocess, "set_preprocessor")]

        class OWTestProvider(OWBaseLearner):
            LEARNER = KNNLearner
            OUTPUT_MODEL_NAME = "Test learner"

            name = "test widget"

        self.assertChannelsEqual(OWTestProvider.inputs, inputs)

    def test_outputs(self):
        expected_outputs = [
            ("Learner", KNNLearner),
            ("TestModel", KNNLearner.__returns__),
            ("Test", Table)
        ]

        class OWTestProvider(OWBaseLearner):
            LEARNER = KNNLearner
            OUTPUT_MODEL_NAME = "TestModel"
            name = "test widget"

            outputs = [("Test", Table)]

        self.assertChannelsEqual(OWTestProvider.outputs, expected_outputs)

    def test_class_without_attributes(self):
        with self.assertRaises(AttributeError):
            class OWTestProvider(OWBaseLearner):
                OUTPUT_MODEL_NAME = "Test learner"
                name = 'test'

        with self.assertRaises(AttributeError):
            class OWTestProvider(OWBaseLearner):
                LEARNER = KNNLearner
                name = 'test'
