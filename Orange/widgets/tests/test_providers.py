import unittest

from Orange.base import Learner, Model
from Orange.data import Table
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets.utils.owlearnerwidget import OWProvidesLearner, OWProvidesLearnerI, ProviderMetaClass
from Orange.widgets.widget import OWWidget


class TestProviderMetaClass(unittest.TestCase):

    def assertChanelsEqual(self, first, second, msg=None):
        self.assertEqual(len(first), len(second), msg)
        self.assertTrue(all(i1.name == i2[0] for i1, i2 in zip(first, second)), msg)

    def test_inputs(self):
        # backward compatibility
        inputs = [("Data", Table, "set_data"),
                  ("Preprocessor", Preprocess, "set_preprocessor")]

        class OWTestProvider(OWProvidesLearner, OWWidget):
            inputs = [("Data", Table, "set_data")] + OWProvidesLearner.inputs

        self.assertEqual(OWTestProvider.inputs, inputs)

        class OWTestProvider(OWProvidesLearner, OWWidget):
            inputs = [("Data", Table, "set_data")] + OWProvidesLearner.inputs
            name = "real widget"

        self.assertChanelsEqual(OWTestProvider.inputs, inputs)

        # new way
        class OWTestProvider(OWProvidesLearnerI):
            LEARNER = Learner
            OUTPUT_MODEL_CLASS = Model
            OUTPUT_MODEL_NAME = "Test learner"

            name = "test widget"

        self.assertChanelsEqual(OWTestProvider.inputs, inputs)

    def test_outputs(self):
        expected_outputs = [
            ("Learner", Learner),
            ("TestModel", Model),
            ("Test", Table)
        ]

        class OWTestProvider(OWProvidesLearner, OWWidget):
            outputs = expected_outputs

        self.assertEqual(OWTestProvider.outputs, expected_outputs)

        class OWTestProvider(OWProvidesLearner, OWWidget):
            outputs = expected_outputs
            name = "real widget"

        self.assertChanelsEqual(OWTestProvider.outputs, expected_outputs)

        class OWTestProvider(OWProvidesLearnerI):
            LEARNER = Learner
            OUTPUT_MODEL_CLASS = Model
            OUTPUT_MODEL_NAME = "TestModel"
            name = "test widget"

            extra_outputs = [("Test", Table)]

        self.assertChanelsEqual(OWTestProvider.outputs, expected_outputs)

    def test_class_without_attributes(self):
        with self.assertRaises(AttributeError):
            class OWTestProvider(OWProvidesLearnerI):
                OUTPUT_MODEL_CLASS = Model
                OUTPUT_MODEL_NAME = "Test learner"
                name = 'test'

        with self.assertRaises(AttributeError):
            class OWTestProvider(OWProvidesLearnerI):
                LEARNER = Learner
                OUTPUT_MODEL_NAME = "Test learner"
                name = 'test'

        with self.assertRaises(AttributeError):
            class OWTestProvider(OWProvidesLearnerI):
                LEARNER = Learner
                OUTPUT_MODEL_CLASS = Model
                name = 'test'
