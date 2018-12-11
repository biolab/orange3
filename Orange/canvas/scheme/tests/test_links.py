"""
Tests for SchemeLink
"""

from ...gui import test
from ...registry.tests import small_testing_registry

from .. import SchemeNode, SchemeLink, IncompatibleChannelTypeError


class TestSchemeLink(test.QAppTestCase):
    def test_link(self):
        import Orange
        reg = small_testing_registry()
        base = "Orange.widgets"
        file_desc = reg.widget(base + ".data.owfile.OWFile")
        discretize_desc = reg.widget(base + ".data.owdiscretize.OWDiscretize")
        bayes_desc = reg.widget(base + ".classify.ownaivebayes.OWNaiveBayes")

        file_node = SchemeNode(file_desc)
        discretize_node = SchemeNode(discretize_desc)
        bayes_node = SchemeNode(bayes_desc)

        link1 = SchemeLink(file_node, file_node.output_channel("Data"),
                           discretize_node,
                           discretize_node.input_channel("Data"))

        self.assertTrue(link1.source_type() is Orange.data.Table)
        self.assertTrue(link1.sink_type() is Orange.data.Table)

        with self.assertRaises(ValueError):
            SchemeLink(discretize_node, "Data",
                       file_node, "$$$[")

        with self.assertRaises(IncompatibleChannelTypeError):
            SchemeLink(bayes_node, "Learner", discretize_node, "Data")
