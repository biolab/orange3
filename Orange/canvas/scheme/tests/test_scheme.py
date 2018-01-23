"""
Tests for Scheme
"""

from ...gui import test
from ...registry.tests import small_testing_registry
from ...registry import WidgetRegistry

from .. import (
    Scheme, SchemeNode, SchemeLink, SchemeTextAnnotation,
    SchemeTopologyError, SinkChannelError,
    DuplicatedLinkError, IncompatibleChannelTypeError, SchemeCycleError
)

from ...registry.description import WidgetDescription, CategoryDescription
from ...registry.description import InputSignal, OutputSignal
import unittest


class TestScheme(test.QCoreAppTestCase):
    def test_scheme(self):
        reg = small_testing_registry()
        base = "Orange.widgets"
        file_desc = reg.widget(base + ".data.owfile.OWFile")
        discretize_desc = reg.widget(base + ".data.owdiscretize.OWDiscretize")
        bayes_desc = reg.widget(base + ".classify.ownaivebayes.OWNaiveBayes")
        # Create the scheme
        scheme = Scheme()

        self.assertEqual(scheme.title, "")
        self.assertEqual(scheme.description, "")

        nodes_added = []
        links_added = []
        annotations_added = []

        scheme.node_added.connect(nodes_added.append)
        scheme.node_removed.connect(nodes_added.remove)

        scheme.link_added.connect(links_added.append)
        scheme.link_removed.connect(links_added.remove)

        scheme.annotation_added.connect(annotations_added.append)
        scheme.annotation_removed.connect(annotations_added.remove)

        w1 = scheme.new_node(file_desc)
        self.assertTrue(len(nodes_added) == 1)
        self.assertTrue(isinstance(nodes_added[-1], SchemeNode))
        self.assertTrue(nodes_added[-1] is w1)

        w2 = scheme.new_node(discretize_desc)
        self.assertTrue(len(nodes_added) == 2)
        self.assertTrue(isinstance(nodes_added[-1], SchemeNode))
        self.assertTrue(nodes_added[-1] is w2)

        w3 = scheme.new_node(bayes_desc)
        self.assertTrue(len(nodes_added) == 3)
        self.assertTrue(isinstance(nodes_added[-1], SchemeNode))
        self.assertTrue(nodes_added[-1] is w3)

        self.assertTrue(len(links_added) == 0)
        l1 = SchemeLink(w1, "Data", w2, "Data")
        scheme.add_link(l1)
        self.assertTrue(len(links_added) == 1)
        self.assertTrue(isinstance(links_added[-1], SchemeLink))
        self.assertTrue(links_added[-1] is l1)

        l2 = SchemeLink(w1, "Data", w3, "Data")
        scheme.add_link(l2)
        self.assertTrue(len(links_added) == 2)
        self.assertTrue(isinstance(links_added[-1], SchemeLink))
        self.assertTrue(links_added[-1] is l2)

        # Test find_links.
        found = scheme.find_links(w1, None, w2, None)
        self.assertSequenceEqual(found, [l1])
        found = scheme.find_links(None, None, w3, None)
        self.assertSequenceEqual(found, [l2])

        scheme.remove_link(l2)
        self.assertTrue(l2 not in links_added)

        # Add a link to itself.
        self.assertRaises(SchemeTopologyError, scheme.new_link,
                          w2, "Data", w2, "Data")

        # Add an link with incompatible types
        self.assertRaises(IncompatibleChannelTypeError,
                          scheme.new_link, w3, "Learner", w2, "Data")

        # Add a link to a node with no input channels
        self.assertRaises(ValueError, scheme.new_link,
                          w2, "Data", w1, "Data")

        # add back l2 for the folowing checks
        scheme.add_link(l2)

        # Add a duplicate link
        self.assertRaises(DuplicatedLinkError, scheme.new_link,
                          w1, "Data", w3, "Data")

        # Add a link to an already connected sink channel
        self.assertRaises(SinkChannelError, scheme.new_link,
                          w2, "Data", w3, "Data")

        text_annot = SchemeTextAnnotation((0, 0, 100, 20), "Text")
        scheme.add_annotation(text_annot)
        self.assertSequenceEqual(annotations_added, [text_annot])
        self.assertSequenceEqual(scheme.annotations, annotations_added)

        arrow_annot = SchemeTextAnnotation((0, 100), (100, 100))
        scheme.add_annotation(arrow_annot)
        self.assertSequenceEqual(annotations_added, [text_annot, arrow_annot])
        self.assertSequenceEqual(scheme.annotations, annotations_added)

        scheme.remove_annotation(text_annot)
        self.assertSequenceEqual(annotations_added, [arrow_annot])
        self.assertSequenceEqual(scheme.annotations, annotations_added)


class TestSchemeCycleDetection(unittest.TestCase):
    """
    Test class dedicated to check that the cycle detection and the management
    of the :attr:`allows_cycle` of the :class:`WidgetDescription` is well
    managed
    """
    iNewDesc = 0

    def setUp(self):
        self.reg = WidgetRegistry()

        # Create the scheme
        self.scheme = Scheme()

        self.reg.register_category(
            CategoryDescription(name='test',
                                qualified_name="Orange.widgets.test")
        )

    def test_simplecycle(self):
        """Make sure we succeed to create a cycle with one widget having
        the 'allows_cycle' flag and we fail if this flag is not here"""

        def create_simple_cycle(allows_cycle):
            desc_node_1 = self.create_node_desc(allows_cycle=False)
            self.reg.register_widget(desc_node_1)
            desc_1 = self.reg.widget(desc_node_1.qualified_name)

            desc_node_2 = self.create_node_desc(allows_cycle=allows_cycle)
            self.reg.register_widget(desc_node_2)
            desc_2 = self.reg.widget(desc_node_2.qualified_name)
            n1 = self.scheme.new_node(desc_1)
            n2 = self.scheme.new_node(desc_2)
            source_channel = n1.output_channel("Data")
            sink_channel = n2.input_channel("Data")
            lupstream = SchemeLink(n1, source_channel, n2, sink_channel)
            self.scheme.add_link(lupstream)
            ldownstream = SchemeLink(n2, "Data", n1, "Data")
            self.scheme.add_link(ldownstream)

        create_simple_cycle(allows_cycle=True)
        with self.assertRaises(SchemeCycleError):
            create_simple_cycle(allows_cycle=False)

    def create_node_desc(self, allows_cycle):
        """Create a simple widget taking a string in input and returning a
        string.

        :param allows_cycle: true if the node allow cycle
        """

        inputs = [
            InputSignal("Data", "builtins.str", "processing"),
            InputSignal("Data2", "builtins.str", "processing"),
            InputSignal("Data3", "builtins.str", "processing")
        ]

        outputs = [
            OutputSignal("Data", "builtins.str"),
            OutputSignal("Data2", "builtins.str"),
            OutputSignal("Data3", "builtins.str")
        ]
        name = 'desc_' + str(self.iNewDesc)
        node_desc = WidgetDescription(name=name,
                                      id='id_' + name,
                                      inputs=inputs,
                                      outputs=outputs,
                                      qualified_name='Orange.widgets.test.' + name,
                                      allows_cycle=allows_cycle,
                                      category='test')
        self.iNewDesc = self.iNewDesc + 1
        return node_desc

    def create_node(self):
        desc_node_1 = self.create_node_desc(allows_cycle=True)
        self.reg.register_widget(desc_node_1)
        desc_1 = self.reg.widget(desc_node_1.qualified_name)
        return self.scheme.new_node(desc_1)

    def link(self, node1, node2, channel):
        src_channel_1 = node1.output_channel(channel)
        sink_channel_2 = node2.input_channel(channel)
        link = SchemeLink(node1, src_channel_1, node2, sink_channel_2)
        self.scheme.add_link(link)

    def test_cycle_case_8(self):
        """Make sure under no condition we can create a '8 cycle'.
        Such as defined in the issue #2629
        """
        n0 = self.create_node()
        n1 = self.create_node()
        n2 = self.create_node()

        self.link(n0, n1, "Data")
        self.link(n1, n2, "Data")
        self.link(n2, n1, "Data2")
        with self.assertRaises(SchemeCycleError):
            self.link(n1, n0, "Data2")

    def test_extrapolate_cycles(self):
        """Make sure the algorithm of cycle detection is working"""
        n0 = self.create_node()
        n1 = self.create_node()
        n2 = self.create_node()
        n3 = self.create_node()
        n4 = self.create_node()
        n5 = self.create_node()
        n6 = self.create_node()

        self.link(n0, n1, "Data")
        self.link(n1, n2, "Data")
        self.link(n2, n1, "Data2")
        self.link(n2, n3, "Data")
        self.link(n3, n4, "Data")
        self.link(n4, n5, "Data")
        self.link(n5, n3, "Data2")

        source_channel = n5.output_channel("Data")
        sink_channel = n6.input_channel("Data")
        link = SchemeLink(n5, source_channel, n6, sink_channel)
        cycles = self.scheme.extrapolate_cycles(link)
        self.assertTrue(len(cycles) == 2)
        if len(cycles[0]) == 2:
            cyclen1n2 = cycles[0]
            cyclen3n4n5 = cycles[1]
        else:
            cyclen1n2 = cycles[1]
            cyclen3n4n5 = cycles[0]

        self.assertTrue(len(cyclen1n2) == 2)
        self.assertTrue(len(cyclen3n4n5) == 3)
        self.assertTrue(n1 in cyclen1n2 and n2 in cyclen1n2)
        self.assertTrue(n3 in cyclen3n4n5 and n4 in cyclen3n4n5 and n5 in cyclen3n4n5)
        self.scheme.add_link(link)

    def test_simple_cycle_with_subcycle(self):
        """Make sure we cannot create a 'simple cycle' containing at one
        'sub-cycle'. Such as defined in the issue #2629
        """
        n0 = self.create_node()
        n1 = self.create_node()
        n2 = self.create_node()

        self.link(n0, n1, "Data")
        self.link(n1, n2, "Data")
        self.link(n2, n1, "Data2")

        sourceChannel = n2.output_channel("Data")
        sinkChannel = n0.input_channel("Data")
        link = SchemeLink(n2, sourceChannel, n0, sinkChannel)

        cycles = self.scheme.extrapolate_cycles(link)
        self.assertTrue(len(cycles) == 2)
        with self.assertRaises(SchemeCycleError):
            self.scheme.add_link(link)
