"""
Tests for Scheme
"""

from ...gui import test
from ...registry.tests import small_testing_registry

from .. import (
    Scheme, SchemeNode, SchemeLink, SchemeTextAnnotation,
    SchemeArrowAnnotation, SchemeTopologyError, SinkChannelError,
    DuplicatedLinkError, IncompatibleChannelTypeError
)


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
