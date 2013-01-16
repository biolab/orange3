"""Test read write
"""
from ...gui import test
from ...registry import global_registry

from .. import Scheme, SchemeNode, SchemeLink, \
               SchemeArrowAnnotation, SchemeTextAnnotation

from ..readwrite import scheme_to_ows_stream, parse_scheme


class TestReadWrite(test.QAppTestCase):
    def test_io(self):
        from StringIO import StringIO
        reg = global_registry()

        base = "Orange.OrangeWidgets"
        file_desc = reg.widget(base + ".Data.OWFile.OWFile")
        discretize_desc = reg.widget(base + ".Data.OWDiscretize.OWDiscretize")
        bayes_desc = reg.widget(base + ".Classify.OWNaiveBayes.OWNaiveBayes")

        scheme = Scheme()
        file_node = SchemeNode(file_desc)
        discretize_node = SchemeNode(discretize_desc)
        bayes_node = SchemeNode(bayes_desc)

        scheme.add_node(file_node)
        scheme.add_node(discretize_node)
        scheme.add_node(bayes_node)

        scheme.add_link(SchemeLink(file_node, "Data",
                                   discretize_node, "Data"))

        scheme.add_link(SchemeLink(discretize_node, "Data",
                                   bayes_node, "Data"))

        scheme.add_annotation(SchemeArrowAnnotation((0, 0), (10, 10)))
        scheme.add_annotation(SchemeTextAnnotation((0, 100, 200, 200), "$$"))

        stream = StringIO()
        scheme_to_ows_stream(scheme, stream)

        stream.seek(0)

        scheme_1 = parse_scheme(Scheme(), stream)

        self.assertTrue(len(scheme.nodes) == len(scheme_1.nodes))
        self.assertTrue(len(scheme.links) == len(scheme_1.links))
        self.assertTrue(len(scheme.annotations) == len(scheme_1.annotations))

        for n1, n2 in zip(scheme.nodes, scheme_1.nodes):
            self.assertEqual(n1.position, n2.position)
            self.assertEqual(n1.title, n2.title)

        for link1, link2 in zip(scheme.links, scheme_1.links):
            self.assertEqual(link1.source_type(), link2.source_type())
            self.assertEqual(link1.sink_type(), link2.sink_type())

            self.assertEqual(link1.source_channel.name,
                             link2.source_channel.name)

            self.assertEqual(link1.sink_channel.name,
                             link2.sink_channel.name)

            self.assertEqual(link1.enabled, link2.enabled)

        for annot1, annot2 in zip(scheme.annotations, scheme_1.annotations):
            self.assertIs(type(annot1), type(annot2))
            if isinstance(annot1, SchemeTextAnnotation):
                self.assertEqual(annot1.text, annot2.text)
                self.assertEqual(annot1.rect, annot2.rect)
            else:
                self.assertEqual(annot1.start_pos, annot2.start_pos)
                self.assertEqual(annot1.end_pos, annot2.end_pos)
