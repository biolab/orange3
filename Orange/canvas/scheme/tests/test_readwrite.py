"""Test read write
"""
from xml.etree import ElementTree as ET
from io import BytesIO

from ...gui import test
from ...registry import global_registry, WidgetRegistry, WidgetDescription

from .. import Scheme, SchemeNode, SchemeLink, \
               SchemeArrowAnnotation, SchemeTextAnnotation

from .. import readwrite
from ..readwrite import scheme_to_ows_stream, parse_scheme, scheme_load


class TestReadWrite(test.QAppTestCase):
    def test_io(self):
        reg = global_registry()

        base = "Orange.widgets"
        file_desc = reg.widget(base + ".data.owfile.OWFile")
        discretize_desc = reg.widget(base + ".data.owdiscretize.OWDiscretize")
        bayes_desc = reg.widget(base + ".classify.ownaivebayes.OWNaiveBayes")

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

        stream = BytesIO()
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

    def test_io2(self):
        reg = global_registry()

        base = "Orange.widgets"
        file_desc = reg.widget(base + ".data.owfile.OWFile")
        discretize_desc = reg.widget(base + ".data.owdiscretize.OWDiscretize")
        bayes_desc = reg.widget(base + ".classify.ownaivebayes.OWNaiveBayes")

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

        stream = BytesIO()
        scheme_to_ows_stream(scheme, stream)

        stream.seek(0)

        scheme_1 = scheme_load(Scheme(), stream)

        self.assertEqual(len(scheme.nodes), len(scheme_1.nodes))
        self.assertEqual(len(scheme.links), len(scheme_1.links))
        self.assertEqual(len(scheme.annotations), len(scheme_1.annotations))

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

    def test_safe_evals(self):
        s = readwrite.string_eval(r"'\x00\xff'")
        self.assertEqual(s, chr(0) + chr(255))

        with self.assertRaises(ValueError):
            readwrite.string_eval("[1, 2]")

        t = readwrite.tuple_eval("(1, 2.0, 'a')")
        self.assertEqual(t, (1, 2.0, 'a'))

        with self.assertRaises(ValueError):
            readwrite.tuple_eval("u'string'")

        with self.assertRaises(ValueError):
            readwrite.tuple_eval("(1, [1, [2, ]])")

        self.assertIs(readwrite.terminal_eval("True"), True)
        self.assertIs(readwrite.terminal_eval("False"), False)
        self.assertIs(readwrite.terminal_eval("None"), None)

        self.assertEqual(readwrite.terminal_eval("42"), 42)
        self.assertEqual(readwrite.terminal_eval("'42'"), '42')

    def test_literal_dump(self):
        struct = {1: [{(1, 2): ""}],
                  True: 1.0,
                  None: None}

        s = readwrite.literal_dumps(struct)
        self.assertEqual(readwrite.literal_loads(s), struct)

        with self.assertRaises(ValueError):
            recur = [1]
            recur.append(recur)
            readwrite.literal_dumps(recur)

        with self.assertRaises(TypeError):
            readwrite.literal_dumps(self)

    def test_1_0_parse(self):
        tree = ET.parse(BytesIO(FOOBAR_v10))
        parsed = readwrite.parse_ows_etree_v_1_0(tree)
        self.assertIsInstance(parsed, readwrite._scheme)
        self.assertEqual(parsed.version, "1.0")
        self.assertTrue(len(parsed.nodes) == 2)
        self.assertTrue(len(parsed.links) == 2)

        qnames = [node.qualified_name for node in parsed.nodes]
        self.assertSetEqual(set(qnames), set(["foo", "bar"]))

        reg = foo_registry()

        parsed = readwrite.resolve_1_0(parsed, reg)

        qnames = [node.qualified_name for node in parsed.nodes]
        self.assertSetEqual(set(qnames),
                            set(["package.foo", "frob.bar"]))
        projects = [node.project_name for node in parsed.nodes]
        self.assertSetEqual(set(projects), set(["Foo", "Bar"]))

    def test_resolve_replaced(self):
        tree = ET.parse(BytesIO(FOOBAR_v20))
        parsed = readwrite.parse_ows_etree_v_2_0(tree)

        self.assertIsInstance(parsed, readwrite._scheme)
        self.assertEqual(parsed.version, "2.0")
        self.assertTrue(len(parsed.nodes) == 2)
        self.assertTrue(len(parsed.links) == 2)

        qnames = [node.qualified_name for node in parsed.nodes]
        self.assertSetEqual(set(qnames), set(["package.foo", "package.bar"]))

        reg = foo_registry()

        parsed = readwrite.resolve_replaced(parsed, reg)

        qnames = [node.qualified_name for node in parsed.nodes]
        self.assertSetEqual(set(qnames),
                            set(["package.foo", "frob.bar"]))
        projects = [node.project_name for node in parsed.nodes]
        self.assertSetEqual(set(projects), set(["Foo", "Bar"]))


def foo_registry():
    reg = WidgetRegistry()
    reg.register_widget(
        WidgetDescription(
            name="Foo",
            id="foooo",
            qualified_name="package.foo",
            project_name="Foo"
        )
    )
    reg.register_widget(
        WidgetDescription(
            name="Bar",
            id="barrr",
            qualified_name="frob.bar",
            project_name="Bar",
            replaces=["package.bar"]

        )
    )
    return reg


FOOBAR_v10 = b"""<?xml version="1.0" ?>
<schema>
    <widgets>
        <widget caption="Foo" widgetName="foo" xPos="1" yPos="2"/>
        <widget caption="Bar" widgetName="bar" xPos="2" yPos="3"/>
    </widgets>
    <channels>
        <channel enabled="1" inWidgetCaption="Foo" outWidgetCaption="Bar"
                 signals="[('foo', 'bar')]"/>
        <channel enabled="0" inWidgetCaption="Foo" outWidgetCaption="Bar"
                 signals="[('foo1', 'bar1')]"/>
    </channels>
    <settings settingsDictionary="{}"/>
</schema>
"""

FOOBAR_v20 = b"""<?xml version="1.0" ?>
<scheme title="FooBar" description="Foo to the bar" version="2.0">
    <nodes>
        <node id="0" title="Foo" position="1, 2" project_name="Foo"
              qualified_name="package.foo" />
        <node id="1" title="Bar" position="2, 3" project_name="Foo"
              qualified_name="package.bar" />
    </nodes>
    <links>
        <link enabled="true" id="0" sink_channel="bar" sink_node_id="1"
              source_channel="foo" source_node_id="0" />
        <link enabled="false" id="1" sink_channel="bar1" sink_node_id="1"
              source_channel="foo1" source_node_id="0" />
    </links>
</scheme>
"""
