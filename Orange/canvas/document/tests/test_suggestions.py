# pylint: disable=protected-access
from Orange.canvas.document import SchemeEditWidget
from Orange.canvas.document.interactions import NewLinkAction
from Orange.canvas.document.suggestions import Suggestions
from Orange.canvas.gui.test import QAppTestCase
from Orange.canvas.registry import global_registry
from Orange.canvas.registry.qt import QtWidgetRegistry
from Orange.canvas.scheme import SchemeLink, Scheme, SchemeNode


class TestSuggestions(QAppTestCase):
    def test_load_default(self):
        suggestions = Suggestions()
        suggestions._Suggestions__load_default_suggestions()

    def test_log_link(self):
        suggestions = Suggestions()

        reg = QtWidgetRegistry(global_registry())

        w = SchemeEditWidget()
        scheme = Scheme()
        w.setScheme(scheme)

        base = "Orange.widgets."
        file_desc = reg.widget(base + "data.owfile.OWFile")
        disc_desc = reg.widget(base + "data.owdiscretize.OWDiscretize")

        node1 = SchemeNode(file_desc, title="title1",
                           position=(100, 100))
        w.addNode(node1)
        node2 = SchemeNode(disc_desc, title="title2",
                           position=(300, 100))
        w.addNode(node2)

        link = SchemeLink(node1, "Data", node2, "Data")
        src_name = link.source_node.description.name
        sink_name = link.sink_node.description.name
        link_key = (src_name, sink_name, NewLinkAction.FROM_SOURCE)

        suggestions.set_direction(NewLinkAction.FROM_SOURCE)

        freq = suggestions._Suggestions__link_frequencies[link_key]
        source_prob = suggestions._Suggestions__source_probability[src_name][sink_name]

        w.addLink(link)

        new_freq = suggestions._Suggestions__link_frequencies[link_key]
        new_source_prob = suggestions._Suggestions__source_probability[src_name][sink_name]

        self.assertEqual(freq + 1, new_freq)
        self.assertEqual(source_prob + 1, new_source_prob)
