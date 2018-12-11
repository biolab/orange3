"""
Tests for scheme document.
"""

from ..schemeedit import SchemeEditWidget
from ...scheme import Scheme, SchemeNode, SchemeLink, SchemeTextAnnotation

from ...registry.tests import small_testing_registry

from ...gui.test import QAppTestCase


class TestSchemeEdit(QAppTestCase):
    def test_schemeedit(self):
        reg = small_testing_registry()

        w = SchemeEditWidget()

        scheme = Scheme()

        w.setScheme(scheme)

        self.assertIs(w.scheme(), scheme)
        self.assertTrue(not w.isModified())

#        w.setModified(True)
#        self.assertTrue(w.isModified())

        scheme = Scheme()

        w.setScheme(scheme)
        self.assertIs(w.scheme(), scheme)
        self.assertTrue(not w.isModified())

        w.show()

        base = "Orange.widgets."
        file_desc = reg.widget(base + "data.owfile.OWFile")
        disc_desc = reg.widget(base + "data.owdiscretize.OWDiscretize")

        node_list = []
        link_list = []
        annot_list = []

        scheme.node_added.connect(node_list.append)
        scheme.node_removed.connect(node_list.remove)

        scheme.link_added.connect(link_list.append)
        scheme.link_removed.connect(link_list.remove)

        scheme.annotation_added.connect(annot_list.append)
        scheme.annotation_removed.connect(annot_list.remove)

        node = SchemeNode(file_desc, title="title1", position=(100, 100))
        w.addNode(node)

        self.assertSequenceEqual(node_list, [node])
        self.assertSequenceEqual(scheme.nodes, node_list)

        self.assertTrue(w.isModified())

        stack = w.undoStack()
        stack.undo()

        self.assertSequenceEqual(node_list, [])
        self.assertSequenceEqual(scheme.nodes, node_list)
        self.assertTrue(not w.isModified())

        stack.redo()

        node1 = SchemeNode(disc_desc, title="title2",
                           position=(300, 100))
        w.addNode(node1)

        self.assertSequenceEqual(node_list, [node, node1])
        self.assertSequenceEqual(scheme.nodes, node_list)
        self.assertTrue(w.isModified())

        link = SchemeLink(node, "Data", node1, "Data")
        w.addLink(link)

        self.assertSequenceEqual(link_list, [link])

        stack.undo()
        stack.undo()

        stack.redo()
        stack.redo()

        w.removeNode(node1)

        self.assertSequenceEqual(link_list, [])
        self.assertSequenceEqual(node_list, [node])

        stack.undo()

        self.assertSequenceEqual(link_list, [link])
        self.assertSequenceEqual(node_list, [node, node1])

        w.removeLink(link)

        self.assertSequenceEqual(link_list, [])

        stack.undo()

        self.assertSequenceEqual(link_list, [link])

        annotation = SchemeTextAnnotation((200, 300, 50, 20), "text")
        w.addAnnotation(annotation)
        self.assertSequenceEqual(annot_list, [annotation])

        stack.undo()
        self.assertSequenceEqual(annot_list, [])

        stack.redo()
        self.assertSequenceEqual(annot_list, [annotation])

        w.resize(600, 400)
        self.app.exec_()
