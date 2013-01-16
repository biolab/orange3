from PyQt4.QtGui import QGraphicsView, QPainter

from ..scene import CanvasScene
from .. import items
from ... import scheme

from ...gui.test import QAppTestCase


class TestScene(QAppTestCase):
    def setUp(self):
        QAppTestCase.setUp(self)
        self.scene = CanvasScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(QPainter.Antialiasing | \
                                 QPainter.TextAntialiasing)
        self.view.show()
        self.view.resize(400, 300)

    def test_scene(self):
        """Test basic scene functionality.
        """
        file_desc, disc_desc, bayes_desc = self.widget_desc()

        file_item = items.NodeItem(file_desc)
        disc_item = items.NodeItem(disc_desc)
        bayes_item = items.NodeItem(bayes_desc)

        file_item = self.scene.add_node_item(file_item)
        disc_item = self.scene.add_node_item(disc_item)
        bayes_item = self.scene.add_node_item(bayes_item)

        # Remove a node
        self.scene.remove_node_item(bayes_item)
        self.assertSequenceEqual(self.scene.node_items(),
                                 [file_item, disc_item])

        # And add it again
        self.scene.add_node_item(bayes_item)
        self.assertSequenceEqual(self.scene.node_items(),
                                 [file_item, disc_item, bayes_item])

        # Adding the same item again should raise an exception
        with self.assertRaises(ValueError):
            self.scene.add_node_item(bayes_item)

        # Add links
        link1 = self.scene.new_link_item(file_item, "Data", disc_item, "Data")
        link2 = self.scene.new_link_item(disc_item, "Data", bayes_item, "Data")

        link1a = self.scene.add_link_item(link1)
        link2a = self.scene.add_link_item(link2)
        self.assertEqual(link1, link1a)
        self.assertEqual(link2, link2a)
        self.assertSequenceEqual(self.scene.link_items(), [link1, link2])

        # Remove links
        self.scene.remove_link_item(link2)
        self.scene.remove_link_item(link1)
        self.assertSequenceEqual(self.scene.link_items(), [])

        self.assertTrue(link1.sourceItem is None and link1.sinkItem is None)
        self.assertTrue(link2.sourceItem is None and link2.sinkItem is None)

        self.assertSequenceEqual(file_item.outputAnchors(), [])
        self.assertSequenceEqual(disc_item.inputAnchors(), [])
        self.assertSequenceEqual(disc_item.outputAnchors(), [])
        self.assertSequenceEqual(bayes_item.outputAnchors(), [])

        # And add one link again
        link1 = self.scene.new_link_item(file_item, "Data", disc_item, "Data")
        link1 = self.scene.add_link_item(link1)
        self.assertSequenceEqual(self.scene.link_items(), [link1])

        self.assertTrue(file_item.outputAnchors())
        self.assertTrue(disc_item.inputAnchors())

        self.app.exec_()

    def test_scene_with_scheme(self):
        """Test scene through modifying the scheme.
        """
        test_scheme = scheme.Scheme()
        self.scene.set_scheme(test_scheme)

        node_items = []
        link_items = []

        self.scene.node_item_added.connect(node_items.append)
        self.scene.node_item_removed.connect(node_items.remove)
        self.scene.link_item_added.connect(link_items.append)
        self.scene.link_item_removed.connect(link_items.remove)

        file_desc, disc_desc, bayes_desc = self.widget_desc()
        file_node = scheme.SchemeNode(file_desc)
        disc_node = scheme.SchemeNode(disc_desc)
        bayes_node = scheme.SchemeNode(bayes_desc)

        nodes = [file_node, disc_node, bayes_node]
        test_scheme.add_node(file_node)
        test_scheme.add_node(disc_node)
        test_scheme.add_node(bayes_node)

        self.assertTrue(len(self.scene.node_items()) == 3)
        self.assertSequenceEqual(self.scene.node_items(), node_items)

        for node, item in zip(nodes, node_items):
            self.assertIs(item, self.scene.item_for_node(node))

        # Remove a widget
        test_scheme.remove_node(bayes_node)
        self.assertTrue(len(self.scene.node_items()) == 2)
        self.assertSequenceEqual(self.scene.node_items(), node_items)

        # And add it again
        test_scheme.add_node(bayes_node)
        self.assertTrue(len(self.scene.node_items()) == 3)
        self.assertSequenceEqual(self.scene.node_items(), node_items)

        # Add links
        link1 = test_scheme.new_link(file_node, "Data", disc_node, "Data")
        link2 = test_scheme.new_link(disc_node, "Data", bayes_node, "Data")
        self.assertTrue(len(self.scene.link_items()) == 2)
        self.assertSequenceEqual(self.scene.link_items(), link_items)

        # Remove links
        test_scheme.remove_link(link1)
        test_scheme.remove_link(link2)
        self.assertTrue(len(self.scene.link_items()) == 0)
        self.assertSequenceEqual(self.scene.link_items(), link_items)

        # And add one link again
        test_scheme.add_link(link1)
        self.assertTrue(len(self.scene.link_items()) == 1)
        self.assertSequenceEqual(self.scene.link_items(), link_items)
        self.app.exec_()

    def test_scheme_construction(self):
        """Test construction (editing) of the scheme through the scene.
        """
        test_scheme = scheme.Scheme()
        self.scene.set_scheme(test_scheme)

        node_items = []
        link_items = []

        self.scene.node_item_added.connect(node_items.append)
        self.scene.node_item_removed.connect(node_items.remove)
        self.scene.link_item_added.connect(link_items.append)
        self.scene.link_item_removed.connect(link_items.remove)

        file_desc, disc_desc, bayes_desc = self.widget_desc()
        file_node = scheme.SchemeNode(file_desc)

        file_item = self.scene.add_node(file_node)
        self.scene.commit_scheme_node(file_node)

        self.assertSequenceEqual(self.scene.node_items(), [file_item])
        self.assertSequenceEqual(node_items, [file_item])
        self.assertSequenceEqual(test_scheme.nodes, [file_node])

        disc_node = scheme.SchemeNode(disc_desc)
        bayes_node = scheme.SchemeNode(bayes_desc)

        disc_item = self.scene.add_node(disc_node)
        bayes_item = self.scene.add_node(bayes_node)

        self.assertSequenceEqual(self.scene.node_items(),
                                 [file_item, disc_item, bayes_item])
        self.assertSequenceEqual(self.scene.node_items(), node_items)

        # The scheme is still the same.
        self.assertSequenceEqual(test_scheme.nodes, [file_node])

        # Remove items
        self.scene.remove_node(disc_node)
        self.scene.remove_node(bayes_node)

        self.assertSequenceEqual(self.scene.node_items(), [file_item])
        self.assertSequenceEqual(node_items, [file_item])
        self.assertSequenceEqual(test_scheme.nodes, [file_node])

        # Add them again this time also in the scheme.
        disc_item = self.scene.add_node(disc_node)
        bayes_item = self.scene.add_node(bayes_node)

        self.scene.commit_scheme_node(disc_node)
        self.scene.commit_scheme_node(bayes_node)

        self.assertSequenceEqual(self.scene.node_items(),
                                 [file_item, disc_item, bayes_item])
        self.assertSequenceEqual(self.scene.node_items(), node_items)
        self.assertSequenceEqual(test_scheme.nodes,
                                 [file_node, disc_node, bayes_node])

        link1 = scheme.SchemeLink(file_node, "Data", disc_node, "Data")
        link2 = scheme.SchemeLink(disc_node, "Data", bayes_node, "Data")
        link_item1 = self.scene.add_link(link1)
        link_item2 = self.scene.add_link(link2)

        self.assertSequenceEqual(self.scene.link_items(),
                                 [link_item1, link_item2])
        self.assertSequenceEqual(self.scene.link_items(), link_items)
        self.assertSequenceEqual(test_scheme.links, [])

        # Commit the links
        self.scene.commit_scheme_link(link1)
        self.scene.commit_scheme_link(link2)

        self.assertSequenceEqual(self.scene.link_items(),
                                 [link_item1, link_item2])
        self.assertSequenceEqual(self.scene.link_items(), link_items)
        self.assertSequenceEqual(test_scheme.links,
                                 [link1, link2])

        self.app.exec_()

    def widget_desc(self):
        from ...registry.tests import small_testing_registry
        reg = small_testing_registry()

        file_desc = reg.widget(
            "Orange.OrangeWidgets.Data.OWFile.OWFile"
        )

        discretize_desc = reg.widget(
            "Orange.OrangeWidgets.Data.OWDiscretize.OWDiscretize"
        )

        bayes_desc = reg.widget(
            "Orange.OrangeWidgets.Classify.OWNaiveBayes.OWNaiveBayes"
        )

        return file_desc, discretize_desc, bayes_desc
