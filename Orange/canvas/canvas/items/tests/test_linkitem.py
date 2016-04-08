import time

from ..linkitem import LinkItem

from .. import NodeItem, AnchorPoint

from ....registry.tests import small_testing_registry

from . import TestItems


class TestLinkItem(TestItems):
    def test_linkitem(self):
        reg = small_testing_registry()

        data_desc = reg.category("Data")

        file_desc = reg.widget("Orange.widgets.data.owfile.OWFile")

        file_item = NodeItem()
        file_item.setWidgetDescription(file_desc)
        file_item.setWidgetCategory(data_desc)
        file_item.setPos(0, 100)

        discretize_desc = reg.widget(
            "Orange.widgets.data.owdiscretize.OWDiscretize"
        )

        discretize_item = NodeItem()
        discretize_item.setWidgetDescription(discretize_desc)
        discretize_item.setWidgetCategory(data_desc)
        discretize_item.setPos(200, 100)
        classify_desc = reg.category("Classify")

        bayes_desc = reg.widget(
            "Orange.widgets.classify.ownaivebayes.OWNaiveBayes"
        )

        nb_item = NodeItem()
        nb_item.setWidgetDescription(bayes_desc)
        nb_item.setWidgetCategory(classify_desc)
        nb_item.setPos(400, 100)

        self.scene.addItem(file_item)
        self.scene.addItem(discretize_item)
        self.scene.addItem(nb_item)

        link = LinkItem()
        anchor1 = file_item.newOutputAnchor()
        anchor2 = discretize_item.newInputAnchor()

        self.assertSequenceEqual(file_item.outputAnchors(), [anchor1])
        self.assertSequenceEqual(discretize_item.inputAnchors(), [anchor2])

        link.setSourceItem(file_item, anchor1)
        link.setSinkItem(discretize_item, anchor2)

        # Setting an item and an anchor not in the item's anchors raises
        # an error.
        with self.assertRaises(ValueError):
            link.setSourceItem(file_item, AnchorPoint())

        self.assertSequenceEqual(file_item.outputAnchors(), [anchor1])

        anchor2 = file_item.newOutputAnchor()

        link.setSourceItem(file_item, anchor2)
        self.assertSequenceEqual(file_item.outputAnchors(), [anchor1, anchor2])
        self.assertIs(link.sourceAnchor, anchor2)

        file_item.removeOutputAnchor(anchor1)

        self.scene.addItem(link)

        link = LinkItem()
        link.setSourceItem(discretize_item)
        link.setSinkItem(nb_item)

        self.scene.addItem(link)

        self.assertTrue(len(nb_item.inputAnchors()) == 1)
        self.assertTrue(len(discretize_item.outputAnchors()) == 1)
        self.assertTrue(len(discretize_item.inputAnchors()) == 1)
        self.assertTrue(len(file_item.outputAnchors()) == 1)

        link.removeLink()

        self.assertTrue(len(nb_item.inputAnchors()) == 0)
        self.assertTrue(len(discretize_item.outputAnchors()) == 0)
        self.assertTrue(len(discretize_item.inputAnchors()) == 1)
        self.assertTrue(len(file_item.outputAnchors()) == 1)

        self.app.exec_()

    def test_dynamic_link(self):
        link = LinkItem()
        anchor1 = AnchorPoint()
        anchor2 = AnchorPoint()

        self.scene.addItem(link)
        self.scene.addItem(anchor1)
        self.scene.addItem(anchor2)

        link.setSourceItem(None, anchor1)
        link.setSinkItem(None, anchor2)

        anchor2.setPos(100, 100)

        link.setSourceName("1")
        link.setSinkName("2")

        link.setDynamic(True)
        self.assertTrue(link.isDynamic())

        link.setDynamicEnabled(True)
        self.assertTrue(link.isDynamicEnabled())

        def advance():
            clock = time.clock()
            link.setDynamic(clock > 3)
            link.setDynamicEnabled(int(clock) % 2 == 0)
            self.singleShot(0, advance)

        advance()

        self.app.exec_()
