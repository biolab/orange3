import time

from PyQt4.QtGui import QGraphicsView, QPainter, QPainterPath

from ...gui.test import QAppTestCase

from ..layout import AnchorLayout
from ..scene import CanvasScene
from ..items import NodeItem, LinkItem


class TestAnchorLayout(QAppTestCase):
    def setUp(self):
        QAppTestCase.setUp(self)
        self.scene = CanvasScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.show()
        self.view.resize(600, 400)

    def test_layout(self):
        file_desc, disc_desc, bayes_desc = self.widget_desc()
        file_item = NodeItem()
        file_item.setWidgetDescription(file_desc)
        file_item.setPos(0, 150)
        self.scene.add_node_item(file_item)

        bayes_item = NodeItem()
        bayes_item.setWidgetDescription(bayes_desc)
        bayes_item.setPos(200, 0)
        self.scene.add_node_item(bayes_item)

        disc_item = NodeItem()
        disc_item.setWidgetDescription(disc_desc)
        disc_item.setPos(200, 300)
        self.scene.add_node_item(disc_item)

        link = LinkItem()
        link.setSourceItem(file_item)
        link.setSinkItem(disc_item)
        self.scene.add_link_item(link)

        link = LinkItem()
        link.setSourceItem(file_item)
        link.setSinkItem(bayes_item)
        self.scene.add_link_item(link)

        layout = AnchorLayout()
        self.scene.addItem(layout)
        self.scene.set_anchor_layout(layout)

        layout.invalidateNode(file_item)
        layout.activate()

        p1, p2 = file_item.outputAnchorItem.anchorPositions()
        self.assertGreater(p1, p2)

        self.scene.node_item_position_changed.connect(layout.invalidateNode)

        path = QPainterPath()
        path.addEllipse(125, 0, 50, 300)

        def advance():
            t = time.clock()
            bayes_item.setPos(path.pointAtPercent(t % 1.0))
            disc_item.setPos(path.pointAtPercent((t + 0.5) % 1.0))

            self.singleShot(20, advance)

        advance()

        self.app.exec_()

    def widget_desc(self):
        from ...registry.tests import small_testing_registry
        reg = small_testing_registry()

        file_desc = reg.widget(
            "Orange.widgets.data.owfile.OWFile"
        )

        discretize_desc = reg.widget(
            "Orange.widgets.data.owdiscretize.OWDiscretize"
        )

        bayes_desc = reg.widget(
            "Orange.widgets.classify.ownaivebayes.OWNaiveBayes"
        )

        return file_desc, discretize_desc, bayes_desc
