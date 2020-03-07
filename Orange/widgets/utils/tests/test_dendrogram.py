import numpy as np

from AnyQt.QtCore import Qt, QPoint
from AnyQt.QtWidgets import QGraphicsScene

from orangewidget.tests.base import GuiTest

from Orange.clustering import hierarchical
from Orange.widgets.utils.dendrogram import DendrogramWidget


class TestDendrogramWidget(GuiTest):
    def setUp(self) -> None:
        super().setUp()
        self.scene = QGraphicsScene()
        self.widget = DendrogramWidget()
        self.scene.addItem(self.widget)

    def tearDown(self) -> None:
        self.scene.clear()
        del self.widget
        super().tearDown()

    def test_widget(self):
        w = self.widget

        T = hierarchical.Tree
        C = hierarchical.ClusterData
        S = hierarchical.SingletonData

        def t(h: float, left: T, right: T):
            return T(C((left.value.first, right.value.last), h), (left, right))

        def leaf(r, index):
            return T(S((r, r + 1), 0.0, index))

        T = hierarchical.Tree

        w.set_root(t(0.0, leaf(0, 0), leaf(1, 1)))
        w.resize(w.effectiveSizeHint(Qt.PreferredSize))
        h = w.height_at(QPoint())
        self.assertEqual(h, 0)
        h = w.height_at(QPoint(10, 0))
        self.assertEqual(h, 0)

        self.assertEqual(w.pos_at_height(0).x(), w.rect().x())
        self.assertEqual(w.pos_at_height(1).x(), w.rect().x())

        height = np.finfo(float).eps
        w.set_root(t(height, leaf(0, 0), leaf(1, 1)))

        h = w.height_at(QPoint())
        self.assertEqual(h, height)
        h = w.height_at(QPoint(w.size().width(), 0))
        self.assertEqual(h, 0)

        self.assertEqual(w.pos_at_height(0).x(), w.rect().right())
        self.assertEqual(w.pos_at_height(height).x(), w.rect().left())
