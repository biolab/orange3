import unittest

from AnyQt.QtCore import Qt, QSizeF

from orangewidget.tests.base import GuiTest
from Orange.widgets.utils.graphicstextlist import TextListWidget, scaled


class TestTextListWidget(GuiTest):
    def test_setItems(self):
        w = TextListWidget()
        w.setItems([])
        self.assertEqual(w.count(), 0)
        w.setItems(["Aa"])
        self.assertEqual(w.count(), 1)
        w.setItems(["Aa", "Bb"])
        self.assertEqual(w.count(), 2)
        w.clear()
        self.assertEqual(w.count(), 0)
        w.clear()

    def test_orientation(self):
        w = TextListWidget()
        w.setItems(['x' * 20] * 2)
        w.setOrientation(Qt.Vertical)
        self.assertEqual(w.orientation(), Qt.Vertical)
        sh = w.effectiveSizeHint(Qt.PreferredSize)
        self.assertGreater(sh.width(), sh.height())
        w.setOrientation(Qt.Horizontal)
        sh = w.effectiveSizeHint(Qt.PreferredSize)
        self.assertLess(sh.width(), sh.height())

    def test_alignment(self):
        w = TextListWidget()
        w.setItems(["a"])
        w.resize(200, 100)
        w.setAlignment(Qt.AlignRight)
        self.assertEqual(w.alignment(), Qt.AlignRight)
        item = w.childItems()[0].childItems()[0]

        def brect(item):
            return item.mapRectToItem(w, item.boundingRect())

        self.assertEqual(brect(item).right(), 200)
        w.setAlignment(Qt.AlignLeft)
        self.assertEqual(brect(item).left(), 0)

        w.setAlignment(Qt.AlignHCenter)
        self.assertTrue(90 <= brect(item).center().x() < 110)

        w.setAlignment(Qt.AlignTop)
        self.assertEqual(brect(item).top(), 0)

        w.setAlignment(Qt.AlignBottom)
        self.assertEqual(brect(item).bottom(), 100)

        w.setAlignment(Qt.AlignVCenter)
        self.assertTrue(45 <= brect(item).center().y() < 55)


class TestUtils(unittest.TestCase):
    def test_scaled(self):
        cases_keep_aspect = [
            (QSizeF(100, 100), QSizeF(200, 300), QSizeF(200, 200)),
            (QSizeF(100, 100), QSizeF(300, 200), QSizeF(200, 200)),
            (QSizeF(100, 100), QSizeF(300, -1), QSizeF(300, 300)),
            (QSizeF(100, 100), QSizeF(-1, 300), QSizeF(300, 300)),
            (QSizeF(100, 100), QSizeF(-1, -1), QSizeF(100, 100)),
        ]
        for size, const, expected in cases_keep_aspect:
            s = scaled(size, const)
            self.assertEqual(s, expected, f"scaled({size}, {const})")

        cases_keep_aspect_by_expaindig = [
            (QSizeF(100, 100), QSizeF(200, 300), QSizeF(300, 300)),
            (QSizeF(100, 100), QSizeF(300, 200), QSizeF(300, 300)),
            (QSizeF(100, 100), QSizeF(300, -1), QSizeF(300, 300)),
            (QSizeF(100, 100), QSizeF(-1, 300), QSizeF(300, 300)),
            (QSizeF(100, 100), QSizeF(-1, -1), QSizeF(100, 100)),
        ]

        for size, const, expected in cases_keep_aspect_by_expaindig:
            s = scaled(size, const, Qt.KeepAspectRatioByExpanding)
            self.assertEqual(
                s, expected,
                f"scaled({size}, {const}, Qt.KeepAspectRatioByExpanding)"
            )

        self.assertEqual(
            scaled(QSizeF(0, 0), QSizeF(100, 100)), QSizeF(0, 0)
        )
