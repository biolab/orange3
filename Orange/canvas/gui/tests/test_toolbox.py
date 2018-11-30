"""
Tests for ToolBox widget.

"""

from AnyQt.QtWidgets import QLabel, QListView, QSpinBox, QAbstractButton
from AnyQt.QtGui import QIcon

from .. import test
from .. import toolbox


class TestToolBox(test.QAppTestCase):
    def test_tool_box(self):
        w = toolbox.ToolBox()
        style = self.app.style()
        icon = QIcon(style.standardIcon(style.SP_FileIcon))
        p1 = QLabel("A Label")
        p2 = QListView()
        p3 = QLabel("Another\nlabel")
        p4 = QSpinBox()

        i1 = w.addItem(p1, "T1", icon)
        i2 = w.addItem(p2, "Tab " * 10, icon, "a tab")
        i3 = w.addItem(p3, "t3")
        i4 = w.addItem(p4, "t4")

        self.assertSequenceEqual([i1, i2, i3, i4], range(4))
        self.assertEqual(w.count(), 4)

        for i, item in enumerate([p1, p2, p3, p4]):
            self.assertIs(item, w.widget(i))
            b = w.tabButton(i)
            a = w.tabAction(i)
            self.assertIsInstance(b,  QAbstractButton)
            self.assertIs(b.defaultAction(), a)

        w.show()
        w.removeItem(2)

        self.assertEqual(w.count(), 3)
        self.assertIs(w.widget(2), p4)

        p3 = QLabel("Once More Unto the Breach")

        w.insertItem(2, p3, "Dear friend")

        self.assertEqual(w.count(), 4)

        self.assertIs(w.widget(1), p2)
        self.assertIs(w.widget(2), p3)
        self.assertIs(w.widget(3), p4)

        self.app.exec_()
