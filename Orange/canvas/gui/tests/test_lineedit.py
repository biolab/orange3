"""
Test for searchwidget

"""

from AnyQt.QtWidgets import QAction, QStyle, QMenu

from ..lineedit import LineEdit

from ..test import QAppTestCase


class TestSearchWidget(QAppTestCase):
    def test_lineedit(self):
        """test LineEdit
        """
        line = LineEdit()
        line.show()

        action1 = QAction(line.style().standardIcon(QStyle.SP_ArrowBack),
                          "Search", line)
        menu = QMenu()
        menu.addAction("Regex")
        menu.addAction("Wildcard")
        action1.setMenu(menu)

        line.setAction(action1, LineEdit.LeftPosition)
        self.assertIs(line.actionAt(LineEdit.LeftPosition), action1)
        self.assertTrue(line.button(LineEdit.LeftPosition) is not None)
        self.assertTrue(line.button(LineEdit.RightPosition) is None)

        with self.assertRaises(ValueError):
            line.removeActionAt(100)

        line.removeActionAt(LineEdit.LeftPosition)
        self.assertIs(line.actionAt(LineEdit.LeftPosition), None)

        line.setAction(action1, LineEdit.LeftPosition)

        action2 = QAction(line.style().standardIcon(QStyle.SP_TitleBarCloseButton),
                          "Delete", line)
        line.setAction(action2, LineEdit.RightPosition)

        line.setPlaceholderText("Search")
        self.assertEqual(line.placeholderText(), "Search")

        b = line.button(LineEdit.RightPosition)
        b.setFlat(False)
        self.app.exec_()
