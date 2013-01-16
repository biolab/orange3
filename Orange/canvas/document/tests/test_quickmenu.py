from PyQt4.QtCore import QPoint
from ..quickmenu import QuickMenu, SuggestMenuPage, FlattenedTreeItemModel, \
                        ToolTree, QAction

from ...gui.test import QAppTestCase


class TestMenu(QAppTestCase):
    def test_menu(self):
        from PyQt4.QtGui import QStringListModel
        menu = QuickMenu()

        def triggered(action):
            print "Triggered", action.text()

        def hovered(action):
            print "Hover", action.text()

        menu.triggered.connect(triggered)
        menu.hovered.connect(hovered)

        items_page = ToolTree()
        model = QStringListModel(["one", "two", "file not found"])
        items_page.setModel(model)
        menu.addPage("w", items_page)

        page_c = ToolTree()
        menu.addPage("c", page_c)

        menu.popup(QPoint(200, 200))
        menu.activateWindow()

        self.app.exec_()

    def test_menu_with_registry(self):
        from ...registry.qt import run_discovery

        registry = run_discovery()
        menu = QuickMenu()
        menu.setModel(registry.model())

        triggered_action = []

        def triggered(action):
            print "Triggered", action.text()
            self.assertIsInstance(action, QAction)
            triggered_action.append(action)

        def hovered(action):
            self.assertIsInstance(action, QAction)
            print "Hover", action.text()

        menu.triggered.connect(triggered)
        menu.hovered.connect(hovered)
        self.app.setActiveWindow(menu)

        rval = menu.exec_(QPoint(200, 200))

        if triggered_action:
            self.assertIs(triggered_action[0], rval)

    def test_search(self):
        from ...registry.qt import run_discovery

        registry = run_discovery()

        menu = SuggestMenuPage()

        menu.setModel(registry.model())
        menu.show()
        menu.setFilterFixedString("la")
        self.singleShot(2500, lambda: menu.setFilterFixedString("ba"))
        self.app.exec_()

    def test_flattened_model(self):
        from PyQt4.QtGui import QStringListModel
        model = QStringListModel(["0", "1", "2", "3"])
        flat = FlattenedTreeItemModel()
        flat.setSourceModel(model)

        def get(row):
            return flat.index(row, 0).data().toPyObject()

        self.assertEqual(get(0), "0")
        self.assertEqual(get(1), "1")
        self.assertEqual(get(3), "3")
        self.assertEqual(flat.rowCount(), model.rowCount())
        self.assertEqual(flat.columnCount(), 1)
