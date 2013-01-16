"""
Test for tooltree

"""

from PyQt4.QtGui import QStandardItemModel, QStandardItem, QAction
from PyQt4.QtCore import Qt

from ..tooltree import ToolTree, FlattenedTreeItemModel

from ...registry.qt import QtWidgetRegistry
from ...registry.tests import small_testing_registry

from ..test import QAppTestCase


class TestToolTree(QAppTestCase):
    def test_tooltree(self):
        tree = ToolTree()
        role = tree.actionRole()
        model = QStandardItemModel()
        tree.setModel(model)
        item = QStandardItem("One")
        item.setData(QAction("One", tree), role)
        model.appendRow([item])

        cat = QStandardItem("A Category")
        item = QStandardItem("Two")
        item.setData(QAction("Two", tree), role)
        cat.appendRow([item])
        item = QStandardItem("Three")
        item.setData(QAction("Three", tree), role)
        cat.appendRow([item])

        model.appendRow([cat])

        def p(action):
            print("triggered", action.text())

        tree.triggered.connect(p)

        tree.show()

        self.app.exec_()

    def test_tooltree_registry(self):
        reg = QtWidgetRegistry(small_testing_registry())

        tree = ToolTree()
        tree.setModel(reg.model())
        tree.setActionRole(reg.WIDGET_ACTION_ROLE)
        tree.show()

        def p(action):
            print("triggered", action.text())

        tree.triggered.connect(p)

        self.app.exec_()

    def test_flattened(self):
        reg = QtWidgetRegistry(small_testing_registry())
        source = reg.model()

        model = FlattenedTreeItemModel()
        model.setSourceModel(source)

        tree = ToolTree()
        tree.setActionRole(reg.WIDGET_ACTION_ROLE)
        tree.setModel(model)
        tree.show()

        changed = []
        model.dataChanged.connect(
            lambda start, end: changed.append((start, end))
        )

        item = source.item(0).child(0)

        item.setText("New text")

        self.assertTrue(len(changed) == 1)
        self.assertEquals(changed[-1][0].data(Qt.DisplayRole).toString(),
                          "New text")

        self.assertEquals(model.data(model.index(1)).toString(), "New text")

        model.setFlatteningMode(FlattenedTreeItemModel.InternalNodesDisabled)

        self.assertFalse(model.index(0, 0).flags() & Qt.ItemIsEnabled)

        model.setFlatteningMode(FlattenedTreeItemModel.LeavesOnly)

        self.assertTrue(model.rowCount() == 3)

        def p(action):
            print("triggered", action.text())

        tree.triggered.connect(p)

        self.app.exec_()
