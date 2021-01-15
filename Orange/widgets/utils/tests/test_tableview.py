from AnyQt.QtCore import Qt
from AnyQt.QtGui import QStandardItemModel
from AnyQt.QtTest import QTest, QSignalSpy

from Orange.widgets.tests.base import GuiTest
from Orange.widgets.utils.tableview import TableView


class TestTableView(GuiTest):
    def test_table_view_selection_finished(self):
        model = QStandardItemModel()
        model.setRowCount(10)
        model.setColumnCount(4)

        view = TableView()
        view.setModel(model)
        view.adjustSize()

        spy = QSignalSpy(view.selectionFinished)
        rect0 = view.visualRect(model.index(0, 0))
        rect4 = view.visualRect(model.index(4, 2))
        QTest.mousePress(
            view.viewport(), Qt.LeftButton, Qt.NoModifier, rect0.center(),
        )
        self.assertEqual(len(spy), 0)
        QTest.mouseRelease(
            view.viewport(), Qt.LeftButton, Qt.NoModifier, rect4.center(),
        )
        self.assertEqual(len(spy), 1)

    def test_table_view_default_vsection_size(self):
        view = TableView()
        vheader = view.verticalHeader()
        font = view.font()
        font.setPixelSize(38)
        view.setFont(font)
        self.assertGreaterEqual(vheader.defaultSectionSize(), 38)
