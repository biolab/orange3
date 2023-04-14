from AnyQt.QtWidgets import QAbstractButton

from orangewidget.tests.base import GuiTest

import Orange
from Orange.widgets.data.utils.models import RichTableModel, TableSliceProxy
from Orange.widgets.data.utils.tableview import RichTableView
from Orange.widgets.utils.itemselectionmodel import BlockSelectionModel


class TableViewTest(GuiTest):
    def setUp(self) -> None:
        super().setUp()
        self.data = Orange.data.Table("iris")[::10]
        self.data.domain.attributes[0].attributes["A"] = "a"
        self.data.domain.class_var.attributes["A"] = "b"

    def tearDown(self) -> None:
        del self.data
        super().tearDown()

    def test_tableview(self):
        view = RichTableView()
        model = RichTableModel(self.data)
        view.setModel(model)
        self.assertIsInstance(view.selectionModel(), BlockSelectionModel)
        model.setRichHeaderFlags(RichTableModel.Name | RichTableModel.Labels |
                                 RichTableModel.Icon)
        view.grab()
        self.assertIn("A", view.cornerText())
        model.setRichHeaderFlags(RichTableModel.Name)
        self.assertEqual(view.cornerText(), "")

    def test_tableview_toggle_select_all(self):
        view = RichTableView()
        model = RichTableModel(self.data)
        view.setModel(model)
        b = view.findChild(QAbstractButton)
        b.click()
        self.assertEqual(len(view.selectionModel().selectedRows(0)),
                         model.rowCount())
        b.click()
        self.assertEqual(len(view.selectionModel().selectedRows(0)), 0)

    def test_selection(self):
        view = RichTableView()
        model = RichTableModel(self.data)
        view.setModel(model)
        view.setBlockSelection([1, 2], [2, 3])
        sel = [(idx.row(), idx.column()) for idx in view.selectedIndexes()]
        self.assertEqual(sorted(sel), [(1, 2), (1, 3), (2, 2), (2, 3)])
        self.assertEqual(view.blockSelection(), ([1, 2], [2, 3]))

        model_ = TableSliceProxy(rowSlice=slice(1, None, 1))
        model_.setSourceModel(model)
        view.setModel(model_)
        view.setBlockSelection([1, 2], [2, 3])
        sel = [(idx.row(), idx.column()) for idx in view.selectedIndexes()]
        self.assertEqual(sorted(sel), [(0, 2), (0, 3), (1, 2), (1, 3)])
        self.assertEqual(view.blockSelection(), ([1, 2], [2, 3]))

    def test_basket_column(self):
        model = RichTableModel(self.data.to_sparse())
        view = RichTableView()
        view.setModel(model)
        model.setRichHeaderFlags(RichTableModel.Name | RichTableModel.Labels)
        view.grab()
