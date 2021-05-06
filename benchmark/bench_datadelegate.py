from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QStyledItemDelegate

from orangewidget.tests.base import GuiTest

from Orange.data import Table
from Orange.widgets.data.owtable import RichTableModel
from Orange.widgets.utils.itemdelegates import DataDelegate
from Orange.widgets.utils.tableview import TableView

from .base import benchmark, Benchmark


class BenchDataDelegate(GuiTest, Benchmark):
    def setUp(self) -> None:
        super().setUp()
        data = Table("brown-selected")
        self.view = TableView()
        self.delegate = DataDelegate(
            self.view, roles=(
                Qt.DisplayRole, Qt.BackgroundRole, Qt.TextAlignmentRole
            )
        )
        self.view.setItemDelegate(self.delegate)
        self.model = RichTableModel(data)
        self.view.setModel(self.model)
        self.view.resize(1024, 760)
        self.option = self.view.viewOptions()
        self.index_00 = self.model.index(0, 0)

    def tearDown(self) -> None:
        super().tearDown()
        self.view.setModel(None)
        del self.view
        del self.model
        del self.delegate
        del self.option
        del self.index_00

    @benchmark(number=3, warmup=1, repeat=10)
    def bench_paint(self):
        _ = self.view.grab()

    @benchmark(number=3, warmup=1, repeat=10)
    def bench_init_style_option(self):
        self.delegate.initStyleOption(self.option, self.index_00)

