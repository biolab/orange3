from AnyQt.QtCore import Qt

from orangewidget.tests.base import GuiTest

from Orange.data import Table
from Orange.widgets.data.owtable import RichTableModel, TableBarItemDelegate
from Orange.widgets.utils.itemdelegates import DataDelegate
from Orange.widgets.utils.tableview import TableView

from .base import benchmark, Benchmark


class BaseBenchTableView(GuiTest, Benchmark):
    def setUp(self) -> None:
        super().setUp()
        self.view = TableView()
        self.view.resize(1024, 760)

    def tearDown(self) -> None:
        super().tearDown()
        self.view.setModel(None)
        del self.view


class BaseBenchTableModel(BaseBenchTableView):
    def setUp(self) -> None:
        super().setUp()
        data = Table("brown-selected")
        self.model = RichTableModel(data)
        self.view.setModel(self.model)


class BenchTableView(BaseBenchTableView):
    def setUp(self) -> None:
        super().setUp()
        data = Table("brown-selected")
        self.delegate = DataDelegate(
            self.view, roles=(
                Qt.DisplayRole, Qt.BackgroundRole, Qt.TextAlignmentRole
            )
        )
        self.view.setItemDelegate(self.delegate)
        self.model = RichTableModel(data)
        self.view.setModel(self.model)
        self.option = self.view.viewOptions()
        self.index_00 = self.model.index(0, 0)

    def tearDown(self) -> None:
        super().tearDown()
        del self.model
        del self.delegate
        del self.option
        del self.index_00

    @benchmark(number=10, warmup=1, repeat=3)
    def bench_paint(self):
        _ = self.view.grab()

    @benchmark(number=10, warmup=1, repeat=3)
    def bench_init_style_option(self):
        self.delegate.initStyleOption(self.option, self.index_00)


class BenchTableBarItemDelegate(BenchTableView):
    def setUp(self) -> None:
        super().setUp()
        self.delegate = TableBarItemDelegate(self.view)
        # self.delegate = gui.TableBarItem()
        self.view.setItemDelegate(self.delegate)
