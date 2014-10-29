"""
ModelMap Projection Rank
------------------------

A widget for ranking projection quality.

"""
import sys
import numpy as np
import Orange

from PyQt4.QtGui import QApplication, QTableView, QStandardItemModel, QStandardItem
from PyQt4.QtCore import Qt, pyqtSignal as Signal, pyqtSlot as Slot

from Orange.data import Table
from Orange.data.sql.table import SqlTable
from Orange.widgets.settings import DomainContextHandler
from Orange.widgets.widget import OWWidget, gui, Default, AttributeList
from Orange.statistics import contingency


def std(f):
    x = np.array(range(len(f)))
    # normalize; we do not prefer attributes with many values
    x = x / x.mean()
    xf = np.multiply(f, x)
    x2f = np.multiply(f, np.power(x, 2))
    return np.sqrt((np.sum(x2f) - np.power(np.sum(xf), 2) / np.sum(f)) / (np.sum(f) - 1))


def p_index(ct):
    """Projection pursuit projection index."""
    ni, nj = ct.shape

    # compute standard deviation
    s = std(np.sum(ct, axis=1)) * std(np.sum(ct, axis=0))

    pairs = [(v1, v2) for v1 in range(ni) for v2 in range(nj)]

    d = sum(ct[pairs[p1]] * ct[pairs[p2]] * max(1.4142135623730951 - np.sqrt(
        np.power((pairs[p1][0] - pairs[p2][0]) / float(ni - 1), 2) + np.power(
            (pairs[p1][1] - pairs[p2][1]) / float(nj - 1), 2)), 0.) for p1 in range(len(pairs)) for p2 in range(p1))

    ssum = len(pairs) * (len(pairs) - 1) / 2.

    return s * d / ssum, s, d / ssum


class OWMPR(OWWidget):
    name = 'ModelMap Projection Rank'
    description = 'Ranking projections by estimating projection quality'
    icon = "icons/ModelMap.svg"

    inputs = [('Data', Table, 'set_data', Default)]
    outputs = [('Features', AttributeList)]
    want_main_area = False
    settingsHandler = DomainContextHandler()

    variable_changed = Signal()

    def __init__(self):
        super().__init__()
        self.data = None
        self.progress = None

        self.infoa = gui.widgetLabel(self.controlArea, "No data loaded.")

        self.projectionTable = QTableView()
        self.controlArea.layout().addWidget(self.projectionTable)
        self.projectionTable.setSelectionBehavior(QTableView.SelectRows)
        self.projectionTable.setSelectionMode(QTableView.SingleSelection)
        self.projectionTable.setSortingEnabled(True)

        self.projectionTableModel = QStandardItemModel(self)
        self.projectionTableModel.setHorizontalHeaderLabels(["P-Index", "", ""])
        self.projectionTable.setModel(self.projectionTableModel)

        self.projectionTable.setColumnWidth(0, 90)
        self.projectionTable.sortByColumn(0, Qt.DescendingOrder)
        self.projectionTable.selectionModel().selectionChanged.connect(self.on_selection_changed)

        gui.button(self.controlArea, self, "Rank Projections", callback=self.rank, default=True)
        self.resize(370, 600)

    def set_data(self, data):
        self.data = data
        self.infoa.setText("Data set: {}".format(data.name) if self.data else "No data loaded.")

    def rank(self):
        if self.progress:
            return

        disc = Orange.feature.discretization.EqualWidth(n=10)

        ndomain = Orange.data.Domain(
                [disc(self.data, attr) if type(attr) == Orange.data.variable.ContinuousVariable
                 else attr for attr in self.data.domain.attributes], self.data.domain.class_vars)

        t = self.data.from_table(ndomain, self.data)

        attrs = t.domain.attributes

        tables = {}
        l = 0
        self.progress = gui.ProgressBar(self, len(attrs) * (len(attrs) - 1) / 2)
        for i in range(len(attrs)):
            for j in range(i):
                ct = np.array(contingency.get_contingency(t, attrs[j], attrs[i]))
                pindex, _, _ = p_index(ct)
                tables[i, j] = ct

                item = QStandardItem()
                item.setData(float(pindex), Qt.DisplayRole)
                self.projectionTableModel.setItem(l, 0, item)

                item = QStandardItem()
                item.setData(attrs[i].name, Qt.DisplayRole)
                self.projectionTableModel.setItem(l, 1, item)

                item = QStandardItem()
                item.setData(attrs[j].name, Qt.DisplayRole)
                self.projectionTableModel.setItem(l, 2, item)

                self.progress.advance()
                l += 1

        self.progress.finish()
        self.progress = None

    def on_selection_changed(self, selected, deselected):
        """Called when the ranks view selection changes."""
        a1 = selected.indexes()[1].data().replace('D_', '')
        a2 = selected.indexes()[2].data().replace('D_', '')
        d = self.data.domain
        self.send("Features", AttributeList([d[a1], d[a2]]))


#test widget appearance
if __name__ == '__main__':
    a = QApplication(sys.argv)
    ow = OWMPR()
    ow.show()
    data = Orange.data.Table('zoo.tab')
    ow.set_data(data)
    ow.handleNewSignals()
    a.exec()
    #save settings
    ow.saveSettings()
