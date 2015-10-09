import copy

from PyQt4 import QtGui
from PyQt4.QtCore import Qt, QAbstractTableModel

import Orange
from Orange.widgets import widget, settings, gui
from Orange.widgets.utils.colorpalette import ColorPaletteGenerator, \
    ContinuousPaletteGenerator


class ColorTableModel(QAbstractTableModel):
    def __init__(self, colors):
        QAbstractTableModel.__init__(self)
        self.colors = colors

    def columnCount(self, parent):
        return 0 if parent.isValid() else 2

    def rowCount(self, parent):
        return 0 if parent.isValid() else len(self.colors)

    def data(self, index, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return self.super().data(index, role)
        row, col = index.row(), index.column()
        if col == 0:
            return self.colors[row][0]


class OWColor(widget.OWWidget):
    name = "Color"
    description = "Set color legend for variables"
    icon = "icons/Colors.svg"

    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Data", Orange.data.Table)]

    settingsHandler = settings.PerfectDomainContextHandler()
    colors = settings.ContextSetting([])

    def __init__(self):
        super().__init__()
        self.data = None

        self.color_table = QtGui.QTableView()
        self.mainArea.layout().addWidget(self.color_table)

        self.color_model = ColorTableModel(self)
        self.color_table.setModel(self.color_model)

        self.resize(690, 500)

    def set_data(self, data):
        self.colors = []
        if data is None:
            self.data = None
        else:
            continuous_palette = ContinuousPaletteGenerator(Qt.blue, Qt.yellow)

            def create_part(part):
                vars = []
                for var in part:
                    if var.is_discrete or var.is_continuous:
                        var = var.make_proxy()
                        if hasattr(var, "colors"):
                            var.colors = copy.copy(var.colors)
                        elif var.is_discrete:
                            n_values = len(var.values)
                            palette = ColorPaletteGenerator(n_values)
                            var.colors = palette.getRGB(range(n_values))
                        else:
                            var.colors = continuous_palette
                        self.colors.append(var.name, var.colors)
                    vars.append(var)

            domain = data.domain
            domain = data.Domain(create_part(domain.attributes),
                                 create_part(domain.class_vars),
                                 create_part(domain.metas))
            self.data = data.Table(domain, data)
        self.commit()

    def commit(self):
        self.send("data", self.data)


if __name__ == "__main__":
    a = QtGui.QApplication([])
    ow = OWColor()
    ow.set_data(Orange.data.Table("heart_disease.tab"))
    ow.show()
    a.exec_()
    ow.saveSettings()
