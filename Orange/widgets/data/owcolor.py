import copy

from PyQt4.QtCore import Qt, QAbstractTableModel, SIGNAL, QModelIndex
from PyQt4.QtGui import QStyledItemDelegate, QColor, QHeaderView, QFont, \
    QColorDialog, QTableView, QPixmap, qRgb, QImage
import numpy as np

import Orange
from Orange.widgets import widget, settings, gui
from Orange.widgets.utils.colorpalette import ColorPaletteGenerator, \
    ContinuousPaletteGenerator, ColorPaletteDlg

ColorRole = next(gui.OrangeUserRole)

def _encode_color(color):
    return "#{}{}{}".format(*[("0" + hex(x)[2:])[-2:] for x in color])


class HorizontalGridDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        painter.save()
        painter.setPen(QColor(212, 212, 212))
        painter.drawLine(option.rect.bottomLeft(), option.rect.bottomRight())
        painter.restore()
        QStyledItemDelegate.paint(self, painter, option, index)


# noinspection PyMethodOverriding
class ColorTableModel(QAbstractTableModel):
    def __init__(self):
        QAbstractTableModel.__init__(self)
        self.colors = []

    def set_data(self, colors):
        self.colors = colors
        self.emit(SIGNAL("dataChanged(QModelIndex, QModelIndex)"),
                  self.index(0, 0), self.index(self.n_columns(), self.n_rows()))

    def rowCount(self, parent):
        return 0 if parent.isValid() else self.n_rows()

    def columnCount(self, parent):
        return 0 if parent.isValid() else self.n_columns()

    def n_rows(self):
        return len(self.colors)

    def data(self, index, role=Qt.DisplayRole):
        # Only valid for the first column
        row, col = index.row(), index.column()
        if role == Qt.DisplayRole:
            return self.colors[row][0]
        if role == Qt.FontRole:
            font = QFont()
            font.setBold(True)
            return font
        if role == Qt.TextAlignmentRole:
            return Qt.AlignRight | Qt.AlignVCenter


class DiscColorTableModel(ColorTableModel):
    def n_columns(self):
        return bool(self.colors) and \
               1 + max(len(labels) for _, labels, __ in self.colors)

    def data(self, index, role=Qt.DisplayRole):
        row, col = index.row(), index.column()
        if col == 0:
            return ColorTableModel.data(self, index, role)
        name, labels, colors = self.colors[row]
        if col > len(labels):
            return
        if role == Qt.DisplayRole:
            return labels[col - 1]
        color = colors[col - 1]
        if role == Qt.DecorationRole:
            return QColor(*color)
        if role == Qt.ToolTipRole:
            return _encode_color(color)
        if role == ColorRole:
            return color

    # noinspection PyMethodOverriding
    def setData(self, index, value, role):
        row, col = index.row(), index.column()
        if role == ColorRole:
            self.colors[row][2][col - 1][:] = value[:3]
            self.emit(SIGNAL("dataChanged(QModelIndex, QModelIndex)"),
                      index, index)


class ContColorTableModel(ColorTableModel):
    def n_columns(self):
        return 2

    def data(self, index, role=Qt.DisplayRole):
        row, col = index.row(), index.column()
        if col == 0:
            return ColorTableModel.data(self, index, role)
        if col > 1:
            return
        colors = self.colors[row][1]
        if role == Qt.DecorationRole:
            continuous_palette = ContinuousPaletteGenerator(*colors)
            line = continuous_palette.getRGB(np.arange(0, 1, 1 / 256))
            data = np.arange(0, 256, dtype=np.int8).reshape(1, 256).repeat(16, 0)
            img = QImage(data, 256, 16, QImage.Format_Indexed8)
            img.setColorCount(256)
            img.setColorTable([qRgb(*x) for x in line])
            img.data = data
            return img
        if role == Qt.ToolTipRole:
            return "{} - {}".format(_encode_color(colors[0]),
                                    _encode_color(colors[1]))
        if role == ColorRole:
            return colors

    # noinspection PyMethodOverriding
    def setData(self, index, value, role):
        row, col = index.row(), index.column()
        if role == ColorRole:
            self.colors[row][1] = value
            self.emit(SIGNAL("dataChanged(QModelIndex, QModelIndex)"),
                      index, index)


class OWColor(widget.OWWidget):
    name = "Color"
    description = "Set color legend for variables"
    icon = "icons/Colors.svg"

    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Data", Orange.data.Table)]

    settingsHandler = settings.PerfectDomainContextHandler()
    disc_colors = settings.ContextSetting([])
    cont_colors = settings.ContextSetting([])
    color_settings = settings.Setting(None)
    selected_schema_index = settings.Setting(0)

    want_main_area = False

    def __init__(self):
        super().__init__()
        self.data = None
        self.disc_colors = []
        self.cont_colors = []

        def prepare_table(box, model, on_click):
            view = QTableView()
            view.horizontalHeader().hide()
            view.verticalHeader().hide()
            view.setShowGrid(False)
            view.setSelectionMode(QTableView.NoSelection)
            view.setItemDelegate(HorizontalGridDelegate())
            view.horizontalHeader().setResizeMode(QHeaderView.ResizeToContents)
            box.layout().addWidget(view)
            view.clicked.connect(on_click)
            view.setModel(model)
            return view

        box = gui.widgetBox(self.controlArea, "Discrete variables",
                            orientation="horizontal")
        self.disc_model = DiscColorTableModel()
        self.disc_view = prepare_table(box, self.disc_model, self.disc_clicked)

        box = gui.widgetBox(self.controlArea, "Numeric variables",
                            orientation="horizontal")
        self.cont_model = ContColorTableModel()
        self.cont_view = prepare_table(box, self.cont_model, self.cont_clicked)

    def set_data(self, data):
        self.disc_colors = []
        self.cont_colors = []
        if data is None:
            self.data = None
        else:
            def create_part(part):
                vars = []
                for var in part:
                    if not (var.is_discrete or var.is_continuous):
                        vars.append(var)
                        continue
                    var = var.make_proxy()
                    if hasattr(var, "colors"):
                        var.colors = copy.copy(var.colors)
                    if var.is_discrete:
                        if not hasattr(var, "colors"):
                            n_values = len(var.values)
                            palette = ColorPaletteGenerator(n_values)
                            var.colors = palette.getRGB(range(n_values))
                        self.disc_colors.append(
                            (var.name, var.values, var.colors))
                    else:
                        if not hasattr(var, "colors"):
                            var.colors = ((0, 0, 255), (255, 255, 0), False)
                        self.cont_colors.append([var.name, var.colors])
                    vars.append(var)
                return vars

            domain = data.domain
            domain = Orange.data.Domain(create_part(domain.attributes),
                                        create_part(domain.class_vars),
                                        create_part(domain.metas))
            self.data = Orange.data.Table(domain, data)
            self.disc_model.set_data(self.disc_colors)
            self.cont_model.set_data(self.cont_colors)
            print(self.disc_colors)
            print(self.cont_colors)
            self.disc_view.resizeColumnsToContents()
            self.cont_view.resizeColumnsToContents()
        self.commit()

    def commit(self):
        self.send("Data", self.data)

    def disc_clicked(self, index):
        color = self.disc_model.data(index, ColorRole)
        if color is None:
            return
        dlg = QColorDialog(QColor(*color))
        if dlg.exec():
            color = dlg.selectedColor()
            self.disc_model.setData(index, color.getRgb(), ColorRole)

    def cont_clicked(self, index):
        from_c, to_c, black = self.cont_model.data(index, ColorRole)
        dlg = ColorPaletteDlg(self)
        dlg.createContinuousPalette("", "Gradient palette", black,
                                    QColor(*from_c), QColor(*to_c))
        dlg.setColorSchemas(self.color_settings, self.selected_schema_index)
        if dlg.exec():
            self.cont_model.setData(index,
                                    (dlg.contLeft.getColor().getRgb(),
                                     dlg.contRight.getColor().getRgb(),
                                     dlg.contpassThroughBlack), \
                                    ColorRole)
            self.color_settings = dlg.getColorSchemas()
            self.selected_schema_index = dlg.selectedSchemaIndex


if __name__ == "__main__":
    from PyQt4 import QtGui
    a = QtGui.QApplication([])
    ow = OWColor()
    ow.set_data(Orange.data.Table("heart_disease.tab"))
    ow.show()
    a.exec_()
    ow.saveSettings()
