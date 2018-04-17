"""
Widget for assigning colors to variables
"""

import numpy as np
from AnyQt.QtCore import Qt, QSize, QAbstractTableModel
from AnyQt.QtGui import QColor, QFont, QImage, QBrush, qRgb
from AnyQt.QtWidgets import (
    QHeaderView, QColorDialog, QTableView, QApplication
)

import Orange
from Orange.widgets import widget, settings, gui
from Orange.widgets.gui import HorizontalGridDelegate
from Orange.widgets.utils.colorpalette import \
    ContinuousPaletteGenerator, ColorPaletteDlg
from Orange.widgets.widget import Input, Output

ColorRole = next(gui.OrangeUserRole)


# noinspection PyMethodOverriding
class ColorTableModel(QAbstractTableModel):
    """Base color model for discrete and continuous attributes. The model
    handles the first column; other columns are handled in the derived classes
    """

    def __init__(self):
        QAbstractTableModel.__init__(self)
        self.variables = []

    @staticmethod
    def _encode_color(color):
        return "#{}{}{}".format(*[("0" + hex(x)[2:])[-2:] for x in color])

    def flags(self, _):
        return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def set_data(self, variables):
        self.modelAboutToBeReset.emit()
        self.variables = variables
        self.modelReset.emit()

    def rowCount(self, parent):
        return 0 if parent.isValid() else self.n_rows()

    def columnCount(self, parent):
        return 0 if parent.isValid() else self.n_columns()

    def n_rows(self):
        return len(self.variables)

    def data(self, index, role=Qt.DisplayRole):
        # pylint: disable=missing-docstring
        # Only valid for the first column
        row = index.row()
        if role == Qt.DisplayRole or role == Qt.EditRole:
            return self.variables[row].name
        if role == Qt.FontRole:
            font = QFont()
            font.setBold(True)
            return font
        if role == Qt.TextAlignmentRole:
            return Qt.AlignRight | Qt.AlignVCenter

    def setData(self, index, value, role):
        # pylint: disable=missing-docstring
        # Only valid for the first column
        if role == Qt.EditRole:
            self.variables[index.row()].name = value
        else:
            return False
        self.dataChanged.emit(index, index)
        return True


class DiscColorTableModel(ColorTableModel):
    """A model that stores the colors corresponding to values of discrete
    variables. Colors are shown as decorations."""

    # The class only overloads the methods from the base class
    # pylint: disable=missing-docstring
    def n_columns(self):
        return bool(self.variables) and \
               1 + max(len(var.values) for var in self.variables)

    def data(self, index, role=Qt.DisplayRole):
        # pylint: disable=too-many-return-statements
        row, col = index.row(), index.column()
        if col == 0:
            return ColorTableModel.data(self, index, role)
        var = self.variables[row]
        if col > len(var.values):
            return
        if role == Qt.DisplayRole or role == Qt.EditRole:
            return var.values[col - 1]
        try:
            color = var.colors[col - 1]
        except (AttributeError, IndexError):
            return
        if role == Qt.DecorationRole:
            return QColor(*color)
        if role == Qt.ToolTipRole:
            return self._encode_color(color)
        if role == ColorRole:
            return var.colors[col - 1]

    # noinspection PyMethodOverriding
    def setData(self, index, value, role):
        row, col = index.row(), index.column()
        if col == 0:
            return ColorTableModel.setData(self, index, value, role)
        if role == ColorRole:
            self.variables[row].set_color(col - 1, value[:3])
        elif role == Qt.EditRole:
            self.variables[row].values[col - 1] = value
        else:
            return False
        self.dataChanged.emit(index, index)
        return True


class ContColorTableModel(ColorTableModel):
    """A model that stores the colors corresponding to values of discrete
    variables. Colors are shown as decorations."""

    # The class only overloads the methods from the base class, except
    # copy_to_all that is documented
    # pylint: disable=missing-docstring
    @staticmethod
    def n_columns():
        return 3

    def data(self, index, role=Qt.DisplayRole):
        def _column0():
            return ColorTableModel.data(self, index, role)

        def _column1():
            if role == Qt.DecorationRole:
                continuous_palette = ContinuousPaletteGenerator(*var.colors)
                line = continuous_palette.getRGB(np.arange(0, 1, 1 / 256))
                data = np.arange(0, 256, dtype=np.int8). \
                    reshape((1, 256)). \
                    repeat(16, 0)
                img = QImage(data, 256, 16, QImage.Format_Indexed8)
                img.setColorCount(256)
                img.setColorTable([qRgb(*x) for x in line])
                img.data = data
                return img
            if role == Qt.ToolTipRole:
                return "{} - {}".format(self._encode_color(var.colors[0]),
                                        self._encode_color(var.colors[1]))
            if role == ColorRole:
                return var.colors

        def _column2():
            if role == Qt.SizeHintRole:
                return QSize(100, 1)
            if role == Qt.ForegroundRole:
                return QBrush(Qt.blue)
            if row == self.mouse_row and role == Qt.DisplayRole:
                return "Copy to all"

        row, col = index.row(), index.column()
        var = self.variables[row]
        if 0 <= col <= 2:
            return [_column0, _column1, _column2][col]()

    # noinspection PyMethodOverriding
    def setData(self, index, value, role):
        row, col = index.row(), index.column()
        if col == 0:
            return ColorTableModel.setData(self, index, value, role)
        if role == ColorRole:
            self.variables[row].colors = value
        else:
            return False
        self.dataChanged.emit(index, index)
        return True

    def copy_to_all(self, index):
        colors = self.variables[index.row()].colors
        for row in range(self.n_rows()):
            self.variables[row].colors = colors
        self.dataChanged.emit(self.index(0, 1), self.index(self.n_rows(), 1))



class ColorTable(QTableView):
    """The base table view for discrete and continuous attributes."""

    # pylint: disable=missing-docstring
    def __init__(self, model):
        QTableView.__init__(self)
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setShowGrid(False)
        self.setSelectionMode(QTableView.NoSelection)
        self.setItemDelegate(HorizontalGridDelegate())
        self.setModel(model)

    def mouseReleaseEvent(self, event):
        index = self.indexAt(event.pos())
        if not index.isValid():
            return
        rect = self.visualRect(index)
        self.handle_click(index, event.pos().x() - rect.x())


class DiscreteTable(ColorTable):
    """Table view for discrete variables"""

    def handle_click(self, index, x_offset):
        """Handle click events for the first column (call the inherited
        edit method) and the second (call method for changing the palette)"""
        if self.model().data(index, Qt.EditRole) is None:
            return
        if index.column() == 0 or x_offset > 24:
            self.edit(index)
        else:
            self.change_color(index)

    def change_color(self, index):
        """Invoke palette editor and set the color"""
        color = self.model().data(index, ColorRole)
        if color is None:
            return
        dlg = QColorDialog(QColor(*color))
        if dlg.exec():
            color = dlg.selectedColor()
            self.model().setData(index, color.getRgb(), ColorRole)


class ContinuousTable(ColorTable):
    """Table view for continuous variables"""

    def __init__(self, master, model):
        ColorTable.__init__(self, model)
        self.master = master
        self.viewport().setMouseTracking(True)
        self.model().mouse_row = None

    def mouseMoveEvent(self, event):
        """Store the hovered row index in the model, trigger viewport update"""
        pos = event.pos()
        ind = self.indexAt(pos)
        self.model().mouse_row = ind.row()
        super().mouseMoveEvent(event)
        self.viewport().update()

    def leaveEvent(self, _):
        """Remove the stored the hovered row index, trigger viewport update"""
        self.model().mouse_row = None
        self.viewport().update()

    def handle_click(self, index, _):
        """Call the specific methods for handling clicks for each column"""
        if index.column() == 0:
            self.edit(index)
        elif index.column() == 1:
            self.change_color(index)
        elif index.column() == 2:
            self.model().copy_to_all(index)

    def change_color(self, index):
        """Invoke palette editor and set the color"""
        from_c, to_c, black = self.model().data(index, ColorRole)
        master = self.master
        dlg = ColorPaletteDlg(master)
        dlg.createContinuousPalette("", "Gradient palette", black,
                                    QColor(*from_c), QColor(*to_c))
        dlg.setColorSchemas(master.color_settings, master.selected_schema_index)
        if dlg.exec():
            self.model().setData(index,
                                 (dlg.contLeft.getColor().getRgb(),
                                  dlg.contRight.getColor().getRgb(),
                                  dlg.contpassThroughBlack),
                                 ColorRole)
            master.color_settings = dlg.getColorSchemas()
            master.selected_schema_index = dlg.selectedSchemaIndex


class OWColor(widget.OWWidget):
    name = "Color"
    description = "Set color legend for variables."
    icon = "icons/Colors.svg"

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        data = Output("Data", Orange.data.Table)

    settingsHandler = settings.PerfectDomainContextHandler(
        match_values=settings.PerfectDomainContextHandler.MATCH_VALUES_ALL)
    disc_data = settings.ContextSetting([])
    cont_data = settings.ContextSetting([])
    color_settings = settings.Setting(None)
    selected_schema_index = settings.Setting(0)
    auto_apply = settings.Setting(True)

    want_main_area = False

    def __init__(self):
        super().__init__()
        self.data = None
        self.orig_domain = self.domain = None
        self.disc_colors = []
        self.cont_colors = []

        box = gui.hBox(self.controlArea, "Discrete Variables")
        self.disc_model = DiscColorTableModel()
        disc_view = self.disc_view = DiscreteTable(self.disc_model)
        disc_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.disc_model.dataChanged.connect(self._on_data_changed)
        box.layout().addWidget(disc_view)

        box = gui.hBox(self.controlArea, "Numeric Variables")
        self.cont_model = ContColorTableModel()
        cont_view = self.cont_view = ContinuousTable(self, self.cont_model)
        cont_view.setColumnWidth(1, 256)
        self.cont_model.dataChanged.connect(self._on_data_changed)
        box.layout().addWidget(cont_view)

        box = gui.auto_commit(
            self.controlArea, self, "auto_apply", "Apply",
            orientation=Qt.Horizontal, checkbox_label="Apply automatically")
        box.button.setFixedWidth(180)
        box.layout().insertStretch(0)

    def sizeHint(self):
        return QSize(500, 570)

    def _create_proxies(self, variables):
        part_vars = []
        for var in variables:
            if var.is_discrete or var.is_continuous:
                var = var.make_proxy()
                if var.is_discrete:
                    var.values = var.values[:]
                    self.disc_colors.append(var)
                else:
                    self.cont_colors.append(var)
            part_vars.append(var)
        return part_vars

    @Inputs.data
    def set_data(self, data):
        """Handle data input signal"""
        self.closeContext()
        self.disc_colors = []
        self.cont_colors = []
        if data is None:
            self.data = self.domain = None
        else:
            domain = self.orig_domain = data.domain
            domain = Orange.data.Domain(self._create_proxies(domain.attributes),
                                        self._create_proxies(domain.class_vars),
                                        self._create_proxies(domain.metas))
            self.openContext(data)
            self.data = data.transform(domain)
            self.disc_model.set_data(self.disc_colors)
            self.cont_model.set_data(self.cont_colors)
            self.disc_view.resizeColumnsToContents()
            self.cont_view.resizeColumnsToContents()
        self.commit()

    def storeSpecificSettings(self):
        # Store the colors that were changed -- but not others
        self.current_context.disc_data = \
            [(var.name, var.values, "colors" in var.attributes and var.colors)
             for var in self.disc_colors]
        self.current_context.cont_data = \
            [(var.name, "colors" in var.attributes and var.colors)
             for var in self.cont_colors]

    def retrieveSpecificSettings(self):
        disc_data = getattr(self.current_context, "disc_data", ())
        for var, (name, values, colors) in zip(self.disc_colors, disc_data):
            var.name = name
            var.values = values[:]
            if colors is not False:
                var.colors = colors
        cont_data = getattr(self.current_context, "cont_data", ())
        for var, (name, colors) in zip(self.cont_colors, cont_data):
            var.name = name
            if colors is not False:
                var.colors = colors

    def _on_data_changed(self, *args):
        self.commit()

    def commit(self):
        self.Outputs.data.send(self.data)

    def send_report(self):
        """Send report"""
        def _report_variables(variables, orig_variables):
            from Orange.widgets.report import colored_square as square

            def was(n, o):
                return n if n == o else "{} (was: {})".format(n, o)

            # definition of td element for continuous gradient
            # with support for pre-standard css (needed at least for Qt 4.8)
            max_values = max(
                (len(var.values) for var in variables if var.is_discrete),
                default=1)
            defs = ("-webkit-", "-o-", "-moz-", "")
            cont_tpl = '<td colspan="{}">' \
                       '<span class="legend-square" style="width: 100px; '.\
                format(max_values) + \
                " ".join(map(
                    "background: {}linear-gradient("
                    "left, rgb({{}}, {{}}, {{}}), {{}}rgb({{}}, {{}}, {{}}));"
                    .format, defs)) + \
                '"></span></td>'

            rows = ""
            for var, ovar in zip(variables, orig_variables):
                if var.is_discrete:
                    values = "    \n".join(
                        "<td>{} {}</td>".
                        format(square(*var.colors[i]), was(value, ovalue))
                        for i, (value, ovalue) in
                        enumerate(zip(var.values, ovar.values)))
                elif var.is_continuous:
                    col = var.colors
                    colors = col[0][:3] + ("black, " * col[2], ) + col[1][:3]
                    values = cont_tpl.format(*colors * len(defs))
                else:
                    continue
                name = was(var.name, ovar.name)
                rows += '<tr style="height: 2em">\n' \
                        '  <th style="text-align: right">{}</th>{}\n</tr>\n'. \
                    format(name, values)
            return rows

        if not self.data:
            return
        domain = self.data.domain
        orig_domain = self.orig_domain
        sections = (
            (name, _report_variables(vars, ovars))
            for name, vars, ovars in (
                ("Features", domain.attributes, orig_domain.attributes),
                ("Outcome" + "s" * (len(domain.class_vars) > 1),
                 domain.class_vars, orig_domain.class_vars),
                ("Meta attributes", domain.metas, orig_domain.metas)))
        table = "".join("<tr><th>{}</th></tr>{}".format(name, rows)
                        for name, rows in sections if rows)
        if table:
            self.report_raw("<table>{}</table>".format(table))


if __name__ == "__main__":
    a = QApplication([])
    WIDGET = OWColor()
    WIDGET.set_data(Orange.data.Table("heart_disease.tab"))
    WIDGET.show()
    a.exec_()
    WIDGET.saveSettings()
