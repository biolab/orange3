from itertools import chain

import numpy as np

from AnyQt.QtCore import Qt, QSize, QAbstractTableModel, QModelIndex, QTimer
from AnyQt.QtGui import QColor, QFont, QBrush
from AnyQt.QtWidgets import QHeaderView, QColorDialog, QTableView, QComboBox

import Orange
from Orange.util import color_to_hex
from Orange.widgets import widget, settings, gui
from Orange.widgets.gui import HorizontalGridDelegate
from Orange.widgets.utils import itemmodels, colorpalettes
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.widget import Input, Output
from orangewidget.settings import IncompatibleContext

ColorRole = next(gui.OrangeUserRole)
StripRole = next(gui.OrangeUserRole)


class AttrDesc:
    """
    Describes modifications that will be applied to variable.

    Provides methods that return either the modified value or the original

    Attributes:
        var (Variable): an instance of variable
        new_name (str or `None`): a changed name or `None`
    """
    def __init__(self, var):
        self.var = var
        self.new_name = None

    @property
    def name(self):
        return self.new_name or self.var.name

    @name.setter
    def name(self, name):
        self.new_name = name


class DiscAttrDesc(AttrDesc):
    """
    Describes modifications that will be applied to variable.

    Provides methods that return either the modified value or the original

    Attributes:
        var (DiscreteVariable): an instance of variable
        name (str or `None`): a changed name or `None`
        new_colors (list of tuple or None): new colors as tuples (R, G, B)
        new_values (list of str or None): new names for values, if changed
    """
    def __init__(self, var):
        super().__init__(var)
        self.new_colors = None
        self.new_values = None

    @property
    def colors(self):
        if self.new_colors is None:
            return self.var.colors
        else:
            return self.new_colors

    def set_color(self, i, color):
        if self.new_colors is None:
            self.new_colors = list(self.var.colors)
        self.new_colors[i] = color

    @property
    def values(self):
        return tuple(self.new_values or self.var.values)

    def set_value(self, i, value):
        if not self.new_values:
            self.new_values = list(self.var.values)
        self.new_values[i] = value

    def create_variable(self):
        new_var = self.var.copy(name=self.name, values=self.values)
        new_var.colors = np.asarray(self.colors)
        return new_var


class ContAttrDesc(AttrDesc):
    """
    Describes modifications that will be applied to variable.

    Provides methods that return either the modified value or the original

    Attributes:
        var (ContinuousVariable): an instance of variable
        name (str or `None`): a changed name or `None`
        palette_name (str or None): name of palette or None if unmodified
    """
    def __init__(self, var):
        super().__init__(var)
        if var.palette.name not in colorpalettes.ContinuousPalettes:
            self.new_palette_name = colorpalettes.DefaultContinuousPaletteName
        else:
            self.new_palette_name = None

    @property
    def palette_name(self):
        return self.new_palette_name or self.var.palette.name

    @palette_name.setter
    def palette_name(self, palette_name):
        self.new_palette_name = palette_name

    def create_variable(self):
        new_var = self.var.copy(name=self.name)
        new_var.attributes["palette"] = self.palette_name
        return new_var


class ColorTableModel(QAbstractTableModel):
    """
    Base color model for discrete and continuous variables. The model handles:
    - the first column - variable name (including setData)
    - flags
    - row count, computed as len(attrdescs)

    Attribute:
        attrdescs (list of AttrDesc): attrdescs with user-defined changes
    """
    def __init__(self):
        QAbstractTableModel.__init__(self)
        self.attrdescs = []

    @staticmethod
    def flags(_):  # pragma: no cover
        return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def set_data(self, attrdescs):
        self.modelAboutToBeReset.emit()
        self.attrdescs = attrdescs
        self.modelReset.emit()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.attrdescs)

    def data(self, index, role=Qt.DisplayRole):
        # Only valid for the first column; derived classes implement the rest
        row = index.row()
        if role in (Qt.DisplayRole, Qt.EditRole):
            return self.attrdescs[row].name
        if role == Qt.FontRole:
            font = QFont()
            font.setBold(True)
            return font
        if role == Qt.TextAlignmentRole:
            return Qt.AlignRight | Qt.AlignVCenter
        return None

    def setData(self, index, value, role):
        # Only valid for the first column; derived classes implement the rest
        if role == Qt.EditRole:
            self.attrdescs[index.row()].name = value
        else:
            return False
        self.dataChanged.emit(index, index)
        return True


class DiscColorTableModel(ColorTableModel):
    """
    A model that stores the colors corresponding to values of discrete
    variables. Colors are shown as decorations.
    """
    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return 1 + max((len(row.var.values) for row in self.attrdescs),
                       default=0)

    def data(self, index, role=Qt.DisplayRole):
        # pylint: disable=too-many-return-statements
        row, col = index.row(), index.column()
        if col == 0:
            return super().data(index, role)

        desc = self.attrdescs[row]
        if col > len(desc.var.values):
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            return desc.values[col - 1]

        color = desc.colors[col - 1]
        if role == Qt.DecorationRole:
            return QColor(*color)
        if role == Qt.ToolTipRole:
            return color_to_hex(color)
        if role == ColorRole:
            return color
        return None

    def setData(self, index, value, role):
        row, col = index.row(), index.column()
        if col == 0:
            return super().setData(index, value, role)

        desc = self.attrdescs[row]
        if role == ColorRole:
            desc.set_color(col - 1, value[:3])
        elif role == Qt.EditRole:
            desc.set_value(col - 1, value)
        else:
            return False
        self.dataChanged.emit(index, index)
        return True


class ContColorTableModel(ColorTableModel):
    """A model that stores the colors corresponding to values of discrete
    variables. Colors are shown as decorations.

    Attributes:
        mouse_row (int): the row over which the mouse is hovering
    """
    def __init__(self):
        super().__init__()
        self.mouse_row = None

    def set_mouse_row(self, row):
        self.mouse_row = row

    @staticmethod
    def columnCount(parent=QModelIndex()):
        return 0 if parent.isValid() else 3

    def data(self, index, role=Qt.DisplayRole):
        def _column0():
            return ColorTableModel.data(self, index, role)

        def _column1():
            palette = colorpalettes.ContinuousPalettes[desc.palette_name]
            if role == Qt.ToolTipRole:
                return palette.friendly_name
            if role == ColorRole:
                return palette
            if role == StripRole:
                return palette.color_strip(128, 16)
            if role == Qt.SizeHintRole:
                return QSize(150, 16)
            return None

        def _column2():
            if role == Qt.SizeHintRole:
                return QSize(100, 1)
            if role == Qt.ForegroundRole:
                return QBrush(Qt.blue)
            if row == self.mouse_row and role == Qt.DisplayRole:
                return "Copy to all"
            return None

        row, col = index.row(), index.column()
        desc = self.attrdescs[row]
        if 0 <= col <= 2:
            return [_column0, _column1, _column2][col]()

    # noinspection PyMethodOverriding
    def setData(self, index, value, role):
        row, col = index.row(), index.column()
        if col == 0:
            return super().setData(index, value, role)
        if role == ColorRole:
            self.attrdescs[row].palette_name = value.name
        else:
            return False
        self.dataChanged.emit(index, index)
        return True

    def copy_to_all(self, index):
        palette_name = self.attrdescs[index.row()].palette_name
        for desc in self.attrdescs:
            desc.palette_name = palette_name
        self.dataChanged.emit(self.index(0, 1), self.index(self.rowCount(), 1))


class ColorStripDelegate(HorizontalGridDelegate):
    def __init__(self, view):
        super().__init__()
        self.view = view

    def createEditor(self, parent, option, index):
        class Combo(QComboBox):
            def __init__(self, parent, initial_data):
                super().__init__(parent)
                model = itemmodels.ContinuousPalettesModel(icon_width=128)
                self.setModel(model)
                self.setCurrentIndex(model.indexOf(initial_data))
                self.setIconSize(QSize(128, 16))
                QTimer.singleShot(0, self.showPopup)

            def hidePopup(self):
                super().hidePopup()
                view.closeEditor(self, ColorStripDelegate.NoHint)

        def select(i):
            self.view.model().setData(
                index,
                combo.model().index(i, 0).data(Qt.UserRole),
                ColorRole)

        view = self.view
        combo = Combo(parent, index.data(ColorRole))
        combo.currentIndexChanged[int].connect(select)
        return combo

    def paint(self, painter, option, index):
        strip = index.data(StripRole)
        rect = option.rect
        painter.drawPixmap(
            rect.x() + 13, rect.y() + (rect.height() - strip.height()) / 2,
            strip)
        super().paint(painter, option, index)


class ColorTable(QTableView):
    """
    The base table view for discrete and continuous attributes.

    Sets the basic properties of the table and implementes mouseRelease that
    calls handle_click with appropriate index. It also prepares a grid_deleagte
    that is used in derived classes.
    """
    def __init__(self, model):
        QTableView.__init__(self)
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setShowGrid(False)
        self.setSelectionMode(QTableView.NoSelection)
        self.setModel(model)
        # View doesn't take ownership of delegates, so we store it here
        self.grid_delegate = HorizontalGridDelegate()

    def mouseReleaseEvent(self, event):
        index = self.indexAt(event.pos())
        if not index.isValid():
            return
        rect = self.visualRect(index)
        self.handle_click(index, event.pos().x() - rect.x())


class DiscreteTable(ColorTable):
    """Table view for discrete variables"""
    def __init__(self, model):
        super().__init__(model)
        self.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.setItemDelegate(self.grid_delegate)
        self.setEditTriggers(QTableView.NoEditTriggers)

    def handle_click(self, index, x_offset):
        """
        Handle click events for the first column (call the edit method)
        and the second (call method for changing the palette)
        """
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

    def __init__(self, model):
        super().__init__(model)
        self.viewport().setMouseTracking(True)
        # View doesn't take ownership of delegates, so we must store it
        self.color_delegate = ColorStripDelegate(self)
        self.setItemDelegateForColumn(0, self.grid_delegate)
        self.setItemDelegateForColumn(1, self.color_delegate)
        self.setColumnWidth(1, 256)
        self.setEditTriggers(
            QTableView.SelectedClicked | QTableView.DoubleClicked)

    def mouseMoveEvent(self, event):
        """Store the hovered row index in the model, trigger viewport update"""
        pos = event.pos()
        ind = self.indexAt(pos)
        self.model().set_mouse_row(ind.row())
        super().mouseMoveEvent(event)
        self.viewport().update()

    def leaveEvent(self, _):
        """Remove the stored the hovered row index, trigger viewport update"""
        self.model().set_mouse_row(None)
        self.viewport().update()

    def handle_click(self, index, _):
        """Call the specific methods for handling clicks for each column"""
        if index.column() < 2:
            self.edit(index)
        elif index.column() == 2:
            self.model().copy_to_all(index)


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
    disc_descs = settings.ContextSetting([])
    cont_descs = settings.ContextSetting([])
    color_settings = settings.Setting(None)
    selected_schema_index = settings.Setting(0)
    auto_apply = settings.Setting(True)

    settings_version = 2

    want_main_area = False

    def __init__(self):
        super().__init__()
        self.data = None
        self.orig_domain = self.domain = None

        box = gui.hBox(self.controlArea, "Discrete Variables")
        self.disc_model = DiscColorTableModel()
        self.disc_view = DiscreteTable(self.disc_model)
        self.disc_model.dataChanged.connect(self._on_data_changed)
        box.layout().addWidget(self.disc_view)

        box = gui.hBox(self.controlArea, "Numeric Variables")
        self.cont_model = ContColorTableModel()
        self.cont_view = ContinuousTable(self.cont_model)
        self.cont_model.dataChanged.connect(self._on_data_changed)
        box.layout().addWidget(self.cont_view)

        box = gui.auto_apply(self.controlArea, self, "auto_apply")
        box.button.setFixedWidth(180)
        box.layout().insertStretch(0)

        self.info.set_input_summary(self.info.NoInput)
        self.info.set_output_summary(self.info.NoOutput)

    @staticmethod
    def sizeHint():  # pragma: no cover
        return QSize(500, 570)

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.disc_descs = []
        self.cont_descs = []
        if data is None:
            self.data = self.domain = None
            self.info.set_input_summary(self.info.NoInput)
        else:
            self.data = data
            self.info.set_input_summary(len(data), format_summary_details(data))
            for var in chain(data.domain.variables, data.domain.metas):
                if var.is_discrete:
                    self.disc_descs.append(DiscAttrDesc(var))
                elif var.is_continuous:
                    self.cont_descs.append(ContAttrDesc(var))

        self.disc_model.set_data(self.disc_descs)
        self.cont_model.set_data(self.cont_descs)
        self.openContext(data)
        self.disc_view.resizeColumnsToContents()
        self.cont_view.resizeColumnsToContents()
        self.unconditional_commit()

    def _on_data_changed(self):
        self.commit()

    def commit(self):
        def make(variables):
            new_vars = []
            for var in variables:
                source = disc_dict if var.is_discrete else cont_dict
                desc = source.get(var.name)
                new_vars.append(desc.create_variable() if desc else var)
            return new_vars

        if self.data is None:
            self.Outputs.data.send(None)
            self.info.set_output_summary(self.info.NoOutput)
            return

        disc_dict = {desc.var.name: desc for desc in self.disc_descs}
        cont_dict = {desc.var.name: desc for desc in self.cont_descs}

        dom = self.data.domain
        new_domain = Orange.data.Domain(
            make(dom.attributes), make(dom.class_vars), make(dom.metas))
        new_data = self.data.transform(new_domain)
        self.info.set_output_summary(len(new_data),
                                     format_summary_details(new_data))
        self.Outputs.data.send(new_data)

    def send_report(self):
        """Send report"""
        def _report_variables(variables):
            from Orange.widgets.report import colored_square as square

            def was(n, o):
                return n if n == o else f"{n} (was: {o})"

            max_values = max(
                (len(var.values) for var in variables if var.is_discrete),
                default=1)

            rows = ""
            disc_dict = {k.var.name: k for k in self.disc_descs}
            cont_dict = {k.var.name: k for k in self.cont_descs}
            for var in variables:
                if var.is_discrete:
                    desc = disc_dict[var.name]
                    value_cols = "    \n".join(
                        f"<td>{square(*color)} {was(value, old_value)}</td>"
                        for color, value, old_value in
                        zip(desc.colors, desc.values, var.values))
                elif var.is_continuous:
                    desc = cont_dict[var.name]
                    pal = colorpalettes.ContinuousPalettes[desc.palette_name]
                    value_cols = f'<td colspan="{max_values}">' \
                                 f'{pal.friendly_name}</td>'
                else:
                    continue
                names = was(desc.name, desc.var.name)
                rows += '<tr style="height: 2em">\n' \
                        f'  <th style="text-align: right">{names}</th>' \
                        f'  {value_cols}\n' \
                        '</tr>\n'
            return rows

        if not self.data:
            return
        dom = self.data.domain
        sections = (
            (name, _report_variables(variables))
            for name, variables in (
                ("Features", dom.attributes),
                ("Outcome" + "s" * (len(dom.class_vars) > 1), dom.class_vars),
                ("Meta attributes", dom.metas)))
        table = "".join(f"<tr><th>{name}</th></tr>{rows}"
                        for name, rows in sections if rows)
        if table:
            self.report_raw(r"<table>{table}</table>")

    @classmethod
    def migrate_context(cls, context, version):
        if not version or version < 2:
            raise IncompatibleContext


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWColor).run(Orange.data.Table("heart_disease.tab"))
