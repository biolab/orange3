from math import isnan

from AnyQt.QtCore import Qt, QIdentityProxyModel, QModelIndex

from orangewidget.gui import OrangeUserRole

import Orange
from Orange.widgets import gui
from Orange.widgets.utils.itemmodels import TableModel

_BarRole = gui.TableBarItem.BarRole


class RichTableModel(TableModel):
    """A TableModel with some extra bells and whistles/

    (adds support for gui.BarRole, include variable labels and icons
    in the header)
    """
    #: Rich header data flags.
    Name, Labels, Icon = 1, 2, 4

    #: Qt.ItemData role to retrieve variable's header attributes.
    LabelsItemsRole = next(OrangeUserRole)

    def __init__(self, sourcedata, parent=None):
        super().__init__(sourcedata, parent)

        self._header_flags = RichTableModel.Name
        self._continuous = [var.is_continuous for var in self.vars]
        labels = []
        for var in self.vars:
            if isinstance(var, Orange.data.Variable):
                labels.extend(var.attributes.keys())
        self._labels = list(sorted(
            {label for label in labels if not label.startswith("_")}))

    def data(self, index, role=Qt.DisplayRole):
        # pylint: disable=arguments-differ
        if role == _BarRole and self._continuous[index.column()]:
            val = super().data(index, TableModel.ValueRole)
            if val is None or isnan(val):
                return None

            dist = super().data(index, TableModel.VariableStatsRole)
            if dist is not None and dist.max > dist.min:
                return (val - dist.min) / (dist.max - dist.min)
            else:
                return None
        elif role == Qt.TextAlignmentRole and self._continuous[index.column()]:
            return Qt.AlignRight | Qt.AlignVCenter
        else:
            return super().data(index, role)

    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            var = super().headerData(
                section, orientation, TableModel.VariableRole)
            if var is None:
                return super().headerData(
                    section, orientation, Qt.DisplayRole)

            lines = []
            if self._header_flags & RichTableModel.Name:
                lines.append(var.name)
            if self._header_flags & RichTableModel.Labels:
                lines.extend(str(var.attributes.get(label, ""))
                             for label in self._labels)
            return "\n".join(lines)
        elif orientation == Qt.Horizontal and \
                role == RichTableModel.LabelsItemsRole:
            var = super().headerData(
                section, orientation, TableModel.VariableRole)
            if var is None:
                return []
            return [(label, var.attributes.get(label))
                    for label in self._labels]
        elif orientation == Qt.Horizontal and role == Qt.DecorationRole and \
                self._header_flags & RichTableModel.Icon:
            var = super().headerData(
                section, orientation, TableModel.VariableRole)
            if var is not None:
                return gui.attributeIconDict[var]
            else:
                return None
        else:
            return super().headerData(section, orientation, role)

    def setRichHeaderFlags(self, flags):
        if flags != self._header_flags:
            self._header_flags = flags
            self.headerDataChanged.emit(
                Qt.Horizontal, 0, self.columnCount() - 1)

    def richHeaderFlags(self):
        return self._header_flags


# This is used for sub-setting large (SQL) models. Largely untested probably
# broken.
class TableSliceProxy(QIdentityProxyModel):
    def __init__(self, parent=None, rowSlice=slice(0, None, 1), **kwargs):
        super().__init__(parent, **kwargs)
        self.__rowslice = slice(0, None, 1)
        self.setRowSlice(rowSlice)

    def setRowSlice(self, rowslice):
        if rowslice.step is not None and rowslice.step != 1:
            raise ValueError("invalid stride")

        if self.__rowslice != rowslice:
            self.beginResetModel()
            self.__rowslice = rowslice
            self.endResetModel()

    def index(
            self, row: int, column: int, _parent: QModelIndex = QModelIndex()
    ) -> QModelIndex:
        return self.createIndex(row, column)

    def parent(self, _child: QModelIndex) -> QModelIndex:
        return QModelIndex()

    def sibling(self, row: int, column: int, _idx: QModelIndex) -> QModelIndex:
        return self.index(row, column)

    def mapToSource(self, proxyindex):
        model = self.sourceModel()
        if model is None or not proxyindex.isValid():
            return QModelIndex()

        row, col = proxyindex.row(), proxyindex.column()
        row = row + self.__rowslice.start
        if 0 <= row < model.rowCount():
            return model.createIndex(row, col)
        else:
            return QModelIndex()

    def mapFromSource(self, sourceindex):
        model = self.sourceModel()
        if model is None or not sourceindex.isValid():
            return QModelIndex()
        row, col = sourceindex.row(), sourceindex.column()
        row = row - self.__rowslice.start
        if 0 <= row < self.rowCount():
            return self.createIndex(row, col)
        else:
            return QModelIndex()

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        count = super().rowCount()
        start, stop, step = self.__rowslice.indices(count)
        assert step == 1
        return stop - start
