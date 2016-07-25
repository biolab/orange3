import numpy as np

from PyQt4.QtCore import Qt, QAbstractItemModel, QModelIndex, QLineF, QSize
from PyQt4.QtGui import QTreeView, QColor, QItemDelegate,\
    QPainter, QPen, QBrush, QComboBox

from Orange.classification import OrangeTreeLearner
from Orange.widgets.settings import PerfectDomainContextHandler
from Orange.data import Table
from Orange.tree import OrangeTreeModel
from Orange.widgets import widget, gui
from Orange.widgets.widget import OWWidget


class BarDelegate(QItemDelegate):
    BarRole = next(gui.OrangeUserRole)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.color_schema = None

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setHeight(int(size.height() * 1.3))
        size.setWidth(size.width() + 10)
        return size

    def paint(self, painter, option, index):
        painter.save()
        self.drawBackground(painter, option, index)
        rect = option.rect
        value = index.data(self.BarRole)
        if value is not None:
            pw = 3
            hmargin = 5
            x = rect.left() + hmargin
            width = rect.width() - 2 * hmargin
            vmargin = 1
            textoffset = pw + vmargin * 2
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)
            baseline = rect.bottom() - textoffset / 2
            fact = width / np.sum(value)
            for prop, color in zip(value, self.color_schema):
                if prop == 0:
                    continue
                painter.setPen(QPen(QBrush(color), pw))
                to_x = x + prop * fact
                line = QLineF(x, baseline, to_x, baseline)
                painter.drawLine(line)
                x = to_x
            painter.restore()
            text_rect = rect.adjusted(0, 0, 0, -textoffset)
        else:
            text_rect = rect
        text = str(index.data(Qt.DisplayRole))
        self.drawDisplay(painter, option, text_rect, text)
        painter.restore()


class TreeModel(QAbstractItemModel):
    def __init__(self):
        super().__init__()
        self.tree = self.root = self.parents = self.tot_inst = None

    def set_tree(self, tree=None):
        self.beginResetModel()
        self.tree = tree
        self.root = tree.root
        self.parents = {}
        self._compute_parents(self.root)
        self.tot_inst = float(np.sum(tree.root.value))
        self.endResetModel()

    def _compute_parents(self, node):
        if node.children:
            for child in node.children:
                if child:
                    self.parents[child] = node
                    self._compute_parents(child)

    def _remove_parents(self, node):
        if node.children:
            for child in node.children:
                if child:
                    del self.parents[child]
                    self._remove_parents(child)

    def index(self, row, column, parent):
        node = parent.internalPointer() if parent.isValid() else self.root
        if node.children and row < len(node.children):
            return self.createIndex(row, column, node.children[row])
        else:
            return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        node = index.internalPointer()
        parent = self.parents.get(node)
        grandpa = self.parents.get(parent)
        if not grandpa:
            return QModelIndex()
        return self.createIndex(grandpa.children.index(parent), 0, parent)

    def rowCount(self, parent):
        if parent.column() > 0:
            return 0
        node = parent.internalPointer() if parent.isValid() else self.root
        return bool(node and node.children) and len(node.children)

    def columnCount(self, parent):
        return 2

    def data(self, index, role):
        if not index.isValid():
            return None
        node = index.internalPointer()
        col = index.column()
        dist = node.value
        modus = np.argmax(dist)
        tot = np.sum(dist)
        if role == Qt.DisplayRole:
            if col == 0:
                parent = self.parents[node]
                value = parent.describe_branch(parent.children.index(node))
                op = "= " if value[0] not in "≤>" else ""
                return "{} {}{}".format(parent.attr.name, op, value)
            if col == 1:
                prediction = self.tree.domain.class_var.values[modus]
                perc = int(round(100 * dist[modus] / tot))
                return "{} ({} %)".format(prediction, perc)
        elif role == BarDelegate.BarRole and col == 1:
            return dist / tot
        elif role == Qt.ToolTipRole:
            values = self.tree.domain.class_var.values
            return "{} instances; majority: {} ({} %)<br/>" \
                   "<nobr>Class distribution ({}): {} ({})</nobr>".format(
                       int(tot),
                       values[modus], int(round(dist[modus] / tot * 100)),
                       " : ".join(values),
                       " : ".join(str(int(val)) for val in dist),
                       " : ".join("{} %".format(int(round(val / tot * 100)))
                                  for val in dist))

    def node_for_index(self, index):
        return index.internalPointer()

    def cut_subtree(self, index):
        node = index.internalPointer()
        if node.children is None:
            return
        self.beginRemoveRows(index, 0, len(node.children))
        self._remove_parents(node)
        node.children = None
        self.endRemoveRows()

    def set_subtree(self, index, new_node):
        self.beginResetModel()
        node = index.internalPointer()
        parent = self.parents[node]
        child_idx = parent.children.index(node)
        parent.children[child_idx] = new_node
        self.parents = {}
        self._compute_parents(self.root)
        self.endResetModel()


class OWInteractiveTreeBuilder(OWWidget):
    """Widget for manually building and editing a tree"""

    name = "Interactive tree builder"
    icon = "icons/InteractiveTree.svg"
    priority = 10

    inputs = [("Data", Table, "set_dataset")]
    outputs = [("Selected Data", Table, widget.Default),
               ("Tree", OrangeTreeModel)]

    settingsHandler = PerfectDomainContextHandler()

    def __init__(self):
        super().__init__()
        self.tree = None

        box = gui.vBox(self.controlArea, box="Current Node")
        self.info_label = gui.widgetLabel(box, "")
        box = gui.vBox(self.controlArea)
        gui.widgetLabel(box, "Add/Set split: ")
        self.combo = QComboBox()
        self.combo.activated[int].connect(self.attr_selected)
        box.layout().addWidget(self.combo)
        gui.separator(box, 20, 20)
        self.prune_button = gui.button(
            box, self, "Cut Subtree", autoDefault=False, callback=self.prune)
        gui.button(
            box, self, "Build Subtree", autoDefault=False, callback=self.build)
        gui.rubber(self.controlArea)

        self.model = TreeModel()

        self.view = QTreeView()
        self.view.setModel(self.model)
        header = self.view.header()
        header.setResizeMode(0, header.Stretch)
        header.setResizeMode(1, header.ResizeToContents)
        header.setStretchLastSection(False)
        header.hide()
        self.delegate = BarDelegate(self)
        self.view.setItemDelegate(self.delegate)
        self.view.selectionModel().selectionChanged.connect(self.selection_changed)

        self.mainArea.layout().addWidget(self.view)

    def sizeHint(self):
        return QSize(600, 400)

    def attr_selected(self, i):
        if not i:
            return
        attr = self.tree.data.domain.attributes[i - 1]
        if attr.is_discrete:
            sel_index = self.view.selectedIndexes()[0]


    def set_dataset(self, data):
        domain = data.domain
        self.tree = OrangeTreeLearner()(data)
        self.model.set_tree(self.tree)
        self.view.resizeColumnToContents(1)
        self.view.expandAll()
        self.delegate.color_schema = \
            [QColor(*c) for c in domain.class_var.colors]
        self.combo.clear()
        self.combo.addItem("(Choose Variable)")
        for attr in domain.attributes:
            self.combo.addItem(gui.attributeIconDict[attr], attr.name)

    def selection_changed(self, selected, _):
        node = selected.indexes()[0].internalPointer()
        self.prune_button.setEnabled(bool(node.children))
        data = self.tree.instances[node.subset]
        dist = node.value
        tot = float(np.sum(dist))
        modus = np.argmax(dist)
        values = data.domain.class_var.values
        self.send("Selected Data", data)
        self.info_label.setText(
            "{} instances<br/><br/>Majority: {} ({} %)".format(
                int(tot), values[modus], int(round(dist[modus] / tot * 100))))

    def prune(self):
        self.model.cut_subtree(self.view.selectedIndexes()[0])

    def build(self):
        def fix_subsets(node):
            node.subset = super_subset[node.subset]
            if node.children:
                for child in node.children:
                    if child is not None:
                        fix_subsets(child)
        index = self.view.selectedIndexes()[0]
        node = self.model.node_for_index(index)
        super_subset = node.subset
        data = self.tree.instances[super_subset]
        print(data)
        print(len(data))
        subtree = OrangeTreeLearner()(data)
        self.model.set_subtree(index, subtree.root)
        print(len(subtree.root.subset))
        self.view.expandAll()

    def send_report(self):
        pass


def main():
    from PyQt4.QtGui import QApplication
    app = QApplication([])
    ow = OWInteractiveTreeBuilder()
    ow.set_dataset(Table("titanic"))
    ow.show()
    app.exec()

if __name__ == "__main__":
    main()
