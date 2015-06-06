import math
import itertools
from collections import defaultdict

from PyQt4 import QtGui, QtCore
import numpy

import Orange

from Orange.widgets import widget
from Orange.widgets import gui
from Orange.widgets.utils import itemmodels


class OWMergeData(widget.OWWidget):
    name = "Merge Data"
    description = "Merge data sets based on values of selected data feature."
    icon = "icons/MergeData.svg"
    priority = 1110

    inputs = [("Data A", Orange.data.Table, "setDataA", widget.Default),
              ("Data B", Orange.data.Table, "setDataB")]
    outputs = [("Merged Data A+B", Orange.data.Table, ),
               ("Merged Data B+A", Orange.data.Table, )]

    want_main_area = False

    def __init__(self, parent=None):
        super().__init__(parent)

        # data
        self.dataA = None
        self.dataB = None

        # GUI
        w = QtGui.QWidget(self)
        self.controlArea.layout().addWidget(w)
        grid = QtGui.QGridLayout()
        grid.setMargin(0)
        w.setLayout(grid)

        # attribute A selection
        boxAttrA = gui.widgetBox(
            self, self.tr("Attribute A"), addToLayout=False)
        grid.addWidget(boxAttrA, 0, 0)
        self.attrViewA = QtGui.QListView(
            selectionMode=QtGui.QListView.SingleSelection
        )

        self.attrModelA = itemmodels.VariableListModel()
        self.attrViewA.setModel(self.attrModelA)
        self.attrViewA.selectionModel().selectionChanged.connect(
            self._selectedAttrAChanged)

        boxAttrA.layout().addWidget(self.attrViewA)

        # attribute  B selection
        boxAttrB = gui.widgetBox(
            self, self.tr("Attribute B"), addToLayout=False)
        grid.addWidget(boxAttrB, 0, 1)
        self.attrViewB = QtGui.QListView(
            selectionMode=QtGui.QListView.SingleSelection
        )

        self.attrModelB = itemmodels.VariableListModel()
        self.attrViewB.setModel(self.attrModelB)
        self.attrViewB.selectionModel().selectionChanged.connect(
            self._selectedAttrBChanged)

        boxAttrB.layout().addWidget(self.attrViewB)

        # info A
        boxDataA = gui.widgetBox(
            self, self.tr("Data A Input"), addToLayout=False)
        grid.addWidget(boxDataA, 1, 0)
        self.infoBoxDataA = gui.widgetLabel(boxDataA, self.dataInfoText(None))

        # info B
        boxDataB = gui.widgetBox(
            self, self.tr("Data B Input"), addToLayout=False)
        grid.addWidget(boxDataB, 1, 1)
        self.infoBoxDataB = gui.widgetLabel(boxDataB, self.dataInfoText(None))

        # resize
        self.resize(400, 500)

    def setDataA(self, data):
        #self.closeContext()
        self.dataA = data
        if data is not None:
            self.attrModelA[:] = allvars(data)
        else:
            self.attrModelA[:] = []

        self.infoBoxDataA.setText(self.dataInfoText(data))

    def setDataB(self, data):
        #self.closeContext()
        self.dataB = data
        if data is not None:
            self.attrModelB[:] = allvars(data)
        else:
            self.attrModelB[:] = []

        self.infoBoxDataB.setText(self.dataInfoText(data))

    def handleNewSignals(self):
        self._invalidate()

    def dataInfoText(self, data):
        ninstances = 0
        nvariables = 0
        if data is not None:
            ninstances = len(data)
            nvariables = len(data.domain)

        instances = self.tr("%n instance(s)", None, ninstances)
        attributes = self.tr("%n variable(s)", None, nvariables)
        return "\n".join([instances, attributes])

    def selectedIndexA(self):
        return selected_row(self.attrViewA)

    def selectedIndexB(self):
        return selected_row(self.attrViewB)

    def commit(self):
        indexA = self.selectedIndexA()
        indexB = self.selectedIndexB()
        if indexA is None or indexB is None:
            return

        varA = self.attrModelA[indexA]
        varB = self.attrModelB[indexB]

        AB = merge(self.dataA, varA, self.dataB, varB)
        BA = merge(self.dataB, varB, self.dataA, varA)

        self.send("Merged Data A+B", AB)
        self.send("Merged Data B+A", BA)

    def _selectedAttrAChanged(self, *args):
        self._invalidate()

    def _selectedAttrBChanged(self, *args):
        self._invalidate()

    def _invalidate(self):
        self.commit()


def selected_row(view):
    rows = view.selectionModel().selectedRows()
    if rows:
        return rows[0].row()
    else:
        return None


def allvars(data):
    return data.domain.attributes + data.domain.class_vars + data.domain.metas


def merge(A, varA, B, varB):
    join_indices = left_join_indices(A, B, (varA,), (varB,))
    seen_set = set()

    def seen(val):
        return val in seen_set or bool(seen_set.add(val))

    merge_indices = [(i, j) for i, j in join_indices if not seen(i)]

    all_vars_A = set(A.domain.variables + A.domain.metas)
    iter_vars_B = itertools.chain(
        enumerate(B.domain.variables),
        ((-i, m) for i, m in enumerate(B.domain.metas, start=1))
    )
    reduced_indices_B = [i for i, var in iter_vars_B if not var in all_vars_A]
    reduced_B = B[:, list(reduced_indices_B)]

    return join_table_by_indices(A, reduced_B, merge_indices)


def group_table_indices(table, key_vars, exclude_unknown=False):
    """
    Group table indices based on values of selected columns (`key_vars`).

    Return a dictionary mapping all unique value combinations (keys)
    into a list of indices in the table where they are present.

    :param Orange.data.Table table:
    :param list-of-Orange.data.FeatureDescriptor] key_vars:
    :param bool exclude_unknown:

    """
    groups = defaultdict(list)
    for i, inst in enumerate(table):
        key = [inst[a] for a in key_vars]
        if exclude_unknown and any(math.isnan(k) for k in key):
            continue
        key = tuple([str(k) for k in key])
        groups[key].append(i)
    return groups


def left_join_indices(table1, table2, vars1, vars2):
    key_map1 = group_table_indices(table1, vars1)
    key_map2 = group_table_indices(table2, vars2)
    indices = []
    for i, inst in enumerate(table1):
        key = tuple([str(inst[v]) for v in vars1])
        if key in key_map1 and key in key_map2:
            for j in key_map2[key]:
                indices.append((i, j))
        else:
            indices.append((i, None))
    return indices


def right_join_indices(table1, table2, vars1, vars2):
    indices = left_join_indices(table2, table1, vars2, vars1)
    return [(j, i) for i, j in indices]


def inner_join_indices(table1, table2, vars1, vars2):
    indices = left_join_indices(table1, table2, vars1, vars2)
    return [(i, j) for i, j in indices if j is not None]


def left_join(left, right, left_vars, right_vars):
    """
    Left join `left` and `right` on values of `left/right_vars`.
    """
    indices = left_join_indices(left, right, left_vars, right_vars)
    return join_table_by_indices(left, right, indices)


def right_join(left, right, left_vars, right_vars):
    """
    Right join left and right on attributes attr1 and attr2
    """
    indices = right_join_indices(left, right, left_vars, right_vars)
    return join_table_by_indices(left, right, indices)


def inner_join(left, right, left_vars, right_vars):
    indices = inner_join_indices(left, right, left_vars, right_vars)
    return join_table_by_indices(left, right, indices)


def join_table_by_indices(left, right, indices):
    domain = Orange.data.Domain(
        left.domain.attributes + right.domain.attributes,
        left.domain.class_vars + right.domain.class_vars,
        left.domain.metas + right.domain.metas
    )
    X = join_array_by_indices(left.X, right.X, indices)
    Y = join_array_by_indices(numpy.c_[left.Y], numpy.c_[right.Y], indices)
    metas = join_array_by_indices(left.metas, right.metas, indices)

    return Orange.data.Table.from_numpy(domain, X, Y, metas)


def join_array_by_indices(left, right, indices, masked=float("nan")):
    left_masked = [masked] * left.shape[1]
    right_masked = [masked] * right.shape[1]

    leftparts = []
    rightparts = []
    for i, j in indices:
        if i is not None:
            leftparts.append(left[i])
        else:
            leftparts.append(left_masked)
        if j is not None:
            rightparts.append(right[j])
        else:
            rightparts.append(right_masked)

    def hstack_blocks(blocks):
        return numpy.hstack(list(map(numpy.vstack, blocks)))

    return hstack_blocks((leftparts, rightparts))


def test():
    app = QtGui.QApplication([])

    w = OWMergeData()
    zoo = Orange.data.Table("zoo")
    A = zoo[:, [0, 1, 2, "type", -1]]
    B = zoo[:, [3, 4, 5, "type", -1]]
    w.setDataA(A)
    w.setDataB(B)
    w.handleNewSignals()
    w.show()
    app.exec_()


if __name__ == "__main__":
    test()
