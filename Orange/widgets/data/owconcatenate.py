"""
Concatenate
===========

Concatenate (append) two or more data sets.

"""

from collections import OrderedDict
from functools import reduce
from itertools import chain, repeat
from operator import itemgetter

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt

import numpy

import Orange.data
from Orange.widgets import widget, gui, settings


class OWConcatenate(widget.OWWidget):
    name = "Concatenate"
    description = "Concatenate (append) two or more data sets."
    priority = 1111
    icon = "icons/Concatenate.svg"

    inputs = [("Primary Data", Orange.data.Table,
               "set_primary_data", widget.Default),
              ("Additional Data", Orange.data.Table,
               "set_more_data", widget.Multiple)]
    outputs = [("Data", Orange.data.Table)]

    #: Domain merging operations
    MergeUnion, MergeIntersection = 0, 1

    #: Domain role of the "Source ID" attribute.
    ClassRole, AttributeRole, MetaRole = 0, 1, 2

    #: Selected domain merging operation
    merge_type = settings.Setting(0)
    #: Append source ID column
    append_source_column = settings.Setting(False)
    #: Selected "Source ID" domain role
    source_column_role = settings.Setting(0)
    #: User specified name for the "Source ID" attr
    source_attr_name = settings.Setting("Source ID")

    want_main_area = False

    def __init__(self, parent=None):
        super().__init__(parent)

        self.primary_data = None
        self.more_data = OrderedDict()

        mergebox = gui.widgetBox(self.controlArea, "Domains merging")
        box = gui.radioButtons(
            mergebox, self, "merge_type",
            callback=self._merge_type_changed)

        gui.widgetLabel(
            box, self.tr("When there is no primary table, " +
                         "the domain should be:"))

        gui.appendRadioButton(
            box, self.tr("Union of attributes appearing in all tables"))

        gui.appendRadioButton(
            box, self.tr("Intersection of attributes in all tables"))

        gui.separator(box)

        label = gui.widgetLabel(
            box,
            self.tr("The resulting table will have class only if there " +
                    "is no conflict between input classes."))
        label.setWordWrap(True)

        ###
        box = gui.widgetBox(
            self.controlArea, self.tr("Source identification"),
            addSpace=False)

        cb = gui.checkBox(
            box, self, "append_source_column",
            self.tr("Append data source IDs"))

        ibox = gui.indentedBox(box, sep=gui.checkButtonOffsetHint(cb))

        form = QtGui.QFormLayout(
            spacing=8,
            labelAlignment=Qt.AlignLeft,
            formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QtGui.QFormLayout.AllNonFixedFieldsGrow
        )

        form.addRow(
            self.tr("Feature name"),
            gui.lineEdit(ibox, self, "source_attr_name", valueType=str))

        form.addRow(
            self.tr("Place"),
            gui.comboBox(
                ibox, self, "source_column_role",
                items=[self.tr("Class attribute"),
                       self.tr("Attribute"),
                       self.tr("Meta attribute")])
        )

        ibox.layout().addLayout(form)

        cb.disables.append(ibox)
        cb.makeConsistent()

        gui.button(
            self.controlArea, self, self.tr("Apply Changes"),
            callback=self.apply, default=True
        )

        gui.rubber(self.controlArea)
        self.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)

    def set_primary_data(self, data):
        self.primary_data = data

    def set_more_data(self, data=None, id=None):
        if data is None:
            del self.more_data[id]
        else:
            self.more_data[id] = data

    def handleNewSignals(self):
        self.apply()

    def apply(self):
        tables = []
        if self.primary_data is not None:
            tables = [self.primary_data] + list(self.more_data.values())
            domain = self.primary_data.domain
        elif self.more_data:
            tables = self.more_data.values()
            if self.merge_type == OWConcatenate.MergeUnion:
                domain = reduce(domain_union,
                                (table.domain for table in tables))
            else:
                domain = reduce(domain_intersection,
                                (table.domain for table in tables))

        tables = [Orange.data.Table.from_table(domain, table)
                  for table in tables]

        if tables:
            data = concat(tables)
            if self.append_source_column:
                source_var = Orange.data.DiscreteVariable(
                    self.source_attr_name,
                    values=["{}".format(i) for i in range(len(tables))]
                )
                source_values = list(
                    chain(*(repeat(i, len(table))
                            for i, table in enumerate(tables)))
                )
                places = ["class_vars", "attributes", "metas"]
                place = places[self.source_column_role]

                data = append_columns(
                    data, **{place: [(source_var, source_values)]}
                )
        else:
            data = None

        self.send("Data", data)

    def _merge_type_changed(self, ):
        if self.primary_data is None and self.more_data:
            self.apply()


def concat(tables):
    Xs = [table.X for table in tables]
    Ys = [table.Y for table in tables]
    metas = [table.metas for table in tables]

    domain = tables[0].domain

    X = numpy.vstack(Xs)
    Y = numpy.vstack(Ys)
    metas = numpy.vstack(metas)
    return Orange.data.Table.from_numpy(domain, X, Y, metas)


def unique(seq):
    seen_set = set()
    for el in seq:
        if el not in seen_set:
            yield el
            seen_set.add(el)


def domain_union(A, B):
    union = Orange.data.Domain(
        tuple(unique(A.attributes + B.attributes)),
        tuple(unique(A.class_vars + B.class_vars)),
        tuple(unique(A.metas + B.metas))
    )
    return union


def domain_intersection(A, B):
    def tuple_intersection(t1, t2):
        inters = set(t1) & set(t2)
        return tuple(el for el in t1 + t2 if el in inters)

    intersection = Orange.data.Domain(
        tuple_intersection(A.attributes, B.attributes),
        tuple_intersection(A.class_vars, B.class_vars),
        tuple_intersection(A.metas, B.metas),
    )

    return intersection


#:: (Table, **{place: [(Variable, values)]}) -> Table
def append_columns(data, attributes=(), class_vars=(), metas=()):
    domain = data.domain
    new_attributes = tuple(map(itemgetter(0), attributes))
    new_class_vars = tuple(map(itemgetter(0), class_vars))
    new_metas = tuple(map(itemgetter(0), metas))

    new_domain = Orange.data.Domain(
        domain.attributes + new_attributes,
        domain.class_vars + new_class_vars,
        domain.metas + new_metas
    )

    def ascolumn(array):
        array = numpy.asarray(array)
        if array.ndim < 2:
            array = array.reshape((-1, 1))
        return array

    attr_cols = [ascolumn(col) for _, col in attributes]
    class_cols = [ascolumn(col) for _, col in class_vars]
    metas = [ascolumn(col) for _, col in metas]

    X = numpy.hstack((data.X,) + tuple(attr_cols))
    Y = numpy.hstack((data.Y,) + tuple(class_cols))
    metas = numpy.hstack((data.metas,) + tuple(metas))

    new_data = Orange.data.Table.from_numpy(new_domain, X, Y, metas)
    return new_data


def main():
    app = QtGui.QApplication([])
    w = OWConcatenate()
    data_a = Orange.data.Table("iris")
    data_b = Orange.data.Table("zoo")
    w.set_more_data(data_a, 0)
    w.set_more_data(data_b, 1)
    w.handleNewSignals()
    w.show()

    app.exec_()


if __name__ == "__main__":
    main()
