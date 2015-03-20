from PyQt4 import QtGui

import Orange

from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting

#: Purging flags
SortValues, RemoveConstant, RemoveUnusedValues = 1, 2, 4


class OWPurgeDomain(widget.OWWidget):
    name = "Purge Domain"
    description = "Removes redundant values and attributes, sorts values."
    icon = "icons/PurgeDomain.svg"
    category = "Data"
    keywords = ["data", "purge", "domain"]

    inputs = [("Data", Orange.data.Table, "setData")]
    outputs = [("Data", Orange.data.Table)]

    removeValues = Setting(1)
    removeAttributes = Setting(1)
    removeClassAttribute = Setting(1)
    removeClasses = Setting(1)
    autoSend = Setting(False)
    sortValues = Setting(True)
    sortClasses = Setting(True)

    want_main_area = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None

        self.preRemoveValues = 1
        self.preRemoveClasses = 1

        self.removedAttrs = "-"
        self.reducedAttrs = "-"
        self.resortedAttrs = "-"
        self.classAttr = "-"

        boxAt = gui.widgetBox(self.controlArea, "Attributes")
        gui.checkBox(boxAt, self, 'sortValues', 'Sort attribute values',
                     callback=self.optionsChanged)
        gui.separator(boxAt, 2)
        rua = gui.checkBox(
            boxAt, self, "removeAttributes",
            "Remove attributes with less than two values",
            callback=self.removeAttributesChanged)
        ruv = gui.checkBox(
            gui.indentedBox(boxAt, sep=gui.checkButtonOffsetHint(rua)),
            self,
            "removeValues",
            "Remove unused attribute values",
            callback=self.optionsChanged
        )
        rua.disables = [ruv]
        rua.makeConsistent()

        boxAt = gui.widgetBox(self.controlArea, "Classes", addSpace=True)
        gui.checkBox(boxAt, self, 'sortClasses', 'Sort classes',
                     callback=self.optionsChanged)
        gui.separator(boxAt, 2)
        rua = gui.checkBox(
            boxAt, self, "removeClassAttribute",
            "Remove class attribute if there are less than two classes",
            callback=self.removeClassesChanged
        )
        ruv = gui.checkBox(
            gui.indentedBox(boxAt, sep=gui.checkButtonOffsetHint(rua)),
            self,
            "removeClasses",
            "Remove unused class values",
            callback=self.optionsChanged
        )
        rua.disables = [ruv]
        rua.makeConsistent()

        box3 = gui.widgetBox(self.controlArea, 'Statistics', addSpace=True)
        gui.label(box3, self, "Removed attributes: %(removedAttrs)s")
        gui.label(box3, self, "Reduced attributes: %(reducedAttrs)s")
        gui.label(box3, self, "Resorted attributes: %(resortedAttrs)s")
        gui.label(box3, self, "Class attribute: %(classAttr)s")

        gui.auto_commit(self.controlArea, self, "autoSend", "Send Data",
                        checkbox_label="Send automatically",
                        orientation="horizontal")
        gui.rubber(self.controlArea)

    def setData(self, dataset):
        if dataset is not None:
            self.data = dataset
            self.unconditional_commit()
        else:
            self.removedAttrs = "-"
            self.reducedAttrs = "-"
            self.resortedAttrs = "-"
            self.classAttr = "-"
            self.send("Data", None)
            self.data = None

    def removeAttributesChanged(self):
        if not self.removeAttributes:
            self.preRemoveValues = self.removeValues
            self.removeValues = False
        else:
            self.removeValues = self.preRemoveValues
        self.optionsChanged()

    def removeClassesChanged(self):
        if not self.removeClassAttribute:
            self.preRemoveClasses = self.removeClasses
            self.removeClasses = False
        else:
            self.removeClasses = self.preRemoveClasses
        self.optionsChanged()

    def optionsChanged(self):
        self.commit()

    def commit(self):
        if self.data is None:
            return

        self.reducedAttrs = 0
        self.removedAttrs = 0
        self.resortedAttrs = 0

        attr_flags = sum([SortValues * self.sortValues,
                          RemoveConstant * self.removeAttributes,
                          RemoveUnusedValues * self.removeValues])

        class_flags = sum([SortValues * self.sortClasses,
                           RemoveConstant * self.removeClassAttribute,
                           RemoveUnusedValues * self.removeClasses])
        domain = self.data.domain

        attrs_state = [purge_var_M(var, self.data, attr_flags)
                       for var in domain.attributes]
        class_vars_state = [purge_var_M(var, self.data, class_flags)
                            for var in domain.class_vars]

        nremoved = len([st for st in attrs_state if is_removed(st)])
        nreduced = len([st for st in attrs_state
                        if not is_removed(st) and is_reduced(st)])
        nsorted = len([st for st in attrs_state
                       if not is_removed(st) and is_sorted(st)])

        self.removedAttrs = nremoved
        self.reducedAttrs = nreduced
        self.resortedAttrs = nsorted

        if class_vars_state:
            # TODO: Extend the reporting for multi-class domains
            st = class_vars_state[0]
            if isinstance(st, Var):
                self.classAttr = "Class is unchanged"
            elif is_removed(st):
                self.classAttr = "Class is removed"
            else:
                status = " and ".join(
                    [s for s, predicate in zip(["sorted", "reduced"],
                                               [is_sorted, is_reduced])
                     if predicate(st)]
                )
                self.classAttr = "Class is " + status

        attrs = tuple(merge_transforms(st).var for st in attrs_state
                      if not is_removed(st))
        class_vars = tuple(merge_transforms(st).var for st in class_vars_state
                           if not is_removed(st))

        newdomain = Orange.data.Domain(attrs, class_vars, domain.metas)
        if newdomain.attributes != domain.attributes or \
                newdomain.class_vars != domain.class_vars:
            data = Orange.data.Table.from_table(newdomain, self.data)
        else:
            data = self.data

        self.send("Data", data)


import numpy
from collections import namedtuple

# Define a simple Purge expression 'language'.
#: A input variable (leaf expression).
Var = namedtuple("Var", ["var"])
#: Removed variable (can only ever be present as a root node).
Removed = namedtuple("Removed", ["sub", "var"])
#: A reduced variable
Reduced = namedtuple("Reduced", ["sub", "var"])
#: A sorted variable
Sorted = namedtuple("Sorted", ["sub", "var"])
#: A general (lookup) transformed variable.
#: (this node is returned as a result of `merge` which joins consecutive
#: Removed/Reduced nodes into a single Transformed node)
Transformed = namedtuple("Transformed", ["sub", "var"])


def is_var(exp):
    """Is `exp` a `Var` node."""
    return isinstance(exp, Var)

def is_removed(exp):
    """Is `exp` a `Removed` node."""
    return isinstance(exp, Removed)

def _contains(exp, cls):
    """Does `node` contain a sub node of type `cls`"""
    if isinstance(exp, cls):
        return True
    elif isinstance(exp, Var):
        return False
    else:
        return _contains(exp.sub, cls)

def is_reduced(exp):
    """Does `exp` contain a `Reduced` node."""
    return _contains(exp, Reduced)

def is_sorted(exp):
    """Does `exp` contain a `Reduced` node."""
    return _contains(exp, Sorted)


def merge_transforms(exp):
    """
    Merge consecutive Removed, Reduced or Transformed nodes.

    .. note:: Removed nodes are returned unchanged.

    """
    if isinstance(exp, (Var, Removed)):
        return exp
    elif isinstance(exp, (Reduced, Sorted, Transformed)):
        prev = merge_transforms(exp.sub)
        if isinstance(prev, (Reduced, Sorted, Transformed)):
            B = exp.var.compute_value
            assert isinstance(B, Lookup)
            A = B.variable.compute_value
            assert isinstance(A, Lookup)

            new_var = Orange.data.DiscreteVariable(
                exp.var.name,
                values=exp.var.values,
                ordered=exp.var.ordered
            )
            new_var.compute_value = merge_lookup(A, B)
            assert isinstance(prev.sub, Var)
            return Transformed(prev.sub, new_var)
        else:
            assert prev is exp.sub
            return exp
    else:
        raise TypeError


def purge_var_M(var, data, flags):
    state = Var(var)
    if flags & RemoveConstant:
        var = remove_constant(state.var, data)
        if var is None:
            return Removed(state, state.var)

    if isinstance(state.var, Orange.data.DiscreteVariable):
        if flags & RemoveUnusedValues:
            newattr = remove_unused_values(state.var, data)

            if newattr is not state.var:
                state = Reduced(state, newattr)

            if flags & RemoveConstant and len(state.var.values) < 2:
                return Removed(state, state.var)

        if flags & SortValues:
            newattr = sort_var_values(state.var)
            if newattr is not state.var:
                state = Sorted(state, newattr)

    return state


def purge_domain(data, attribute_flags=RemoveConstant | RemoveUnusedValues,
                 class_flags=RemoveConstant | RemoveUnusedValues):

    attrs = [purge_var_M(var, data, attribute_flags)
             for var in data.domain.attributes]
    class_vars = [purge_var_M(var, data, class_flags)
                  for var in data.domain.class_vars]

    attrs = [var for var in attrs if not is_removed(var)]
    class_vars = [var for var in class_vars if not is_removed(var)]
    attrs = [merge_transforms(var).var for var in attrs]
    class_vars = [merge_transforms(var).var for var in class_vars]

    return Orange.data.Domain(attrs, class_vars, data.domain.metas)


def has_at_least_two_values(data, var):
    ((dist, _), ) = data._compute_distributions([var])
    if isinstance(var, Orange.data.ContinuousVariable):
        dist = dist[1, :]
    return numpy.sum(dist > 0.0) > 1


def remove_constant(var, data):
    if isinstance(var, Orange.data.ContinuousVariable):
        if not has_at_least_two_values(data, var):
            return None
        else:
            return var
    elif isinstance(var, Orange.data.DiscreteVariable):
        if len(var.values) < 2:
            return None
        else:
            return var
    else:
        return var


def remove_unused_values(var, data):
    column_data = Orange.data.Table.from_table(
        Orange.data.Domain([var]),
        data
    )
    array = column_data.X.ravel()
    mask = numpy.isfinite(array)
    unique = numpy.array(numpy.unique(array[mask]), dtype=int)

    if len(unique) == len(var.values):
        return var

    used_values = [var.values[i] for i in unique]
    new_var = Orange.data.DiscreteVariable(
        "R_{}".format(var.name),
        values=used_values
    )
    translation_table = numpy.array([numpy.NaN] * len(var.values))
    translation_table[unique] = range(len(new_var.values))

    if 0 >= var.base_value < len(var.values):
        base = translation_table[var.base_value]
        if numpy.isfinite(base):
            new_var.base_value = int(base)

    new_var.compute_value = Lookup(var, translation_table)
    return new_var


def sort_var_values(var):
    newvalues = list(sorted(var.values))

    if newvalues == list(var.values):
        return var

    translation_table = numpy.array(
        [float(newvalues.index(value)) for value in var.values]
    )

    newvar = Orange.data.DiscreteVariable(var.name, values=newvalues)
    newvar.compute_value = Lookup(var, translation_table)
    return newvar

from Orange.preprocess.transformation import Lookup


class Lookup(Lookup):
    def transform(self, column):
        mask = numpy.isnan(column)
        column_valid = numpy.where(mask, 0, column)
        values = self.lookup_table[numpy.array(column_valid, dtype=int)]
        return numpy.where(mask, numpy.nan, values)


def merge_lookup(A, B):
    """
    Merge two consecutive Lookup transforms into one.
    """
    lookup_table = numpy.array(A.lookup_table)
    mask = numpy.isfinite(lookup_table)
    indices = numpy.array(lookup_table[mask], dtype=int)
    lookup_table[mask] = B.lookup_table[indices]
    return Lookup(A.variable, lookup_table)

if __name__ == "__main__":
    appl = QtGui.QApplication([])
    ow = OWPurgeDomain()
    data = Orange.data.Table("car.tab")
    subset = [inst for inst in data
              if inst["buying"] == "v-high"]
    subset = Orange.data.Table(data.domain, subset)
    # The "buying" should be removed and the class "y" reduced
    ow.setData(subset)
    ow.show()
    appl.exec_()
    ow.saveSettings()
