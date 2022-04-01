"""
Feature Constructor

A widget for defining (constructing) new features from values
of other variables.

"""
import re
import copy
import functools
import builtins
import math
import random
import logging
import ast
import types
import unicodedata

from traceback import format_exception_only
from collections import namedtuple, OrderedDict
from itertools import chain, count
from typing import List, Dict, Any

import numpy as np

from AnyQt.QtWidgets import (
    QSizePolicy, QAbstractItemView, QComboBox, QFormLayout, QLineEdit,
    QHBoxLayout, QVBoxLayout, QStackedWidget, QStyledItemDelegate,
    QPushButton, QMenu, QListView, QFrame, QLabel, QMessageBox)
from AnyQt.QtGui import QKeySequence
from AnyQt.QtCore import Qt, pyqtSignal as Signal, pyqtProperty as Property
from orangewidget.utils.combobox import ComboBoxSearch

import Orange
from Orange.data.util import get_unique_names
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler
from Orange.widgets.utils import itemmodels, vartype
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets import report
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Msg, Input, Output

FeatureDescriptor = \
    namedtuple("FeatureDescriptor", ["name", "expression"])

ContinuousDescriptor = \
    namedtuple("ContinuousDescriptor",
               ["name", "expression", "number_of_decimals"])
DateTimeDescriptor = \
    namedtuple("DateTimeDescriptor",
               ["name", "expression"])
DiscreteDescriptor = \
    namedtuple("DiscreteDescriptor",
               ["name", "expression", "values", "ordered"])

StringDescriptor = namedtuple("StringDescriptor", ["name", "expression"])

#warningIcon = gui.createAttributePixmap('!', QColor((202, 0, 32)))

def make_variable(descriptor, compute_value):
    if isinstance(descriptor, ContinuousDescriptor):
        return Orange.data.ContinuousVariable(
            descriptor.name,
            descriptor.number_of_decimals,
            compute_value)
    if isinstance(descriptor, DateTimeDescriptor):
        return Orange.data.TimeVariable(
            descriptor.name,
            compute_value=compute_value, have_date=True, have_time=True)
    elif isinstance(descriptor, DiscreteDescriptor):
        return Orange.data.DiscreteVariable(
            descriptor.name,
            values=descriptor.values,
            compute_value=compute_value)
    elif isinstance(descriptor, StringDescriptor):
        return Orange.data.StringVariable(
            descriptor.name,
            compute_value=compute_value)
    else:
        raise TypeError


def selected_row(view):
    """
    Return the index of selected row in a `view` (:class:`QListView`)

    The view's selection mode must be a QAbstractItemView.SingleSelction
    """
    if view.selectionMode() in [QAbstractItemView.MultiSelection,
                                QAbstractItemView.ExtendedSelection]:
        raise ValueError("invalid 'selectionMode'")

    sel_model = view.selectionModel()
    indexes = sel_model.selectedRows()
    if indexes:
        assert len(indexes) == 1
        return indexes[0].row()
    else:
        return None


class FeatureEditor(QFrame):
    ExpressionTooltip = """
Use variable names as values in expression.
Categorical features are passed as strings
(note the change in behaviour from Orange 3.30).

""".lstrip()

    FUNCTIONS = dict(chain([(key, val) for key, val in math.__dict__.items()
                            if not key.startswith("_")],
                           [(key, val) for key, val in builtins.__dict__.items()
                            if key in {"str", "float", "int", "len",
                                       "abs", "max", "min"}]))
    featureChanged = Signal()
    featureEdited = Signal()

    modifiedChanged = Signal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QFormLayout(
            fieldGrowthPolicy=QFormLayout.ExpandingFieldsGrow
        )
        layout.setContentsMargins(0, 0, 0, 0)
        self.nameedit = QLineEdit(
            placeholderText="Name...",
            sizePolicy=QSizePolicy(QSizePolicy.Minimum,
                                   QSizePolicy.Fixed)
        )
        self.expressionedit = QLineEdit(
            placeholderText="Expression...",
            toolTip=self.ExpressionTooltip)

        self.attrs_model = itemmodels.VariableListModel(
            ["Select Feature"], parent=self)
        self.attributescb = ComboBoxSearch(
            minimumContentsLength=16,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon,
            sizePolicy=QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        )
        self.attributescb.setModel(self.attrs_model)

        sorted_funcs = sorted(self.FUNCTIONS)
        self.funcs_model = itemmodels.PyListModelTooltip(
            chain(["Select Function"], sorted_funcs),
            chain([''], [self.FUNCTIONS[func].__doc__ for func in sorted_funcs])
        )
        self.funcs_model.setParent(self)

        self.functionscb = ComboBoxSearch(
            minimumContentsLength=16,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon,
            sizePolicy=QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
        self.functionscb.setModel(self.funcs_model)

        hbox = QHBoxLayout()
        hbox.addWidget(self.attributescb)
        hbox.addWidget(self.functionscb)

        layout.addRow(self.nameedit, self.expressionedit)
        layout.addRow(self.tr(""), hbox)
        self.setLayout(layout)

        self.nameedit.editingFinished.connect(self._invalidate)
        self.expressionedit.textChanged.connect(self._invalidate)
        self.attributescb.currentIndexChanged.connect(self.on_attrs_changed)
        self.functionscb.currentIndexChanged.connect(self.on_funcs_changed)

        self._modified = False

    def setModified(self, modified):
        if not isinstance(modified, bool):
            raise TypeError

        if self._modified != modified:
            self._modified = modified
            self.modifiedChanged.emit(modified)

    def modified(self):
        return self._modified

    modified = Property(bool, modified, setModified,
                        notify=modifiedChanged)

    def setEditorData(self, data, domain):
        self.nameedit.setText(data.name)
        self.expressionedit.setText(data.expression)
        self.setModified(False)
        self.featureChanged.emit()
        self.attrs_model[:] = ["Select Feature"]
        if domain is not None and not domain.empty():
            self.attrs_model[:] += chain(domain.attributes,
                                         domain.class_vars,
                                         domain.metas)

    def editorData(self):
        return FeatureDescriptor(name=self.nameedit.text(),
                                 expression=self.nameedit.text())

    def _invalidate(self):
        self.setModified(True)
        self.featureEdited.emit()
        self.featureChanged.emit()

    def on_attrs_changed(self):
        index = self.attributescb.currentIndex()
        if index > 0:
            attr = sanitized_name(self.attrs_model[index].name)
            self.insert_into_expression(attr)
            self.attributescb.setCurrentIndex(0)

    def on_funcs_changed(self):
        index = self.functionscb.currentIndex()
        if index > 0:
            func = self.funcs_model[index]
            if func in ["atan2", "fmod", "ldexp", "log",
                        "pow", "copysign", "hypot"]:
                self.insert_into_expression(func + "(,)")
                self.expressionedit.cursorBackward(False, 2)
            elif func in ["e", "pi"]:
                self.insert_into_expression(func)
            else:
                self.insert_into_expression(func + "()")
                self.expressionedit.cursorBackward(False)
            self.functionscb.setCurrentIndex(0)

    def insert_into_expression(self, what):
        cp = self.expressionedit.cursorPosition()
        ct = self.expressionedit.text()
        text = ct[:cp] + what + ct[cp:]
        self.expressionedit.setText(text)
        self.expressionedit.setFocus()


class ContinuousFeatureEditor(FeatureEditor):
    ExpressionTooltip = "A numeric expression\n\n" \
                        + FeatureEditor.ExpressionTooltip

    def editorData(self):
        return ContinuousDescriptor(
            name=self.nameedit.text(),
            number_of_decimals=None,
            expression=self.expressionedit.text()
        )


class DateTimeFeatureEditor(FeatureEditor):
    ExpressionTooltip = FeatureEditor.ExpressionTooltip + \
        "Result must be a string in ISO-8601 format " \
        "(e.g. 2019-07-30T15:37:27 or a part thereof),\n" \
        "or a number of seconds since Jan 1, 1970."

    def editorData(self):
        return DateTimeDescriptor(
            name=self.nameedit.text(),
            expression=self.expressionedit.text()
        )


class DiscreteFeatureEditor(FeatureEditor):
    ExpressionTooltip = FeatureEditor.ExpressionTooltip + \
        "Result must be a string, if values are not explicitly given\n" \
        "or a zero-based integer indices into a list of values given below."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        tooltip = \
            "If values are given, above expression must return zero-based " \
            "integer indices into that list."
        self.valuesedit = QLineEdit(placeholderText="A, B ...", toolTip=tooltip)
        self.valuesedit.textChanged.connect(self._invalidate)

        layout = self.layout()
        label = QLabel(self.tr("Values (optional)"))
        label.setToolTip(tooltip)
        layout.addRow(label, self.valuesedit)

    def setEditorData(self, data, domain):
        self.valuesedit.setText(
            ", ".join(v.replace(",", r"\,") for v in data.values))

        super().setEditorData(data, domain)

    def editorData(self):
        values = self.valuesedit.text()
        values = re.split(r"(?<!\\),", values)
        values = tuple(filter(None, [v.replace(r"\,", ",").strip() for v in values]))
        return DiscreteDescriptor(
            name=self.nameedit.text(),
            values=values,
            ordered=False,
            expression=self.expressionedit.text()
        )


class StringFeatureEditor(FeatureEditor):
    ExpressionTooltip = "A string expression\n\n" \
                        + FeatureEditor.ExpressionTooltip

    def editorData(self):
        return StringDescriptor(
            name=self.nameedit.text(),
            expression=self.expressionedit.text()
        )


_VarMap = {
    DiscreteDescriptor: vartype(Orange.data.DiscreteVariable("d")),
    ContinuousDescriptor: vartype(Orange.data.ContinuousVariable("c")),
    DateTimeDescriptor: vartype(Orange.data.TimeVariable("t")),
    StringDescriptor: vartype(Orange.data.StringVariable("s"))
}


@functools.lru_cache(20)
def variable_icon(dtype):
    vtype = _VarMap.get(dtype, dtype)
    return gui.attributeIconDict[vtype]


class FeatureItemDelegate(QStyledItemDelegate):
    @staticmethod
    def displayText(value, _):
        return value.name + " := " + value.expression


class DescriptorModel(itemmodels.PyListModel):
    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DecorationRole:
            value = self[index.row()]
            return variable_icon(type(value))
        else:
            return super().data(index, role)


def freevars(exp, env):
    """
    Return names of all free variables in a parsed (expression) AST.

    Parameters
    ----------
    exp : ast.AST
        An expression ast (ast.parse(..., mode="single"))
    env : List[str]
        Environment

    Returns
    -------
    freevars : List[str]

    See also
    --------
    ast

    """
    # pylint: disable=too-many-return-statements,too-many-branches
    etype = type(exp)
    if etype in [ast.Expr, ast.Expression]:
        return freevars(exp.body, env)
    elif etype == ast.BoolOp:
        return sum((freevars(v, env) for v in exp.values), [])
    elif etype == ast.BinOp:
        return freevars(exp.left, env) + freevars(exp.right, env)
    elif etype == ast.UnaryOp:
        return freevars(exp.operand, env)
    elif etype == ast.Lambda:
        args = exp.args
        assert isinstance(args, ast.arguments)
        argnames = [a.arg for a in args.args]
        argnames += [args.vararg.arg] if args.vararg else []
        argnames += [a.arg for a in args.kwonlyargs] if args.kwonlyargs else []
        argnames += [args.kwarg] if args.kwarg else []
        return freevars(exp.body, env + argnames)
    elif etype == ast.IfExp:
        return (freevars(exp.test, env) + freevars(exp.body, env) +
                freevars(exp.orelse, env))
    elif etype == ast.Dict:
        return sum((freevars(v, env)
                    for v in chain(exp.keys, exp.values)), [])
    elif etype == ast.Set:
        return sum((freevars(v, env) for v in exp.elts), [])
    elif etype in [ast.SetComp, ast.ListComp, ast.GeneratorExp, ast.DictComp]:
        env_ext = []
        vars_ = []
        for gen in exp.generators:
            target_names = freevars(gen.target, [])  # assigned names
            vars_iter = freevars(gen.iter, env)
            env_ext += target_names
            vars_ifs = list(chain(*(freevars(ifexp, env + target_names)
                                    for ifexp in gen.ifs or [])))
            vars_ += vars_iter + vars_ifs

        if etype == ast.DictComp:
            vars_ = (freevars(exp.key, env_ext) +
                     freevars(exp.value, env_ext) +
                     vars_)
        else:
            vars_ = freevars(exp.elt, env + env_ext) + vars_
        return vars_
    # Yield, YieldFrom???
    elif etype == ast.Compare:
        return sum((freevars(v, env)
                    for v in [exp.left] + exp.comparators), [])
    elif etype == ast.Call:
        return sum(map(lambda e: freevars(e, env),
                       chain([exp.func],
                             exp.args or [],
                             [k.value for k in exp.keywords or []])),
                   [])
    elif etype == ast.Starred:
        # a 'starred' call parameter (e.g. a and b in `f(x, *a, *b)`
        return freevars(exp.value, env)
    elif etype in [ast.Num, ast.Str, ast.Ellipsis, ast.Bytes, ast.NameConstant]:
        return []
    elif etype == ast.Constant:
        return []
    elif etype == ast.Attribute:
        return freevars(exp.value, env)
    elif etype == ast.Subscript:
        return freevars(exp.value, env) + freevars(exp.slice, env)
    elif etype == ast.Name:
        return [exp.id] if exp.id not in env else []
    elif etype == ast.List:
        return sum((freevars(e, env) for e in exp.elts), [])
    elif etype == ast.Tuple:
        return sum((freevars(e, env) for e in exp.elts), [])
    elif etype == ast.Slice:
        return sum((freevars(e, env)
                    for e in filter(None, [exp.lower, exp.upper, exp.step])),
                   [])
    elif etype == ast.ExtSlice:
        return sum((freevars(e, env) for e in exp.dims), [])
    elif etype == ast.Index:
        return freevars(exp.value, env)
    elif etype == ast.keyword:
        return freevars(exp.value, env)
    else:
        raise ValueError(exp)


class FeatureConstructorHandler(DomainContextHandler):
    """Context handler that filters descriptors"""

    def is_valid_item(self, setting, item, attrs, metas):
        """Check if descriptor `item` can be used with given domain.

        Return True if descriptor's expression contains only
        available variables and descriptors name does not clash with
        existing variables.
        """
        if item.name in attrs or item.name in metas:
            return False

        try:
            exp_ast = ast.parse(item.expression, mode="eval")
        # ast.parse can return arbitrary errors, not only SyntaxError
        # pylint: disable=broad-except
        except Exception:
            return False

        available = dict(globals()["__GLOBALS"])
        for var in attrs:
            available[sanitized_name(var)] = None
        for var in metas:
            available[sanitized_name(var)] = None

        if freevars(exp_ast, available):
            return False
        return True


class OWFeatureConstructor(OWWidget):
    name = "Feature Constructor"
    description = "Construct new features (data columns) from a set of " \
                  "existing features in the input dataset."
    category = "Transform"
    icon = "icons/FeatureConstructor.svg"
    keywords = ['function', 'lambda']
    priority = 2240

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        data = Output("Data", Orange.data.Table)

    want_main_area = False

    settingsHandler = FeatureConstructorHandler()
    descriptors = ContextSetting([])
    currentIndex = ContextSetting(-1)
    expressions_with_values = ContextSetting(False)
    settings_version = 2

    EDITORS = [
        (ContinuousDescriptor, ContinuousFeatureEditor),
        (DateTimeDescriptor, DateTimeFeatureEditor),
        (DiscreteDescriptor, DiscreteFeatureEditor),
        (StringDescriptor, StringFeatureEditor)
    ]

    class Error(OWWidget.Error):
        more_values_needed = Msg("Categorical feature {} needs more values.")
        invalid_expressions = Msg("Invalid expressions: {}.")

    class Warning(OWWidget.Warning):
        renamed_var = Msg("Recently added variable has been renamed, "
                           "to avoid duplicates.\n")

    def __init__(self):
        super().__init__()
        self.data = None
        self.editors = {}

        box = gui.vBox(self.controlArea, "Variable Definitions")

        toplayout = QHBoxLayout()
        toplayout.setContentsMargins(0, 0, 0, 0)
        box.layout().addLayout(toplayout)

        self.editorstack = QStackedWidget(
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding,
                                   QSizePolicy.MinimumExpanding)
        )

        for descclass, editorclass in self.EDITORS:
            editor = editorclass()
            editor.featureChanged.connect(self._on_modified)
            self.editors[descclass] = editor
            self.editorstack.addWidget(editor)

        self.editorstack.setEnabled(False)

        buttonlayout = QVBoxLayout(spacing=10)
        buttonlayout.setContentsMargins(0, 0, 0, 0)

        self.addbutton = QPushButton(
            "New", toolTip="Create a new variable",
            minimumWidth=120,
            shortcut=QKeySequence.New
        )

        def unique_name(fmt, reserved):
            candidates = (fmt.format(i) for i in count(1))
            return next(c for c in candidates if c not in reserved)

        def generate_newname(fmt):
            return unique_name(fmt, self.reserved_names())

        menu = QMenu(self.addbutton)
        cont = menu.addAction("Numeric")
        cont.triggered.connect(
            lambda: self.addFeature(
                ContinuousDescriptor(generate_newname("X{}"), "", 3))
        )
        disc = menu.addAction("Categorical")
        disc.triggered.connect(
            lambda: self.addFeature(
                DiscreteDescriptor(generate_newname("D{}"), "", (), False))
        )
        string = menu.addAction("Text")
        string.triggered.connect(
            lambda: self.addFeature(
                StringDescriptor(generate_newname("S{}"), ""))
        )
        datetime = menu.addAction("Date/Time")
        datetime.triggered.connect(
            lambda: self.addFeature(
                DateTimeDescriptor(generate_newname("T{}"), ""))
        )

        menu.addSeparator()
        self.duplicateaction = menu.addAction("Duplicate Selected Variable")
        self.duplicateaction.triggered.connect(self.duplicateFeature)
        self.duplicateaction.setEnabled(False)
        self.addbutton.setMenu(menu)

        self.removebutton = QPushButton(
            "Remove", toolTip="Remove selected variable",
            minimumWidth=120,
            shortcut=QKeySequence.Delete
        )
        self.removebutton.clicked.connect(self.removeSelectedFeature)

        buttonlayout.addWidget(self.addbutton)
        buttonlayout.addWidget(self.removebutton)
        buttonlayout.addStretch(10)

        toplayout.addLayout(buttonlayout, 0)
        toplayout.addWidget(self.editorstack, 10)

        # Layout for the list view
        layout = QVBoxLayout(spacing=1, margin=0)
        self.featuremodel = DescriptorModel(parent=self)

        self.featureview = QListView(
            minimumWidth=200, minimumHeight=50,
            sizePolicy=QSizePolicy(QSizePolicy.Minimum,
                                   QSizePolicy.MinimumExpanding)
        )

        self.featureview.setItemDelegate(FeatureItemDelegate(self))
        self.featureview.setModel(self.featuremodel)
        self.featureview.selectionModel().selectionChanged.connect(
            self._on_selectedVariableChanged
        )

        layout.addWidget(self.featureview)

        box.layout().addLayout(layout, 1)

        self.fix_button = gui.button(
            self.buttonsArea, self, "Upgrade Expressions",
            callback=self.fix_expressions)
        self.fix_button.setHidden(True)
        gui.button(self.buttonsArea, self, "Send", callback=self.apply, default=True)

    def setCurrentIndex(self, index):
        index = min(index, len(self.featuremodel) - 1)
        self.currentIndex = index
        if index >= 0:
            itemmodels.select_row(self.featureview, index)
            desc = self.featuremodel[min(index, len(self.featuremodel) - 1)]
            editor = self.editors[type(desc)]
            self.editorstack.setCurrentWidget(editor)
            editor.setEditorData(desc, self.data.domain if self.data else None)
        self.editorstack.setEnabled(index >= 0)
        self.duplicateaction.setEnabled(index >= 0)
        self.removebutton.setEnabled(index >= 0)

    def _on_selectedVariableChanged(self, selected, *_):
        index = selected_row(self.featureview)
        if index is not None:
            self.setCurrentIndex(index)
        else:
            self.setCurrentIndex(-1)

    def _on_modified(self):
        if self.currentIndex >= 0:
            self.Warning.clear()
            editor = self.editorstack.currentWidget()
            proposed = editor.editorData().name
            uniq = get_unique_names(self.reserved_names(self.currentIndex),
                                    proposed)

            feature = editor.editorData()
            if editor.editorData().name != uniq:
                self.Warning.renamed_var()
                feature = feature.__class__(uniq, *feature[1:])

            self.featuremodel[self.currentIndex] = feature
            self.descriptors = list(self.featuremodel)

    def setDescriptors(self, descriptors):
        """
        Set a list of variable descriptors to edit.
        """
        self.descriptors = descriptors
        self.featuremodel[:] = list(self.descriptors)

    def reserved_names(self, idx_=None):
        varnames = []
        if self.data is not None:
            varnames = [var.name for var in
                        self.data.domain.variables + self.data.domain.metas]
        varnames += [desc.name for idx, desc in enumerate(self.featuremodel)
                     if idx != idx_]
        return set(varnames)

    @Inputs.data
    @check_sql_input
    def setData(self, data=None):
        """Set the input dataset."""
        self.closeContext()

        self.data = data
        self.expressions_with_values = False

        self.descriptors = []
        self.currentIndex = -1
        if self.data is not None:
            self.openContext(data)

        # disconnect from the selection model while reseting the model
        selmodel = self.featureview.selectionModel()
        selmodel.selectionChanged.disconnect(self._on_selectedVariableChanged)

        self.featuremodel[:] = list(self.descriptors)
        self.setCurrentIndex(self.currentIndex)

        selmodel.selectionChanged.connect(self._on_selectedVariableChanged)
        self.fix_button.setHidden(not self.expressions_with_values)
        self.editorstack.setEnabled(self.currentIndex >= 0)

    def handleNewSignals(self):
        if self.data is not None:
            self.apply()
        else:
            self.Outputs.data.send(None)
            self.fix_button.setHidden(True)

    def addFeature(self, descriptor):
        self.featuremodel.append(descriptor)
        self.setCurrentIndex(len(self.featuremodel) - 1)
        editor = self.editorstack.currentWidget()
        editor.nameedit.setFocus()
        editor.nameedit.selectAll()

    def removeFeature(self, index):
        del self.featuremodel[index]
        index = selected_row(self.featureview)
        if index is not None:
            self.setCurrentIndex(index)
        elif index is None and self.featuremodel.rowCount():
            # Deleting the last item clears selection
            self.setCurrentIndex(self.featuremodel.rowCount() - 1)

    def removeSelectedFeature(self):
        if self.currentIndex >= 0:
            self.removeFeature(self.currentIndex)

    def duplicateFeature(self):
        desc = self.featuremodel[self.currentIndex]
        self.addFeature(copy.deepcopy(desc))

    @staticmethod
    def check_attrs_values(attr, data):
        for i in range(len(data)):
            for var in attr:
                if not math.isnan(data[i, var]) \
                        and int(data[i, var]) >= len(var.values):
                    return var.name
        return None

    def _validate_descriptors(self, desc):

        def validate(source):
            try:
                return validate_exp(ast.parse(source, mode="eval"))
            # ast.parse can return arbitrary errors, not only SyntaxError
            # pylint: disable=broad-except
            except Exception:
                return False

        final = []
        invalid = []
        for d in desc:
            if validate(d.expression):
                final.append(d)
            else:
                final.append(d._replace(expression=""))
                invalid.append(d)

        if invalid:
            self.Error.invalid_expressions(", ".join(s.name for s in invalid))

        return final

    def apply(self):
        def report_error(err):
            log = logging.getLogger(__name__)
            log.error("", exc_info=True)
            self.error("".join(format_exception_only(type(err), err)).rstrip())

        self.Error.clear()

        if self.data is None:
            return

        desc = list(self.featuremodel)
        desc = self._validate_descriptors(desc)
        try:
            new_variables = construct_variables(
                desc, self.data, self.expressions_with_values)
        # user's expression can contain arbitrary errors
        except Exception as err:  # pylint: disable=broad-except
            report_error(err)
            return

        attrs = [var for var in new_variables if var.is_primitive()]
        metas = [var for var in new_variables if not var.is_primitive()]
        new_domain = Orange.data.Domain(
            self.data.domain.attributes + tuple(attrs),
            self.data.domain.class_vars,
            metas=self.data.domain.metas + tuple(metas)
        )

        try:
            for variable in new_variables:
                variable.compute_value.mask_exceptions = False
            data = self.data.transform(new_domain)
        # user's expression can contain arbitrary errors
        # pylint: disable=broad-except
        except Exception as err:
            report_error(err)
            return
        finally:
            for variable in new_variables:
                variable.compute_value.mask_exceptions = True

        disc_attrs_not_ok = self.check_attrs_values(
            [var for var in attrs if var.is_discrete], data)
        if disc_attrs_not_ok:
            self.Error.more_values_needed(disc_attrs_not_ok)
            return

        self.Outputs.data.send(data)

    def send_report(self):
        items = OrderedDict()
        for feature in self.featuremodel:
            if isinstance(feature, DiscreteDescriptor):
                desc = "categorical"
                if feature.values:
                    desc += " with values " \
                            + ", ".join(f"'{val}'" for val in feature.values)
                if feature.ordered:
                    desc += "; ordered"
            elif isinstance(feature, ContinuousDescriptor):
                desc = "numeric"
            elif isinstance(feature, DateTimeDescriptor):
                desc = "date/time"
            else:
                desc = "text"
            items[feature.name] = f"{feature.expression} ({desc})"
        self.report_items(
            report.plural("Constructed feature{s}", len(items)), items)

    def fix_expressions(self):
        dlg = QMessageBox(
            QMessageBox.Question,
            "Fix Expressions",
            "This widget's behaviour has changed. Values of categorical "
            "variables are now inserted as their textual representations "
            "(strings); previously they appeared as integer numbers, with an "
            "attribute '.value' that contained the text.\n\n"
            "The widget currently runs in compatibility mode. After "
            "expressions are updated, manually check for their correctness.")
        dlg.addButton("Update", QMessageBox.ApplyRole)
        dlg.addButton("Cancel", QMessageBox.RejectRole)
        if dlg.exec() == QMessageBox.RejectRole:
            return

        def fixer(mo):
            var = domain[mo.group(2)]
            if mo.group(3) == ".value":  # uses string; remove `.value`
                return "".join(mo.group(1, 2, 4))
            # Uses ints: get them by indexing
            return mo.group(1) + "{" + \
                   ", ".join(f"'{val}': {i}"
                             for i, val in enumerate(var.values)) + \
                   f"}}[{var.name}]" + mo.group(4)

        domain = self.data.domain
        disc_vars = "|".join(f"{var.name}"
                             for var in chain(domain.variables, domain.metas)
                             if var.is_discrete)
        expr = re.compile(r"(^|\W)(" + disc_vars + r")(\.value)?(\W|$)")
        self.descriptors[:] = [
            descriptor._replace(
                expression=expr.sub(fixer, descriptor.expression))
            for descriptor in self.descriptors]

        self.expressions_with_values = False
        self.fix_button.hide()
        index = self.currentIndex
        self.featuremodel[:] = list(self.descriptors)
        self.setCurrentIndex(index)
        self.apply()

    @classmethod
    def migrate_context(cls, context, version):
        if version is None or version < 2:
            used_vars = set(chain(*(
                freevars(ast.parse(descriptor.expression, mode="eval"), [])
                for descriptor in context.values["descriptors"]
                if descriptor.expression)))
            disc_vars = {name
                         for (name, vtype) in chain(context.attributes.items(),
                                                    context.metas.items())
                         if vtype == 1}
            if used_vars & disc_vars:
                context.values["expressions_with_values"] = True


def validate_exp(exp):
    """
    Validate an `ast.AST` expression.

    Only expressions with no list,set,dict,generator comprehensions
    are accepted.

    Parameters
    ----------
    exp : ast.AST
        A parsed abstract syntax tree

    """
    # pylint: disable=too-many-branches
    if not isinstance(exp, ast.AST):
        raise TypeError("exp is not a 'ast.AST' instance")

    etype = type(exp)
    if etype in [ast.Expr, ast.Expression]:
        return validate_exp(exp.body)
    elif etype == ast.BoolOp:
        return all(map(validate_exp, exp.values))
    elif etype == ast.BinOp:
        return all(map(validate_exp, [exp.left, exp.right]))
    elif etype == ast.UnaryOp:
        return validate_exp(exp.operand)
    elif etype == ast.IfExp:
        return all(map(validate_exp, [exp.test, exp.body, exp.orelse]))
    elif etype == ast.Dict:
        return all(map(validate_exp, chain(exp.keys, exp.values)))
    elif etype == ast.Set:
        return all(map(validate_exp, exp.elts))
    elif etype == ast.Compare:
        return all(map(validate_exp, [exp.left] + exp.comparators))
    elif etype == ast.Call:
        subexp = chain([exp.func], exp.args or [],
                       [k.value for k in exp.keywords or []])
        return all(map(validate_exp, subexp))
    elif etype == ast.Starred:
        assert isinstance(exp.ctx, ast.Load)
        return validate_exp(exp.value)
    elif etype in [ast.Num, ast.Str, ast.Bytes, ast.Ellipsis, ast.NameConstant]:
        return True
    elif etype == ast.Constant:
        return True
    elif etype == ast.Attribute:
        return True
    elif etype == ast.Subscript:
        return all(map(validate_exp, [exp.value, exp.slice]))
    elif etype in {ast.List, ast.Tuple}:
        assert isinstance(exp.ctx, ast.Load)
        return all(map(validate_exp, exp.elts))
    elif etype == ast.Name:
        return True
    elif etype == ast.Slice:
        return all(map(validate_exp,
                       filter(None, [exp.lower, exp.upper, exp.step])))
    elif etype == ast.ExtSlice:
        return all(map(validate_exp, exp.dims))
    elif etype == ast.Index:
        return validate_exp(exp.value)
    elif etype == ast.keyword:
        return validate_exp(exp.value)
    else:
        raise ValueError(exp)


def construct_variables(descriptions, data, use_values=False):
    # subs
    variables = []
    source_vars = data.domain.variables + data.domain.metas
    for desc in descriptions:
        desc, func = bind_variable(desc, source_vars, data, use_values)
        var = make_variable(desc, func)
        variables.append(var)
    return variables


def sanitized_name(name):
    sanitized = re.sub(r"\W", "_", name)
    if sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


def bind_variable(descriptor, env, data, use_values):
    """
    (descriptor, env) ->
        (descriptor, (instance -> value) | (table -> value list))
    """
    if not descriptor.expression.strip():
        return descriptor, FeatureFunc("nan", [], {"nan": float("nan")})

    exp_ast = ast.parse(descriptor.expression, mode="eval")
    freev = unique(freevars(exp_ast, []))
    variables = {unicodedata.normalize("NFKC", sanitized_name(v.name)): v
                 for v in env}
    source_vars = [(name, variables[name]) for name in freev
                   if name in variables]

    values = {}
    cast = None
    nan = float("nan")

    if isinstance(descriptor, DiscreteDescriptor):
        if not descriptor.values:
            str_func = FeatureFunc(descriptor.expression, source_vars,
                                   use_values=use_values)
            values = sorted({str(x) for x in str_func(data)})
            values = {name: i for i, name in enumerate(values)}
            descriptor = descriptor._replace(values=values)

            def cast(x):  # pylint: disable=function-redefined
                return values.get(x, nan)

        else:
            values = [sanitized_name(v) for v in descriptor.values]
            values = {name: i for i, name in enumerate(values)}

    if isinstance(descriptor, DateTimeDescriptor):
        parse = Orange.data.TimeVariable("_").parse

        def cast(e):  # pylint: disable=function-redefined
            if isinstance(e, (int, float)):
                return e
            if e == "" or e is None:
                return np.nan
            return parse(e)

    func = FeatureFunc(descriptor.expression, source_vars, values, cast,
                       use_values=use_values)
    return descriptor, func


def make_lambda(expression, args, env=None):
    # type: (ast.Expression, List[str], Dict[str, Any]) -> types.FunctionType
    """
    Create an lambda function from a expression AST.

    Parameters
    ----------
    expression : ast.Expression
        The body of the lambda.
    args : List[str]
        A list of positional argument names
    env : Optional[Dict[str, Any]]
        Extra environment to capture in the lambda's closure.

    Returns
    -------
    func : types.FunctionType
    """
    # lambda *{args}* : EXPRESSION
    lambda_ = ast.Lambda(
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=arg, annotation=None) for arg in args],
            varargs=None,
            varargannotation=None,
            kwonlyargs=[],
            kwarg=None,
            kwargannotation=None,
            defaults=[],
            kw_defaults=[]),
        body=expression.body,
    )
    lambda_ = ast.copy_location(lambda_, expression.body)
    # lambda **{env}** : lambda *{args}*: EXPRESSION
    outer = ast.Lambda(
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name, annotation=None) for name in (env or {})],
            varargs=None,
            varargannotation=None,
            kwonlyargs=[],
            kwarg=None,
            kwargannotation=None,
            defaults=[],
            kw_defaults=[],
        ),
        body=lambda_,
    )
    exp = ast.Expression(body=outer, lineno=1, col_offset=0)
    ast.fix_missing_locations(exp)
    GLOBALS = __GLOBALS.copy()
    GLOBALS["__builtins__"] = {}
    # pylint: disable=eval-used
    fouter = eval(compile(exp, "<lambda>", "eval"), GLOBALS)
    assert isinstance(fouter, types.FunctionType)
    finner = fouter(**env)
    assert isinstance(finner, types.FunctionType)
    return finner


__ALLOWED = [
    "Ellipsis", "False", "None", "True", "abs", "all", "any", "acsii",
    "bin", "bool", "bytearray", "bytes", "chr", "complex", "dict",
    "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "getattr", "hasattr", "hash", "hex", "id", "int", "iter", "len",
    "list", "map", "memoryview", "next", "object",
    "oct", "ord", "pow", "range", "repr", "reversed", "round",
    "set", "slice", "sorted", "str", "tuple", "type",
    "zip"
]

__GLOBALS = {name: getattr(builtins, name) for name in __ALLOWED
             if hasattr(builtins, name)}

__GLOBALS.update({name: getattr(math, name) for name in dir(math)
                  if not name.startswith("_")})

__GLOBALS.update({
    "normalvariate": random.normalvariate,
    "gauss": random.gauss,
    "expovariate": random.expovariate,
    "gammavariate": random.gammavariate,
    "betavariate": random.betavariate,
    "lognormvariate": random.lognormvariate,
    "paretovariate": random.paretovariate,
    "vonmisesvariate": random.vonmisesvariate,
    "weibullvariate": random.weibullvariate,
    "triangular": random.triangular,
    "uniform": random.uniform,
    "nanmean": lambda *args: np.nanmean(args),
    "nanmin": lambda *args: np.nanmin(args),
    "nanmax": lambda *args: np.nanmax(args),
    "nansum": lambda *args: np.nansum(args),
    "nanstd": lambda *args: np.nanstd(args),
    "nanmedian": lambda *args: np.nanmedian(args),
    "nancumsum": lambda *args: np.nancumsum(args),
    "nancumprod": lambda *args: np.nancumprod(args),
    "nanargmax": lambda *args: np.nanargmax(args),
    "nanargmin": lambda *args: np.nanargmin(args),
    "nanvar": lambda *args: np.nanvar(args),
    "mean": lambda *args: np.mean(args),
    "min": lambda *args: np.min(args),
    "max": lambda *args: np.max(args),
    "sum": lambda *args: np.sum(args),
    "std": lambda *args: np.std(args),
    "median": lambda *args: np.median(args),
    "cumsum": lambda *args: np.cumsum(args),
    "cumprod": lambda *args: np.cumprod(args),
    "argmax": lambda *args: np.argmax(args),
    "argmin": lambda *args: np.argmin(args),
    "var": lambda *args: np.var(args)})


class FeatureFunc:
    """
    Parameters
    ----------
    expression : str
        An expression string
    args : List[Tuple[str, Orange.data.Variable]]
        A list of (`name`, `variable`) tuples where `name` is the name of
        a variable as used in `expression`, and `variable` is the variable
        instance used to extract the corresponding column/value from a
        Table/Instance.
    extra_env : Optional[Dict[str, Any]]
        Extra environment specifying constant values to be made available
        in expression. It must not shadow names in `args`
    cast: Optional[Callable]
        A function for casting the expressions result to the appropriate
        type (e.g. string representation of date/time variables to floats)
    """
    def __init__(self, expression, args, extra_env=None, cast=None, use_values=False):
        self.expression = expression
        self.args = args
        self.extra_env = dict(extra_env or {})
        self.func = make_lambda(ast.parse(expression, mode="eval"),
                                [name for name, _ in args], self.extra_env)
        self.cast = cast
        self.mask_exceptions = True
        self.use_values = use_values

    def __call__(self, instance, *_):
        if isinstance(instance, Orange.data.Table):
            return [self(inst) for inst in instance]
        else:
            try:
                args = [str(instance[var]) if var.is_string
                    else var.values[int(instance[var])] if var.is_discrete and not self.use_values
                    else instance[var]
                    for _, var in self.args]
                y = self.func(*args)
            # user's expression can contain arbitrary errors
            # this also covers missing attributes
            except:  # pylint: disable=bare-except
                if not self.mask_exceptions:
                    raise
                return np.nan
            if self.cast:
                y = self.cast(y)
            return y

    def __reduce__(self):
        return type(self), (self.expression, self.args,
                            self.extra_env, self.cast)

    def __repr__(self):
        return "{0.__name__}{1!r}".format(*self.__reduce__())


def unique(seq):
    seen = set()
    unique_el = []
    for el in seq:
        if el not in seen:
            unique_el.append(el)
            seen.add(el)
    return unique_el


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWFeatureConstructor).run(Orange.data.Table("iris"))
