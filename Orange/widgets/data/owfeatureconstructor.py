"""
Feature Constructor

A widget for defining (constructing) new features from values
of other variables.

"""
import sys
import re
import copy
import functools
import builtins
import math
import random
from collections import namedtuple, OrderedDict
from itertools import chain, count

from AnyQt.QtWidgets import (
    QSizePolicy, QAbstractItemView, QComboBox, QFormLayout, QLineEdit,
    QHBoxLayout, QVBoxLayout, QStackedWidget, QStyledItemDelegate,
    QPushButton, QMenu, QListView, QFrame
)
from AnyQt.QtGui import QIcon, QKeySequence
from AnyQt.QtCore import Qt, pyqtSignal as Signal, pyqtProperty as Property

import Orange
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler
from Orange.widgets.utils import itemmodels, vartype
from Orange.widgets.utils.sql import check_sql_input
from Orange.canvas import report
from Orange.widgets.widget import OWWidget, Msg

FeatureDescriptor = \
    namedtuple("FeatureDescriptor", ["name", "expression"])

ContinuousDescriptor = \
    namedtuple("ContinuousDescriptor",
               ["name", "expression", "number_of_decimals"])
DiscreteDescriptor = \
    namedtuple("DiscreteDescriptor",
               ["name", "expression", "values", "base_value", "ordered"])

StringDescriptor = namedtuple("StringDescriptor", ["name", "expression"])


def make_variable(descriptor, compute_value):
    if isinstance(descriptor, ContinuousDescriptor):
        return Orange.data.ContinuousVariable(
            descriptor.name,
            descriptor.number_of_decimals,
            compute_value)
    elif isinstance(descriptor, DiscreteDescriptor):
        return Orange.data.DiscreteVariable(
            descriptor.name,
            values=descriptor.values,
            ordered=descriptor.ordered,
            base_value=descriptor.base_value,
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
    FUNCTIONS = dict(chain([(key, val) for key, val in math.__dict__.items()
                            if not key.startswith("_")],
                           [("str", str)]))
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
            placeholderText="Expression..."
        )

        self.attrs_model = itemmodels.VariableListModel(
            ["Select Feature"], parent=self)
        self.attributescb = QComboBox(
            minimumContentsLength=16,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon,
            sizePolicy=QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        )
        self.attributescb.setModel(self.attrs_model)

        sorted_funcs = sorted(self.FUNCTIONS)
        self.funcs_model = itemmodels.PyListModelTooltip()
        self.funcs_model.setParent(self)

        self.funcs_model[:] = chain(["Select Function"], sorted_funcs)
        self.funcs_model.tooltips[:] = chain(
            [''],
            [self.FUNCTIONS[func].__doc__ for func in sorted_funcs])

        self.functionscb = QComboBox(
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
        if not type(modified) is bool:
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
        if domain is not None and (domain or domain.metas):
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

    def editorData(self):
        return ContinuousDescriptor(
            name=self.nameedit.text(),
            number_of_decimals=3,
            expression=self.expressionedit.text()
        )


class DiscreteFeatureEditor(FeatureEditor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.valuesedit = QLineEdit()
        self.valuesedit.textChanged.connect(self._invalidate)

        layout = self.layout()
        layout.addRow(self.tr("Values"), self.valuesedit)

    def setEditorData(self, data, domain):
        self.valuesedit.setText(
            ", ".join(v.replace(",", r"\,") for v in data.values))

        super().setEditorData(data, domain)

    def editorData(self):
        values = self.valuesedit.text()
        values = re.split(r"(?<!\\),", values)
        values = tuple(v.replace(r"\,", ",").strip() for v in values)
        return DiscreteDescriptor(
            name=self.nameedit.text(),
            values=values,
            base_value=-1,
            ordered=False,
            expression=self.expressionedit.text()
        )


class StringFeatureEditor(FeatureEditor):
    def editorData(self):
        return StringDescriptor(
            name=self.nameedit.text(),
            expression=self.expressionedit.text()
        )


_VarMap = {
    DiscreteDescriptor: vartype(Orange.data.DiscreteVariable()),
    ContinuousDescriptor: vartype(Orange.data.ContinuousVariable()),
    StringDescriptor: vartype(Orange.data.StringVariable())
}


@functools.lru_cache(20)
def variable_icon(dtype):
    vtype = _VarMap.get(dtype, dtype)
    try:
        return gui.attributeIconDict[vtype]
    except Exception:
        return QIcon()


class FeatureItemDelegate(QStyledItemDelegate):
    def displayText(self, value, locale):
        return value.name + " := " + value.expression


class DescriptorModel(itemmodels.PyListModel):
    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DecorationRole:
            value = self[index.row()]
            return variable_icon(type(value))
        else:
            return super().data(index, role)


class FeatureConstructorSettingsHandler(DomainContextHandler):
    """Context handler that filters descriptors"""

    def is_valid_item(self, setting, descriptor, attrs, metas):
        """Check if descriptor can be used with given domain.

        Return True if descriptor's expression contains only
        available variables and descriptors name does not clash with
        existing variables.
        """
        if descriptor.name in attrs or descriptor.name in metas:
            return False

        try:
            exp_ast = ast.parse(descriptor.expression, mode="eval")
        except Exception:
            return False

        for name in freevars(exp_ast, []):
            if not (name in attrs or name in metas):
                return False
        return True


class OWFeatureConstructor(OWWidget):
    name = "Feature Constructor"
    description = "Construct new features (data columns) from a set of " \
                  "existing features in the input data set."
    icon = "icons/FeatureConstructor.svg"
    inputs = [("Data", Orange.data.Table, "setData")]
    outputs = [("Data", Orange.data.Table)]

    want_main_area = False

    settingsHandler = FeatureConstructorSettingsHandler()
    descriptors = ContextSetting([])
    currentIndex = ContextSetting(-1)

    EDITORS = [
        (ContinuousDescriptor, ContinuousFeatureEditor),
        (DiscreteDescriptor, DiscreteFeatureEditor),
        (StringDescriptor, StringFeatureEditor)
    ]

    class Error(OWWidget.Error):
        more_values_needed = Msg("Discrete feature {} needs more values.")

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

        def reserved_names():
            varnames = []
            if self.data is not None:
                varnames = [var.name for var in
                            self.data.domain.variables + self.data.domain.metas]
            varnames += [desc.name for desc in self.featuremodel]
            return set(varnames)

        def generate_newname(fmt):
            return unique_name(fmt, reserved_names())

        menu = QMenu(self.addbutton)
        cont = menu.addAction("Continuous")
        cont.triggered.connect(
            lambda: self.addFeature(
                ContinuousDescriptor(generate_newname("X{}"), "", 3))
        )
        disc = menu.addAction("Discrete")
        disc.triggered.connect(
            lambda: self.addFeature(
                DiscreteDescriptor(generate_newname("D{}"), "",
                                   ("A", "B"), -1, False))
        )
        string = menu.addAction("String")
        string.triggered.connect(
            lambda: self.addFeature(
                StringDescriptor(generate_newname("S{}"), ""))
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
            minimumWidth=200,
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

        box = gui.hBox(self.controlArea)
        box.layout().addWidget(self.report_button)
        self.report_button.setMinimumWidth(180)
        gui.rubber(box)
        commit = gui.button(box, self, "Send", callback=self.apply,
                            default=True)
        commit.setMinimumWidth(180)

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
            editor = self.editorstack.currentWidget()
            self.featuremodel[self.currentIndex] = editor.editorData()
            self.descriptors = list(self.featuremodel)

    def setDescriptors(self, descriptors):
        """
        Set a list of variable descriptors to edit.
        """
        self.descriptors = descriptors
        self.featuremodel[:] = list(self.descriptors)

    @check_sql_input
    def setData(self, data=None):
        """Set the input dataset."""
        self.closeContext()

        self.data = data

        if self.data is not None:
            descriptors = list(self.descriptors)
            currindex = self.currentIndex
            self.descriptors = []
            self.currentIndex = -1
            self.openContext(data)

            if descriptors != self.descriptors or \
                    self.currentIndex != currindex:
                # disconnect from the selection model while reseting the model
                selmodel = self.featureview.selectionModel()
                selmodel.selectionChanged.disconnect(
                    self._on_selectedVariableChanged)

                self.featuremodel[:] = list(self.descriptors)
                self.setCurrentIndex(self.currentIndex)

                selmodel.selectionChanged.connect(
                    self._on_selectedVariableChanged)

        self.editorstack.setEnabled(self.currentIndex >= 0)

    def handleNewSignals(self):
        if self.data is not None:
            self.apply()
        else:
            self.send("Data", None)

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
        elif index is None and len(self.featuremodel) > 0:
            # Deleting the last item clears selection
            self.setCurrentIndex(len(self.featuremodel) - 1)

    def removeSelectedFeature(self):
        if self.currentIndex >= 0:
            self.removeFeature(self.currentIndex)

    def duplicateFeature(self):
        desc = self.featuremodel[self.currentIndex]
        self.addFeature(copy.deepcopy(desc))

    def check_attrs_values(self, attr, data):
        for i in range(len(data)):
            for var in attr:
                if not math.isnan(data[i, var]) \
                        and int(data[i, var]) >= len(var.values):
                    return var.name
        return None

    def apply(self):
        if self.data is None:
            return

        desc = list(self.featuremodel)

        def validate(source):
            try:
                return validate_exp(ast.parse(source, mode="eval"))
            except Exception:
                return False

        def remove_invalid_expression(desc):
            return (desc if validate(desc.expression)
                    else desc._replace(expression=""))

        desc = map(remove_invalid_expression, desc)
        source_vars = tuple(self.data.domain) + self.data.domain.metas
        new_variables = construct_variables(desc, source_vars)

        attrs = [var for var in new_variables if var.is_primitive()]
        metas = [var for var in new_variables if not var.is_primitive()]
        new_domain = Orange.data.Domain(
            self.data.domain.attributes + tuple(attrs),
            self.data.domain.class_vars,
            metas=self.data.domain.metas + tuple(metas)
        )

        self.Error.clear()
        try:
            data = Orange.data.Table(new_domain, self.data)
        except Exception as err:
            self.error(err.args[0])
            return
        disc_attrs_not_ok = self.check_attrs_values(
            [var for var in attrs if var.is_discrete], data)
        if disc_attrs_not_ok:
            self.Error.more_values_needed(disc_attrs_not_ok)
            return

        self.send("Data", data)

    def send_report(self):
        items = OrderedDict()
        for feature in self.featuremodel:
            if isinstance(feature, DiscreteDescriptor):
                items[feature.name] = "{} (discrete with values {}{})".format(
                    feature.expression, feature.values,
                    "; ordered" * feature.ordered)
            elif isinstance(feature, ContinuousDescriptor):
                items[feature.name] = "{} (numeric)".format(feature.expression)
            else:
                items[feature.name] = "{} (text)".format(feature.expression)
        self.report_items(
            report.plural("Constructed feature{s}", len(items)), items)



import ast


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
    elif etype == ast.Call and sys.version_info < (3, 5):
        return sum((freevars(e, env)
                    for e in [exp.func] + (exp.args or []) +
                    ([k.value for k in exp.keywords or []]) +
                    ([exp.starargs] if exp.starargs else []) +
                    ([exp.kwargs] if exp.kwargs else [])),
                   [])
    elif etype == ast.Call:
        return sum(map(lambda e: freevars(e, env),
                       chain([exp.func],
                             exp.args or [],
                             [k.value for k in exp.keywords or []])),
                   [])
    elif sys.version_info >= (3, 5) and etype == ast.Starred:
        # a 'starred' call parameter (e.g. a and b in `f(x, *a, *b)`
        return freevars(exp.value, env)
    elif etype in [ast.Num, ast.Str, ast.Ellipsis, ast.Bytes]:
        return []
    elif sys.version_info >= (3, 4) and etype == ast.NameConstant:
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
        if sys.version_info < (3, 5):
            extra = [exp.starargs, exp.kwargs]
            subexp = chain(subexp, *filter(None, extra))
        return all(map(validate_exp, subexp))
    elif sys.version_info >= (3, 5) and etype == ast.Starred:
        assert isinstance(exp.ctx, ast.Load)
        return validate_exp(exp.value)
    elif etype in [ast.Num, ast.Str, ast.Bytes, ast.Ellipsis]:
        return True
    elif sys.version_info >= (3, 4) and etype == ast.NameConstant:
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


def construct_variables(descriptions, source_vars):
    # subs
    variables = []
    for desc in descriptions:
        _, func = bind_variable(desc, source_vars)
        var = make_variable(desc, func)
        variables.append(var)
    return variables


def sanitized_name(name):
    return re.sub(r"\W", "_", name)


def bind_variable(descriptor, env):
    """
    (descriptor, env) ->
        (descriptor, (instance -> value) | (table -> value list))
    """
    if not descriptor.expression.strip():
        return (descriptor, lambda _: float("nan"))

    exp_ast = ast.parse(descriptor.expression, mode="eval")
    freev = unique(freevars(exp_ast, []))
    variables = {sanitized_name(v.name): v for v in env}
    source_vars = [(name, variables[name]) for name in freev
                   if name in variables]

    values = []
    if isinstance(descriptor, DiscreteDescriptor):
        values = [sanitized_name(v) for v in descriptor.values]
    return descriptor, FeatureFunc(exp_ast, source_vars, values)


def make_lambda(expression, args, values):
    def make_arg(name):
        if sys.version_info >= (3, 0):
            return ast.arg(arg=name, annotation=None)
        else:
            return ast.Name(id=arg, ctx=ast.Param(), lineno=1, col_offset=0)

    lambda_ = ast.Lambda(
        args=ast.arguments(
            args=[make_arg(arg) for arg in args + values],
            varargs=None,
            varargannotation=None,
            kwonlyargs=[],
            kwarg=None,
            kwargannotation=None,
            defaults=[ast.Num(i) for i in range(len(values))],
            kw_defaults=[]),
        body=expression.body,
    )
    lambda_ = ast.copy_location(lambda_, expression.body)
    exp = ast.Expression(body=lambda_, lineno=1, col_offset=0)
    ast.dump(exp)
    ast.fix_missing_locations(exp)
    GLOBALS = __GLOBALS.copy()
    GLOBALS["__builtins__"] = {}
    return eval(compile(exp, "<lambda>", "eval"), GLOBALS)


__ALLOWED = [
    "Ellipsis", "False", "None", "True", "abs", "all", "any", "acsii",
    "bin", "bool", "bytearray", "bytes", "chr", "complex", "dict",
    "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "getattr", "hasattr", "hash", "hex", "id", "int", "iter", "len",
    "list", "map", "max", "memoryview", "min", "next", "object",
    "oct", "ord", "pow", "range", "repr", "reversed", "round",
    "set", "slice", "sorted", "str", "sum", "tuple", "type",
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
    "uniform": random.uniform}
)


class FeatureFunc:
    def __init__(self, expression, args, values):
        self.expression = expression
        self.args = args
        self.values = values
        self.func = make_lambda(expression, [name for name, _ in args], values)

    def __call__(self, instance, *_):
        if isinstance(instance, Orange.data.Table):
            return [self(inst) for inst in instance]
        else:
            args = [instance[var] for _, var in self.args]
            return self.func(*args)


def unique(seq):
    seen = set()
    unique_el = []
    for el in seq:
        if el not in seen:
            unique_el.append(el)
            seen.add(el)
    return unique_el


def main(argv=sys.argv):
    from AnyQt.QtWidgets import QApplication
    app = QApplication(list(argv))
    argv = app.arguments()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "iris"

    w = OWFeatureConstructor()
    w.show()
    w.raise_()
    data = Orange.data.Table(filename)
    w.setData(data)
    w.handleNewSignals()
    app.exec_()
    w.setData(None)
    w.handleNewSignals()
    w.saveSettings()
    return 0

if __name__ == "__main__":
    sys.exit(main())
