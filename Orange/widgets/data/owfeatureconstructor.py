"""
Feature Constructor

A widget for defining (constructing) new features from values
of other variables.

"""
import sys
import re
import copy
import unicodedata
import functools
import builtins
import math
import random
from collections import namedtuple, Counter
from itertools import chain
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt, QEvent, pyqtSignal as Signal, pyqtProperty as Property

import Orange

from Orange.widgets import widget, gui
from Orange.widgets.settings import DomainContextHandler, Setting, ContextSetting
from Orange.widgets.utils import itemmodels, vartype

from .owpythonscript import PythonSyntaxHighlighter

FeatureDescriptor = \
    namedtuple("FeatureDescriptor", ["name", "expression"])

ContinuousDescriptor = \
    namedtuple("ContinuousDescriptor",
               ["name", "expression", "number_of_decimals"])
DiscreteDescriptor = \
    namedtuple("DiscreteDescriptor",
               ["name", "expression", "values", "base_value", "ordered"])

StringDescriptor = namedtuple("StringDescriptor", ["name", "expression"])


@functools.lru_cache(50)
def make_variable(descriptor, compute_value=None):
    if compute_value is None:
        if descriptor.expression.strip():
            compute_value = \
                lambda instance: eval(descriptor.expression,
                                      {"instance": instance, "_": instance})
        else:
            compute_value = lambda _: float("nan")

    if isinstance(descriptor, ContinuousDescriptor):
        return Orange.data.ContinuousVariable(descriptor.name, descriptor.number_of_decimals, compute_value)
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


def is_valid_expression(exp):
    try:
        ast.parse(exp, mode="eval")
        return True
    except Exception:
        return False


class ActionToolBarButton(QtGui.QToolButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ActionToolBar(QtGui.QFrame):
    iconSizeChanged = Signal(QtCore.QSize)
    actionTriggered = Signal(QtGui.QAction)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QtGui.QHBoxLayout(spacing=1)
        layout.setContentsMargins(0, 0, 0, 0)

        if "sizePolicy" not in kwargs:
            self.setSizePolicy(QSizePolicy.MinimumExpanding,
                               QSizePolicy.Minimum)

        self.setLayout(layout)
        layout.addStretch()

        self._actions = []

    def clear(self):
        for action in reversed(self.actions()):
            self.removeAction(action)

    def iconSize(self):
        if self._iconSize is None:
            style = self.style()
            pm = style.pixelMetric(QtGui.QStyle.PM_ToolBarIconSize)
            return QtCore.QSize(pm, pm)
        else:
            return self._iconSize

    def setIconSize(self, size):
        if self._iconSize != size:
            changed = self.iconSize() != size
            self._iconSize = size
            if changed:
                self.iconSizeChanged.emit(self.iconSize())

    def buttonForAction(self, action):
        for ac, button in self._actions:
            if action is ac:
                return button
        return None

    def actionEvent(self, event):
        super().actionEvent(event)

        if event.type() == QEvent.ActionAdded:
            self._insertActionBefore(event.action(), event.before())
        elif event.type() == QEvent.ActionRemoved:
            self._removeAction(event.action())
        elif event.type() == QEvent.ActionChanged:
            self._updateAction(event.action())

    def _insertActionBefore(self, action, before=None):
        index = len(self._actions)
        if action is not None:
            actions = [a for a, _ in self._actions]
            try:
                index = actions.index(before)
            except ValueError:
                pass

        button = self._button(action)
        self._actions.insert(index, (action, button))
        self.layout().insertWidget(index, button)

        button.triggered.connect(self.actionTriggered)

    def _removeAction(self, action):
        actions = [a for a, _ in self._actions]
        try:
            index = actions.index(action)
        except ValueError:
            raise
        else:
            _, button = self._actions[index]
            self.layout().takeAt(index)
            button.hide()
            button.deleteLater()
            del self._actions[index]

    def _updateAction(self, action):
        pass

    def _button(self, action):
        b = ActionToolBarButton(
            toolButtonStyle=Qt.ToolButtonIconOnly,
            sizePolicy=QSizePolicy(QSizePolicy.Minimum,
                                   QSizePolicy.Minimum)
        )
        b.setDefaultAction(action)
        b.setPopupMode(QtGui.QToolButton.InstantPopup)
        return b


def selected_row(view):
    if view.selectionMode() in [QtGui.QAbstractItemView.MultiSelection,
                                QtGui.QAbstractItemView.ExtendedSelection]:
        raise ValueError("invalid 'selectionMode'")

    sel_model = view.selectionModel()
    indexes = sel_model.selectedRows()
    if indexes:
        assert len(indexes) == 1
        return indexes[0].row()
    else:
        return None


class FeatureEditor(QtGui.QFrame):
    FUNCTIONS = dict(chain([(key, val) for key, val in math.__dict__.items()
                            if not key.startswith("_")],
                           [("str", str)]))
    featureChanged = Signal()
    featureEdited = Signal()

    modifiedChanged = Signal([], [bool])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QtGui.QFormLayout(
            fieldGrowthPolicy=QtGui.QFormLayout.ExpandingFieldsGrow
        )
        self.nameedit = QtGui.QLineEdit(
            sizePolicy=QSizePolicy(QSizePolicy.Minimum,
                                   QSizePolicy.Fixed)
        )
        self.expressionedit = QtGui.QLineEdit()

        self.attrs_model = itemmodels.VariableListModel(["Select feature"])
        self.attributescb = QtGui.QComboBox(
            minimumContentsLength=12,
            sizeAdjustPolicy=QtGui.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.attributescb.setModel(self.attrs_model)

        sorted_funcs = sorted(self.FUNCTIONS)
        self.funcs_model = itemmodels.PyListModelTooltip()
        self.funcs_model[:] = chain(["Select function"], sorted_funcs)
        self.funcs_model.tooltips[:] = chain(
            [''],
            [self.FUNCTIONS[func].__doc__ for func in sorted_funcs])

        self.functionscb = QtGui.QComboBox()
        self.functionscb.setModel(self.funcs_model)

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.attributescb)
        hbox.addWidget(self.functionscb)

        layout.addRow(self.tr("Name"), self.nameedit)
        layout.addRow(self.tr("Expression"), self.expressionedit)
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
            self.modifiedChanged.emit()
            self.modifiedChanged[bool].emit(modified)

    def modified(self):
        return self._modified

    modified = Property(bool, modified, setModified,
                        notify=modifiedChanged)

    def setEditorData(self, data, domain):
        self.nameedit.setText(data.name)
        self.expressionedit.setText(data.expression)
        self.setModified(False)
        self.featureChanged.emit()
        self.attrs_model[:] = ["Select feature"]
        if domain:
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ndecimalsedit = QtGui.QSpinBox(minimum=1, maximum=9)
        self.layout().insertRow(1, self.tr("# decimals"), self.ndecimalsedit)
        self.ndecimalsedit.editingFinished.connect(self._invalidate)

        self.setTabOrder(self.nameedit, self.ndecimalsedit)
        self.setTabOrder(self.ndecimalsedit, self.expressionedit)

    def setEditorData(self, data, domain):
        self.ndecimalsedit.setValue(data.number_of_decimals)
        super().setEditorData(data, domain)

    def editorData(self):
        return ContinuousDescriptor(
            name=self.nameedit.text(),
            number_of_decimals=self.ndecimalsedit.value(),
            expression=self.expressionedit.text()
        )


class DiscreteFeatureEditor(FeatureEditor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        valueslayout = QtGui.QVBoxLayout(spacing=1)
        valueslayout.setContentsMargins(0, 0, 0, 0)

        self.valuesmodel = itemmodels.PyListModel(
            [],
            flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        )
        self.valuesedit = QtGui.QListView(
            sizePolicy=QSizePolicy(QSizePolicy.Minimum,
                                   QSizePolicy.MinimumExpanding)
        )
        self.valuesedit.setModel(self.valuesmodel)

        toolbar = ActionToolBar()

        addaction = QtGui.QAction(
            "+", toolbar,
            toolTip="Add a value"
        )
        addaction.triggered.connect(self.addValue)

        removeaction = QtGui.QAction(
            unicodedata.lookup("MINUS SIGN"), toolbar,
            toolTip="Remove selected value",
            #             shortcut=QtGui.QKeySequence.Delete,
            #             shortcutContext=Qt.WidgetShortcut
        )
        removeaction.triggered.connect(self.removeValue)

        toolbar.addAction(addaction)
        toolbar.addAction(removeaction)

        valueslayout.addWidget(self.valuesedit)
        valueslayout.addWidget(toolbar)

        self.baseedit = QtGui.QComboBox()
        self.baseedit.setModel(self.valuesmodel)
        self.orderededit = QtGui.QCheckBox(text=self.tr("Ordered"))

        layout = self.layout()
        layout.insertRow(1, self.tr("Values"), valueslayout)
        layout.insertRow(2, self.tr("Base Value"), self.baseedit)
        layout.insertRow(3, self.orderededit)

        self.valuesmodel.rowsInserted.connect(self._invalidate)
        self.valuesmodel.rowsRemoved.connect(self._invalidate)
        self.valuesmodel.dataChanged.connect(self._invalidate)

        self.baseedit.activated.connect(self._invalidate)
        self.orderededit.clicked.connect(self._invalidate)

        self.setTabOrder(self.nameedit, self.valuesedit)
        self.setTabOrder(self.valuesedit, toolbar)
        self.setTabOrder(toolbar, self.baseedit)
        self.setTabOrder(self.baseedit, self.orderededit)
        self.setTabOrder(self.orderededit, self.expressionedit)

    def setEditorData(self, data, domain):
        self.valuesmodel[:] = data.values
        self.baseedit.setCurrentIndex(data.base_value)
        super().setEditorData(data, domain)

    def editorData(self):
        return DiscreteDescriptor(
            name=self.nameedit.text(),
            values=tuple(self.valuesmodel),
            base_value=self.baseedit.currentIndex(),
            ordered=self.orderededit.isChecked(),
            expression=self.expressionedit.text()
        )

    def addValue(self, name=None):
        if name is not None:
            name = "%s" % self.valuesmodel.rowCount()

        self.valuesmodel.append(name)
        index = self.valuesmodel.index(len(self.valuesmodel) - 1)
        self.valuesedit.setCurrentIndex(index)
        self.valuesedit.edit(index)

    def removeValue(self):
        index = selected_row(self.valuesedit)
        if index is not None:
            del self.valuesmodel[index]


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
        return QtGui.QIcon()


class FeatureItemDelegate(QtGui.QStyledItemDelegate):
    def displayText(self, value, locale):
        return value.name + " := " + value.expression

    def _initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        model = index.model()
        data = model.data(index, Qt.DisplayRole)
        icon = variable_icon(type(data))
        option.icon = icon
        option.decorationSize = icon.actualSize(
            option.decorationSize, QtGui.QIcon.Normal, QtGui.QIcon.Off)


class DescriptorModel(itemmodels.PyListModel):
    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DecorationRole:
            value = self[index.row()]
            return variable_icon(type(value))
        else:
            return super().data(index, role)


class OWFeatureConstructor(widget.OWWidget):
    name = "Feature Constructor"
    description = "Construct new features (data columns) from a set of " \
                  "existing features in the input data set."
    icon = "icons/FeatureConstructor.svg"
    inputs = [("Data", Orange.data.Table, "setData")]
    outputs = [("Data", Orange.data.Table)]
    want_main_area = False

    settingsHandler = DomainContextHandler()
    descriptors = ContextSetting([])
    currentIndex = ContextSetting(-1)

    EDITORS = [
        (ContinuousDescriptor, ContinuousFeatureEditor),
        (DiscreteDescriptor, DiscreteFeatureEditor),
        (StringDescriptor, StringFeatureEditor)
    ]

    def __init__(self):
        super().__init__()
        self.data = None
        self.editors = {}

        box = QtGui.QGroupBox(
            title=self.tr("Feature Definitions")
        )

        box.setLayout(QtGui.QHBoxLayout())

        self.controlArea.layout().addWidget(box)

        # Layout for the list view
        layout = QtGui.QVBoxLayout(spacing=1, margin=0)
        self.featuremodel = DescriptorModel()

        self.featuremodel.wrap(self.descriptors)
        self.featureview = QtGui.QListView(
            minimumWidth=200,
            sizePolicy=QSizePolicy(QSizePolicy.Minimum,
                                   QSizePolicy.MinimumExpanding)
        )

        self.featureview.setItemDelegate(FeatureItemDelegate())
        self.featureview.setModel(self.featuremodel)
        self.featureview.selectionModel().selectionChanged.connect(
            self._on_selectedVariableChanged
        )

        self.featuretoolbar = ActionToolBar()
        self.addaction = QtGui.QAction(
            "+", self,
            toolTip="Create a new feature",
            shortcut=QtGui.QKeySequence.New
        )
        menu = QtGui.QMenu()
        cont = menu.addAction("Continuous")
        cont.triggered.connect(
            lambda: self.addFeature(ContinuousDescriptor("Name", "", 2))
        )
        disc = menu.addAction("Discrete")
        disc.triggered.connect(
            lambda: self.addFeature(
                DiscreteDescriptor("Name", "", ("0", "1"), -1, False))
        )

        string = menu.addAction("String")
        string.triggered.connect(
            lambda: self.addFeature(StringDescriptor("Name", ""))
        )

        menu.addSeparator()
        self.duplicateaction = menu.addAction("Duplicate selected feature")
        self.duplicateaction.triggered.connect(self.duplicateFeature)
        self.duplicateaction.setEnabled(False)

        self.addaction.setMenu(menu)

        self.removeaction = QtGui.QAction(
            unicodedata.lookup("MINUS SIGN"), self,
            toolTip="Remove selected feature",
            #             shortcut=QtGui.QKeySequence.Delete,
            #             shortcutContext=Qt.WidgetShortcut
        )
        self.removeaction.triggered.connect(self.removeSelectedFeature)
        self.featuretoolbar.addAction(self.addaction)
        self.featuretoolbar.addAction(self.removeaction)

        layout.addWidget(self.featureview)
        layout.addWidget(self.featuretoolbar)

        box.layout().addLayout(layout, 1)

        self.editorstack = QtGui.QStackedWidget(
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding,
                                   QSizePolicy.MinimumExpanding)
        )

        for descclass, editorclass in self.EDITORS:
            editor = editorclass()
            editor.featureChanged.connect(self._on_modified)
            self.editors[descclass] = editor
            self.editorstack.addWidget(editor)

        self.editorstack.setEnabled(False)

        box.layout().addWidget(self.editorstack, 3)

        gui.button(self.controlArea, self, "Commit", callback=self.apply,
                   default=True)

    def setCurrentIndex(self, index):
        index = min(index, len(self.featuremodel) - 1)
        self.currentIndex = index
        if index >= 0:
            itemmodels.select_row(self.featureview, index)
            desc = self.featuremodel[min(index, len(self.featuremodel) - 1)]
            editor = self.editors[type(desc)]
            self.editorstack.setCurrentWidget(editor)
            editor.setEditorData(desc,
                                 self.data.domain if self.data else None)
        self.editorstack.setEnabled(index >= 0)
        self.duplicateaction.setEnabled(index >= 0)
        self.removeaction.setEnabled(index >= 0)

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

    def setDescriptors(self, descriptors):
        """
        Set a list of variable descriptors to edit.
        """
        self.descriptors = descriptors
        self.featuremodel[:] = list(self.descriptors)

    def setData(self, data=None):
        self.closeContext()

        self.featuremodel.wrap([])
        self.currentIndex = -1
        self.data = data

        if self.data is not None:
            self.openContext(data)
            self.featuremodel.wrap(self.descriptors)
            self.setCurrentIndex(self.currentIndex)

        self.editorstack.setEnabled(len(self.featuremodel) > 0)

        self._invalidate()

    def handleNewSignals(self):
        if self.data is not None:
            self.apply()

    def _invalidate(self):
        pass

    def addFeature(self, descriptor):
        self.featuremodel.append(descriptor)
        self.setCurrentIndex(len(self.featuremodel) - 1)
        editor = self.editorstack.currentWidget()
        editor.nameedit.setFocus()
        editor.nameedit.selectAll()
        self._invalidate()

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

        def remove_invalid_expression(desc):
            return (desc if is_valid_expression(desc.expression)
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

        self.error(0)
        try:
            data = Orange.data.Table(new_domain, self.data)
        except Exception as err:
            self.error(0, repr(err.args[0]))
            return

        self.error(1)
        disc_attrs_not_ok = self.check_attrs_values(
            [var for var in attrs if var.is_discrete], data)
        if disc_attrs_not_ok:
            self.error(1, 'Discrete variable %s needs more values.' %
                       disc_attrs_not_ok)
            return

        self.send("Data", data)


import ast


class RewriteNames(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names
        self.rewrites = []

    def visit_Str(self, node):
        if node.s in self.names:
            new = ast.Subscript(
                value=ast.Name(id="value", ctx=ast.Load()),
                slice=ast.Index(value=ast.Str(s=node.s)),
                ctx=ast.Load()
            )
            self.rewrites.append((node.s))
            return ast.copy_location(new)
        else:
            return node


def bind_names(exp, domain):
    names = [f.name for f in domain.features]
    transf = RewriteNames(names)
    return transf.visit(exp)


def freevars(exp, env):
    etype = type(exp)
    if etype in [ast.Expr, ast.Expression]:
        return freevars(exp.body, env)
    elif etype == ast.BoolOp:
        return sum((freevars(v, env) for v in exp.values), [])
    elif etype == ast.BinOp:
        return freevars(exp.left, env) + freevars(exp.right, env)
    elif etype == ast.UnaryOp:
        return freevars(exp.operand, env)
    elif etype == ast.IfExp:
        return (freevars(exp.test, env) + freevars(exp.body, env) +
                freevars(exp.orelse, env))
    elif etype == ast.Dict:
        return sum((freevars(v, env) for v in exp.values), [])
    elif etype == ast.Set:
        return sum((freevars(v, env) for v in exp.elts), [])
    elif etype in [ast.SetComp, ast.ListComp, ast.GeneratorExp]:
        raise NotImplementedError
    elif etype == ast.DictComp:
        raise NotImplementedError
    # Yield, YieldFrom???
    elif etype == ast.Compare:
        return sum((freevars(v, env) for v in [exp.left] + exp.comparators), [])
    elif etype == ast.Call:
        return sum((freevars(e, env)
                    for e in [exp.func] + (exp.args or []) +
                    (exp.keywords or []) +
                    (exp.starargs or []) +
                    (exp.kwargs or [])),
                   [])
    elif etype in [ast.Num, ast.Str, ast.Ellipsis]:
        #     elif etype in [ast.Num, ast.Str, ast.Ellipsis, ast.Bytes]:
        return []
    elif etype == ast.Attribute:
        return freevars(exp.value, env)
    elif etype == ast.Subscript:
        return freevars(exp.value, env) + freevars(exp.slice, env),
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
    return eval(compile(exp, "<lambda>", "eval"), __GLOBALS)


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


if __name__ == "__main__":
    app = QtGui.QApplication([])
    w = OWFeatureConstructor()
    w.show()
    data = Orange.data.Table("iris")
    w.setData(data)
    w.handleNewSignals()
    app.exec_()
    w.setData(None)
    w.saveSettings()
