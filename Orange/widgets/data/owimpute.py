
import collections
from collections import namedtuple

from PyQt4 import QtGui
from PyQt4.QtGui import (
    QWidget, QGroupBox, QRadioButton, QPushButton, QHBoxLayout,
    QVBoxLayout, QStackedLayout, QComboBox, QLineEdit,
    QDoubleValidator, QButtonGroup
)

from PyQt4.QtCore import Qt, QMargins

import Orange.data
import Orange.classification
from Orange.data import filter as data_filter

from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget
from Orange.widgets.utils import itemmodels


def _margins(margins, container):
    if isinstance(margins, tuple):
        left, top, right, bottom = margins
    elif isinstance(margins, int):
        left = top = right = bottom = margins
    elif isinstance(margins, QMargins):
        left, top, right, bottom = \
            margins.left(), margins.top(), margins.right(), margins.bottom()
    else:
        raise TypeError

    container_margins = container.getContentsMargins()
    margins = [c if m == -1 else m
               for c, m in zip([left, top, right, bottom],
                               container_margins)]
    return margins


def layout(orientation=Qt.Vertical, margins=None, spacing=None,):
    if orientation == Qt.Vertical:
        lay = QVBoxLayout()
    else:
        lay = QHBoxLayout()

    if margins is not None:
        left, top, right, bottom = _margins(margins, lay)
        lay.setContentsMargins(left, right, top, bottom)
    return lay


def group_box(title=None, layout=None, margin=None, flat=False, ):
    gb = QGroupBox(title=title, flat=flat)
    if layout is not None:
        gb.setLayout(layout)
    return gb


def widget(layout=None, tooltip=None, objname=None, enabled=True,):
    w = QWidget(toolTip=tooltip, objectName=objname, enabled=enabled)
    if layout is not None:
        w.setLayout(layout)
    return w


def radio_button(text="", checked=False, group=None, group_id=None):
    button = QRadioButton(text, checked=checked)
    if group is not None:
        group.addButton(button, )
        if group_id is not None:
            group.setId(button, group_id)
    return button


def push_button(text="", checked=False, checkable=False,
                group=None, group_id=None, **kwargs):
    button = QPushButton(text, checked=checked, checkable=checkable, **kwargs)
    if group is not None:
        group.addButton(button)
        if group_id is not None:
            group.setId(button, group_id)
    return button


def commit_widget(button_text="Commit", button_default=True,
                  check_text="Commit on any change", checked=False,
                  modified=False, clicked=None):
    w = widget(layout=layout(margins=0))
    button = push_button(button_text, clicked=clicked,
                         default=button_default)
    button.setDefault(button_default)

    check = QtGui.QCheckBox(check_text, checked=checked)
    action = QtGui.QAction(
        button_text, w,
        objectName="action-commit",
    )
    button.clicked.connect(action.trigger)
    w.commit_action = action
    w.commit_button = button
    w.auto_commit_check = check

    w.layout().addWidget(check)
    w.layout().addWidget(button)
    return w


class DisplayFormatDelegate(QtGui.QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        state = index.data(Qt.UserRole)
        var = index.model()[index.row()]
        if state:
            fmt = state.method.format
            text = fmt.format(var=var, params=state.params,
                              **state.method._asdict())
            option.text = text


METHODS = (
    {"name": "Default (above)",
     "short": "",
     "description": "As above so below",
     "format": "{var.name}"},
    {"name": "Don't impute",
     "short": "leave",
     "description": "I",
     "format": "{var.name} -> leave"},
    {"name": "Average/Most frequent",
     "short": "avg",
     "description": "Replace with average/modus for the column",
     "format": "{var.name} -> avg"},
    {"name": "As a distinct value",
     "short": "as_value",
     "description": "",
     "format": "{var.name} -> new value"},
    {"name": "Model-based imputer",
     "short": "model",
     "description": "",
     "format": "{var.name} -> {params[0]!r}"},
    {"name": "Random values",
     "short": "random",
     "description": "Replace with a random value",
     "format": "{var.name} -> random"},
    {"name": "Remove instances with unknown values",
     "short": "drop",
     "description": "",
     "format": "{var.name} -> drop"},
    {"name": "Value",
     "short": "value",
     "description": "",
     "format": "{var.name} -> {params[0]!s}"},
)


Method = namedtuple(
    "Method",
    ["name", "short", "description", "format"]
)


class Method(Method):
    pass


State = namedtuple("State", ["method", "params"])


class State(State):
    def __new__(cls, method, params=()):
        return super().__new__(cls, method, params)

    def _asdict(self):
        return {"method": self.method._asdict(),
                "params": self.params}

# state
#  - selected default
#  - for each variable (indexed by (vartype, name)):
#     - selected method (method index, *params)

# vartype * name -> method
# data method = Method(name) | Method2(name, (*params))


METHODS = [Method(**m) for m in METHODS]


class OWImpute(OWWidget):
    name = "Impute"
    description = "Imputes missing values in the data table."
    icon = "icons/Impute.svg"
    priority = 2130

    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Learner", Orange.classification.Fitter, "set_fitter")]
    outputs = [("Data", Orange.data.Table)]

    METHODS = METHODS

    settingsHandler = settings.DomainContextHandler()

    default_method = settings.Setting(1)
    variable_methods = settings.ContextSetting({})
    autocommit = settings.Setting(False)

    want_main_area = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.modified = False

        box = group_box(self.tr("Default method"),
                        layout=layout(Qt.Vertical))
        self.controlArea.layout().addWidget(box)

        bgroup = QButtonGroup()

        for i, m in enumerate(self.METHODS[1:-1], 1):
            b = radio_button(m.name, checked=i == self.default_method,
                             group=bgroup, group_id=i)
            box.layout().addWidget(b)

        self.defbggroup = bgroup

        bgroup.buttonClicked[int].connect(self.set_default_method)
        box = group_box(self.tr("Individual attribute settings"),
                        layout=layout(Qt.Horizontal))
        self.controlArea.layout().addWidget(box)

        self.varview = QtGui.QListView(
            selectionMode=QtGui.QListView.ExtendedSelection
        )
        self.varview.setItemDelegate(DisplayFormatDelegate())
        self.varmodel = itemmodels.VariableListModel()
        self.varview.setModel(self.varmodel)
        self.varview.selectionModel().selectionChanged.connect(
            self._on_var_selection_changed
        )
        self.selection = self.varview.selectionModel()

        box.layout().addWidget(self.varview)

        method_layout = layout(Qt.Vertical, margins=0)
        box.layout().addLayout(method_layout)

        methodbox = group_box(layout=layout(Qt.Vertical))

        bgroup = QButtonGroup()
        for i, m in enumerate(self.METHODS):
            b = radio_button(m.name, group=bgroup, group_id=i)
            methodbox.layout().addWidget(b)

        assert self.METHODS[-1].short == "value"

        self.value_stack = value_stack = QStackedLayout()
        self.value_combo = QComboBox(activated=self._on_value_changed)
        self.value_line = QLineEdit(editingFinished=self._on_value_changed)
        self.value_line.setValidator(QDoubleValidator())
        value_stack.addWidget(self.value_combo)
        value_stack.addWidget(self.value_line)
        methodbox.layout().addLayout(value_stack)

        bgroup.buttonClicked[int].connect(
            self.set_method_for_current_selection
        )
        reset_button = push_button("Restore all to default",
                                   clicked=self.reset_var_methods,
                                   default=False, autoDefault=False)

        method_layout.addWidget(methodbox)
        method_layout.addStretch(2)
        method_layout.addWidget(reset_button)
        self.varmethodbox = methodbox
        self.varbgroup = bgroup

        commitbox = group_box("Commit", layout=layout(margins=0))

        cwidget = commit_widget(
            button_text="Commit",
            button_default=True,
            check_text="Commit on any change",
            checked=self.autocommit,
            clicked=self.commit
        )

        def toggle_auto_commit(b):
            self.autocommit = b
            if self.modified:
                self.commit()

        cwidget.auto_commit_check.toggled[bool].connect(toggle_auto_commit)
        commitbox.layout().addWidget(cwidget)

        self.addAction(cwidget.commit_action)
        self.controlArea.layout().addWidget(commitbox)

        self.data = None
        self.fitter = None

    def set_default_method(self, index):
        """
        Set the current selected default imputation method.
        """
        if self.default_method != index:
            self.default_method = index
            self.defbggroup.button(index).setChecked(True)
            self._invalidate()

    def set_data(self, data):
        self.closeContext()
        self.clear()
        self.data = data
        if data is not None:
            self.varmodel[:] = data.domain.variables
            self.openContext(data.domain)
            self.restore_state(self.variable_methods)
            itemmodels.select_row(self.varview, 0)

        self.commit()

    def set_fitter(self, fitter):
        self.fitter = fitter

        if self.data is not None and \
                any(state.model.short == "model" for state in
                    map(self.state_for_column, range(len(self.data.domain)))):
            self.commit()

    def restore_state(self, state):
        for i, var in enumerate(self.varmodel):
            key = variable_key(var)
            if key in state:
                index = self.varmodel.index(i)
                self.varmodel.setData(index, state[key], Qt.UserRole)

    def clear(self):
        self.varmodel[:] = []
        self.variable_methods = {}
        self.data = None
        self.modified = False

    def state_for_column(self, column):
        """
        #:: int -> State
        Return the effective imputation state for `column`.

        :param int column:
        :rtype State:

        """
        var = self.varmodel[column]

        state = self.variable_methods.get(variable_key(var), None)
        if state is None or state.method == METHODS[0]:
            state = State(METHODS[self.default_method], ())
        return state

    def imputer_for_column(self, column):
        state = self.state_for_column(column)
        data = self.data
        var = data.domain[column]
        method, params = state
        if method.short == "leave":
            return None
        elif method.short == "avg":
            return column_imputer_average(var, data)
        elif method.short == "model":
            fitter = self.fitter if self.fitter is not None else MeanFitter()
            return column_imputer_by_model(var, data, fitter=fitter)
        elif method.short == "random":
            return column_imputer_random(var, data)
        elif method.short == "value":
            return column_imputer_defaults(var, data, float(params[0]))
        elif method.short == "as_value":
            return column_imputer_as_value(var, data)
        else:
            assert False

    def commit(self):
        if self.data is not None:
            states = [self.state_for_column(i)
                      for i in range(len(self.varmodel))]

            # Columns to filter unknowns by dropping rows.
            filter_columns = [i for i, state in enumerate(states)
                              if state.method.short == "drop"]

            impute_columns = [i for i, state in enumerate(states)
                              if state.method.short not in ["drop", "leave"]]

            imputers = [(self.varmodel[i], self.imputer_for_column(i))
                        for i in impute_columns]

            data = self.data

            if imputers:
                table_imputer = ImputerModel(data.domain, dict(imputers))
                data = table_imputer(data)

            if filter_columns:
                filter_ = data_filter.IsDefined(filter_columns)
                data = filter_(data)
        else:
            data = None

        self.send("Data", data)
        self.modified = False

    def _invalidate(self):
        self.modified = True
        if self.autocommit:
            self.commit()

    def _on_var_selection_changed(self):
        indexes = self.selection.selectedIndexes()

        vars = [self.varmodel[index.row()] for index in indexes]
        defstate = State(METHODS[0], ())
        states = [self.variable_methods.get(variable_key(var), defstate)
                  for var in vars]
        all_cont = all(isinstance(var, Orange.data.ContinuousVariable)
                       for var in vars)
        states = list(unique(states))
        method = None
        params = ()
        state = None
        if len(states) == 1:
            state = states[0]
            method, params = state
            mindex = METHODS.index(method)
            self.varbgroup.button(mindex).setChecked(True)
        elif self.varbgroup.checkedButton() is not None:
            self.varbgroup.setExclusive(False)
            self.varbgroup.checkedButton().setChecked(False)
            self.varbgroup.setExclusive(True)

        values, enabled, stack_index = [], False, 0
        value, value_index = "0.0", 0
        if all_cont:
            enabled, stack_index = True, 1
            if method is not None and method.short == "value":
                value = params[0]

        elif len(vars) == 1 and \
                isinstance(vars[0], Orange.data.DiscreteVariable):
            values, enabled, stack_index = vars[0].values, True, 0
            if method is not None and method.short == "value":
                try:
                    value_index = values.index(params[0])
                except IndexError:
                    pass

        self.value_stack.setCurrentIndex(stack_index)
        self.value_stack.setEnabled(enabled)

        if stack_index == 0:
            self.value_combo.clear()
            self.value_combo.addItems(values)
            self.value_combo.setCurrentIndex(value_index)
        else:
            self.value_line.setText(value)

    def _on_value_changed(self):
        # The "fixed" value in the widget has been changed by the user.
        index = self.varbgroup.checkedId()
        self.set_method_for_current_selection(index)

    def set_method_for_current_selection(self, methodindex):
        indexes = self.selection.selectedIndexes()
        self.set_method_for_indexes(indexes, methodindex)

    def set_method_for_indexes(self, indexes, methodindex):
        method = METHODS[methodindex]
        params = (None,)
        if method.short == "value":
            if self.value_stack.currentIndex() == 0:
                value = self.value_combo.currentIndex()
            else:
                value = self.value_line.text()
            params = (value, )
        elif method.short == "model":
            params = ("model", )
        state = State(method, params)

        for index in indexes:
            self.varmodel.setData(index, state, Qt.UserRole)
            var = self.varmodel[index.row()]
            self.variable_methods[variable_key(var)] = state

        self._invalidate()

    def reset_var_methods(self):
        indexes = map(self.varmodel.index, range(len(self.varmodel)))
        self.set_method_for_indexes(indexes, 0)


def variable_key(variable):
    return (variable.var_type, variable.name)


def unique(iterable):
    seen = set()
    for el in iterable:
        if el not in seen:
            seen.add(el)
            yield el


def translate_domain(X, domain):
    if isinstance(domain, tuple):
        domain = Orange.data.Domain(domain)

    if X.domain != domain:
        if isinstance(X, Orange.data.Table):
            X = Orange.data.Table.from_table(domain, X)
        elif isinstance(X, Orange.data.Instance):
            X = domain.convert(X)
        else:
            # Storage??
            raise TypeError

    return X


def column_imputer(variable, table):
    """
    column_imputer :: Variable -> Table -> ColumnImputerModel
    """
    pass


class ColumnImputerModel(object):
    def __init__(self, domain, codomain, transformers):
        if isinstance(domain, tuple):
            domain = Orange.data.Domain(domain)
        if isinstance(codomain, tuple):
            codomain = Orange.data.Domain(codomain)

        self.domain = domain
        self.codomain = codomain
        self.transformers = transformers

    def __call__(self, data):
        raise NotImplementedError()


def learn_model_for(fitter, variable, data):
    """
    Learn a model for `variable`
    """
    attrs = [attr for attr in data.domain.attributes
             if attr is not variable]
    domain = Orange.data.Domain(attrs, (variable,))
    data = Orange.data.Table.from_table(domain, data)
    return fitter(data)


from Orange.classification.naive_bayes import BayesLearner


def column_imputer_by_model(variable, table, *, fitter=BayesLearner()):
    model = learn_model_for(fitter, variable, table)
    assert model.domain.class_vars == (variable,)
    return ColumnImputerFromModel(table.domain, model.domain.class_vars, model)


class ColumnImputerFromModel(ColumnImputerModel):
    def __init__(self, domain, codomain, model):
        transform = ModelTransform(model.domain.class_var, model)
        super().__init__(model.domain, codomain, (transform,))
        self.model = model

    def __call__(self, data):
        trans = self.transformers[0]
        filter_ = data_filter.IsDefined([trans.variable], negate=True)
        data_with_unknowns = filter_(data)
        values = trans(data_with_unknowns)

        domain = Orange.data.Domain([trans.variable])
        X = Orange.data.Table.from_table(domain, data)
        X.X[numpy.isnan(X), :] = values
        return X


from Orange.statistics import basic_stats
from Orange.statistics import distribution


def column_imputer_defaults(variable, table, default):
    transform = ReplaceUnknowns(variable, default)
    return ColumnImputerDefaults(table.domain, (variable,),
                                 [transform], [default])


def column_imputer_maximal(variable, table):
    stats = basic_stats.BasicStats(table, variable)
    return column_imputer_defaults(variable, table, stats.max)


def column_imputer_minimal(variable, table):
    stats = basic_stats.BasicStats(table, variable)
    return column_imputer_defaults(variable, table, stats.min)


def column_imputer_average(variable, table):
    stats = basic_stats.BasicStats(table, variable)
    return column_imputer_defaults(variable, table, stats.mean)


def column_imputer_modus(variable, table):
    stat = distribution.get_distribution(table, variable)
    column_imputer_defaults(variable, table, stat.modus())


class ColumnImputerDefaults(ColumnImputerModel):
    def __init__(self, domain, codomain, transformers, defaults):
        super().__init__(domain, codomain, transformers)
        self.defaults = defaults

    def __call__(self, data, weight=None):
        data = translate_domain(data, self.codomain)
        defaults, X = numpy.broadcast_arrays([self.defaults], data.X)
        X = numpy.where(numpy.isnan(X), defaults, X)
        return Orange.data.Table.from_numpy(self.codomain, X)


def column_imputer_as_value(variable, table):
    if isinstance(variable, Orange.data.DiscreteVariable):
        fmt = "{var.name}"
        value = "N/A"
        var = Orange.data.DiscreteVariable(
            fmt.format(var=variable),
            values=variable.values + [value],
            base_value=variable.base_value
        )
        var.get_value_from = Lookup(
            variable,
            numpy.arange(len(variable.values), dtype=int),
            unknown=len(variable.values)
        )
        codomain = [var]
        transformers = [var.get_value_from]
    elif isinstance(variable, Orange.data.ContinuousVariable):
        fmt = "{var.name}_def"
        var = Orange.data.DiscreteVariable(
            fmt.format(var=variable),
            values=("undef", "def"),
        )
        var.get_value_from = IsDefined(variable)
        codomain = [variable, var]
        stats = basic_stats.BasicStats(table, variable)
        transformers = [ReplaceUnknowns(variable, stats.mean),
                        var.get_value_from]
    else:
        raise TypeError(type(variable))

    return ColumnImputerAsValue(
            table.domain, Orange.data.Domain(codomain), transformers)


class ColumnImputerAsValue(ColumnImputerModel):
    def __init__(self, domain, codomain, transformers):
        super().__init__(domain, codomain, transformers)

    def __call__(self, data, ):
        data = translate_domain(data, self.domain)
        data = translate_domain(data, self.codomain)

        variable = self.codomain[0]
        if isinstance(variable, Orange.data.ContinuousVariable):
            tr = self.transformers[0]
            assert isinstance(tr, ReplaceUnknowns)
            c = tr(data[:, variable])
            cindex = data.domain.index(variable)
            data.X[:, cindex] = c
        return data


def column_imputer_random(variable, data):
    if isinstance(variable, Orange.data.DiscreteVariable):
        transformer = RandomTransform(variable)
    elif isinstance(variable, Orange.data.ContinuousVariable):
        transformer = RandomTransform(variable)
    return RandomImputerModel((variable,), (variable,), (transformer,))


class RandomImputerModel(ColumnImputerModel):
    def __call__(self, data):
        data = translate_domain(data, self.codomain)
        trans = self.transformers[0]
        values = trans(data).reshape((-1, 1))

        X = data[:, trans.variable].X
        values = numpy.where(numpy.isnan(X), values, X)
        return Orange.data.Table.from_numpy(self.codomain, values)


# Why even need this?
class NullColumnImputer(ColumnImputerModel):
    def __init__(self, domain, codomain, transformers):
        super().__init__(domain, codomain, transformers)

    def __call__(self, data, weight=None):
        data = translate_domain(data, self.codomain)
        return data


from functools import reduce
import numpy
from Orange.feature.transformation import \
    ColumnTransformation, Lookup, Identity


class IsDefined(ColumnTransformation):
    def _transform(self, c):
        return ~numpy.isnan(c)


class Lookup(Lookup):
    def __init__(self, variable, lookup_table, unknown=None):
        super().__init__(variable, lookup_table)
        self.unknown = unknown

    def _transform(self, column):
        if self.unknown is None:
            unknown = numpy.nan
        else:
            unknown = self.unknown

        mask = numpy.isnan(column)
        column_valid = numpy.where(mask, 0, column)
        values = self.lookup_table[numpy.array(column_valid, dtype=int)]
        return numpy.where(mask, unknown, values)


class ReplaceUnknowns(ColumnTransformation):
    def __init__(self, variable, value=0):
        super().__init__(variable)
        self.value = value

    def _transform(self, c):
        return numpy.where(numpy.isnan(c), self.value, c)


class RandomTransform(ColumnTransformation):
    def __init__(self, variable, dist=None):
        super().__init__(variable)
        self.dist = dist

    def _transform(self, c):
        if isinstance(self.variable, Orange.data.DiscreteVariable):
            if self.dist is not None:
                pass
            else:
                c = numpy.random.randint(len(self.variable.values),
                                         size=c.shape)
        else:
            if self.dist is not None:
                pass
            else:
                c = numpy.random.normal(size=c.shape)
        return c


class ModelTransform(ColumnTransformation):
    def __init__(self, variable, model):
        super().__init__(variable)
        self.model = model

    def __call__(self, data):
        return self.model(data)


# Rename to TableImputer (Model?)
class ImputerModel(object):
    """
    A fitted Imputation model.

    :param domain:
        Imputer domain.
    :param columnimputers:
        A mapping of columns in `domain` to a :class:`ColumnImputerModel`.

    """
    def __init__(self, domain, columnimputers={}):
        self.columnimputers = columnimputers
        self.domain = domain

        col_models = [(var, columnimputers.get(var, None))
                      for var in domain.variables]
        # variables for the codomain
        codomain_attrs = []
        codomain_class_vars = []

        # column imputers for all variables in the domain
        col_imputers = []
        for i, (var, imp) in enumerate(col_models):
            if isinstance(imp, ColumnImputerModel):
                pass
            elif isinstance(imp, Orange.classification.Model):
                imp = ColumnImputerFromModel(domain, imp.class_vars, imp)
            elif isinstance(imp, collections.Callable):
                raise NotImplementedError
                imp = ColumnImputerFromCallable(var, imp)
            elif imp is None:
                imp = NullColumnImputer(domain, (var,), (Identity(var),))

            col_imputers.append((var, imp))

            if i < len(domain.attributes):
                codomain_attrs.extend(imp.codomain)
            else:
                codomain_class_vars.extend(imp.codomain)

        self.codomain = Orange.data.Domain(
            codomain_attrs, codomain_class_vars, domain.metas
        )

        self.transformers = []
        self.columnimputers = dict(col_imputers)
        for var, colimp in col_imputers:
            self.transformers.append(
                (var, tuple(zip(colimp.codomain, colimp.transformers)))
            )

    def __call__(self, X, weight=None):
        X = translate_domain(X, self.domain)
        Xp = translate_domain(X, self.codomain)

        if Xp is X:
            Xp = Xp.copy()

        nattrs = len(Xp.domain.attributes)
        for var in X.domain:
            col_imputer = self.columnimputers[var]
            if isinstance(col_imputer, NullColumnImputer):
                continue

            if not self._is_var_transform(col_imputer):
                cols = col_imputer(X)
                for i, cv in enumerate(col_imputer.codomain):
                    cvindex = Xp.domain.index(cv)
                    if cvindex < len(Xp.domain.attributes):
                        Xp.X[:, cvindex] = cols.X[:, i]
                    else:
                        Xp.Y[:, nattrs - cvindex] = cols.X[:, i]

        return Xp

    def _is_var_transform(self, imputer):
        """
        Is `imputer` implemented as a Varible.get_value_from.

        """
        for var, t in zip(imputer.codomain, imputer.transformers):
            if var.get_value_from and var.get_value_from is t:
                pass
            else:
                return False
        return False


"""

Imputation:
    Should be a standard X -> X' transform.
    i.e. Given an X of domain D returns the X' of a domain
    D' where the number of instances in X' might not be the same
    and the D' might not be the same.

    F = Imputer(X)
    D' = F.codomain
    rows = F.filter(X)
    X' = F(X)
    assert X'.domain == D'
    assert X'.rowids == rows

    Issue 1: The filter might/should be a separate step. I.e.

    F = Imputer(X) o Filter(X, ...) # Imputer(X) | Filter(X)

    F.domain, F.codomain
    reduce((+), map(F.transform, F.domain)) == F.codomain

    A ColumnImputer is a mapping [var] -> [var', [var1', ...]].
    The mapping is specified with either var'.get_value_from or
    by ColumnImputer i.e. ColumnImputer must contain a Transformation for
    each codomain variable
    data transform = Transform of Variable * (Variable * Transformation) list
    i.e.
        source var -> [(new_var, Transformation), ...]

"""

from Orange.classification import Fitter, Model


class MeanFitter(Fitter):
    def fit_storage(self, data):
        dist = distribution.get_distribution(data, data.domain.class_var)
        domain = Orange.data.Domain((), (data.domain.class_var,))
        return MeanPredictor(domain, dist)


class MeanPredictor(Model):
    def __init__(self, domain, distribution):
        super().__init__(domain)
        self.distribution = distribution
        self.mean = distribution.mean()

    def predict(self, X):
        return numpy.zeros(len(X)) + self.mean


import unittest


class Test(unittest.TestCase):
    def test_impute_defaults(self):
        nan = numpy.nan
        data = [
            [1.0, nan, 0.0],
            [2.0, 1.0, 3.0],
            [nan, nan, nan]
        ]
        data = Orange.data.Table.from_numpy(None, numpy.array(data))

        cimp1 = column_imputer_average(data.domain[0], data)
        self.assertIsInstance(cimp1.transformers[0], ReplaceUnknowns)
        trans = cimp1.transformers[0]
        self.assertEqual(trans.value, 1.5)
        self.assertTrue((trans(data) == [1.0, 2.0, 1.5]).all())

        cimp2 = column_imputer_maximal(data.domain[1], data)
        trans = cimp2.transformers[0]
        self.assertTrue((trans(data) == [1.0, 1.0, 1.0]).all())

        cimp3 = column_imputer_minimal(data.domain[2], data)
        trans = cimp3.transformers[0]
        self.assertTrue((trans(data) == [0.0, 3.0, 0.0]).all())

        imputer = ImputerModel(
            data.domain,
            {data.domain[0]: cimp1,
             data.domain[1]: cimp2,
             data.domain[2]: cimp3}
        )
        idata = imputer(data)
        self.assertClose(idata.X,
                         [[1.0, 1.0, 0.0],
                          [2.0, 1.0, 3.0],
                          [1.5, 1.0, 0.0]])

    def test_impute_as_value(self):
        nan = numpy.nan
        data = [
            [1.0, nan, 0.0],
            [2.0, 1.0, 3.0],
            [nan, nan, nan]
        ]
        domain = Orange.data.Domain(
            (Orange.data.DiscreteVariable("A", values=["0", "1", "2"]),
             Orange.data.ContinuousVariable("B"),
             Orange.data.ContinuousVariable("C"))
        )
        data = Orange.data.Table.from_numpy(domain, numpy.array(data))

        cimp1 = column_imputer_as_value(domain[0], data)
        self.assertEqual(len(cimp1.codomain), 1)
        self.assertEqual(cimp1.codomain[0].name, "A")
        self.assertEqual(cimp1.codomain[0].values, ["0", "1", "2", "N/A"])
        self.assertEqual(len(cimp1.transformers), 1)

        trans = cimp1.transformers[0]
        self.assertClose(trans(data), [1.0, 2.0, 3.0])
        self.assertEqual(list(inst[0] for inst in cimp1(data)),
                         ["1", "2", "N/A"])

        cimp2 = column_imputer_as_value(domain[1], data)
        self.assertEqual(len(cimp2.transformers), 2)
        self.assertEqual(cimp2.codomain[0], domain[1])
        self.assertIsInstance(cimp2.codomain[1], Orange.data.DiscreteVariable)
        self.assertEqual(cimp2.codomain[1].values, ["undef", "def"])

        self.assertClose(cimp2.transformers[0](data), [1.0, 1.0, 1.0])
        self.assertClose(cimp2.transformers[1](data), [0, 1, 0])

        idata = cimp2(data)
        self.assertEqual(idata.domain, cimp2.codomain)
        self.assertClose(idata.X, [[1, 0], [1, 1], [1, 0]])

        cimp3 = column_imputer_as_value(domain[2], data)
        imputer = ImputerModel(
            domain,
            {var: cimp for (var, cimp) in zip(domain, (cimp1, cimp2, cimp3))}
        )
        idata = imputer(data)
        self.assertEqual(
            reduce(tuple.__add__,
                   (tuple(cimp.codomain) for cimp in (cimp1, cimp2, cimp3)),
                   ),
            tuple(idata.domain)
        )

        self.assertClose(
            idata.X,
            [[1, 1.0, 0, 0.0, 1],
             [2, 1.0, 1, 3.0, 1],
             [3, 1.0, 0, 1.5, 0]]
        )

    def test_impute_by_model(self):
        from Orange.classification.majority import MajorityFitter

        nan = numpy.nan
        data = [
            [1.0, nan, 0.0],
            [2.0, 1.0, 3.0],
            [nan, nan, nan]
        ]
        domain = Orange.data.Domain(
            (Orange.data.DiscreteVariable("A", values=["0", "1", "2"]),
             Orange.data.ContinuousVariable("B"),
             Orange.data.ContinuousVariable("C"))
        )
        data = Orange.data.Table.from_numpy(domain, numpy.array(data))

        cimp1 = column_imputer_by_model(domain[0], data,
                                        fitter=MajorityFitter())
        self.assertEqual(tuple(cimp1.codomain), (domain[0],))

        cimp2 = column_imputer_by_model(domain[1], data, fitter=MeanFitter())
        cimp3 = column_imputer_by_model(domain[2], data, fitter=MeanFitter())

        imputer = ImputerModel(
            data.domain,
            {data.domain[0]: cimp1,
             data.domain[1]: cimp2,
             data.domain[2]: cimp3}
        )
        idata = imputer(data)
        self.assertClose(idata.X,
                         [[1.0, 1.0, 0.0],
                          [2.0, 1.0, 3.0],
                          [1.0, 1.0, 1.5]])

    def test_impute_random(self):
        nan = numpy.nan
        data = [
            [1.0, nan, 0.0],
            [2.0, 1.0, 3.0],
            [nan, nan, nan]
        ]
        domain = Orange.data.Domain(
            (Orange.data.DiscreteVariable("A", values=["0", "1", "2"]),
             Orange.data.ContinuousVariable("B"),
             Orange.data.ContinuousVariable("C"))
        )
        data = Orange.data.Table.from_numpy(domain, numpy.array(data))

        cimp1 = column_imputer_random(domain[0], data)
        self.assertTrue(not numpy.any(numpy.isnan(cimp1(data).X)))

        cimp2 = column_imputer_random(domain[1], data)
        self.assertTrue(not numpy.any(numpy.isnan(cimp2(data).X)))

        cimp3 = column_imputer_random(domain[2], data)
        self.assertTrue(not numpy.any(numpy.isnan(cimp3(data).X)))

        imputer = ImputerModel(
            data.domain,
            {data.domain[0]: cimp1,
             data.domain[1]: cimp2,
             data.domain[2]: cimp3}
        )
        idata = imputer(data)
        self.assertTrue(not numpy.any(numpy.isnan(idata.X)))

        definedmask = ~numpy.isnan(data.X)
        self.assertClose(data.X[definedmask],
                         idata.X[definedmask])

    def assertClose(self, X, Y, delta=1e-9, msg=None):
        X, Y = numpy.asarray(X), numpy.asarray(Y)
        if not (numpy.abs(X - Y) <= delta).all():
            standardMsg = "%s != %s to within delta %f" % (X, Y, delta)
            msg = self._formatMessage(msg, standardMsg)
            raise self.failureException(msg)


if __name__ == "__main__":
    app = QtGui.QApplication([])
    w = OWImpute()
    w.show()
    data = Orange.data.Table("brown-selected")
    w.set_data(data)
    app.exec_()
