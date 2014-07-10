from collections import namedtuple

from PyQt4 import QtGui
from PyQt4.QtGui import (
    QListView, QHBoxLayout, QStyledItemDelegate, QDialogButtonBox
)
from PyQt4.QtCore import Qt

import Orange.data
import Orange.feature.discretization as disc

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels

__all__ = ["OWDiscretize"]

# 'Default' method delegates to 'method'
Default = namedtuple("Default", ["method"])
Leave = namedtuple("Leave", [])
MDL = namedtuple("MDL", [])
EqualFreq = namedtuple("EqualFreq", ["k"])
EqualWidth = namedtuple("EqualWidth", ["k"])
Remove = namedtuple("Remove", [])
Custom = namedtuple("Custom", ["points"])

METHODS = [
    (Default, ),
    (Leave, ),
    (MDL, ),
    (EqualFreq, ),
    (EqualWidth, ),
    (Remove, ),
    (Custom, )
]

_dispatch = {
    Default:
        lambda m, data, var: _dispatch[type(m.method)](m.method, data, var),
    Leave: lambda m, data, var: var,
    MDL: lambda m, data, var: EntropyMDL()(data, var),
    EqualFreq: lambda m, data, var: disc.EqualFreq(m.k)(data, var),
    EqualWidth: lambda m, data, var: disc.EqualWidth(m.k)(data, var),
    Remove: lambda m, data, var: None,
    Custom: lambda m, data, var: disc._discretized_var(data, var, m.points)
}


# Variable discretization state
DState = namedtuple(
    "DState",
    ["method",    # discretization method
     "points",    # induced cut points
     "disc_var"]  # induced discretized variable
)


def is_derived(var):
    return var.get_value_from is not None


def is_discretized(var):
    return is_derived(var) and isinstance(var.get_value_from, disc.Discretizer)


def variable_key(var):
    return var.var_type, var.name


def button_group_reset(group):
    button = group.checkedButton()
    if button is not None:
        group.setExclusive(False)
        button.setChecked(False)
        group.setExclusive(True)


class DiscDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        state = index.data(Qt.UserRole)

        if state is not None:
            extra = self.cutsText(state)
            option.text = option.text + ": " + extra

    def cutsText(self, state):
        method = state.method
        name = None
        # Need a better way to distinguish discretization states
        # i.e. between 'induced no points v.s. 'removed by choice'
        if state.points is None and state.disc_var is not None:
            points = ""
        elif state.points is None:
            points = "..."
        elif state.points == []:
            points = "<removed>"
        else:
            points = ", ".join(map("{:.2f}".format, state.points))

        if isinstance(method, Default):
            name = None
        elif isinstance(method, Leave):
            name = "(leave)"
        elif isinstance(method, MDL):
            name = "(entropy)"
        elif isinstance(method, EqualFreq):
            name = "(equal frequency k={})".format(method.k)
        elif isinstance(method, EqualWidth):
            name = "(equal width k={})".format(method.k)
        elif isinstance(method, Remove):
            name = "(removed)"
        elif isinstance(method, Custom):
            name = "(custom)"
        else:
            assert False

        if name is not None:
            return points + " " + name
        else:
            return points


class OWDiscretize(widget.OWWidget):
    name = "Discretize"
    description = "Discretization of continuous attributes."
    icon = "icons/Discretize.svg"
    inputs = [{"name": "Data",
               "type": Orange.data.Table,
               "handler": "set_data",
               "doc": "Input data table"}]

    outputs = [{"name": "Data",
                "type": Orange.data.Table,
                "doc": "Table with discretized features"}]

    settingsHandler = settings.DomainContextHandler()
    saved_var_states = settings.ContextSetting({})

    default_method = settings.Setting(0)
    default_k = settings.Setting(5)

    # Discretization methods
    Default, Leave, MDL, EqualFreq, EqualWidth, Remove, Custom = range(7)

    want_main_area = False

    def  __init__(self, parent=None):
        super().__init__(parent)

        #: input data
        self.data = None
        #: Current variable discretization state
        self.var_state = {}
        #: Saved variable discretization settings (context setting)
        self.saved_var_states = {}

        self.method = 0
        self.k = 5

        box = gui.widgetBox(
            self.controlArea, self.tr("Default Discretization"))
        rbox = gui.radioButtons(
            box, self, "default_method", callback=self._default_disc_changed)

        options = [
            self.tr("Default"),
            self.tr("Leave continuous"),
            self.tr("Entropy-MDL discretization"),
            self.tr("Equal-frequency discretization"),
            self.tr("Equal-width discretization"),
            self.tr("Remove continuous attributes")
        ]

        for opt in options[1:5]:
            gui.appendRadioButton(rbox, opt)

        gui.hSlider(gui.indentedBox(rbox),
                    self, "default_k", minValue=2, maxValue=10,
                    label="Num. of intervals:",
                    callback=self._default_disc_changed)

        gui.appendRadioButton(rbox, options[-1])

        vlayout = QHBoxLayout()
        box = gui.widgetBox(
            self.controlArea, "Individual Attribute Settings",
            orientation=vlayout
        )

        # List view with all attributes
        self.varview = QListView(
            selectionMode=QListView.ExtendedSelection
        )
        self.varview.setItemDelegate(DiscDelegate())
        self.varmodel = itemmodels.VariableListModel()
        self.varview.setModel(self.varmodel)
        self.varview.selectionModel().selectionChanged.connect(
            self._var_selection_changed
        )

        vlayout.addWidget(self.varview)
        # Controls for individual attr settings
        controlbox = gui.radioButtons(
            box, self, "method", callback=self._disc_method_changed
        )
        vlayout.addWidget(controlbox)

        for opt in options[:5]:
            gui.appendRadioButton(controlbox, opt)

        gui.hSlider(gui.indentedBox(controlbox),
                    self, "k", minValue=2, maxValue=10,
                    label="Num. of intervals:",
                    callback=self._disc_method_changed)

        gui.appendRadioButton(controlbox, options[-1])

        gui.rubber(controlbox)
        controlbox.setEnabled(False)

        self.controlbox = controlbox

        bbox = QDialogButtonBox(QDialogButtonBox.Apply)
        self.controlArea.layout().addWidget(bbox)
        bbox.accepted.connect(self.commit)
        button = bbox.button(QDialogButtonBox.Apply)
        button.clicked.connect(self.commit)

    def set_data(self, data):
        self.closeContext()
        self.data = data
        if self.data is not None:
            self._initialize(data)
            self.openContext(data)
            # Restore the per variable discretization settings
            self._restore(self.saved_var_states)
            # Complete the induction of cut points
            self._update_points()
        else:
            self._clear()

        self.commit()

    def _initialize(self, data):
        # Initialize the default variable states for new data.
        self.class_var = data.domain.class_var
        cvars = [var for var in data.domain
                 if isinstance(var, Orange.data.ContinuousVariable)]
        self.varmodel[:] = cvars
        self._reset()

    def _restore(self, saved_state):
        # Restore variable states from a saved_state dictionary.
        for i, var in enumerate(self.varmodel):
            key = variable_key(var)
            if key in saved_state:
                state = saved_state[key]
                self._set_var_state(i, state)

    def _reset(self):
        # restore the individual variable settings back to defaults.
        def_method = self._current_default_method()
        self.var_state = {}
        for i in range(len(self.varmodel)):
            state = DState(Default(def_method), None, None)
            self._set_var_state(i, state)

    def _set_var_state(self, index, state):
        # set the state of variable at `index` to `state`.
        self.var_state[index] = state
        self.varmodel.setData(self.varmodel.index(index), state, Qt.UserRole)

    def _clear(self):
        self.data = None
        self.varmodel[:] = []
        self.var_state = {}
        self.saved_var_states = {}

    def _update_points(self):
        """
        Update the induced cut points.
        """
        def induce_cuts(method, data, var):
            dvar = _dispatch[type(method)](method, data, var)
            if dvar is None:
                # removed
                return [], None
            elif dvar is var:
                # no transformation took place
                return None, var
            elif is_discretized(dvar):
                return dvar.get_value_from.points, dvar
            else:
                assert False

        for i, var in enumerate(self.varmodel):
            state = self.var_state[i]
            if state.points is None and state.disc_var is None:
                points, dvar = induce_cuts(state.method, self.data, var)
                new_state = state._replace(points=points, disc_var=dvar)
                self._set_var_state(i, new_state)

    def _method_index(self, method):
        return METHODS.index((type(method), ))

    def _current_default_method(self):
        method = self.default_method + 1
        k = self.default_k
        if method == OWDiscretize.Leave:
            def_method = Leave()
        elif method == OWDiscretize.MDL:
            def_method = MDL()
        elif method == OWDiscretize.EqualFreq:
            def_method = EqualFreq(k)
        elif method == OWDiscretize.EqualWidth:
            def_method = EqualWidth(k)
        elif method == OWDiscretize.Remove:
            def_method = Remove()
        else:
            assert False
        return def_method

    def _current_method(self):
        if self.method == OWDiscretize.Default:
            method = Default(self._current_default_method())
        elif self.method == OWDiscretize.Leave:
            method = Leave()
        elif self.method == OWDiscretize.MDL:
            method = MDL()
        elif self.method == OWDiscretize.EqualFreq:
            method = EqualFreq(self.k)
        elif self.method == OWDiscretize.EqualWidth:
            method = EqualWidth(self.k)
        elif self.method == OWDiscretize.Remove:
            method = Remove()
        elif self.method == OWDiscretize.Custom:
            method = Custom(self.cutpoints)
        else:
            assert False
        return method

    def _default_disc_changed(self):
        method = self._current_default_method()
        state = DState(Default(method), None, None)
        for i, _ in enumerate(self.varmodel):
            if isinstance(self.var_state[i].method, Default):
                self._set_var_state(i, state)

        self._update_points()

    def _disc_method_changed(self):
        indices = self.selected_indices()
        method = self._current_method()
        state = DState(method, None, None)

        for idx in indices:
            self._set_var_state(idx, state)

        self._update_points()

    def _var_selection_changed(self, *args):
        indices = self.selected_indices()
        # set of all methods for the current selection
        methods = [self.var_state[i].method for i in indices]
        mset = set(methods)
        self.controlbox.setEnabled(len(mset) > 0)
        if len(mset) == 1:
            method = mset.pop()
            self.method = self._method_index(method)

            if isinstance(method, (EqualFreq, EqualWidth)):
                self.k = method.k
            elif isinstance(method, Custom):
                self.cutpoints = method.points

        else:
            # deselect the current button
            self.method = -1
            bg = self.controlbox.group
            button_group_reset(bg)

    def selected_indices(self):
        rows = self.varview.selectionModel().selectedRows()
        return [index.row() for index in rows]

    def discretized_var(self, source):
        index = list(self.varmodel).index(source)
        state = self.var_state[index]
        if state.disc_var is None:
            return None
        elif state.disc_var is source:
            return source
        elif state.points == []:
            return None
        else:
            return state.disc_var

    def discretized_domain(self):
        """
        Return the current effective discretized domain.
        """
        if self.data is None:
            return None

        def disc_var(source):
            if isinstance(source, Orange.data.ContinuousVariable):
                return self.discretized_var(source)
            else:
                return source

        attributes = [disc_var(v) for v in self.data.domain.attributes]
        attributes = [v for v in attributes if v is not None]

        class_var = disc_var(self.data.domain.class_var)

        domain = Orange.data.Domain(
            attributes, class_var,
            metas=self.data.domain.metas
        )
        return domain

    def commit(self):
        output = None
        if self.data is not None:
            domain = self.discretized_domain()
            output = Orange.data.Table.from_table(domain, self.data)
        self.send("Data", output)

    def storeSpecificSettings(self):
        super().storeSpecificSettings()
        self.saved_var_states = {
            variable_key(var):
                self.var_state[i]._replace(points=None, disc_var=None)
            for i, var in enumerate(self.varmodel)
        }


# Entropy-MDL discretization.
# ===========================

import numpy


def normalize(X, axis=None, out=None):
    """
    Normalize `X` array so it sums to 1.0 over the `axis`.

    Parameters
    ----------
    X : array
        Array to normalize.
    axis : optional int
        Axis over which the resulting array sums to 1.
    out : optional array
        Output array of the same shape as X.
    """
    X = numpy.asarray(X, dtype=float)
    scale = numpy.sum(X, axis=axis, keepdims=True)
    if out is None:
        return X / scale
    else:
        if out is not X:
            assert out.shape == X.shape
            out[:] = X
        out /= scale
        return out


def entropy_normalized(D, axis=None):
    """
    Compute the entropy of distribution array `D`.

    `D` must be a distribution (i.e. sum to 1.0 over `axis`)

    Parameters
    ----------
    D : array
        Distribution.
    axis : optional int
        Axis of `D` along which to compute the entropy.

    """
    # req: (numpy.sum(D, axis=axis) >= 0).all()
    # req: (numpy.sum(D, axis=axis) <= 1).all()
    # req: numpy.all(numpy.abs(numpy.sum(D, axis=axis) - 1) < 1e-9)

    D = numpy.asarray(D)
    Dc = numpy.clip(D, numpy.finfo(D.dtype).eps, 1.0)
    return - numpy.sum(D * numpy.log2(Dc), axis=axis)


def entropy(D, axis=None):
    """
    Compute the entropy of distribution `D`.

    Parameters
    ----------
    D : array
        Distribution.
    axis : optional int
        Axis of `D` along which to compute the entropy.

    """
    D = normalize(D, axis=axis)
    return entropy_normalized(D, axis=axis)


def entropy_cuts_sorted(CS):
    """
    Return the class information entropy induced by partitioning
    the `CS` distribution at all N-1 candidate cut points.

    Parameters
    ----------
    CS : (N, K) array of class distributions.
    """
    CS = numpy.asarray(CS)
    # |--|-------|--------|
    #  S1    ^       S2
    # S1 contains all points which are <= to cut point
    # Cumulative distributions for S1 and S2 (left right set)
    # i.e. a cut at index i separates the CS into S1Dist[i] and S2Dist[i]
    S1Dist = numpy.cumsum(CS, axis=0)[:-1]
    S2Dist = numpy.cumsum(CS[::-1], axis=0)[-2::-1]

    # Entropy of S1[i] and S2[i] sets
    ES1 = entropy(S1Dist, axis=1)
    ES2 = entropy(S2Dist, axis=1)

    # Number of cases in S1[i] and S2[i] sets
    S1_count = numpy.sum(S1Dist, axis=1)
    S2_count = numpy.sum(S2Dist, axis=1)

    # Number of all cases
    S_count = numpy.sum(CS)

    ES1w = ES1 * S1_count / S_count
    ES2w = ES2 * S2_count / S_count

    # E(A, T; S) Class information entropy of the partition S
    E = ES1w + ES2w

    return E, ES1, ES2


def entropy_disc(X, C):
    """
    Entropy discretization.

    :param X: (N, 1) array
    :param C: (N, K) array (class probabilities must sum(axis=1) to 1 )

    :rval:
    """
    sort_ind = numpy.argsort(X, axis=0)
    X = X[sort_ind]
    C = C[sort_ind]
    return entropy_discretize_sorted(X, C)


def entropy_discretize_sorted(C):
    """
    Entropy discretization on a sorted C.

    :param C: (N, K) array of class distributions.

    """
    E, ES1, ES2 = entropy_cuts_sorted(C)
    # TODO: Also get the left right distribution counts from
    # entropy_cuts_sorted,

    # Note the + 1
    cut_index = numpy.argmin(E) + 1

    # Distribution of classed in S1, S2 and S
    S1_c = numpy.sum(C[:cut_index], axis=0)
    S2_c = numpy.sum(C[cut_index:], axis=0)
    S_c = S1_c + S2_c

    ES = entropy(numpy.sum(C, axis=0))
    ES1, ES2 = ES1[cut_index - 1], ES2[cut_index - 1]

    # Information gain of the best split
    Gain = ES - E[cut_index - 1]
    # Number of classes in S, S1 and S2 (with non zero counts)
    k = numpy.sum(S_c > 0)
    k1 = numpy.sum(S1_c > 0)
    k2 = numpy.sum(S2_c > 0)

    assert k > 0
    delta = numpy.log2(3 ** k - 2) - (k * ES - k1 * ES1 - k2 * ES2)
    N = numpy.sum(S_c)

    if Gain > numpy.log2(N - 1) / N + delta / N:
        # Accept the cut point and recursively split the subsets.
        left, right = [], []
        if k1 > 1 and cut_index > 1:
            left = entropy_discretize_sorted(C[:cut_index, :])
        if k2 > 1 and cut_index < len(C) - 1:
            right = entropy_discretize_sorted(C[cut_index:, :])
        return left + [cut_index] + [i + cut_index for i in right]
    else:
        return []


class EntropyMDL(disc.Discretization):
    def __call__(self, data, attribute):
        from Orange.statistics import contingency as c
        cont = c.get_contingency(data, attribute)
        values, I = join_contingency(cont)
        cut_ind = numpy.array(entropy_discretize_sorted(I))
        if len(cut_ind) > 0:
            points = values[cut_ind - 1]
            return disc._discretized_var(data, attribute, points)
        else:
            return None


def join_contingency(contingency):
    """
    Join contingency list into a single ordered distribution.
    """
    k = len(contingency)
    values = numpy.r_[tuple(contingency[i][0] for i in range(k))]
#    counts = numpy.r_[tuple(contingency[i][1] for i in range(k))]
    I = numpy.zeros((len(values), k))
    start = 0
    for i in range(k):
        counts = contingency[i][1]
        span = len(counts)
        I[start: start + span, i] = contingency[i][1]
        start += span

    sort_ind = numpy.argsort(values)

    values, I = values[sort_ind], I[sort_ind, :]

    unique, uniq_index = numpy.unique(values, return_index=True)
    #
    spans = numpy.diff(numpy.r_[uniq_index, len(values)])
    I = [numpy.sum(I[start:start + span], axis=0)
         for start, span in zip(uniq_index, spans)]
    I = numpy.array(I)
    assert I.shape[0] == unique.shape[0]
    return unique, I


def main():
    app = QtGui.QApplication([])
    w = OWDiscretize()
    data = Orange.data.Table("iris")
    data = Orange.data.Table("brown-selected")
    w.set_data(data)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    import sys
    sys.exit(main())
