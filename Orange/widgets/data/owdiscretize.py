from collections import namedtuple

from PyQt4 import QtGui
from PyQt4.QtGui import (
    QListView, QHBoxLayout, QStyledItemDelegate, QDialogButtonBox
)
from PyQt4.QtCore import Qt

import Orange.data
import Orange.preprocess.discretize as disc

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels, vartype
from Orange.widgets.widget import OutputSignal, InputSignal

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
    MDL: lambda m, data, var: disc.EntropyMDL()(data, var),
    EqualFreq: lambda m, data, var: disc.EqualFreq(m.k)(data, var),
    EqualWidth: lambda m, data, var: disc.EqualWidth(m.k)(data, var),
    Remove: lambda m, data, var: None,
    Custom: lambda m, data, var:
        disc.Discretizer.create_discretized_var(var, m.points)
}


# Variable discretization state
DState = namedtuple(
    "DState",
    ["method",    # discretization method
     "points",    # induced cut points
     "disc_var"]  # induced discretized variable
)


def is_derived(var):
    return var.compute_value is not None


def is_discretized(var):
    return isinstance(var.compute_value, disc.Discretizer)


def variable_key(var):
    return vartype(var), var.name


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
    description = "Discretize the continuous data features."
    icon = "icons/Discretize.svg"
    inputs = [InputSignal("Data", Orange.data.Table, "set_data",
                          doc="Input data table")]
    outputs = [OutputSignal("Data", Orange.data.Table,
                            doc="Table with discretized features")]

    settingsHandler = settings.DomainContextHandler()
    saved_var_states = settings.ContextSetting({})

    default_method = settings.Setting(0)
    default_k = settings.Setting(5)
    autosend = settings.Setting(True)

    #: Discretization methods
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
        self.default_bbox = rbox = gui.radioButtons(
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

        s = gui.hSlider(gui.indentedBox(rbox),
                        self, "default_k", minValue=2, maxValue=10,
                        label="Num. of intervals:",
                        callback=self._default_disc_changed)
        s.setTracking(False)

        gui.appendRadioButton(rbox, options[-1])

        vlayout = QHBoxLayout()
        box = gui.widgetBox(
            self.controlArea, "Individual Attribute Settings",
            orientation=vlayout, spacing=8
        )

        # List view with all attributes
        self.varview = QListView(selectionMode=QListView.ExtendedSelection)
        self.varview.setItemDelegate(DiscDelegate())
        self.varmodel = itemmodels.VariableListModel()
        self.varview.setModel(self.varmodel)
        self.varview.selectionModel().selectionChanged.connect(
            self._var_selection_changed
        )

        vlayout.addWidget(self.varview)
        # Controls for individual attr settings
        self.bbox = controlbox = gui.radioButtons(
            box, self, "method", callback=self._disc_method_changed
        )
        vlayout.addWidget(controlbox)

        for opt in options[:5]:
            gui.appendRadioButton(controlbox, opt)

        s = gui.hSlider(gui.indentedBox(controlbox),
                        self, "k", minValue=2, maxValue=10,
                        label="Num. of intervals:",
                        callback=self._disc_method_changed)
        s.setTracking(False)

        gui.appendRadioButton(controlbox, "Remove attribute")

        gui.rubber(controlbox)
        controlbox.setEnabled(False)

        self.controlbox = controlbox

        gui.auto_commit(self.controlArea, self, "autosend", "Apply",
                        orientation="horizontal",
                        checkbox_label="Send data after every change")

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
        self.unconditional_commit()

    def _initialize(self, data):
        # Initialize the default variable states for new data.
        self.class_var = data.domain.class_var
        cvars = [var for var in data.domain if var.is_continuous]
        self.varmodel[:] = cvars

        class_var = data.domain.class_var
        has_disc_class = data.domain.has_discrete_class

        self.default_bbox.buttons[self.MDL - 1].setEnabled(has_disc_class)
        self.bbox.buttons[self.MDL].setEnabled(has_disc_class)

        # If the newly disabled MDL button is checked then change it
        if not has_disc_class and self.default_method == self.MDL - 1:
            self.default_method = 0
        if not has_disc_class and self.method == self.MDL:
            self.method = 0

        # Reset (initialize) the variable discretization states.
        self._reset()

    def _restore(self, saved_state):
        # Restore variable states from a saved_state dictionary.
        def_method = self._current_default_method()
        for i, var in enumerate(self.varmodel):
            key = variable_key(var)
            if key in saved_state:
                state = saved_state[key]
                if isinstance(state.method, Default):
                    state = DState(Default(def_method), None, None)
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
        self.default_bbox.buttons[self.MDL - 1].setEnabled(True)
        self.bbox.buttons[self.MDL].setEnabled(True)

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
                return dvar.compute_value.points, dvar
            else:
                assert False
        for i, var in enumerate(self.varmodel):
            state = self.var_state[i]
            if state.points is None and state.disc_var is None:
                points, dvar = induce_cuts(state.method, self.data, var)
                new_state = state._replace(points=points, disc_var=dvar)
                self._set_var_state(i, new_state)
        self.commit()

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
            if source and source.is_continuous:
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
            output = self.data.from_table(domain, self.data)
        self.send("Data", output)

    def storeSpecificSettings(self):
        super().storeSpecificSettings()
        self.saved_var_states = {
            variable_key(var):
                self.var_state[i]._replace(points=None, disc_var=None)
            for i, var in enumerate(self.varmodel)
        }


def main():
    app = QtGui.QApplication([])
    w = OWDiscretize()
    data = Orange.data.Table("brown-selected")
    w.set_data(data)
    w.set_data(None)
    w.set_data(data)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    import sys
    sys.exit(main())
