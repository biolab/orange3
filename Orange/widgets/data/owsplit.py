from functools import partial

import numpy as np

from AnyQt.QtCore import Qt

from orangewidget.settings import Setting

from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler
from Orange.widgets.widget import OWWidget, Msg, Output, Input
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.data import \
    Table, Domain, DiscreteVariable, StringVariable, ContinuousVariable
from Orange.data.util import SharedComputeValue, get_unique_names


def get_substrings(values, delimiter):
    return sorted({ss.strip() for s in values for ss in s.split(delimiter)}
                  - {""})


class SplitColumnBase:
    def __init__(self, data, attr, delimiter):
        self.attr = attr
        self.delimiter = delimiter
        column = set(data.get_column(self.attr))
        self.new_values = tuple(get_substrings(column, self.delimiter))

    def __eq__(self, other):
        return self.attr == other.attr \
               and self.delimiter == other.delimiter \
               and self.new_values == other.new_values

    def __hash__(self):
        return hash((self.attr, self.delimiter, self.new_values))


class SplitColumnOneHot(SplitColumnBase):
    InheritEq = True

    def __call__(self, data):
        column = data.get_column(self.attr)
        values = [{ss.strip() for ss in s.split(self.delimiter)}
                  for s in column]
        return {v: np.array([i for i, xs in enumerate(values) if v in xs],
                            dtype=int)
                for v in self.new_values}


class SplitColumnCounts(SplitColumnBase):
    InheritEq = True

    def __call__(self, data):
        column = data.get_column(self.attr)
        values = [[ss.strip() for ss in s.split(self.delimiter)]
                  for s in column]
        return {v: np.array([xs.count(v) for xs in values], dtype=float)
                for v in self.new_values}


class StringEncodingBase(SharedComputeValue):
    def __init__(self, fn, new_feature):
        super().__init__(fn)
        self.new_feature = new_feature

    def __eq__(self, other):
        return super().__eq__(other) and self.new_feature == other.new_feature

    def __hash__(self):
        return super().__hash__() ^ hash(self.new_feature)

    def compute(self, data, shared_data):
        raise NotImplementedError  # silence pylint

class OneHotStrings(StringEncodingBase):
    InheritEq = True

    def compute(self, data, shared_data):
        indices = shared_data[self.new_feature]
        col = np.zeros(len(data))
        col[indices] = 1
        return col


class CountStrings(StringEncodingBase):
    InheritEq = True

    def compute(self, data, shared_data):
        return shared_data[self.new_feature]


class DiscreteEncoding:
    def __init__(self, variable, delimiter, onehot, value):
        self.variable = variable
        self.delimiter = delimiter
        self.onehot = onehot
        self.value = value

    def __call__(self, data):
        column = data.get_column(self.variable).astype(float)
        col = np.zeros(len(column))
        col[np.isnan(column)] = np.nan
        for val_idx, value in enumerate(self.variable.values):
            parts = value.split(self.delimiter)
            if self.onehot:
                col[column == val_idx] = int(self.value in parts)
            else:
                col[column == val_idx] = parts.count(self.value)
        return col

    def __eq__(self, other):
        return self.variable == other.variable \
               and self.value == other.value \
               and self.delimiter == other.delimiter \
               and self.onehot == other.onehot

    def __hash__(self):
        return hash((self.variable, self.value, self.delimiter, self.onehot))


class OWSplit(OWWidget):
    name = "Split"
    description = "Split text or categorical variables into indicator variables"
    category = "Transform"
    icon = "icons/Split.svg"
    keywords = "text, columns, word, encoding, questionnaire, survey, term, counts, indicator"
    priority = 700

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    class Warning(OWWidget.Warning):
        no_disc = Msg("Data contains only numeric variables.")

    want_main_area = False
    resizing_enabled = False

    Categorical, Numerical, Counts = range(3)
    OutputLabels = ("Categorical (No, Yes)", "Numerical (0, 1)", "Counts")

    settingsHandler = DomainContextHandler()
    attribute = ContextSetting(None)
    delimiter = ContextSetting(";")
    output_type = ContextSetting(Categorical)
    auto_apply = Setting(True)

    def __init__(self):
        super().__init__()
        self.data = None

        variable_select_box = gui.vBox(self.controlArea, "Variable")

        gui.comboBox(variable_select_box, self, "attribute",
                     orientation=Qt.Horizontal, searchable=True,
                     callback=self.apply.deferred,
                     model=DomainModel(valid_types=(StringVariable,
                                                    DiscreteVariable)))
        le = gui.lineEdit(
                variable_select_box, self, "delimiter", "Delimiter: ",
                orientation=Qt.Horizontal, callback=self.apply.deferred,
                controlWidth=20)
        le.box.layout().addStretch(1)
        le.setAlignment(Qt.AlignCenter)

        gui.radioButtonsInBox(
            self.controlArea, self, "output_type", self.OutputLabels,
            box="Output Values",
            callback=self.apply.deferred)

        gui.auto_apply(self.buttonsArea, self, commit=self.apply)

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.data = data

        model = self.controls.attribute.model()
        model.set_domain(data.domain if data is not None else None)
        self.Warning.no_disc(shown=data is not None and not model)
        if not model:
            self.attribute = None
            self.data = None
            return
        self.attribute = model[0]
        self.openContext(data)
        self.apply.now()

    @gui.deferred
    def apply(self):
        if self.attribute is None:
            self.Outputs.data.send(None)
            return
        var = self.data.domain[self.attribute]
        values, computer = self._get_compute_value(var)
        new_columns = self._get_new_columns(values, computer)
        new_domain = Domain(
            self.data.domain.attributes + new_columns,
            self.data.domain.class_vars, self.data.domain.metas
        )
        extended_data = self.data.transform(new_domain)
        self.Outputs.data.send(extended_data)

    def _get_compute_value(self, var):
        if var.is_discrete:
            values = get_substrings(var.values, self.delimiter)
            computer = partial(
                DiscreteEncoding,
                var, self.delimiter, self.output_type != self.Counts)
        else:
            if self.output_type == self.Counts:
                sc = SplitColumnCounts(self.data, var, self.delimiter)
                computer = partial(CountStrings, sc)
            else:
                sc = SplitColumnOneHot(self.data, var, self.delimiter)
                computer = partial(OneHotStrings, sc)
            values = sc.new_values
        return values, computer

    def _get_new_columns(self, values, computer):
        names = get_unique_names(self.data.domain, values, equal_numbers=False)
        if self.output_type == self.Categorical:
            return tuple(
                DiscreteVariable(
                    name, ("No", "Yes"), compute_value=computer(value))
                for value, name in zip(values, names))
        else:
            return tuple(
                ContinuousVariable(
                    name, compute_value=computer(value))
                for value, name in zip(values, names))


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWSplit).run(Table.from_file("tests/orange-in-education.tab"))
