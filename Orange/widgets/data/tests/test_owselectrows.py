# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import (
    Table, ContinuousVariable, StringVariable, DiscreteVariable)
from Orange.widgets.data.owselectrows import OWSelectRows, FilterDiscreteType
from Orange.widgets.tests.base import WidgetTest

from Orange.data.filter import FilterContinuous, FilterString

CFValues = {
    FilterContinuous.Equal: ["5.4"],
    FilterContinuous.NotEqual: ["5.4"],
    FilterContinuous.Less: ["5.4"],
    FilterContinuous.LessEqual: ["5.4"],
    FilterContinuous.Greater: ["5.4"],
    FilterContinuous.GreaterEqual: ["5.4"],
    FilterContinuous.Between: ["5.4", "6.0"],
    FilterContinuous.Outside: ["5.4", "6.0"],
    FilterContinuous.IsDefined: [],
}


SFValues = {
    FilterString.Equal: ["aardwark"],
    FilterString.NotEqual: ["aardwark"],
    FilterString.Less: ["aardwark"],
    FilterString.LessEqual: ["aardwark"],
    FilterString.Greater: ["aardwark"],
    FilterString.GreaterEqual: ["aardwark"],
    FilterString.Between: ["aardwark", "cat"],
    FilterString.Outside: ["aardwark"],
    FilterString.Contains: ["aa"],
    FilterString.StartsWith: ["aa"],
    FilterString.EndsWith: ["ark"],
    FilterString.IsDefined: []
}

DFValues = {
    FilterDiscreteType.Equal: [0],
    FilterDiscreteType.NotEqual: [0],
    FilterDiscreteType.In: [0, 1],
    FilterDiscreteType.IsDefined: [],
}


class TestOWSelectRows(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSelectRows)  # type: OWSelectRows

    def test_filter_cont(self):
        iris = Table("iris")[::5]
        self.widget.auto_commit = True
        self.widget.set_data(iris)

        for i, (op, _) in enumerate(OWSelectRows.Operators[ContinuousVariable]):
            self.widget.remove_all()
            self.widget.add_row(0, i, CFValues[op])
            self.widget.conditions_changed()
            self.widget.unconditional_commit()

    def test_filter_str(self):
        zoo = Table("zoo")[::5]
        self.widget.auto_commit = False
        self.widget.set_data(zoo)
        var_idx = len(zoo.domain)
        for i, (op, _) in enumerate(OWSelectRows.Operators[StringVariable]):
            self.widget.remove_all()
            self.widget.add_row(var_idx, i, SFValues[op])
            self.widget.conditions_changed()
            self.widget.unconditional_commit()

    def test_filter_disc(self):
        lenses = Table("lenses")
        self.widget.auto_commit = False
        self.widget.set_data(lenses)

        for i, (op, _) in enumerate(OWSelectRows.Operators[DiscreteVariable]):
            self.widget.remove_all()
            self.widget.add_row(0, i, DFValues[op])
            self.widget.conditions_changed()
            self.widget.unconditional_commit()
