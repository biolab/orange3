import numpy as np

from AnyQt.QtWidgets import QGridLayout, QLabel, QLineEdit, QSizePolicy
from AnyQt.QtCore import QSize, Qt

from Orange.data import StringVariable, DiscreteVariable, Domain
from Orange.data.table import Table
from Orange.statistics.util import bincount
from Orange.preprocess.transformation import Transformation, Lookup
from Orange.widgets import gui, widget
from Orange.widgets.settings import DomainContextHandler, ContextSetting
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import Msg


def map_by_substring(a, patterns):
    res = np.full(len(a), np.nan)
    for val_idx, pattern in reversed(list(enumerate(patterns))):
        res[np.char.find(a, pattern) != -1] = val_idx
    return res


class ClassFromStringSubstring(Transformation):
    def __init__(self, variable, patterns):
        super().__init__(variable)
        self.patterns = patterns

    def transform(self, c):
        nans = np.equal(c, None)
        c = c.astype(str)
        c[nans] = ""
        res = map_by_substring(c, self.patterns)
        res[nans] = np.nan
        return res


class ClassFromDiscreteSubstring(Lookup):
    def __init__(self, variable, patterns):
        lookup_table = map_by_substring(variable.values, patterns)
        super().__init__(variable, lookup_table)


class OWCreateClass(widget.OWWidget):
    name = "Create Class"
    description = "Create class attribute from a string attribute"
    icon = "icons/CreateClass.svg"
    category = "Data"
    keywords = ["data"]

    inputs = [("Data", Table, "set_data")]
    outputs = [("Data", Table)]

    want_main_area = False

    settingsHandler = DomainContextHandler()
    attribute = ContextSetting(None)
    rules = ContextSetting({})

    TRANSFORMERS = {StringVariable: ClassFromStringSubstring,
                    DiscreteVariable: ClassFromDiscreteSubstring}

    class Warning(widget.OWWidget.Warning):
        no_nonnumeric_vars = Msg("Data contains only numeric variables.")

    def __init__(self):
        super().__init__()
        self.data = None
        self.line_edits = []
        self.remove_buttons = []
        self.counts = []
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        box = gui.hBox(self.controlArea)
        gui.widgetLabel(box, "Class from column: ", addSpace=12)
        gui.comboBox(
            box, self, "attribute", callback=self.update_rules,
            model=DomainModel(valid_types=(StringVariable, DiscreteVariable)),
            sizePolicy=(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))

        self.rules_box = rules_box = QGridLayout()
        self.controlArea.layout().addLayout(self.rules_box)
        self.add_button = gui.button(None, self, "+", flat=True,
                                     callback=self.add_row,
                                     minimumSize=QSize(12, 20))
        self.rules_box.setColumnMinimumWidth(1, 80)
        self.rules_box.setColumnMinimumWidth(0, 10)
        self.rules_box.setColumnStretch(0, 1)
        self.rules_box.setColumnStretch(1, 1)
        self.rules_box.setColumnStretch(2, 100)
        rules_box.addWidget(QLabel("Name"), 0, 1)
        rules_box.addWidget(QLabel("Pattern"), 0, 2)
        rules_box.addWidget(QLabel("#Instances"), 0, 3, 1, 2)
        self.update_rules()

        box = gui.hBox(self.controlArea)
        gui.rubber(box)
        gui.button(box, self, "Apply", autoDefault=False, callback=self.apply)

    @property
    def active_rules(self):
        return self.rules.setdefault(self.attribute and self.attribute.name,
                                     [["C1", ""], ["C2", ""]])

    def rules_to_edits(self):
        for editr, textr in zip(self.line_edits, self.active_rules):
            for edit, text in zip(editr, textr):
                edit.setText(text)

    def set_data(self, data):
        self.closeContext()
        self.rules = {}
        self.data = data
        model = self.controls.attribute.model()
        model.set_domain(data and data.domain)
        self.Warning.no_nonnumeric_vars(shown=data is not None and not model)
        if not model:
            self.attribute = None
            self.send("Data", None)
            return
        self.attribute = model[0]
        self.openContext(data)
        self.update_rules()
        self.apply()

    def update_rules(self):
        self.adjust_n_rule_rows()
        self.rules_to_edits()
        self.update_counts()

    def adjust_n_rule_rows(self):
        def _add_line():
            self.line_edits.append([])
            n_lines = len(self.line_edits)
            for coli in range(1, 3):
                edit = QLineEdit()
                self.line_edits[-1].append(edit)
                self.rules_box.addWidget(edit, n_lines, coli)
                edit.textChanged.connect(self.sync_edit)
            button = gui.button(
                None, self, label='×', flat=True, height=20,
                styleSheet='* {font-size: 16pt; color: silver}'
                           '*:hover {color: black}',
                callback=self.remove_row)
            button.setMinimumSize(QSize(12, 20))
            self.remove_buttons.append(button)
            self.rules_box.addWidget(button, n_lines, 0)
            self.counts.append([])
            for coli, kwargs in enumerate(
                    (dict(alignment=Qt.AlignRight),
                     dict(alignment=Qt.AlignLeft, styleSheet="color: gray"))):
                label = QLabel(**kwargs)
                self.counts[-1].append(label)
                self.rules_box.addWidget(label, n_lines, 3 + coli)

        def _remove_line():
            for edit in self.line_edits.pop():
                edit.deleteLater()
            self.remove_buttons.pop().deleteLater()
            for label in self.counts.pop():
                label.deleteLater()

        def _fix_tab_order():
            prev = None
            for row, rule in zip(self.line_edits, self.active_rules):
                for col_idx, edit in enumerate(row):
                    edit.row, edit.col_idx = rule, col_idx
                    if prev is not None:
                        self.setTabOrder(prev, edit)
                    prev = edit

        n = len(self.active_rules)
        while n > len(self.line_edits):
            _add_line()
        while len(self.line_edits) > n:
            _remove_line()
        self.rules_box.addWidget(self.add_button, n + 1, 0)
        _fix_tab_order()

    def add_row(self):
        self.active_rules.append(["", ""])
        self.adjust_n_rule_rows()

    def remove_row(self):
        remove_idx = self.remove_buttons.index(self.sender())
        del self.active_rules[remove_idx]
        self.update_rules()

    def sync_edit(self, text):
        edit = self.sender()
        edit.row[edit.col_idx] = text
        self.update_counts()

    def update_counts(self):
        def _set_labels(labels, matching, total_matching):
            n_matched = int(np.sum(matching))
            n_before = int(np.sum(total_matching)) - n_matched
            labels[0].setText("{}".format(n_matched))
            if n_before:
                labels[1].setText("+ {}".format(n_before))

        def _string_counts(data):
            data = data.astype(str)
            data = data[~np.char.equal(data, "")]
            remaining = np.array(data)
            for labels, (_, pattern) in zip(self.counts, self.active_rules):
                matching = np.char.find(remaining, pattern) != -1
                total_matching = np.char.find(data, pattern) != -1
                _set_labels(labels, matching, total_matching)
                remaining = remaining[~matching]
                if len(remaining) == 0:
                    break

        def _discrete_counts(data):
            attr_vals = np.array(attr.values)
            bins = bincount(data, max_val=len(attr.values) - 1)[0]
            remaining = np.array(bins)
            for labels, (_, pattern) in zip(self.counts, self.active_rules):
                matching = np.char.find(attr_vals, pattern) != -1
                _set_labels(labels, remaining[matching], bins[matching])
                remaining[matching] = 0
                if not np.any(remaining):
                    break

        for labels in self.counts:
            for label in labels:
                label.setText("")
        attr = self.attribute
        if attr is None:
            return
        data = self.data.get_column_view(attr)[0]
        if isinstance(attr, StringVariable):
            _string_counts(data)
        else:
            _discrete_counts(data)

    def apply(self):
        if not self.attribute or not self.active_rules:
            self.send("Data", None)
            return
        domain = self.data.domain
        names, patterns = \
            zip(*((name.strip(), pattern)
                  for name, pattern in self.active_rules if name.strip()))
        transformer = self.TRANSFORMERS[type(self.attribute)]
        new_class = DiscreteVariable(
            "class", names, compute_value=transformer(self.attribute, patterns))
        new_domain = Domain(domain.attributes, new_class,
                            domain.metas + domain.class_vars)
        new_data = Table(new_domain, self.data)
        self.send("Data", new_data)


def main():  # pragma: no cover
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    table = Table("zoo")
    ow = OWCreateClass()
    ow.show()
    ow.set_data(table)
    a.exec()
    ow.saveSettings()

if __name__ == "__main__":  # pragma: no cover
    main()
