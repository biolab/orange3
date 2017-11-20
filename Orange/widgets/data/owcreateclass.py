"""Widget for creating classes from non-numeric attribute by substrings"""
import re
from itertools import count

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
from Orange.widgets.widget import Msg, Input, Output


def map_by_substring(a, patterns, case_sensitive, match_beginning):
    """
    Map values in a using a list of patterns. The patterns are considered in
    order of appearance.

    Args:
        a (np.array): input array of `dtype` `str`
        patterns (list of str): list of stirngs
        case_sensitive (bool): case sensitive match
        match_beginning (bool): match only at the beginning of the string

    Returns:
        np.array of floats representing indices of matched patterns
    """
    res = np.full(len(a), np.nan)
    if not case_sensitive:
        a = np.char.lower(a)
        patterns = (pattern.lower() for pattern in patterns)
    for val_idx, pattern in reversed(list(enumerate(patterns))):
        indices = np.char.find(a, pattern)
        matches = indices == 0 if match_beginning else indices != -1
        res[matches] = val_idx
    return res


class ValueFromStringSubstring(Transformation):
    """
    Transformation that computes a discrete variable from a string variable by
    pattern matching.

    Given patterns `["abc", "a", "bc", ""]`, string data
    `["abcd", "aa", "bcd", "rabc", "x"]` is transformed to values of the new
    attribute with indices`[0, 1, 2, 0, 3]`.

    Args:
        variable (:obj:`~Orange.data.StringVariable`): the original variable
        patterns (list of str): list of string patterns
        case_sensitive (bool, optional): if set to `True`, the match is case
            sensitive
        match_beginning (bool, optional): if set to `True`, the pattern must
            appear at the beginning of the string
    """
    def __init__(self, variable, patterns,
                 case_sensitive=False, match_beginning=False):
        super().__init__(variable)
        self.patterns = patterns
        self.case_sensitive = case_sensitive
        self.match_beginning = match_beginning

    def transform(self, c):
        """
        Transform the given data.

        Args:
            c (np.array): an array of type that can be cast to dtype `str`

        Returns:
            np.array of floats representing indices of matched patterns
        """
        nans = np.equal(c, None)
        c = c.astype(str)
        c[nans] = ""
        res = map_by_substring(
            c, self.patterns, self.case_sensitive, self.match_beginning)
        res[nans] = np.nan
        return res


class ValueFromDiscreteSubstring(Lookup):
    """
    Transformation that computes a discrete variable from discrete variable by
    pattern matching.

    Say that the original attribute has values
    `["abcd", "aa", "bcd", "rabc", "x"]`. Given patterns
    `["abc", "a", "bc", ""]`, the values are mapped to the values of the new
    attribute with indices`[0, 1, 2, 0, 3]`.

    Args:
        variable (:obj:`~Orange.data.DiscreteVariable`): the original variable
        patterns (list of str): list of string patterns
        case_sensitive (bool, optional): if set to `True`, the match is case
            sensitive
        match_beginning (bool, optional): if set to `True`, the pattern must
            appear at the beginning of the string
    """
    def __init__(self, variable, patterns,
                 case_sensitive=False, match_beginning=False):
        super().__init__(variable, [])
        self.case_sensitive = case_sensitive
        self.match_beginning = match_beginning
        self.patterns = patterns  # Finally triggers computation of the lookup

    def __setattr__(self, key, value):
        """__setattr__ is overloaded to recompute the lookup table when the
        patterns, the original attribute or the flags change."""
        super().__setattr__(key, value)
        if hasattr(self, "patterns") and \
                key in ("case_sensitive", "match_beginning", "patterns",
                        "variable"):
            self.lookup_table = map_by_substring(
                self.variable.values, self.patterns,
                self.case_sensitive, self.match_beginning)


class OWCreateClass(widget.OWWidget):
    name = "Create Class"
    description = "Create class attribute from a string attribute"
    icon = "icons/CreateClass.svg"
    category = "Data"
    keywords = ["data"]

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    want_main_area = False

    settingsHandler = DomainContextHandler()
    attribute = ContextSetting(None)
    class_name = ContextSetting("class")
    rules = ContextSetting({})
    match_beginning = ContextSetting(False)
    case_sensitive = ContextSetting(False)

    TRANSFORMERS = {StringVariable: ValueFromStringSubstring,
                    DiscreteVariable: ValueFromDiscreteSubstring}

    class Warning(widget.OWWidget.Warning):
        no_nonnumeric_vars = Msg("Data contains only numeric variables.")

    class Error(widget.OWWidget.Error):
        class_name_duplicated = Msg("Class name duplicated.")
        class_name_empty = Msg("Class name should not be empty.")

    def __init__(self):
        super().__init__()
        self.data = None

        # The following lists are of the same length as self.active_rules

        #: list of pairs with counts of matches for each patter when the
        #     patterns are applied in order and when applied on the entire set,
        #     disregarding the preceding patterns
        self.match_counts = []

        #: list of list of QLineEdit: line edit pairs for each pattern
        self.line_edits = []
        #: list of QPushButton: list of remove buttons
        self.remove_buttons = []
        #: list of list of QLabel: pairs of labels with counts
        self.counts = []

        combo = gui.comboBox(
            self.controlArea, self, "attribute", label="From column: ",
            box=True, orientation=Qt.Horizontal, callback=self.update_rules,
            model=DomainModel(valid_types=(StringVariable, DiscreteVariable)))
        # Don't use setSizePolicy keyword argument here: it applies to box,
        # not the combo
        combo.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)

        patternbox = gui.vBox(self.controlArea, box=True)
        #: QWidget: the box that contains the remove buttons, line edits and
        #    count labels. The lines are added and removed dynamically.
        self.rules_box = rules_box = QGridLayout()
        patternbox.layout().addLayout(self.rules_box)
        box = gui.hBox(patternbox)
        gui.button(
            box, self, "+", callback=self.add_row, autoDefault=False, flat=True,
            minimumSize=(QSize(20, 20)))
        gui.rubber(box)
        self.rules_box.setColumnMinimumWidth(1, 70)
        self.rules_box.setColumnMinimumWidth(0, 10)
        self.rules_box.setColumnStretch(0, 1)
        self.rules_box.setColumnStretch(1, 1)
        self.rules_box.setColumnStretch(2, 100)
        rules_box.addWidget(QLabel("Name"), 0, 1)
        rules_box.addWidget(QLabel("Substring"), 0, 2)
        rules_box.addWidget(QLabel("#Instances"), 0, 3, 1, 2)
        self.update_rules()

        gui.lineEdit(
            self.controlArea, self, "class_name",
            label="Name for the new class:",
            box=True, orientation=Qt.Horizontal)

        optionsbox = gui.vBox(self.controlArea, box=True)
        gui.checkBox(
            optionsbox, self, "match_beginning", "Match only at the beginning",
            callback=self.options_changed)
        gui.checkBox(
            optionsbox, self, "case_sensitive", "Case sensitive",
            callback=self.options_changed)

        box = gui.hBox(self.controlArea)
        gui.rubber(box)
        gui.button(box, self, "Apply", autoDefault=False, width=180,
                   callback=self.apply)

        # TODO: Resizing upon changing the number of rules does not work
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    @property
    def active_rules(self):
        """
        Returns the class names and patterns corresponding to the currently
            selected attribute. If the attribute is not yet in the dictionary,
            set the default.
        """
        return self.rules.setdefault(self.attribute and self.attribute.name,
                                     [["", ""], ["", ""]])

    def rules_to_edits(self):
        """Fill the line edites with the rules from the current settings."""
        for editr, textr in zip(self.line_edits, self.active_rules):
            for edit, text in zip(editr, textr):
                edit.setText(text)

    @Inputs.data
    def set_data(self, data):
        """Input data signal handler."""
        self.closeContext()
        self.rules = {}
        self.data = data
        model = self.controls.attribute.model()
        model.set_domain(data and data.domain)
        self.Warning.no_nonnumeric_vars(shown=data is not None and not model)
        if not model:
            self.attribute = None
            self.Outputs.data.send(None)
            return
        self.attribute = model[0]
        self.openContext(data)
        self.update_rules()
        self.apply()

    def update_rules(self):
        """Called when the rules are changed: adjust the number of lines in
        the form and fill them, update the counts. The widget does not have
        auto-apply."""
        self.adjust_n_rule_rows()
        self.rules_to_edits()
        self.update_counts()
        # TODO: Indicator that changes need to be applied

    def options_changed(self):
        self.update_counts()

    def adjust_n_rule_rows(self):
        """Add or remove lines if needed and fix the tab order."""
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
                autoDefault=False, callback=self.remove_row)
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
        _fix_tab_order()

    def add_row(self):
        """Append a new row at the end."""
        self.active_rules.append(["", ""])
        self.adjust_n_rule_rows()
        self.update_counts()

    def remove_row(self):
        """Remove a row."""
        remove_idx = self.remove_buttons.index(self.sender())
        del self.active_rules[remove_idx]
        self.update_rules()
        self.update_counts()

    def sync_edit(self, text):
        """Handle changes in line edits: update the active rules and counts"""
        edit = self.sender()
        edit.row[edit.col_idx] = text
        self.update_counts()

    def class_labels(self):
        """Construct a list of class labels. Empty labels are replaced with
        C1, C2, C3. If C<n> already appears in the list of values given by
        the user, the labels start at C<n+1> instead.
        """
        largest_c = max((int(label[1:]) for label, _ in self.active_rules
                         if re.match("^C\\d+", label)),
                        default=0)
        class_count = count(largest_c + 1)
        return [label_edit.text() or "C{}".format(next(class_count))
                for label_edit, _ in self.line_edits]

    def update_counts(self):
        """Recompute and update the counts of matches."""
        def _matcher(strings, pattern):
            """Return indices of strings into patterns; consider case
            sensitivity and matching at the beginning. The given strings are
            assumed to be in lower case if match is case insensitive. Patterns
            are fixed on the fly."""
            if not self.case_sensitive:
                pattern = pattern.lower()
            indices = np.char.find(strings, pattern.strip())
            return indices == 0 if self.match_beginning else indices != -1

        def _lower_if_needed(strings):
            return strings if self.case_sensitive else np.char.lower(strings)

        def _string_counts():
            """
            Generate pairs of arrays for each rule until running out of data
            instances. np.sum over the two arrays in each pair gives the
            number of matches of the remaining instances (considering the
            order of patterns) and of the original data.

            For _string_counts, the arrays contain bool masks referring to the
            original data
            """
            nonlocal data
            data = data.astype(str)
            data = data[~np.char.equal(data, "")]
            data = _lower_if_needed(data)
            remaining = np.array(data)
            for _, pattern in self.active_rules:
                matching = _matcher(remaining, pattern)
                total_matching = _matcher(data, pattern)
                yield matching, total_matching
                remaining = remaining[~matching]
                if len(remaining) == 0:
                    break

        def _discrete_counts():
            """
            Generate pairs similar to _string_counts, except that the arrays
            contain bin counts for the attribute's values matching the pattern.
            """
            attr_vals = np.array(attr.values)
            attr_vals = _lower_if_needed(attr_vals)
            bins = bincount(data, max_val=len(attr.values) - 1)[0]
            remaining = np.array(bins)
            for _, pattern in self.active_rules:
                matching = _matcher(attr_vals, pattern)
                yield remaining[matching], bins[matching]
                remaining[matching] = 0
                if not np.any(remaining):
                    break

        def _clear_labels():
            """Clear all labels"""
            for lab_matched, lab_total in self.counts:
                lab_matched.setText("")
                lab_total.setText("")

        def _set_labels():
            """Set the labels to show the counts"""
            for (n_matched, n_total), (lab_matched, lab_total), (lab, patt) in \
                    zip(self.match_counts, self.counts, self.active_rules):
                n_before = n_total - n_matched
                lab_matched.setText("{}".format(n_matched))
                if n_before and (lab or patt):
                    lab_total.setText("+ {}".format(n_before))
                    if n_matched:
                        tip = "{} of the {} matching instances are already " \
                              "covered above".format(n_before, n_total)
                    else:
                        tip = "All matching instances are already covered above"
                    lab_total.setToolTip(tip)
                    lab_matched.setToolTip(tip)

        def _set_placeholders():
            """Set placeholders for empty edit lines"""
            matches = [n for n, _ in self.match_counts] + \
                      [0] * len(self.line_edits)
            for n_matched, (_, patt) in zip(matches, self.line_edits):
                if not patt.text():
                    patt.setPlaceholderText(
                        "(remaining instances)" if n_matched else "(unused)")

            labels = self.class_labels()
            for label, (lab_edit, _) in zip(labels, self.line_edits):
                if not lab_edit.text():
                    lab_edit.setPlaceholderText(label)

        _clear_labels()
        attr = self.attribute
        if attr is None:
            return
        counters = {StringVariable: _string_counts,
                    DiscreteVariable: _discrete_counts}
        data = self.data.get_column_view(attr)[0]
        self.match_counts = [[int(np.sum(x)) for x in matches]
                             for matches in counters[type(attr)]()]
        _set_labels()
        _set_placeholders()

    def apply(self):
        """Output the transformed data."""
        self.Error.clear()
        self.class_name = self.class_name.strip()
        if not self.attribute:
            self.Outputs.data.send(None)
            return
        domain = self.data.domain
        if not len(self.class_name):
            self.Error.class_name_empty()
        if self.class_name in domain:
            self.Error.class_name_duplicated()
        if not len(self.class_name) or self.class_name in domain:
            self.Outputs.data.send(None)
            return
        rules = self.active_rules
        # Transposition + stripping
        valid_rules = [label or pattern or n_matches
                       for (label, pattern), n_matches in
                       zip(rules, self.match_counts)]
        patterns = [pattern
                    for (_, pattern), valid in zip(rules, valid_rules)
                    if valid]
        names = [name for name, valid in zip(self.class_labels(), valid_rules)
                 if valid]
        transformer = self.TRANSFORMERS[type(self.attribute)]
        compute_value = transformer(
            self.attribute, patterns, self.case_sensitive, self.match_beginning)
        new_class = DiscreteVariable(
            self.class_name, names, compute_value=compute_value)
        new_domain = Domain(
            domain.attributes, new_class, domain.metas + domain.class_vars)
        new_data = self.data.transform(new_domain)
        self.Outputs.data.send(new_data)

    def send_report(self):
        def _cond_part():
            rule = "<b>{}</b> ".format(class_name)
            if patt:
                rule += "if <b>{}</b> contains <b>{}</b>".format(
                    self.attribute.name, patt)
            else:
                rule += "otherwise"
            return rule

        def _count_part():
            if not n_matched:
                return "all {} matching instances are already covered " \
                       "above".format(n_total)
            elif n_matched < n_total and patt:
                return "{} matching instances (+ {} that are already " \
                       "covered above".format(n_matched, n_total - n_matched)
            else:
                return "{} matching instances".format(n_matched)

        if not self.attribute:
            return
        self.report_items("Input", [("Source attribute", self.attribute.name)])
        output = ""
        names = self.class_labels()
        for (n_matched, n_total), class_name, (lab, patt) in \
                zip(self.match_counts, names, self.active_rules):
            if lab or patt or n_total:
                output += "<li>{}; {}</li>".format(_cond_part(), _count_part())
        if output:
            self.report_items("Output", [("Class name", self.class_name)])
            self.report_raw("<ol>{}</ol>".format(output))


def main():  # pragma: no cover
    """Simple test for manual inspection of the widget"""
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
