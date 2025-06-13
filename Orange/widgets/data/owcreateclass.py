"""Widget for creating classes from non-numeric attribute by substrings"""
import re
from itertools import count
from typing import Optional, Sequence

import numpy as np

from AnyQt.QtWidgets import QGridLayout, QLabel, QLineEdit, QSizePolicy, QWidget
from AnyQt.QtCore import Qt

from Orange.data import StringVariable, DiscreteVariable, Domain
from Orange.data.table import Table
from Orange.statistics.util import bincount
from Orange.preprocess.transformation import Transformation, Lookup
from Orange.widgets import gui, widget
from Orange.widgets.settings import DomainContextHandler, ContextSetting
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.localization import pl
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg, Input, Output


def map_by_substring(
        a: np.ndarray,
        patterns: list[str],
        case_sensitive: bool, match_beginning: bool, regular_expressions: bool,
        map_values: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    Map values in a using a list of patterns. The patterns are considered in
    order of appearance.

    Flags `match_beginning` and `regular_expressions` are incompatible.

    Args:
        a (np.array): input array of `dtype` `str`
        patterns (list of str): list of strings
        case_sensitive (bool): case sensitive match
        match_beginning (bool): match only at the beginning of the string
        map_values (list of int, optional):
            list of len(patterns); return values for each pattern
        regular_expressions (bool): use regular expressions

    Returns:
        np.array of floats representing indices of matched patterns
    """
    assert not (regular_expressions and match_beginning)
    if map_values is None:
        map_values = np.arange(len(patterns))
    else:
        map_values = np.array(map_values, dtype=int)
    res = np.full(len(a), np.nan)
    if not case_sensitive and not regular_expressions:
        a = np.char.lower(a)
        patterns = (pattern.lower() for pattern in patterns)
    for val_idx, pattern in reversed(list(enumerate(patterns))):
        # Note that similar code repeats in update_counts. Any changes here
        # should be reflected there.
        if regular_expressions:
            re_pattern = re.compile(pattern,
                                    re.IGNORECASE if not case_sensitive else 0)
            matches = np.array([bool(re_pattern.search(s)) for s in a],
                                dtype=bool)
        else:
            indices = np.char.find(a, pattern)
            matches = indices == 0 if match_beginning else indices != -1
        res[matches] = map_values[val_idx]
    return res


class _EqHashMixin:
    def __eq__(self, other):
        return super().__eq__(other) \
               and self.patterns == other.patterns \
               and self.case_sensitive == other.case_sensitive \
               and self.match_beginning == other.match_beginning \
               and self.regular_expressions == other.regular_expressions \
               and np.all(self.map_values == other.map_values)

    def __hash__(self):
        return hash((type(self), self.variable,
                     tuple(self.patterns),
                     self.case_sensitive, self.match_beginning,
                     self.regular_expressions,
                     None if self.map_values is None else tuple(self.map_values)
                    ))

class ValueFromStringSubstring(_EqHashMixin, Transformation):
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
        map_values (list of int, optional): return values for each pattern
        regular_expressions (bool, optional): if set to `True`, the patterns are
    """
    # regular_expressions was added later and at the end (instead of with other
    # flags) for compatibility with older existing pickles
    def __init__(
            self,
            variable: StringVariable,
            patterns: list[str],
            case_sensitive: bool = False,
            match_beginning: bool = False,
            map_values: Optional[Sequence[int]] = None,
            regular_expressions: bool = False):
        super().__init__(variable)
        self.patterns = patterns
        self.case_sensitive = case_sensitive
        self.match_beginning = match_beginning
        self.regular_expressions = regular_expressions
        self.map_values = map_values

    InheritEq = True

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
            c, self.patterns,
            self.case_sensitive, self.match_beginning, self.regular_expressions,
            self.map_values)
        res[nans] = np.nan
        return res


class ValueFromDiscreteSubstring(_EqHashMixin, Lookup):
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
        map_values (list of int, optional): return values for each pattern
        regular_expressions (bool, optional): if set to `True`, the patterns are

    """
    # regular_expressions was added later and at the end (instead of with other
    # flags) for compatibility with older existing pickles
    def __init__(
            self,
            variable: DiscreteVariable,
            patterns: list[str],
            case_sensitive: bool = False,
            match_beginning: bool = False,
            map_values: Optional[Sequence[int]] = None,
            regular_expressions: bool = False):
        super().__init__(variable, [])
        self.case_sensitive = case_sensitive
        self.match_beginning = match_beginning
        self.map_values = map_values
        self.regular_expressions = regular_expressions
        self.patterns = patterns  # Finally triggers computation of the lookup

    InheritEq = True

    def __setattr__(self, key, value):
        """__setattr__ is overloaded to recompute the lookup table when the
        patterns, the original attribute or the flags change."""
        super().__setattr__(key, value)
        if hasattr(self, "patterns") and \
                key in ("case_sensitive", "match_beginning", "patterns",
                        "variable", "map_values"):
            self.lookup_table = map_by_substring(
                self.variable.values, self.patterns,
                self.case_sensitive, self.match_beginning,
                self.regular_expressions, self.map_values)

def unique_in_order_mapping(a: Sequence[str]) -> tuple[list[str], list[int]]:
    """ Return
    - unique elements of the input list (in the order of appearance)
    - indices of the input list onto the returned uniques
    """
    first_position = {}
    unique_in_order = []
    mapping = []
    for e in a:
        if e not in first_position:
            first_position[e] = len(unique_in_order)
            unique_in_order.append(e)
        mapping.append(first_position[e])
    return unique_in_order, mapping


class OWCreateClass(widget.OWWidget):
    name = "Create Class"
    description = "Create class attribute from a string attribute"
    icon = "icons/CreateClass.svg"
    category = "Transform"
    keywords = "create class"
    priority = 2300

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    want_main_area = False
    buttons_area_orientation = Qt.Vertical

    settingsHandler = DomainContextHandler()
    attribute = ContextSetting(None)
    class_name = ContextSetting("class")
    rules = ContextSetting({})
    match_beginning = ContextSetting(False)
    case_sensitive = ContextSetting(False)
    regular_expressions = ContextSetting(False)

    TRANSFORMERS = {StringVariable: ValueFromStringSubstring,
                    DiscreteVariable: ValueFromDiscreteSubstring}

    # Cached variables are used so that two instances of the widget with the
    # same settings will create the same variable. The usual `make` wouldn't
    # work here because variables with `compute_value` are not reused.
    cached_variables = {}

    class Warning(widget.OWWidget.Warning):
        no_nonnumeric_vars = Msg("Data contains only numeric variables.")

    class Error(widget.OWWidget.Error):
        class_name_duplicated = Msg("Class name duplicated.")
        class_name_empty = Msg("Class name should not be empty.")
        invalid_regular_expression = Msg("Invalid regular expression: {}")

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

        gui.lineEdit(
            self.controlArea, self, "class_name",
            orientation=Qt.Horizontal, box="New Class Name")

        variable_select_box = gui.vBox(self.controlArea, "Match by Substring")

        combo = gui.comboBox(
            variable_select_box, self, "attribute", label="From column:",
            orientation=Qt.Horizontal, searchable=True,
            callback=self.update_rules,
            model=DomainModel(valid_types=(StringVariable, DiscreteVariable)))
        # Don't use setSizePolicy keyword argument here: it applies to box,
        # not the combo
        combo.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)

        patternbox = gui.vBox(variable_select_box)
        #: QWidget: the box that contains the remove buttons, line edits and
        #    count labels. The lines are added and removed dynamically.
        self.rules_box = rules_box = QGridLayout()
        rules_box.setSpacing(4)
        rules_box.setContentsMargins(4, 4, 4, 4)
        self.rules_box.setColumnMinimumWidth(1, 70)
        self.rules_box.setColumnMinimumWidth(0, 10)
        self.rules_box.setColumnStretch(0, 1)
        self.rules_box.setColumnStretch(1, 1)
        self.rules_box.setColumnStretch(2, 100)
        rules_box.addWidget(QLabel("Name"), 0, 1)
        rules_box.addWidget(QLabel("Substring"), 0, 2)
        rules_box.addWidget(QLabel("Count"), 0, 3, 1, 2)
        self.update_rules()

        widg = QWidget(patternbox)
        widg.setLayout(rules_box)
        patternbox.layout().addWidget(widg)

        box = gui.hBox(patternbox)
        gui.rubber(box)
        gui.button(box, self, "+", callback=self.add_row,
                   autoDefault=False, width=34,
                   sizePolicy=(QSizePolicy.Maximum,
                               QSizePolicy.Maximum))

        optionsbox = gui.vBox(self.controlArea, "Options")
        gui.checkBox(
            optionsbox, self, "regular_expressions", "Use regular expressions",
            callback=self.options_changed)
        gui.checkBox(
            optionsbox, self, "match_beginning", "Match only at the beginning",
            stateWhenDisabled=False,
            callback=self.options_changed)
        gui.checkBox(
            optionsbox, self, "case_sensitive", "Case sensitive",
            callback=self.options_changed)

        gui.rubber(self.controlArea)

        gui.button(self.buttonsArea, self, "Apply", callback=self.apply)

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
        model.set_domain(data.domain if data is not None else None)
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
        self.controls.match_beginning.setEnabled(not self.regular_expressions)
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
                None, self, label='×', width=33,
                autoDefault=False, callback=self.remove_row,
                sizePolicy=(QSizePolicy.Maximum,
                            QSizePolicy.Maximum)
            )
            self.remove_buttons.append(button)
            self.rules_box.addWidget(button, n_lines, 0)
            self.counts.append([])
            for coli, kwargs in enumerate(
                    ({},
                     {"styleSheet": "color: gray"})):
                label = QLabel(alignment=Qt.AlignCenter, **kwargs)
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
        return [label_edit.text() or f"C{next(class_count)}"
                for label_edit, _ in self.line_edits]

    def invalid_patterns(self):
        if not self.regular_expressions:
            return None
        for _, pattern in self.active_rules:
            try:
                re.compile(pattern)
            except re.error:
                return pattern
        return None

    def update_counts(self):
        """Recompute and update the counts of matches."""
        if self.regular_expressions:
            def _matcher(strings, pattern):
                # Note that similar code repeats in map_by_substring.
                # Any changes here should be reflected there.
                re_pattern = re.compile(
                    pattern,
                    re.IGNORECASE if not self.case_sensitive else 0)
                return np.array([bool(re_pattern.search(s)) for s in strings],
                                 dtype=bool)

            def _lower_if_needed(strings):
                return strings
        else:
            def _matcher(strings, pattern):
                """Return indices of strings into patterns; consider case
                sensitivity and matching at the beginning. The given strings are
                assumed to be in lower case if match is case insensitive. Patterns
                are fixed on the fly."""
                # Note that similar code repeats in map_by_substring.
                # Any changes here should be reflected there.
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
                if not remaining.size:
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
                lab_matched.setText(f"{n_matched}")
                if n_before and (lab or patt):
                    lab_total.setText(f"+ {n_before}")
                    if n_matched:
                        tip = f"{n_before} o" \
                              f"f {n_total} matching {pl(n_total, 'instance')} " \
                              f"{pl(n_before, 'is|are')} already covered above."
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
        if (invalid := self.invalid_patterns()) is not None:
            self.Error.invalid_regular_expression(invalid)
            return
        self.Error.invalid_regular_expression.clear()

        attr = self.attribute
        if attr is None:
            return
        counters = {StringVariable: _string_counts,
                    DiscreteVariable: _discrete_counts}
        data = self.data.get_column(attr)
        self.match_counts = [[int(np.sum(x)) for x in matches]
                             for matches in counters[type(attr)]()]
        _set_labels()
        _set_placeholders()

    def apply(self):
        """Output the transformed data."""
        self.Error.clear()
        if (invalid := self.invalid_patterns()) is not None:
            self.Error.invalid_regular_expression(invalid)
            self.Outputs.data.send(None)
            return

        self.class_name = self.class_name.strip()
        if not self.attribute:
            self.Outputs.data.send(None)
            return
        domain = self.data.domain
        if not self.class_name:
            self.Error.class_name_empty()
        if self.class_name in domain:
            self.Error.class_name_duplicated()
        if not self.class_name or self.class_name in domain:
            self.Outputs.data.send(None)
            return
        new_class = self._create_variable()
        new_domain = Domain(
            domain.attributes, new_class, domain.metas + domain.class_vars)
        new_data = self.data.transform(new_domain)
        self.Outputs.data.send(new_data)

    def _create_variable(self):
        rules = self.active_rules
        # Transposition + stripping
        valid_rules = [label or pattern or n_matches
                       for (label, pattern), n_matches in
                       zip(rules, self.match_counts)]
        patterns = tuple(
            pattern for (_, pattern), valid in zip(rules, valid_rules) if valid)
        names = tuple(
            name for name, valid in zip(self.class_labels(), valid_rules)
            if valid)
        transformer = self.TRANSFORMERS[type(self.attribute)]

        # join patterns with the same names
        names, map_values = unique_in_order_mapping(names)
        names = tuple(str(a) for a in names)
        map_values = tuple(map_values)

        var_key = (self.attribute, self.class_name, names,
                   patterns, self.case_sensitive, self.match_beginning,
                   self.regular_expressions, map_values)
        if var_key in self.cached_variables:
            return self.cached_variables[var_key]

        compute_value = transformer(
            self.attribute, patterns, self.case_sensitive,
            self.match_beginning and not self.regular_expressions,
            map_values, self.regular_expressions)
        new_var = DiscreteVariable(
            self.class_name, names, compute_value=compute_value)
        self.cached_variables[var_key] = new_var
        return new_var

    def send_report(self):
        # Pylint gives false positives: these functions are always called from
        # within the loop
        # pylint: disable=undefined-loop-variable
        def _cond_part():
            rule = f"<b>{class_name}</b> "
            if patt:
                rule += f"if <b>{self.attribute.name}</b> contains <b>{patt}</b>"
            else:
                rule += "otherwise"
            return rule

        def _count_part():
            aca = "already covered above"
            if not n_matched:
                if n_total == 1:
                    return f"the single matching instance is {aca}"
                elif n_total == 2:
                    return f"both matching instances are {aca}"
                else:
                    return f"all {n_total} matching instances are {aca}"
            elif not patt:
                return f"{n_matched} {pl(n_matched, 'instance')}"
            else:
                m = f"{n_matched} matching {pl(n_matched, 'instance')}"
                if n_matched < n_total:
                    n_already = n_total - n_matched
                    m += f" (+{n_already} that {pl(n_already, 'is|are')} {aca})"
                return m

        if not self.attribute:
            return
        self.report_items("Input", [("Source attribute", self.attribute.name)])
        output = ""
        names = self.class_labels()
        for (n_matched, n_total), class_name, (lab, patt) in \
                zip(self.match_counts, names, self.active_rules):
            if lab or patt or n_total:
                output += f"<li>{_cond_part()}; {_count_part()}</li>"
        if output:
            self.report_items("Output", [("Class name", self.class_name)])
            self.report_raw(f"<ol>{output}</ol>")


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWCreateClass).run(Table("zoo"))
