# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import unittest
from unittest.mock import patch

import numpy as np

from Orange.data import Table, StringVariable, DiscreteVariable
from Orange.widgets.data.owcreateclass import (
    OWCreateClass,
    map_by_substring, ValueFromStringSubstring, ValueFromDiscreteSubstring)
from Orange.widgets.tests.base import WidgetTest


class TestHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patterns = ["abc", "a", "bc", ""]
        cls.arr = np.array(["abcd", "aa", "bcd", "rabc", "x"])

    def test_map_by_substring(self):
        np.testing.assert_equal(
            map_by_substring(self.arr,
                             ["abc", "a", "bc", ""],
                             case_sensitive=True, match_beginning=False),
            [0, 1, 2, 0, 3])
        np.testing.assert_equal(
            map_by_substring(self.arr,
                             ["abc", "a", "Bc", ""],
                             case_sensitive=True, match_beginning=False),
            [0, 1, 3, 0, 3])
        np.testing.assert_equal(
            map_by_substring(self.arr,
                             ["abc", "a", "Bc", ""],
                             case_sensitive=False, match_beginning=False),
            [0, 1, 2, 0, 3])
        np.testing.assert_equal(
            map_by_substring(self.arr,
                             ["abc", "a", "bc", ""],
                             case_sensitive=False, match_beginning=True),
            [0, 1, 2, 3, 3])
        np.testing.assert_equal(
            map_by_substring(self.arr, ["", ""], False, False),
            0)
        self.assertTrue(np.all(np.isnan(
            map_by_substring(self.arr, [], False, False))))

    def test_value_from_string_substring(self):
        trans = ValueFromStringSubstring(StringVariable(), self.patterns)
        arr2 = np.hstack((self.arr.astype(object), [None]))

        with patch('Orange.widgets.data.owcreateclass.map_by_substring') as mbs:
            trans.transform(self.arr)
            a, patterns, case_sensitive, match_beginning = mbs.call_args[0]
            np.testing.assert_equal(a, self.arr)
            self.assertEqual(patterns, self.patterns)
            self.assertFalse(case_sensitive)
            self.assertFalse(match_beginning)

            trans.transform(arr2)
            a, patterns, *_ = mbs.call_args[0]
            np.testing.assert_equal(a,
                                    np.hstack((self.arr.astype(str), "")))

        np.testing.assert_equal(trans.transform(arr2),
                                [0, 1, 2, 0, 3, np.nan])

    def test_value_string_substring_flags(self):
        trans = ValueFromStringSubstring(StringVariable(), self.patterns)
        with patch('Orange.widgets.data.owcreateclass.map_by_substring') as mbs:
            trans.case_sensitive = True
            trans.transform(self.arr)
            case_sensitive, match_beginning = mbs.call_args[0][-2:]
            self.assertTrue(case_sensitive)
            self.assertFalse(match_beginning)

            trans.case_sensitive = False
            trans.match_beginning = True
            trans.transform(self.arr)
            case_sensitive, match_beginning = mbs.call_args[0][-2:]
            self.assertFalse(case_sensitive)
            self.assertTrue(match_beginning)

    def test_value_from_discrete_substring(self):
        trans = ValueFromDiscreteSubstring(
            DiscreteVariable(values=self.arr), self.patterns)
        np.testing.assert_equal(trans.lookup_table, [0, 1, 2, 0, 3])

    def test_value_from_discrete_substring_flags(self):
        trans = ValueFromDiscreteSubstring(
            DiscreteVariable(values=self.arr), self.patterns)
        with patch('Orange.widgets.data.owcreateclass.map_by_substring') as mbs:
            trans.case_sensitive = True
            a, patterns, case_sensitive, match_beginning = mbs.call_args[0]
            np.testing.assert_equal(a, self.arr)
            self.assertEqual(patterns, self.patterns)
            self.assertTrue(case_sensitive)
            self.assertFalse(match_beginning)

            trans.case_sensitive = False
            trans.match_beginning = True
            a, patterns, case_sensitive, match_beginning = mbs.call_args[0]
            np.testing.assert_equal(a, self.arr)
            self.assertEqual(patterns, self.patterns)
            self.assertFalse(case_sensitive)
            self.assertTrue(match_beginning)

            arr2 = self.arr[::-1]
            trans.variable = DiscreteVariable(values=arr2)
            a, patterns, case_sensitive, match_beginning = mbs.call_args[0]
            np.testing.assert_equal(a, arr2)
            self.assertEqual(patterns, self.patterns)
            self.assertFalse(case_sensitive)
            self.assertTrue(match_beginning)

            patt2 = self.patterns[::-1]
            trans.patterns = patt2
            a, patterns, case_sensitive, match_beginning = mbs.call_args[0]
            np.testing.assert_equal(a, arr2)
            self.assertEqual(patterns, patt2)
            self.assertFalse(case_sensitive)
            self.assertTrue(match_beginning)


class TestOWCreateClass(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWCreateClass)
        self.heart = Table("heart_disease")
        self.zoo = Table("zoo")
        self.no_attributes = Table("iris")[:, :4]

    def _test_default_rules(self):
        self.assertEqual(self.widget.active_rules, [["", ""], ["", ""]])
        for i, (label, pattern) in enumerate(self.widget.line_edits):
            self.assertEqual(label.text(), "".format(i + 1))
            self.assertEqual(pattern.text(), "")

    def _set_attr(self, attr):
        attr_combo = self.widget.controls.attribute
        idx = attr_combo.model().indexOf(attr)
        attr_combo.setCurrentIndex(idx)
        attr_combo.activated.emit(idx)

    def _check_counts(self, expected):
        for countrow, expectedrow in zip(self.widget.counts, expected):
            for count, exp in zip(countrow, expectedrow):
                self.assertEqual(count.text(), exp)

    def test_no_data(self):
        widget = self.widget
        self.send_signal(self.widget.Inputs.data, None)
        attr_combo = widget.controls.attribute

        self.assertFalse(attr_combo.model())
        self.assertIsNone(widget.attribute)
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        # That's all. I don't care if line edits retain the content

        widget.apply()
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

    def test_no_useful_data(self):
        widget = self.widget
        self.send_signal(self.widget.Inputs.data, self.no_attributes)
        attr_combo = widget.controls.attribute

        self.assertFalse(attr_combo.model())
        self.assertIsNone(widget.attribute)
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertTrue(widget.Warning.no_nonnumeric_vars.is_shown())

        widget.apply()
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

        self.send_signal(self.widget.Inputs.data, self.heart)
        self.assertFalse(widget.Warning.no_nonnumeric_vars.is_shown())

    def test_string_data(self):
        widget = self.widget
        self.send_signal(self.widget.Inputs.data, self.zoo)
        self._set_attr(self.zoo.domain.metas[0])
        widget.line_edits[0][1].setText("a")

        self._check_counts([["54", ""], ["47", ""]])

        widget.apply()
        outdata = self.get_output(self.widget.Outputs.data)
        classes = outdata.get_column_view("class")[0]
        attr = outdata.get_column_view("name")[0].astype(str)
        has_a = np.char.find(attr, "a") != -1
        np.testing.assert_equal(classes[has_a], 0)
        np.testing.assert_equal(classes[~has_a], 1)

    def _set_thal(self):
        widget = self.widget
        thal = self.heart.domain["thal"]
        self._set_attr(thal)

        widget.line_edits[0][0].setText("Cls1")
        widget.line_edits[1][0].setText("Cls2")
        widget.line_edits[0][1].setText("eversa")
        widget.line_edits[1][1].setText("efect")

    def _check_thal(self):
        widget = self.widget
        self.assertEqual(widget.attribute.name, "thal")
        # This line indeed tests an implementational detail, but the one I
        # got wrong. Until implementation changes, it makes me feel safe.
        self.assertIs(widget.rules["thal"], widget.active_rules)

        self._check_counts([["117", ""], ["18", "+ 117"]])

        widget.apply()
        outdata = self.get_output(self.widget.Outputs.data)
        self.assertEqual(outdata.domain.class_var.values, ["Cls1", "Cls2"])
        classes = outdata.get_column_view("class")[0]
        attr = outdata.get_column_view("thal")[0]
        thal = self.heart.domain["thal"]
        reversable = np.equal(attr, thal.values.index("reversable defect"))
        fixed = np.equal(attr, thal.values.index("fixed defect"))
        np.testing.assert_equal(classes[reversable], 0)
        np.testing.assert_equal(classes[fixed], 1)
        self.assertTrue(np.all(np.isnan(classes[~(reversable | fixed)])))

    def test_flow_and_context_handling(self):
        widget = self.widget
        self.send_signal(self.widget.Inputs.data, self.heart)
        self._test_default_rules()

        widget.apply()
        outdata = self.get_output(self.widget.Outputs.data)
        self.assertEqual(outdata.domain.class_var.values, ["C1"])
        classes = outdata.get_column_view("class")[0]
        np.testing.assert_equal(classes, 0)

        thal = self.heart.domain["thal"]
        self._set_attr(thal)
        self.assertIs(widget.rules["thal"], widget.active_rules)
        self._test_default_rules()

        self._set_thal()
        self._check_thal()

        gender = self.heart.domain["gender"]
        self._set_attr(gender)
        self.assertIs(widget.rules["gender"], widget.active_rules)
        self._test_default_rules()

        widget.line_edits[0][1].setText("ema")
        self._check_counts([["97", ""], ["206", ""]])
        widget.line_edits[1][1].setText("ma")
        self._check_counts([["97", ""], ["206", "+ 97"]])

        widget.apply()
        outdata = self.get_output(self.widget.Outputs.data)
        self.assertEqual(outdata.domain.class_var.values, ["C1", "C2"])
        classes = outdata.get_column_view("class")[0]
        attr = outdata.get_column_view("gender")[0]
        female = np.equal(attr, gender.values.index("female"))
        np.testing.assert_equal(classes[female], 0)
        np.testing.assert_equal(classes[~female], 1)

        self._set_attr(thal)
        self._check_thal()

        prev_rules = widget.rules
        self.send_signal(self.widget.Inputs.data, self.zoo)
        self.assertIsNot(widget.rules, prev_rules)

        self.send_signal(self.widget.Inputs.data, self.heart)
        self._check_thal()

        # Check that sending None as data does not ruin the context, and that
        # the empty context does not match the true one later
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNot(widget.rules, prev_rules)

        self.send_signal(self.widget.Inputs.data, self.heart)
        self._check_thal()

        self.send_signal(self.widget.Inputs.data, self.no_attributes)
        self.assertIsNot(widget.rules, prev_rules)

        self.send_signal(self.widget.Inputs.data, self.heart)
        self._check_thal()

    def test_add_remove_lines(self):
        widget = self.widget
        self.send_signal(self.widget.Inputs.data, self.heart)
        self._set_thal()
        widget.add_row()
        self.assertEqual(len(widget.line_edits), 3)
        widget.line_edits[2][0].setText("Cls3")
        widget.line_edits[2][1].setText("a")
        # Observing counts suffices to deduct that rules are set correctly
        self._check_counts([["117", ""], ["18", "+ 117"], ["166", "+ 117"]])

        widget.add_row()
        widget.add_row()
        widget.line_edits[3][1].setText("c")
        widget.line_edits[4][1].setText("b")
        widget.apply()
        outdata = self.get_output(self.widget.Outputs.data)
        # Ignore classes without labels
        self._check_counts([["117", ""], ["18", "+ 117"], ["166", "+ 117"],
                            ["", ""], ["", ""]])
        self.assertEqual(outdata.domain.class_var.values,
                         ["Cls1", "Cls2", "Cls3"])

        widget.remove_buttons[1].click()
        self._check_counts([["117", ""], ["166", "+ 117"], ["18", "+ 117"],
                            ["", ""], ["", ""]])
        self.assertEqual([lab.text() for _, lab in widget.line_edits],
                         ["eversa", "a", "c", "b"])

        # Removal preserves integrity of connections
        widget.line_edits[1][1].setText("efect")
        widget.line_edits[2][1].setText("a")
        self._check_counts([["117", ""], ["18", "+ 117"], ["166", "+ 117"],
                            ["", ""]])

        while widget.remove_buttons:
            widget.remove_buttons[0].click()
        widget.apply()
        outdata = self.get_output(self.widget.Outputs.data)
        np.testing.assert_equal(self.heart.X, outdata.X)
        self.assertTrue(np.all(np.isnan(outdata.Y)))

    def test_options(self):
        def _transformer_flags():
            widget.apply()
            outdata = self.get_output(self.widget.Outputs.data)
            transformer = outdata.domain.class_var.compute_value
            return transformer.case_sensitive, transformer.match_beginning

        widget = self.widget
        self.send_signal(self.widget.Inputs.data, self.heart)
        self.assertEqual(_transformer_flags(), (False, False))
        widget.controls.case_sensitive.click()
        self.assertEqual(_transformer_flags(), (True, False))
        widget.controls.case_sensitive.click()
        widget.controls.match_beginning.click()
        self.assertEqual(_transformer_flags(), (False, True))

    def test_report(self):
        """Report does not crash"""
        widget = self.widget
        widget.send_report()

        self.send_signal(self.widget.Inputs.data, self.heart)
        thal = self.heart.domain["thal"]
        self._set_attr(thal)
        widget.line_edits[0][0].setText("Cls3")
        widget.line_edits[0][1].setText("a")
        widget.send_report()

        widget.line_edits[1][1].setText("b")
        widget.send_report()

        widget.line_edits[1][1].setText("c")
        widget.send_report()

if __name__ == "__main__":
    unittest.main()
