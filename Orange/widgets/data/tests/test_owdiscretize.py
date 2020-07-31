# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,unsubscriptable-object
from unittest.mock import Mock

from Orange.data import Table
from Orange.widgets.data.owdiscretize import OWDiscretize, Default, EqualFreq, \
    Remove, Leave, Custom
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.utils.itemmodels import select_row


class TestOWDiscretize(WidgetTest):
    def setUp(self):
        super().setUp()
        self.widget = self.create_widget(OWDiscretize)

    def test_empty_data(self):
        """No crash on empty data"""
        data = Table("iris")
        widget = self.widget
        widget.default_method = 3
        self.send_signal(self.widget.Inputs.data,
                         Table.from_domain(data.domain))
        widget.unconditional_commit()

    def test_summary(self):
        """Check if status bar is updated when data is received"""
        input_sum = self.widget.info.set_input_summary = Mock()
        output_sum = self.widget.info.set_output_summary = Mock()

        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)
        input_sum.assert_called_with(len(data), format_summary_details(data))
        output = self.get_output(self.widget.Outputs.data)
        output_sum.assert_called_with(len(output),
                                      format_summary_details(output))

        input_sum.reset_mock()
        output_sum.reset_mock()
        self.send_signal(self.widget.Inputs.data, None)
        input_sum.assert_called_once()
        self.assertEqual(input_sum.call_args[0][0].brief, "")
        output_sum.assert_called_once()
        self.assertEqual(output_sum.call_args[0][0].brief, "")

    def test_select_method(self):
        widget = self.widget
        data = Table("iris")[::5]
        self.send_signal(self.widget.Inputs.data, data)

        model = widget.varmodel
        view = widget.varview
        defbg = widget.default_button_group
        varbg = widget.variable_button_group
        self.assertSequenceEqual(list(model), data.domain.attributes)
        defbg.button(OWDiscretize.EqualFreq).click()
        self.assertEqual(widget.default_method, OWDiscretize.EqualFreq)
        self.assertTrue(
            all(isinstance(m, Default) and isinstance(m.method, EqualFreq)
                for m in map(widget.method_for_index,
                             range(len(data.domain.attributes)))))

        # change method for first variable
        select_row(view, 0)
        varbg.button(OWDiscretize.Remove).click()
        met = widget.method_for_index(0)
        self.assertIsInstance(met, Remove)

        # select a second var
        selmodel = view.selectionModel()
        selmodel.select(model.index(2), selmodel.Select)
        # the current checked button must unset
        self.assertEqual(varbg.checkedId(), -1)

        varbg.button(OWDiscretize.Leave).click()
        self.assertIsInstance(widget.method_for_index(0), Leave)
        self.assertIsInstance(widget.method_for_index(2), Leave)
        # reset both back to default
        varbg.button(OWDiscretize.Default).click()
        self.assertIsInstance(widget.method_for_index(0), Default)
        self.assertIsInstance(widget.method_for_index(2), Default)
