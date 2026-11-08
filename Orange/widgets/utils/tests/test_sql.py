import unittest
from unittest.mock import patch, MagicMock

from orangewidget.utils.signals import MultiInput
from Orange.data import Table, Domain
from Orange.data.sql.table import AUTO_DL_LIMIT, SqlTable
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.signals import Input
from Orange.widgets.utils.sql import check_sql_input, check_sql_input_sequence
from Orange.widgets.widget import OWWidget


class TestSQLDecorator(WidgetTest):
    class MockWidget(OWWidget):
        name = "MockWidget"
        keywords = "mockwidget"

        NotCalled = object()

        class Inputs:
            data = Input("Data", Table)
            additional_data = MultiInput("Additional Data", Table)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.called_with = self.NotCalled

        @Inputs.data
        @check_sql_input
        def set_data(self, obj):
            self.called_with = obj

        @Inputs.additional_data
        @check_sql_input_sequence
        def set_additional_data(self, index, obj):
            self.called_with = index, obj

        @Inputs.additional_data.insert
        @check_sql_input_sequence
        def insert_more_data(self, *_):
            pass

        @Inputs.additional_data.remove
        def remove_more_data(self, *_):
            pass

        def pop_called_with(self):
            t = self.called_with
            self.called_with = self.NotCalled
            return t

    def setUp(self):
        self.widget = self.create_widget(self.MockWidget)

    def test_inputs_check_sql(self):
        """Test if check_sql_input is called when data is sent to a widget."""
        d = Table()
        self.send_signal(self.widget.Inputs.data, d)
        self.assertIs(self.widget.pop_called_with(), d)

        a_table = object()
        with patch("Orange.widgets.utils.sql.Table",
                   MagicMock(return_value=a_table)) as table_mock, \
                patch("Orange.widgets.utils.state_summary.format_summary_details"):
            d = SqlTable(None, None, MagicMock())
            d.domain = Domain([])

            with patch.object(SqlTable, "__len__",
                              return_value=AUTO_DL_LIMIT - 1):
                self.send_signal(self.widget.Inputs.data, d)
                table_mock.assert_called_once_with(d)
                self.assertIs(self.widget.pop_called_with(), a_table)
                table_mock.reset_mock()

            with patch.object(SqlTable, "__len__",
                              return_value=AUTO_DL_LIMIT + 1):
                self.send_signal(self.widget.Inputs.data, d)
                table_mock.assert_not_called()
                self.assertIs(self.widget.pop_called_with(), None)
                self.assertTrue(self.widget.Error.download_sql_data.is_shown())
                table_mock.reset_mock()

            self.send_signal(self.widget.Inputs.data, None)
            table_mock.assert_not_called()
            self.assertIs(self.widget.pop_called_with(), None)
            self.assertFalse(self.widget.Error.download_sql_data.is_shown())

    def test_check_sql_input_sequence(self):
        """Test if check_sql_input_sequence is called when data is sent to a widget."""
        d = Table()
        self.send_signal(self.widget.Inputs.additional_data, d)

        a_table = object()
        with patch("Orange.widgets.utils.sql.Table",
                   MagicMock(return_value=a_table)) as table_mock, \
                patch("Orange.widgets.utils.state_summary.format_summary_details"):
            d = SqlTable(None, None, MagicMock())
            d.domain = Domain([])

            with patch.object(SqlTable, "__len__",
                              return_value=AUTO_DL_LIMIT - 1):
                self.send_signal(self.widget.Inputs.additional_data, d)
                table_mock.assert_called_once_with(d)
                index, obj = self.widget.pop_called_with()
                self.assertIs(index, 0)
                self.assertIs(obj, a_table)
                table_mock.reset_mock()

            with patch.object(SqlTable, "__len__",
                              return_value=AUTO_DL_LIMIT + 1):
                self.send_signal(self.widget.Inputs.additional_data, d)
                table_mock.assert_not_called()
                index, obj = self.widget.pop_called_with()
                self.assertIs(index, 0)
                self.assertIs(obj, None)
                self.assertTrue(self.widget.Error.download_sql_data.is_shown())
                table_mock.reset_mock()

            self.send_signal(self.widget.Inputs.additional_data, None)
            table_mock.assert_not_called()
            index, obj = self.widget.pop_called_with()
            self.assertIs(index, 0)
            self.assertIs(obj, None)
            self.assertFalse(self.widget.Error.download_sql_data.is_shown())


if __name__ == "__main__":
    unittest.main()
