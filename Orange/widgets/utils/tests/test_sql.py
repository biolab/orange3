import unittest
from unittest.mock import patch, MagicMock

from Orange.data import Table
from Orange.data.sql.table import AUTO_DL_LIMIT, SqlTable
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.signals import Input
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import OWWidget


class TestSQLDecorator(WidgetTest):
    class MockWidget(OWWidget):
        name = "MockWidget"

        NotCalled = object()

        class Inputs:
            data = Input("Data", Table)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.called_with = self.NotCalled

        @Inputs.data
        @check_sql_input
        def set_data(self, obj):
            self.called_with = obj

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
                   MagicMock(return_value=a_table)) as table_mock:
            d = SqlTable(None, None, MagicMock())

            d.approx_len = MagicMock(return_value=AUTO_DL_LIMIT - 1)
            self.send_signal(self.widget.Inputs.data, d)
            table_mock.assert_called_once_with(d)
            self.assertIs(self.widget.pop_called_with(), a_table)
            table_mock.reset_mock()

            d.approx_len = MagicMock(return_value=AUTO_DL_LIMIT + 1)
            self.send_signal(self.widget.Inputs.data, d)
            table_mock.assert_not_called()
            self.assertIs(self.widget.pop_called_with(), None)
            self.assertTrue(self.widget.Error.download_sql_data.is_shown())
            table_mock.reset_mock()

            self.send_signal(self.widget.Inputs.data, None)
            table_mock.assert_not_called()
            self.assertIs(self.widget.pop_called_with(), None)
            self.assertFalse(self.widget.Error.download_sql_data.is_shown())


if __name__ == "__main__":
    unittest.main()
