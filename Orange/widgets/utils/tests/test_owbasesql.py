# pylint: disable=missing-docstring, protected-access
import unittest
from unittest.mock import Mock
from collections import OrderedDict
from types import SimpleNamespace

from Orange.data import Table
from Orange.data.sql.backend import Backend
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.owbasesql import OWBaseSql
from Orange.data.sql.backend.base import BackendError


USERNAME = "UN"
PASSWORD = "PASS"


class BrokenBackend(Backend):  # pylint: disable=abstract-method
    def __init__(self, connection_params):
        super().__init__(connection_params)
        raise BackendError("Error connecting to DB.")


class TestableSqlWidget(OWBaseSql):
    name = "SQL"

    def __init__(self):
        self.mocked_backend = Mock()
        super().__init__()

    def get_backend(self):
        return self.mocked_backend

    def get_table(self) -> Table:
        return Table("iris")

    @staticmethod
    def _credential_manager(_, __):  # pylint: disable=arguments-differ
        return SimpleNamespace(username=USERNAME, password=PASSWORD)


class TestOWBaseSql(WidgetTest):
    def setUp(self):
        self.host, self.port, self.db = "host", "port", "DB"
        settings = {"host": self.host, "port": self.port,
                    "database": self.db, "schema": ""}
        self.widget = self.create_widget(TestableSqlWidget,
                                         stored_settings=settings)

    def test_connect(self):
        self.widget.mocked_backend.assert_called_once_with(
            {"host": "host", "port": "port", "database": self.db,
             "user": USERNAME, "password": PASSWORD})
        self.assertDictEqual(
            self.widget.database_desc,
            OrderedDict((("Host", "host"), ("Port", "port"),
                         ("Database", self.db), ("User name", USERNAME))))

    def test_connection_error(self):
        self.widget.get_backend = Mock(return_value=BrokenBackend)
        self.widget.connectbutton.click()
        self.assertTrue(self.widget.Error.connection.is_shown())
        self.assertIsNone(self.widget.database_desc)

    def test_output(self):
        self.widget.open_table()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.data))
        self.assertIsNotNone(self.widget.data_desc_table)

    def test_output_error(self):
        self.widget.get_table = lambda: None
        self.widget.open_table()
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertIsNone(self.widget.data_desc_table)

    def test_missing_database_parameter(self):
        self.widget.open_table()
        self.widget.databasetext.setText("")
        self.widget.mocked_backend.reset_mock()
        self.widget.connectbutton.click()
        self.widget.mocked_backend.assert_not_called()
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertIsNone(self.widget.data_desc_table)
        self.assertFalse(self.widget.Error.connection.is_shown())

    def test_report(self):
        self.widget.report_button.click()  # DB connection
        self.widget.open_table()
        self.widget.report_button.click()  # table
        self.widget.databasetext.setText("")
        self.widget.connectbutton.click()
        self.widget.report_button.click()  # empty


if __name__ == "__main__":
    unittest.main()
