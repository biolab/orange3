# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
from unittest import mock

from Orange.data import Table
from Orange.widgets.data.owsql import OWSql
from Orange.widgets.tests.base import WidgetTest, simulate
from Orange.tests.sql.base import DataBaseTest as dbt


class TestOWSqlConnected(WidgetTest, dbt):
    def setUpDB(self):
        # pylint: disable=attribute-defined-outside-init
        self.widget = self.create_widget(OWSql)
        self.params, _ = self.create_iris_sql_table()
        self.iris = Table("iris")

    def tearDownDB(self):
        self.drop_iris_sql_table()

    @dbt.run_on(["postgres"])
    def test_connection(self):
        """Test if a connection to the database can be established"""
        self.set_connection_params()
        self.widget.connect()

        self.assertFalse(self.widget.Error.connection.is_shown())
        self.assertIsNotNone(self.widget.database_desc)
        tables = ["Select a table", "Custom SQL"]
        self.assertTrue(set(self.widget.tables).issuperset(set(tables)))

    @dbt.run_on(["postgres"])
    def test_output_iris(self):
        """Test if iris data can be fetched from database"""
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

        self.set_connection_params()
        self.widget.connect()

        idx = list(map(str, self.widget.tables)).index("iris")
        self.widget.tablecombo.setCurrentIndex(idx)
        self.widget.select_table()

        output = self.get_output(self.widget.Outputs.data)
        self.assertIsNotNone(output)
        self.assertEqual(len(output), len(self.iris))
        iris_domain = set(map(str, self.iris.domain.attributes))
        output_domain = set(map(str, output.domain.attributes))
        self.assertTrue(output_domain.issuperset(iris_domain))

    def set_connection_params(self):
        """Set database connection parameters on widget"""
        port = ''
        if self.params['port'] is not None:
            port += ':' + str(self.params['port'])
        self.widget.servertext.setText(self.params['host'] + port)
        self.widget.databasetext.setText(self.params['database'])
        self.widget.usernametext.setText(self.params['user'])
        self.widget.passwordtext.setText(self.params['password'])


class TestOWSql(WidgetTest):

    @mock.patch('Orange.widgets.data.owsql.Backend')
    def test_missing_extension(self, mock_backends):
        """Test for correctly handled missing backend extension"""
        backend = mock.Mock()
        backend().display_name = "PostgreSQL"
        backend().missing_extension = ["missing extension"]
        backend().list_tables.return_value = []
        mock_backends.available_backends.return_value = [backend]

        settings = {"host": "host", "port": "port",
                    "database": "DB", "schema": "",
                    "username": "username", "password": "password"}
        widget = self.create_widget(OWSql, stored_settings=settings)

        self.assertTrue(widget.Warning.missing_extension.is_shown())
        self.assertTrue(widget.download)
        self.assertFalse(widget.downloadcb.isEnabled())

    @mock.patch('Orange.widgets.data.owsql.Backend')
    def test_non_postgres(self, mock_backends):
        """Test if download is enforced for non postgres backends"""
        backend = mock.Mock()
        backend().display_name = "database"
        del backend().missing_extension
        backend().list_tables.return_value = []
        mock_backends.available_backends.return_value = [backend]

        settings = {"host": "host", "port": "port",
                    "database": "DB", "schema": "",
                    "username": "username", "password": "password"}
        widget = self.create_widget(OWSql, stored_settings=settings)

        self.assertTrue(widget.download)
        self.assertFalse(widget.downloadcb.isEnabled())

    @mock.patch('Orange.widgets.data.owsql.Table',
                mock.PropertyMock(return_value=Table('iris')))
    @mock.patch('Orange.widgets.data.owsql.SqlTable')
    @mock.patch('Orange.widgets.data.owsql.Backend')
    def test_restore_table(self, mock_backends, mock_sqltable):
        """Test if selected table is restored from settings"""
        backend = mock.Mock()
        backend().display_name = "database"
        del backend().missing_extension
        backend().list_tables.return_value = ["a", "b", "c"]
        mock_backends.available_backends.return_value = [backend]
        mock_sqltable().approx_len.return_value = 100

        settings = {"host": "host", "port": "port", "database": "DB",
                    "schema": "", "username": "username",
                    "password": "password", "table": "b"}
        widget = self.create_widget(OWSql, stored_settings=settings)
        self.assertEqual(widget.tablecombo.currentText(), "b")

    @mock.patch("Orange.data.sql.backend.base.Backend.available_backends")
    def test_selected_backend(self, mocked_backends: mock.Mock):
        b1, b2 = mock.Mock(), mock.Mock()
        b1.display_name = "B1"
        b2.display_name = "B2"
        mocked_backends.return_value = [b1, b2]

        widget = self.create_widget(OWSql)
        self.assertEqual(widget.backendcombo.currentText(), "B1")

        simulate.combobox_activate_index(widget.backendcombo, 1)
        self.assertEqual(widget.backendcombo.currentText(), "B2")

        settings = widget.settingsHandler.pack_data(widget)
        widget = self.create_widget(OWSql, stored_settings=settings)
        self.assertEqual(widget.backendcombo.currentText(), "B2")

        settings = widget.settingsHandler.pack_data(widget)
        settings["selected_backend"] = "B3"
        widget = self.create_widget(OWSql, stored_settings=settings)
        self.assertEqual(widget.backendcombo.currentText(), "B1")

        mocked_backends.return_value = []
        settings = widget.settingsHandler.pack_data(widget)
        widget = self.create_widget(OWSql, stored_settings=settings)
        self.assertEqual(widget.backendcombo.currentText(), "")


if __name__ == "__main__":
    unittest.main()
