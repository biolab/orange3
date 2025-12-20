# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
from unittest import mock

from Orange.data import Table
from Orange.widgets.data.owsql import OWSql
from Orange.widgets.tests.base import WidgetTest, simulate
from Orange.tests.sql.base import DataBaseTest as dbt

mock_msgbox = mock.MagicMock()
mock_msgbox().addButton.return_value = "NO"
mock_msgbox().clickedButton.return_value = "NO"


def mock_sqltable(*args, **_):
    table = Table(args[1])
    table.get_domain = lambda **_: table.domain
    table.download_data = lambda *_: 1
    return table


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
        tables = ["Select a table"]
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
        backend().n_tables.return_value = 3
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

    @mock.patch('Orange.widgets.data.owsql.Backend')
    def test_data_source(self, mocked_backends: mock.Mock):
        widget: OWSql = self.create_widget(OWSql)
        widget.controls.data_source.buttons[OWSql.CUSTOM_SQL].click()

        backend = mock.Mock()
        backend().display_name = "Dummy Backend"
        backend().list_tables.return_value = ["a", "b", "c"]
        backend().n_tables.return_value = 3
        mocked_backends.available_backends.return_value = [backend]

        settings = {"selected_backend": "Dummy Backend",
                    "host": "host", "port": "port", "database": "DB",
                    "schema": "", "username": "username",
                    "password": "password"}
        widget: OWSql = self.create_widget(OWSql, stored_settings=settings)
        self.assertEqual(widget.tablecombo.currentText(), "Select a table")
        self.assertFalse(widget.tablecombo.isHidden())
        self.assertTrue(widget.tabletext.isHidden())
        self.assertTrue(widget.custom_sql.isHidden())

        widget.controls.data_source.buttons[OWSql.CUSTOM_SQL].click()
        self.assertEqual(widget.tablecombo.currentText(), "Select a table")
        self.assertFalse(widget.tablecombo.isHidden())
        self.assertTrue(widget.tabletext.isHidden())
        self.assertFalse(widget.custom_sql.isHidden())

        widget.controls.data_source.buttons[OWSql.TABLE].click()
        self.assertEqual(widget.tablecombo.currentText(), "Select a table")
        self.assertFalse(widget.tablecombo.isHidden())
        self.assertTrue(widget.tabletext.isHidden())
        self.assertTrue(widget.custom_sql.isHidden())

    @mock.patch('Orange.widgets.data.owsql.MAX_TABLES', 2)
    @mock.patch('Orange.widgets.data.owsql.SqlTable',
                mock.Mock(side_effect=mock_sqltable))
    @mock.patch('Orange.widgets.data.owsql.Backend')
    def test_table_text(self, mocked_backends: mock.Mock):
        backend = mock.Mock()
        backend().display_name = "Dummy Backend"
        backend().list_tables.return_value = ["iris", "zoo", "titanic"]
        backend().n_tables.return_value = 3
        mocked_backends.available_backends.return_value = [backend]

        settings = {"selected_backend": "Dummy Backend",
                    "host": "host", "port": "port", "database": "DB",
                    "schema": "", "username": "username",
                    "password": "password"}
        widget: OWSql = self.create_widget(OWSql, stored_settings=settings)
        self.assertTrue(widget.tablecombo.isHidden())
        self.assertFalse(widget.tabletext.isHidden())
        widget.tabletext.setText("zoo")
        widget.select_table()
        output = self.get_output(widget.Outputs.data, widget=widget)
        self.assertIsInstance(output, Table)
        self.assertEqual(len(output), 101)

    @mock.patch('Orange.widgets.data.owsql.AUTO_DL_LIMIT', 120)
    @mock.patch('Orange.widgets.data.owsql.is_postgres',
                mock.Mock(return_value=True))
    @mock.patch('Orange.widgets.data.owsql.QMessageBox', mock_msgbox)
    @mock.patch('Orange.widgets.data.owsql.SqlTable',
                mock.Mock(side_effect=mock_sqltable))
    @mock.patch('Orange.widgets.data.owsql.Backend')
    def test_auto_dl_limit(self, mocked_backends: mock.Mock):
        backend = mock.Mock()
        backend().display_name = "Dummy Backend"
        backend().list_tables.return_value = ["iris", "zoo", "titanic"]
        backend().n_tables.return_value = 3
        mocked_backends.available_backends.return_value = [backend]

        settings = {"selected_backend": "Dummy Backend",
                    "host": "host", "port": "port", "database": "DB",
                    "schema": "", "username": "username",
                    "password": "password"}
        widget: OWSql = self.create_widget(OWSql, stored_settings=settings)
        widget.tablecombo.setCurrentIndex(2)
        widget.select_table()
        output = self.get_output(widget.Outputs.data, widget=widget)
        self.assertIsInstance(output, Table)
        self.assertEqual(len(output), 101)

        widget.tablecombo.setCurrentIndex(1)
        widget.select_table()
        output = self.get_output(widget.Outputs.data, widget=widget)
        self.assertIsNone(output)


if __name__ == "__main__":
    unittest.main()
