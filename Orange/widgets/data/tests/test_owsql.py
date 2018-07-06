# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

from Orange.data import Table
from Orange.widgets.data.owsql import OWSql
from Orange.widgets.tests.base import WidgetTest
from Orange.tests.sql.base import create_iris, parse_uri, sql_test


@sql_test
class TestOWSql(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSql)
        params, _ = create_iris()
        self.params = parse_uri(params)
        self.iris = Table("iris")

    def test_connection(self):
        """Test if a connection to the database can be established"""
        self.set_connection_params()
        self.widget.connect()

        self.assertFalse(self.widget.Error.connection.is_shown())
        self.assertIsNotNone(self.widget.database_desc)
        tables = ["Select a table", "Custom SQL"]
        self.assertTrue(set(self.widget.tables).issuperset(set(tables)))

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

    def test_missing_extension(self):
        """Test for correctly handled missing backend extension"""
        self.set_connection_params()
        self.widget.backends[0] = unittest.mock.Mock()
        self.widget.backends[0]().missing_extension = "missing extension"
        self.widget.backends[0]().list_tables = lambda x: []
        self.widget.connect()

        self.assertTrue(self.widget.Warning.missing_extension.is_shown())
        self.assertTrue(self.widget.download)
        self.assertFalse(self.widget.downloadcb.isEnabled())

    def set_connection_params(self):
        """Set database connection parameters on widget"""
        port = ''
        if self.params['port'] is not None:
            port += ':' + str(self.params['port'])
        self.widget.servertext.setText(self.params['host'] + port)
        self.widget.databasetext.setText(self.params['database'])
        self.widget.usernametext.setText(self.params['user'])
        self.widget.passwordtext.setText(self.params['password'])


if __name__ == "__main__":
    unittest.main()
