from unittest.mock import patch, Mock

import requests

from Orange.widgets.data.owdatasets import OWDataSets
from Orange.widgets.tests.base import WidgetTest


class TestOWDataSets(WidgetTest):
    @patch("Orange.widgets.data.owdatasets.list_remote",
           Mock(side_effect=requests.exceptions.ConnectionError))
    @patch("Orange.widgets.data.owdatasets.list_local", Mock(return_value={}))
    @patch("Orange.widgets.data.owdatasets.log", Mock())
    def test_no_internet_connection(self):
        w = self.create_widget(OWDataSets)  # type: OWDataSets
        self.assertTrue(w.Error.no_remote_datasets.is_shown())

    @patch("Orange.widgets.data.owdatasets.list_remote",
           Mock(side_effect=requests.exceptions.ConnectionError))
    @patch("Orange.widgets.data.owdatasets.list_local",
           Mock(return_value={('core', 'foo.tab'): {}}))
    @patch("Orange.widgets.data.owdatasets.log", Mock())
    def test_only_local(self):
        w = self.create_widget(OWDataSets)  # type: OWDataSets
        self.assertTrue(w.Warning.only_local_datasets.is_shown())
        self.assertEqual(w.view.model().rowCount(), 1)

    @patch("Orange.widgets.data.owdatasets.list_remote",
           Mock(side_effect=requests.exceptions.ConnectionError))
    @patch("Orange.widgets.data.owdatasets.list_local",
           Mock(return_value={('core', 'foo.tab'): {},
                              ('core', 'bar.tab'): {}}))
    @patch("Orange.widgets.data.owdatasets.log", Mock())
    def test_filtering(self):
        w = self.create_widget(OWDataSets)  # type: OWDataSets
        self.assertEqual(w.view.model().rowCount(), 2)
        w.filterLineEdit.setText("foo")
        self.assertEqual(w.view.model().rowCount(), 1)
        w.filterLineEdit.setText("baz")
        self.assertEqual(w.view.model().rowCount(), 0)
        w.filterLineEdit.setText("")
        self.assertEqual(w.view.model().rowCount(), 2)
