import unittest
from unittest.mock import patch, Mock

import requests

from AnyQt.QtCore import QItemSelectionModel

from Orange.widgets.data.owdatasets import OWDataSets
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.state_summary import format_summary_details


class TestOWDataSets(WidgetTest):
    @patch("Orange.widgets.data.owdatasets.OWDataSets.list_remote",
           Mock(side_effect=requests.exceptions.ConnectionError))
    @patch("Orange.widgets.data.owdatasets.OWDataSets.list_local",
           Mock(return_value={}))
    @patch("Orange.widgets.data.owdatasets.log", Mock())
    def test_no_internet_connection(self):
        w = self.create_widget(OWDataSets)  # type: OWDataSets
        self.wait_until_stop_blocking(w)
        self.assertTrue(w.Error.no_remote_datasets.is_shown())

    @patch("Orange.widgets.data.owdatasets.OWDataSets.list_remote",
           Mock(side_effect=requests.exceptions.ConnectionError))
    @patch("Orange.widgets.data.owdatasets.OWDataSets.list_local",
           Mock(return_value={('core', 'foo.tab'): {}}))
    @patch("Orange.widgets.data.owdatasets.log", Mock())
    def test_only_local(self):
        w = self.create_widget(OWDataSets)  # type: OWDataSets
        self.wait_until_stop_blocking(w)
        self.assertTrue(w.Warning.only_local_datasets.is_shown())
        self.assertEqual(w.view.model().rowCount(), 1)

    @patch("Orange.widgets.data.owdatasets.OWDataSets.list_remote",
           Mock(side_effect=requests.exceptions.ConnectionError))
    @patch("Orange.widgets.data.owdatasets.OWDataSets.list_local",
           Mock(return_value={('core', 'foo.tab'): {},
                              ('core', 'bar.tab'): {}}))
    @patch("Orange.widgets.data.owdatasets.log", Mock())
    def test_filtering(self):
        w = self.create_widget(OWDataSets)  # type: OWDataSets
        self.wait_until_stop_blocking(w)
        self.assertEqual(w.view.model().rowCount(), 2)
        w.filterLineEdit.setText("foo")
        self.assertEqual(w.view.model().rowCount(), 1)
        w.filterLineEdit.setText("baz")
        self.assertEqual(w.view.model().rowCount(), 0)
        w.filterLineEdit.setText("")
        self.assertEqual(w.view.model().rowCount(), 2)

    @patch("Orange.widgets.data.owdatasets.OWDataSets.list_remote",
           Mock(return_value={('core', 'iris.tab'): {}}))
    @patch("Orange.widgets.data.owdatasets.OWDataSets.list_local",
           Mock(return_value={}))
    @patch("Orange.widgets.data.owdatasets.ensure_local",
           Mock(return_value="iris.tab"))
    def test_download_iris(self):
        w = self.create_widget(OWDataSets)  # type: OWDataSets
        self.wait_until_stop_blocking(w)
        # select the only dataset
        sel_type = QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows
        w.view.selectionModel().select(w.view.model().index(0, 0), sel_type)
        w.commit()
        iris = self.get_output(w.Outputs.data, w)
        self.assertEqual(len(iris), 150)

    @patch("Orange.widgets.data.owdatasets.OWDataSets.list_remote",
           Mock(side_effect=requests.exceptions.ConnectionError))
    @patch("Orange.widgets.data.owdatasets.OWDataSets.list_local",
           Mock(return_value={('dir1', 'dir2', 'foo.tab'): {},
                              ('bar.tab',): {}}))
    @patch("Orange.widgets.data.owdatasets.log", Mock())
    def test_dir_depth(self):
        w = self.create_widget(OWDataSets)  # type: OWDataSets
        self.wait_until_stop_blocking(w)
        self.assertEqual(w.view.model().rowCount(), 2)

    @patch("Orange.widgets.data.owdatasets.OWDataSets.list_remote",
           Mock(return_value={('core', 'iris.tab'): {}}))
    @patch("Orange.widgets.data.owdatasets.OWDataSets.list_local",
           Mock(return_value={}))
    @patch("Orange.widgets.data.owdatasets.ensure_local",
           Mock(return_value="iris.tab"))
    def test_summary(self):
        """Check if status bar is updated when data is received"""
        widget = self.create_widget(OWDataSets)
        output_sum = widget.info.set_output_summary = Mock()
        self.wait_until_stop_blocking(widget)
        # select the only dataset
        sel_type = QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows
        widget.view.selectionModel().select(widget.view.model().index(0, 0), sel_type)
        widget.commit()
        output = self.get_output(widget.Outputs.data, widget)
        output_sum.assert_called_with(len(output),
                                      format_summary_details(output))

if __name__ == "__main__":
    unittest.main()
