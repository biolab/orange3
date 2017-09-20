from unittest.mock import patch, Mock

import requests

from Orange.widgets.data.owdatasets import OWDataSets
from Orange.widgets.tests.base import WidgetTest


class TestOWDataSets(WidgetTest):
    @patch("Orange.widgets.data.owdatasets.list_remote",
           Mock(side_effect=requests.exceptions.ConnectionError))
    @patch("Orange.widgets.data.owdatasets.log", Mock())
    def test_works_without_internet_connection(self):
        w = self.create_widget(OWDataSets)  # type: OWDataSets
        self.assertTrue(w.Error.no_remote_datasets.is_shown())
