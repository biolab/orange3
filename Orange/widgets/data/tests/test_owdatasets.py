import unittest
from unittest.mock import patch, Mock

import requests

from AnyQt.QtCore import QItemSelectionModel, Qt

from Orange.widgets.data.owdatasets import OWDataSets
from Orange.widgets.tests.base import WidgetTest


class TestOWDataSets(WidgetTest):
    @patch("Orange.widgets.data.owdatasets.list_remote",
           Mock(side_effect=requests.exceptions.ConnectionError))
    @patch("Orange.widgets.data.owdatasets.list_local",
           Mock(return_value={}))
    @patch("Orange.widgets.data.owdatasets.log", Mock())
    def test_no_internet_connection(self):
        w = self.create_widget(OWDataSets)  # type: OWDataSets
        self.wait_until_stop_blocking(w)
        self.assertTrue(w.Error.no_remote_datasets.is_shown())

    @patch("Orange.widgets.data.owdatasets.list_remote",
           Mock(side_effect=requests.exceptions.ConnectionError))
    @patch("Orange.widgets.data.owdatasets.list_local",
           Mock(return_value={('core', 'foo.tab'): {}}))
    @patch("Orange.widgets.data.owdatasets.log", Mock())
    @WidgetTest.skipNonEnglish
    def test_only_local(self):
        w = self.create_widget(OWDataSets)  # type: OWDataSets
        self.wait_until_stop_blocking(w)
        self.assertTrue(w.Warning.only_local_datasets.is_shown())
        self.assertEqual(w.view.model().rowCount(), 1)

    @patch("Orange.widgets.data.owdatasets.list_remote",
           Mock(side_effect=requests.exceptions.ConnectionError))
    @patch("Orange.widgets.data.owdatasets.list_local",
           Mock(return_value={('core', 'foo.tab'): {"language": "English"},
                              ('core', 'bar.tab'): {"language": "Slovenščina"}}))
    @patch("Orange.widgets.data.owdatasets.log", Mock())
    def test_filtering(self):
        w = self.create_widget(OWDataSets)  # type: OWDataSets
        model = w.view.model()
        model.setLanguage(None)
        self.wait_until_stop_blocking(w)
        self.assertEqual(model.rowCount(), 2)
        w.filterLineEdit.setText("foo")
        self.assertEqual(model.rowCount(), 1)
        w.filterLineEdit.setText("baz")
        self.assertEqual(model.rowCount(), 0)
        w.filterLineEdit.setText("")
        self.assertEqual(model.rowCount(), 2)

        model.setLanguage("Slovenščina")
        self.assertEqual(model.rowCount(), 1)
        self.assertEqual(model.index(0, 0).data(Qt.UserRole).title, "bar.tab")

        model.setLanguage("English")
        self.assertEqual(model.rowCount(), 1)
        self.assertEqual(model.index(0, 0).data(Qt.UserRole).title, "foo.tab")

        model.setLanguage(None)
        self.assertEqual(model.rowCount(), 2)

    @patch("Orange.widgets.data.owdatasets.list_remote",
           Mock(return_value={('core', 'foo.tab'): {"language": "English"},
                              ('core', 'bar.tab'): {"language": "Slovenščina"}}))
    @patch("Orange.widgets.data.owdatasets.list_local",
           Mock(return_value={}))
    def test_remember_language(self):
        w = self.create_widget(OWDataSets)  # type: OWDataSets
        self.wait_until_stop_blocking(w)
        w.language_combo.setCurrentText("Slovenščina")
        w.language_combo.activated.emit(w.language_combo.currentIndex())
        settings = w.settingsHandler.pack_data(w)

        w2 = self.create_widget(OWDataSets, stored_settings=settings)
        self.wait_until_stop_blocking(w2)
        self.assertEqual(w2.language_combo.currentText(), "Slovenščina")

        settings["language"] = "Klingon"
        w2 = self.create_widget(OWDataSets, stored_settings=settings)
        self.wait_until_stop_blocking(w2)
        self.assertEqual(w2.language_combo.currentText(), "Klingon")

    @patch("Orange.widgets.data.owdatasets.list_remote",
           Mock(return_value={('core', 'iris.tab'): {}}))
    @patch("Orange.widgets.data.owdatasets.list_local",
           Mock(return_value={}))
    @patch("Orange.widgets.data.owdatasets.ensure_local",
           Mock(return_value="iris.tab"))
    @WidgetTest.skipNonEnglish
    def test_download_iris(self):
        w = self.create_widget(OWDataSets)  # type: OWDataSets
        self.wait_until_stop_blocking(w)
        # select the only dataset
        sel_type = QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows
        w.view.selectionModel().select(w.view.model().index(0, 0), sel_type)
        self.assertEqual(w.selected_id, "core/iris.tab")
        w.commit()
        iris = self.get_output(w.Outputs.data, w)
        self.assertEqual(len(iris), 150)

    @patch("Orange.widgets.data.owdatasets.list_remote",
           Mock(return_value={('dir1', 'dir2', 'foo.tab'): {}}))
    @patch("Orange.widgets.data.owdatasets.list_local",
           Mock(return_value={}))
    @patch("Orange.widgets.data.owdatasets.ensure_local",
           Mock(return_value="iris.tab"))
    @WidgetTest.skipNonEnglish
    def test_download_multidir(self):
        w = self.create_widget(OWDataSets)  # type: OWDataSets
        self.wait_until_stop_blocking(w)
        # select the only dataset
        sel_type = QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows
        w.view.selectionModel().select(w.view.model().index(0, 0), sel_type)
        self.assertEqual(w.selected_id, "dir1/dir2/foo.tab")
        w.commit()
        iris = self.get_output(w.Outputs.data, w)
        self.assertEqual(len(iris), 150)

    @patch("Orange.widgets.data.owdatasets.list_remote",
           Mock(side_effect=requests.exceptions.ConnectionError))
    @patch("Orange.widgets.data.owdatasets.list_local",
           Mock(return_value={('dir1', 'dir2', 'foo.tab'): {},
                              ('bar.tab',): {}}))
    @patch("Orange.widgets.data.owdatasets.log", Mock())
    @WidgetTest.skipNonEnglish
    def test_dir_depth(self):
        w = self.create_widget(OWDataSets)  # type: OWDataSets
        self.wait_until_stop_blocking(w)
        self.assertEqual(w.view.model().rowCount(), 2)

    def test_migrate_selected_id(self):
        settings = {}
        OWDataSets.migrate_settings(settings, 0)
        self.assertNotIn("selected_id", settings)

        settings = {"selected_id": None}
        OWDataSets.migrate_settings(settings, 0)
        self.assertEqual(settings["selected_id"], None)

        settings = {"selected_id": "dir1\\bar"}
        OWDataSets.migrate_settings(settings, 0)
        self.assertEqual(settings["selected_id"], "dir1/bar")

        settings = {"selected_id": "dir1/bar"}
        OWDataSets.migrate_settings(settings, 0)
        self.assertEqual(settings["selected_id"], "dir1/bar")


if __name__ == "__main__":
    unittest.main()
