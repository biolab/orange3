import time
import unittest
from unittest.mock import patch, Mock

import requests

from AnyQt.QtCore import QItemSelectionModel, Qt

from Orange.widgets.data.owdatasets import OWDataSets, Namespace as DSNamespace, \
    GENERAL_DOMAIN, ALL_DOMAINS
from Orange.widgets.tests.base import WidgetTest


class TestOWDataSets(WidgetTest):
    def setUp(self):
        # Most tests check the iniitialization of widget under different
        # conditions, therefore mocks are needed prior to calling createWidget.
        # Inherited methods will set self.widget; here we set it to None to
        # avoid lint errors.
        self.widget = None

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
           Mock(side_effect=requests.exceptions.ConnectionError))
    @patch("Orange.widgets.data.owdatasets.list_local",
           Mock(return_value={('core', 'foo.tab'): {"domain": None},
                              ('edu', 'bar.tab'): {"domain": "edu"}}))
    @patch("Orange.widgets.data.owdatasets.log", Mock())
    def test_filtering_by_domain(self):
        w = self.create_widget(OWDataSets)  # type: OWDataSets
        model = w.view.model()
        model.setDomain(GENERAL_DOMAIN)
        self.wait_until_stop_blocking(w)
        self.assertEqual(model.rowCount(), 1)

        model.setDomain(ALL_DOMAINS)
        self.wait_until_stop_blocking(w)
        self.assertEqual(model.rowCount(), 2)

        model.setDomain("edu")
        self.assertEqual(model.rowCount(), 1)
        self.assertEqual(model.index(0, 0).data(Qt.UserRole).title, "bar.tab")

        model.setDomain("baz")
        self.assertEqual(model.rowCount(), 0)

    @patch("Orange.widgets.data.owdatasets.list_local",
           Mock(return_value={('core', 'foo.tab'): {"domain": None},
                              ('core', 'bar.tab'): {"domain": "edu"}}))
    @patch("Orange.widgets.data.owdatasets.log", Mock())
    @patch("Orange.widgets.data.owdatasets.OWDataSets.commit", Mock())
    def test_change_domain(self):
        def wait_and_return(_):
            time.sleep(0.2)
            return {('core', 'foo.tab'): {"domain": "edu"},
                    ('core', 'bar.tab'): {"domain": "edu"}}
        with patch("Orange.widgets.data.owdatasets.list_remote",
                   new=wait_and_return):
            self.widget = w = self.create_widget(OWDataSets,
                                   stored_settings={"selected_id": "bar.tab",
                                                    "domain": "edu"})
            self.wait_until_stop_blocking()
            self.assertEqual(w.selected_id, "bar.tab")
            self.assertEqual(w.domain_combo.currentText(), "edu")

            self.widget = w = self.create_widget(OWDataSets,
                                   stored_settings={"selected_id": "foo.tab",
                                                    "domain": "(core)"})
            self.wait_until_stop_blocking()
            self.assertEqual(w.selected_id, "foo.tab")
            self.assertEqual(w.domain_combo.currentText(), "edu")

            self.widget = w = self.create_widget(OWDataSets,
                                   stored_settings={"selected_id": "bar.tab",
                                                    "domain": "(core)"})
            self.wait_until_stop_blocking()
            self.assertEqual(w.selected_id, "bar.tab")
            self.assertEqual(w.domain_combo.currentText(), "edu")

    def __titles(self, widget):
        model = widget.view.model()
        return {
            model.index(row, 0).data(Qt.UserRole).title
            for row in range(model.rowCount())}

    @patch("Orange.widgets.data.owdatasets.list_remote",
           Mock(side_effect=requests.exceptions.ConnectionError))
    @patch("Orange.widgets.data.owdatasets.list_local",
           Mock(return_value={
               ('core', 'foo.tab'): {"title": "an unlisted data set",
                                     "publication_status": DSNamespace.UNLISTED},
               ('core', 'bar.tab'): {"title": "a published data set",
                                     "publication_status": DSNamespace.PUBLISHED},
               ('core', 'baz.tab'): {"title": "an unp unp",
                                     "publication_status": DSNamespace.PUBLISHED}
           }))
    @patch("Orange.widgets.data.owdatasets.log", Mock())
    def test_filtering_unlisted(self):
        def titles():
            return self.__titles(w)

        w = self.create_widget(OWDataSets)  # type: OWDataSets
        model = w.view.model()
        self.assertEqual(titles(), {"a published data set", "an unp unp"})

        model.setFilterFixedString("unp")
        self.assertEqual(titles(), {"an unp unp"})

        model.setFilterFixedString("an U")
        self.assertEqual(titles(), {"an unlisted data set", "an unp unp"})

        model.setFilterFixedString("")
        self.assertEqual(titles(), {"a published data set", "an unp unp"})

        model.setFilterFixedString(None)
        self.assertEqual(titles(), {"a published data set", "an unp unp"})

    @patch("Orange.widgets.data.owdatasets.list_remote",
           Mock(return_value={('core', 'foo.tab'): {"title": "Foo data set",
                                                    "language": "English"},
                              ('core', 'bar.tab'): {"title": "Bar data set",
                                                    "domain": "Testing"},
                              ('core', 'bax.tab'): {"title": "Bax data set",
                                                    "language": "Slovenščina"}
                              }))
    @patch("Orange.widgets.data.owdatasets.list_local",
           Mock(return_value={}))
    @patch("Orange.widgets.data.owdatasets.OWDataSets.commit", Mock())
    def test_filter_overrides_language_and_domain(self):
        w = self.create_widget(OWDataSets)  # type: OWDataSets
        self.wait_until_stop_blocking(w)
        w.language_combo.setCurrentText("Slovenščina")
        w.language_combo.activated.emit(w.language_combo.currentIndex())
        w.domain_combo.setCurrentText(w.GENERAL_DOMAIN_LABEL)
        w.domain_combo.activated.emit(w.domain_combo.currentIndex())

        self.assertEqual(self.__titles(w), {"Bax data set"})

        w.filterLineEdit.setText("data ")
        self.assertEqual(self.__titles(w), {"Foo data set",
                                            "Bar data set",
                                            "Bax data set"})
        self.assertEqual(w.language_combo.currentText(), w.ALL_LANGUAGES)
        self.assertFalse(w.language_combo.isEnabled())
        self.assertEqual(w.domain_combo.currentText(), w.ALL_DOMAINS_LABEL)
        self.assertFalse(w.domain_combo.isEnabled())

        w.filterLineEdit.setText("da")
        self.assertEqual(self.__titles(w), {"Bax data set"})
        self.assertEqual(w.language_combo.currentText(), "Slovenščina")
        self.assertTrue(w.language_combo.isEnabled())
        self.assertEqual(w.domain_combo.currentText(), w.GENERAL_DOMAIN_LABEL)
        self.assertTrue(w.domain_combo.isEnabled())


        w.filterLineEdit.setText("bar d")
        self.assertEqual(self.__titles(w), {"Bar data set"})

        w.filterLineEdit.setText("bax d")
        self.assertEqual(self.__titles(w), {"Bax data set"})

        w.language_combo.setCurrentText("English")
        w.language_combo.activated.emit(2)
        self.assertEqual(self.__titles(w), {"Bax data set"})

        settings = w.settingsHandler.pack_data(w)

        w2 = self.create_widget(OWDataSets, stored_settings=settings)
        self.wait_until_stop_blocking(w2)
        self.assertEqual(w2.language_combo.currentText(), "English")
        self.assertEqual(self.__titles(w2), {"Foo data set"})

        w.selected_id = "bax.tab"
        settings = w.settingsHandler.pack_data(w)
        w2 = self.create_widget(OWDataSets, stored_settings=settings)
        self.wait_until_stop_blocking(w2)
        self.assertEqual(w2.language_combo.currentText(), w2.ALL_LANGUAGES)
        self.assertFalse(w2.language_combo.isEnabled())
        self.assertEqual(w2.filterLineEdit.text(), "bax d")
        self.assertEqual(self.__titles(w2), {"Bax data set"})


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
           Mock(return_value={('core', 'foo.tab'): {"language": "English"},
                              ('core', 'bar.tab'): {"language": "Slovenščina"}}))
    @patch("Orange.widgets.data.owdatasets.list_local",
           Mock(return_value={}))
    def test_remember_all_languages(self):
        w = self.create_widget(OWDataSets)  # type: OWDataSets
        self.wait_until_stop_blocking(w)
        w.language_combo.setCurrentText(w.ALL_LANGUAGES)
        w.language_combo.activated.emit(w.language_combo.currentIndex())
        settings = w.settingsHandler.pack_data(w)

        w2 = self.create_widget(OWDataSets, stored_settings=settings)
        self.wait_until_stop_blocking(w2)
        self.assertEqual(w2.language_combo.currentText(), w2.ALL_LANGUAGES)

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
        w.commit()
        self.assertEqual(w.selected_id, "iris.tab")
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
        self.assertEqual(settings["selected_id"], "bar")

        settings = {"selected_id": "dir1/bar"}
        OWDataSets.migrate_settings(settings, 0)
        self.assertEqual(settings["selected_id"], "bar")

        settings = {"selected_id": "bar"}
        OWDataSets.migrate_settings(settings, 0)
        self.assertEqual(settings["selected_id"], "bar")


if __name__ == "__main__":
    unittest.main()
