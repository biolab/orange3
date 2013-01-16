import logging

from PyQt4.QtGui import QTreeView
from PyQt4.QtCore import QSettings

from ...gui import test

from ..settings import UserSettingsDialog, UserSettingsModel
from ...utils.settings import Settings, config_slot


class TestUserSettings(test.QAppTestCase):
    def setUp(self):
        logging.basicConfig()
        test.QAppTestCase.setUp(self)

    def test(self):
        settings = UserSettingsDialog()
        settings.show()

        self.app.exec_()

    def test_settings_model(self):
        store = QSettings(QSettings.IniFormat, QSettings.UserScope,
                          "biolab.si", "Orange Canvas UnitTests")

        defaults = [config_slot("S1", bool, True, "Something"),
                    config_slot("S2", unicode, "I an not a String",
                                "Disregard the string.")]

        settings = Settings(defaults=defaults, store=store)
        model = UserSettingsModel(settings=settings)

        self.assertEqual(model.rowCount(), len(settings))

        view = QTreeView()
        view.setHeaderHidden(False)

        view.setModel(model)

        view.show()
        self.app.exec_()
