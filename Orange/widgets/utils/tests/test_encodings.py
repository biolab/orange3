import os
from unittest import mock

from AnyQt.QtCore import QSettings

from Orange.widgets.tests.base import GuiTest
from Orange.widgets.utils.encodings import SelectEncodingsWidget


def mock_settings():
    return QSettings(os.devnull, QSettings.IniFormat)


class TestSelectEncodingsWidget(GuiTest):

    @mock.patch("Orange.widgets.utils.encodings.QSettings", mock_settings)
    def test_widget(self):
        w = SelectEncodingsWidget()
        model = w.model()
        w.reset()
        enc = w.selectedEncodings()
        self.assertLess(len(enc), model.rowCount())
        w.selectAll()
        enc = w.selectedEncodings()
        self.assertEqual(len(enc), model.rowCount())
        w.clearAll()
        enc = w.selectedEncodings()
        self.assertEqual(len(enc), 0)
