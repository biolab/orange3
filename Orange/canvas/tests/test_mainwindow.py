# pylint: disable=all
from unittest.mock import patch

from Orange.canvas.mainwindow import MainWindow, OUserSettingsDialog
from Orange.widgets.tests.base import GuiTest


class TestMainWindow(GuiTest):
    def test_settings_dialog(self):
        mw = MainWindow()
        with patch.object(OUserSettingsDialog, "exec", lambda self: 0), \
                patch.object(OUserSettingsDialog, "show", lambda self: None):
            mw.open_canvas_settings()
