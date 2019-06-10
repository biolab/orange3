import unittest.mock

from AnyQt.QtCore import Qt, QEvent
from AnyQt.QtTest import QTest
from AnyQt.QtWidgets import QWidget, QApplication

from Orange.canvas.utils.overlay import NotificationWidget, NotificationOverlay
from Orange.widgets.tests.base import GuiTest


class TestOverlay(GuiTest):
    def setUp(self) -> None:
        self.container = QWidget()
        stdb = NotificationWidget.Ok | NotificationWidget.Close
        self.overlay = NotificationOverlay(self.container)
        self.notif = NotificationWidget(title="lol",
                                        text="hihi",
                                        standardButtons=stdb)
        self.container.show()
        QTest.qWaitForWindowExposed(self.container)
        self.overlay.show()
        QTest.qWaitForWindowExposed(self.overlay)

    def tearDown(self) -> None:
        NotificationOverlay.overlayInstances = []
        NotificationOverlay.notifQueue = []
        self.container = None
        self.overlay = None
        self.notif = None

    def test_notification_dismiss(self):
        mock = unittest.mock.MagicMock()
        self.notif.clicked.connect(mock)
        NotificationOverlay.registerNotification(self.notif)
        QTest.mouseClick(self.notif._dismiss_button, Qt.LeftButton)
        mock.assert_called_once_with(self.notif._dismiss_button)

    def test_notification_message(self):
        self.notif.setText("Hello world! It's so nice here")
        QApplication.sendPostedEvents(self.notif, QEvent.LayoutRequest)
        self.assertTrue(self.notif.geometry().isValid())

        button_ok = self.notif.button(NotificationWidget.Ok)
        button_close = self.notif.button(NotificationWidget.Close)

        self.assertTrue(all([button_ok, button_close]))
        self.assertIs(self.notif.button(NotificationWidget.Ok), button_ok)
        self.assertIs(self.notif.button(NotificationWidget.Close), button_close)

        button = self.notif.button(NotificationWidget.Ok)
        self.assertIsNot(button, None)
        self.assertTrue(self.notif.buttonRole(button),
                        NotificationWidget.AcceptRole)

        mock = unittest.mock.MagicMock()
        self.notif.accepted.connect(mock)

        NotificationOverlay.registerNotification(self.notif)

        cloned = NotificationOverlay.overlayInstances[0].currentWidget()
        self.assertTrue(cloned.isVisible())
        button = cloned._msgwidget.button(NotificationWidget.Ok)
        QTest.mouseClick(button, Qt.LeftButton)
        self.assertFalse(cloned.isVisible())

        mock.assert_called_once()

    def test_queued_notifications(self):
        surveyDialogButtons = NotificationWidget.Ok | NotificationWidget.Close
        surveyDialog = NotificationWidget(title="Survey",
                                          text="We want to understand our users better.\n"
                                               "Would you like to take a short survey?",
                                          standardButtons=surveyDialogButtons)

        def handle_survey_response(b):
            self.assertEquals(self.notif.buttonRole(b), NotificationWidget.DismissRole)

        self.notif.clicked.connect(handle_survey_response)

        NotificationOverlay.registerNotification(self.notif)
        notif1 = NotificationOverlay.overlayInstances[0]._widgets[0]
        button = notif1._dismiss_button

        NotificationOverlay.registerNotification(surveyDialog)
        notif2 = NotificationOverlay.overlayInstances[0]._widgets[1]

        self.assertTrue(notif1.isVisible())
        self.assertFalse(notif2.isVisible())

        QTest.mouseClick(button, Qt.LeftButton)

        self.assertFalse(notif1.isVisible())
        self.assertTrue(notif2.isVisible())

