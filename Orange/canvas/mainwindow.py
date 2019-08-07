from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import (
    QFormLayout, QCheckBox, QLineEdit, QWidget, QVBoxLayout,
)
from orangecanvas.application.settings import UserSettingsDialog
from orangecanvas.document.usagestatistics import UsageStatistics
from orangewidget.workflow.mainwindow import OWCanvasMainWindow

from orangecanvas.utils.overlay import NotificationOverlay


class OUserSettingsDialog(UserSettingsDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        w = self.widget(0)  # 'General' tab
        layout = w.layout()
        assert isinstance(layout, QFormLayout)
        cb = QCheckBox(self.tr("Automatically check for updates"))
        cb.setAttribute(Qt.WA_LayoutUsesWidgetRect)

        layout.addRow("Updates", cb)
        self.bind(cb, "checked", "startup/check-updates")

        # Error Reporting Tab
        tab = QWidget()
        self.addTab(tab, self.tr("Error Reporting"),
                    toolTip="Settings related to error reporting")

        form = QFormLayout()
        line_edit_mid = QLineEdit()
        self.bind(line_edit_mid, "text", "error-reporting/machine-id")
        form.addRow("Machine ID:", line_edit_mid)

        box = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        cb1 = QCheckBox(
            self.tr(""),
            toolTip=self.tr(
                "Share anonymous usage statistics to improve Orange")
        )
        self.bind(cb1, "checked", "error-reporting/send-statistics")
        cb1.clicked.connect(UsageStatistics.set_enabled)
        layout.addWidget(cb1)
        box.setLayout(layout)
        form.addRow(self.tr("Share Anonymous Statistics"), box)

        tab.setLayout(form)


class MainWindow(OWCanvasMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.notification_overlay = NotificationOverlay(self.scheme_widget)
        self.notification_server = None

    def open_canvas_settings(self):
        # type: () -> None
        """Reimplemented."""
        dlg = OUserSettingsDialog(self, windowTitle=self.tr("Preferences"))
        dlg.show()
        status = dlg.exec_()
        if status == 0:
            self.user_preferences_changed_notify_all()

    def set_notification_server(self, notif_server):
        self.notification_server = notif_server

        # populate notification overlay with current notifications
        for notif in self.notification_server.getNotificationQueue():
            self.notification_overlay.addNotification(notif)

        notif_server.newNotification.connect(self.notification_overlay.addNotification)
        notif_server.nextNotification.connect(self.notification_overlay.nextWidget)

    def create_new_window(self):  # type: () -> CanvasMainWindow
        window = super().create_new_window()
        window.set_notification_server(self.notification_server)
        return window
