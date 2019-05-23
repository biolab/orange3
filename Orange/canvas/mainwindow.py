import os
from typing import Optional

from AnyQt.QtCore import Qt, QSettings
from AnyQt.QtWidgets import (
    QAction, QFileDialog, QMenu, QMenuBar, QWidget, QMessageBox
)
from AnyQt.QtGui import QKeySequence

from orangecanvas.application.canvasmain import CanvasMainWindow
from Orange.canvas.widgetsscheme import WidgetsScheme
from Orange.canvas import config
from Orange.widgets.report.owreport import HAVE_REPORT, OWReport


def _insert_action(mb, menuid, beforeactionid, action):
    # type: (QMenuBar, str, str, QAction) -> bool
    """
    Insert an action into one of a QMenuBar's menu.

    Parameters
    ----------
    mb : QMenuBar
        The menu bar
    menuid : str
        The target menu's objectName. The menu must be a child of `mb`.
    beforeactionid : str
        The objectName of the action before which the action will be inserted.
    action : QAction
        The action to insert

    Returns
    -------
    success: bool
        True if the actions was successfully inserted (the menu and before
        actions were found), False otherwise
    """
    def find_action(widget, name):  # type: (QWidget, str) -> Optional[QAction]
        for a in widget.actions():
            if a.objectName() == name:
                return a
        return None

    menu = mb.findChild(QMenu, menuid)
    if menu is not None:
        sep = find_action(menu, beforeactionid)
        if sep:
            menu.insertAction(sep, action)
            return True
    return False


class OWCanvasMainWindow(CanvasMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.show_report_action = QAction(
            "Show report", self,
            objectName="action-show-report",
            toolTip="Show a report window",
            shortcut=QKeySequence(Qt.ShiftModifier | Qt.Key_R),
            enabled=HAVE_REPORT,
        )
        self.show_report_action.triggered.connect(self.show_report_view)
        self.open_report_action = QAction(
            "Open Report...", self,
            objectName="action-open-report",
            toolTip="Open a saved report",
            enabled=HAVE_REPORT,
        )
        self.open_report_action.triggered.connect(self.open_report)
        self.reset_widget_settings_action = QAction(
            self.tr("Reset Widget Settings..."), self,
            triggered=self.reset_widget_settings
        )

        menubar = self.menuBar()
        # Insert the 'Load report' in the File menu ...
        _insert_action(menubar, "file-menu", "open-actions-separator",
                       self.open_report_action)
        # ... and 'Show report' in the View menu.
        _insert_action(menubar, "view-menu", "view-visible-actions-separator",
                       self.show_report_action)

        _insert_action(menubar, "options-menu", "canvas-addons-action",
                       self.reset_widget_settings_action)

    def open_report(self):
        """
        Present an 'Open report' dialog to the user, load a '.report' file
        (as saved by OWReport) and create a new canvas window associated
        with the OWReport instance.
        """
        settings = QSettings()
        KEY = "report/file-dialog-dir"
        start_dir = settings.value(KEY, "", type=str)
        dlg = QFileDialog(
            self,
            windowTitle=self.tr("Open Report"),
            acceptMode=QFileDialog.AcceptOpen,
            fileMode=QFileDialog.ExistingFile,
        )
        if os.path.isdir(start_dir):
            dlg.setDirectory(start_dir)

        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setNameFilters(["Report (*.report)"])

        def accepted():
            directory = dlg.directory().absolutePath()
            filename = dlg.selectedFiles()[0]
            settings.setValue(KEY, directory)
            self._open_report(filename)

        dlg.accepted.connect(accepted)
        dlg.exec()

    def _open_report(self, filename):
        """
        Open and load a '*.report' from 'filename'
        """
        report = OWReport.load(filename)
        # Create a new window for the report
        if self.is_transient():
            window = self
        else:
            window = self.create_new_window()
        # toggle the window modified flag (this will clear the 'is_transient'
        # flag on the new window)
        window.setWindowModified(True)
        window.setWindowModified(False)

        report.setParent(window, Qt.Window)
        sc = window.current_document().scheme()  # type: WidgetsScheme
        sc.set_report_view(report)

        window.show()
        window.raise_()
        window.show_report_view()

        report._build_html()
        report.table.selectRow(0)
        report.show()
        report.raise_()

    def show_report_view(self):
        """
        Show the 'Report' view for the current workflow.
        """
        sc = self.current_document().scheme()  # type: WidgetsScheme
        sc.show_report_view()

    def reset_widget_settings(self):
        mb = QMessageBox(
            self,
            windowTitle="Clear settings",
            text="Orange needs to be restarted for the changes to take effect.",
            icon=QMessageBox.Information,
            informativeText="Press OK to close Orange now.",
            standardButtons=QMessageBox.Ok | QMessageBox.Cancel,
        )
        res = mb.exec()
        if res == QMessageBox.Ok:
            # Touch a finely crafted file inside the settings directory.
            # The existence of this file is checked by the canvas main
            # function and is deleted there.
            fname = os.path.join(config.widget_settings_dir(),
                                 "DELETE_ON_START")
            os.makedirs(config.widget_settings_dir(), exist_ok=True)
            with open(fname, "a"):
                pass

            if not self.close():
                QMessageBox(
                    self,
                    text="Settings will still be reset at next application start",
                    icon=QMessageBox.Information
                ).exec()

