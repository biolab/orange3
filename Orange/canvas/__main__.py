"""
Orange Canvas main entry point

"""
from contextlib import closing

import os
import sys
import gc
import re
import time
import logging
import signal
from logging.handlers import RotatingFileHandler
import optparse
import pickle
import shlex
import shutil
from unittest.mock import patch
from urllib.request import urlopen, Request

import pkg_resources

from AnyQt.QtGui import QFont, QColor, QPalette, QDesktopServices, QIcon
from AnyQt.QtCore import (
    Qt, QDir, QUrl, QSettings, QThread, pyqtSignal, QT_VERSION
)
import pyqtgraph

import orangecanvas
from orangecanvas.registry import qt
from orangecanvas.registry import WidgetRegistry, set_global_registry
from orangecanvas.registry import cache
from orangecanvas.application.application import CanvasApplication
from orangecanvas.application.outputview import TextStream, ExceptHook
from orangecanvas.gui.splashscreen import SplashScreen
from orangecanvas import config as canvasconfig
from orangecanvas.main import (
    fix_win_pythonw_std_stream, fix_set_proxy_env, fix_macos_nswindow_tabbing,
    breeze_dark,
)

from orangewidget.workflow.errorreporting import handle_exception

from Orange.canvas import config
from Orange.canvas.utils.overlay import NotificationWidget, NotificationOverlay
from Orange.canvas.mainwindow import MainWindow
from Orange.widgets import gui


log = logging.getLogger(__name__)


def make_sql_logger(level=logging.INFO):
    sql_log = logging.getLogger('sql_log')
    sql_log.setLevel(level)
    handler = RotatingFileHandler(os.path.join(config.log_dir(), 'sql.log'),
                                  maxBytes=1e7, backupCount=2)
    sql_log.addHandler(handler)


def make_stdout_handler(level, fmt=None):
    handler = logging.StreamHandler()
    handler.setLevel(level)
    if fmt:
        handler.setFormatter(logging.Formatter(fmt))
    return handler


def setup_notifications():
    settings = QSettings()
    # If run for the fifth time, prompt short survey
    show_survey = settings.value("startup/show-short-survey", True, type=bool) and \
                  settings.value("startup/launch-count", 0, type=int) >= 5
    if show_survey:
        surveyDialogButtons = NotificationWidget.Ok | NotificationWidget.Close
        surveyDialog = NotificationWidget(icon=QIcon(gui.resource_filename("icons/information.png")),
                                          title="Survey",
                                          text="We want to understand our users better.\n"
                                               "Would you like to take a short survey?",
                                          standardButtons=surveyDialogButtons)

        def handle_survey_response(button):
            if surveyDialog.buttonRole(button) == NotificationWidget.AcceptRole:
                success = QDesktopServices.openUrl(
                    QUrl("https://orange.biolab.si/survey/short.html"))
                settings.setValue("startup/show-short-survey", not success)
            elif surveyDialog.buttonRole(button) == NotificationWidget.RejectRole:
                settings.setValue("startup/show-short-survey", False)

        surveyDialog.clicked.connect(handle_survey_response)

        NotificationOverlay.registerNotification(surveyDialog)

    # data collection permission
    if not settings.value("error-reporting/permission-requested", False, type=bool):
        permDialogButtons = NotificationWidget.Ok | NotificationWidget.Close
        permDialog = NotificationWidget(icon=QIcon(gui.resource_filename(
                                                                "../../distribute/icon-48.png")),
                                        title="Anonymous Usage Statistics",
                                        text="Do you wish to opt-in to sharing "
                                             "statistics about how you use Orange?\n"
                                             "All data is anonymized and used "
                                             "exclusively for understanding how users "
                                             "interact with Orange.",
                                        standardButtons=permDialogButtons)
        btnOK = permDialog.button(NotificationWidget.AcceptRole)
        btnOK.setText("Allow")

        def handle_permission_response(button):
            if permDialog.buttonRole(button) != permDialog.DismissRole:
                settings.setValue("error-reporting/permission-requested", True)
            if permDialog.buttonRole(button) == permDialog.AcceptRole:
                settings.setValue("error-reporting/send-statistics", True)

        permDialog.clicked.connect(handle_permission_response)

        NotificationOverlay.registerNotification(permDialog)


def check_for_updates():
    settings = QSettings()
    check_updates = settings.value('startup/check-updates', True, type=bool)
    last_check_time = settings.value('startup/last-update-check-time', 0, type=int)
    ONE_DAY = 86400

    if check_updates and time.time() - last_check_time > ONE_DAY:
        settings.setValue('startup/last-update-check-time', int(time.time()))

        from Orange.version import version as current

        class GetLatestVersion(QThread):
            resultReady = pyqtSignal(str)

            def run(self):
                try:
                    request = Request('https://orange.biolab.si/version/',
                                      headers={
                                          'Accept': 'text/plain',
                                          'Accept-Encoding': 'gzip, deflate',
                                          'Connection': 'close',
                                          'User-Agent': self.ua_string()})
                    contents = urlopen(request, timeout=10).read().decode()
                # Nothing that this fails with should make Orange crash
                except Exception:  # pylint: disable=broad-except
                    log.exception('Failed to check for updates')
                else:
                    self.resultReady.emit(contents)

            @staticmethod
            def ua_string():
                is_anaconda = 'Continuum' in sys.version or 'conda' in sys.version
                return 'Orange{orange_version}:Python{py_version}:{platform}:{conda}'.format(
                    orange_version=current,
                    py_version='.'.join(sys.version[:3]),
                    platform=sys.platform,
                    conda='Anaconda' if is_anaconda else '',
                )

        def compare_versions(latest):
            version = pkg_resources.parse_version
            skipped = settings.value('startup/latest-skipped-version', "", type=str)
            if version(latest) <= version(current) or \
                    latest == skipped:
                return

            questionButtons = NotificationWidget.Ok | NotificationWidget.Close
            question = NotificationWidget(icon=QIcon(gui.resource_filename('icons/Dlg_down3.png')),
                                          title='Orange Update Available',
                                          text='Current version: <b>{}</b><br>'
                                               'Latest version: <b>{}</b>'.format(current, latest),
                                          textFormat=Qt.RichText,
                                          standardButtons=questionButtons,
                                          acceptLabel="Download",
                                          rejectLabel="Skip this Version")

            def handle_click(b):
                if question.buttonRole(b) == question.RejectRole:
                    settings.setValue('startup/latest-skipped-version', latest)
                if question.buttonRole(b) == question.AcceptRole:
                    QDesktopServices.openUrl(QUrl("https://orange.biolab.si/download/"))

            question.clicked.connect(handle_click)

            NotificationOverlay.registerNotification(question)

        thread = GetLatestVersion()
        thread.resultReady.connect(compare_versions)
        thread.start()
        return thread
    return None


def main(argv=None):
    # Allow termination with CTRL + C
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    # Disable pyqtgraph's atexit and QApplication.aboutToQuit cleanup handlers.
    pyqtgraph.setConfigOption("exitCleanup", False)

    if argv is None:
        argv = sys.argv

    usage = "usage: %prog [options] [workflow_file]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--no-discovery",
                      action="store_true",
                      help="Don't run widget discovery "
                           "(use full cache instead)")
    parser.add_option("--force-discovery",
                      action="store_true",
                      help="Force full widget discovery "
                           "(invalidate cache)")
    parser.add_option("--clear-widget-settings",
                      action="store_true",
                      help="Remove stored widget setting")
    parser.add_option("--no-welcome",
                      action="store_true",
                      help="Don't show welcome dialog.")
    parser.add_option("--no-splash",
                      action="store_true",
                      help="Don't show splash screen.")
    parser.add_option("-l", "--log-level",
                      help="Logging level (0, 1, 2, 3, 4)",
                      type="int", default=1)
    parser.add_option("--style",
                      help="QStyle to use",
                      type="str", default=None)
    parser.add_option("--stylesheet",
                      help="Application level CSS style sheet to use",
                      type="str", default=None)
    parser.add_option("--qt",
                      help="Additional arguments for QApplication",
                      type="str", default=None)

    (options, args) = parser.parse_args(argv[1:])

    levels = [logging.CRITICAL,
              logging.ERROR,
              logging.WARN,
              logging.INFO,
              logging.DEBUG]

    # Fix streams before configuring logging (otherwise it will store
    # and write to the old file descriptors)
    fix_win_pythonw_std_stream()

    # Try to fix macOS automatic window tabbing (Sierra and later)
    fix_macos_nswindow_tabbing()

    logging.basicConfig(
        level=levels[options.log_level],
        handlers=[make_stdout_handler(levels[options.log_level])]
    )
    # set default application configuration
    config_ = config.Config()
    canvasconfig.set_default(config_)
    log.info("Starting 'Orange Canvas' application.")

    qt_argv = argv[:1]

    style = options.style
    defaultstylesheet = "orange.qss"
    fusiontheme = None

    if style is not None:
        if style.startswith("fusion:"):
            qt_argv += ["-style", "fusion"]
            _, _, fusiontheme = style.partition(":")
        else:
            qt_argv += ["-style", style]

    if options.qt is not None:
        qt_argv += shlex.split(options.qt)

    qt_argv += args

    if QT_VERSION >= 0x50600:
        CanvasApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    log.debug("Starting CanvasApplicaiton with argv = %r.", qt_argv)
    app = CanvasApplication(qt_argv)
    config_.init()
    if app.style().metaObject().className() == "QFusionStyle":
        if fusiontheme == "breeze-dark":
            app.setPalette(breeze_dark())
            defaultstylesheet = "darkorange.qss"

    palette = app.palette()
    if style is None and palette.color(QPalette.Window).value() < 127:
        log.info("Switching default stylesheet to darkorange")
        defaultstylesheet = "darkorange.qss"

    # Initialize SQL query and execution time logger (in SqlTable)
    sql_level = min(levels[options.log_level], logging.INFO)
    make_sql_logger(sql_level)

    clear_settings_flag = os.path.join(
        config.widget_settings_dir(), "DELETE_ON_START")

    if options.clear_widget_settings or \
            os.path.isfile(clear_settings_flag):
        log.info("Clearing widget settings")
        shutil.rmtree(
            config.widget_settings_dir(),
            ignore_errors=True)

    # Set http_proxy environment variables, after (potentially) clearing settings
    fix_set_proxy_env()

    # Setup file log handler for the select logger list - this is always
    # at least INFO
    level = min(levels[options.log_level], logging.INFO)
    file_handler = logging.FileHandler(
        filename=os.path.join(config.log_dir(), "canvas.log"),
        mode="w"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")
    )
    file_handler.setLevel(level)

    for namespace in ["orangecanvas", "orangewidget", "Orange"]:
        logger = logging.getLogger(namespace)
        logger.setLevel(level)
        logger.addHandler(file_handler)

    # intercept any QFileOpenEvent requests until the main window is
    # fully initialized.
    # NOTE: The QApplication must have the executable ($0) and filename
    # arguments passed in argv otherwise the FileOpen events are
    # triggered for them (this is done by Cocoa, but QApplicaiton filters
    # them out if passed in argv)

    open_requests = []

    def onrequest(url):
        log.info("Received an file open request %s", url)
        open_requests.append(url)

    app.fileOpenRequest.connect(onrequest)

    settings = QSettings()

    settings.setValue('startup/launch-count', settings.value('startup/launch-count', 0, int) + 1)

    stylesheet = options.stylesheet or defaultstylesheet
    stylesheet_string = None

    if stylesheet != "none":
        if os.path.isfile(stylesheet):
            with open(stylesheet, "r") as f:
                stylesheet_string = f.read()
        else:
            if not os.path.splitext(stylesheet)[1]:
                # no extension
                stylesheet = os.path.extsep.join([stylesheet, "qss"])

            pkg_name = orangecanvas.__name__
            resource = "styles/" + stylesheet

            if pkg_resources.resource_exists(pkg_name, resource):
                stylesheet_string = \
                    pkg_resources.resource_string(pkg_name, resource).decode()

                base = pkg_resources.resource_filename(pkg_name, "styles")

                pattern = re.compile(
                    r"^\s@([a-zA-Z0-9_]+?)\s*:\s*([a-zA-Z0-9_/]+?);\s*$",
                    flags=re.MULTILINE
                )

                matches = pattern.findall(stylesheet_string)

                for prefix, search_path in matches:
                    QDir.addSearchPath(prefix, os.path.join(base, search_path))
                    log.info("Adding search path %r for prefix, %r",
                             search_path, prefix)

                stylesheet_string = pattern.sub("", stylesheet_string)

            else:
                log.info("%r style sheet not found.", stylesheet)

    # Add the default canvas_icons search path
    dirpath = os.path.abspath(os.path.dirname(orangecanvas.__file__))
    QDir.addSearchPath("canvas_icons", os.path.join(dirpath, "icons"))

    canvas_window = MainWindow()
    canvas_window.setAttribute(Qt.WA_DeleteOnClose)
    canvas_window.setWindowIcon(config.application_icon())

    if stylesheet_string is not None:
        canvas_window.setStyleSheet(stylesheet_string)

    if not options.force_discovery:
        reg_cache = cache.registry_cache()
    else:
        reg_cache = None

    widget_registry = qt.QtWidgetRegistry()
    widget_discovery = config_.widget_discovery(
        widget_registry, cached_descriptions=reg_cache)

    want_splash = \
        settings.value("startup/show-splash-screen", True, type=bool) and \
        not options.no_splash

    if want_splash:
        pm, rect = config.splash_screen()
        splash_screen = SplashScreen(pixmap=pm, textRect=rect)
        splash_screen.setFont(QFont("Helvetica", 12))
        color = QColor("#FFD39F")

        def show_message(message):
            splash_screen.showMessage(message, color=color)

        widget_registry.category_added.connect(show_message)

    log.info("Running widget discovery process.")

    cache_filename = os.path.join(config.cache_dir(), "widget-registry.pck")
    if options.no_discovery:
        with open(cache_filename, "rb") as f:
            widget_registry = pickle.load(f)
        widget_registry = qt.QtWidgetRegistry(widget_registry)
    else:
        if want_splash:
            splash_screen.show()
        widget_discovery.run(config.widgets_entry_points())
        if want_splash:
            splash_screen.hide()
            splash_screen.deleteLater()

        # Store cached descriptions
        cache.save_registry_cache(widget_discovery.cached_descriptions)
        with open(cache_filename, "wb") as f:
            pickle.dump(WidgetRegistry(widget_registry), f)

    set_global_registry(widget_registry)
    canvas_window.set_widget_registry(widget_registry)
    canvas_window.show()
    canvas_window.raise_()

    want_welcome = \
        settings.value("startup/show-welcome-screen", True, type=bool) \
        and not options.no_welcome

    # Process events to make sure the canvas_window layout has
    # a chance to activate (the welcome dialog is modal and will
    # block the event queue, plus we need a chance to receive open file
    # signals when running without a splash screen)
    app.processEvents()

    app.fileOpenRequest.connect(canvas_window.open_scheme_file)

    if want_welcome and not args and not open_requests:
        canvas_window.welcome_dialog()

    elif args:
        log.info("Loading a scheme from the command line argument %r",
                 args[0])
        canvas_window.load_scheme(args[0])
    elif open_requests:
        log.info("Loading a scheme from an `QFileOpenEvent` for %r",
                 open_requests[-1])
        canvas_window.load_scheme(open_requests[-1].toLocalFile())

    # initialize notifications
    setup_notifications()

    # local references prevent destruction
    update_check = check_for_updates()

    # Tee stdout and stderr into Output dock
    log_view = canvas_window.output_view()

    stdout = TextStream()
    stdout.stream.connect(log_view.write)
    if sys.stdout:
        stdout.stream.connect(sys.stdout.write)
        stdout.flushed.connect(sys.stdout.flush)

    stderr = TextStream()
    error_writer = log_view.formated(color=Qt.red)
    stderr.stream.connect(error_writer.write)
    if sys.stderr:
        stderr.stream.connect(sys.stderr.write)
        stderr.flushed.connect(sys.stderr.flush)

    log.info("Entering main event loop.")
    excepthook = ExceptHook(stream=stderr)
    excepthook.handledException.connect(handle_exception)
    try:
        with closing(stdout),\
             closing(stderr),\
             patch('sys.excepthook', excepthook),\
             patch('sys.stderr', stderr),\
             patch('sys.stdout', stdout):
            status = app.exec_()
    except BaseException:
        log.error("Error in main event loop.", exc_info=True)
        status = 42

    del canvas_window
    del update_check

    app.processEvents()
    app.flush()
    # Collect any cycles before deleting the QApplication instance
    gc.collect()

    del app
    return status


if __name__ == "__main__":
    sys.exit(main())
