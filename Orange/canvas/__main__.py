"""
Orange Canvas main entry point

"""

import os
import sys
import gc
import re
import logging
from logging.handlers import RotatingFileHandler
import optparse
import pickle
import shlex
import shutil
from unittest.mock import patch

import pkg_resources

from PyQt4.QtGui import QFont, QColor, QDesktopServices, QMessageBox
from PyQt4.QtCore import Qt, QDir, QUrl

from Orange import canvas
from Orange.canvas.application.application import CanvasApplication
from Orange.canvas.application.canvasmain import CanvasMainWindow
from Orange.canvas.application.outputview import TextStream, ExceptHook

from Orange.canvas.gui.splashscreen import SplashScreen
from Orange.canvas.config import cache_dir
from Orange.canvas import config
from Orange.canvas.utils.qtcompat import QSettings

from Orange.canvas.registry import qt
from Orange.canvas.registry import WidgetRegistry, set_global_registry
from Orange.canvas.registry import cache

log = logging.getLogger(__name__)


# Allow termination with CTRL + C
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)


def fix_osx_10_9_private_font():
    # Fix fonts on Os X (QTBUG 47206, 40833, 32789)
    if sys.platform == "darwin":
        import platform
        try:
            version = platform.mac_ver()[0]
            version = float(version[:version.rfind(".")])
            if version >= 10.11:  # El Capitan
                QFont.insertSubstitution(".SF NS Text", "Helvetica Neue")
            elif version >= 10.10:  # Yosemite
                QFont.insertSubstitution(".Helvetica Neue DeskInterface",
                                         "Helvetica Neue")
            elif version >= 10.9:
                QFont.insertSubstitution(".Lucida Grande UI", "Lucida Grande")
        except AttributeError:
            pass


def fix_win_pythonw_std_stream():
    """
    On windows when running without a console (using pythonw.exe) the
    std[err|out] file descriptors are invalid and start throwing exceptions
    when their buffer is flushed (`http://bugs.python.org/issue706263`_)

    """
    if sys.platform == "win32" and \
            os.path.basename(sys.executable) == "pythonw.exe":
        if sys.stdout is None:
            sys.stdout = open(os.devnull, "w")
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")


def make_sql_logger(level=logging.INFO):
    sql_log = logging.getLogger('sql_log')
    sql_log.setLevel(level)
    handler = RotatingFileHandler(os.path.join(config.log_dir(), 'sql.log'),
                                  maxBytes=1e7, backupCount=2)
    sql_log.addHandler(handler)


def main(argv=None):
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
                      type="str", default="orange.qss")
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

    # Try to fix fonts on OSX Mavericks
    fix_osx_10_9_private_font()

    # File handler should always be at least INFO level so we need
    # the application root level to be at least at INFO.
    root_level = min(levels[options.log_level], logging.INFO)
    rootlogger = logging.getLogger(canvas.__name__)
    rootlogger.setLevel(root_level)

    # Initialize SQL query and execution time logger (in SqlTable)
    sql_level = min(levels[options.log_level], logging.INFO)
    make_sql_logger(sql_level)

    # Standard output stream handler at the requested level
    stream_hander = logging.StreamHandler()
    stream_hander.setLevel(level=levels[options.log_level])
    rootlogger.addHandler(stream_hander)

    log.info("Starting 'Orange Canvas' application.")

    qt_argv = argv[:1]

    if options.style is not None:
        qt_argv += ["-style", options.style]

    if options.qt is not None:
        qt_argv += shlex.split(options.qt)

    qt_argv += args

    log.debug("Starting CanvasApplicaiton with argv = %r.", qt_argv)
    app = CanvasApplication(qt_argv)

    # NOTE: config.init() must be called after the QApplication constructor
    config.init()

    clear_settings_flag = os.path.join(
        config.widget_settings_dir(), "DELETE_ON_START")

    if options.clear_widget_settings or \
            os.path.isfile(clear_settings_flag):
        log.info("Clearing widget settings")
        shutil.rmtree(
            config.widget_settings_dir(),
            ignore_errors=True)

    file_handler = logging.FileHandler(
        filename=os.path.join(config.log_dir(), "canvas.log"),
        mode="w"
    )

    file_handler.setLevel(root_level)
    rootlogger.addHandler(file_handler)

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

    stylesheet = options.stylesheet
    stylesheet_string = None

    if stylesheet != "none":
        if os.path.isfile(stylesheet):
            stylesheet_string = open(stylesheet, "rb").read()
        else:
            if not os.path.splitext(stylesheet)[1]:
                # no extension
                stylesheet = os.path.extsep.join([stylesheet, "qss"])

            pkg_name = canvas.__name__
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
    dirpath = os.path.abspath(os.path.dirname(canvas.__file__))
    QDir.addSearchPath("canvas_icons", os.path.join(dirpath, "icons"))

    canvas_window = CanvasMainWindow()
    canvas_window.setWindowIcon(config.application_icon())

    if stylesheet_string is not None:
        canvas_window.setStyleSheet(stylesheet_string)

    if not options.force_discovery:
        reg_cache = cache.registry_cache()
    else:
        reg_cache = None

    widget_discovery = qt.QtWidgetDiscovery(cached_descriptions=reg_cache)

    widget_registry = qt.QtWidgetRegistry()

    widget_discovery.found_category.connect(
        widget_registry.register_category
    )
    widget_discovery.found_widget.connect(
        widget_registry.register_widget
    )

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

        widget_discovery.discovery_start.connect(splash_screen.show)
        widget_discovery.discovery_process.connect(show_message)
        widget_discovery.discovery_finished.connect(splash_screen.hide)

    log.info("Running widget discovery process.")

    cache_filename = os.path.join(cache_dir(), "widget-registry.pck")
    if options.no_discovery:
        widget_registry = pickle.load(open(cache_filename, "rb"))
        widget_registry = qt.QtWidgetRegistry(widget_registry)
    else:
        widget_discovery.run(config.widgets_entry_points())
        # Store cached descriptions
        cache.save_registry_cache(widget_discovery.cached_descriptions)
        pickle.dump(WidgetRegistry(widget_registry),
                     open(cache_filename, "wb"))
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

    # If run for the first time, open a browser tab with a survey
    show_survey = settings.value("startup/show-survey", True, type=bool)
    if show_survey:
        question = QMessageBox(
            QMessageBox.Question,
            'Orange Survey',
            'We would like to know more about how our software is used.\n\n'
            'Would you care to fill our short 1-minute survey?',
            QMessageBox.Yes | QMessageBox.No)
        question.setDefaultButton(QMessageBox.Yes)
        later = question.addButton('Ask again later', QMessageBox.NoRole)
        question.setEscapeButton(later)

        def handle_response(result):
            if result == QMessageBox.Yes:
                success = QDesktopServices.openUrl(
                    QUrl("http://orange.biolab.si/survey/short.html"));
                settings.setValue("startup/show-survey", not success)
            else:
                settings.setValue("startup/show-survey", result != QMessageBox.No)

        question.finished.connect(handle_response)
        question.show()

    # Tee stdout and stderr into Output dock
    log_view = canvas_window.log_view()

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

    sys.excepthook = ExceptHook(stream=stderr)

    log.info("Entering main event loop.")
    try:
        with patch('sys.stderr', stderr),\
             patch('sys.stdout', stdout):
            status = app.exec_()
    except BaseException:
        log.error("Error in main event loop.", exc_info=True)

    canvas_window.deleteLater()
    app.processEvents()
    app.flush()
    del canvas_window

    # Collect any cycles before deleting the QApplication instance
    gc.collect()

    del app
    return status


if __name__ == "__main__":
    sys.exit(main())
