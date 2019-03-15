"""
Orange Canvas main entry point

"""

import os
import sys
import gc
import re
import time
import logging
from logging.handlers import RotatingFileHandler
import optparse
import pickle
import shlex
import shutil
from unittest.mock import patch
from urllib.request import urlopen, Request, getproxies

import pkg_resources

from AnyQt.QtGui import QFont, QColor, QPalette, QDesktopServices
from AnyQt.QtWidgets import QMessageBox
from AnyQt.QtCore import (
    Qt, QDir, QUrl, QSettings, QThread, pyqtSignal, QT_VERSION
)

from Orange import canvas
from Orange.canvas.application.application import CanvasApplication
from Orange.canvas.application.canvasmain import CanvasMainWindow
from Orange.canvas.application.outputview import TextStream, ExceptHook
from Orange.canvas.application.errorreporting import handle_exception

from Orange.canvas.gui.splashscreen import SplashScreen
from Orange.canvas.config import cache_dir
from Orange.canvas import config
from Orange.canvas.document.usagestatistics import UsageStatistics

from Orange.canvas.registry import qt
from Orange.canvas.registry import WidgetRegistry, set_global_registry
from Orange.canvas.registry import cache

log = logging.getLogger(__name__)

# The modules below are imported on the fly (and used just there) for clarity
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position

# Allow termination with CTRL + C
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Disable pyqtgraph's atexit and QApplication.aboutToQuit cleanup handlers.
import pyqtgraph
pyqtgraph.setConfigOption("exitCleanup", False)


default_proxies = None


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


def fix_macos_nswindow_tabbing():
    """
    Disable automatic NSWindow tabbing on macOS Sierra and higher.

    See QTBUG-61707
    """
    import ctypes
    import ctypes.util
    import platform

    if sys.platform != "darwin":
        return
    ver, _, _ = platform.mac_ver()
    ver = tuple(map(int, ver.split(".")[:2]))
    if ver < (10, 12):
        return

    c_char_p, c_void_p = ctypes.c_char_p, ctypes.c_void_p
    id = Sel = Class = c_void_p

    def annotate(func, restype, argtypes):
        func.restype = restype
        func.argtypes = argtypes
        return func
    try:
        libobjc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("libobjc"))
        # Load AppKit.framework which contains NSWindow class
        # pylint: disable=unused-variable
        AppKit = ctypes.cdll.LoadLibrary(ctypes.util.find_library("AppKit"))
        objc_getClass = annotate(
            libobjc.objc_getClass, Class, [c_char_p])
        objc_msgSend = annotate(
            libobjc.objc_msgSend, id, [id, Sel])
        sel_registerName = annotate(
            libobjc.sel_registerName, Sel, [c_char_p])
        class_getClassMethod = annotate(
            libobjc.class_getClassMethod, c_void_p, [Class, Sel])
    except (OSError, AttributeError):
        return

    NSWindow = objc_getClass(b"NSWindow")
    if NSWindow is None:
        return
    setAllowsAutomaticWindowTabbing = sel_registerName(
        b'setAllowsAutomaticWindowTabbing:'
    )
    # class_respondsToSelector does not work (for class methods)
    if class_getClassMethod(NSWindow, setAllowsAutomaticWindowTabbing):
        # [NSWindow setAllowsAutomaticWindowTabbing: NO]
        objc_msgSend(
            NSWindow,
            setAllowsAutomaticWindowTabbing,
            ctypes.c_bool(False),
        )


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


def fix_set_proxy_env():
    """
    Set http_proxy/https_proxy environment variables (for requests, pip, ...)
    from user-specified settings or, if none, from system settings on OS X
    and from registry on Windos.
    """
    # save default proxies so that setting can be reset
    global default_proxies
    if default_proxies is None:
        default_proxies = getproxies()  # can also read windows and macos settings

    settings = QSettings()
    proxies = getproxies()
    for scheme in set(["http", "https"]) | set(proxies):
        from_settings = settings.value("network/" + scheme + "-proxy", "", type=str)
        from_default = default_proxies.get(scheme, "")
        env_scheme = scheme + '_proxy'
        if from_settings:
            os.environ[env_scheme] = from_settings
        elif from_default:
            os.environ[env_scheme] = from_default  # crucial for windows/macos support
        else:
            os.environ.pop(env_scheme, "")


def make_sql_logger(level=logging.INFO):
    sql_log = logging.getLogger('sql_log')
    sql_log.setLevel(level)
    handler = RotatingFileHandler(os.path.join(config.log_dir(), 'sql.log'),
                                  maxBytes=1e7, backupCount=2)
    sql_log.addHandler(handler)


def show_survey():
    # If run for the first time, open a browser tab with a survey
    settings = QSettings()
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
                    QUrl("https://orange.biolab.si/survey/short.html"))
                settings.setValue("startup/show-survey", not success)
            else:
                settings.setValue("startup/show-survey", result != QMessageBox.No)

        question.finished.connect(handle_response)
        question.show()
        return question


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
            if version(latest) <= version(current):
                return
            question = QMessageBox(
                QMessageBox.Information,
                'Orange Update Available',
                'A newer version of Orange is available.<br><br>'
                '<b>Current version:</b> {}<br>'
                '<b>Latest version:</b> {}'.format(current, latest),
                textFormat=Qt.RichText)
            ok = question.addButton('Download Now', question.AcceptRole)
            question.setDefaultButton(ok)
            question.addButton('Remind Later', question.RejectRole)
            question.finished.connect(
                lambda:
                question.clickedButton() == ok and
                QDesktopServices.openUrl(QUrl("https://orange.biolab.si/download/")))
            question.show()

        thread = GetLatestVersion()
        thread.resultReady.connect(compare_versions)
        thread.start()
        return thread

def send_usage_statistics():

    class SendUsageStatistics(QThread):
        def run(self):
            try:
                usage = UsageStatistics()
                usage.send_statistics()
            except Exception:  # pylint: disable=broad-except
                # exceptions in threads would crash Orange
                log.warning("Failed to send usage statistics.")

    thread = SendUsageStatistics()
    thread.start()
    return thread

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

    # Try to fix fonts on OSX Mavericks
    fix_osx_10_9_private_font()

    # Try to fix macOS automatic window tabbing (Sierra and later)
    fix_macos_nswindow_tabbing()

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
    if app.style().metaObject().className() == "QFusionStyle":
        if fusiontheme == "breeze-dark":
            app.setPalette(breeze_dark())
            defaultstylesheet = "darkorange.qss"

    palette = app.palette()
    if style is None and palette.color(QPalette.Window).value() < 127:
        log.info("Switching default stylesheet to darkorange")
        defaultstylesheet = "darkorange.qss"

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

    # Set http_proxy environment variables, after (potentially) clearing settings
    fix_set_proxy_env()

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
    canvas_window.setAttribute(Qt.WA_DeleteOnClose)
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
        with open(cache_filename, "rb") as f:
            widget_registry = pickle.load(f)
        widget_registry = qt.QtWidgetRegistry(widget_registry)
    else:
        widget_discovery.run(config.widgets_entry_points())
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

    # local references prevent destruction
    survey = show_survey()
    update_check = check_for_updates()
    send_stat = send_usage_statistics()

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

    log.info("Entering main event loop.")
    try:
        with patch('sys.excepthook',
                   ExceptHook(stream=stderr,
                              handledException=handle_exception)),\
             patch('sys.stderr', stderr),\
             patch('sys.stdout', stdout):
            status = app.exec_()
    except BaseException:
        log.error("Error in main event loop.", exc_info=True)
        status = 42

    del canvas_window
    del survey
    del update_check
    del send_stat

    app.processEvents()
    app.flush()
    # Collect any cycles before deleting the QApplication instance
    gc.collect()

    del app
    return status


def breeze_dark():
    # 'Breeze Dark' color scheme from KDE.
    text = QColor(239, 240, 241)
    textdisabled = QColor(98, 108, 118)
    window = QColor(49, 54, 59)
    base = QColor(35, 38, 41)
    highlight = QColor(61, 174, 233)
    link = QColor(41, 128, 185)

    light = QColor(69, 76, 84)
    mid = QColor(43, 47, 52)
    dark = QColor(28, 31, 34)
    shadow = QColor(20, 23, 25)

    palette = QPalette()
    palette.setColor(QPalette.Window, window)
    palette.setColor(QPalette.WindowText, text)
    palette.setColor(QPalette.Disabled, QPalette.WindowText, textdisabled)
    palette.setColor(QPalette.Base, base)
    palette.setColor(QPalette.AlternateBase, window)
    palette.setColor(QPalette.ToolTipBase, window)
    palette.setColor(QPalette.ToolTipText, text)
    palette.setColor(QPalette.Text, text)
    palette.setColor(QPalette.Disabled, QPalette.Text, textdisabled)
    palette.setColor(QPalette.Button, window)
    palette.setColor(QPalette.ButtonText, text)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, textdisabled)
    palette.setColor(QPalette.BrightText, Qt.white)
    palette.setColor(QPalette.Highlight, highlight)
    palette.setColor(QPalette.HighlightedText, text)
    palette.setColor(QPalette.Light, light)
    palette.setColor(QPalette.Mid, mid)
    palette.setColor(QPalette.Dark, dark)
    palette.setColor(QPalette.Shadow, shadow)
    palette.setColor(QPalette.Link, link)
    return palette


if __name__ == "__main__":
    sys.exit(main())
