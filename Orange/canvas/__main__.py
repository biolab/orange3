"""
Orange Canvas main entry point

"""
import uuid
from collections import defaultdict
from contextlib import closing

import os
import sys
from datetime import date

import gc
import re
import time
import logging
from logging.handlers import RotatingFileHandler
import signal
import optparse
import pickle
import shlex
import shutil
from unittest.mock import patch
from urllib.request import urlopen, Request

import pkg_resources
import yaml

from AnyQt.QtGui import QFont, QColor, QPalette, QDesktopServices, QIcon
from AnyQt.QtCore import (
    Qt, QDir, QUrl, QSettings, QThread, pyqtSignal, QT_VERSION, QFile
)
import pyqtgraph

import orangecanvas
from orangecanvas import config as canvasconfig
from orangecanvas.registry import qt, WidgetRegistry, set_global_registry, cache
from orangecanvas.application.application import CanvasApplication
from orangecanvas.application.outputview import TextStream, ExceptHook
from orangecanvas.document.usagestatistics import UsageStatistics
from orangecanvas.gui.splashscreen import SplashScreen
from orangecanvas.utils.after_exit import run_after_exit
from orangecanvas.utils.overlay import Notification, NotificationServer
from orangecanvas.main import (
    fix_win_pythonw_std_stream, fix_set_proxy_env, fix_macos_nswindow_tabbing,
    breeze_dark,
)

from orangewidget.workflow.errorreporting import handle_exception

from Orange import canvas
from Orange.util import literal_eval, requirementsSatisfied, resource_filename
from Orange.version import version as current, release as is_release
from Orange.canvas import config
from Orange.canvas.mainwindow import MainWindow
from Orange.widgets.settings import widget_settings_dir


log = logging.getLogger(__name__)

statistics_server_url = os.getenv(
    'ORANGE_STATISTICS_API_URL', "https://orange.biolab.si/usage-statistics"
)


def ua_string():
    is_anaconda = 'Continuum' in sys.version or 'conda' in sys.version
    return 'Orange{orange_version}:Python{py_version}:{platform}:{conda}'.format(
        orange_version=current,
        py_version='.'.join(str(a) for a in sys.version_info[:3]),
        platform=sys.platform,
        conda='Anaconda' if is_anaconda else ''
    )


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


def check_for_updates():
    settings = QSettings()
    check_updates = settings.value('startup/check-updates', True, type=bool)
    last_check_time = settings.value('startup/last-update-check-time', 0, type=int)
    ONE_DAY = 86400

    if check_updates and time.time() - last_check_time > ONE_DAY:
        settings.setValue('startup/last-update-check-time', int(time.time()))

        class GetLatestVersion(QThread):
            resultReady = pyqtSignal(str)

            def run(self):
                try:
                    request = Request('https://orange.biolab.si/version/',
                                      headers={
                                          'Accept': 'text/plain',
                                          'Accept-Encoding': 'gzip, deflate',
                                          'Connection': 'close',
                                          'User-Agent': ua_string()})
                    contents = urlopen(request, timeout=10).read().decode()
                # Nothing that this fails with should make Orange crash
                except Exception:  # pylint: disable=broad-except
                    log.exception('Failed to check for updates')
                else:
                    self.resultReady.emit(contents)

        def compare_versions(latest):
            version = pkg_resources.parse_version
            skipped = settings.value('startup/latest-skipped-version', "", type=str)
            if version(latest) <= version(current) or \
                    latest == skipped:
                return

            notif = Notification(title='Orange Update Available',
                                 text='Current version: <b>{}</b><br>'
                                      'Latest version: <b>{}</b>'.format(current, latest),
                                 accept_button_label="Download",
                                 reject_button_label="Skip this Version",
                                 icon=QIcon(resource_filename("canvas/icons/update.png")))

            def handle_click(role):
                if role == notif.RejectRole:
                    settings.setValue('startup/latest-skipped-version', latest)
                if role == notif.AcceptRole:
                    QDesktopServices.openUrl(QUrl("https://orange.biolab.si/download/"))

            notif.clicked.connect(handle_click)
            canvas.notification_server_instance.registerNotification(notif)

        thread = GetLatestVersion()
        thread.resultReady.connect(compare_versions)
        thread.start()
        return thread
    return None


def open_link(url: QUrl):
    if url.scheme() == "orange":
        # define custom actions within Orange here
        if url.host() == "enable-statistics":
            settings = QSettings()

            settings.setValue("reporting/send-statistics", True)
            UsageStatistics.set_enabled(True)

            if not settings.contains('reporting/machine-id'):
                settings.setValue('reporting/machine-id', str(uuid.uuid4()))
    else:
        QDesktopServices.openUrl(url)


class YAMLNotification:
    """
    Class used for safe loading of yaml file.
    """
    # pylint: disable=redefined-builtin
    def __init__(self, id=None, type=None, start=None, end=None, requirements=None, icon=None,
                 title=None, text=None, link=None, accept_button_label=None,
                 reject_button_label=None):
        self.id = id
        self.type = type
        self.start = start
        self.end = end
        self.requirements = requirements
        self.icon = icon
        self.title = title
        self.text = text
        self.link = link
        self.accept_button_label = accept_button_label
        self.reject_button_label = reject_button_label

    def toNotification(self):
        return Notification(title=self.title,
                            text=self.text,
                            accept_button_label=self.accept_button_label,
                            reject_button_label=self.reject_button_label,
                            icon=QIcon(resource_filename(self.icon)))

    @staticmethod
    def yamlConstructor(loader, node):
        fields = loader.construct_mapping(node)
        return YAMLNotification(**fields)


yaml.add_constructor(u'!Notification', YAMLNotification.yamlConstructor, Loader=yaml.SafeLoader)


def pull_notifications():
    settings = QSettings()
    check_notifs = settings.value("notifications/check-notifications", True, bool)
    if not check_notifs:
        return None

    Version = pkg_resources.parse_version

    # create settings_dict for notif requirements purposes (read-only)
    spec = canvasconfig.spec + config.spec
    settings_dict = canvasconfig.Settings(defaults=spec, store=settings)

    # map of installed addon name -> version
    installed_list = [ep.dist for ep in config.addon_entry_points()
                      if ep.dist is not None]
    installed = defaultdict(lambda: "-1")
    for addon in installed_list:
        installed[addon.project_name] = addon.version

    # get set of already displayed notification IDs, stored in settings["notifications/displayed"]
    displayedIDs = literal_eval(settings.value("notifications/displayed", "set()", str))

    # get notification feed from Github
    class GetNotifFeed(QThread):
        resultReady = pyqtSignal(str)

        def run(self):
            try:
                request = Request('https://orange.biolab.si/notification-feed',
                                  headers={
                                      'Accept': 'text/plain',
                                      'Connection': 'close',
                                      'User-Agent': ua_string(),
                                      'Cache-Control': 'no-cache',
                                      'Pragma': 'no-cache'})
                contents = urlopen(request, timeout=10).read().decode()
            # Nothing that this fails with should make Orange crash
            except Exception:  # pylint: disable=broad-except
                log.warning('Failed to pull notification feed')
            else:
                self.resultReady.emit(contents)

    thread = GetNotifFeed()

    def parse_yaml_notification(YAMLnotif: YAMLNotification):
        # check if notification has been displayed and responded to previously
        if YAMLnotif.id and YAMLnotif.id in displayedIDs:
            return

        # check if type is filtered by user
        allowAnnouncements = settings.value('notifications/announcements', True, bool)
        allowBlog = settings.value('notifications/blog', True, bool)
        allowNewFeatures = settings.value('notifications/new-features', True, bool)
        if YAMLnotif.type and \
                (YAMLnotif.type == 'announcement' and not allowAnnouncements
                 or YAMLnotif.type == 'blog' and not allowBlog
                 or YAMLnotif.type == 'new-features' and not allowNewFeatures):
            return

        # check time validity
        today = date.today()
        if (YAMLnotif.start and YAMLnotif.start > today) or \
                (YAMLnotif.end and YAMLnotif.end < today):
            return

        # check requirements
        reqs = YAMLnotif.requirements
        # Orange/addons version
        if reqs and 'installed' in reqs and \
                not requirementsSatisfied(reqs['installed'], installed, req_type=Version):
            return
        # local config values
        if reqs and 'local_config' in reqs and \
                not requirementsSatisfied(reqs['local_config'], settings_dict):
            return

        # if no custom icon is set, default to notif type icon
        if YAMLnotif.icon is None and YAMLnotif.type is not None:
            YAMLnotif.icon = "canvas/icons/" + YAMLnotif.type + ".png"

        # instantiate and return Notification
        notif = YAMLnotif.toNotification()

        # connect link to notification
        notif.accepted.connect(lambda: open_link(QUrl(YAMLnotif.link)))

        # remember notification id
        def remember_notification(role):
            # if notification was accepted or rejected, write its ID to preferences
            if role == notif.DismissRole or YAMLnotif.id is None:
                return

            displayedIDs.add(YAMLnotif.id)
            settings.setValue("notifications/displayed", repr(displayedIDs))
        notif.clicked.connect(remember_notification)

        # display notification
        canvas.notification_server_instance.registerNotification(notif)

    def setup_notification_feed(feed_str):
        feed = yaml.safe_load(feed_str)
        for YAMLnotif in feed:
            parse_yaml_notification(YAMLnotif)

    thread.resultReady.connect(setup_notification_feed)
    thread.start()
    return thread


def send_usage_statistics():
    def send_statistics(url):
        """Send the statistics to the remote at `url`"""
        import json
        import requests
        settings = QSettings()
        if not settings.value("reporting/send-statistics", False,
                              type=bool):
            log.info("Not sending usage statistics (preferences setting).")
            return
        if not UsageStatistics.is_enabled():
            log.info("Not sending usage statistics (disabled).")
            return

        if settings.contains('reporting/machine-id'):
            machine_id = settings.value('reporting/machine-id')
        else:
            machine_id = str(uuid.uuid4())
            settings.setValue('reporting/machine-id', machine_id)

        is_anaconda = 'Continuum' in sys.version or 'conda' in sys.version

        data = UsageStatistics.load()
        for d in data:
            d["Orange Version"] = d.pop("Application Version", "")
            d["Anaconda"] = is_anaconda
            d["UUID"] = machine_id
        try:
            r = requests.post(url, files={'file': json.dumps(data)})
            if r.status_code != 200:
                log.warning("Error communicating with server while attempting to send "
                            "usage statistics. Status code %d", r.status_code)
                return
            # success - wipe statistics file
            log.info("Usage statistics sent.")
            with open(UsageStatistics.filename(), 'w', encoding="utf-8") as f:
                json.dump([], f)
        except (ConnectionError, requests.exceptions.RequestException):
            log.warning("Connection error while attempting to send usage statistics.")
        except Exception:  # pylint: disable=broad-except
            log.warning("Failed to send usage statistics.", exc_info=True)

    class SendUsageStatistics(QThread):
        def run(self):
            try:
                send_statistics(statistics_server_url)
            except Exception:  # pylint: disable=broad-except
                # exceptions in threads would crash Orange
                log.warning("Failed to send usage statistics.")

    thread = SendUsageStatistics()
    thread.start()
    return thread


# pylint: disable=too-many-locals,too-many-branches
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

    # set pyqtgraph colors
    def onPaletteChange():
        p = app.palette()
        bg = p.base().color().name()
        fg = p.windowText().color().name()

        log.info('Setting pyqtgraph background to %s', bg)
        pyqtgraph.setConfigOption('background', bg)
        log.info('Setting pyqtgraph foreground to %s', fg)
        pyqtgraph.setConfigOption('foreground', fg)

    app.paletteChanged.connect(onPaletteChange)
    onPaletteChange()

    palette = app.palette()
    if style is None and palette.color(QPalette.Window).value() < 127:
        log.info("Switching default stylesheet to darkorange")
        defaultstylesheet = "darkorange.qss"

    # Initialize SQL query and execution time logger (in SqlTable)
    sql_level = min(levels[options.log_level], logging.INFO)
    make_sql_logger(sql_level)

    clear_settings_flag = os.path.join(widget_settings_dir(), "DELETE_ON_START")

    if options.clear_widget_settings or \
            os.path.isfile(clear_settings_flag):
        log.info("Clearing widget settings")
        shutil.rmtree(widget_settings_dir(), ignore_errors=True)

    # Set http_proxy environment variables, after (potentially) clearing settings
    fix_set_proxy_env()

    # Setup file log handler for the select logger list - this is always
    # at least INFO
    level = min(levels[options.log_level], logging.INFO)
    file_handler = logging.FileHandler(
        filename=os.path.join(config.log_dir(), "canvas.log"),
        mode="w"
    )
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    stream = TextStream()
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    for namespace in ["orangecanvas", "orangewidget", "Orange"]:
        logger = logging.getLogger(namespace)
        logger.setLevel(level)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    # intercept any QFileOpenEvent requests until the main window is
    # fully initialized.
    # NOTE: The QApplication must have the executable ($0) and filename
    # arguments passed in argv otherwise the FileOpen events are
    # triggered for them (this is done by Cocoa, but QApplicaiton filters
    # them out if passed in argv)

    open_requests = []

    def onrequest(url):
        log.info("Received an file open request %s", url)
        path = url.path()
        exists = QFile(path).exists()
        if exists and \
                ('pydevd.py' not in url.path() and  # PyCharm debugger
                 'run_profiler.py' not in url.path()):  # PyCharm profiler
            open_requests.append(url)

    app.fileOpenRequest.connect(onrequest)

    settings = QSettings()
    settings.setValue('startup/launch-count', settings.value('startup/launch-count', 0, int) + 1)

    if settings.value("reporting/send-statistics", False, type=bool) \
            and is_release:
        UsageStatistics.set_enabled(True)

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

                if 'dark' in stylesheet:
                    app.setProperty('darkMode', True)

            else:
                log.info("%r style sheet not found.", stylesheet)

    # Add the default canvas_icons search path
    dirpath = os.path.abspath(os.path.dirname(orangecanvas.__file__))
    QDir.addSearchPath("canvas_icons", os.path.join(dirpath, "icons"))

    canvas_window = MainWindow()
    canvas_window.setAttribute(Qt.WA_DeleteOnClose)
    canvas_window.setWindowIcon(config.application_icon())
    canvas_window.connect_output_stream(stream)

    # initialize notification server, set to initial canvas
    notif_server = NotificationServer()
    canvas.notification_server_instance = notif_server
    canvas_window.set_notification_server(notif_server)

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

    if args:
        log.info("Loading a scheme from the command line argument %r",
                 args[0])
        canvas_window.load_scheme(args[0])
    elif open_requests:
        log.info("Loading a scheme from an `QFileOpenEvent` for %r",
                 open_requests[-1])
        canvas_window.load_scheme(open_requests[-1].toLocalFile())
    else:
        swp_loaded = canvas_window.ask_load_swp_if_exists()
        if not swp_loaded and want_welcome:
            canvas_window.welcome_dialog()

    # local references prevent destruction
    update_check = check_for_updates()
    send_stat = send_usage_statistics()
    pull_notifs = pull_notifications()

    # Tee stdout and stderr into Output dock
    log_view = canvas_window.output_view()

    stdout = TextStream()
    stdout.stream.connect(log_view.write)
    if sys.stdout:
        stdout.stream.connect(sys.stdout.write)
        stdout.flushed.connect(sys.stdout.flush)

    stderr = TextStream()
    error_writer = log_view.formatted(color=Qt.red)
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
             closing(stream), \
             patch('sys.excepthook', excepthook),\
             patch('sys.stderr', stderr),\
             patch('sys.stdout', stdout):
            status = app.exec()
    except BaseException:
        log.error("Error in main event loop.", exc_info=True)
        status = 42

    del canvas_window
    del update_check
    del send_stat
    del pull_notifs

    app.processEvents()
    app.flush()
    # Collect any cycles before deleting the QApplication instance
    gc.collect()

    del app

    if status == 96:
        log.info('Restarting via exit code 96.')
        run_after_exit([sys.executable, sys.argv[0]])

    return status


if __name__ == "__main__":
    sys.exit(main())
