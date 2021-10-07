"""
Orange Canvas main entry point

"""
import argparse
import uuid
import os
import sys
import time
import logging
import signal
import shutil

from logging.handlers import RotatingFileHandler
from collections import defaultdict
from datetime import date
from urllib.request import urlopen, Request

import pkg_resources
import yaml

from AnyQt.QtGui import QColor, QDesktopServices, QIcon, QPalette
from AnyQt.QtCore import QUrl, QSettings, QThread, pyqtSignal

import pyqtgraph

from orangecanvas import config as canvasconfig
from orangecanvas.application.outputview import ExceptHook
from orangecanvas.document.usagestatistics import UsageStatistics
from orangecanvas.utils.overlay import Notification, NotificationServer
from orangecanvas.main import Main

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


class OMain(Main):
    DefaultConfig = "Orange.canvas.config.Config"

    def run(self, argv):
        # Allow termination with CTRL + C
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        # Disable pyqtgraph's atexit and QApplication.aboutToQuit cleanup handlers.
        pyqtgraph.setConfigOption("exitCleanup", False)
        super().run(argv)

    def argument_parser(self) -> argparse.ArgumentParser:
        parser = super().argument_parser()
        parser.add_argument(
            "--clear-widget-settings", action="store_true",
            help="Clear stored widget setting/defaults",
        )
        return parser

    def setup_logging(self):
        super().setup_logging()
        make_sql_logger(self.options.log_level)

    def setup_application(self):
        super().setup_application()
        clear_settings_flag = os.path.join(widget_settings_dir(),
                                           "DELETE_ON_START")
        # NOTE: No OWWidgetBase subclass should be imported before this
        options = self.options
        if options.clear_widget_settings or \
                os.path.isfile(clear_settings_flag):
            log.info("Clearing widget settings")
            shutil.rmtree(widget_settings_dir(), ignore_errors=True)

        notif_server = NotificationServer()
        canvas.notification_server_instance = notif_server

        self._update_check = check_for_updates()
        self._send_stat = send_usage_statistics()
        self._pull_notifs = pull_notifications()

        settings = QSettings()
        settings.setValue('startup/launch-count',
                          settings.value('startup/launch-count', 0, int) + 1)

        if settings.value("reporting/send-statistics", False, type=bool) \
                and is_release:
            UsageStatistics.set_enabled(True)

        app = self.application

        # set pyqtgraph colors
        def onPaletteChange():
            p = app.palette()
            bg = p.base().color().name()
            fg = p.windowText().color().name()

            log.info('Setting pyqtgraph background to %s', bg)
            pyqtgraph.setConfigOption('background', bg)
            log.info('Setting pyqtgraph foreground to %s', fg)
            pyqtgraph.setConfigOption('foreground', fg)
            app.setProperty('darkMode', p.color(QPalette.Base).value() < 127)

        app.paletteChanged.connect(onPaletteChange)
        onPaletteChange()

    def show_splash_message(self, message: str, color=QColor("#FFD39F")):
        super().show_splash_message(message, color)

    def create_main_window(self):
        window = MainWindow()
        window.set_notification_server(canvas.notification_server_instance)
        return window

    def setup_sys_redirections(self):
        super().setup_sys_redirections()
        if isinstance(sys.excepthook, ExceptHook):
            sys.excepthook.handledException.connect(handle_exception)

    def tear_down_sys_redirections(self):
        if isinstance(sys.excepthook, ExceptHook):
            sys.excepthook.handledException.disconnect(handle_exception)
        super().tear_down_sys_redirections()


def main(argv=None):
    return OMain().run(argv)


if __name__ == "__main__":
    sys.exit(main())
