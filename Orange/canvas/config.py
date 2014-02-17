"""
Orange Canvas Configuration

"""

import os
import sys
import logging
import pickle as pickle
import itertools

import pkg_resources

from PyQt4.QtGui import (
    QDesktopServices, QPainter, QFont, QFontMetrics, QColor, QPixmap
)

from PyQt4.QtCore import Qt, QCoreApplication, QPoint, QRect

from .utils.settings import Settings, config_slot

# Import QSettings from qtcompat module (compatibility with PyQt < 4.8.3
from .utils.qtcompat import QSettings

log = logging.getLogger(__name__)


def init():
    """
    Initialize the QCoreApplication.organizationDomain, applicationName,
    applicationVersion and the default settings format. Will only run once.

    .. note:: This should not be run before QApplication has been initialized.
              Otherwise it can break Qt's plugin search paths.

    """
    dist = pkg_resources.get_distribution("Orange")
    version = dist.version
    # Use only major.minor
    version = ".".join(version.split(".", 2)[:2])

    QCoreApplication.setOrganizationDomain("biolab.si")
    QCoreApplication.setApplicationName("Orange Canvas")
    QCoreApplication.setApplicationVersion(version)
    QSettings.setDefaultFormat(QSettings.IniFormat)

    # Make it a null op.
    global init
    init = lambda: None

rc = {}


spec = \
    [("startup/show-splash-screen", bool, True,
      "Show splash screen at startup"),

     ("startup/show-welcome-screen", bool, True,
      "Show Welcome screen at startup"),

     ("stylesheet", str, "orange",
      "QSS stylesheet to use"),

     ("schemeinfo/show-at-new-scheme", bool, True,
      "Show Scheme Properties when creating a new Scheme"),

     ("mainwindow/scheme-margins-enabled", bool, False,
      "Show margins around the scheme view"),

     ("mainwindow/show-scheme-shadow", bool, True,
      "Show shadow around the scheme view"),

     ("mainwindow/toolbox-dock-exclusive", bool, True,
      "Should the toolbox show only one expanded category at the time"),

     ("mainwindow/toolbox-dock-floatable", bool, False,
      "Is the canvas toolbox floatable (detachable from the main window)"),

     ("mainwindow/toolbox-dock-movable", bool, True,
      "Is the canvas toolbox movable (between left and right edge)"),

     ("mainwindow/toolbox-dock-use-popover-menu", bool, True,
      "Use a popover menu to select a widget when clicking on a category "
      "button"),

     ("mainwindow/number-of-recent-schemes", int, 15,
      "Number of recent schemes to keep in history"),

     ("schemeedit/show-channel-names", bool, True,
      "Show channel names"),

     ("schemeedit/show-link-state", bool, True,
      "Show link state hints."),

     ("schemeedit/enable-node-animations", bool, True,
      "Enable node animations."),

     ("schemeedit/freeze-on-load", bool, False,
      "Freeze signal propagation when loading a scheme."),

     ("quickmenu/trigger-on-double-click", bool, True,
      "Show quick menu on double click."),

     ("quickmenu/trigger-on-right-click", bool, True,
      "Show quick menu on right click."),

     ("quickmenu/trigger-on-space-key", bool, True,
      "Show quick menu on space key press."),

     ("quickmenu/trigger-on-any-key", bool, False,
      "Show quick menu on double click."),

     ("logging/level", int, 1, "Logging level"),

     ("logging/show-on-error", bool, True, "Show log window on error"),

     ("logging/dockable", bool, True, "Allow log window to be docked"),

     ("output/redirect-stderr", bool, True,
      "Redirect and display standard error output"),

     ("output/redirect-stdout", bool, True,
      "Redirect and display standard output"),

     ("output/stay-on-top", bool, True, ""),

     ("output/show-on-error", bool, True, "Show output window on error"),

     ("output/dockable", bool, True, "Allow output window to be docked"),

     ("help/stay-on-top", bool, True, ""),

     ("help/dockable", bool, True, "Allow help window to be docked"),

     ("help/open-in-external-browser", bool, False,
      "Open help in an external browser")
     ]

spec = [config_slot(*t) for t in spec]


def settings():
    init()
    store = QSettings()
    settings = Settings(defaults=spec, store=store)
    return settings


def data_dir():
    """Return the application data directory. If the directory path
    does not yet exists then create it.

    """
    import Orange
    init()
    datadir = QDesktopServices.storageLocation(QDesktopServices.DataLocation)
    datadir = str(datadir)
    datadir = os.path.join(datadir, Orange.__version__)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    return datadir


def cache_dir():
    """Return the application cache directory. If the directory path
    does not yet exists then create it.

    """
    import Orange
    init()
    cachedir = QDesktopServices.storageLocation(QDesktopServices.CacheLocation)
    cachedir = str(cachedir)
    cachedir = os.path.join(cachedir, Orange.__version__)
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)
    return cachedir


def log_dir():
    """
    Return the application log directory.
    """
    init()
    if sys.platform == "darwin":
        name = str(QCoreApplication.applicationName())
        logdir = os.path.join(os.path.expanduser("~/Library/Logs"), name)
    else:
        logdir = data_dir()

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir


def open_config():
    global rc
    app_dir = data_dir()
    filename = os.path.join(app_dir, "canvas-rc.pck")
    if os.path.exists(filename):
        with open(os.path.join(app_dir, "canvas-rc.pck"), "rb") as f:
            rc.update(pickle.load(f))


def save_config():
    app_dir = data_dir()
    with open(os.path.join(app_dir, "canvas-rc.pck"), "wb") as f:
        pickle.dump(rc, f)


def recent_schemes():
    """Return a list of recently accessed schemes.
    """
    app_dir = data_dir()
    recent_filename = os.path.join(app_dir, "recent.pck")
    recent = []
    if os.path.isdir(app_dir) and os.path.isfile(recent_filename):
        with open(recent_filename, "rb") as f:
            recent = pickle.load(f)

    # Filter out files not found on the file system
    recent = [(title, path) for title, path in recent \
              if os.path.exists(path)]
    return recent


def save_recent_scheme_list(scheme_list):
    """Save the list of recently accessed schemes
    """
    app_dir = data_dir()
    recent_filename = os.path.join(app_dir, "recent.pck")

    if os.path.isdir(app_dir):
        with open(recent_filename, "wb") as f:
            pickle.dump(scheme_list, f)


WIDGETS_ENTRY = "orange.widgets"


# This could also be achieved by declaring the entry point in
# Orange's setup.py, but that would not guaranty this entry point
# is the first in a list.

def default_entry_point():
    """
    Return a default orange.widgets entry point for loading
    default Orange Widgets.

    """
    dist = pkg_resources.get_distribution("Orange")
    ep = pkg_resources.EntryPoint("Orange Widgets", "Orange.widgets",
                                  dist=dist)
    return ep


def widgets_entry_points():
    """
    Return an `EntryPoint` iterator for all 'orange.widget' entry
    points plus the default Orange Widgets.

    """
    ep_iter = pkg_resources.iter_entry_points(WIDGETS_ENTRY)
    chain = [[default_entry_point()],
             ep_iter
             ]
    return itertools.chain(*chain)


def splash_screen():
    """
    """
    pm = QPixmap(
        pkg_resources.resource_filename(
            __name__, "icons/orange-splash-screen.png")
    )

    version = QCoreApplication.applicationVersion()
    size = 21 if len(version) < 5 else 16
    font = QFont("Helvetica")
    font.setPixelSize(size)
    font.setBold(True)
    font.setItalic(True)
    font.setLetterSpacing(QFont.AbsoluteSpacing, 2)
    metrics = QFontMetrics(font)
    br = metrics.boundingRect(version).adjusted(-5, 0, 5, 0)
    br.moveCenter(QPoint(436, 224))

    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)
    p.setFont(font)
    p.setPen(QColor("#231F20"))
    p.drawText(br, Qt.AlignCenter, version)
    p.end()
    return pm, QRect(88, 193, 200, 20)
