"""
Orange Canvas Configuration

"""
import os
import sys
import itertools
from distutils.version import LooseVersion

from typing import Dict, Any, Optional, Iterable, List

import pkg_resources
import requests

from AnyQt.QtGui import QPainter, QFont, QFontMetrics, QColor, QPixmap, QIcon
from AnyQt.QtCore import Qt, QPoint, QRect

from orangecanvas import config

from . import discovery
from . import widgetsscheme

import Orange

# generated from biolab/orange3-addons repository
OFFICIAL_ADDON_LIST = "https://orange.biolab.si/addons/list"

WIDGETS_ENTRY = "orange.widgets"
ADDONS_ENTRY = "orange3.addon"


class Config(config.Config):
    """
    Orange application config
    """
    OrganizationDomain = "biolab.si"
    ApplicationName = "Orange Canvas"
    ApplicationVersion = Orange.__version__

    @staticmethod
    def application_icon():
        """
        Return the main application icon.
        """
        path = pkg_resources.resource_filename(
            __name__, "icons/orange-canvas.svg"
        )
        return QIcon(path)

    @staticmethod
    def splash_screen():

        path = pkg_resources.resource_filename(
            __name__, "icons/orange-splash-screen.png")
        pm = QPixmap(path)

        version = Config.ApplicationVersion
        if version:
            version_parsed = LooseVersion(version)
            version_comp = version_parsed.version
            version = ".".join(map(str, version_comp[:2]))
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

    @staticmethod
    def widgets_entry_points():
        """
        Return an `EntryPoint` iterator for all 'orange.widget' entry
        points plus the default Orange Widgets.
        """
        # This could also be achieved by declaring the entry point in
        # Orange's setup.py, but that would not guaranty this entry point
        # is the first in a list.
        dist = pkg_resources.get_distribution("Orange3")
        default_ep = pkg_resources.EntryPoint(
            "Orange Widgets", "Orange.widgets", dist=dist)
        return itertools.chain(
            (default_ep,), pkg_resources.iter_entry_points(WIDGETS_ENTRY))

    @staticmethod
    def addon_entry_points():
        return Config.widgets_entry_points()

    @staticmethod
    def addon_defaults_list(session=None):
        # type: (Optional[requests.Session]) -> List[Dict[str, Any]]
        """
        Return a list of available add-ons.
        """
        if session is None:
            session = requests.Session()
        return session.get(OFFICIAL_ADDON_LIST).json()

    @staticmethod
    def core_packages():
        # type: () -> List[str]
        """
        Return a list of 'core packages'

        These packages constitute required the application framework. They
        cannot be removes via the 'Add-on/plugins' manager. They however can
        be updated. The package that defines the application's `main()` entry
        point must always be in this list.
        """
        return ["Orange3 >=3.20,<4.0a"]

    @staticmethod
    def examples_entry_points():
        # type: () -> Iterable[pkg_resources.EntryPoint]
        """
        Return an iterator over the entry points yielding 'Example Workflows'
        """
        # `iter_entry_points` yields them in unspecified order, so we insert
        # our first
        default_ep = pkg_resources.EntryPoint(
            "Orange3", "Orange.canvas.workflows",
            dist=pkg_resources.get_distribution("Orange3"))

        return itertools.chain(
            (default_ep,),
            pkg_resources.iter_entry_points("orange.widgets.tutorials")
        )

    widget_discovery = discovery.WidgetDiscovery
    workflow_constructor = widgetsscheme.WidgetsScheme

    APPLICATION_URLS = {
        #: Submit a bug report action in the Help menu
        "Bug Report": "https://github.com/biolab/orange3/issues",
        #: A url quick tour/getting started url
        "Quick Start": "https://orange.biolab.si/getting-started/",
        #: The 'full' documentation, should be something like current /docs/
        #: but specific for 'Visual Programing' only
        "Documentation": "https://orange.biolab.si/toolbox/",
        #: YouTube tutorials
        "Screencasts":
            "https://www.youtube.com/watch"
            "?v=HXjnDIgGDuI&list=PLmNPvQr9Tf-ZSDLwOzxpvY-HrE0yv-8Fy&index=1",
        #: Used for 'Submit Feedback' action in the help menu
        "Feedback": "https://orange.biolab.si/survey/long.html",
    }


def init():
    # left for backwards compatibility
    raise RuntimeError("This is not the init you are looking for.")


def data_dir():
    """
    Return the application data directory. If the directory path
    does not yet exists then create it.
    """

    from Orange.misc import environ
    path = os.path.join(environ.data_dir(), "canvas")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass
    return path


def cache_dir():
    """Return the application cache directory. If the directory path
    does not yet exists then create it.

    """
    from Orange.misc import environ
    path = os.path.join(environ.cache_dir(), "canvas")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass
    return path


def log_dir():
    """
    Return the application log directory.
    """
    if sys.platform == "darwin":
        name = Config.ApplicationName
        logdir = os.path.join(os.path.expanduser("~/Library/Logs"), name)
    else:
        logdir = data_dir()

    try:
        os.makedirs(logdir, exist_ok=True)
    except OSError:
        pass
    return logdir


def widget_settings_dir():
    """
    Return the widget settings directory.
    """
    from Orange.misc import environ
    return environ.widget_settings_dir()


def widgets_entry_points():
    return Config.widgets_entry_points()


def splash_screen():
    return Config.splash_screen()


def application_icon():
    return Config.application_icon()
