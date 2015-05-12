"""

"""
import os
import logging
import io

from urllib.parse import urljoin

from PyQt4.QtCore import QObject, QUrl

from PyQt4.QtNetwork import (
    QNetworkAccessManager, QNetworkDiskCache, QNetworkRequest, QNetworkReply
)


from .intersphinx import read_inventory_v1, read_inventory_v2

from .. import config

log = logging.getLogger(__name__)


class HelpProvider(QObject):
    def __init__(self, parent=None):
        QObject.__init__(self, parent)

    def search(self, description):
        raise NotImplementedError


class IntersphinxHelpProvider(HelpProvider):
    def __init__(self, parent=None, target=None, inventory=None):
        HelpProvider.__init__(self, parent)
        self.target = target

        if inventory is None:
            if is_url_local(self.target):
                inventory = os.path.join(self.target, "objects.inv")
            else:
                inventory = urljoin(self.target, "objects.inv")

        self.inventory = inventory

        self.islocal = bool(QUrl(inventory).toLocalFile())
        self.items = None

        self._fetch_inventory()

    def search(self, description):
        if description.help_ref:
            ref = description.help_ref
        else:
            ref = description.name

        if not self.islocal and not self._reply.isFinished():
            self._reply.waitForReadyRead(2000)

        if self.items is None:
            labels = {}
        else:
            labels = self.items.get("std:label", {})
        entry = labels.get(ref.lower(), None)
        if entry is not None:
            _, _, url, _ = entry
            return url
        else:
            raise KeyError(ref)

    def _fetch_inventory(self):
        cache_dir = config.cache_dir()
        cache_dir = os.path.join(cache_dir, "help", "intersphinx")

        try:
            os.makedirs(cache_dir)
        except OSError:
            pass

        url = QUrl(self.inventory)

        if not self.islocal:
            # fetch and cache the inventory file
            manager = QNetworkAccessManager(self)
            cache = QNetworkDiskCache()
            cache.setCacheDirectory(cache_dir)
            manager.setCache(cache)
            req = QNetworkRequest(url)

            self._reply = manager.get(req)
            manager.finished.connect(self._on_finished)
        else:
            self._load_inventory(open(str(url.toLocalFile()), "rb"))

    def _on_finished(self, reply):
        if reply.error() != QNetworkReply.NoError:
            log.error("An error occurred while fetching "
                      "intersphinx inventory {0!r}".format(self.inventory))

            self._error = reply.error(), reply.errorString()

        else:
            contents = bytes(reply.readAll())
            self._load_inventory(io.BytesIO(contents))

    def _load_inventory(self, stream):
        version = stream.readline().rstrip()
        if self.islocal:
            join = os.path.join
        else:
            join = urljoin

        if version == b"# Sphinx inventory version 1":
            items = read_inventory_v1(stream, self.target, join)
        elif version == b"# Sphinx inventory version 2":
            items = read_inventory_v2(stream, self.target, join)
        else:
            log.error("Invalid/unknown intersphinx inventory format.")
            self._error = (ValueError,
                           "{0} does not seem to be an intersphinx "
                           "inventory file".format(self.target))
            items = None

        self.items = items


def qurl_query_items(url):
    items = []
    for key, value in url.queryItems():
        items.append((str(key), str(value)))
    return items


def is_url_local(url):
    return bool(QUrl(url).toLocalFile())
