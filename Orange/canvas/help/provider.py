"""

"""
import os
import logging
import io

from urllib.parse import urljoin

from html import parser
from xml.etree.ElementTree import TreeBuilder, Element

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


class BaseInventoryProvider(HelpProvider):
    def __init__(self, inventory, parent=None):
        super().__init__(parent)
        self.inventory = QUrl(inventory)

        if not self.inventory.scheme() and not self.inventory.isEmpty():
            self.inventory.setScheme("file")

        self._error = None
        self._fetch_inventory(self.inventory)

    def _fetch_inventory(self, url):
        cache_dir = config.cache_dir()
        cache_dir = os.path.join(cache_dir, "help", type(self).__qualname__)

        try:
            os.makedirs(cache_dir)
        except OSError:
            pass

        url = QUrl(self.inventory)
        if not url.isLocalFile():
            # fetch and cache the inventory file.
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
                      "help inventory '{0}'".format(self.inventory))
            self._error = reply.error(), reply.errorString()

        else:
            contents = bytes(reply.readAll())
            self._load_inventory(io.BytesIO(contents))

    def _load_inventory(self, stream):
        raise NotImplementedError()


class IntersphinxHelpProvider(BaseInventoryProvider):
    def __init__(self, inventory, target=None, parent=None):
        self.target = target
        self.items = None
        super().__init__(inventory, parent)

    def search(self, description):
        if description.help_ref:
            ref = description.help_ref
        else:
            ref = description.name

        if not self.inventory.isLocalFile() and not self._reply.isFinished():
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

    def _load_inventory(self, stream):
        version = stream.readline().rstrip()
        if self.inventory.isLocalFile():
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


class SimpleHelpProvider(HelpProvider):
    def __init__(self, parent=None, baseurl=None):
        super().__init__(parent)
        self.baseurl = baseurl

    def search(self, description):
        if description.help_ref:
            ref = description.help_ref
        else:
            raise KeyError()

        url = QUrl(self.baseurl).resolved(QUrl(ref))
        if url.isLocalFile():
            path = url.toLocalFile()
            fragment = url.fragment()
            if os.path.isfile(path):
                return url
            elif os.path.isfile("{}.html".format(path)):
                url = QUrl.fromLocalFile("{}.html".format(path))
                url.setFragment(fragment)
                return url
            elif os.path.isdir(path) and \
                    os.path.isfile(os.path.join(path, "index.html")):
                url = QUrl.fromLocalFile(os.path.join(path, "index.html"))
                url.setFragment(fragment)
                return url
            else:
                raise KeyError()
        else:
            if url.scheme() in ["http", "https"]:
                path = url.path()
                if not (path.endswith(".html") or path.endswith("/")):
                    url.setPath(path + ".html")
        return url


class HtmlIndexProvider(BaseInventoryProvider):
    """
    Provide help links from an html help index page.
    """
    class _XHTMLParser(parser.HTMLParser):
        # A helper class for parsing XHTML into an xml.etree.ElementTree
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.builder = TreeBuilder(element_factory=Element)

        def handle_starttag(self, tag, attrs):
            self.builder.start(tag, dict(attrs),)

        def handle_endtag(self, tag):
            self.builder.end(tag)

        def handle_data(self, data):
            self.builder.data(data)

    def __init__(self, inventory, parent=None, xpathquery=None):
        self.root = None
        self.items = {}
        self.xpathquery = xpathquery

        super().__init__(inventory, parent)

    def _load_inventory(self, stream):
        contents = io.TextIOWrapper(stream, encoding="utf-8").read()
        try:
            self.items = self._parse(contents)
        except Exception:
            log.exception("Error parsing")

    def _parse(self, stream):
        parser = HtmlIndexProvider._XHTMLParser(
            strict=True, convert_charrefs=True)
        parser.feed(stream)
        self.root = parser.builder.close()

        path = self.xpathquery or ".//div[@id='widgets']//li/a"

        items = {}
        for el in self.root.findall(path):
            href = el.attrib.get("href", None)
            name = el.text.lower()
            items[name] = href

        if not items:
            log.warning("No help references found. Wrong configuration??")
        return items

    def search(self, desc):
        if not self.inventory.isLocalFile() and not self._reply.isFinished():
            self._reply.waitForReadyRead(2000)

        if self.items is None:
            labels = {}
        else:
            labels = self.items

        entry = labels.get(desc.name.lower(), None)
        if entry is not None:
            return self.inventory.resolved(QUrl(entry))
        else:
            raise KeyError()


def qurl_query_items(url):
    items = []
    for key, value in url.queryItems():
        items.append((str(key), str(value)))
    return items


def is_url_local(url):
    return bool(QUrl(url).toLocalFile())
