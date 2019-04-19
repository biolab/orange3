"""

"""
import sys
import os
import string
import itertools
import logging
import email
import urllib.parse

from distutils.version import StrictVersion

from operator import itemgetter
from sysconfig import get_path

import pkg_resources

from AnyQt.QtCore import QObject, QUrl, QDir, QT_VERSION

from . import provider


log = logging.getLogger(__name__)


class HelpManager(QObject):
    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self._registry = None
        self._initialized = False
        self._providers = {}

    def set_registry(self, registry):
        """
        Set the widget registry for which the manager should
        provide help.

        """
        if self._registry is not registry:
            self._registry = registry
            self._initialized = False
            self.initialize()

    def registry(self):
        """
        Return the previously set with set_registry.
        """
        return self._registry

    def initialize(self):
        if self._initialized:
            return

        reg = self._registry
        all_projects = set(desc.project_name for desc in reg.widgets())

        providers = []
        for project in set(all_projects) - set(self._providers.keys()):
            provider = None
            try:
                dist = pkg_resources.get_distribution(project)
                provider = get_help_provider_for_distribution(dist)
            except Exception:
                log.exception("Error while initializing help "
                              "provider for %r", project)

            if provider:
                providers.append((project, provider))
                provider.setParent(self)

        self._providers.update(dict(providers))
        self._initialized = True

    def get_help(self, url):
        """
        """
        self.initialize()
        if url.scheme() == "help" and url.authority() == "search":
            return self.search(qurl_query_items(url))
        else:
            return url

    def description_by_id(self, desc_id):
        reg = self._registry
        return get_by_id(reg, desc_id)

    def search(self, query):
        self.initialize()

        if isinstance(query, QUrl):
            query = qurl_query_items(query)

        query = dict(query)
        desc_id = query["id"]
        desc = self.description_by_id(desc_id)

        provider = None
        if desc.project_name:
            provider = self._providers.get(desc.project_name)

        # TODO: Ensure initialization of the provider
        if provider:
            return provider.search(desc)
        else:
            raise KeyError(desc_id)


def get_by_id(registry, descriptor_id):
    for desc in registry.widgets():
        if desc.qualified_name == descriptor_id:
            return desc

    raise KeyError(descriptor_id)


if QT_VERSION < 0x50000:
    def qurl_query_items(url):
        items = []
        for key, value in url.queryItems():
            items.append((str(key), str(value)))
        return items
else:
    # QUrl has no queryItems
    def qurl_query_items(url):
        if not url.hasQuery():
            return []
        querystr = url.query()
        return urllib.parse.parse_qsl(querystr)


def get_help_provider_for_description(desc):
    if desc.project_name:
        dist = pkg_resources.get_distribution(desc.project_name)
        return get_help_provider_for_distribution(dist)


def is_develop_egg(dist):
    """
    Is the distribution installed in development mode (setup.py develop)
    """
    meta_provider = dist._provider
    egg_info_dir = os.path.dirname(meta_provider.egg_info)
    egg_name = pkg_resources.to_filename(dist.project_name)
    return meta_provider.egg_info.endswith(egg_name + ".egg-info") \
           and os.path.exists(os.path.join(egg_info_dir, "setup.py"))


def left_trim_lines(lines):
    """
    Remove all unnecessary leading space from lines.
    """
    lines_striped = list(zip(lines[1:], list(map(str.lstrip, lines[1:]))))
    lines_striped = list(filter(itemgetter(1), lines_striped))
    indent = min([len(line) - len(striped) \
                  for line, striped in lines_striped] + [sys.maxsize])

    if indent < sys.maxsize:
        return [line[indent:] for line in lines]
    else:
        return list(lines)


def trim_trailing_lines(lines):
    """
    Trim trailing blank lines.
    """
    lines = list(lines)
    while lines and not lines[-1]:
        lines.pop(-1)
    return lines


def trim_leading_lines(lines):
    """
    Trim leading blank lines.
    """
    lines = list(lines)
    while lines and not lines[0]:
        lines.pop(0)
    return lines


def trim(string):
    """
    Trim a string in PEP-256 compatible way
    """
    lines = string.expandtabs().splitlines()

    lines = list(map(str.lstrip, lines[:1])) + left_trim_lines(lines[1:])

    return  "\n".join(trim_leading_lines(trim_trailing_lines(lines)))


# Fields allowing multiple use (from PEP-0345)
MULTIPLE_KEYS = ["Platform", "Supported-Platform", "Classifier",
                 "Requires-Dist", "Provides-Dist", "Obsoletes-Dist",
                 "Project-URL"]


def parse_meta(contents):
    message = email.message_from_string(contents)
    meta = {}
    for key in set(message.keys()):
        if key in MULTIPLE_KEYS:
            meta[key] = message.get_all(key)
        else:
            meta[key] = message.get(key)

    version = StrictVersion(meta["Metadata-Version"])

    if version >= StrictVersion("1.3") and "Description" not in meta:
        desc = message.get_payload()
        if desc:
            meta["Description"] = desc
    return meta


def get_meta_entry(dist, name):
    """
    Get the contents of the named entry from the distributions PKG-INFO file
    """
    meta = get_dist_meta(dist)
    return meta.get(name)


def get_dist_url(dist):
    """
    Return the 'url' of the distribution (as passed to setup function)
    """
    return get_meta_entry(dist, "Home-page")


def get_dist_meta(dist):
    if dist.has_metadata("PKG-INFO"):
        # egg-info
        contents = dist.get_metadata("PKG-INFO")
    elif dist.has_metadata("METADATA"):
        # dist-info
        contents = dist.get_metadata("METADATA")
    else:
        contents = None

    if contents is not None:
        return parse_meta(contents)
    else:
        return {}


def _replacements_for_dist(dist):
    replacements = {"PROJECT_NAME": dist.project_name,
                    "PROJECT_NAME_LOWER": dist.project_name.lower(),
                    "PROJECT_VERSION": dist.version,
                    "DATA_DIR": get_path("data")}
    try:
        replacements["URL"] = get_dist_url(dist)
    except KeyError:
        pass

    if is_develop_egg(dist):
        replacements["DEVELOP_ROOT"] = dist.location

    return replacements


def qurl_from_path(urlpath):
    if QDir(urlpath).isAbsolute():
        # deal with absolute paths including windows drive letters
        return QUrl.fromLocalFile(urlpath)
    return QUrl(urlpath, QUrl.TolerantMode)


def create_intersphinx_provider(entry_point):
    locations = entry_point.resolve()
    replacements = _replacements_for_dist(entry_point.dist)

    formatter = string.Formatter()

    for target, inventory in locations:
        # Extract all format fields
        format_iter = formatter.parse(target)
        if inventory:
            format_iter = itertools.chain(format_iter,
                                          formatter.parse(inventory))
        # Names used in both target and inventory
        fields = {name for _, name, _, _ in format_iter if name}

        if not set(fields) <= set(replacements.keys()):
            log.warning("Invalid replacement fields %s",
                        set(fields) - set(replacements.keys()))
            continue

        target = formatter.format(target, **replacements)
        if inventory:
            inventory = formatter.format(inventory, **replacements)

        targeturl = qurl_from_path(target)
        if not targeturl.isValid():
            continue

        if targeturl.isLocalFile():
            if os.path.exists(os.path.join(target, "objects.inv")):
                inventory = QUrl.fromLocalFile(
                    os.path.join(target, "objects.inv"))
            else:
                log.info("Local doc root '%s' does not exist.", target)
                continue

        else:
            if not inventory:
                # Default inventory location
                inventory = targeturl.resolved(QUrl("objects.inv"))

        if inventory is not None:
            return provider.IntersphinxHelpProvider(
                inventory=inventory, target=target)
    return None


def create_html_provider(entry_point):
    locations = entry_point.resolve()
    replacements = _replacements_for_dist(entry_point.dist)

    formatter = string.Formatter()

    for target in locations:
        # Extract all format fields
        format_iter = formatter.parse(target)
        fields = {name for _, name, _, _ in format_iter if name}

        if not set(fields) <= set(replacements.keys()):
            log.warning("Invalid replacement fields %s",
                        set(fields) - set(replacements.keys()))
            continue
        target = formatter.format(target, **replacements)

        targeturl = qurl_from_path(target)
        if not targeturl.isValid():
            continue

        if targeturl.isLocalFile():
            if not os.path.exists(target):
                log.info("Local doc root '%s' does not exist.", target)
                continue

        if target:
            return provider.SimpleHelpProvider(
                baseurl=QUrl.fromLocalFile(target))

    return None


def create_html_inventory_provider(entry_point):
    locations = entry_point.resolve()
    replacements = _replacements_for_dist(entry_point.dist)

    formatter = string.Formatter()

    for target, xpathquery in locations:
        if isinstance(target, (tuple, list)):
            pass

        # Extract all format fields
        format_iter = formatter.parse(target)
        fields = {name for _, name, _, _ in format_iter if name}

        if not set(fields) <= set(replacements.keys()):
            log.warning("Invalid replacement fields %s",
                        set(fields) - set(replacements.keys()))
            continue

        target = formatter.format(target, **replacements)

        targeturl = qurl_from_path(target)
        if not targeturl.isValid():
            continue

        if targeturl.isLocalFile():
            if not os.path.exists(target):
                log.info("Local doc root '%s' does not exist", target)
                continue

            inventory = QUrl.fromLocalFile(target)
        else:
            inventory = QUrl(target)

        return provider.HtmlIndexProvider(
            inventory=inventory, xpathquery=xpathquery)

    return None

_providers = {
    "intersphinx": create_intersphinx_provider,
    "html-simple": create_html_provider,
    "html-index": create_html_inventory_provider,
}


def get_help_provider_for_distribution(dist):
    entry_points = dist.get_entry_map().get("orange.canvas.help", {})
    provider = None
    for name, entry_point in list(entry_points.items()):
        create = _providers.get(name, None)
        if create:
            try:
                provider = create(entry_point)
            except pkg_resources.DistributionNotFound as err:
                log.warning("Unsatisfied dependencies (%r)", err)
                continue
            except Exception as ex:
                log.exception("Exception {}".format(ex))
            if provider:
                log.info("Created %s provider for %s",
                         type(provider), dist)
                break

    return provider
