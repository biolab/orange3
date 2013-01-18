"""

"""
import sys
import os
import string
import itertools
import logging

from operator import itemgetter

import pkg_resources

from .provider import IntersphinxHelpProvider

from PyQt4.QtCore import QObject, QUrl

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
                              "provider for %r", desc.project_name)

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
        if desc.id == descriptor_id:
            return desc

    raise KeyError(descriptor_id)


def qurl_query_items(url):
    items = []
    for key, value in url.queryItems():
        items.append((str(key), str(value)))
    return items


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
    lines_striped = list(zip(lines[1:], list(map(string.lstrip, lines[1:]))))
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


def parse_pkg_info(contents):
    lines = contents.expandtabs().splitlines()
    parsed = {}
    current_block = None
    for line in lines:
        if line.startswith(" "):
            parsed[current_block].append(line)
        elif line.strip():
            current_block, block_contents = line.split(": ", 1)
            if current_block == "Classifier":
                if current_block not in parsed:
                    parsed[current_block] = [trim(block_contents)]
                else:
                    parsed[current_block].append(trim(block_contents))
            else:
                parsed[current_block] = [block_contents]

    for key, val in list(parsed.items()):
        if key != "Classifier":
            parsed[key] = trim("\n".join(val))

    return parsed


def get_pkg_info_entry(dist, name):
    """
    Get the contents of the named entry from the distributions PKG-INFO file
    """
    pkg_info = parse_pkg_info(dist.get_metadata("PKG-INFO"))
    return pkg_info[name]


def get_dist_url(dist):
    """
    Return the 'url' of the distribution (as passed to setup function)
    """
    return get_pkg_info_entry(dist, "Home-page")


def create_intersphinx_provider(entry_point):
    locations = entry_point.load()
    dist = entry_point.dist

    replacements = {"PROJECT_NAME": dist.project_name,
                    "PROJECT_NAME_LOWER": dist.project_name.lower(),
                    "PROJECT_VERSION": dist.version}
    try:
        replacements["URL"] = get_dist_url(dist)
    except KeyError:
        pass

    formatter = string.Formatter()

    for target, inventory in locations:
        # Extract all format fields
        format_iter = formatter.parse(target)
        if inventory:
            format_iter = itertools.chain(format_iter,
                                          formatter.parse(inventory))
        fields = list(map(itemgetter(1), format_iter))
        fields = [_f for _f in set(fields) if _f]

        if "DEVELOP_ROOT" in fields and is_develop_egg(dist):
            target = formatter.format(target, DEVELOP_ROOT=dist.location)

            if os.path.exists(target) and \
                    os.path.exists(os.path.join(target, "objects.inv")):
                return IntersphinxHelpProvider(target=target)
            else:
                continue
        elif fields:
            try:
                target = formatter.format(target, **replacements)
                if inventory:
                    inventory = formatter.format(inventory, **replacements)
            except KeyError:
                log.exception("Error while formating intersphinx url.")
                continue

            return IntersphinxHelpProvider(target=target, inventory=inventory)
        else:
            return IntersphinxHelpProvider(target=target, inventory=inventory)

    return None


_providers = {"intersphinx": create_intersphinx_provider}


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
            except Exception:
                log.exception("Exception")
            if provider:
                log.info("Created %s provider for %s",
                         type(provider), dist)
                break

    return provider
