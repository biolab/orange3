import codecs
from contextlib import closing
from itertools import groupby
import json
import logging
import platform
from urllib.parse import urlencode
from urllib.request import urlopen, build_opener
import uuid
from time import time

import pip

from AnyQt.QtCore import QSettings

try:
    from Orange.version import full_version as VERSION_STR
except ImportError:
    VERSION_STR = '???'

TELEMETRY_POST_URL = "http://127.0.0.1:8000/telemetry/v1/"
GEO_URL = "http://freegeoip.net/json/"

log = logging.getLogger()


class Telemetry():
    def __init__(self):
        self.schemes = {}
        self.scheme_id = 0
        self.start_time = time()
        self.last_time = time()
        self.end_time = 0

    def add_scheme(self, scheme):
        def canonicalize_dict(x):
            return sorted(x.items(), key=lambda x: hash(x[0]))

        def unique_and_count(lst):
            grouper = groupby(sorted(map(canonicalize_dict, lst)))
            return [dict(k + [("count", len(list(g)))]) for k, g in grouper]

        nodes = []
        for node in scheme.nodes:
            desc = node.description
            nodes.append({"name": desc.name,
                          "qualified_name": desc.qualified_name,
                         })

        self.schemes["scheme_{}".format(self.scheme_id)] = unique_and_count(nodes)
        now = time()
        self.schemes["scheme_duration_{}".format(self.scheme_id)] = now - self.last_time
        self.last_time = now
        self.scheme_id += 1

    def _orange(self):
        INSTALLED_PACKAGES = ', '.join(sorted("%s==%s" % (i.project_name, i.version)
                                              for i in pip.get_installed_distributions()))

        machine_id = QSettings().value('error-reporting/machine-id', '', type=str)

        ENVIRONMENT = 'Python {} on {} {} {} {}'.format(
            platform.python_version(), platform.system(), platform.release(),
            platform.version(), platform.machine())
        MACHINE_ID = machine_id or str(uuid.getnode())

        orange = {"version": VERSION_STR,
                  "environment": ENVIRONMENT,
                  "packages": INSTALLED_PACKAGES,
                  "machine_id": MACHINE_ID}
        return orange

    def _geo(self):
        geo = {}
        try:
            with closing(urlopen(GEO_URL)) as response:
                reader = codecs.getreader("utf-8")
                geo = json.load(reader(response))
        except Exception:
            pass
        return geo

    def _prepare_data(self):
        def merge_dicts(lst):
            z = lst[0].copy()
            for i in range(1, len(lst)):
                z.update(lst[i])
            return z
        return merge_dicts([{"start_time": self.start_time},
                            self._geo(),
                            self._orange(),
                            self.schemes,
                            {"end_time:": self.end_time}])

    def send(self):
        self.end_time = time()
        data = self._prepare_data()

        def _post_telemetry(data):
            try:
                opener = build_opener()
                u = opener.open(TELEMETRY_POST_URL)
                url = u.geturl()
                urlopen(url, timeout=10, data=urlencode(data).encode("utf8"))
            except Exception as e:
                e.__context__ = None
                log.exception("Telemetry failed.", exc_info=e)

        _post_telemetry(data)
