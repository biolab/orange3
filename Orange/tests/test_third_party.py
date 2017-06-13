from unittest import TestCase

from pkg_resources import parse_version


class TestPkgResources(TestCase):
    def test_parse_version(self):
        self.assertGreater(parse_version('3.4.1'), parse_version('3.4.0'))
        self.assertGreater(parse_version('3.4.1'), parse_version('3.4.dev'))
        self.assertGreater(parse_version('3.4.0'), parse_version('3.4~1'))
