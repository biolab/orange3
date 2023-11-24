from unittest import TestCase

from packaging.version import Version


class TestPkgResources(TestCase):
    def test_parse_version(self):
        self.assertGreater(Version('3.4.1'), Version('3.4.0'))
        self.assertGreater(Version('3.4.1'), Version('3.4.dev'))
        self.assertGreater(Version('3.4.1'), Version('3.4.1.dev'))
        self.assertLess(Version('3.4.1'), Version('3.4.2.dev'))
