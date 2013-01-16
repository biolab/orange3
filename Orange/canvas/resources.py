"""
Orange Canvas Resource Loader

"""

import os
import logging

log = logging.getLogger(__name__)


def package_dirname(package):
    """Return the directory path where package is located.

    """
    if isinstance(package, basestring):
        package = __import__(package, fromlist=[""])
    filename = package.__file__
    dirname = os.path.dirname(filename)
    return dirname


def package(qualified_name):
    """Return the enclosing package name where qualified_name is located.

    `qualified_name` can be a module inside the package or even an object
    inside the module. If a package name itself is provided it is returned.

    """
    try:
        module = __import__(qualified_name, fromlist=[""])
    except ImportError:
        # qualified_name could name an object inside a module/package
        if "." in qualified_name:
            qualified_name, attr_name = qualified_name.rsplit(".", 1)
            module = __import__(qualified_name, fromlist=[attr_name])
        else:
            raise

    if module.__package__ is not None:
        # the modules enclosing package
        return module.__package__
    else:
        # 'qualified_name' is itself the package
        assert(module.__name__ == qualified_name)
        return qualified_name

dirname = os.path.abspath(os.path.dirname(__file__))

DEFAULT_SEARCH_PATHS = \
    [("", dirname),
     ("", os.path.join(dirname, "../OrangeWidgets"))]

del dirname


def default_search_paths():
    return DEFAULT_SEARCH_PATHS


def add_default_search_paths(search_paths):
    DEFAULT_SEARCH_PATHS.extend(search_paths)


def search_paths_from_description(desc):
    """Return the search paths for the Category/WidgetDescription.
    """
    paths = []
    if desc.package:
        dirname = package_dirname(desc.package)
        paths.append(("", dirname))
    elif desc.qualified_name:
        dirname = package_dirname(package(desc.qualified_name))
        paths.append(("", dirname))

    if hasattr(desc, "search_paths"):
        paths.extend(desc.search_paths)
    return paths


class resource_loader(object):
    def __init__(self, search_paths=[]):
        self._search_paths = []
        self.add_search_paths(search_paths)

    @classmethod
    def from_description(cls, desc):
        """Construct an resource from a Widget or Category
        description.

        """
        paths = search_paths_from_description(desc)
        return icon_loader(search_paths=paths)

    def add_search_paths(self, paths):
        """Add `paths` to the list of search paths.
        """
        self._search_paths.extend(paths)

    def search_paths(self):
        """Return a list of all search paths.
        """
        return self._search_paths + default_search_paths()

    def split_prefix(self, path):
        """Split prefixed path.
        """
        if self.is_valid_prefixed(path) and ":" in path:
            prefix, path = path.split(":", 1)
        else:
            prefix = ""
        return prefix, path

    def is_valid_prefixed(self, path):
        i = path.find(":")
        return i != 1

    def find(self, name):
        """Find a resource matching `name`.
        """
        prefix, path = self.split_prefix(name)
        if prefix == "" and self.match(path):
            return path
        elif self.is_valid_prefixed(path):
            for pp, search_path in self.search_paths():
                if pp == prefix and \
                        self.match(os.path.join(search_path, path)):
                    return os.path.join(search_path, path)

        return None

    def match(self, path):
        return os.path.exists(path)

    def get(self, name):
        return self.load(name)

    def load(self, name):
        return self.open(name).read()

    def open(self, name):
        path = self.find(name)
        if path is not None:
            return open(path, "rb")
        else:
            raise IOError(2, "Cannot find %r" % name)

import glob


class icon_loader(resource_loader):
    DEFAULT_ICON = "icons/Unknown.png"

    def match(self, path):
        if resource_loader.match(self, path):
            return True
        return self.is_icon_glob(path)

    def icon_glob(self, path):
        name, ext = os.path.splitext(path)
        pattern = name + "_*" + ext
        return glob.glob(pattern)

    def is_icon_glob(self, path):
        name, ext = os.path.splitext(path)
        pattern = name + "_*" + ext
        return bool(glob.glob(pattern))

    def get(self, name, default=None):
        path = self.find(name)
        if not path:
            path = self.find(self.DEFAULT_ICON if default is None else default)
        if not path:
            raise IOError(2, "Cannot find %r in %s" % \
                          (name, self.search_paths()))
        if self.is_icon_glob(path):
            icons = self.icon_glob(path)
        else:
            icons = [path]

        from PyQt4.QtGui import QIcon

        icon = QIcon()
        for path in icons:
            icon.addFile(path)
        return icon

    def open(self, name):
        raise NotImplementedError

    def load(self, name):
        return self.get(name)


import unittest


class TestIconLoader(unittest.TestCase):
    def setUp(self):
        from PyQt4.QtGui import QApplication
        self.app = QApplication([])

    def tearDown(self):
        self.app.exit()
        del self.app

    def test_loader(self):
        loader = icon_loader()
        self.assertEqual(loader.search_paths(), DEFAULT_SEARCH_PATHS)
        icon = loader.get("icons/CanvasIcon.png")
        self.assertTrue(not icon.isNull())

        path = loader.find(":icons/CanvasIcon.png")
        self.assertTrue(os.path.isfile(path))
        icon = loader.get(":icons/CanvasIcon.png")
        self.assertTrue(not icon.isNull())

    def test_from_desc(self):
        from .registry.description import (
            WidgetDescription, CategoryDescription
        )

        desc = WidgetDescription.from_module(
            "Orange.OrangeWidgets.Data.OWFile"
        )

        loader = icon_loader.from_description(desc)
        path = loader.find(desc.icon)
        self.assertTrue(os.path.isfile(path))
        icon = loader.get(desc.icon)
        self.assertTrue(not icon.isNull())

        desc = CategoryDescription.from_package("Orange.OrangeWidgets.Data")
        loader = icon_loader.from_description(desc)
        path = loader.find("icons/file.svg")
        self.assertTrue(os.path.isfile(path))
        icon = loader.get("icons/file.svg")
        self.assertTrue(not icon.isNull())

    def test_package_reflection(self):
        from Orange.OrangeWidgets.Data import OWFile
        from Orange.OrangeWidgets import Data
        package_name = Data.__name__
        p1 = package("Orange.OrangeWidgets.Data.OWFile.OWFile")
        self.assertEqual(p1, package_name)

        p2 = package("Orange.OrangeWidgets.Data.OWFile")
        self.assertEqual(p2, package_name)

        p3 = package("Orange.OrangeWidgets.Data")
        self.assertEqual(p3, package_name)

        p4 = package(OWFile.__name__)
        self.assertEqual(p4, package_name)

        dirname = package_dirname(package_name)
        self.assertEqual(dirname, os.path.dirname(Data.__file__))


if __name__ == "__main__":
    unittest.main()
