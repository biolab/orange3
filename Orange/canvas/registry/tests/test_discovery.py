"""
Test widget discovery

"""

import os
import logging

import unittest

from ..discovery import WidgetDiscovery, widget_descriptions_from_package

from ..description import CategoryDescription, WidgetDescription


class TestDiscovery(unittest.TestCase):

    def setUp(self):
        logging.basicConfig()

    def discovery_class(self):
        return WidgetDiscovery()

    def test_handle(self):
        disc = self.discovery_class()

        desc = CategoryDescription(name="C", qualified_name="M.C")
        disc.handle_category(desc)

        desc = WidgetDescription(name="SomeWidget", id="some.widget",
                                 qualified_name="Some.Widget",
                                 category="C",)
        disc.handle_widget(desc)

    def test_process_module(self):
        disc = self.discovery_class()
        disc.process_category_package(
            "Orange.widgets.data"
        )
        disc.process_widget_module(
            "Orange.widgets.classify.ownaivebayes"
        )

    def test_process_loader(self):
        disc = self.discovery_class()

        def callable(discovery):
            desc = CategoryDescription(
                name="Data", qualified_name="Data")

            discovery.handle_category(desc)

            desc = WidgetDescription.from_module(
                "Orange.widgets.data.owfile"
            )
            discovery.handle_widget(desc)

        disc.process_loader(callable)

    def test_process_iter(self):
        disc = self.discovery_class()
        cat_desc = CategoryDescription.from_package(
            "Orange.widgets.data"
        )
        wid_desc = widget_descriptions_from_package(
            "Orange.widgets.data"
        )
        disc.process_iter([cat_desc] + wid_desc)

    def test_run(self):
        disc = self.discovery_class()
        disc.run("example.does.not.exist.but.it.does.not.matter.")
