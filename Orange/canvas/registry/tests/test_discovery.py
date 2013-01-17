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

        desc = WidgetDescription(name="SomeWidget", category="C",
                                 qualified_name="Some.Widget")
        disc.handle_widget(desc)

    def test_file(self):
        from Orange.OrangeWidgets.Data import OWFile
        disc = self.discovery_class()
        disc.process_file(OWFile.__file__)

    def test_process_directory(self):
        from Orange.OrangeWidgets import Data, Visualize
        data_dirname = os.path.dirname(Data.__file__)
        visualize_dirname = os.path.dirname(Visualize.__file__)

        disc = self.discovery_class()
        disc.process_directory(data_dirname)
        disc.process_directory(visualize_dirname)

    def test_process_module(self):
        disc = self.discovery_class()
        disc.process_category_package(
            "Orange.OrangeWidgets.Data"
        )
        disc.process_widget_module(
            "Orange.OrangeWidgets.Classify.OWNaiveBayes"
        )

    def test_process_loader(self):
        disc = self.discovery_class()

        def callable(discovery):
            desc = CategoryDescription(
                name="Data", qualified_name="Data")

            discovery.handle_category(desc)

            desc = WidgetDescription.from_module(
                "Orange.OrangeWidgets.Data.OWFile"
            )
            discovery.handle_widget(desc)

        disc.process_loader(callable)

    def test_process_iter(self):
        disc = self.discovery_class()
        cat_desc = CategoryDescription.from_package(
            "Orange.OrangeWidgets.Data"
        )
        wid_desc = widget_descriptions_from_package(
            "Orange.OrangeWidgets.Data"
        )
        disc.process_iter([cat_desc] + wid_desc)

    def test_run(self):
        disc = self.discovery_class()
        disc.run()
