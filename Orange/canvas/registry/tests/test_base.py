"""
Test WidgetRegistry.
"""

import logging
from operator import attrgetter

import unittest

from ..base import WidgetRegistry
from .. import description


class TestRegistry(unittest.TestCase):
    def setUp(self):
        logging.basicConfig()

    def test_registry_const(self):
        reg = WidgetRegistry()

        data_desc = description.CategoryDescription.from_package(
            "Orange.widgets.data"
        )

        reg.register_category(data_desc)

        self.assertTrue(reg.has_category(data_desc.name))
        self.assertSequenceEqual(reg.categories(), [data_desc])
        self.assertIs(reg.category(data_desc.name), data_desc)

        file_desc = description.WidgetDescription.from_module(
            "Orange.widgets.data.owfile"
        )
        file_desc.category = "Data"
        reg.register_widget(file_desc)

        self.assertTrue(reg.has_widget(file_desc.qualified_name))
        self.assertSequenceEqual(reg.widgets("Data"), [file_desc])
        self.assertIs(reg.widget(file_desc.qualified_name), file_desc)

        # ValueError adding a description with the same qualified name
        with self.assertRaises(ValueError):
            desc = description.WidgetDescription(
                name="A name",
                id=file_desc.id,
                qualified_name=file_desc.qualified_name
            )
            reg.register_widget(desc)

        discretize_desc = description.WidgetDescription.from_module(
            "Orange.widgets.data.owdiscretize"
        )
        discretize_desc.category = "Data"
        reg.register_widget(discretize_desc)

        self.assertTrue(reg.has_widget(discretize_desc.qualified_name))
        self.assertIs(reg.widget(discretize_desc.qualified_name),
                      discretize_desc)

        self.assertSetEqual(set(reg.widgets("Data")),
                            set([file_desc, discretize_desc]))

        classify_desc = description.CategoryDescription.from_package(
            "Orange.widgets.classify"
        )
        classify_desc.category = "Classify"
        reg.register_category(classify_desc)

        self.assertTrue(reg.has_category(classify_desc.name))
        self.assertIs(reg.category(classify_desc.name), classify_desc)
        self.assertSetEqual(set(reg.categories()),
                            set([data_desc, classify_desc]))

        bayes_desc = description.WidgetDescription.from_module(
            "Orange.widgets.classify.ownaivebayes"
        )
        bayes_desc.category = "Classify"
        reg.register_widget(bayes_desc)

        self.assertTrue(reg.has_widget(bayes_desc.qualified_name))
        self.assertIs(reg.widget(bayes_desc.qualified_name), bayes_desc)
        self.assertSequenceEqual(reg.widgets("Classify"), [bayes_desc])

        # Test copy constructor
        reg1 = WidgetRegistry(reg)
        self.assertTrue(reg1.has_category(data_desc.name))
        self.assertTrue(reg1.has_category(classify_desc.name))
        self.assertSequenceEqual(reg.categories(), reg1.categories())

        # Test 'widgets()'
        self.assertSetEqual(set(reg1.widgets()),
                            set([file_desc, discretize_desc, bayes_desc]))

        # Test ordering by priority
        self.assertSequenceEqual(
            reg.widgets("Data"),
            sorted([file_desc, discretize_desc], key=attrgetter("priority")))

        self.assertTrue(
            all(isinstance(desc.priority, int)
                for desc in [file_desc, discretize_desc, bayes_desc]))
