# pylint: disable=unsubscriptable-object
import unittest

from Orange.data import Table
from Orange.widgets.data.owpurgedomain import OWPurgeDomain
from Orange.widgets.tests.base import WidgetTest


class TestOWPurgeDomain(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWPurgeDomain)
        self.iris = Table("iris")

    def test_minimum_size(self):
        pass


if __name__ == "__main__":
    unittest.main()
