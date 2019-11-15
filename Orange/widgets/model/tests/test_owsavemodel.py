import unittest

from Orange.classification.majority import MajorityLearner
from Orange.data import Table
from Orange.widgets.model.owsavemodel import OWSaveModel
from Orange.widgets.utils.save.tests.test_owsavebase import \
    SaveWidgetsTestBaseMixin
from Orange.widgets.tests.base import WidgetTest


class OWSaveTestBase(WidgetTest, SaveWidgetsTestBaseMixin):
    def setUp(self):
        self.widget = self.create_widget(OWSaveModel)
        self.model = MajorityLearner(Table("iris"))


if __name__ == "__main__":
    unittest.main()
