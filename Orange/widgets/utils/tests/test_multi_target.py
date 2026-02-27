import unittest

from Orange.data import Table, Domain, DiscreteVariable
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.signals import Input
from Orange.widgets.utils.multi_target import check_multiple_targets_input
from Orange.widgets.widget import OWWidget


class TestMultiTargetDecorator(WidgetTest):
    class MockWidget(OWWidget):
        name = "MockWidget"
        keywords = "mockwidget"

        NotCalled = object()

        class Inputs:
            data = Input("Data", Table)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.called_with = self.NotCalled

        @Inputs.data
        @check_multiple_targets_input
        def set_data(self, obj):
            self.called_with = obj

        def pop_called_with(self):
            t = self.called_with
            self.called_with = self.NotCalled
            return t

    def setUp(self):
        self.widget = self.create_widget(self.MockWidget)
        self.data = Table("iris")

    def test_check_multiple_targets_input(self):
        class_vars = [self.data.domain.class_var,
                      DiscreteVariable("c1", values=("a", "b"))]
        domain = Domain(self.data.domain.attributes, class_vars=class_vars)
        multiple_targets_data = self.data.transform(domain)
        self.send_signal(self.widget.Inputs.data, multiple_targets_data)
        self.assertTrue(self.widget.Error.multiple_targets_data.is_shown())
        self.assertIs(self.widget.pop_called_with(), None)

        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertFalse(self.widget.Error.multiple_targets_data.is_shown())
        self.assertIs(self.widget.pop_called_with(), self.data)


if __name__ == "__main__":
    unittest.main()
