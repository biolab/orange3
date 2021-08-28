# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,unsubscriptable-object
from unittest.mock import patch

import numpy as np

from Orange.data import Table
from Orange.widgets.data.owrandomize import OWRandomize
from Orange.widgets.tests.base import WidgetTest


class TestOWRandomize(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.zoo = Table("zoo")

    def setUp(self):
        self.widget = self.create_widget(OWRandomize)

    def test_data(self):
        """Check widget's data and output with data on the input"""
        self.assertEqual(self.widget.data, None)
        self.send_signal(self.widget.Inputs.data, self.zoo)
        self.assertEqual(self.widget.data, self.zoo)
        output = self.get_output(self.widget.Outputs.data)
        np.testing.assert_array_equal(output.X, self.zoo.X)
        np.testing.assert_array_equal(output.metas, self.zoo.metas)
        self.assertTrue((output.Y != self.zoo.Y).any())
        self.assertTrue((np.sort(output.Y, axis=0) ==
                         np.sort(self.zoo.Y, axis=0)).all())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

    def test_shuffling(self):
        """Check widget's output for all types of shuffling"""
        self.send_signal(self.widget.Inputs.data, self.zoo)
        self.widget.class_check.setChecked(True)
        self.widget.attrs_check.setChecked(True)
        self.widget.metas_check.setChecked(True)
        output = self.get_output(self.widget.Outputs.data)
        self.assertTrue((output.X != self.zoo.X).any())
        self.assertTrue((np.sort(output.X, axis=0) ==
                         np.sort(self.zoo.X, axis=0)).all())
        self.assertTrue((output.Y != self.zoo.Y).any())
        self.assertTrue((np.sort(output.Y, axis=0) ==
                         np.sort(self.zoo.Y, axis=0)).all())
        self.assertTrue((output.metas != self.zoo.metas).any())
        self.assertTrue((np.sort(output.metas, axis=0) ==
                         np.sort(self.zoo.metas, axis=0)).all())

    def test_scope(self):
        self.send_signal(self.widget.Inputs.data, self.zoo)
        output = self.get_output(self.widget.Outputs.data)
        n_zoo = len(self.zoo)
        s = int(self.widget.scope_prop / 100 * n_zoo)
        self.assertGreater(sum((output.Y == self.zoo.Y).astype(int)), n_zoo - s)
        self.assertLessEqual(sum((output.Y != self.zoo.Y).astype(int)), s)

    def test_replicable_shuffling(self):
        """Check widget's output for replicable shuffling """
        self.send_signal(self.widget.Inputs.data, self.zoo)
        self.widget.replicable_check.setChecked(True)
        output = self.get_output(self.widget.Outputs.data)
        np.testing.assert_array_equal(output.X, self.zoo.X)
        np.testing.assert_array_equal(output.metas, self.zoo.metas)
        self.assertTrue((output.Y != self.zoo.Y).any())
        self.assertTrue((np.sort(output.Y, axis=0) ==
                         np.sort(self.zoo.Y, axis=0)).all())
        self.widget.commit.now()
        output2 = self.get_output(self.widget.Outputs.data)
        np.testing.assert_array_equal(output.X, output2.X)
        np.testing.assert_array_equal(output.Y, output2.Y)
        np.testing.assert_array_equal(output.metas, output2.metas)

    def test_unconditional_commit_on_new_signal(self):
        with patch.object(self.widget.commit, 'now') as apply:
            self.widget.auto_apply = False
            apply.reset_mock()
            self.send_signal(self.widget.Inputs.data, self.zoo)
            apply.assert_called()
