# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access

import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import QRectF, QPointF, QEvent, Qt
from AnyQt.QtGui import QMouseEvent

from Orange.data import Table, DiscreteVariable, ContinuousVariable, Domain
from Orange.widgets.data import owpaintdata
from Orange.widgets.data.owpaintdata import OWPaintData
from Orange.widgets.tests.base import WidgetTest, datasets


class TestOWPaintData(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(
            OWPaintData,
            stored_settings={
                "autocommit": True
            }
        )  # type: OWPaintData

    def test_empty_data(self):
        """No crash on empty data"""
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.data,
                         Table.from_domain(data.domain))

    def test_var_name_duplicates(self):
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.attr1 = 'atr1'
        self.widget.attr2 = 'atr1'
        self.widget._attr_name_changed()
        self.assertTrue(self.widget.Warning.renamed_vars.is_shown())
        self.widget.attr2 = 'atr2'
        self.widget._attr_name_changed()
        self.assertFalse(self.widget.Warning.renamed_vars.is_shown())

    def test_nan_data(self):
        data = datasets.missing_data_2()
        self.send_signal(self.widget.Inputs.data, data)

    def test_output_shares_internal_buffer(self):
        data = Table("iris")[::5]
        self.send_signal(self.widget.Inputs.data, data)
        output1 = self.get_output(self.widget.Outputs.data)
        output1_copy = output1.copy()
        self.widget._add_command(
            owpaintdata.SelectRegion(QRectF(0.25, 0.25, 0.5, 0.5))
        )
        self.widget._add_command(
            owpaintdata.MoveSelection(QPointF(0.1, 0.1))
        )
        output2 = self.get_output(self.widget.Outputs.data)
        self.assertIsNot(output1, output2)

        np.testing.assert_equal(output1.X, output1_copy.X)
        np.testing.assert_equal(output1.Y, output1_copy.Y)

        self.assertTrue(np.any(output1.X != output2.X))

    def test_20_values_class(self):
        domain = Domain(
            [ContinuousVariable("A"),
             ContinuousVariable("B")],
            DiscreteVariable("C", values=[chr(ord("a") + i) for i in range(20)])
        )
        data = Table.from_list(domain, [[0.1, 0.2, "a"], [0.4, 0.7, "t"]])
        self.send_signal(self.widget.Inputs.data, data)

    def test_sparse_data(self):
        """
        Show warning msg when data is sparse.
        GH-2298
        GH-2163
        """
        data = Table("iris")[::25].copy()
        with data.unlocked():
            data.X = sp.csr_matrix(data.X)
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Warning.sparse_not_supported.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.sparse_not_supported.is_shown())

    def test_load_empty_data(self):
        """
        It should not crash when old workflow with no data is loaded.
        GH-2399
        """
        self.create_widget(OWPaintData, stored_settings={"data": []})

    def test_reset_to_input(self):
        """Checks if the data resets to input when Reset to Input is pressed"""
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), len(data))
        self.widget.set_current_tool(self.widget.TOOLS[1][2]) # PutInstanceTool
        tool = self.widget.current_tool
        event = QMouseEvent(QEvent.MouseButtonPress, QPointF(0.17, 0.17),
                            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        tool.mousePressEvent(event)
        output = self.get_output(self.widget.Outputs.data)
        self.assertNotEqual(len(output), len(data))
        self.assertEqual(len(output), 151)
        self.widget.reset_to_input()
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), len(data))

        self.send_signal(self.widget.Inputs.data, data)
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), len(data))
        self.widget.set_current_tool(self.widget.TOOLS[5][2])  # ClearTool
        output = self.get_output(self.widget.Outputs.data)
        self.assertIsNone(output)
        self.widget.reset_to_input()
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), len(data))
