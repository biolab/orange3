# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access
import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import scipy.sparse as sp

from AnyQt.QtCore import QRectF, QPointF, QPoint ,QEvent, Qt
from AnyQt.QtGui import QMouseEvent
from AnyQt.QtTest import QTest

from orangecanvas.gui.test import mouseMove

from Orange.data import Table, DiscreteVariable, ContinuousVariable, Domain
from Orange.widgets.utils import itemmodels
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

    def test_tools_interaction(self):
        def mouse_path(stroke, button=Qt.LeftButton, delay=50):
            assert len(stroke) > 2
            QTest.mousePress(viewport, button, pos=stroke[0], delay=delay)
            for p in stroke[1:-1]:
                mouseMove(viewport, button, pos=p, delay=delay)
            QTest.mouseRelease(viewport, button, pos=stroke[-1], delay=delay)

        def assert_close(p1, p2):
            assert_almost_equal(np.array(p1), np.array(p2))

        w = self.widget
        w.adjustSize()
        viewport = w.plotview.viewport()
        center = viewport.rect().center()
        # Put single point
        w.set_current_tool(owpaintdata.PutInstanceTool)
        QTest.mouseClick(viewport, Qt.LeftButton)
        p0 = w.data[0]
        # Air brush stroke
        w.set_current_tool(owpaintdata.AirBrushTool)
        mouse_path([center, center + QPoint(5, 5), center + QPoint(5, 10), center + QPoint(0, 10)])

        w.set_current_tool(owpaintdata.SelectTool)

        # Draw selection rect
        mouse_path([center - QPoint(100, 100), center, center + QPoint(100, 100)])
        # Move selection
        mouse_path([center, center + QPoint(30, 30), center + QPoint(50, 50)])
        self.assertNotEqual(w.data[0], p0)
        count = len(w.data)

        w.current_tool.delete()  #
        self.assertNotEqual(len(w.data), count)

        w.set_current_tool(owpaintdata.ClearTool)
        self.assertEqual(len(w.data), 0)
        w.undo_stack.undo() # clear
        w.undo_stack.undo() # delete selection
        w.undo_stack.undo() # move
        assert_close(w.data[0], p0)

        stroke = [center - QPoint(10, 10), center, center + QPoint(10, 10)]

        w.set_current_tool(owpaintdata.MagnetTool)
        mouse_path(stroke)
        w.undo_stack.undo()
        assert_close(w.data[0], p0)

        w.set_current_tool(owpaintdata.JitterTool)
        mouse_path(stroke)
        w.undo_stack.undo()
        assert_close(w.data[0], p0)

    def test_add_remove_class(self):
        def put_instance():
            w.set_current_tool(owpaintdata.PutInstanceTool)
            QTest.mouseClick(viewport, Qt.LeftButton)

        def assert_class_column_equal(data):
            assert_array_equal(np.array(w.data)[:, 2].ravel(), data)

        w = self.widget
        viewport = w.plotview.viewport()
        put_instance()
        itemmodels.select_row(w.classValuesView, 1)
        put_instance()
        w.add_new_class_label()
        itemmodels.select_row(w.classValuesView, 2)
        put_instance()
        self.assertSequenceEqual(w.class_model, ["C1", "C2", "C3"])
        assert_class_column_equal([0, 1, 2])
        itemmodels.select_row(w.classValuesView, 0)
        w.remove_selected_class_label()
        self.assertSequenceEqual(w.class_model, ["C2", "C3"])
        assert_class_column_equal([0, 1])
        w.undo_stack.undo()
        self.assertSequenceEqual(w.class_model, ["C1", "C2", "C3"])
        assert_class_column_equal([0, 1, 2])


class TestCommands(unittest.TestCase):
    def test_merge_cmd(self):  # pylint: disable=import-outside-toplevel
        from Orange.widgets.data.owpaintdata import (
            Append, Move, DeleteIndices, Composite, merge_cmd
        )

        def merge(a, b):
            return merge_cmd(Composite(a, b))

        a1 = Append(np.array([[0., 0., 1.], [1., 1., 0.]]))
        a2 = Append(np.array([[2., 2., 1,]]))
        c = merge(a1, a2)
        self.assertIsInstance(c, Append)
        assert_array_equal(c.points, np.array([[0., 0., 1.], [1, 1, 0], [2, 2, 1]]))
        m1 = Move(range(2), np.array([1., 1., 0.]))
        m2 = Move(range(2), np.array([.5, .5, -1]))
        c = merge(m1, m2)
        self.assertIsInstance(c, Move)
        assert_array_equal(c.delta,  np.array([1.5, 1.5, -1]))
        c = merge(m1, Move(range(100, 102), np.array([1., 1., 1.])))
        self.assertIsInstance(c, Composite)

        c = merge(m1, Move([100, 105], np.array([0., 0, 0])))
        self.assertIsInstance(c, Composite)

        d1 = DeleteIndices(range(0, 3))
        d2 = DeleteIndices(range(3, 5))
        c = merge(d1, d2)
        self.assertIsInstance(c, DeleteIndices)
        self.assertEqual(c.indices, range(0, 5))
