# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from AnyQt.QtCore import QEvent, QPoint, Qt
from AnyQt.QtGui import QMouseEvent

from Orange.data import Table, DiscreteVariable, Domain
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.visualize.owmosaic import OWMosaicDisplay, MosaicVizRank


class TestOWMosaicDisplay(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data

    def setUp(self):
        self.widget = self.create_widget(OWMosaicDisplay)

    def _select_data(self):
        self.widget.select_area(1, QMouseEvent(
            QEvent.MouseButtonPress, QPoint(), Qt.LeftButton,
            Qt.LeftButton, Qt.KeyboardModifiers()))
        return [2, 3, 9, 23, 29, 30, 34, 35, 37, 42, 47, 49]


# Derive from WidgetTest to simplify creation of the Mosaic widget, although
# we are actually testing the MosaicVizRank dialog and not the widget

# These tests are rather crude: the main challenge of this widget is to handle
# user interactions and interrupts, e.g. changing the widget settings or
# getting new data while the computation is running.
class MosaicVizRankTests(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris.tab")
        dom = Domain(cls.iris.domain.attributes, [])
        cls.iris_no_class = Table(dom, cls.iris)

    def setUp(self):
        self.widget = self.create_widget(OWMosaicDisplay)
        self.vizrank = self.widget.vizrank

    def test_count(self):
        """MosaicVizrank correctly computes the number of combinations"""
        widget = self.widget
        vizrank = self.vizrank
        widget.set_data(self.iris)

        widget.interior_coloring = widget.PEARSON
        vizrank.max_attrs = 2
        self.assertEqual(vizrank.state_count(), 10)  # 5x4 / 2
        vizrank.max_attrs = 3
        self.assertEqual(vizrank.state_count(), 20)  # above + 5x4x3 / 2x3
        vizrank.max_attrs = 4
        self.assertEqual(vizrank.state_count(), 25)  # above + 5x4x3x2 / 2x3x4

        widget.interior_coloring = widget.CLASS_DISTRIBUTION
        vizrank.max_attrs = 2
        self.assertEqual(vizrank.state_count(), 10)  # 4 + 4x3 / 2
        vizrank.max_attrs = 3
        self.assertEqual(vizrank.state_count(), 14)  # above + 4x3x2 / 3x2
        vizrank.max_attrs = 4
        self.assertEqual(vizrank.state_count(), 15)  # above + 4x3x2x1 / 2x3x4

        widget.set_data(self.iris_no_class)
        vizrank.max_attrs = 2
        self.assertEqual(vizrank.state_count(), 6)  # 4x3 / 2
        vizrank.max_attrs = 3
        self.assertEqual(vizrank.state_count(), 10)  # above + 4x3x2 / 3x2
        vizrank.max_attrs = 4
        self.assertEqual(vizrank.state_count(), 11)  # above + 4x3x2x1 / 2x3x4

    def test_iteration(self):
        """MosaicVizrank correctly iterates through states"""
        widget = self.widget
        vizrank = self.vizrank
        widget.set_data(self.iris)
        vizrank.compute_attr_order()

        widget.interior_coloring = widget.CLASS_DISTRIBUTION
        vizrank.max_attrs = 4
        self.assertEqual([state.copy()
                          for state in vizrank.iterate_states(None)],
                         [[0], [1], [2], [3],
                          [0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3],
                          [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3],
                          [0, 1, 2, 3]])
        self.assertEqual([state.copy()
                          for state in vizrank.iterate_states([0, 3])],
                         [[0, 3], [1, 3], [2, 3],
                          [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3],
                          [0, 1, 2, 3]])

        vizrank.max_attrs = 2
        self.assertEqual([state.copy()
                          for state in vizrank.iterate_states(None)],
                         [[0], [1], [2], [3],
                          [0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]])
        self.assertEqual([state.copy()
                          for state in vizrank.iterate_states([0, 3])],
                         [[0, 3], [1, 3], [2, 3]])

        widget.interior_coloring = widget.PEARSON
        vizrank.max_attrs = 4
        self.assertEqual([state.copy()
                          for state in vizrank.iterate_states(None)],
                         [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3],
                          [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3],
                          [0, 1, 2, 3]])
        self.assertEqual([state.copy()
                          for state in vizrank.iterate_states([0, 3])],
                         [[0, 3], [1, 3], [2, 3],
                          [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3],
                          [0, 1, 2, 3]])

        vizrank.max_attrs = 2
        self.assertEqual([state.copy()
                          for state in vizrank.iterate_states(None)],
                         [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]])
        self.assertEqual([state.copy()
                          for state in vizrank.iterate_states([0, 3])],
                         [[0, 3], [1, 3], [2, 3]])

    def test_row_for_state(self):
        """MosaicVizrank returns table row corresponding to the state"""
        self.widget.set_data(self.iris)
        self.vizrank.attr_ordering = [DiscreteVariable(n) for n in "abcd"]
        items = self.vizrank.row_for_state(0, [1, 3, 0])
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.text(), "a, b, d")
        self.assertEqual(
            item.data(self.vizrank._AttrRole),
            tuple(self.vizrank.attr_ordering[i] for i in [0, 1, 3]))

    def test_does_not_crash(self):
        """MosaicVizrank computes rankings without crashing"""
        widget = self.widget
        vizrank = self.vizrank
        widget.set_data(self.iris)
        vizrank.max_attrs = 2

        widget.interior_coloring = widget.PEARSON
        vizrank.toggle()
        self.assertEqual(vizrank.rank_model.rowCount(), 10)  # 4x5 / 2
        widget.interior_coloring = widget.CLASS_DISTRIBUTION
        vizrank.toggle()
        self.assertEqual(vizrank.rank_model.rowCount(), 10)  # 4 + 4x5 / 2

        widget.set_data(self.iris_no_class)
        vizrank.toggle()
        self.assertEqual(vizrank.rank_model.rowCount(), 6)  # 3x4 / 2

        data = Table("housing.tab")
        widget.set_data(data)
        vizrank.toggle()
