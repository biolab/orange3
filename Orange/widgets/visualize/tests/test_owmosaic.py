# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from PyQt4.QtCore import QEvent, QPoint, Qt
from PyQt4.QtGui import QMouseEvent

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
class MosaicVizRankTests(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris.tab")

    def setUp(self):
        self.widget = self.create_widget(OWMosaicDisplay)
        self.vizrank = MosaicVizRank(self.widget)

    def test_count(self):
        """MosaicVizrank correctly computes the number of combinations"""
        widget = self.widget
        vizrank = self.vizrank
        widget.set_data(self.iris)

        widget.interior_coloring = widget.PEARSON
        vizrank.max_attrs = 2
        self.assertEqual(vizrank.state_count(), 6)
        vizrank.max_attrs = 3
        self.assertEqual(vizrank.state_count(), 10)
        vizrank.max_attrs = 4
        self.assertEqual(vizrank.state_count(), 11)

        widget.interior_coloring = widget.CLASS_DISTRIBUTION
        vizrank.max_attrs = 2
        self.assertEqual(vizrank.state_count(), 10)
        vizrank.max_attrs = 3
        self.assertEqual(vizrank.state_count(), 14)
        vizrank.max_attrs = 4
        self.assertEqual(vizrank.state_count(), 15)

    def test_iteration(self):
        """MosaicVizrank correctly iterates through states"""
        widget = self.widget
        vizrank = self.vizrank
        widget.set_data(self.iris)

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
        self.vizrank.attrs = [DiscreteVariable(n) for n in "abcd"]
        items = self.vizrank.row_for_state(0, [1, 3, 0])
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.text(), "a, b, d")
        self.assertEqual(
            item.data(self.vizrank._AttrRole),
            tuple(self.vizrank.attrs[i] for i in [0, 1, 3]))

    def test_does_not_crash(self):
        """MosaicVizrank computes rankings without crashing"""
        widget = self.widget
        vizrank = self.vizrank
        widget.set_data(self.iris)
        vizrank.max_attrs = 2

        widget.interior_coloring = widget.PEARSON
        vizrank.toggle()
        self.assertEqual(vizrank.rank_model.rowCount(), 6)
        widget.interior_coloring = widget.CLASS_DISTRIBUTION
        vizrank.toggle()
        self.assertEqual(vizrank.rank_model.rowCount(), 10)

        data = Table("housing.tab")
        widget.set_data(data)
        vizrank.toggle()

        data = Table(Domain(self.iris.domain.attributes, []), self.iris)
        widget.set_data(data)
        vizrank.toggle()
        self.assertEqual(vizrank.rank_model.rowCount(), 6)
