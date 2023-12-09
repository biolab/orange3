import unittest
from unittest.mock import patch, Mock

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QStandardItem
from AnyQt.QtWidgets import QDialog
from AnyQt.QtTest import QSignalSpy

from orangewidget.tests.base import GuiTest
from Orange.widgets.widget import OWWidget
from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets.visualize.utils.vizrank import (
    RunState, VizRankMixin,
    VizRankDialog, VizRankDialogAttrs, VizRankDialogAttrPair,
    VizRankDialogNAttrs
)


class MockDialog(VizRankDialog):
    class Task:
        interrupt = False
        set_partial_result = Mock()

        def is_interruption_requested(self):
            return self.interrupt

    task = Task()

    def __init__(self, parent=None):
        super().__init__(parent)

    # False positive?! "Number of parameters was 4 in 'ConcurrentMixin.start'
    # and is now 4 in overriding 'MockDialog.start' method".
    # pylint: disable=arguments-differ
    def start(self, func, *args, **kwargs):
        func(*args, self.task)

    def state_generator(self):
        return range(10)

    def state_count(self):
        return 10

    def compute_score(self, state):
        return 10 * (state % 2) + state // 2 if state != 3 else None

    row_for_state = Mock()


class TestVizRankDialog(GuiTest):

    @patch.object(VizRankDialog, "start")
    @patch.object(VizRankDialog, "prepare_run",
                  new=lambda self: setattr(self.run_state,
                                           "state", RunState.Ready))
    def test_init_and_button(self, run_vizrank):
        dialog = VizRankDialog(None)
        self.assertEqual(dialog.run_state.state, RunState.Initialized)
        self.assertEqual(dialog.button.text(), dialog.button_labels[RunState.Initialized])
        dialog.button.click()
        run_vizrank.assert_called_once()
        self.assertEqual(dialog.run_state.state, RunState.Running)
        self.assertEqual(dialog.button.text(), dialog.button_labels[RunState.Running])

        dialog.button.click()
        run_vizrank.assert_called_once()
        self.assertEqual(dialog.run_state.state, RunState.Paused)
        self.assertEqual(dialog.button.text(), dialog.button_labels[RunState.Paused])

        dialog.button.click()
        self.assertEqual(run_vizrank.call_count, 2)
        self.assertEqual(dialog.run_state.state, RunState.Running)
        self.assertEqual(dialog.button.text(), dialog.button_labels[RunState.Running])

    def test_running(self):
        dialog = MockDialog()
        dialog.task.interrupt = False
        dialog.start_computation()
        result = dialog.task.set_partial_result.call_args[0][0]
        self.assertEqual(result.scores, [0, 1, 2, 3, 4, 10, 12, 13, 14])
        self.assertEqual(result.completed_states, 10)

        dialog.rank_model.appendRow(QStandardItem("foo"))
        dialog.rank_model.appendRow(QStandardItem("bar"))
        dialog.on_done(result)
        self.assertEqual(dialog.run_state.state, RunState.Done)
        self.assertEqual(dialog.button.text(), dialog.button_labels[RunState.Done])
        self.assertFalse(dialog.button.isEnabled())
        selection = dialog.rank_table.selectedIndexes()
        self.assertEqual(len(selection), 1)
        self.assertEqual(selection[0].row(), 0)

    def test_interruption(self):
        dialog = MockDialog()
        dialog.task.interrupt = True
        dialog.start_computation()
        result = dialog.task.set_partial_result.call_args[0][0]
        self.assertEqual(result.scores, [0])
        self.assertEqual(result.completed_states, 1)


class TestVizRankMixin(GuiTest):
    def setUp(self):
        self.mock_dialog = Mock()
        self.mock_dialog.__name__ = "foo"

        class Widget(OWWidget, VizRankMixin(self.mock_dialog)):
            pass

        self.widget = Widget()

    def test_button(self):
        widget, dialog = self.widget, self.mock_dialog

        # False positives, pylint: disable=attribute-defined-outside-init)
        widget.start_vizrank = Mock()
        widget.raise_vizrank = Mock()

        button = widget.vizrank_button("Let's Vizrank")
        self.assertEqual(button.text(), "Let's Vizrank")
        self.assertFalse(button.isEnabled())

        widget.disable_vizrank("Too lazy to rank to-day.")
        self.assertEqual(button.text(), "Let's Vizrank")
        self.assertEqual(button.toolTip(), "Too lazy to rank to-day.")
        self.assertFalse(button.isEnabled())
        dialog.assert_not_called()

        widget.init_vizrank()
        dialog.assert_called_once()
        dialog.reset_mock()
        self.assertEqual(button.text(), "Let's Vizrank")
        self.assertEqual(button.toolTip(), "")
        self.assertTrue(button.isEnabled())

        widget.disable_vizrank("Too lazy to rank to-day.")
        self.assertEqual(button.text(), "Let's Vizrank")
        self.assertEqual(button.toolTip(), "Too lazy to rank to-day.")
        self.assertFalse(button.isEnabled())
        dialog.assert_not_called()

        widget.init_vizrank()
        dialog.assert_called_once()  # new data, new dialog!
        dialog.reset_mock()

        button.click()
        widget.start_vizrank.assert_called_once()
        widget.raise_vizrank.assert_called_once()

    def test_no_button(self):
        widget, dialog = self.widget, self.mock_dialog

        widget.disable_vizrank("Too lazy to rank to-day.")
        dialog.assert_not_called()

        widget.init_vizrank()
        dialog.assert_called_once()
        dialog.reset_mock()
        widget.disable_vizrank("Too lazy to rank to-day.")

        widget.disable_vizrank("Too lazy to rank to-day.")
        dialog.assert_not_called()

        widget.init_vizrank()
        dialog.assert_called_once()
        dialog.reset_mock()
        widget.disable_vizrank("Too lazy to rank to-day.")

    def test_init_vizrank(self):
        widget, dialog = self.widget, self.mock_dialog
        a, b = Mock(), Mock()
        widget.init_vizrank(a, b)
        dialog.assert_called_with(widget, a, b)


class TestVizRankDialogWithData(GuiTest):
    def setUp(self):
        self.attrs = tuple(ContinuousVariable(n) for n in "abcdef")
        self.class_var = ContinuousVariable("y")
        self.variables = (*self.attrs, self.class_var)
        self.metas = tuple(ContinuousVariable(n) for n in "mn")
        self.data = Table.from_list(
            Domain(self.attrs, self.class_var, self.metas),
            [[0] * 9])


class TestVizRankDialogAttrs(TestVizRankDialogWithData):
    def test_init(self):
        dialog = VizRankDialogAttrs(None, self.data)
        self.assertIs(dialog.data, self.data)
        self.assertEqual(dialog.attrs, self.variables)
        self.assertIsNone(dialog.attr_color)

        dialog = VizRankDialogAttrs(None, self.data, self.attrs[1:])
        self.assertIs(dialog.data, self.data)
        self.assertEqual(dialog.attrs, self.attrs[1:])
        self.assertIsNone(dialog.attr_color)

        dialog = VizRankDialogAttrs(None, self.data, self.attrs[1:], self.attrs[3])
        self.assertIs(dialog.data, self.data)
        self.assertEqual(dialog.attrs, self.attrs[1:])
        self.assertIs(dialog.attr_color, self.attrs[3])

    def test_attr_order(self):
        # pylint: disable=abstract-method
        dialog = VizRankDialogAttrs(None, self.data)
        self.assertEqual(dialog.attr_order, self.variables)

        class OrderedAttr(VizRankDialogAttrs):
            call_count = 0

            def score_attributes(self):
                self.call_count += 1
                return self.attrs[::-1]

        dialog = OrderedAttr(None, self.data)
        self.assertEqual(dialog.attr_order, self.variables[::-1])
        self.assertEqual(dialog.attr_order, self.variables[::-1])
        self.assertEqual(dialog.call_count, 1)

    def test_row_for_state(self):
        # pylint: disable=protected-access
        dialog = VizRankDialogAttrs(None, self.data)
        item = dialog.row_for_state(0, [3, 1])[0]
        self.assertEqual(item.data(Qt.DisplayRole), "d, b")
        self.assertEqual(item.data(dialog._AttrRole), [self.attrs[3], self.attrs[1]])

        dialog.sort_names_in_row = True
        item = dialog.row_for_state(0, [3, 1])[0]
        self.assertEqual(item.data(Qt.DisplayRole), "b, d")
        self.assertEqual(item.data(dialog._AttrRole), [self.attrs[1], self.attrs[3]])

    def test_autoselect(self):
        # False positive, pylint: disable=no-value-for-parameter
        class Widget(OWWidget, VizRankMixin(VizRankDialogAttrs)):
            pass
        widget = Widget()
        try:
            widget.init_vizrank(self.data)
            dialog = widget.vizrank_dialog
            for state in ([0, 1], [0, 3], [1, 3], [3, 1]):
                dialog.rank_model.appendRow(dialog.row_for_state(0, state))

            widget.vizrankAutoSelect.emit([self.attrs[1], self.attrs[3]])
            selection = dialog.rank_table.selectedIndexes()
            self.assertEqual(len(selection), 1)
            self.assertEqual(selection[0].row(), 2)

            widget.vizrankAutoSelect.emit([self.attrs[3], self.attrs[1]])
            selection = dialog.rank_table.selectedIndexes()
            self.assertEqual(len(selection), 1)
            self.assertEqual(selection[0].row(), 3)

            widget.vizrankAutoSelect.emit([self.attrs[3], self.attrs[0]])
            selection = dialog.rank_table.selectedIndexes()
            self.assertEqual(len(selection), 0)
        finally:
            widget.onDeleteWidget()


class TestVizRankDialogAttrPair(TestVizRankDialogWithData):
    def test_count_and_generator(self):
        dialog = VizRankDialogAttrPair(None, self.data, self.attrs[:5])
        self.assertEqual(dialog.state_count(), 5 * 4 / 2)
        self.assertEqual(
            list(dialog.state_generator()),
            [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3),
             (0, 4), (1, 4), (2, 4), (3, 4)])


class TestVizRankDialogNAttrs(TestVizRankDialogWithData):
    def test_spin_interaction(self):
        dialog = VizRankDialogNAttrs(None,
                                     self.data, self.attrs[:5], None, 4)
        spy = QSignalSpy(dialog.runStateChanged)

        with patch.object(
                VizRankDialog, "pause_computation",
                side_effect=lambda: dialog.set_run_state(RunState.Paused)):
            with patch.object(
                    VizRankDialog, "start_computation",
                    side_effect=lambda: dialog.set_run_state(RunState.Running)):
                spin = dialog.n_attrs_spin
                self.assertEqual(spin.value(), 4)
                self.assertEqual(spin.maximum(), 5)

                dialog.start_computation()
                self.assertEqual(dialog.run_state.state, RunState.Running)
                self.assertEqual(spy[-1][1]["n_attrs"], 4)

                spin.setValue(3)
                # Ranking must be paused
                self.assertEqual(dialog.run_state.state, RunState.Paused)
                # Label should be changed to "restart with ..."
                self.assertNotEqual(dialog.button.text(),
                                    dialog.button_labels[RunState.Paused])

                spin.setValue(4)
                self.assertEqual(dialog.run_state.state, RunState.Paused)
                # Label should be reset to "Continue"
                self.assertEqual(dialog.button.text(),
                                 dialog.button_labels[RunState.Paused])

        # Remove the side-effect so that we see that start_computation
        # resets the state to Initialized before calling super
        with patch.object(VizRankDialog, "start_computation"):
            dialog.start_computation()
            self.assertEqual(spy[-1][1]["n_attrs"], 4)
            # Here, the state must not be reset to Initialized
            self.assertEqual(dialog.run_state.state, RunState.Paused)
            # But now manually set it to appropriate state
            dialog.set_run_state(RunState.Running)

            spin.setValue(3)
            self.assertEqual(dialog.run_state.state, RunState.Paused)
            self.assertNotEqual(dialog.button.text(), dialog.button_labels[RunState.Paused])

            dialog.start_computation()
            self.assertEqual(dialog.run_state.state, RunState.Initialized)
            self.assertEqual(spy[-1][1]["n_attrs"], 3)


if __name__ == "__main__":
    unittest.main()
