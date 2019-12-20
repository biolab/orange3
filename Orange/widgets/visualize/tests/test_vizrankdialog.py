from itertools import chain
import unittest
from unittest.mock import Mock
from queue import Queue

from AnyQt.QtGui import QStandardItem

from Orange.data import Table
from Orange.widgets.visualize.utils import (
    VizRankDialog, Result, run_vizrank, QueuedScore
)
from Orange.widgets.tests.base import WidgetTest


def compute_score(x):
    return (x[0] + 1) / (x[1] + 1)


class TestRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Table("iris")

    def test_Result(self):
        res = Result(queue=Queue(), scores=[])
        self.assertIsInstance(res.queue, Queue)
        self.assertIsInstance(res.scores, list)

    def test_run_vizrank(self):
        scores, task = [], Mock()
        # run through all states
        task.is_interruption_requested.return_value = False
        states = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        res = run_vizrank(compute_score, lambda initial: chain(states),
                          None, scores, 0, 6, task)

        next_state = self.assertQueueEqual(
            res.queue, [0, 0, 0, 3, 2, 5], compute_score,
            states, states[1:] + [None])
        self.assertIsNone(next_state)
        res_scores = sorted([compute_score(x) for x in states])
        self.assertListEqual(res.scores, res_scores)
        self.assertIsNot(scores, res.scores)
        self.assertEqual(task.set_partial_result.call_count, 2)
        self.assertEqual(task.set_progress_value.call_count, 7)

    def test_run_vizrank_interrupt(self):
        scores, task = [], Mock()
        # interrupt calculation in third iteration
        task.is_interruption_requested.side_effect = lambda: \
            True if task.is_interruption_requested.call_count > 2 else False
        states = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        res = run_vizrank(compute_score, lambda initial: chain(states),
                          None, scores, 0, 6, task)

        next_state = self.assertQueueEqual(
            res.queue, [0, 0], compute_score, states[:2], states[1:3])
        self.assertEqual(next_state, (0, 3))
        res_scores = sorted([compute_score(x) for x in states[:2]])
        self.assertListEqual(res.scores, res_scores)
        self.assertIsNot(scores, res.scores)
        self.assertEqual(task.set_partial_result.call_count, 1)
        self.assertEqual(task.set_progress_value.call_count, 3)
        task.set_progress_value.assert_called_with(int(1 / 6 * 100))

        # continue calculation through all states
        task.is_interruption_requested.side_effect = lambda: False
        i = states.index(next_state)
        res = run_vizrank(compute_score, lambda initial: chain(states[i:]),
                          None, res_scores, 2, 6, task)

        next_state = self.assertQueueEqual(
            res.queue, [0, 3, 2, 5], compute_score, states[2:],
            states[3:] + [None])
        self.assertIsNone(next_state)
        res_scores = sorted([compute_score(x) for x in states])
        self.assertListEqual(res.scores, res_scores)
        self.assertIsNot(scores, res.scores)
        self.assertEqual(task.set_partial_result.call_count, 3)
        self.assertEqual(task.set_progress_value.call_count, 8)
        task.set_progress_value.assert_called_with(int(5 / 6 * 100))

    def assertQueueEqual(self, queue, positions, f, states, next_states):
        self.assertIsInstance(queue, Queue)
        for qs in (QueuedScore(position=p, score=f(s), state=s, next_state=ns)
                   for p, s, ns in zip(positions, states, next_states)):
            result = queue.get_nowait()
            self.assertEqual(result.position, qs.position)
            self.assertEqual(result.state, qs.state)
            self.assertEqual(result.next_state, qs.next_state)
            self.assertEqual(result.score, qs.score)
            next_state = result.next_state
        return next_state


class TestVizRankDialog(WidgetTest):
    def test_on_partial_result(self):
        def iterate_states(initial_state):
            if initial_state is not None:
                return chain(states[states.index(initial_state):])
            return chain(states)

        def invoke_on_partial_result():
            widget.on_partial_result(run_vizrank(
                widget.compute_score,
                widget.iterate_states,
                widget.saved_state,
                widget.scores,
                widget.saved_progress,
                widget.state_count(),
                task
            ))

        task = Mock()
        states = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        widget = VizRankDialog(None)
        widget.progressBarInit()
        widget.compute_score = compute_score
        widget.iterate_states = iterate_states
        widget.row_for_state = lambda sc, _: [QStandardItem(str(sc))]
        widget.state_count = lambda: len(states)

        # interrupt calculation in third iteration
        task.is_interruption_requested.side_effect = lambda: \
            True if task.is_interruption_requested.call_count > 2 else False
        invoke_on_partial_result()
        self.assertEqual(widget.rank_model.rowCount(), 2)
        for row, score in enumerate(
                sorted([compute_score(x) for x in states[:2]])):
            self.assertEqual(widget.rank_model.item(row, 0).text(), str(score))
        self.assertEqual(widget.saved_progress, 2)
        task.set_progress_value.assert_called_with(int(1 / 6 * 100))

        # continue calculation through all states
        task.is_interruption_requested.side_effect = lambda: False
        invoke_on_partial_result()
        self.assertEqual(widget.rank_model.rowCount(), 6)
        for row, score in enumerate(
                sorted([compute_score(x) for x in states])):
            self.assertEqual(widget.rank_model.item(row, 0).text(), str(score))
        self.assertEqual(widget.saved_progress, 6)
        task.set_progress_value.assert_called_with(int(5 / 6 * 100))


if __name__ == "__main__":
    unittest.main()
