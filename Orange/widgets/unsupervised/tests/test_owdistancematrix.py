# pylint: disable=protected-access

import unittest
from functools import partial
from unittest.mock import patch

import numpy as np
from AnyQt.QtCore import Qt

from orangewidget.settings import Context

from Orange.misc import DistMatrix
from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.distance import Euclidean
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owdistancematrix import OWDistanceMatrix


class TestOWDistanceMatrix(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWDistanceMatrix)
        self.iris = Table("iris")[:5]
        self.distances = Euclidean(self.iris)

    def test_set_distances(self):
        assert isinstance(self.widget, OWDistanceMatrix)

        # Distances with row data
        self.widget.set_distances(self.distances)
        self.assertIn(self.iris.domain[0], self.widget.annot_combo.model())
        self.widget.send_report()

        # Distances without row data
        self.distances.row_items = None
        self.widget.set_distances(self.distances)
        self.assertNotIn(self.iris.domain[0], self.widget.annot_combo.model())
        self.widget.send_report()

        # Non-square distances, no labels
        distances = DistMatrix(np.array([[1, 2, 3], [4, 5, 6]]))
        self.widget.set_distances(distances)
        self.assertEqual(self.widget.annot_combo.model().rowCount(), 2)
        self.widget.send_report()

        # Non-square distances, row labels
        distances = DistMatrix(np.array([[1, 2, 3], [4, 5, 6]]))
        distances.row_items = list("ab")
        self.widget.set_distances(distances)
        self.assertEqual(self.widget.annot_combo.model().rowCount(), 3)
        self.widget.send_report()

        # Non-square distances, column labels
        distances = DistMatrix(np.array([[1, 2, 3], [4, 5, 6]]))
        distances.col_items = list("def")
        self.widget.set_distances(distances)
        self.assertEqual(self.widget.annot_combo.model().rowCount(), 3)
        self.widget.send_report()

        # Non-square distances, both labels
        distances = DistMatrix(np.array([[1, 2, 3], [4, 5, 6]]))
        distances.row_items = list("ab")
        distances.col_items = list("def")
        self.widget.set_distances(distances)
        self.assertEqual(self.widget.annot_combo.model().rowCount(), 3)
        self.widget.send_report()

        # Non-square distances, no labels
        distances = DistMatrix(np.array([[1, 2, 3], [4, 5, 6]]))
        self.widget.set_distances(distances)
        self.assertEqual(self.widget.annot_combo.model().rowCount(), 2)
        self.widget.send_report()

    def test_context_attribute(self):
        distances = Euclidean(self.iris, axis=0)
        annotations = ["None", "Enumerate"]
        self.widget.set_distances(distances)
        self.widget.openContext(distances, annotations)

    def test_unconditional_commit_on_new_signal(self):
        with patch.object(self.widget.commit, 'now') as commit:
            self.widget.auto_commit = False
            commit.reset_mock()
            self.send_signal(self.widget.Inputs.distances, self.distances)
            commit.assert_called()

    def test_labels(self):
        x, y = (ContinuousVariable(c) for c in "xy")
        s = StringVariable("s")
        grades = Table.from_list(
            Domain([x, y], [], [s]),
            [[91.0, 89.0, "Bill"],
             [51.0, 100.0, "Cynthia"],
             [9.0, 61.0, "Demi"],
             [49.0, 92.0, "Fred"],
             [91.0, 49.0, "George"]
             ]
        )

        header = self.widget.tablemodel.headerData

        distances = Euclidean(grades)
        self.widget.set_distances(distances)
        ac = self.widget.annot_combo
        idx = ac.model().indexOf(grades.domain.metas[0])
        ac.setCurrentIndex(idx)
        ac.activated.emit(idx)
        self.assertEqual(header(2, Qt.Horizontal, Qt.DisplayRole), "Demi")
        self.assertIsNone(header(2, Qt.Horizontal, Qt.BackgroundRole))
        self.assertEqual(header(2, Qt.Vertical, Qt.DisplayRole), "Demi")
        self.assertIsNone(header(2, Qt.Vertical, Qt.BackgroundRole))

        idx = ac.model().indexOf(grades.domain.attributes[0])
        ac.setCurrentIndex(idx)
        ac.activated.emit(idx)
        self.assertIn("9", header(2, Qt.Horizontal, Qt.DisplayRole))
        self.assertIsNotNone(header(2, Qt.Horizontal, Qt.BackgroundRole))
        self.assertIn("9", header(2, Qt.Vertical, Qt.DisplayRole))
        self.assertIsNotNone(header(2, Qt.Vertical, Qt.BackgroundRole))

    def test_num_meta_labels_w_nan(self):
        x, y = (ContinuousVariable(c) for c in "xy")
        s = StringVariable("s")
        data = Table.from_list(
            Domain([x], [], [y, s]),
            [[0, 1, "a"],
             [1, np.nan, "b"]]
        )
        distances = Euclidean(data)
        self.widget.set_distances(distances)
        ac = self.widget.annot_combo
        idx = ac.model().indexOf(y)
        ac.setCurrentIndex(idx)
        ac.activated.emit(idx)

        header = self.widget.tablemodel.headerData
        self.assertEqual(header(0, Qt.Horizontal, Qt.DisplayRole), "1")
        self.assertEqual(header(1, Qt.Horizontal, Qt.DisplayRole), "?")
        self.assertIsNotNone(header(1, Qt.Horizontal, Qt.BackgroundRole))
        self.assertEqual(header(0, Qt.Vertical, Qt.DisplayRole), "1")
        self.assertEqual(header(1, Qt.Vertical, Qt.DisplayRole), "?")
        self.assertIsNotNone(header(1, Qt.Vertical, Qt.BackgroundRole))

    def test_choose_label(self):
        self.assertIs(OWDistanceMatrix._choose_label(self.iris),
                      self.iris.domain.class_var)

        domain = Domain([ContinuousVariable(x) for x in "xyz"],
                        ContinuousVariable("t"),
                        [ContinuousVariable("m")] +
                        [StringVariable(c) for c in "abc"]
                        )
        data = Table.from_numpy(
            domain,
            np.zeros((4, 3), dtype=float),
            np.arange(4, dtype=float),
            np.array([[0, "a", "a", "a"],
                      [1, "b", "b", "b"],
                      [2, "a", "c", "b"],
                      [0, "b", "a", "a"]])
        )
        self.assertIs(OWDistanceMatrix._choose_label(data),
                      domain.metas[2])
        domain2 = Domain(domain.attributes, domain.class_var, domain.metas[:-2])
        self.assertIs(OWDistanceMatrix._choose_label(data.transform(domain2)),
                      domain.metas[1])

    def test_non_square_labels(self):
        widget = self.widget
        ac = self.widget.annot_combo

        dist = DistMatrix([[1, 2, 3], [4, 5, 6]])
        dist.row_items = DistMatrix._labels_to_tables(["aa", "bb"])
        dist.col_items = DistMatrix._labels_to_tables(["cc", "dd", "ee"])
        self.send_signal(widget.Inputs.distances, dist)
        self.assertEqual(ac.model().rowCount(), 3)

        header = partial(widget.tablemodel.headerData, role=Qt.DisplayRole)
        ac.setCurrentIndex(0)
        ac.activated.emit(0)
        self.assertIsNone(header(1, Qt.Horizontal))
        self.assertIsNone(header(1, Qt.Vertical))

        ac.setCurrentIndex(1)
        ac.activated.emit(1)
        self.assertEqual(header(1, Qt.Horizontal), "2")
        self.assertEqual(header(1, Qt.Vertical), "2")

        ac.setCurrentIndex(2)
        ac.activated.emit(2)
        self.assertEqual(header(1, Qt.Horizontal), "dd")
        self.assertEqual(header(1, Qt.Vertical), "bb")

    @WidgetTest.skipNonEnglish
    def test_migrate_settings_v1_and_use_them(self):
        ind = [1, 2, 5, 6, 7, 8]
        context = Context(
            values={'__version__': 1},
            dim=10,
            annotations=['None', 'Enumerate',
                         'sepal length', 'sepal width', 'petal length',
                         'petal width', 'iris'],
            annotation='petal length',
            selection=ind)
        widget = self.create_widget(
            OWDistanceMatrix, stored_settings={"__version__": 1, "context_settings": [context]})
        iris = Table("iris")[:10]
        distances = Euclidean(iris)
        self.send_signal(widget.Inputs.distances, distances)
        self.assertEqual(widget.annotation_idx, 4)
        self.assertEqual(widget.tableview.selectionModel().selectedItems(), ind)
        outm = self.get_output(widget.Outputs.distances)
        np.testing.assert_equal(outm, distances.submatrix(ind, ind))

    def test_square_settings(self):
        widget = self.widget
        self.send_signal(widget.Inputs.distances, self.distances)
        widget._set_selection([0, 3, 4])
        widget.annotation_idx = 3

        self.send_signal(widget.Inputs.distances, None)
        self.assertEqual(widget._get_selection(), ([], True))
        self.assertEqual(widget.annotation_idx, 0)

        self.send_signal(widget.Inputs.distances, self.distances)
        self.assertEqual(widget._get_selection(), ([0, 3, 4], True))
        self.assertEqual(widget.annotation_idx, 3)

        matrix = DistMatrix(np.array([[1, 2, 3], [4, 5, 6]]))
        matrix.row_items = list("ab")

        self.send_signal(widget.Inputs.distances,matrix)
        self.assertEqual(widget._get_selection(), (([], []), False))
        self.assertEqual(widget.annotation_idx, 2)

        widget._set_selection(([0], [0, 2]))
        widget.annotation_idx = 0

        self.send_signal(widget.Inputs.distances, self.distances)
        self.assertEqual(widget._get_selection(), ([0, 3, 4], True))
        self.assertEqual(widget.annotation_idx, 3)

        self.send_signal(widget.Inputs.distances, matrix)
        self.assertEqual(widget._get_selection(), (([0], [0, 2]), False))
        self.assertEqual(widget.annotation_idx, 0)


if __name__ == "__main__":
    unittest.main()
