# pylint: disable=all
from itertools import product
from unittest.mock import patch

import numpy as np
from AnyQt.QtCore import QSize
from AnyQt.QtGui import QImage, QPainter
from AnyQt.QtWidgets import QStyleOptionViewItem

from orangewidget.tests.base import GuiTest

from Orange.misc import DistMatrix
from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.distance import Euclidean
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owdistancematrix import OWDistanceMatrix, \
    DistanceMatrixModel, TableBorderItem


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

        # Distances without row data
        self.distances.row_items = None
        self.widget.set_distances(self.distances)
        self.assertNotIn(self.iris.domain[0], self.widget.annot_combo.model())

    def test_not_symmetric(self):
        w = self.widget
        self.send_signal(w.Inputs.distances, DistMatrix([[1, 2, 3], [4, 5, 6]]))
        self.assertTrue(w.Error.not_symmetric.is_shown())
        self.send_signal(w.Inputs.distances, None)
        self.assertFalse(w.Error.not_symmetric.is_shown())

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

        distances = Euclidean(grades)
        self.widget.set_distances(distances)
        ac = self.widget.annot_combo
        idx = ac.model().indexOf(grades.domain.metas[0])
        ac.setCurrentIndex(idx)
        ac.activated.emit(idx)
        self.assertIsNone(self.widget.tablemodel.label_colors)

    def test_num_meta_labels(self):
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
        self.assertEqual(self.widget.tablemodel.labels, ["1", "?"])

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


class TestDelegates(GuiTest):
    def test_delegate(self):
        model = DistanceMatrixModel()
        matrix = np.array([[0.0, 0.1, 0.2], [0.1, 0.0, 0.1], [0.2, 0.1, 0.0]])
        model.set_data(matrix)
        delegate = TableBorderItem()
        for row, col in product(range(model.rowCount()),
                                range(model.columnCount())):
            index = model.index(row, col)
            option = QStyleOptionViewItem()
            size = delegate.sizeHint(option, index).expandedTo(QSize(30, 18))
            delegate.initStyleOption(option, index)
            img = QImage(size, QImage.Format_ARGB32_Premultiplied)
            painter = QPainter(img)
            try:
                delegate.paint(painter, option, index)
            finally:
                painter.end()
