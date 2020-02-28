# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import warnings

import numpy as np

from AnyQt.QtCore import QPoint, Qt
from AnyQt.QtWidgets import QGraphicsScene, QGraphicsView
from AnyQt.QtTest import QTest

from orangewidget.tests.base import GuiTest
import Orange.misc
from Orange.clustering import hierarchical
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.distance import Euclidean
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.unsupervised.owhierarchicalclustering import \
    OWHierarchicalClustering, DendrogramWidget


class TestOWHierarchicalClustering(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.distances = Euclidean(cls.data)
        cls.signal_name = "Distances"
        cls.signal_data = cls.distances
        cls.same_input_output_domain = False

    def setUp(self):
        self.widget = self.create_widget(OWHierarchicalClustering)

    def _select_data(self):
        items = self.widget.dendrogram._items
        cluster = items[sorted(list(items.keys()))[4]]
        self.widget.dendrogram.set_selected_items([cluster])
        return [14, 15, 32, 33]

    def _compare_selected_annotated_domains(self, selected, annotated):
        self.assertEqual(annotated.domain.variables,
                         selected.domain.variables)
        self.assertNotIn("Other", selected.domain.metas[0].values)
        self.assertIn("Other", annotated.domain.metas[0].values)
        self.assertLess(set(var.name for var in selected.domain.metas),
                        set(var.name for var in annotated.domain.metas))

    def test_selection_box_output(self):
        """Check output if Selection method changes"""
        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.annotated_data))

        # change selection to 'Height ratio'
        self.widget.selection_box.buttons[1].click()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.annotated_data))

        # change selection to 'Top N'
        self.widget.selection_box.buttons[2].click()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.annotated_data))

    def test_all_zero_inputs(self):
        d = Orange.misc.DistMatrix(np.zeros((10, 10)))
        self.widget.set_distances(d)

    def test_annotation_settings_retrieval(self):
        """Check whether widget retrieves correct settings for annotation"""
        widget = self.widget

        dist_names = Orange.misc.DistMatrix(
            np.zeros((4, 4)), self.data, axis=0)
        dist_no_names = Orange.misc.DistMatrix(np.zeros((10, 10)), axis=1)

        self.send_signal(self.widget.Inputs.distances, self.distances)
        # Check that default is set (class variable)
        self.assertEqual(widget.annotation, self.data.domain.class_var)

        var2 = self.data.domain[2]
        widget.annotation = var2
        # Iris now has var2 as annotation

        self.send_signal(self.widget.Inputs.distances, dist_no_names)
        self.assertEqual(widget.annotation, "Enumeration")  # Check default
        widget.annotation = "None"
        # Pure matrix with axis=1 now has None as annotation

        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertIs(widget.annotation, var2)
        self.send_signal(self.widget.Inputs.distances, dist_no_names)
        self.assertEqual(widget.annotation, "None")

        self.send_signal(self.widget.Inputs.distances, dist_names)
        self.assertEqual(widget.annotation, "Name")  # Check default
        widget.annotation = "Enumeration"
        # Pure matrix with axis=1 has Enumerate as annotation

        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertIs(widget.annotation, var2)
        self.send_signal(self.widget.Inputs.distances, dist_no_names)
        self.assertEqual(widget.annotation, "None")
        self.send_signal(self.widget.Inputs.distances, dist_names)
        self.assertEqual(widget.annotation, "Enumeration")
        self.send_signal(self.widget.Inputs.distances, dist_no_names)
        self.assertEqual(widget.annotation, "None")

    def test_domain_loses_class(self):
        widget = self.widget
        self.send_signal(self.widget.Inputs.distances, self.distances)
        data = self.data[:, :4]
        distances = Euclidean(data)
        self.send_signal(self.widget.Inputs.distances, distances)

    def test_infinite_distances(self):
        """
        Scipy does not accept infinite distances and neither does this widget.
        Error is shown.
        GH-2380
        """
        table = Table.from_list(
            Domain(
                [ContinuousVariable("a")],
                [DiscreteVariable("b", values=("y", ))]),
            list(zip([1.79e308, -1e120],
                     "yy"))
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*", RuntimeWarning)
            distances = Euclidean(table)
        self.assertFalse(self.widget.Error.not_finite_distances.is_shown())
        self.send_signal(self.widget.Inputs.distances, distances)
        self.assertTrue(self.widget.Error.not_finite_distances.is_shown())
        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertFalse(self.widget.Error.not_finite_distances.is_shown())

    def test_output_cut_ratio(self):
        self.send_signal(self.widget.Inputs.distances, self.distances)

        # no data is selected
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        annotated = self.get_output(self.widget.Outputs.annotated_data)
        self.assertIsNotNone(annotated)

        # selecting clusters with cutoff should select all data
        QTest.mousePress(
            self.widget.view.headerView().viewport(),
            Qt.LeftButton, Qt.NoModifier,
            QPoint(100, 10)
        )
        selected = self.get_output(self.widget.Outputs.selected_data)
        annotated = self.get_output(self.widget.Outputs.annotated_data)
        self.assertEqual(len(selected), len(self.data))
        self.assertIsNotNone(annotated)

    def test_retain_selection(self):
        """Hierarchical Clustering didn't retain selection. GH-1563"""
        self.send_signal(self.widget.Inputs.distances, self.distances)
        self._select_data()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.selected_data))
        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertIsNotNone(self.get_output(self.widget.Outputs.selected_data))

    def test_restore_state(self):
        self.send_signal(self.widget.Inputs.distances, self.distances)
        self._select_data()
        ids_1 = self.get_output(self.widget.Outputs.selected_data).ids
        state = self.widget.settingsHandler.pack_data(self.widget)
        w = self.create_widget(
            OWHierarchicalClustering, stored_settings=state
        )
        self.send_signal(w.Inputs.distances, self.distances, widget=w)
        ids_2 = self.get_output(w.Outputs.selected_data, widget=w).ids
        self.assertSequenceEqual(list(ids_1), list(ids_2))


class TestDendrogramWidget(GuiTest):
    def setUp(self) -> None:
        super().setUp()
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.widget = DendrogramWidget()
        self.scene.addItem(self.widget)

    def tearDown(self) -> None:
        self.scene.clear()
        del self.widget
        del self.view
        super().tearDown()

    def test_widget(self):
        w = self.widget

        T = hierarchical.Tree
        C = hierarchical.ClusterData
        S = hierarchical.SingletonData

        def t(h: float, left: T, right: T):
            return T(C((left.value.first, right.value.last), h), (left, right))

        def leaf(r, index):
            return T(S((r, r + 1), 0.0, index))

        T = hierarchical.Tree

        w.set_root(t(0.0, leaf(0, 0), leaf(1, 1)))
        w.resize(w.effectiveSizeHint(Qt.PreferredSize))
        h = w.height_at(QPoint())
        self.assertEqual(h, 0)
        h = w.height_at(QPoint(10, 0))
        self.assertEqual(h, 0)

        self.assertEqual(w.pos_at_height(0).x(), w.rect().x())
        self.assertEqual(w.pos_at_height(1).x(), w.rect().x())

        height = np.finfo(float).eps
        w.set_root(t(height, leaf(0, 0), leaf(1, 1)))

        h = w.height_at(QPoint())
        self.assertEqual(h, height)
        h = w.height_at(QPoint(w.size().width(), 0))
        self.assertEqual(h, 0)

        self.assertEqual(w.pos_at_height(0).x(), w.rect().right())
        self.assertEqual(w.pos_at_height(height).x(), w.rect().left())
