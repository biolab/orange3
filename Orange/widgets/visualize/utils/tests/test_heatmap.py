import numpy as np

from AnyQt.QtCore import Qt, QPoint
from AnyQt.QtGui import QFont
from AnyQt.QtTest import QTest, QSignalSpy
from AnyQt.QtWidgets import QGraphicsScene, QGraphicsView

from orangecanvas.gui.test import mouseMove
from orangewidget.tests.base import GuiTest

from Orange.clustering.hierarchical import Tree, SingletonData, ClusterData
from Orange.widgets.visualize.utils.heatmap import HeatmapGridWidget, \
    GradientColorMap, CategoricalColorMap, CategoricalColorLegend


class _GraphicsGuiTest(GuiTest):
    scene: QGraphicsScene
    view: QGraphicsView

    def setUp(self) -> None:
        super().setUp()
        self.view = QGraphicsView()
        self.scene = QGraphicsScene(self.view)
        self.view.setScene(self.scene)

    def tearDown(self) -> None:
        self.scene.clear()
        self.scene.deleteLater()
        self.scene = None
        self.view.deleteLater()
        self.view = None
        super().tearDown()


class TestHeatmapGridWidget(_GraphicsGuiTest):
    _c2 = Tree(ClusterData((0, 1), 0.5), (
        Tree(SingletonData((0, 0), 0, 0), ()),
        Tree(SingletonData((1, 1), 0, 1), ()),
    ))
    _Data = {
        "0-0": HeatmapGridWidget.Parts(
            rows=[], columns=[], data=np.zeros(shape=(0, 0)), span=(0, 0)
        ),
        "1-0": HeatmapGridWidget.Parts(
            rows=[], columns=[], data=np.zeros(shape=(0, 0)), span=(0, 0)
        ),
        "0-1": HeatmapGridWidget.Parts(
            rows=[], columns=[HeatmapGridWidget.ColumnItem("a", [0])],
            data=np.zeros(shape=(0, 1)), span=(0, 1)
        ),
        "1-1": HeatmapGridWidget.Parts(
            rows=[HeatmapGridWidget.RowItem("a", [0])],
            columns=[HeatmapGridWidget.ColumnItem("a", [0])],
            data=np.zeros(shape=(1, 1)), span=(0, 1),
            row_names=["a"], col_names=["b"],
        ),
        "2-2-split": HeatmapGridWidget.Parts(
            rows=[
                HeatmapGridWidget.RowItem("a", [0]),
                HeatmapGridWidget.RowItem("b", [1]),
            ],
            columns=[
                HeatmapGridWidget.ColumnItem("a", [0]),
                HeatmapGridWidget.ColumnItem("a", [1]),
            ],
            data=np.zeros(shape=(2, 2)), span=(-1, 1),
            row_names=["a", "b"],
            col_names=["b", "b"],
        ),
        "2-2-cl": HeatmapGridWidget.Parts(
            rows=[HeatmapGridWidget.RowItem("", [0, 1], _c2)],
            columns=[HeatmapGridWidget.ColumnItem("", [0, 1], _c2)],
            data=np.zeros(shape=(2, 2)), span=(-1, 1),
            row_names=["a", "b"],
            col_names=["b", "b"],
        ),
        "2-2": HeatmapGridWidget.Parts(
            rows=[HeatmapGridWidget.RowItem("", [0, 1])],
            columns=[HeatmapGridWidget.ColumnItem("", [0, 1])],
            data=np.zeros(shape=(2, 2)), span=(-1, 1),
            row_names=["a", "b"],
            col_names=["b", "b"],
        )
    }

    def test_widget(self):
        w = HeatmapGridWidget()
        self.scene.addItem(w)

        for p in self._Data.values():
            w.setHeatmaps(p)

        w.headerGeometry()
        w.footerGeometry()

    def test_widget_annotations(self):
        w = HeatmapGridWidget()
        self.scene.addItem(w)
        w.setHeatmaps(self._Data["2-2"])
        # Coverage. The game.
        w.setLegendVisible(True)
        w.setLegendVisible(False)

        w.setShowAverages(True)
        w.setShowAverages(False)

        w.setRowLabels(None)
        w.setRowLabels(["1", "2"])

        w.setRowLabelsVisible(False)
        w.setRowLabelsVisible(True)

        w.setColumnLabels(None)
        w.setColumnLabels(["1", "2"])

        w.setAspectRatioMode(Qt.IgnoreAspectRatio)
        w.setAspectRatioMode(Qt.KeepAspectRatio)
        w.setAspectRatioMode(Qt.KeepAspectRatioByExpanding)

        for pos in (
                HeatmapGridWidget.NoPosition,
                HeatmapGridWidget.PositionTop,
                HeatmapGridWidget.PositionBottom,
                HeatmapGridWidget.PositionTop | HeatmapGridWidget.PositionBottom,
        ):
            w.setColumnLabelsPosition(pos)

        w.setRowSideColorAnnotations(
            np.array([0, 1]),
            CategoricalColorMap(np.array([[255] * 3, [0] * 3]),
                                names=["a", "b"])
        )
        w.setRowSideColorAnnotations(None)

    def test_selection(self):
        w = HeatmapGridWidget()
        self.scene.addItem(w)
        w.setHeatmaps(self._Data["2-2"])
        view = self.view
        w.resize(w.effectiveSizeHint(Qt.PreferredSize))
        h = w.layout().itemAt(w.Row0, w.Col0 + 1)
        pos = view.mapFromScene(h.scenePos())
        spy = QSignalSpy(w.selectionFinished)
        QTest.mouseClick(
            view.viewport(), Qt.LeftButton, pos=pos + QPoint(1, 1)
        )
        self.assertSequenceEqual(list(spy), [[]])
        self.assertSequenceEqual(w.selectedRows(), [0])
        spy = QSignalSpy(w.selectionFinished)
        QTest.mouseClick(
            view.viewport(), Qt.LeftButton, Qt.ControlModifier,
            pos=pos + QPoint(1, 1)
        )
        self.assertSequenceEqual(list(spy), [[]])
        self.assertSequenceEqual(w.selectedRows(), [])

        spy = QSignalSpy(w.selectionFinished)
        QTest.mousePress(view.viewport(), Qt.LeftButton, pos=pos + QPoint(1, 1))
        mouseMove(view.viewport(), Qt.LeftButton, pos=pos + QPoint(20, 20))
        QTest.mouseRelease(view.viewport(), Qt.LeftButton,
                           pos=pos + QPoint(30, 40))
        self.assertSequenceEqual(list(spy), [[]])

        spy_fin = QSignalSpy(w.selectionFinished)
        spy_chn = QSignalSpy(w.selectionChanged)
        w.selectRows([1])
        self.assertSequenceEqual(list(spy_fin), [])
        self.assertSequenceEqual(list(spy_chn), [[]])

    def test_colormap(self):
        w = HeatmapGridWidget()
        self.scene.addItem(w)
        w.setHeatmaps(self._Data["2-2"])
        w.setColorMap(GradientColorMap([[255] * 3, [0] * 3]))
        w.setColorMap(GradientColorMap([[255] * 3, [0] * 3], center=0))


class TestCategoricalColorLegend(_GraphicsGuiTest):
    def ensure_scene_polished(self):
        self.view.grab()

    def test_font_propagation(self):
        cmap = CategoricalColorMap(np.array([[255] * 3, [0] * 3]),
                                   names=["a", "b"])
        w = CategoricalColorLegend(cmap, title="Title")
        self.scene.addItem(w)
        font = QFont("Windings")
        w.setFont(font)
        # needs to be polished for FontChange to be delivered
        self.ensure_scene_polished()
        self.assertEqual(w.layout().itemAt(0).item.font().family(),
                         font.family())
