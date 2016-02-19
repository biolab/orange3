from PyQt4.QtGui import QPainterPath, QGraphicsEllipseItem

from .. import NodeItem, AnchorPoint, NodeAnchorItem

from . import TestItems


class TestNodeItem(TestItems):
    def setUp(self):
        TestItems.setUp(self)
        from ....registry.tests import small_testing_registry
        self.reg = small_testing_registry()

        self.data_desc = self.reg.category("Data")
        self.classify_desc = self.reg.category("Classify")

        self.file_desc = self.reg.widget(
            "Orange.widgets.data.owfile.OWFile"
        )
        self.discretize_desc = self.reg.widget(
            "Orange.widgets.data.owdiscretize.OWDiscretize"
        )
        self.bayes_desc = self.reg.widget(
            "Orange.widgets.classify.ownaivebayes.OWNaiveBayes"
        )

    def test_nodeitem(self):
        file_item = NodeItem()
        file_item.setWidgetDescription(self.file_desc)
        file_item.setWidgetCategory(self.data_desc)

        file_item.setTitle("File Node")
        self.assertEqual(file_item.title(), "File Node")

        file_item.setProcessingState(True)
        self.assertEqual(file_item.processingState(), True)

        file_item.setProgress(50)
        self.assertEqual(file_item.progress(), 50)

        file_item.setProgress(100)
        self.assertEqual(file_item.progress(), 100)

        file_item.setProgress(101)
        self.assertEqual(file_item.progress(), 100, "Progress overshots")

        file_item.setProcessingState(False)
        self.assertEqual(file_item.processingState(), False)
        self.assertEqual(file_item.progress(), -1,
                         "setProcessingState does not clear the progress.")

        self.scene.addItem(file_item)
        file_item.setPos(100, 100)

        discretize_item = NodeItem()
        discretize_item.setWidgetDescription(self.discretize_desc)
        discretize_item.setWidgetCategory(self.data_desc)

        self.scene.addItem(discretize_item)
        discretize_item.setPos(300, 100)

        nb_item = NodeItem()
        nb_item.setWidgetDescription(self.bayes_desc)
        nb_item.setWidgetCategory(self.classify_desc)

        self.scene.addItem(nb_item)
        nb_item.setPos(500, 100)

        positions = []
        anchor = file_item.newOutputAnchor()
        anchor.scenePositionChanged.connect(positions.append)

        file_item.setPos(110, 100)
        self.assertTrue(len(positions) > 0)

        file_item.setErrorMessage("message")
        file_item.setWarningMessage("message")
        file_item.setInfoMessage("I am alive")

        file_item.setErrorMessage(None)
        file_item.setWarningMessage(None)
        file_item.setInfoMessage(None)

        file_item.setInfoMessage("I am back.")

        def progress():
            self.singleShot(10, progress)
            p = (nb_item.progress() + 1) % 100
            nb_item.setProgress(p)

            if p > 50:
                nb_item.setInfoMessage("Over 50%")
                file_item.setWarningMessage("Second")
            else:
                nb_item.setInfoMessage(None)
                file_item.setWarningMessage(None)

            discretize_item.setAnchorRotation(50 - p)

        progress()

        self.app.exec_()

    def test_nodeanchors(self):
        file_item = NodeItem()
        file_item.setWidgetDescription(self.file_desc)
        file_item.setWidgetCategory(self.data_desc)

        file_item.setTitle("File Node")

        self.scene.addItem(file_item)
        file_item.setPos(100, 100)

        discretize_item = NodeItem()
        discretize_item.setWidgetDescription(self.discretize_desc)
        discretize_item.setWidgetCategory(self.data_desc)

        self.scene.addItem(discretize_item)
        discretize_item.setPos(300, 100)

        nb_item = NodeItem()
        nb_item.setWidgetDescription(self.bayes_desc)
        nb_item.setWidgetCategory(self.classify_desc)

        with self.assertRaises(ValueError):
            file_item.newInputAnchor()

        anchor = file_item.newOutputAnchor()
        self.assertIsInstance(anchor, AnchorPoint)

        self.app.exec_()

    def test_anchoritem(self):
        anchoritem = NodeAnchorItem(None)
        self.scene.addItem(anchoritem)

        path = QPainterPath()
        path.addEllipse(0, 0, 100, 100)

        anchoritem.setAnchorPath(path)

        anchor = AnchorPoint()
        anchoritem.addAnchor(anchor)

        ellipse1 = QGraphicsEllipseItem(-3, -3, 6, 6)
        ellipse2 = QGraphicsEllipseItem(-3, -3, 6, 6)
        self.scene.addItem(ellipse1)
        self.scene.addItem(ellipse2)

        anchor.scenePositionChanged.connect(ellipse1.setPos)

        with self.assertRaises(ValueError):
            anchoritem.addAnchor(anchor)

        anchor1 = AnchorPoint()
        anchoritem.addAnchor(anchor1)

        anchor1.scenePositionChanged.connect(ellipse2.setPos)

        self.assertSequenceEqual(anchoritem.anchorPoints(), [anchor, anchor1])

        self.assertSequenceEqual(anchoritem.anchorPositions(), [0.5, 0.5])
        anchoritem.setAnchorPositions([0.5, 0.0])

        self.assertSequenceEqual(anchoritem.anchorPositions(), [0.5, 0.0])

        def advance():
            t = anchoritem.anchorPositions()
            t = [(t + 0.05) % 1.0 for t in t]
            anchoritem.setAnchorPositions(t)
            self.singleShot(20, advance)

        advance()

        self.app.exec_()
