from PyQt4.QtGui import QGraphicsScene, QGraphicsView
from PyQt4.QtCore import Qt

from ...gui import test
from ..editlinksdialog import EditLinksDialog, EditLinksNode, \
                              GraphicsTextWidget
from ...scheme import SchemeNode


class TestLinksEditDialog(test.QAppTestCase):
    def test_links_edit(self):
        from ...registry.tests import small_testing_registry

        dlg = EditLinksDialog()
        reg = small_testing_registry()
        file_desc = reg.widget("Orange.OrangeWidgets.Data.OWFile.OWFile")
        bayes_desc = reg.widget("Orange.OrangeWidgets.Classify.OWNaiveBayes."
                                "OWNaiveBayes")
        source_node = SchemeNode(file_desc, title="This is File")
        sink_node = SchemeNode(bayes_desc)

        source_channel = source_node.output_channel("Data")
        sink_channel = sink_node.input_channel("Data")
        links = [(source_channel, sink_channel)]

        dlg.setNodes(source_node, sink_node)

        dlg.show()
        dlg.setLinks(links)

        self.assertSequenceEqual(dlg.links(), links)
        status = dlg.exec_()

        self.assertTrue(dlg.links() == [] or \
                        dlg.links() == links)

    def test_graphicstextwidget(self):
        scene = QGraphicsScene()
        view = QGraphicsView(scene)

        text = GraphicsTextWidget()
        text.setHtml("<center><b>a text</b></center><p>paragraph</p>")
        scene.addItem(text)
        view.show()
        view.resize(400, 300)

        self.app.exec_()

    def test_editlinksnode(self):
        from ...registry.tests import small_testing_registry

        reg = small_testing_registry()
        file_desc = reg.widget("Orange.OrangeWidgets.Data.OWFile.OWFile")
        bayes_desc = reg.widget("Orange.OrangeWidgets.Classify.OWNaiveBayes."
                                "OWNaiveBayes")
        source_node = SchemeNode(file_desc, title="This is File")
        sink_node = SchemeNode(bayes_desc)

        scene = QGraphicsScene()
        view = QGraphicsView(scene)

        node = EditLinksNode(node=source_node)
        scene.addItem(node)

        node = EditLinksNode(direction=Qt.RightToLeft)
        node.setSchemeNode(sink_node)

        node.setPos(300, 0)
        scene.addItem(node)

        view.show()
        view.resize(800, 300)
        self.app.exec_()
