from PyQt4.QtGui import QBrush
from PyQt4.QtCore import Qt

from Orange.regression.tree import TreeRegressor
from Orange.widgets import gui
from Orange.widgets.settings import Setting, ClassValuesContextHandler
from Orange.widgets.classify.owclassificationtreegraph import (OWTreeGraph,
                                                               TreeNode)
from Orange.widgets.utils.colorpalette import ContinuousPaletteGenerator


class RegressionTreeNode(TreeNode):
    def impurity(self):
        """
        :return: Impurity in a particular node.
        """
        return self.tree.impurity[self.i]

    def get_distribution(self):
        """
        :return: Average value in a particular node.
        """
        return self.tree.value[self.node_id]


class OWRegressionTreeGraph(OWTreeGraph):
    name = "Regression Tree Viewer"
    description = "A graphical visualization of a regression tree."
    icon = "icons/RegressionTreeGraph.svg"
    priority = 35

    settingsHandler = ClassValuesContextHandler()
    color_index = Setting(0)
    color_settings = Setting(None)
    selected_color_settings_index = Setting(0)

    inputs = [("Regression Tree", TreeRegressor, "ctree")]
    NODE = RegressionTreeNode

    def __init__(self):
        super().__init__()
        box = gui.vBox(self.controlArea, "Nodes", addSpace=True)
        self.color_combo = gui.comboBox(
            box, self, "color_index", orientation=Qt.Horizontal, items=[],
            label="Colors", callback=self.toggle_color,
            contentsLength=8)
        gui.separator(box)
        gui.rubber(self.controlArea)

    def ctree(self, model=None):
        if model is not None:
            self.color_combo.clear()
            self.color_combo.addItem("Default")
            self.color_combo.addItem("Instances in node")
            self.color_combo.addItem("Impurity")
            self.color_combo.setCurrentIndex(self.color_index)
            self.scene.colorPalette = \
                ContinuousPaletteGenerator(*model.domain.class_var.colors)
        super().ctree(model)


    def update_node_info(self, node):
        distr = node.get_distribution()
        total = node.num_instances()
        total_tree = self.tree.n_node_samples[0]
        impurity = node.impurity()
        text = "{:2.1f}<br/>".format(sum(distr.reshape(1)))
        text += "{:2.1f}%, {}/{}<br/>".format(100 * total / total_tree,
                                              total, total_tree)
        text += "{:2.3f}".format(impurity)
        text = self._update_node_info_attr_name(node, text)
        node.setHtml('<p style="line-height: 120%; margin-bottom: 0">'
                     '{}</p>'.
                     format(text))

    def toggle_node_color(self):
        palette = self.scene.colorPalette
        all_instances = self.tree.n_node_samples[0]
        max_impurity = self.tree.impurity[0]
        for node in self.scene.nodes():
            li = [0.5, node.num_instances() / all_instances,
                  node.impurity() / max_impurity][self.color_index]
            node.backgroundBrush = QBrush(palette[self.color_index].lighter(
                180 - li * 150))
        self.scene.update()

    def send_report(self):
        if not self.tree:
            return
        self.report_items((
            ("Tree size", self.info.text()),
            ("Edge widths",
             ("Fixed", "Relative to root", "Relative to parent")[
                 self.line_width_method])))
        self.report_plot(self.scene)


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication
    from Orange.data import Table
    from Orange.regression.tree import TreeRegressionLearner

    a = QApplication(sys.argv)
    ow = OWRegressionTreeGraph()
    data = Table("housing")
    reg = TreeRegressionLearner(max_depth=5)(data)
    reg.instances = data

    ow.ctree(reg)
    ow.show()
    ow.raise_()
    a.exec_()
    ow.saveSettings()
