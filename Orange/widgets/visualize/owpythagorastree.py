"""Pythagorean tree viewer for visualizing trees."""
from math import sqrt, log

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor, QPainter
from AnyQt.QtWidgets import QLabel, QSizePolicy

from Orange.base import TreeModel, SklModel
from Orange.data.table import Table
from Orange.widgets import gui, settings
from Orange.widgets.utils.annotated_data import (
    create_annotated_table,
    ANNOTATED_DATA_SIGNAL_NAME
)
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.visualize.pythagorastreeviewer import (
    PythagorasTreeViewer,
    SquareGraphicsItem,
    ContinuousTreeNode,
)
from Orange.widgets.visualize.utils.owlegend import (
    AnchorableGraphicsView,
    Anchorable,
    OWDiscreteLegend,
    OWContinuousLegend,
)
from Orange.widgets.visualize.utils.scene import \
    UpdateItemsOnSelectGraphicsScene
from Orange.widgets.visualize.utils.tree.skltreeadapter import SklTreeAdapter
from Orange.widgets.visualize.utils.tree.treeadapter import TreeAdapter
from Orange.widgets.visualize.utils.view import (
    PannableGraphicsView,
    ZoomableGraphicsView,
    PreventDefaultWheelEvent,
)
from Orange.widgets.widget import OWWidget


class OWPythagorasTree(OWWidget):
    name = 'Pythagorean Tree'
    description = 'Pythagorean Tree visualization for tree like-structures.'
    icon = 'icons/PythagoreanTree.svg'

    priority = 1000

    class Inputs:
        tree = Input("Tree", TreeModel)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    # Enable the save as feature
    graph_name = 'scene'

    # Settings
    depth_limit = settings.ContextSetting(10)
    target_class_index = settings.ContextSetting(0)
    size_calc_idx = settings.Setting(0)
    size_log_scale = settings.Setting(2)
    tooltips_enabled = settings.Setting(True)
    show_legend = settings.Setting(False)

    LEGEND_OPTIONS = {
        'corner': Anchorable.BOTTOM_RIGHT,
        'offset': (10, 10),
    }

    def __init__(self):
        super().__init__()
        # Instance variables
        self.model = None
        self.instances = None
        self.clf_dataset = None
        # The tree adapter instance which is passed from the outside
        self.tree_adapter = None
        self.legend = None

        self.color_palette = None

        # Different methods to calculate the size of squares
        self.SIZE_CALCULATION = [
            ('Normal', lambda x: x),
            ('Square root', lambda x: sqrt(x)),
            ('Logarithmic', lambda x: log(x * self.size_log_scale + 1)),
        ]

        # CONTROL AREA
        # Tree info area
        box_info = gui.widgetBox(self.controlArea, 'Tree Info')
        self.info = gui.widgetLabel(box_info)

        # Display settings area
        box_display = gui.widgetBox(self.controlArea, 'Display Settings')
        self.depth_slider = gui.hSlider(
            box_display, self, 'depth_limit', label='Depth', ticks=False,
            callback=self.update_depth)
        self.target_class_combo = gui.comboBox(
            box_display, self, 'target_class_index', label='Target class',
            orientation=Qt.Horizontal, items=[], contentsLength=8,
            callback=self.update_colors)
        self.size_calc_combo = gui.comboBox(
            box_display, self, 'size_calc_idx', label='Size',
            orientation=Qt.Horizontal,
            items=list(zip(*self.SIZE_CALCULATION))[0], contentsLength=8,
            callback=self.update_size_calc)
        self.log_scale_box = gui.hSlider(
            box_display, self, 'size_log_scale',
            label='Log scale factor', minValue=1, maxValue=100, ticks=False,
            callback=self.invalidate_tree)

        # Plot properties area
        box_plot = gui.widgetBox(self.controlArea, 'Plot Properties')
        self.cb_show_tooltips = gui.checkBox(
            box_plot, self, 'tooltips_enabled', label='Enable tooltips',
            callback=self.update_tooltip_enabled)
        self.cb_show_legend = gui.checkBox(
            box_plot, self, 'show_legend', label='Show legend',
            callback=self.update_show_legend)

        # Stretch to fit the rest of the unsused area
        gui.rubber(self.controlArea)

        self.controlArea.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Expanding)

        # MAIN AREA
        self.scene = TreeGraphicsScene(self)
        self.scene.selectionChanged.connect(self.commit)
        self.view = TreeGraphicsView(self.scene, padding=(150, 150))
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.mainArea.layout().addWidget(self.view)

        self.ptree = PythagorasTreeViewer()
        self.scene.addItem(self.ptree)
        self.view.set_central_widget(self.ptree)

        self.resize(800, 500)
        # Clear the widget to correctly set the intial values
        self.clear()

    @Inputs.tree
    def set_tree(self, model=None):
        """When a different tree is given."""
        self.clear()
        self.model = model

        if model is not None:
            self.instances = model.instances
            # this bit is important for the regression classifier
            if self.instances is not None and \
                    self.instances.domain != model.domain:
                self.clf_dataset = self.instances.transform(self.model.domain)
            else:
                self.clf_dataset = self.instances

            self.tree_adapter = self._get_tree_adapter(self.model)
            self.ptree.clear()

            self.ptree.set_tree(
                self.tree_adapter,
                weight_adjustment=self.SIZE_CALCULATION[self.size_calc_idx][1],
                target_class_index=self.target_class_index,
            )

            self._update_depth_slider()
            self.color_palette = self.ptree.root.color_palette
            self._update_legend_colors()
            self._update_legend_visibility()
            self._update_info_box()
            self._update_target_class_combo()

            self._update_main_area()

            # The target class can also be passed from the meta properties
            # This must be set after `_update_target_class_combo`
            if hasattr(model, 'meta_target_class_index'):
                self.target_class_index = model.meta_target_class_index
                self.update_colors()

            # Get meta variables describing what the settings should look like
            # if the tree is passed from the Pythagorean forest widget.
            if hasattr(model, 'meta_size_calc_idx'):
                self.size_calc_idx = model.meta_size_calc_idx
                self.update_size_calc()

            # TODO There is still something wrong with this
            # if hasattr(model, 'meta_depth_limit'):
            #     self.depth_limit = model.meta_depth_limit
            #     self.update_depth()

        self.Outputs.annotated_data.send(create_annotated_table(self.instances, None))

    def clear(self):
        """Clear all relevant data from the widget."""
        self.model = None
        self.instances = None
        self.clf_dataset = None
        self.tree_adapter = None

        if self.legend is not None:
            self.scene.removeItem(self.legend)
        self.legend = None

        self.ptree.clear()
        self._clear_info_box()
        self._clear_target_class_combo()
        self._clear_depth_slider()
        self._update_log_scale_slider()

    def update_depth(self):
        """This method should be called when the depth changes"""
        self.ptree.set_depth_limit(self.depth_limit)

    def update_colors(self):
        """When the target class / node coloring needs to be updated."""
        self.ptree.target_class_changed(self.target_class_index)
        self._update_legend_colors()

    def update_size_calc(self):
        """When the tree size calculation is updated."""
        self._update_log_scale_slider()
        self.invalidate_tree()

    def invalidate_tree(self):
        """When the tree needs to be completely recalculated."""
        if self.model is not None:
            self.ptree.set_tree(
                self.tree_adapter,
                weight_adjustment=self.SIZE_CALCULATION[self.size_calc_idx][1],
                target_class_index=self.target_class_index,
            )
            self.ptree.set_depth_limit(self.depth_limit)
            self._update_main_area()

    def update_tooltip_enabled(self):
        """When the tooltip visibility is changed and need to be updated."""
        self.ptree.tooltip_changed(self.tooltips_enabled)

    def update_show_legend(self):
        """When the legend visibility needs to be updated."""
        self._update_legend_visibility()

    def _update_info_box(self):
        self.info.setText('Nodes: {}\nDepth: {}'.format(
            self.tree_adapter.num_nodes,
            self.tree_adapter.max_depth
        ))

    def _update_depth_slider(self):
        self.depth_slider.parent().setEnabled(True)
        self.depth_slider.setMaximum(self.tree_adapter.max_depth)
        self._set_max_depth()

    def _update_legend_visibility(self):
        if self.legend is not None:
            self.legend.setVisible(self.show_legend)

    def _update_log_scale_slider(self):
        """On calc method combo box changed."""
        self.log_scale_box.parent().setEnabled(
            self.SIZE_CALCULATION[self.size_calc_idx][0] == 'Logarithmic')

    def _clear_info_box(self):
        self.info.setText('No tree on input')

    def _clear_depth_slider(self):
        self.depth_slider.parent().setEnabled(False)
        self.depth_slider.setMaximum(0)

    def _clear_target_class_combo(self):
        self.target_class_combo.clear()
        self.target_class_index = 0
        self.target_class_combo.setCurrentIndex(self.target_class_index)

    def _set_max_depth(self):
        """Set the depth to the max depth and update appropriate actors."""
        self.depth_limit = self.tree_adapter.max_depth
        self.depth_slider.setValue(self.depth_limit)

    def _update_main_area(self):
        # refresh the scene rect, cuts away the excess whitespace, and adds
        # padding for panning.
        self.scene.setSceneRect(self.view.central_widget_rect())
        # reset the zoom level
        self.view.recalculate_and_fit()
        self.view.update_anchored_items()

    def _get_tree_adapter(self, model):
        if isinstance(model, SklModel):
            return SklTreeAdapter(model)
        return TreeAdapter(model)

    def onDeleteWidget(self):
        """When deleting the widget."""
        super().onDeleteWidget()
        self.clear()

    def commit(self):
        """Commit the selected data to output."""
        if self.instances is None:
            self.Outputs.selected_data.send(None)
            self.Outputs.annotated_data.send(None)
            return
        nodes = [i.tree_node.label for i in self.scene.selectedItems()
                 if isinstance(i, SquareGraphicsItem)]
        data = self.tree_adapter.get_instances_in_nodes(nodes)
        self.Outputs.selected_data.send(data)
        selected_indices = self.tree_adapter.get_indices(nodes)
        self.Outputs.annotated_data.send(create_annotated_table(self.instances, selected_indices))

    def send_report(self):
        """Send report."""
        self.report_plot()

    def _update_target_class_combo(self):
        self._clear_target_class_combo()
        label = [x for x in self.target_class_combo.parent().children()
                 if isinstance(x, QLabel)][0]

        if self.instances.domain.has_discrete_class:
            label_text = 'Target class'
            values = [c.title() for c in self.instances.domain.class_vars[0].values]
            values.insert(0, 'None')
        else:
            label_text = 'Node color'
            values = list(ContinuousTreeNode.COLOR_METHODS.keys())
        label.setText(label_text)
        self.target_class_combo.addItems(values)
        self.target_class_combo.setCurrentIndex(self.target_class_index)

    def _update_legend_colors(self):
        if self.legend is not None:
            self.scene.removeItem(self.legend)

        if self.instances.domain.has_discrete_class:
            self._classification_update_legend_colors()
        else:
            self._regression_update_legend_colors()

    def _classification_update_legend_colors(self):
        if self.target_class_index == 0:
            self.legend = OWDiscreteLegend(domain=self.model.domain,
                                           **self.LEGEND_OPTIONS)
        else:
            items = (
                (self.target_class_combo.itemText(self.target_class_index),
                 self.color_palette[self.target_class_index - 1]),
                ('other', QColor('#ffffff'))
            )
            self.legend = OWDiscreteLegend(items=items, **self.LEGEND_OPTIONS)

        self.legend.setVisible(self.show_legend)
        self.scene.addItem(self.legend)

    def _regression_update_legend_colors(self):
        def _get_colors_domain(domain):
            class_var = domain.class_var
            start, end, pass_through_black = class_var.colors
            if pass_through_black:
                lst_colors = [QColor(*c) for c
                              in [start, (0, 0, 0), end]]
            else:
                lst_colors = [QColor(*c) for c in [start, end]]
            return lst_colors

        # The colors are the class mean
        if self.target_class_index == 1:
            values = (np.min(self.clf_dataset.Y), np.max(self.clf_dataset.Y))
            colors = _get_colors_domain(self.model.domain)
            while len(values) != len(colors):
                values.insert(1, -1)
            items = list(zip(values, colors))
        # Colors are the stddev
        elif self.target_class_index == 2:
            values = (0, np.std(self.clf_dataset.Y))
            colors = _get_colors_domain(self.model.domain)
            while len(values) != len(colors):
                values.insert(1, -1)
            items = list(zip(values, colors))
        else:
            items = None

        self.legend = OWContinuousLegend(items=items, **self.LEGEND_OPTIONS)
        self.legend.setVisible(self.show_legend)
        self.scene.addItem(self.legend)


class TreeGraphicsView(
        PannableGraphicsView,
        ZoomableGraphicsView,
        AnchorableGraphicsView,
        PreventDefaultWheelEvent
):
    """QGraphicsView that contains all functionality we will use to display
    tree."""
    pass


class TreeGraphicsScene(UpdateItemsOnSelectGraphicsScene):
    """QGraphicsScene that the tree uses."""
    pass


def main():
    from Orange.modelling import TreeLearner
    from AnyQt.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    ow = OWPythagorasTree()
    data = Table(sys.argv[1] if len(sys.argv) > 1 else 'iris')

    model = TreeLearner(max_depth=1000)(data)
    model.instances = data
    ow.set_tree(model)

    ow.show()
    ow.raise_()
    ow.handleNewSignals()
    app.exec_()


if __name__ == '__main__':
    main()
