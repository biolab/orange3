"""Pythagorean tree viewer for visualizing trees.

The widget currently supports 3 different kinds of trees that may come in handy
though only 2 are implemented - (GENERAL, CLASSIFICATION, REGRESSION) with
(CLASSIFICATION and REGRESSION implemented.

These modes are necessary since it is impossible to have one tree viewer for
both classification and regression type trees, since they need specialized
tootips and colors.

The general tree exists for the purpose of easy extension, since a new `Tree`
type has been introduced to Orange, which this widget accepts (both the
classification and regression tree extend that base Tree class). In case the
widget is ever required to support more general trees, the enum type is there
to be used, and the methods need to be implemented, then it should work for any
kind of trees.

"""
from math import sqrt, log

import numpy as np
from Orange.widgets.visualize.utils.scene import \
    UpdateItemsOnSelectGraphicsScene
from Orange.widgets.visualize.utils.view import (
    PannableGraphicsView,
    ZoomableGraphicsView,
    PreventDefaultWheelEvent
)
from PyQt4 import QtGui

from Orange.base import Tree
from Orange.classification.tree import TreeClassifier
from Orange.data.table import Table
from Orange.regression.tree import TreeRegressor
from Orange.widgets import gui, settings
from Orange.widgets.utils.colorpalette import ContinuousPaletteGenerator
from Orange.widgets.visualize.pythagorastreeviewer import (
    PythagorasTreeViewer,
    SquareGraphicsItem
)
from Orange.widgets.visualize.utils.owlegend import (
    AnchorableGraphicsView,
    Anchorable,
    OWDiscreteLegend,
    OWContinuousLegend
)
from Orange.widgets.visualize.utils.tree.skltreeadapter import \
    SklTreeAdapter
from Orange.widgets.widget import OWWidget


class OWPythagorasTree(OWWidget):
    name = 'Pythagorean Tree'
    description = 'Pythagorean Tree visualization for tree like-structures.'
    icon = 'icons/PythagoreanTree.svg'

    priority = 1000

    inputs = [('Tree', Tree, 'set_tree')]
    outputs = [('Selected Data', Table)]

    # Enable the save as feature
    graph_name = 'scene'

    # Settings
    depth_limit = settings.ContextSetting(10)
    target_class_index = settings.ContextSetting(0)
    size_calc_idx = settings.Setting(0)
    size_log_scale = settings.Setting(2)
    tooltips_enabled = settings.Setting(True)
    show_legend = settings.Setting(False)

    GENERAL, CLASSIFICATION, REGRESSION = range(3)

    LEGEND_OPTIONS = {
        'corner': Anchorable.BOTTOM_RIGHT,
        'offset': (10, 10),
    }

    def __init__(self):
        super().__init__()
        # Instance variables
        self.tree_type = self.GENERAL
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
            # The +1 is there so that we don't get division by 0 exceptions
            ('Logarithmic', lambda x: log(x * self.size_log_scale + 1)),
        ]

        # Color modes for regression trees
        self.REGRESSION_COLOR_CALC = [
            ('None', lambda _, __: QtGui.QColor(255, 255, 255)),
            ('Class mean', self._color_class_mean),
            ('Standard deviation', self._color_stddev),
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
            orientation='horizontal', items=[], contentsLength=8,
            callback=self.update_colors)
        self.size_calc_combo = gui.comboBox(
            box_display, self, 'size_calc_idx', label='Size',
            orientation='horizontal',
            items=list(zip(*self.SIZE_CALCULATION))[0], contentsLength=8,
            callback=self.update_size_calc)
        self.log_scale_box = gui.hSlider(
            box_display, self, 'size_log_scale',
            label='Log scale factor', minValue=1, maxValue=100, ticks=False,
            callback=self.invalidate_tree)

        # Plot properties area
        box_plot = gui.widgetBox(self.controlArea, 'Plot Properties')
        gui.checkBox(
            box_plot, self, 'tooltips_enabled', label='Enable tooltips',
            callback=self.update_tooltip_enabled)
        gui.checkBox(
            box_plot, self, 'show_legend', label='Show legend',
            callback=self.update_show_legend)

        # Stretch to fit the rest of the unsused area
        gui.rubber(self.controlArea)

        self.controlArea.setSizePolicy(
            QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)

        # MAIN AREA
        # The QGraphicsScene doesn't actually require a parent, but not linking
        # the widget to the scene causes errors and a segfault on close due to
        # the way Qt deallocates memory and deletes objects.
        self.scene = TreeGraphicsScene(self)
        self.scene.selectionChanged.connect(self.commit)
        self.view = TreeGraphicsView(self.scene, padding=(150, 150))
        self.view.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.mainArea.layout().addWidget(self.view)

        self.ptree = PythagorasTreeViewer()
        self.scene.addItem(self.ptree)
        self.view.set_central_widget(self.ptree)

        self.resize(800, 500)
        # Clear the widget to correctly set the intial values
        self.clear()

    def set_tree(self, model=None):
        """When a different tree is given."""
        self.clear()
        self.model = model

        if model is not None:
            # We need to know what kind of tree we have in order to properly
            # show colors and tooltips
            if isinstance(model, TreeClassifier):
                self.tree_type = self.CLASSIFICATION
            elif isinstance(model, TreeRegressor):
                self.tree_type = self.REGRESSION
            else:
                self.tree_type = self.GENERAL

            self.instances = model.instances
            # this bit is important for the regression classifier
            if self.instances is not None and \
                    self.instances.domain != model.domain:
                self.clf_dataset = Table.from_table(
                    self.model.domain, self.instances)
            else:
                self.clf_dataset = self.instances

            self.tree_adapter = self._get_tree_adapter(self.model)
            self.color_palette = self._tree_specific('_get_color_palette')()

            self.ptree.clear()
            self.ptree.set_tree(self.tree_adapter)
            self.ptree.set_tooltip_func(self._tree_specific('_get_tooltip'))
            self.ptree.set_node_color_func(
                self._tree_specific('_get_node_color')
            )

            self._tree_specific('_update_legend_colors')()
            self._update_legend_visibility()

            self._update_info_box()
            self._update_depth_slider()

            self._tree_specific('_update_target_class_combo')()

            self._update_main_area()

            # Get meta variables describing pythagoras tree if given from
            # forest.
            if hasattr(model, 'meta_size_calc_idx'):
                self.size_calc_idx = model.meta_size_calc_idx
            if hasattr(model, 'meta_size_log_scale'):
                self.size_log_scale = model.meta_size_log_scale
            # Updating the size calc redraws the whole tree
            if hasattr(model, 'meta_size_calc_idx') or \
                    hasattr(model, 'meta_size_log_scale'):
                self.update_size_calc()
            # The target class can also be passed from the meta properties
            if hasattr(model, 'meta_target_class_index'):
                self.target_class_index = model.meta_target_class_index
                self.update_colors()
            # TODO this messes up the viewport in pythagoras tree viewer
            # it seems the viewport doesn't reset its size if this is applied
            # if hasattr(model, 'meta_depth_limit'):
            #     self.depth_limit = model.meta_depth_limit
            #     self.update_depth()

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

    # CONTROL AREA CALLBACKS
    def update_depth(self):
        """This method should be called when the depth changes"""
        self.ptree.set_depth_limit(self.depth_limit)

    def update_colors(self):
        """When the target class / node coloring needs to be updated."""
        self.ptree.target_class_has_changed()
        self._tree_specific('_update_legend_colors')()

    def update_size_calc(self):
        """When the tree size calculation is updated."""
        self._update_log_scale_slider()
        self.invalidate_tree()

    def invalidate_tree(self):
        """When the tree needs to be recalculated. E.g. change of size calc."""
        if self.model is not None:
            self.tree_adapter = self._get_tree_adapter(self.model)
            self.ptree.set_tree(self.tree_adapter)
            self.ptree.set_depth_limit(self.depth_limit)
            self._update_main_area()

    def update_tooltip_enabled(self):
        """When the tooltip visibility is changed and need to be updated."""
        if self.tooltips_enabled:
            self.ptree.set_tooltip_func(
                self._tree_specific('_get_tooltip')
            )
        else:
            self.ptree.set_tooltip_func(lambda _: None)
        self.ptree.tooltip_has_changed()

    def update_show_legend(self):
        """When the legend visibility needs to be updated."""
        self._update_legend_visibility()

    # MODEL CHANGED CONTROL ELEMENTS UPDATE METHODS
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

    # MODEL REMOVED CONTROL ELEMENTS CLEAR METHODS
    def _clear_info_box(self):
        self.info.setText('No tree on input')

    def _clear_depth_slider(self):
        self.depth_slider.parent().setEnabled(False)
        self.depth_slider.setMaximum(0)

    def _clear_target_class_combo(self):
        self.target_class_combo.clear()
        self.target_class_index = 0
        self.target_class_combo.setCurrentIndex(self.target_class_index)

    # HELPFUL METHODS
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
        return SklTreeAdapter(
            model.tree,
            model.domain,
            adjust_weight=self.SIZE_CALCULATION[self.size_calc_idx][1],
        )

    def onDeleteWidget(self):
        """When deleting the widget."""
        super().onDeleteWidget()
        self.clear()

    def commit(self):
        """Commit the selected data to output."""
        if self.instances is None:
            self.send('Selected Data', None)
            return
        # this is taken almost directly from the owclassificationtreegraph.py
        items = filter(lambda x: isinstance(x, SquareGraphicsItem),
                       self.scene.selectedItems())

        data = self.tree_adapter.get_instances_in_nodes(
            self.clf_dataset, [item.tree_node for item in items])
        self.send('Selected Data', data)

    def send_report(self):
        """Send report."""
        self.report_plot()

    def _tree_specific(self, method):
        """A best effort method getter that somewhat separates logic specific
        to classification and regression trees.
        This relies on conventional naming of specific methods, e.g.
        a method name _get_tooltip would need to be defined like so:
        _classification_get_tooltip and _regression_get_tooltip, since they are
        both specific.

        Parameters
        ----------
        method : str
            Method name that we would like to call.

        Returns
        -------
        callable or None

        """
        if self.tree_type == self.GENERAL:
            return getattr(self, '_general' + method)
        elif self.tree_type == self.CLASSIFICATION:
            return getattr(self, '_classification' + method)
        elif self.tree_type == self.REGRESSION:
            return getattr(self, '_regression' + method)
        else:
            return None

    # CLASSIFICATION TREE SPECIFIC METHODS
    def _classification_update_target_class_combo(self):
        self._clear_target_class_combo()
        list(filter(
            lambda x: isinstance(x, QtGui.QLabel),
            self.target_class_combo.parent().children()
        ))[0].setText('Target class')
        self.target_class_combo.addItem('None')
        values = [c.title() for c in
                  self.tree_adapter.domain.class_vars[0].values]
        self.target_class_combo.addItems(values)

    def _classification_update_legend_colors(self):
        if self.legend is not None:
            self.scene.removeItem(self.legend)

        if self.target_class_index == 0:
            self.legend = OWDiscreteLegend(domain=self.model.domain,
                                           **self.LEGEND_OPTIONS)
        else:
            items = (
                (self.target_class_combo.itemText(self.target_class_index),
                 self.color_palette[self.target_class_index - 1]),
                ('other', QtGui.QColor('#ffffff'))
            )
            self.legend = OWDiscreteLegend(items=items, **self.LEGEND_OPTIONS)

        self.legend.setVisible(self.show_legend)
        self.scene.addItem(self.legend)

    def _classification_get_color_palette(self):
        return [QtGui.QColor(*c) for c in self.model.domain.class_var.colors]

    def _classification_get_node_color(self, adapter, tree_node):
        # this is taken almost directly from the existing classification tree
        # viewer
        colors = self.color_palette
        distribution = adapter.get_distribution(tree_node.label)[0]
        total = np.sum(distribution)

        if self.target_class_index:
            p = distribution[self.target_class_index - 1] / total
            color = colors[self.target_class_index - 1].light(200 - 100 * p)
        else:
            modus = np.argmax(distribution)
            p = distribution[modus] / (total or 1)
            color = colors[int(modus)].light(400 - 300 * p)
        return color

    def _classification_get_tooltip(self, node):
        distribution = self.tree_adapter.get_distribution(node.label)[0]
        total = int(np.sum(distribution))
        if self.target_class_index:
            samples = distribution[self.target_class_index - 1]
            text = ''
        else:
            modus = np.argmax(distribution)
            samples = distribution[modus]
            text = self.tree_adapter.domain.class_vars[0].values[modus] + \
                '<br>'
        ratio = samples / np.sum(distribution)

        rules = self.tree_adapter.rules(node.label)
        sorted_rules = sorted(rules[:-1], key=lambda rule: rule.attr_name)
        rules_str = ''
        if len(rules):
            rules_str += '<br>'.join(str(rule) for rule in sorted_rules)
            rules_str += '<br><b>%s</b>' % rules[-1]

        splitting_attr = self.tree_adapter.attribute(node.label)

        return '<p>' \
            + text \
            + '{}/{} samples ({:2.3f}%)'.format(
                int(samples), total, ratio * 100) \
            + '<hr>' \
            + ('Split by ' + splitting_attr.name
               if not self.tree_adapter.is_leaf(node.label) else '') \
            + ('<br><br>'
               if len(rules) and not self.tree_adapter.is_leaf(node.label)
               else '') \
            + rules_str \
            + '</p>'

    # REGRESSION TREE SPECIFIC METHODS
    def _regression_update_target_class_combo(self):
        self._clear_target_class_combo()
        list(filter(
            lambda x: isinstance(x, QtGui.QLabel),
            self.target_class_combo.parent().children()
        ))[0].setText('Node color')
        self.target_class_combo.addItems(
            list(zip(*self.REGRESSION_COLOR_CALC))[0])
        self.target_class_combo.setCurrentIndex(self.target_class_index)

    def _regression_update_legend_colors(self):
        if self.legend is not None:
            self.scene.removeItem(self.legend)

        def _get_colors_domain(domain):
            class_var = domain.class_var
            start, end, pass_through_black = class_var.colors
            if pass_through_black:
                lst_colors = [QtGui.QColor(*c) for c
                              in [start, (0, 0, 0), end]]
            else:
                lst_colors = [QtGui.QColor(*c) for c in [start, end]]
            return lst_colors

        # Currently, the first index just draws the outline without any color
        if self.target_class_index == 0:
            self.legend = None
            return
        # The colors are the class mean
        elif self.target_class_index == 1:
            values = (np.min(self.clf_dataset.Y), np.max(self.clf_dataset.Y))
            colors = _get_colors_domain(self.model.domain)
            while len(values) != len(colors):
                values.insert(1, -1)

            self.legend = OWContinuousLegend(items=list(zip(values, colors)),
                                             **self.LEGEND_OPTIONS)
        # Colors are the stddev
        elif self.target_class_index == 2:
            values = (0, np.std(self.clf_dataset.Y))
            colors = _get_colors_domain(self.model.domain)
            while len(values) != len(colors):
                values.insert(1, -1)

            self.legend = OWContinuousLegend(items=list(zip(values, colors)),
                                             **self.LEGEND_OPTIONS)

        self.legend.setVisible(self.show_legend)
        self.scene.addItem(self.legend)

    def _regression_get_color_palette(self):
        return ContinuousPaletteGenerator(
            *self.tree_adapter.domain.class_var.colors)

    def _regression_get_node_color(self, adapter, tree_node):
        return self.REGRESSION_COLOR_CALC[self.target_class_index][1](
            adapter, tree_node
        )

    def _color_class_mean(self, adapter, tree_node):
        # calculate node colors relative to the mean of the node samples
        min_mean = np.min(self.clf_dataset.Y)
        max_mean = np.max(self.clf_dataset.Y)
        instances = adapter.get_instances_in_nodes(self.clf_dataset, tree_node)
        mean = np.mean(instances.Y)

        return self.color_palette[(mean - min_mean) / (max_mean - min_mean)]

    def _color_stddev(self, adapter, tree_node):
        # calculate node colors relative to the standard deviation in the node
        # samples
        min_mean, max_mean = 0, np.std(self.clf_dataset.Y)
        instances = adapter.get_instances_in_nodes(self.clf_dataset, tree_node)
        std = np.std(instances.Y)

        return self.color_palette[(std - min_mean) / (max_mean - min_mean)]

    def _regression_get_tooltip(self, node):
        total = self.tree_adapter.num_samples(
            self.tree_adapter.parent(node.label))
        samples = self.tree_adapter.num_samples(node.label)
        ratio = samples / total

        instances = self.tree_adapter.get_instances_in_nodes(
            self.clf_dataset, node)
        mean = np.mean(instances.Y)
        std = np.std(instances.Y)

        rules = self.tree_adapter.rules(node.label)
        sorted_rules = sorted(rules[:-1], key=lambda rule: rule.attr_name)
        rules_str = ''
        if len(rules):
            rules_str += '<br>'.join(str(rule) for rule in sorted_rules)
            rules_str += '<br><b>%s</b>' % rules[-1]

        splitting_attr = self.tree_adapter.attribute(node.label)

        return '<p>Mean: {:2.3f}'.format(mean) \
            + '<br>Standard deviation: {:2.3f}'.format(std) \
            + '<br>{}/{} samples ({:2.3f}%)'.format(
              int(samples), total, ratio * 100) \
            + '<hr>' \
            + ('Split by ' + splitting_attr.name
               if not self.tree_adapter.is_leaf(node.label) else '') \
            + ('<br><br>' if len(rules) and not self.tree_adapter.is_leaf(
               node.label) else '') \
            + rules_str \
            + '</p>'


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
    import sys
    import Orange
    from Orange.classification.tree import TreeLearner

    argv = sys.argv
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = 'iris'

    app = QtGui.QApplication(argv)
    ow = OWPythagorasTree()
    data = Orange.data.Table(filename)
    clf = TreeLearner(max_depth=1000)(data)
    ow.set_tree(clf)

    ow.show()
    ow.raise_()
    ow.handleNewSignals()
    app.exec_()

    sys.exit(0)


if __name__ == '__main__':
    main()
