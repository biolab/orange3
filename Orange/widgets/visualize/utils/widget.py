from xml.sax.saxutils import escape

import numpy as np

from AnyQt.QtCore import QSize, Signal, Qt
from AnyQt.QtWidgets import QApplication

from orangewidget.utils.visual_settings_dlg import VisualSettingsDialog

from Orange.data import (
    Table, ContinuousVariable, Domain, Variable, StringVariable
)
from Orange.data.util import get_unique_names, array_equal
from Orange.data.sql.table import SqlTable
from Orange.statistics.util import bincount

from Orange.widgets import gui, report
from Orange.widgets.settings import (
    Setting, ContextSetting, DomainContextHandler, SettingProvider
)
from Orange.widgets.utils import colorpalettes
from Orange.widgets.utils.annotated_data import (
    create_annotated_table, ANNOTATED_DATA_SIGNAL_NAME, create_groups_table
)
from Orange.widgets.utils.plot import OWPlotGUI
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.visualize.owscatterplotgraph import (
    OWScatterPlotBase, MAX_COLORS
)
from Orange.widgets.visualize.utils.component import OWGraphWithAnchors
from Orange.widgets.widget import OWWidget, Input, Output, Msg


# maximum number of shapes (including Other)
MAX_SHAPES = len(OWScatterPlotBase.CurveSymbols) - 1

MAX_POINTS_IN_TOOLTIP = 5


class OWProjectionWidgetBase(OWWidget, openclass=True):
    """
    Base widget for widgets that use attribute data to set the colors, labels,
    shapes and sizes of points.

    The widgets defines settings `attr_color`, `attr_label`, `attr_shape`
    and `attr_size`, but leaves defining the gui to the derived widgets.
    These are expected to have controls that manipulate these settings,
    and the controls are expected to use attribute models.

    The widgets also defines attributes `data` and `valid_data` and expects
    the derived widgets to use them to store an instances of `data.Table`
    and a bool `np.ndarray` with indicators of valid (that is, shown)
    data points.
    """
    attr_color = ContextSetting(None, required=ContextSetting.OPTIONAL)
    attr_label = ContextSetting(None, required=ContextSetting.OPTIONAL)
    attr_shape = ContextSetting(None, required=ContextSetting.OPTIONAL)
    attr_size = ContextSetting(None, required=ContextSetting.OPTIONAL)

    class Information(OWWidget.Information):
        missing_size = Msg(
            "Points with undefined '{}' are shown in smaller size")
        missing_shape = Msg(
            "Points with undefined '{}' are shown as crossed circles")

    def __init__(self):
        super().__init__()
        self.data = None
        self.valid_data = None

    def init_attr_values(self):
        """
        Set the models for `attr_color`, `attr_shape`, `attr_size` and
        `attr_label`. All values are set to `None`, except `attr_color`
        which is set to the class variable if it exists.
        """
        data = self.data
        domain = data.domain if data and len(data) else None
        for attr in ("attr_color", "attr_shape", "attr_size", "attr_label"):
            getattr(self.controls, attr).model().set_domain(domain)
            setattr(self, attr, None)
        if domain is not None:
            self.attr_color = domain.class_var

    def get_coordinates_data(self):
        """A get coordinated method that returns no coordinates.

        Derived classes must override this method.
        """
        return None, None

    def get_subset_mask(self):
        """
        Return the bool array indicating the points in the subset

        The base method does nothing and would usually be overridden by
        a method that returns indicators from the subset signal.

        Do not confuse the subset with selection.

        Returns:
            (np.ndarray or `None`): a bool array of indicators
        """
        return None

    def get_column(self, attr, filter_valid=True,
                   max_categories=None, return_labels=False):
        """
        Retrieve the data from the given column in the data table

        The method:
        - densifies sparse data,
        - converts arrays with dtype object to floats if the attribute is
          actually primitive,
        - filters out invalid data (if `filter_valid` is `True`),
        - merges infrequent (discrete) values into a single value
          (if `max_categories` is set).

        Tha latter feature is used for shapes and labels, where only a
        specified number of different values is shown, and others are
        merged into category 'Other'. In this case, the method may return
        either the data (e.g. color indices, shape indices) or the list
        of retained values, followed by `['Other']`.

        Args:
            attr (:obj:~Orange.data.Variable): the column to extract
            filter_valid (bool): filter out invalid data (default: `True`)
            max_categories (int): merge infrequent values (default: `None`);
                ignored for non-discrete attributes
            return_labels (bool): return a list of labels instead of data
                (default: `False`)

        Returns:
            (np.ndarray): (valid) data from the column, or a list of labels
        """
        if attr is None:
            return None

        needs_merging = attr.is_discrete \
                        and max_categories is not None \
                        and len(attr.values) >= max_categories
        if return_labels and not needs_merging:
            assert attr.is_discrete
            return attr.values

        all_data = self.data.get_column_view(attr)[0]
        if all_data.dtype == object and attr.is_primitive():
            all_data = all_data.astype(float)
        if filter_valid and self.valid_data is not None:
            all_data = all_data[self.valid_data]
        if not needs_merging:
            return all_data

        dist = bincount(all_data, max_val=len(attr.values) - 1)[0]
        infrequent = np.zeros(len(attr.values), dtype=bool)
        infrequent[np.argsort(dist)[:-(max_categories-1)]] = True
        if return_labels:
            return [value for value, infreq in zip(attr.values, infrequent)
                    if not infreq] + ["Other"]
        else:
            result = all_data.copy()
            freq_vals = [i for i, f in enumerate(infrequent) if not f]
            for i, infreq in enumerate(infrequent):
                if infreq:
                    result[all_data == i] = max_categories - 1
                else:
                    result[all_data == i] = freq_vals.index(i)
            return result

    # Sizes
    def get_size_data(self):
        """Return the column corresponding to `attr_size`"""
        return self.get_column(self.attr_size)

    def impute_sizes(self, size_data):
        """
        Default imputation for size data

        Let the graph handle it, but add a warning if needed.

        Args:
            size_data (np.ndarray): scaled points sizes
        """
        if self.graph.default_impute_sizes(size_data):
            self.Information.missing_size(self.attr_size)
        else:
            self.Information.missing_size.clear()

    def sizes_changed(self):
        self.graph.update_sizes()

    # Colors
    def get_color_data(self):
        """Return the column corresponding to color data"""
        return self.get_column(self.attr_color, max_categories=MAX_COLORS)

    def get_color_labels(self):
        """
        Return labels for the color legend

        Returns:
            (list of str): labels
        """
        if self.attr_color is None:
            return None
        if not self.attr_color.is_discrete:
            return self.attr_color.str_val
        return self.get_column(self.attr_color, max_categories=MAX_COLORS,
                               return_labels=True)

    def is_continuous_color(self):
        """
        Tells whether the color is continuous

        Returns:
            (bool):
        """
        return self.attr_color is not None and self.attr_color.is_continuous

    def get_palette(self):
        """
        Return a palette suitable for the current `attr_color`

        This method must be overridden if the widget offers coloring that is
        not based on attribute values.
        """
        attr = self.attr_color
        if not attr:
            return None
        palette = attr.palette
        if attr.is_discrete and len(attr.values) >= MAX_COLORS:
            values = self.get_color_labels()
            colors = [palette.palette[attr.to_val(value)]
                      for value in values[:-1]] + [[192, 192, 192]]

            palette = colorpalettes.DiscretePalette.from_colors(colors)
        return palette

    def can_draw_density(self):
        """
        Tells whether the current data and settings are suitable for drawing
        densities

        Returns:
            (bool):
        """
        return self.data is not None and self.data.domain is not None and \
            len(self.data) > 1 and self.attr_color is not None

    def colors_changed(self):
        self.graph.update_colors()
        self._update_opacity_warning()
        self.cb_class_density.setEnabled(self.can_draw_density())

    # Labels
    def get_label_data(self, formatter=None):
        """Return the column corresponding to label data"""
        if self.attr_label:
            label_data = self.get_column(self.attr_label)
            if formatter is None:
                formatter = self.attr_label.str_val
            return np.array([formatter(x) for x in label_data])
        return None

    def labels_changed(self):
        self.graph.update_labels()

    # Shapes
    def get_shape_data(self):
        """
        Return labels for the shape legend

        Returns:
            (list of str): labels
        """
        return self.get_column(self.attr_shape, max_categories=MAX_SHAPES)

    def get_shape_labels(self):
        return self.get_column(self.attr_shape, max_categories=MAX_SHAPES,
                               return_labels=True)

    def impute_shapes(self, shape_data, default_symbol):
        """
        Default imputation for shape data

        Let the graph handle it, but add a warning if needed.

        Args:
            shape_data (np.ndarray): scaled points sizes
            default_symbol (str): a string representing the symbol
        """
        if self.graph.default_impute_shapes(shape_data, default_symbol):
            self.Information.missing_shape(self.attr_shape)
        else:
            self.Information.missing_shape.clear()

    def shapes_changed(self):
        self.graph.update_shapes()

    # Tooltip
    def _point_tooltip(self, point_id, skip_attrs=()):
        def show_part(_point_data, singular, plural, max_shown, _vars):
            cols = [escape('{} = {}'.format(var.name, _point_data[var]))
                    for var in _vars[:max_shown + 2]
                    if _vars == domain.class_vars
                    or var not in skip_attrs][:max_shown]
            if not cols:
                return ""
            n_vars = len(_vars)
            if n_vars > max_shown:
                cols[-1] = "... and {} others".format(n_vars - max_shown + 1)
            return \
                "<b>{}</b>:<br/>".format(singular if n_vars < 2 else plural) \
                + "<br/>".join(cols)

        domain = self.data.domain
        parts = (("Class", "Classes", 4, domain.class_vars),
                 ("Meta", "Metas", 4, domain.metas),
                 ("Feature", "Features", 10, domain.attributes))

        point_data = self.data[point_id]
        return "<br/>".join(show_part(point_data, *columns)
                            for columns in parts)

    def get_tooltip(self, point_ids):
        """
        Return the tooltip string for the given points

        The method is called by the plot on mouse hover

        Args:
            point_ids (list): indices into `data`

        Returns:
            (str):
        """
        point_ids = \
            np.flatnonzero(self.valid_data)[np.asarray(point_ids, dtype=int)]
        text = "<hr/>".join(self._point_tooltip(point_id)
                            for point_id in point_ids[:MAX_POINTS_IN_TOOLTIP])
        if len(point_ids) > MAX_POINTS_IN_TOOLTIP:
            text = "{} instances<hr/>{}<hr/>...".format(len(point_ids), text)
        return text

    def keyPressEvent(self, event):
        """Update the tip about using the modifier keys when selecting"""
        super().keyPressEvent(event)
        self.graph.update_tooltip(event.modifiers())

    def keyReleaseEvent(self, event):
        """Update the tip about using the modifier keys when selecting"""
        super().keyReleaseEvent(event)
        self.graph.update_tooltip(event.modifiers())


class OWDataProjectionWidget(OWProjectionWidgetBase, openclass=True):
    """
    Base widget for widgets that get Data and Data Subset (both
    Orange.data.Table) on the input, and output Selected Data and Data
    (both Orange.data.Table).

    Beside that the widget displays data as two-dimensional projection
    of points.
    """
    class Inputs:
        data = Input("Data", Table, default=True)
        data_subset = Input("Data Subset", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    class Warning(OWProjectionWidgetBase.Warning):
        too_many_labels = Msg(
            "Too many labels to show (zoom in or label only selected)")
        subset_not_subset = Msg(
            "Subset data contains some instances that do not appear in "
            "input data")
        subset_independent = Msg(
            "No subset data instances appear in input data")
        transparent_subset = Msg(
            "Increase opacity if subset is difficult to see")

    settingsHandler = DomainContextHandler()
    selection = Setting(None, schema_only=True)
    visual_settings = Setting({}, schema_only=True)
    auto_commit = Setting(True)

    GRAPH_CLASS = OWScatterPlotBase
    graph = SettingProvider(OWScatterPlotBase)
    graph_name = "graph.plot_widget.plotItem"
    embedding_variables_names = ("proj-x", "proj-y")
    buttons_area_orientation = Qt.Vertical

    input_changed = Signal(object)
    output_changed = Signal(object)

    def __init__(self):
        super().__init__()
        self.subset_data = None
        self.subset_indices = None
        self.__pending_selection = self.selection
        self._invalidated = True
        self._domain_invalidated = True
        self.setup_gui()
        VisualSettingsDialog(self, self.graph.parameter_setter.initial_settings)

    # GUI
    def setup_gui(self):
        self._add_graph()
        self._add_controls()
        self._add_buttons()
        self.input_changed.emit(None)
        self.output_changed.emit(None)

    def _add_graph(self):
        box = gui.vBox(self.mainArea, True, margin=0)
        self.graph = self.GRAPH_CLASS(self, box)
        box.layout().addWidget(self.graph.plot_widget)
        self.graph.too_many_labels.connect(
            lambda too_many: self.Warning.too_many_labels(shown=too_many))

    def _add_controls(self):
        self.gui = OWPlotGUI(self)
        area = self.controlArea
        self._point_box = self.gui.point_properties_box(area)
        self._effects_box = self.gui.effects_box(area)
        self._plot_box = self.gui.plot_properties_box(area)

    def _add_buttons(self):
        gui.rubber(self.controlArea)
        self.gui.box_zoom_select(self.buttonsArea)
        gui.auto_send(self.buttonsArea, self, "auto_commit")

    @property
    def effective_variables(self):
        return self.data.domain.attributes

    @property
    def effective_data(self):
        return self.data.transform(Domain(self.effective_variables,
                                          self.data.domain.class_vars,
                                          self.data.domain.metas))

    # Input
    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        data_existed = self.data is not None
        effective_data = self.effective_data if data_existed else None
        self.closeContext()
        self.data = data
        self.check_data()
        self.init_attr_values()
        self.openContext(self.data)
        self._invalidated = not (
            data_existed and self.data is not None and
            array_equal(effective_data.X, self.effective_data.X))
        self._domain_invalidated = not (
            data_existed and self.data is not None and
            effective_data.domain.checksum()
            == self.effective_data.domain.checksum())
        if self._invalidated:
            self.clear()
            self.input_changed.emit(data)
        self.enable_controls()

    def check_data(self):
        self.clear_messages()

    def enable_controls(self):
        self.cb_class_density.setEnabled(self.can_draw_density())

    @Inputs.data_subset
    @check_sql_input
    def set_subset_data(self, subset):
        self.subset_data = subset

    def handleNewSignals(self):
        self._handle_subset_data()
        if self._invalidated:
            self._invalidated = False
            self.setup_plot()
        else:
            self.graph.update_point_props()
        self._update_opacity_warning()
        self.commit.now()

    def _handle_subset_data(self):
        self.Warning.subset_independent.clear()
        self.Warning.subset_not_subset.clear()
        if self.data is None or self.subset_data is None:
            self.subset_indices = set()
        else:
            self.subset_indices = set(self.subset_data.ids)
            ids = set(self.data.ids)
            if not self.subset_indices & ids:
                self.Warning.subset_independent()
            elif self.subset_indices - ids:
                self.Warning.subset_not_subset()

    def _update_opacity_warning(self):
        self.Warning.transparent_subset(
            shown=self.subset_indices and self.graph.alpha_value < 128)

    def get_subset_mask(self):
        if not self.subset_indices:
            return None
        valid_data = self.data[self.valid_data]
        return np.fromiter((ex.id in self.subset_indices for ex in valid_data),
                           dtype=bool, count=len(valid_data))

    # Plot
    def get_embedding(self):
        """A get embedding method.

        Derived classes must override this method. The overridden method
        should return embedding for all data (valid and invalid). Invalid
        data embedding coordinates should be set to 0 (in some cases to Nan).

        The method should also set self.valid_data.

        Returns:
            np.array: Array of embedding coordinates with shape
            len(self.data) x 2
        """
        raise NotImplementedError

    def get_coordinates_data(self):
        embedding = self.get_embedding()
        if embedding is not None and len(embedding[self.valid_data]):
            return embedding[self.valid_data].T
        return None, None

    def setup_plot(self):
        self.graph.reset_graph()
        self.__pending_selection = self.selection or self.__pending_selection
        self.apply_selection()

    # Selection
    def apply_selection(self):
        pending = self.__pending_selection
        if self.data is not None and pending is not None and len(pending) \
                and max(i for i, _ in pending) < self.graph.n_valid:
            index_group = np.array(pending).T
            selection = np.zeros(self.graph.n_valid, dtype=np.uint8)
            selection[index_group[0]] = index_group[1]

            self.selection = self.__pending_selection
            self.__pending_selection = None
            self.graph.selection = selection
            self.graph.update_selection_colors()
            if self.graph.label_only_selected:
                self.graph.update_labels()

    def selection_changed(self):
        sel = None if self.data and isinstance(self.data, SqlTable) \
            else self.graph.selection
        self.selection = [(i, x) for i, x in enumerate(sel) if x] \
            if sel is not None else None
        self.commit.deferred()

    # Output
    @gui.deferred
    def commit(self):
        self.send_data()

    def send_data(self):
        group_sel, data, graph = None, self._get_projection_data(), self.graph
        if graph.selection is not None:
            group_sel = np.zeros(len(data), dtype=int)
            group_sel[self.valid_data] = graph.selection
        selected = self._get_selected_data(
            data, graph.get_selection(), group_sel)
        self.output_changed.emit(selected)
        self.Outputs.selected_data.send(selected)
        self.Outputs.annotated_data.send(
            self._get_annotated_data(data, group_sel,
                                     graph.selection))

    def _get_projection_data(self):
        if self.data is None or self.embedding_variables_names is None:
            return self.data
        variables = self._get_projection_variables()
        data = self.data.transform(Domain(self.data.domain.attributes,
                                          self.data.domain.class_vars,
                                          self.data.domain.metas + variables))
        if data.metas.size:
            with data.unlocked(data.metas):
                data.metas[:, -2:] = self.get_embedding()
        return data

    def _get_projection_variables(self):
        names = get_unique_names(
            self.data.domain, self.embedding_variables_names)
        return ContinuousVariable(names[0]), ContinuousVariable(names[1])

    @staticmethod
    def _get_selected_data(data, selection, group_sel):
        return create_groups_table(data, group_sel, False, "Group") \
            if len(selection) else None

    @staticmethod
    def _get_annotated_data(data, group_sel, graph_sel):
        if data is None:
            return None
        if graph_sel is not None and np.max(graph_sel) > 1:
            return create_groups_table(data, group_sel)
        else:
            if group_sel is None:
                mask = np.full((len(data), ), False)
            else:
                mask = np.nonzero(group_sel)[0]
            return create_annotated_table(data, mask)

    # Report
    def send_report(self):
        if self.data is None:
            return

        caption = self._get_send_report_caption()
        self.report_plot()
        if caption:
            self.report_caption(caption)

    def _get_send_report_caption(self):
        return report.render_items_vert((
            ("Color", self._get_caption_var_name(self.attr_color)),
            ("Label", self._get_caption_var_name(self.attr_label)),
            ("Shape", self._get_caption_var_name(self.attr_shape)),
            ("Size", self._get_caption_var_name(self.attr_size)),
            ("Jittering", self.graph.jitter_size != 0 and
             "{} %".format(self.graph.jitter_size))))

    # Customize plot
    def set_visual_settings(self, key, value):
        self.graph.parameter_setter.set_parameter(key, value)
        self.visual_settings[key] = value

    @staticmethod
    def _get_caption_var_name(var):
        return var.name if isinstance(var, Variable) else var

    # Misc
    def sizeHint(self):
        return QSize(1132, 708)

    def clear(self):
        self.selection = None
        self.graph.selection = None

    def onDeleteWidget(self):
        super().onDeleteWidget()
        self.graph.plot_widget.getViewBox().deleteLater()
        self.graph.plot_widget.clear()
        self.graph.clear()


class OWAnchorProjectionWidget(OWDataProjectionWidget, openclass=True):
    """ Base widget for widgets with graphs with anchors. """
    SAMPLE_SIZE = 100

    GRAPH_CLASS = OWGraphWithAnchors
    graph = SettingProvider(OWGraphWithAnchors)

    class Outputs(OWDataProjectionWidget.Outputs):
        components = Output("Components", Table)

    class Error(OWDataProjectionWidget.Error):
        sparse_data = Msg("Sparse data is not supported")
        no_valid_data = Msg("No projection due to no valid data")
        no_instances = Msg("At least two data instances are required")
        proj_error = Msg("An error occurred while projecting data.\n{}")

    def __init__(self):
        self.projector = self.projection = None
        super().__init__()
        self.graph.view_box.started.connect(self._manual_move_start)
        self.graph.view_box.moved.connect(self._manual_move)
        self.graph.view_box.finished.connect(self._manual_move_finish)

    def check_data(self):
        def error(err):
            err()
            self.data = None

        super().check_data()
        if self.data is not None:
            if self.data.is_sparse():
                error(self.Error.sparse_data)
            elif len(self.data) < 2:
                error(self.Error.no_instances)
            else:
                if not np.sum(np.all(np.isfinite(self.data.X), axis=1)):
                    error(self.Error.no_valid_data)

    def init_projection(self):
        self.projection = None
        if not self.effective_variables:
            return
        try:
            self.projection = self.projector(self.effective_data)
        except Exception as ex:  # pylint: disable=broad-except
            self.Error.proj_error(ex)

    def get_embedding(self):
        self.valid_data = None
        if self.data is None or self.projection is None:
            return None
        embedding = self.projection(self.data).X
        self.valid_data = np.all(np.isfinite(embedding), axis=1)
        return embedding

    def get_anchors(self):
        if self.projection is None:
            return None, None
        components = self.projection.components_
        if components.shape == (1, 1):
            components = np.array([[1.], [0.]])
        return components.T, [a.name for a in self.effective_variables]

    def _manual_move_start(self):
        self.graph.set_sample_size(self.SAMPLE_SIZE)

    def _manual_move(self, anchor_idx, x, y):
        self.projection.components_[:, anchor_idx] = [x, y]
        self.graph.update_coordinates()

    def _manual_move_finish(self, anchor_idx, x, y):
        self._manual_move(anchor_idx, x, y)
        self.graph.set_sample_size(None)
        self.commit.deferred()

    def _get_projection_data(self):
        if self.data is None or self.projection is None:
            return None
        proposed = [a.name for a in self.projection.domain.attributes]
        names = get_unique_names(self.data.domain, proposed)

        if proposed != names:
            attributes = tuple([attr.copy(name=name) for name, attr in
                                zip(names, self.projection.domain.attributes)])
        else:
            attributes = self.projection.domain.attributes
        return self.data.transform(
            Domain(self.data.domain.attributes,
                   self.data.domain.class_vars,
                   self.data.domain.metas + attributes))

    @gui.deferred
    def commit(self):
        super().commit()
        self.send_components()

    def send_components(self):
        components = None
        if self.data is not None and self.projection is not None:
            proposed = [var.name for var in self.effective_variables]
            comp_name = get_unique_names(proposed, 'component')
            meta_attrs = [StringVariable(name=comp_name)]
            domain = Domain(self.effective_variables, metas=meta_attrs)
            components = Table(domain, self._send_components_x().copy(),
                               metas=self._send_components_metas())
            components.name = "components"
        self.Outputs.components.send(components)

    def _send_components_x(self):
        return self.projection.components_

    def _send_components_metas(self):
        variable_names = [a.name for a in self.projection.domain.attributes]
        return np.array(variable_names, dtype=object)[:, None]

    def clear(self):
        super().clear()
        self.projector = self.projection = None


if __name__ == "__main__":
    class OWProjectionWidgetWithName(OWDataProjectionWidget):
        name = "projection"

        def get_embedding(self):
            if self.data is None:
                return None
            self.valid_data = np.any(np.isfinite(self.data.X), 1)
            x_data = self.data.X
            x_data[x_data == np.inf] = np.nan
            x_data = np.nanmean(x_data[self.valid_data], 1)
            y_data = np.ones(len(x_data))
            return np.vstack((x_data, y_data)).T

    app = QApplication([])
    ow = OWProjectionWidgetWithName()
    table = Table("iris")
    ow.set_data(table)
    ow.set_subset_data(table[::10])
    ow.handleNewSignals()
    ow.show()
    app.exec()
    ow.saveSettings()
