'''

.. index:: plot

######################################
GUI elements for plots (``owplotgui``)
######################################

This module contains functions and classes for creating GUI elements commonly used for plots.

.. autoclass:: OrientedWidget
    :show-inheritance:

.. autoclass:: StateButtonContainer
    :show-inheritance:

.. autoclass:: OWToolbar
    :show-inheritance:

.. autoclass:: OWButton
    :show-inheritance:

.. autoclass:: OrangeWidgets.plot.OWPlotGUI
    :members:

'''

import os
import unicodedata

from AnyQt.QtWidgets import QWidget, QToolButton, QVBoxLayout, QHBoxLayout, QGridLayout, QMenu, QAction,\
    QDialog, QSizePolicy, QPushButton, QListView, QLabel
from AnyQt.QtGui import QIcon, QKeySequence
from AnyQt.QtCore import Qt, pyqtSignal, QPoint, QSize, QObject

from Orange.data import ContinuousVariable, DiscreteVariable
from Orange.widgets import gui
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.listfilter import variables_filter
from Orange.widgets.utils.itemmodels import DomainModel

from .owconstants import NOTHING, ZOOMING, SELECT, SELECT_POLYGON, PANNING, SELECTION_ADD,\
    SELECTION_REMOVE, SELECTION_TOGGLE, SELECTION_REPLACE

__all__ = ["AddVariablesDialog", "VariablesSelection",
           "OrientedWidget", "OWToolbar", "StateButtonContainer",
           "OWAction", "OWButton", "OWPlotGUI"]


SIZE_POLICY_ADAPTING = (QSizePolicy.Expanding, QSizePolicy.Ignored)
SIZE_POLICY_FIXED = (QSizePolicy.Minimum, QSizePolicy.Maximum)


class AddVariablesDialog(QDialog):
    add = pyqtSignal()

    def __init__(self, master, model):
        QDialog.__init__(self)

        self.master = master

        self.setWindowFlags(Qt.Tool)
        self.setLayout(QVBoxLayout())
        self.setWindowTitle("Hidden Axes")

        btns_area = gui.widgetBox(
            self, addSpace=0, spacing=9, orientation=Qt.Horizontal,
            sizePolicy=QSizePolicy(*SIZE_POLICY_FIXED)
        )
        self.btn_add = QPushButton(
            "Add", autoDefault=False, sizePolicy=QSizePolicy(*SIZE_POLICY_FIXED)
        )
        self.btn_add.clicked.connect(self._add)
        self.btn_cancel = QPushButton(
            "Cancel", autoDefault=False, sizePolicy=QSizePolicy(*SIZE_POLICY_FIXED)
        )
        self.btn_cancel.clicked.connect(self._cancel)

        btns_area.layout().addWidget(self.btn_add)
        btns_area.layout().addWidget(self.btn_cancel)

        filter_edit, view = variables_filter(model=model)
        self.view_other = view
        view.setMinimumSize(QSize(30, 60))
        view.setSizePolicy(*SIZE_POLICY_ADAPTING)
        view.viewport().setAcceptDrops(True)

        self.layout().addWidget(filter_edit)
        self.layout().addWidget(view)
        self.layout().addWidget(btns_area)

        master = self.master
        box = master.box
        master.master.setEnabled(False)
        self.move(box.mapToGlobal(QPoint(0, box.pos().y() + box.height())))
        self.setFixedWidth(master.master.controlArea.width())
        self.setMinimumHeight(300)
        self.show()
        self.raise_()
        self.activateWindow()

    def _cancel(self):
        self.closeEvent(None)

    def _add(self):
        self.add_variables()
        self.closeEvent(None)

    def closeEvent(self, QCloseEvent):
        self.master.master.setEnabled(True)
        super().closeEvent(QCloseEvent)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.closeEvent(None)
        elif e.key() in [Qt.Key_Return, Qt.Key_Enter]:
            self._add()
        else:
            super().keyPressEvent(e)

    def selected_rows(self, view):
        """ Return the selected rows in the view.
        """
        rows = view.selectionModel().selectedRows()
        model = view.model()
        return [model.mapToSource(r) for r in rows]

    def add_variables(self):
        view = self.view_other
        model = self.master.model_other

        indices = self.selected_rows(view)
        variables = [model.data(ind, Qt.EditRole) for ind in indices]

        for i in sorted((ind.row() for ind in indices), reverse=True):
            del model[i]

        self.master.model_selected.extend(variables)
        self.add.emit()


class VariablesSelection(QObject):
    added = pyqtSignal()
    removed = pyqtSignal()

    def __init__(self, master, model_selected, model_other,
                 widget=None, parent=None):
        super().__init__(parent)
        self.master = master
        self.model_selected = model_selected
        self.model_other = model_other

        params_view = {"sizePolicy": QSizePolicy(*SIZE_POLICY_ADAPTING),
                       "selectionMode": QListView.ExtendedSelection,
                       "dragEnabled": True,
                       "defaultDropAction": Qt.MoveAction,
                       "dragDropOverwriteMode": False,
                       "dragDropMode": QListView.DragDrop}

        self.view_selected = view = gui.listView(
            widget or master.controlArea, master,
            box=True, **params_view
        )
        view.box.setMinimumHeight(120)
        view.viewport().setAcceptDrops(True)

        delete = QAction(
            "Delete", view,
            shortcut=QKeySequence(Qt.Key_Delete),
            triggered=self.__deactivate_selection
        )
        view.addAction(delete)
        view.setModel(self.model_selected)

        addClassLabel = QAction("+", master,
                                toolTip="Add new class label",
                                triggered=self._action_add)
        removeClassLabel = QAction(unicodedata.lookup("MINUS SIGN"), master,
                                   toolTip="Remove selected class label",
                                   triggered=self.__deactivate_selection)

        add_remove = itemmodels.ModelActionsWidget(
            [addClassLabel, removeClassLabel], master)
        add_remove.layout().addStretch(10)
        add_remove.layout().setSpacing(1)
        add_remove.setSizePolicy(*SIZE_POLICY_FIXED)
        view.box.layout().addWidget(add_remove)

        self.add_remove = add_remove
        self.box = add_remove.buttons[1]

    def set_enabled(self, is_enabled):
        self.view_selected.setEnabled(is_enabled)
        for btn in self.add_remove.buttons:
            btn.setEnabled(is_enabled)

    def __deactivate_selection(self):
        view = self.view_selected
        model = self.model_selected
        indices = view.selectionModel().selectedRows()

        variables = [model.data(ind, Qt.EditRole) for ind in indices]

        for i in sorted((ind.row() for ind in indices), reverse=True):
            del model[i]

        self.model_other.extend(variables)
        self.removed.emit()

    def _action_add(self):
        self.add_variables_dialog = AddVariablesDialog(self, self.model_other)
        self.add_variables_dialog.add.connect(lambda: self.added.emit())


class OrientedWidget(QWidget):
    '''
        A simple QWidget with a box layout that matches its ``orientation``.
    '''
    def __init__(self, orientation, parent):
        QWidget.__init__(self, parent)
        if orientation == Qt.Vertical:
            self._layout = QVBoxLayout()
        else:
            self._layout = QHBoxLayout()
        self.setLayout(self._layout)

class OWToolbar(OrientedWidget):
    '''
        A toolbar is a container that can contain any number of buttons.

        :param gui: Used to create containers and buttons
        :type gui: :obj:`.OWPlotGUI`

        :param text: The name of this toolbar
        :type text: str

        :param orientation: The orientation of this toolbar, either Qt.Vertical or Qt.Horizontal
        :type tex: int

        :param buttons: A list of button identifiers to be added to this toolbar
        :type buttons: list of (int or tuple)

        :param parent: The toolbar's parent widget
        :type parent: :obj:`.QWidget`
    '''
    def __init__(self, gui, text, orientation, buttons, parent, nomargin=False):
        OrientedWidget.__init__(self, orientation, parent)
        self.buttons = {}
        self.groups = {}
        i = 0
        n = len(buttons)
        while i < n:
            if buttons[i] == gui.StateButtonsBegin:
                state_buttons = []
                for j in range(i+1, n):
                    if buttons[j] == gui.StateButtonsEnd:
                        s = gui.state_buttons(orientation, state_buttons, self, nomargin)
                        self.buttons.update(s.buttons)
                        self.groups[buttons[i+1]] = s
                        i = j
                        break
                    else:
                        state_buttons.append(buttons[j])
            elif buttons[i] == gui.Spacing:
                self.layout().addSpacing(10)
            elif type(buttons[i] == int):
                self.buttons[buttons[i]] = gui.tool_button(buttons[i], self)
            elif len(buttons[i] == 4):
                gui.tool_button(buttons[i], self)
            else:
                self.buttons[buttons[i][0]] = gui.tool_button(buttons[i], self)
            i = i + 1
        self.layout().addStretch()

    def select_state(self, state):
        # SELECT_RECTANGLE = SELECT
        # SELECT_RIGHTCLICK = SELECT
        state_buttons = {NOTHING: 11, ZOOMING: 11, SELECT: 13, SELECT_POLYGON: 13, PANNING: 12}
        self.buttons[state_buttons[state]].click()

    def select_selection_behaviour(self, selection_behaviour):
        # SelectionAdd = 21
        # SelectionRemove = 22
        # SelectionToggle = 23
        # SelectionOne = 24
        self.buttons[13]._actions[21 + selection_behaviour].trigger()


class StateButtonContainer(OrientedWidget):
    '''
        This class can contain any number of checkable buttons, of which only one can be selected
        at any time.

        :param gui: Used to create containers and buttons
        :type gui: :obj:`.OWPlotGUI`

        :param buttons: A list of button identifiers to be added to this toolbar
        :type buttons: list of (int or tuple)

        :param orientation: The orientation of this toolbar, either Qt.Vertical or Qt.Horizontal
        :type tex: int

        :param parent: The toolbar's parent widget
        :type parent: :obj:`.QWidget`
    '''
    def __init__(self, gui, orientation, buttons, parent, nomargin=False):
        OrientedWidget.__init__(self, orientation, parent)
        self.buttons = {}
        if nomargin:
            self.layout().setContentsMargins(0, 0, 0, 0)
        self._clicked_button = None
        for i in buttons:
            b = gui.tool_button(i, self)
            b.triggered.connect(self.button_clicked)
            self.buttons[i] = b
            self.layout().addWidget(b)

    def button_clicked(self, checked):
        sender = self.sender()
        self._clicked_button = sender
        for button in self.buttons.values():
            button.setDown(button is sender)

    def button(self, id):
        return self.buttons[id]

    def setEnabled(self, enabled):
        OrientedWidget.setEnabled(self, enabled)
        if enabled and self._clicked_button:
            self._clicked_button.click()

class OWAction(QAction):
    '''
      A :obj:`QAction` with convenience methods for calling a callback or
      setting an attribute of the plot.
    '''
    def __init__(self, plot, icon_name=None, attr_name='', attr_value=None, callback=None,
                 parent=None):
        QAction.__init__(self, parent)

        if type(callback) == str:
            callback = getattr(plot, callback, None)
        if callback:
            self.triggered.connect(callback)
        if attr_name:
            self._plot = plot
            self.attr_name = attr_name
            self.attr_value = attr_value
            self.triggered.connect(self.set_attribute)
        if icon_name:
            self.setIcon(
                QIcon(os.path.join(os.path.dirname(__file__),
                                   "../../icons", icon_name + '.png')))
            self.setIconVisibleInMenu(True)

    def set_attribute(self, clicked):
        setattr(self._plot, self.attr_name, self.attr_value)


class OWButton(QToolButton):
    '''
        A custom tool button which signal when its down state changes
    '''
    downChanged = pyqtSignal(bool)

    def __init__(self, action=None, parent=None):
        QToolButton.__init__(self, parent)
        self.setMinimumSize(30, 30)
        if action:
            self.setDefaultAction(action)

    def setDown(self, down):
        if self.isDown() != down:
            self.downChanged[bool].emit(down)
        QToolButton.setDown(self, down)


class OWPlotGUI:
    '''
        This class contains functions to create common user interface elements (QWidgets)
        for configuration and interaction with the ``plot``.

        It provides shorter versions of some methods in :obj:`.gui` that are directly related to an
        :obj:`.OWPlot` object.

        Normally, you don't have to construct this class manually. Instead, first create the plot,
        then use the :attr:`.OWPlot.gui` attribute.

        Most methods in this class have similar arguments, so they are explaned here in a single
        place.

        :param widget: The parent widget which will contain the newly created widget.
        :type widget: QWidget

        :param id: If ``id`` is an ``int``, a button is constructed from the default table.
                   Otherwise, ``id`` must be tuple with 5 or 6 elements. These elements
                   are explained in the next table.
        :type id: int or tuple

        :param ids: A list of widget identifiers
        :type ids: list of id

        :param text: The text displayed on the widget
        :type text: str

        When using widgets that are specific to your visualization and not included here, you have
        to provide your
        own widgets id's. They are a tuple with the following members:

        :param id: An optional unique identifier for the widget.
                   This is only needed if you want to retrive this widget using
                   :obj:`.OWToolbar.buttons`.
        :type id: int or str

        :param text: The text to be displayed on or next to the widget
        :type text: str

        :param attr_name: Name of attribute which will be set when the button is clicked.
                          If this widget is checkable, its check state will be set
                          according to the current value of this attribute.
                          If this parameter is empty or None, no attribute will be read or set.
        :type attr_name: str

        :param attr_value: The value that will be assigned to the ``attr_name`` when the button is
        clicked.
        :type attr: any

        :param callback: Function to be called when the button is clicked.
                         If a string is passed as ``callback``, a method by that name of ``plot``
                         will be called.
                         If this parameter is empty or ``None``, no function will be called
        :type callback: str or function

        :param icon_name: The filename of the icon for this widget, without the '.png' suffix.
        :type icon_name: str

    '''

    JITTER_SIZES = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10]

    def __init__(self, master):
        self._master = master
        self._plot = master.graph
        self.color_model = DomainModel(
            placeholder="(Same color)", valid_types=DomainModel.PRIMITIVE)
        self.shape_model = DomainModel(
            placeholder="(Same shape)", valid_types=DiscreteVariable)
        self.size_model = DomainModel(
            placeholder="(Same size)", valid_types=ContinuousVariable)
        self.label_model = DomainModel(placeholder="(No labels)")
        self.points_models = [self.color_model, self.shape_model,
                              self.size_model, self.label_model]

    Spacing = 0

    ShowLegend = 2
    ShowFilledSymbols = 3
    ShowGridLines = 4
    PointSize = 5
    AlphaValue = 6
    Color = 7
    Shape = 8
    Size = 9
    Label = 10

    Zoom = 11
    Pan = 12
    Select = 13

    ZoomSelection = 15
    ZoomReset = 16

    ToolTipShowsAll = 17
    ClassDensity = 18
    RegressionLine = 19
    LabelOnlySelected = 20

    SelectionAdd = 21
    SelectionRemove = 22
    SelectionToggle = 23
    SelectionOne = 24
    SimpleSelect = 25

    SendSelection = 31
    ClearSelection = 32
    ShufflePoints = 33

    StateButtonsBegin = 35
    StateButtonsEnd = 36

    AnimatePlot = 41
    AnimatePoints = 42
    AntialiasPlot = 43
    AntialiasPoints = 44
    AntialiasLines = 45
    DisableAnimationsThreshold = 48
    AutoAdjustPerformance = 49

    JitterSizeSlider = 51
    JitterNumericValues = 52

    UserButton = 100

    default_zoom_select_buttons = [
        StateButtonsBegin,
        Zoom,
        Pan,
        Select,
        StateButtonsEnd,
        Spacing,
        SendSelection,
        ClearSelection
    ]

    _buttons = {
        Zoom: ('Zoom', 'state', ZOOMING, None, 'Dlg_zoom'),
        ZoomReset: ('Reset zoom', None, None, None, 'Dlg_zoom_reset'),
        Pan: ('Pan', 'state', PANNING, None, 'Dlg_pan_hand'),
        SimpleSelect: ('Select', 'state', SELECT, None, 'Dlg_arrow'),
        Select: ('Select', 'state', SELECT, None, 'Dlg_arrow'),
        SelectionAdd: ('Add to selection', 'selection_behavior', SELECTION_ADD, None,
                       'Dlg_select_add'),
        SelectionRemove: ('Remove from selection', 'selection_behavior', SELECTION_REMOVE, None,
                          'Dlg_select_remove'),
        SelectionToggle: ('Toggle selection', 'selection_behavior', SELECTION_TOGGLE, None,
                          'Dlg_select_toggle'),
        SelectionOne: ('Replace selection', 'selection_behavior', SELECTION_REPLACE, None,
                       'Dlg_arrow'),
        SendSelection: ('Send selection', None, None, 'send_selection', 'Dlg_send'),
        ClearSelection: ('Clear selection', None, None, 'clear_selection', 'Dlg_clear'),
        ShufflePoints: ('ShufflePoints', None, None, 'shuffle_points', 'Dlg_sort')
    }

    _check_boxes = {
        AnimatePlot : ('Animate plot', 'animate_plot', 'update_animations'),
        AnimatePoints : ('Animate points', 'animate_points', 'update_animations'),
        AntialiasPlot : ('Antialias plot', 'antialias_plot', 'update_antialiasing'),
        AntialiasPoints : ('Antialias points', 'antialias_points', 'update_antialiasing'),
        AntialiasLines : ('Antialias lines', 'antialias_lines', 'update_antialiasing'),
        AutoAdjustPerformance : ('Disable effects for large datasets', 'auto_adjust_performance',
                                 'update_performance')
    }

    '''
        The list of built-in buttons. It is a map of
        id : (name, attr_name, attr_value, callback, icon_name)

        .. seealso:: :meth:`.tool_button`
    '''

    def _get_callback(self, name, master=None):
        if type(name) == str:
            return getattr(master or self._plot, name)
        else:
            return name

    def _check_box(self, widget, value, label, cb_name):
        '''
            Adds a :obj:`.QCheckBox` to ``widget``.
            When the checkbox is toggled, the attribute ``value`` of the plot object is set to
            the checkbox' check state, and the callback ``cb_name`` is called.
        '''
        args = dict(master=self._plot, value=value, label=label,
                    callback=self._get_callback(cb_name, self._plot))
        if isinstance(widget.layout(), QGridLayout):
            widget = widget.layout()
        if isinstance(widget, QGridLayout):
            checkbox = gui.checkBox(None, **args)
            widget.addWidget(checkbox, widget.rowCount(), 1)
            return checkbox
        else:
            return gui.checkBox(widget, **args)

    def antialiasing_check_box(self, widget):
        '''
            Creates a check box that toggles the Antialiasing of the plot
        '''
        self._check_box(widget, 'use_antialiasing', 'Use antialiasing', 'update_antialiasing')

    def jitter_size_slider(self, widget):
        return self.add_control(
            widget, gui.valueSlider, "Jittering",
            master=self._plot, value='jitter_size',
            values=getattr(self._plot, "jitter_sizes", self.JITTER_SIZES),
            callback=self._plot.update_jittering)

    def jitter_numeric_check_box(self, widget):
        self._check_box(
            widget=widget,
            value="jitter_continuous", label="Jitter numeric values",
            cb_name="update_jittering")

    def show_legend_check_box(self, widget):
        '''
            Creates a check box that shows and hides the plot legend
        '''
        self._check_box(widget, 'show_legend', 'Show legend',
                        'update_legend_visibility')

    def tooltip_shows_all_check_box(self, widget):
        gui.checkBox(
            widget=widget, master=self._master, value="tooltip_shows_all",
            label='Show all data on mouse hover')

    def class_density_check_box(self, widget):
        self._master.cb_class_density = \
            self._check_box(widget=widget, value="class_density",
                            label="Show color regions",
                            cb_name=self._plot.update_density)

    def regression_line_check_box(self, widget):
        self._master.cb_reg_line = \
            self._check_box(widget=widget, value="show_reg_line",
                            label="Show regression line",
                            cb_name=self._plot.update_regression_line)

    def label_only_selected_check_box(self, widget):
        self._check_box(widget=widget, value="label_only_selected",
                        label="Label only selected points",
                        cb_name=self._plot.update_labels)

    def filled_symbols_check_box(self, widget):
        self._check_box(widget, 'show_filled_symbols', 'Show filled symbols',
                        'update_filled_symbols')

    def grid_lines_check_box(self, widget):
        self._check_box(widget, 'show_grid', 'Show gridlines',
                        'update_grid_visibility')

    def animations_check_box(self, widget):
        '''
            Creates a check box that enabled or disables animations
        '''
        self._check_box(widget, 'use_animations', 'Use animations', 'update_animations')

    def add_control(self, widget, control, label, **args):
        if isinstance(widget.layout(), QGridLayout):
            widget = widget.layout()
        if isinstance(widget, QGridLayout):
            row = widget.rowCount()
            element = control(None, **args)
            widget.addWidget(QLabel(label), row, 0)
            widget.addWidget(element, row, 1)
            return element
        else:
            return control(widget,  label=label, **args)

    def _slider(self, widget, value, label, min_value, max_value, step, cb_name,
                show_number=False):
        return self.add_control(
            widget, gui.hSlider, label,
            master=self._plot, value=value, minValue=min_value,
            maxValue=max_value, step=step, createLabel=show_number,
            callback=self._get_callback(cb_name, self._master))

    def point_size_slider(self, widget, label="Symbol size:   "):
        '''
            Creates a slider that controls point size
        '''
        return self._slider(widget, 'point_width', label, 1, 20, 1, 'sizes_changed')

    def alpha_value_slider(self, widget, label="Opacity: "):
        '''
            Creates a slider that controls point transparency
        '''
        return self._slider(widget, 'alpha_value', label, 0, 255, 10, 'colors_changed')

    def _combo(self, widget, value, label, cb_name, items=(), model=None):
        return self.add_control(
            widget, gui.comboBox, label,
            master=self._master, value=value, items=items, model=model,
            callback=self._get_callback(cb_name, self._master),
            orientation=Qt.Horizontal, valueType=str,
            sendSelectedValue=True, contentsLength=12,
            labelWidth=50)

    def color_value_combo(self, widget, label="Color: "):
        """Creates a combo box that controls point color"""
        self._combo(widget, "attr_color", label, "colors_changed",
                    model=self.color_model)

    def shape_value_combo(self, widget, label="Shape: "):
        """Creates a combo box that controls point shape"""
        self._combo(widget, "attr_shape", label, "shapes_changed",
                    model=self.shape_model)

    def size_value_combo(self, widget, label="Size: "):
        """Creates a combo box that controls point size"""
        self._combo(widget, "attr_size", label, "sizes_changed",
                    model=self.size_model)

    def label_value_combo(self, widget, label="Label: "):
        """Creates a combo box that controls point label"""
        self._combo(widget, "attr_label", label, "labels_changed",
                    model=self.label_model)

    def box_spacing(self, widget):
        if isinstance(widget.layout(), QGridLayout):
            widget = widget.layout()
        if isinstance(widget, QGridLayout):
            space = QWidget()
            space.setFixedSize(12, 12)
            widget.addWidget(space, widget.rowCount(), 0)
        else:
            gui.separator(widget)

    def point_properties_box(self, widget, box=True):
        '''
            Creates a box with controls for common point properties.
            Currently, these properties are point size and transparency.
        '''
        box = self.create_gridbox(widget, box)
        self.add_widgets([
            self.Color,
            self.Shape,
            self.Size,
            self.Label,
            self.LabelOnlySelected], box)
        return box

    def effects_box(self, widget, box=True):
        """
        Create a box with controls for common plot settings
        """
        box = self.create_gridbox(widget, box)
        self.add_widgets([
            self.PointSize,
            self.AlphaValue,
            self.JitterSizeSlider], box)
        return box

    def plot_properties_box(self, widget, box=None):
        """
        Create a box with controls for common plot settings
        """
        return self.create_box([
            self.ClassDensity,
            self.ShowLegend], widget, box, True)

    _functions = {
        ShowFilledSymbols: filled_symbols_check_box,
        JitterSizeSlider: jitter_size_slider,
        JitterNumericValues: jitter_numeric_check_box,
        ShowLegend: show_legend_check_box,
        ShowGridLines: grid_lines_check_box,
        ToolTipShowsAll: tooltip_shows_all_check_box,
        ClassDensity: class_density_check_box,
        RegressionLine: regression_line_check_box,
        LabelOnlySelected: label_only_selected_check_box,
        PointSize: point_size_slider,
        AlphaValue: alpha_value_slider,
        Color: color_value_combo,
        Shape: shape_value_combo,
        Size: size_value_combo,
        Label: label_value_combo,
        Spacing: box_spacing
        }

    def add_widget(self, id, widget):
        if id in self._functions:
            self._functions[id](self, widget)
        elif id in self._check_boxes:
            label, attr, cb = self._check_boxes[id]
            self._check_box(widget, attr, label, cb)

    def add_widgets(self, ids, widget):
        for id in ids:
            self.add_widget(id, widget)

    def create_box(self, ids, widget, box, name):
        '''
            Creates a :obj:`.QGroupBox` with text ``name`` and adds it to ``widget``.
            The ``ids`` argument is a list of widget ID's that will be added to this box
        '''
        if box is None:
            box = gui.vBox(widget, name)
        self.add_widgets(ids, box)
        return box

    def create_gridbox(self, widget, box=True):
        grid = QGridLayout()
        grid.setColumnMinimumWidth(0, 50)
        grid.setColumnStretch(1, 1)
        box = gui.widgetBox(widget, box=box, orientation=grid)
        # This must come after calling widgetBox, since widgetBox overrides it
        grid.setVerticalSpacing(8)
        return box

    def _expand_id(self, id):
        if type(id) == int:
            name, attr_name, attr_value, callback, icon_name = self._buttons[id]
        elif len(id) == 4:
            name, attr_name, attr_value, callback, icon_name = id
            id = -1
        else:
            id, name, attr_name, attr_value, callback, icon_name = id
        return id, name, attr_name, attr_value, callback, icon_name

    def tool_button(self, id, widget):
        '''
            Creates an :obj:`.OWButton` and adds it to the parent ``widget``.
        '''
        id, name, attr_name, attr_value, callback, icon_name = self._expand_id(id)
        if id == OWPlotGUI.Select:
            b = self.menu_button(self.Select,
                                 [self.SelectionOne, self.SelectionAdd,
                                  self.SelectionRemove, self.SelectionToggle],
                                 widget)
        else:
            b = OWButton(parent=widget)
            ac = OWAction(self._plot, icon_name, attr_name, attr_value, callback, parent=b)
            b.setDefaultAction(ac)
        b.setToolTip(name)
        if widget.layout() is not None:
            widget.layout().addWidget(b)
        return b

    def menu_button(self, main_action_id, ids, widget):
        '''
            Creates an :obj:`.OWButton` with a popup-menu and adds it to the parent ``widget``.
        '''
        id, _, attr_name, attr_value, callback, icon_name = self._expand_id(main_action_id)
        b = OWButton(parent=widget)
        m = QMenu(b)
        b.setMenu(m)
        b._actions = {}

        m.triggered[QAction].connect(b.setDefaultAction)

        if main_action_id:
            main_action = OWAction(self._plot, icon_name, attr_name, attr_value, callback,
                                   parent=b)
            m.triggered.connect(main_action.trigger)

        for id in ids:
            id, _, attr_name, attr_value, callback, icon_name = self._expand_id(id)
            a = OWAction(self._plot, icon_name, attr_name, attr_value, callback, parent=m)
            m.addAction(a)
            b._actions[id] = a

        if m.actions():
            b.setDefaultAction(m.actions()[0])
        elif main_action_id:
            b.setDefaultAction(main_action)


        b.setPopupMode(QToolButton.MenuButtonPopup)
        b.setMinimumSize(40, 30)
        return b

    def state_buttons(self, orientation, buttons, widget, nomargin=False):
        '''
            This function creates a set of checkable buttons and connects them so that only one
            may be checked at a time.
        '''
        c = StateButtonContainer(self, orientation, buttons, widget, nomargin)
        if widget.layout() is not None:
            widget.layout().addWidget(c)
        return c

    def toolbar(self, widget, text, orientation, buttons, nomargin=False):
        '''
            Creates an :obj:`.OWToolbar` with the specified ``text``, ``orientation``
            and ``buttons`` and adds it to ``widget``.

            .. seealso:: :obj:`.OWToolbar`
        '''
        t = OWToolbar(self, text, orientation, buttons, widget, nomargin)
        if nomargin:
            t.layout().setContentsMargins(0, 0, 0, 0)
        if widget.layout() is not None:
            widget.layout().addWidget(t)
        return t

    def zoom_select_toolbar(self, widget, text='Zoom / Select', orientation=Qt.Horizontal,
                            buttons=default_zoom_select_buttons, nomargin=False):
        t = self.toolbar(widget, text, orientation, buttons, nomargin)
        t.buttons[self.SimpleSelect].click()
        return t

    def theme_combo_box(self, widget):
        c = gui.comboBox(widget, self._plot, "theme_name", "Theme",
                         callback=self._plot.update_theme, sendSelectedValue=1, valueType=str)
        c.addItem('Default')
        c.addItem('Light')
        c.addItem('Dark')
        return c

    def box_zoom_select(self, parent):
        box_zoom_select = gui.vBox(parent, "Zoom/Select")
        zoom_select_toolbar = self.zoom_select_toolbar(
            box_zoom_select, nomargin=True,
            buttons=[self.StateButtonsBegin,
                     self.SimpleSelect, self.Pan, self.Zoom,
                     self.StateButtonsEnd,
                     self.ZoomReset]
        )
        buttons = zoom_select_toolbar.buttons
        buttons[self.Zoom].clicked.connect(self._plot.zoom_button_clicked)
        buttons[self.Pan].clicked.connect(self._plot.pan_button_clicked)
        buttons[self.SimpleSelect].clicked.connect(self._plot.select_button_clicked)
        buttons[self.ZoomReset].clicked.connect(self._plot.reset_button_clicked)
        return box_zoom_select
