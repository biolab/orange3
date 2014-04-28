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
from Orange.widgets import gui

from .owconstants import *

from PyQt4.QtGui import QWidget, QToolButton, QGroupBox, QVBoxLayout, QHBoxLayout, QIcon, QMenu, QAction
from PyQt4.QtCore import Qt, pyqtSignal, QObject, SIGNAL, SLOT


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
    def __init__(self, gui, text, orientation, buttons, parent, nomargin = False):
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
        #NOTHING = 0
        #ZOOMING = 1
        #SELECT = 2
        #SELECT_POLYGON = 3
        #PANNING = 4
        #SELECT_RECTANGLE = SELECT
        #SELECT_RIGHTCLICK = SELECT
        state_buttons = {0: 11, 1: 11, 2: 13, 3: 13, 4: 12}
        self.buttons[state_buttons[state]].click()

    def select_selection_behaviour(self, selection_behaviour):
        #SelectionAdd = 21
        #SelectionRemove = 22
        #SelectionToggle = 23
        #SelectionOne = 24
        self.buttons[13]._actions[21 + selection_behaviour].trigger()

class StateButtonContainer(OrientedWidget):
    '''
        This class can contain any number of checkable buttons, of which only one can be selected at any time.

        :param gui: Used to create containers and buttons
        :type gui: :obj:`.OWPlotGUI`

        :param buttons: A list of button identifiers to be added to this toolbar
        :type buttons: list of (int or tuple)

        :param orientation: The orientation of this toolbar, either Qt.Vertical or Qt.Horizontal
        :type tex: int

        :param parent: The toolbar's parent widget
        :type parent: :obj:`.QWidget`
    '''
    def __init__(self, gui, orientation, buttons, parent, nomargin = False):
        OrientedWidget.__init__(self, orientation, parent)
        self.buttons = {}
        if nomargin:
            self.layout().setContentsMargins(0, 0, 0, 0)
        self._clicked_button = None
        for i in buttons:
            b = gui.tool_button(i, self)
            QObject.connect(b, SIGNAL("triggered(QAction*)"), self.button_clicked)
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
    def __init__(self, plot, icon_name=None, attr_name='', attr_value=None, callback=None, parent=None):
        QAction.__init__(self, parent)

        if type(callback) == str:
            callback = getattr(plot, callback, None)
        if callback:
            QObject.connect(self, SIGNAL("triggered(bool)"), callback)
        if attr_name:
            self._plot = plot
            self.attr_name = attr_name
            self.attr_value = attr_value
            QObject.connect(self, SIGNAL("triggered(bool)"), self.set_attribute)
        if icon_name:
            self.setIcon(QIcon(os.path.dirname(__file__) + "/../../icons/" + icon_name + '.png'))
            self.setIconVisibleInMenu(True)

    def set_attribute(self, clicked):
        setattr(self._plot, self.attr_name, self.attr_value)


class OWButton(QToolButton):
    '''
        A custom tool button which signal when its down state changes
    '''
    def __init__(self, action=None, parent=None):
        QToolButton.__init__(self, parent)
        self.setMinimumSize(30, 30)
        if action:
            self.setDefaultAction(action)

    def setDown(self, down):
        if self.isDown() != down:
            self.emit(SIGNAL("downChanged(bool)"), down)
        QToolButton.setDown(self, down)

class OWPlotGUI:
    '''
        This class contains functions to create common user interface elements (QWidgets)
        for configuration and interaction with the ``plot``.

        It provides shorter versions of some methods in :obj:`.gui` that are directly related to an :obj:`.OWPlot` object.

        Normally, you don't have to construct this class manually. Instead, first create the plot,
        then use the :attr:`.OWPlot.gui` attribute.

        Most methods in this class have similar arguments, so they are explaned here in a single place.

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

        When using widgets that are specific to your visualization and not included here, you have to provide your
        own widgets id's. They are a tuple with the following members:

        :param id: An optional unique identifier for the widget.
                   This is only needed if you want to retrive this widget using :obj:`.OWToolbar.buttons`.
        :type id: int or str

        :param text: The text to be displayed on or next to the widget
        :type text: str

        :param attr_name: Name of attribute which will be set when the button is clicked.
                          If this widget is checkable, its check state will be set
                          according to the current value of this attribute.
                          If this parameter is empty or None, no attribute will be read or set.
        :type attr_name: str

        :param attr_value: The value that will be assigned to the ``attr_name`` when the button is clicked.
        :type attr: any

        :param callback: Function to be called when the button is clicked.
                         If a string is passed as ``callback``, a method by that name of ``plot`` will be called.
                         If this parameter is empty or ``None``, no function will be called
        :type callback: str or function

        :param icon_name: The filename of the icon for this widget, without the '.png' suffix.
        :type icon_name: str

    '''
    def __init__(self, plot):
        self._plot = plot

    Spacing = 0

    ShowLegend = 2
    ShowFilledSymbols = 3
    ShowGridLines = 4
    PointSize = 5
    AlphaValue = 6

    Zoom = 11
    Pan = 12
    Select = 13

    ZoomSelection = 15

    SelectionAdd = 21
    SelectionRemove = 22
    SelectionToggle = 23
    SelectionOne = 24

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
        Zoom : ('Zoom', 'state', ZOOMING, None, 'Dlg_zoom'),
        Pan : ('Pan', 'state', PANNING, None, 'Dlg_pan_hand'),
        Select : ('Select', 'state', SELECT, None, 'Dlg_arrow'),
        SelectionAdd : ('Add to selection', 'selection_behavior', SELECTION_ADD, None, 'Dlg_select_add'),
        SelectionRemove : ('Remove from selection', 'selection_behavior', SELECTION_REMOVE, None, 'Dlg_select_remove'),
        SelectionToggle : ('Toggle selection', 'selection_behavior', SELECTION_TOGGLE, None, 'Dlg_select_toggle'),
        SelectionOne : ('Replace selection', 'selection_behavior', SELECTION_REPLACE, None, 'Dlg_arrow'),
        SendSelection : ('Send selection', None, None, 'send_selection', 'Dlg_send'),
        ClearSelection : ('Clear selection', None, None, 'clear_selection', 'Dlg_clear'),
        ShufflePoints : ('ShufflePoints', None, None, 'shuffle_points', 'Dlg_sort')
    }

    _check_boxes = {
        AnimatePlot : ('Animate plot', 'animate_plot', 'update_animations'),
        AnimatePoints : ('Animate points', 'animate_points', 'update_animations'),
        AntialiasPlot : ('Antialias plot', 'antialias_plot', 'update_antialiasing'),
        AntialiasPoints : ('Antialias points', 'antialias_points', 'update_antialiasing'),
        AntialiasLines : ('Antialias lines', 'antialias_lines', 'update_antialiasing'),
        AutoAdjustPerformance : ('Disable effects for large data sets', 'auto_adjust_performance', 'update_performance')
    }
    '''
        The list of built-in buttons. It is a map of
        id : (name, attr_name, attr_value, callback, icon_name)

        .. seealso:: :meth:`.tool_button`
    '''

    def _get_callback(self, name):
        if type(name) == str:
            return getattr(self._plot, name, self._plot.replot)
        else:
            return name

    def _check_box(self, widget, value, label, cb_name):
        '''
            Adds a :obj:`.QCheckBox` to ``widget``.
            When the checkbox is toggled, the attribute ``value`` of the plot object is set to the checkbox' check state,
            and the callback ``cb_name`` is called.
        '''
        gui.checkBox(widget, self._plot, value, label, callback=self._get_callback(cb_name))

    def antialiasing_check_box(self, widget):
        '''
            Creates a check box that toggles the Antialiasing of the plot
        '''
        self._check_box(widget, 'use_antialiasing', 'Use antialiasing', 'update_antialiasing')

    def show_legend_check_box(self, widget):
        '''
            Creates a check box that shows and hides the plot legend
        '''
        self._check_box(widget, 'show_legend', 'Show legend', 'update_legend')

    def filled_symbols_check_box(self, widget):
        self._check_box(widget, 'show_filled_symbols', 'Show filled symbols', 'update_filled_symbols')

    def grid_lines_check_box(self, widget):
        self._check_box(widget, 'show_grid', 'Show gridlines', 'update_grid')

    def animations_check_box(self, widget):
        '''
            Creates a check box that enabled or disables animations
        '''
        self._check_box(widget, 'use_animations', 'Use animations', 'update_animations')

    def _slider(self, widget, value, label, min_value, max_value, step, cb_name):
        gui.hSlider(widget, self._plot, value, label=label, minValue=min_value, maxValue=max_value, step=step, callback=self._get_callback(cb_name))

    def point_size_slider(self, widget):
        '''
            Creates a slider that controls point size
        '''
        self._slider(widget, 'point_width', "Symbol size:   ", 1, 20, 1, 'update_point_size')

    def alpha_value_slider(self, widget):
        '''
            Creates a slider that controls point transparency
        '''
        self._slider(widget, 'alpha_value', "Transparency: ", 0, 255, 10, 'update_alpha_value')

    def point_properties_box(self, widget):
        '''
            Creates a box with controls for common point properties.
            Currently, these properties are point size and transparency.
        '''
        return self.create_box([
            self.PointSize,
            self.AlphaValue
            ], widget, "Point properties")

    def plot_settings_box(self, widget):
        '''
            Creates a box with controls for common plot settings
        '''
        return self.create_box([
            self.ShowLegend,
            self.ShowFilledSymbols,
            self.ShowGridLines,
            ], widget, "Plot settings")

    _functions = {
        ShowLegend : show_legend_check_box,
        ShowFilledSymbols : filled_symbols_check_box,
        ShowGridLines : grid_lines_check_box,
        PointSize : point_size_slider,
        AlphaValue : alpha_value_slider,
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

    def create_box(self, ids, widget, name):
        '''
            Creates a :obj:`.QGroupBox` with text ``name`` and adds it to ``widget``.
            The ``ids`` argument is a list of widget ID's that will be added to this box
        '''
        box = gui.widgetBox(widget, name)
        self.add_widgets(ids, box)
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
            b = self.menu_button(self.Select, [self.SelectionOne, self.SelectionAdd, self.SelectionRemove, self.SelectionToggle], widget)
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
        id, name, attr_name, attr_value, callback, icon_name = self._expand_id(main_action_id)
        b = OWButton(parent=widget)
        m = QMenu(b)
        b.setMenu(m)
        b._actions = {}

        QObject.connect(m, SIGNAL("triggered(QAction*)"), b, SLOT("setDefaultAction(QAction*)"))

        if main_action_id:
            main_action = OWAction(self._plot, icon_name, attr_name, attr_value, callback, parent=b)
            QObject.connect(m, SIGNAL("triggered(QAction*)"), main_action, SLOT("trigger()"))

        for id in ids:
            id, name, attr_name, attr_value, callback, icon_name = self._expand_id(id)
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

    def state_buttons(self, orientation, buttons, widget, nomargin = False):
        '''
            This function creates a set of checkable buttons and connects them so that only one
            may be checked at a time.
        '''
        c = StateButtonContainer(self, orientation, buttons, widget, nomargin)
        if widget.layout() is not None:
            widget.layout().addWidget(c)
        return c

    def toolbar(self, widget, text, orientation, buttons, nomargin = False):
        '''
            Creates an :obj:`.OWToolbar` with the specified ``text``, ``orientation`` and ``buttons`` and adds it to ``widget``.

            .. seealso:: :obj:`.OWToolbar`
        '''
        t = OWToolbar(self, text, orientation, buttons, widget, nomargin)
        if nomargin:
            t.layout().setContentsMargins(0, 0, 0, 0)
        if widget.layout() is not None:
            widget.layout().addWidget(t)
        return t

    def zoom_select_toolbar(self, widget, text = 'Zoom / Select', orientation = Qt.Horizontal, buttons = default_zoom_select_buttons, nomargin = False):
        t = self.toolbar(widget, text, orientation, buttons, nomargin)
        t.buttons[self.Select].click()
        return t

    def effects_box(self, widget):
        b = self.create_box([
            self.AnimatePlot,
            self.AnimatePoints,
            self.AntialiasPlot,
        #    self.AntialiasPoints,
        #    self.AntialiasLines,
            self.AutoAdjustPerformance,
            self.DisableAnimationsThreshold], widget, "Visual effects")
        return b

    def theme_combo_box(self, widget):
        c = gui.comboBox(widget, self._plot, "theme_name", "Theme", callback = self._plot.update_theme, sendSelectedValue = 1, valueType = str)
        c.addItem('Default')
        c.addItem('Light')
        c.addItem('Dark')
        return c
