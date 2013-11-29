import os

from PyQt4.QtGui import QListWidget, QIcon, QSizePolicy

from Orange.canvas.utils import environ
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting
from Orange.widgets.widget import OWWidget

ICON_UP = os.path.join(environ.widget_install_dir, "icons/Dlg_up3.png")
ICON_DOWN = os.path.join(environ.widget_install_dir, "icons/Dlg_down3.png")


class OWVisWidget(OWWidget):
    _shown_attributes = ContextSetting([], required=ContextSetting.REQUIRED,
                                           selected='selected_shown', reservoir="_hidden_attributes")
    # Setting above will override these fields
    _hidden_attributes = ()
    selected_shown = ()
    selected_hidden = ()

    @property
    def shown_attributes(self):
        return [a[0] for a in self._shown_attributes]

    @shown_attributes.setter
    def shown_attributes(self, value):
        shown = []
        hidden = []

        domain = self.get_data_domain()
        attr_info = lambda a: (a.name, a.var_type)
        if domain:
            if value:
                shown = value if isinstance(value[0], tuple) else [attr_info(domain[a]) for a in value]
                hidden = [x for x in [attr_info(domain[a]) for a in domain.attributes] if x not in shown]
            else:
                shown = [attr_info(a) for a in domain.attributes]
                if not self.show_all_attributes:
                    hidden = shown[10:]
                    shown = shown[:10]

            if domain.class_var and attr_info(domain.class_var) not in shown:
                hidden += [attr_info(domain.class_var)]

        self._shown_attributes = shown
        self._hidden_attributes = hidden
        self.selected_hidden = []
        self.selected_shown = []
        self.reset_attr_manipulation()

        self.trigger_attributes_changed()

    @property
    def hidden_attributes(self):
        return [a[0] for a in self._hidden_attributes]

    __attribute_selection_area_initialized = False

    #noinspection PyAttributeOutsideInit
    def add_attribute_selection_area(self, parent):
        self.add_shown_attributes(parent)
        self.add_control_buttons(parent)
        self.add_hidden_attributes(parent)
        self.__attribute_selection_area_initialized = True

        self.trigger_attributes_changed()

    #noinspection PyAttributeOutsideInit
    def add_shown_attributes(self, parent):
        self.shown_attributes_area = gui.widgetBox(parent, " Shown attributes ")
        box = gui.widgetBox(self.shown_attributes_area, orientation='horizontal')
        self.shown_attributes_listbox = gui.listBox(
            box, self, "selected_shown", "_shown_attributes",
            callback=self.reset_attr_manipulation, dragDropCallback=self.trigger_attributes_changed,
            enableDragDrop=True, selectionMode=QListWidget.ExtendedSelection)
        controls_box = gui.widgetBox(box, orientation='vertical')
        self.move_attribute_up_button = gui.button(controls_box, self, "", callback=self.move_selection_up,
                                                   tooltip="Move selected attributes up")
        self.move_attribute_up_button.setIcon(QIcon(ICON_UP))
        self.move_attribute_up_button.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding))
        self.move_attribute_up_button.setMaximumWidth(30)

        self.move_attribute_down_button = gui.button(controls_box, self, "", callback=self.move_selection_down,
                                                     tooltip="Move selected attributes down")
        self.move_attribute_down_button.setIcon(QIcon(ICON_DOWN))
        self.move_attribute_down_button.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding))
        self.move_attribute_down_button.setMaximumWidth(30)

    #noinspection PyAttributeOutsideInit
    def add_control_buttons(self, parent):
        self.add_remove_tools_area = gui.widgetBox(parent, 1, orientation="horizontal")
        self.add_attribute_button = gui.button(self.add_remove_tools_area, self, "", callback=self.show_attribute,
                                               tooltip="Add (show) selected attributes")
        self.add_attribute_button.setIcon(QIcon(ICON_UP))
        self.remove_attribute_button = gui.button(self.add_remove_tools_area, self, "",
                                                  callback=self.hide_attribute,
                                                  tooltip="Remove (hide) selected attributes")
        self.remove_attribute_button.setIcon(QIcon(ICON_DOWN))
        self.show_all_attributes_checkbox = gui.checkBox(self.add_remove_tools_area, self, "show_all_attributes",
                                                         "Show all", callback=self.toggle_show_all_attributes)

    #noinspection PyAttributeOutsideInit
    def add_hidden_attributes(self, parent):
        self.hidden_attributes_area = gui.widgetBox(parent, " Hidden attributes ")
        self.hidden_attributes_listbox = gui.listBox(self.hidden_attributes_area, self, "selected_hidden",
                                                     "_hidden_attributes", callback=self.reset_attr_manipulation,
                                                     dragDropCallback=self.trigger_attributes_changed,
                                                     enableDragDrop=True, selectionMode=QListWidget.ExtendedSelection)

    def reset_attr_manipulation(self):
        if not self.__attribute_selection_area_initialized:
            return
        if self.selected_shown:
            mini, maxi = min(self.selected_shown), max(self.selected_shown)
            tight_selection = maxi - mini == len(self.selected_shown) - 1
            valid_selection = mini > 0 and maxi < len(self._shown_attributes)
        else:
            tight_selection = valid_selection = False

        self.move_attribute_up_button.setEnabled(bool(self.selected_shown and tight_selection and valid_selection))
        self.move_attribute_down_button.setEnabled(bool(self.selected_shown and tight_selection and valid_selection))
        self.add_attribute_button.setDisabled(not self.selected_hidden or self.show_all_attributes)
        self.remove_attribute_button.setDisabled(not self.selected_shown or self.show_all_attributes)
        domain = self.get_data_domain()
        if domain and self._hidden_attributes and domain.class_var \
                and self._hidden_attributes[0][0] != domain.class_var.name:
            self.show_all_attributes_checkbox.setChecked(False)

    def get_data_domain(self):
        if hasattr(self, "data") and self.data:
            return self.data.domain
        else:
            return None

    def move_selection_up(self):
        self.move_selected_attributes(-1)

    def move_selection_down(self):
        self.move_selected_attributes(1)

    def move_selected_attributes(self, dir):
        attrs = self._shown_attributes
        mini, maxi = min(self.selected_shown), max(self.selected_shown) + 1
        if dir == -1:
            self._shown_attributes = attrs[:mini - 1] + attrs[mini:maxi] + [attrs[mini - 1]] + attrs[maxi:]
        else:
            self._shown_attributes = attrs[:mini] + [attrs[maxi]] + attrs[mini:maxi] + attrs[maxi + 1:]
        self.selected_shown = [x + dir for x in self.selected_shown]

        self.reset_attr_manipulation()

        self.trigger_attributes_changed()

    def toggle_show_all_attributes(self):
        if self.show_all_attributes:
            self.show_attribute(True)
        self.reset_attr_manipulation()

    def show_attribute(self, add_all=False):
        if add_all:
            self.set_shown_attributes()
        else:
            self.set_shown_attributes(
                self._shown_attributes + [self._hidden_attributes[i] for i in self.selected_hidden])
        self.selected_hidden = []
        self.selected_shown = []
        self.reset_attr_manipulation()

        self.trigger_attributes_changed()

    def hide_attribute(self):
        new_shown = self._shown_attributes[:]
        self.selected_shown.sort(reverse=True)
        for i in self.selected_shown:
            del new_shown[i]
        self.set_shown_attributes(new_shown)

        self.trigger_attributes_changed()

    def trigger_attributes_changed(self):
        if not self.__attribute_selection_area_initialized:
            # Some components trigger this event during the initialization.
            # We ignore those requests, a separate event will be triggered
            # manually when everything is initialized.
            return
        self.attributes_changed()

    def closeContext(self):
        super().closeContext()

        self.shown_attributes = None

    # "Events"
    def attributes_changed(self):
        pass
