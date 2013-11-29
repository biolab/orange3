import os

from PyQt4.QtGui import QListWidget, QIcon, QSizePolicy

from Orange.canvas.utils import environ
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting
from Orange.widgets.widget import OWWidget

ICON_UP = os.path.join(environ.widget_install_dir, "icons/Dlg_up3.png")
ICON_DOWN = os.path.join(environ.widget_install_dir, "icons/Dlg_down3.png")


class OWVisWidget(OWWidget):
    shown_attributes = ContextSetting([], required=ContextSetting.REQUIRED,
                                          selected='selected_shown', reservoir="hidden_attributes")
    # Setting above will override these fields
    selected_shown = ()
    hidden_attributes = ()
    selected_hidden = ()

    #noinspection PyAttributeOutsideInit
    def add_attribute_selection_area(self, parent, callback=None):
        self.selected_shown = []
        self.shown_attributes = []
        self.hidden_attributes = []
        self.selected_hidden = []
        self.on_update_callback = callback

        self.add_shown_attributes(parent)
        self.add_control_buttons(parent)
        self.add_hidden_attributes(parent)

    #noinspection PyAttributeOutsideInit
    def add_shown_attributes(self, parent):
        self.shown_attributes_area = gui.widgetBox(parent, " Shown attributes ")
        box = gui.widgetBox(self.shown_attributes_area, orientation='horizontal')
        self.shown_attributes_listbox = gui.listBox(
            box, self, "selected_shown", "shown_attributes",
            callback=self.reset_attr_manipulation, dragDropCallback=self.attributes_changed,
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
        self.add_attribute_button = gui.button(self.add_remove_tools_area, self, "", callback=self.add_attribute,
                                               tooltip="Add (show) selected attributes")
        self.add_attribute_button.setIcon(QIcon(ICON_UP))
        self.remove_attribute_button = gui.button(self.add_remove_tools_area, self, "",
                                                  callback=self.remove_attribute,
                                                  tooltip="Remove (hide) selected attributes")
        self.remove_attribute_button.setIcon(QIcon(ICON_DOWN))
        self.show_all_attributes_checkbox = gui.checkBox(self.add_remove_tools_area, self, "show_all_attributes",
                                                         "Show all", callback=self.toggle_show_all_attributes)

    #noinspection PyAttributeOutsideInit
    def add_hidden_attributes(self, parent):
        self.hidden_attributes_area = gui.widgetBox(parent, " Hidden attributes ")
        self.hidden_attributes_listbox = gui.listBox(self.hidden_attributes_area, self, "selected_hidden",
                                                     "hidden_attributes", callback=self.reset_attr_manipulation,
                                                     dragDropCallback=self.attributes_changed, enableDragDrop=True,
                                                     selectionMode=QListWidget.ExtendedSelection)

    def reset_attr_manipulation(self):
        if self.selected_shown:
            mini, maxi = min(self.selected_shown), max(self.selected_shown)
            tight_selection = maxi - mini == len(self.selected_shown) - 1
            valid_selection = mini > 0 and maxi < len(self.shown_attributes)
        else:
            tight_selection = valid_selection = False

        self.move_attribute_up_button.setEnabled(bool(self.selected_shown and tight_selection and valid_selection))
        self.move_attribute_down_button.setEnabled(bool(self.selected_shown and tight_selection and valid_selection))
        self.add_attribute_button.setDisabled(not self.selected_hidden or self.show_all_attributes)
        self.remove_attribute_button.setDisabled(not self.selected_shown or self.show_all_attributes)
        domain = self.get_data_domain()
        if domain and self.hidden_attributes and domain.class_var \
                and self.hidden_attributes[0][0] != domain.class_var.name:
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
        if hasattr(self, "graph"):
            self.graph.insideColors = None
            self.graph.clusterClosure = None
            self.graph.potentialsBmp = None

        attrs = self.shown_attributes
        sel = self.selected_shown
        mini, maxi = min(sel), max(sel) + 1
        if dir == -1:
            self.shown_attributes = attrs[:mini - 1] + attrs[mini:maxi] + [attrs[mini - 1]] + attrs[maxi:]
        else:
            self.shown_attributes = attrs[:mini] + [attrs[maxi]] + attrs[mini:maxi] + attrs[maxi + 1:]
        self.selected_shown = [x + dir for x in sel]

        self.reset_attr_manipulation()

        self.attributes_changed()

        self.graph.potentialsBmp = None
        if self.on_update_callback:
            self.on_update_callback()
        if hasattr(self, "graph"):
            self.graph.removeAllSelections()

    def toggle_show_all_attributes(self):
        if self.show_all_attributes:
            self.add_attribute(True)
        self.reset_attr_manipulation()

    def add_attribute(self, addAll=False):
        if hasattr(self, "graph"):
            self.graph.insideColors = None
            self.graph.clusterClosure = None

        if addAll:
            self.setShownAttributeList()
        else:
            self.setShownAttributeList(
                self.shown_attributes + [self.hidden_attributes[i] for i in self.selected_hidden])
        self.selected_hidden = []
        self.selected_shown = []
        self.reset_attr_manipulation()

        self.attributes_changed()

        if hasattr(self, "graph"):
            self.graph.removeAllSelections()

    def remove_attribute(self):
        if hasattr(self, "graph"):
            self.graph.insideColors = None
            self.graph.clusterClosure = None

        new_shown = self.shown_attributes[:]
        self.selected_shown.sort(reverse=True)
        for i in self.selected_shown:
            del new_shown[i]
        self.setShownAttributeList(new_shown)

        self.attributes_changed()

        if self.on_update_callback:
            self.on_update_callback()
        if hasattr(self, "graph"):
            self.graph.removeAllSelections()

    def getShownAttributeList(self):
        return [a[0] for a in self.shown_attributes]


    def setShownAttributeList(self, shownAttributes=None):
        shown = []
        hidden = []

        domain = self.get_data_domain()
        if domain:
            if shownAttributes:
                if type(shownAttributes[0]) == tuple:
                    shown = shownAttributes
                else:
                    shown = [(domain[a].name, domain[a].var_type) for a in shownAttributes]
                hidden = [x for x in [(a.name, a.var_type) for a in domain.attributes] if x not in shown]
            else:
                shown = [(a.name, a.var_type) for a in domain.attributes]
                if not self.show_all_attributes:
                    hidden = shown[10:]
                    shown = shown[:10]

            if domain.class_var and (domain.class_var.name, domain.class_var.var_type) not in shown:
                hidden += [(domain.class_var.name, domain.class_var.var_type)]

        self.shown_attributes = shown
        self.hidden_attributes = hidden
        self.selected_hidden = []
        self.selected_shown = []
        self.reset_attr_manipulation()

        self.send_shown_attributes()

    # "Events"
    def attributes_changed(self):
        pass
