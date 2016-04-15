import os

from PyQt4.QtGui import QListWidget, QIcon, QSizePolicy

from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting
from Orange.widgets.widget import OWWidget
from Orange.widgets.utils import vartype

ICON_UP = gui.resource_filename("icons/Dlg_up3.png")
ICON_DOWN = gui.resource_filename("icons/Dlg_down3.png")


class OWVisWidget(OWWidget):
    _shown_attributes = ContextSetting(default=[], required=ContextSetting.REQUIRED,
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
        attr_info = lambda a: (a.name, vartype(a))
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

        self.trigger_attributes_changed()

    @property
    def hidden_attributes(self):
        return [a[0] for a in self._hidden_attributes]

    __attribute_selection_area_initialized = False

    #noinspection PyAttributeOutsideInit
    def add_attribute_selection_area(self, parent):
        self.add_shown_attributes(parent)
        self.add_hidden_attributes(parent)
        self.__attribute_selection_area_initialized = True

        self.trigger_attributes_changed()

    #noinspection PyAttributeOutsideInit
    def add_shown_attributes(self, parent):
        self.shown_attributes_area = gui.vBox(parent, " Shown attributes ")
        self.shown_attributes_listbox = gui.listBox(
            self.shown_attributes_area, self, "selected_shown", "_shown_attributes",
            dragDropCallback=self.trigger_attributes_changed,
            enableDragDrop=True, selectionMode=QListWidget.ExtendedSelection)

    #noinspection PyAttributeOutsideInit
    def add_hidden_attributes(self, parent):
        self.hidden_attributes_area = gui.vBox(parent, " Hidden attributes ")
        self.hidden_attributes_listbox = gui.listBox(
            self.hidden_attributes_area, self, "selected_hidden",
            "_hidden_attributes",
            dragDropCallback=self.trigger_attributes_changed,
            enableDragDrop=True, selectionMode=QListWidget.ExtendedSelection)

    def get_data_domain(self):
        if hasattr(self, "data") and self.data:
            return self.data.domain
        else:
            return None

    def trigger_attributes_changed(self):
        if not self.__attribute_selection_area_initialized:
            # Some components trigger this event during the initialization.
            # We ignore those requests, a separate event will be triggered
            # manually when everything is initialized.
            return

        self.attributes_changed()

    def closeContext(self):
        super().closeContext()

        self.data = None
        self.shown_attributes = None

    # "Events"
    def attributes_changed(self):
        pass
