import os

from PyQt4.QtGui import QListWidget, QIcon, QSizePolicy

from Orange.canvas.utils import environ
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting
from Orange.widgets.widget import OWWidget
from Orange.widgets.utils import vartype

ICON_UP = os.path.join(environ.widget_install_dir, "icons/Dlg_up3.png")
ICON_DOWN = os.path.join(environ.widget_install_dir, "icons/Dlg_down3.png")


class OWVisWidget(OWWidget):
    _shown_features = ContextSetting(default=[], required=ContextSetting.REQUIRED,
                                       selected='selected_shown', reservoir="_hidden_features")
    # Setting above will override these fields
    _hidden_features = ()
    selected_shown = ()
    selected_hidden = ()

    @property
    def shown_features(self):
        return [a[0] for a in self._shown_features]

    @shown_features.setter
    def shown_features(self, value):
        shown = []
        hidden = []

        domain = self.get_data_domain()
        attr_info = lambda a: (a.name, vartype(a))
        if domain:
            if value:
                shown = value if isinstance(value[0], tuple) else [attr_info(domain[a]) for a in value]
                hidden = [x for x in [attr_info(domain[a]) for a in domain.features] if x not in shown]
            else:
                shown = [attr_info(a) for a in domain.features]
                if not self.show_all_features:
                    hidden = shown[10:]
                    shown = shown[:10]

            if domain.class_var and attr_info(domain.class_var) not in shown:
                hidden += [attr_info(domain.class_var)]

        self._shown_features = shown
        self._hidden_features = hidden
        self.selected_hidden = []
        self.selected_shown = []

        self.trigger_features_changed()

    @property
    def hidden_features(self):
        return [a[0] for a in self._hidden_features]

    __attribute_selection_area_initialized = False

    #noinspection PyAttributeOutsideInit
    def add_attribute_selection_area(self, parent):
        self.add_shown_features(parent)
        self.add_hidden_features(parent)
        self.__attribute_selection_area_initialized = True

        self.trigger_features_changed()

    #noinspection PyAttributeOutsideInit
    def add_shown_features(self, parent):
        self.shown_features_area = gui.widgetBox(parent, " Shown features ")
        self.shown_features_listbox = gui.listBox(
            self.shown_features_area, self, "selected_shown", "_shown_features",
            dragDropCallback=self.trigger_features_changed,
            enableDragDrop=True, selectionMode=QListWidget.ExtendedSelection)

    #noinspection PyAttributeOutsideInit
    def add_hidden_features(self, parent):
        self.hidden_features_area = gui.widgetBox(parent, " Hidden features ")
        self.hidden_features_listbox = gui.listBox(self.hidden_features_area, self, "selected_hidden",
                                                     "_hidden_features",
                                                     dragDropCallback=self.trigger_features_changed,
                                                     enableDragDrop=True, selectionMode=QListWidget.ExtendedSelection)

    def get_data_domain(self):
        if hasattr(self, "data") and self.data:
            return self.data.domain
        else:
            return None

    def trigger_features_changed(self):
        if not self.__attribute_selection_area_initialized:
            # Some components trigger this event during the initialization.
            # We ignore those requests, a separate event will be triggered
            # manually when everything is initialized.
            return

        self.features_changed()

    def closeContext(self):
        super().closeContext()

        self.data = None
        self.shown_features = None

    # "Events"
    def features_changed(self):
        pass
