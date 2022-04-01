from itertools import chain
from typing import Union, Optional, Dict

import numpy as np
from scipy import sparse as sp

from AnyQt.QtWidgets import QFormLayout

from orangewidget.report import bool_str
from orangewidget.utils.widgetpreview import WidgetPreview
from orangewidget.widget import Msg

from Orange.data import \
    Table, Domain, DiscreteVariable, StringVariable, ContinuousVariable
from Orange.data.util import get_unique_names
from Orange.widgets import widget, gui
from Orange.widgets.settings import ContextHandler, Setting, ContextSetting
from Orange.widgets.utils import itemmodels


DEFAULT_ITEM_NAME = "item"
DEFAULT_VALUE_NAME = "value"
DEFAULT_NAME_FOR_ROW = "row"


class MeltContextHandler(ContextHandler):
    # ContextHandler's methods *are supposed to* context-related add arguments
    # pylint: disable=arguments-differ
    def new_context(self, potential_ids):
        context = super().new_context()
        context.potential_ids = {var.name for var in potential_ids}
        return context

    def match(self, context, potential_ids):
        # This handler matches idvar = None only on perfect match.
        # Otherwise, None would match any context, so automated selection in
        # case of a single candidate would never work.
        names = {var.name for var in potential_ids}
        if names == context.potential_ids:
            return self.PERFECT_MATCH
        if context.values["idvar"] in names:
            return self.MATCH
        return self.NO_MATCH

    def encode_setting(self, context, setting, value):
        if setting.name == "idvar":
            return value.name if value is not None else None
        return super().encode_setting(context, setting, value)

    def decode_setting(self, setting, value, potential_ids):
        if setting.name == "idvar":
            if value is None:
                return None
            for var in potential_ids:
                if var.name == value:
                    return var
        return super().decode_setting(setting, value, potential_ids)


class OWMelt(widget.OWWidget):
    name = "Melt"
    description = "Convert wide data to narrow data, a list of item-value pairs"
    category = "Transform"
    icon = "icons/Melt.svg"
    keywords = ["shopping list", "wide", "narrow"]
    priority = 2230

    class Inputs:
        data = widget.Input("Data", Table)

    class Outputs:
        data = widget.Output("Data", Table)

    class Information(widget.OWWidget.Information):
        no_suitable_features = Msg(
            "No columns with unique values\n"
            "Only columns with unique valules are useful for row identifiers.")

    want_main_area = False
    resizing_enabled = False

    settingsHandler = MeltContextHandler()
    idvar: Union[DiscreteVariable, StringVariable, None] = ContextSetting(None)
    only_numeric = Setting(True)
    exclude_zeros = Setting(False)
    item_var_name = Setting("")
    value_var_name = Setting("")
    auto_apply = Setting(True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data: Optional[Table] = None
        self._output_desc: Optional[Dict[str, str]] = None

        box = gui.widgetBox(self.controlArea, "Unique Row Identifier")
        self.idvar_model = itemmodels.VariableListModel(
            [None], placeholder="Row number")
        self.var_cb = gui.comboBox(
            box, self, "idvar", model=self.idvar_model,
            callback=self._invalidate, minimumContentsLength=16,
            tooltip="A column with identifier, like customer's id")

        box = gui.widgetBox(self.controlArea, "Filter")
        gui.checkBox(
            box, self, "only_numeric", "Ignore non-numeric features",
            callback=self._invalidate)
        gui.checkBox(
            box, self, "exclude_zeros", "Exclude zero values",
            callback=self._invalidate,
            tooltip="Besides missing values, also omit items with zero values")

        form = QFormLayout()
        gui.widgetBox(
            self.controlArea, "Names for generated features", orientation=form)
        form.addRow("Item:",
                    gui.lineEdit(
                        None, self, "item_var_name",
                        callback=self._invalidate,
                        placeholderText=DEFAULT_ITEM_NAME,
                        styleSheet="padding-left: 3px"))
        form.addRow("Value:",
                    gui.lineEdit(
                        None, self, "value_var_name",
                        callback=self._invalidate,
                        placeholderText=DEFAULT_VALUE_NAME,
                        styleSheet="padding-left: 3px"))

        gui.auto_apply(self.controlArea, self)

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.Information.clear()
        self.idvar = None
        del self.idvar_model[1:]
        self.data = data
        if data is not None:
            self.idvar_model[1:] = (
                var
                for var in chain(data.domain.variables, data.domain.metas)
                if isinstance(var, (DiscreteVariable, StringVariable))
                and self._is_unique(var))
            if len(self.idvar_model) == 1:
                self.Information.no_suitable_features()
            # If there is a single suitable variable, we guess it's the id
            # If there are multiple, we default to row number and let the user
            # choose
            elif len(self.idvar_model) == 2:
                self.idvar = self.idvar_model[1]
            self.openContext(self.idvar_model[1:])

        self.commit.now()

    def _is_unique(self, var):
        col = self.data.get_column_view(var)[0]
        col = col[self._notnan_mask(col)]
        return len(col) == len(set(col))

    @staticmethod
    def _notnan_mask(col):
        return np.isfinite(col) if col.dtype == float else col != ""

    def _invalidate(self):
        self.commit.deferred()

    @gui.deferred
    def commit(self):
        self.Error.clear()
        if self.data:
            output = self._reshape_to_long()
            self.Outputs.data.send(output)
            self._store_output_desc(output)
        else:
            self.Outputs.data.send(None)
            self._output_desc = None

    def send_report(self):
        self.report_items("Settings", (
            ("Row identifier", self.controls.idvar.currentText()),
            ("Ignore non-numeric features", bool_str(self.only_numeric)),
            ("Exclude zero values", bool_str(self.exclude_zeros))
        ))
        if self._output_desc:
            self.report_items("Output", self._output_desc)

    def _store_output_desc(self, output):
        self._output_desc = {
            "Item column": output.domain.attributes[1].name,
            "Value column": output.domain.class_var.name,
            "Number of items": len(output)
        }

    def _reshape_to_long(self):
        # Get a mask with columns used for data
        useful_vars = self._get_useful_vars()
        item_names = self._get_item_names(useful_vars)
        n_useful = len(item_names)

        # Get identifiers, remove rows with missing id data
        id_names = ()
        if self.idvar:
            idvalues, _ = self.data.get_column_view(self.idvar)
            idmask = self._notnan_mask(idvalues)
            x = self.data.X[idmask]
            idvalues = idvalues[idmask]
            # For string ids, use indices and store names
            if self.idvar.is_string:
                id_names = idvalues
                idvalues = np.arange(len(idvalues))
        else:
            x = self.data.X
            idvalues = np.arange(x.shape[0])

        # Prepare columns of the long list
        if sp.issparse(x):
            xcoo = x.tocoo()
            col_selection = useful_vars[xcoo.col]
            idcol = idvalues[xcoo.row[col_selection]]
            items = xcoo.col[col_selection]
            items = (np.cumsum(useful_vars) - 1)[items]  # renumerate
            values = xcoo.data[col_selection]
        else:
            idcol = np.repeat(idvalues, n_useful)
            items = np.tile(np.arange(n_useful), len(x))
            values = x[:, useful_vars].flatten()

        # Create a mask for removing long-list entries with missing or zero vals
        # There should be no zero values in sparse matrices, but not a lot of
        # code is required to remove them
        selected = self._notnan_mask(values)
        if self.exclude_zeros:
            included = values != 0
            if not self.only_numeric:
                disc_mask = np.array(
                    [var.is_discrete
                     for var, useful in zip(self.data.domain.attributes, useful_vars)
                     if useful])
                if sp.issparse(x):
                    included |= disc_mask[items]
                else:
                    included |= np.tile(disc_mask, len(x))
            selected &= included

        # Filter the long list
        idcol = idcol[selected]
        items = items[selected]
        values = values[selected]

        domain = self._prepare_domain(item_names, id_names)
        return Table.from_numpy(domain, np.vstack((idcol, items)).T, values)

    def _get_useful_vars(self):
        domain = self.data.domain

        if self.exclude_zeros or self.only_numeric:
            cont_vars = np.array([var.is_continuous for var in domain.attributes])
        if self.only_numeric:
            useful_vars = cont_vars
        else:
            useful_vars = np.full(len(domain.attributes), True)

        if self.idvar:
            ididx = domain.index(self.idvar)
            if ididx >= 0:
                useful_vars[ididx] = False
        return useful_vars

    def _get_item_names(self, useful_vars):
        return tuple(
            var.name
            for var, useful in zip(self.data.domain.attributes, useful_vars)
            if useful)

    def _prepare_domain(self, item_names, idnames=()):
        idvar = self.idvar
        if idvar is None:
            idvar = ContinuousVariable(DEFAULT_NAME_FOR_ROW)
        elif self.idvar.is_string:
            idvar = DiscreteVariable(idvar.name, values=tuple(idnames))

        # Renames without a warning: with only three columns, any intelligent
        # user will realize why renaming
        item_var_name, value_var_name = get_unique_names(
            [idvar.name],
            [self.item_var_name or DEFAULT_ITEM_NAME,
             self.value_var_name or DEFAULT_VALUE_NAME]
        )
        item_var = DiscreteVariable(item_var_name, values=item_names)
        value_var = ContinuousVariable(value_var_name)
        return Domain([idvar, item_var], [value_var])


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWMelt).run(Table("zoo")[50:])
