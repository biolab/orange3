import threading
import textwrap

import numpy as np

from Orange.widgets import widget, gui
from Orange.widgets.utils.localization import pl
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input

from Orange.data import \
    Table, StringVariable, DiscreteVariable, ContinuousVariable

try:
    from Orange.data.sql.table import SqlTable
except ImportError:
    def is_sql(_):
        return False
else:
    def is_sql(data):
        return isinstance(data, SqlTable)


class OWDataInfo(widget.OWWidget):
    name = "Data Info"
    id = "orange.widgets.data.info"
    description = "Display basic information about the data set"
    icon = "icons/DataInfo.svg"
    priority = 80
    category = "Data"
    keywords = "data info, information, inspect"

    class Inputs:
        data = Input("Data", Table)

    want_main_area = False
    buttons_area_orientation = None
    resizing_enabled = False

    def __init__(self):
        super().__init__()

        self.data_desc = {}
        self.data_attrs = {}
        self.description = gui.widgetLabel(
            gui.vBox(self.controlArea, box="Data table properties"))
        self.attributes = gui.widgetLabel(
            gui.vBox(self.controlArea, box="Additional attributes"))

    @Inputs.data
    def data(self, data):
        if data is None:
            self.data_desc = self.data_attrs = {}
            self.update_info()
        else:
            self.data_desc = {
                label: value
                for label, func in (("Name", self._p_name),
                                    ("Location", self._p_location),
                                    ("Size", self._p_size),
                                    ("Features", self._p_features),
                                    ("Targets", self._p_targets),
                                    ("Metas", self._p_metas),
                                    ("Missing data", self._p_missing))
                if bool(value := func(data))}
            self.data_attrs = data.attributes
            self.update_info()

            if is_sql(data):
                def set_exact_length():
                    self.data_desc["Size"] = self._p_size(data, exact=True)
                    self.update_info()

                threading.Thread(target=set_exact_length).start()

    def update_info(self):
        style = """<style>
                       th { text-align: right; vertical-align: top; }
                       th, td { padding-top: 4px; line-height: 125%}
                    </style>"""

        def dict_as_table(d):
            return "<table>" + \
                   "".join(f"<tr><th>{label}: </th><td>" + \
                           '<br/>'.join(textwrap.wrap(value, width=60)) + \
                           "</td></tr>"
                           for label, value in d.items()) + \
                   "</table>"

        if not self.data_desc:
            self.description.setText("No data.")
        else:
            self.description.setText(style + dict_as_table(self.data_desc))
        self.attributes.setHidden(not self.data_attrs)
        if self.data_attrs:
            self.attributes.setText(
                style + dict_as_table({k: str(v)
                                       for k, v in self.data_attrs.items()}))

    def send_report(self):
        if self.data_desc:
            self.report_items("Data table properties", self.data_desc)
        if self.data_attrs:
            self.report_items("Additional attributes", self.data_attrs)

    @staticmethod
    def _p_name(data):
        return getattr(data, "name", "-")

    @staticmethod
    def _p_location(data):
        if not is_sql(data):
            return None

        connection_string = ' '.join(
            f'{key}={value}'
            for key, value in data.connection_params.items()
            if value is not None and key != 'password')
        return f"SQL Table using connection:<br/>{connection_string}"

    @staticmethod
    def _p_size(data, exact=False):
        exact = exact or is_sql(data)
        if exact:
            n = len(data)
            desc = f"{n} {pl(n, 'row')}"
        else:
            n = data.approx_len()
            desc = f"~{n} {pl(n, 'row')}"
        ncols = len(data.domain.variables) + len(data.domain.metas)
        desc += f", {ncols} {pl(ncols, 'column')}"

        sparseness = [s for s, m in (("features", data.X_density),
                                     ("meta attributes", data.metas_density),
                                     ("targets", data.Y_density)) if m() > 1]
        if sparseness:
            desc += "; sparse {', '.join(sparseness)}"
        return desc

    @classmethod
    def _p_features(cls, data):
        return cls._pack_var_counts(data.domain.attributes)

    def _p_targets(self, data):
        if class_var := data.domain.class_var:
            if class_var.is_continuous:
                return "numeric target variable"
            else:
                nclasses = len(class_var.values)
                return "categorical outcome with " \
                       f"{nclasses} {pl(nclasses, 'class|classes')}"
        if class_vars := data.domain.class_vars:
            disc_class = self._count(class_vars, DiscreteVariable)
            cont_class = self._count(class_vars, ContinuousVariable)
            if not cont_class:
                return f"{disc_class} categorical {pl(disc_class, 'target')}"
            elif not disc_class:
                return f"{cont_class} numeric {pl(cont_class, 'target')}"
            return "multi-target data,<br/>" + self._pack_var_counts(class_vars)

    @classmethod
    def _p_metas(cls, data):
        return cls._pack_var_counts(data.domain.metas)

    @staticmethod
    def _p_missing(data: Table):
        if is_sql(data):
            return "(not checked for SQL data)"

        counts = []
        for name, part, n_miss in ((pl(len(data.domain.attributes), "feature"),
                                    data.X, data.get_nan_count_attribute()),
                                   (pl(len(data.domain.class_vars), "targets"),
                                    data.Y, data.get_nan_count_class()),
                                   (pl(len(data.domain.metas), "meta variable"),
                                    data.metas, data.get_nan_count_metas())):
            if n_miss:
                counts.append(
                    f"{n_miss} ({n_miss / np.prod(part.shape):.1%}) in {name}")
        if not counts:
            return "none"
        return ", ".join(counts)

    @staticmethod
    def _count(s, tpe):
        return sum(isinstance(x, tpe) for x in s)

    @classmethod
    def _pack_var_counts(cls, s):
        counts = (
            (name, cls._count(s, type_))
            for name, type_ in (("categorical", DiscreteVariable),
                                ("numeric", ContinuousVariable),
                                ("text", StringVariable)))
        return ", ".join(f"{count} {name}" for name, count in counts if count)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWDataInfo).run(Table("heart_disease"))
