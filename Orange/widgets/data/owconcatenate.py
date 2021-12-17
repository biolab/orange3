"""
Concatenate
===========

Concatenate (append) two or more datasets.

"""
from collections import OrderedDict, namedtuple, defaultdict
from functools import reduce
from itertools import chain, count
from typing import List, Optional, Sequence

import numpy as np
from AnyQt.QtWidgets import QFormLayout
from AnyQt.QtCore import Qt

import Orange.data
from Orange.data.util import get_unique_names_duplicates, get_unique_names
from Orange.util import flatten
from Orange.widgets import widget, gui, settings
from Orange.widgets.settings import Setting
from Orange.widgets.utils.annotated_data import add_columns
from Orange.widgets.utils.sql import check_sql_input, check_sql_input_sequence
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, MultiInput, Output, Msg


class OWConcatenate(widget.OWWidget):
    name = "Concatenate"
    description = "Concatenate (append) two or more datasets."
    priority = 1111
    icon = "icons/Concatenate.svg"
    keywords = ["append", "join", "extend"]

    class Inputs:
        primary_data = Input("Primary Data", Orange.data.Table)
        additional_data = MultiInput(
            "Additional Data", Orange.data.Table, default=True
        )

    class Outputs:
        data = Output("Data", Orange.data.Table)

    class Error(widget.OWWidget.Error):
        bow_concatenation = Msg("Inputs must be of the same type.")

    class Warning(widget.OWWidget.Warning):
        renamed_variables = Msg(
            "Variables with duplicated names have been renamed.")

    merge_type: int
    append_source_column: bool
    source_column_role: int
    source_attr_name: str

    #: Domain merging operations
    MergeUnion, MergeIntersection = 0, 1

    #: Domain role of the "Source ID" attribute.
    ClassRole, AttributeRole, MetaRole = 0, 1, 2

    #: Selected domain merging operation
    merge_type = settings.Setting(0)
    #: Append source ID column
    append_source_column = settings.Setting(False)
    #: Selected "Source ID" domain role
    source_column_role = settings.Setting(0)
    #: User specified name for the "Source ID" attr
    source_attr_name = settings.Setting("Source ID")

    ignore_compute_value = settings.Setting(False)

    want_main_area = False
    resizing_enabled = False

    domain_opts = ("all variables that appear in input tables",
                   "only variables that appear in all tables")

    id_roles = ("Class attribute", "Attribute", "Meta attribute")

    auto_commit = Setting(True)

    def __init__(self):
        super().__init__()

        self.primary_data = None
        self._more_data_input: List[Optional[Orange.data.Table]] = []

        self.mergebox = gui.vBox(self.controlArea, "Variable Merging")
        box = gui.radioButtons(
            self.mergebox, self, "merge_type",
            callback=self._merge_type_changed)

        gui.widgetLabel(
            box, self.tr("When there is no primary table, " +
                         "the output should contain:"))

        for opts in self.domain_opts:
            gui.appendRadioButton(box, self.tr(opts))

        gui.separator(box)

        label = gui.widgetLabel(
            box,
            self.tr("The resulting table will have a class only if there " +
                    "is no conflict between input classes."))
        label.setWordWrap(True)

        gui.separator(box)
        gui.checkBox(
            box, self, "ignore_compute_value",
            "Treat variables with the same name as the same variable,\n"
            "even if they are computed using different formulae.",
            callback=self.commit.deferred, stateWhenDisabled=False)
        ###
        box = gui.vBox(
            self.controlArea, self.tr("Source Identification"),
        )

        cb = gui.checkBox(
            box, self, "append_source_column",
            self.tr("Append data source IDs"),
            callback=self._source_changed)

        ibox = gui.indentedBox(box, sep=gui.checkButtonOffsetHint(cb))

        form = QFormLayout(
            spacing=8,
            labelAlignment=Qt.AlignLeft,
            formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow
        )

        form.addRow(
            self.tr("Feature name:"),
            gui.lineEdit(ibox, self, "source_attr_name", valueType=str,
                         callback=self._source_changed))

        form.addRow(
            self.tr("Place:"),
            gui.comboBox(ibox, self, "source_column_role", items=self.id_roles,
                         callback=self._source_changed))

        ibox.layout().addLayout(form)
        mleft, mtop, mright, _ = ibox.layout().getContentsMargins()
        ibox.layout().setContentsMargins(mleft, mtop, mright, 4)

        cb.disables.append(ibox)
        cb.makeConsistent()

        gui.auto_apply(self.buttonsArea, self, "auto_commit")

    @Inputs.primary_data
    @check_sql_input
    def set_primary_data(self, data):
        self.primary_data = data

    @Inputs.additional_data
    @check_sql_input_sequence
    def set_more_data(self, index, data):
        self._more_data_input[index] = data

    @Inputs.additional_data.insert
    @check_sql_input_sequence
    def insert_more_data(self, index, data):
        self._more_data_input.insert(index, data)

    @Inputs.additional_data.remove
    def remove_more_data(self, index):
        self._more_data_input.pop(index)

    @property
    def more_data(self) -> Sequence[Orange.data.Table]:
        return [t for t in self._more_data_input if t is not None]

    def handleNewSignals(self):
        self.mergebox.setDisabled(self.primary_data is not None)
        if self.incompatible_types():
            self.Error.bow_concatenation()
        else:
            self.Error.bow_concatenation.clear()
            self.commit.now()

    def incompatible_types(self):
        types_ = set()
        if self.primary_data is not None:
            types_.add(type(self.primary_data))
        for table in self.more_data:
            types_.add(type(table))
        if len(types_) > 1:
            return True

        return False

    @gui.deferred
    def commit(self):
        self.Warning.renamed_variables.clear()
        tables, domain, source_var = [], None, None
        if self.primary_data is not None:
            tables = [self.primary_data] + list(self.more_data)
            domain = self.primary_data.domain
        elif self.more_data:
            if self.ignore_compute_value:
                tables = self._dumb_tables()
            else:
                tables = self.more_data
            domains = [table.domain for table in tables]
            domain = self.merge_domains(domains)

        if tables and self.append_source_column:
            assert domain is not None
            names = [getattr(t, 'name', '') for t in tables]
            if len(names) != len(set(names)):
                names = ['{} ({})'.format(name, i)
                         for i, name in enumerate(names)]
            source_var = Orange.data.DiscreteVariable(
                get_unique_names(domain, self.source_attr_name),
                values=names
            )
            places = ["class_vars", "attributes", "metas"]
            domain = add_columns(
                domain,
                **{places[self.source_column_role]: (source_var,)})

        tables = [table.transform(domain) for table in tables]
        if tables:
            data = type(tables[0]).concatenate(tables)
            if source_var:
                source_ids = np.array(list(flatten(
                    [i] * len(table) for i, table in enumerate(tables)))).reshape((-1, 1))
                parts = [data.Y, data.X, data.metas]
                with data.unlocked(parts[self.source_column_role]):
                    data[:, source_var] = source_ids
        else:
            data = None

        self.Outputs.data.send(data)

    def _dumb_tables(self):
        def enumerated_parts(domain):
            return enumerate((domain.attributes, domain.class_vars, domain.metas))

        compute_value_groups = defaultdict(set)
        for table in self.more_data:
            for part, part_vars in enumerated_parts(table.domain):
                for var in part_vars:
                    desc = (var.name, type(var), part)
                    compute_value_groups[desc].add(var.compute_value)
        to_dumbify = {desc
                      for desc, compute_values in compute_value_groups.items()
                      if len(compute_values) > 1}

        dumb_tables = []
        for table in self.more_data:
            dumb_domain = Orange.data.Domain(
                *[[var.copy(compute_value=None)
                   if (var.name, type(var), part) in to_dumbify
                   else var
                   for var in part_vars]
                  for part, part_vars in enumerated_parts(table.domain)])
            dumb_table = type(table).from_numpy(
                dumb_domain,
                table.X, table.Y, table.metas, table.W,
                table.attributes, table.ids)
            dumb_tables.append(dumb_table)
        return dumb_tables

    def _merge_type_changed(self, ):
        if self.incompatible_types():
            self.Error.bow_concatenation()
        else:
            self.Error.bow_concatenation.clear()
            if self.primary_data is None and self.more_data:
                self.commit.deferred()

    def _source_changed(self):
        self.commit.deferred()

    def send_report(self):
        items = OrderedDict()
        if self.primary_data is not None:
            items["Domain"] = "from primary data"
        else:
            items["Domain"] = self.tr(self.domain_opts[self.merge_type]).lower()
        if self.append_source_column:
            items["Source data ID"] = "{} (as {})".format(
                self.source_attr_name,
                self.id_roles[self.source_column_role].lower())
        self.report_items(items)

    def merge_domains(self, domains):
        def fix_names(part):
            for i, attr, name in zip(count(), part, name_iter):
                if attr.name != name:
                    part[i] = attr.renamed(name)
                    self.Warning.renamed_variables()

        oper = set.union if self.merge_type == OWConcatenate.MergeUnion \
            else set.intersection
        parts = [self._get_part(domains, oper, part)
                 for part in ("attributes", "class_vars", "metas")]
        all_names = [var.name for var in chain(*parts)]
        name_iter = iter(get_unique_names_duplicates(all_names))
        for part in parts:
            fix_names(part)
        domain = Orange.data.Domain(*parts)
        return domain

    @classmethod
    def _get_part(cls, domains, oper, part):
        # keep the order of variables: first compute union or intersections as
        # sets, then iterate through chained parts
        vars_by_domain = [getattr(domain, part) for domain in domains]
        valid = reduce(oper, map(set, vars_by_domain))
        valid_vars = [var for var in chain(*vars_by_domain) if var in valid]
        return cls._unique_vars(valid_vars)

    @staticmethod
    def _unique_vars(seq: List[Orange.data.Variable]):
        AttrDesc = namedtuple(
            "AttrDesc",
            ("template", "original", "values", "number_of_decimals"))

        attrs = {}
        for el in seq:
            desc = attrs.get(el)
            if desc is None:
                attrs[el] = AttrDesc(el, True,
                                     el.is_discrete and el.values,
                                     el.is_continuous and el.number_of_decimals)
                continue
            if desc.template.is_discrete:
                sattr_values = set(desc.values)
                # don't use sets: keep the order
                missing_values = tuple(
                    val for val in el.values if val not in sattr_values
                )
                if missing_values:
                    attrs[el] = attrs[el]._replace(
                        original=False,
                        values=desc.values + missing_values)
            elif desc.template.is_continuous:
                if el.number_of_decimals > desc.number_of_decimals:
                    attrs[el] = attrs[el]._replace(
                        original=False,
                        number_of_decimals=el.number_of_decimals)

        new_attrs = []
        for desc in attrs.values():
            attr = desc.template
            if desc.original:
                new_attr = attr
            elif desc.template.is_discrete:
                new_attr = attr.copy()
                for val in desc.values[len(attr.values):]:
                    new_attr.add_value(val)
            else:
                assert desc.template.is_continuous
                new_attr = attr.copy(number_of_decimals=desc.number_of_decimals)
            new_attrs.append(new_attr)
        return new_attrs


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWConcatenate).run(
        insert_more_data=[(0, Orange.data.Table("iris")),
                          (1, Orange.data.Table("zoo"))])
