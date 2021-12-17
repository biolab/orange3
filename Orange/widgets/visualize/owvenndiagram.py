"""
Venn Diagram Widget
-------------------

"""

import math
import unicodedata
from collections import namedtuple, defaultdict
from itertools import compress, count
from functools import reduce
from operator import attrgetter
from xml.sax.saxutils import escape
from typing import Dict, Any, List, Mapping, Optional

import numpy as np

from AnyQt.QtWidgets import (
    QGraphicsScene, QGraphicsView, QGraphicsWidget,
    QGraphicsPathItem, QGraphicsTextItem, QStyle, QSizePolicy
)
from AnyQt.QtGui import (
    QPainterPath, QPainter, QTransform, QColor, QBrush, QPen, QPalette
)
from AnyQt.QtCore import Qt, QPointF, QRectF, QLineF
from AnyQt.QtCore import pyqtSignal as Signal

from Orange.data import Table, Domain, StringVariable, RowInstance
from Orange.data.util import get_unique_names_duplicates
from Orange.widgets import widget, gui
from Orange.widgets.settings import (
    DomainContextHandler, ContextSetting, Setting)
from Orange.widgets.utils import itemmodels, colorpalettes
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.utils.sql import check_sql_input_sequence
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import MultiInput, Output, Msg


_InputData = namedtuple("_InputData", ["key", "name", "table"])
_ItemSet = namedtuple("_ItemSet", ["key", "name", "title", "items"])

IDENTITY_STR = "Instance identity"
EQUALITY_STR = "Instance equality"


class VennVariableListModel(itemmodels.VariableListModel):
    def __init__(self):
        super().__init__([IDENTITY_STR, EQUALITY_STR])
        self.same_domains = True

    def set_variables(self, variables, same_domains):
        self[2:] = variables
        self.same_domains = same_domains

    def flags(self, index):
        if index.row() == 1 and not self.same_domains:
            return Qt.NoItemFlags
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled


class OWVennDiagram(widget.OWWidget):
    name = "Venn Diagram"
    description = "A graphical visualization of the overlap of data instances " \
                  "from a collection of input datasets."
    icon = "icons/VennDiagram.svg"
    priority = 280
    keywords = []
    settings_version = 2

    class Inputs:
        data = MultiInput("Data", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    class Error(widget.OWWidget.Error):
        instances_mismatch = Msg("Data sets do not contain the same instances.")
        too_many_inputs = Msg("Venn diagram accepts at most five datasets.")

    class Warning(widget.OWWidget.Warning):
        renamed_vars = Msg("Some variables have been renamed "
                           "to avoid duplicates.\n{}")

    selection: list

    settingsHandler = DomainContextHandler()
    # Indices of selected disjoint areas
    selection = Setting([], schema_only=True)
    #: Output unique items (one output row for every unique instance `key`)
    #: or preserve all duplicates in the output.
    output_duplicates = Setting(False)
    autocommit = Setting(True)
    rowwise = Setting(True)
    selected_feature = ContextSetting(IDENTITY_STR)

    want_main_area = False
    graph_name = "scene"
    atr_types = ['attributes', 'metas', 'class_vars']
    atr_vals = {'metas': 'metas', 'attributes': 'X', 'class_vars': 'Y'}
    row_vals = {'attributes': 'x', 'class_vars': 'y', 'metas': 'metas'}

    def __init__(self):
        super().__init__()

        # Diagram update is in progress
        self._updating = False
        self.__id_gen = count()  # 'key' generator for _InputData
        #: Connected input dataset signals.
        self._data_inputs: List[_InputData] = []
        # Input non-none datasets in the order they were 'connected'.
        self.__data: Optional[Dict[Any, _InputData]] = None
        # Extracted input item sets in the order they were 'connected'
        self.itemsets = {}
        # A list with 2 ** len(self.data) elements that store item sets
        # belonging to each area
        self.disjoint = []
        # A list with  2 ** len(self.data) elements that store keys of tables
        # intersected in each area
        self.area_keys = []

        # Main area view
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setFrameStyle(QGraphicsView.StyledPanel)

        self.controlArea.layout().addWidget(self.view)
        self.vennwidget = VennDiagram()
        self._resize()
        self.vennwidget.itemTextEdited.connect(self._on_itemTextEdited)
        self.scene.selectionChanged.connect(self._on_selectionChanged)

        self.scene.addItem(self.vennwidget)

        box = gui.radioButtonsInBox(
            self.buttonsArea, self, 'rowwise',
            ["Columns (features)", "Rows (instances), matched by", ],
            callback=self._on_matching_changed
        )
        gui.rubber(self.buttonsArea)
        gui.separator(self.buttonsArea, 10, 0)
        gui.comboBox(
            gui.indentedBox(box,
                            gui.checkButtonOffsetHint(box.buttons[0]),
                            Qt.Horizontal,
                            addSpaceBefore=False),
            self, "selected_feature",
            model=VennVariableListModel(),
            callback=self._on_inputAttrActivated,
            tooltip="Instances are identical if originally coming from the "
                    "same row of the same table.\n"
                    "Instances can be check for equality only if described by "
                    "the same variables.")
        box.layout().setSpacing(6)
        box.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)

        self.outputs_box = box = gui.vBox(self.buttonsArea,
                                          sizePolicy=(QSizePolicy.Preferred,
                                                      QSizePolicy.Preferred),
                                          stretch=0)
        gui.rubber(box)
        self.output_duplicates_cb = gui.checkBox(
            box, self, "output_duplicates", "Output duplicates",
            callback=lambda: self.commit(),  # pylint: disable=unnecessary-lambda
            stateWhenDisabled=False,
            attribute=Qt.WA_LayoutUsesWidgetRect)
        gui.auto_send(
            box, self, "autocommit", box=False, contentsMargins=(0, 0, 0, 0))
        gui.rubber(box)
        self._update_duplicates_cb()
        self._queue = []

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize()

    def showEvent(self, event):
        super().showEvent(event)
        self._resize()

    def _resize(self):
        # vennwidget draws so that the diagram fits into its geometry,
        # while labels take further 120 pixels, hence -120 in below formula
        size = max(200, min(self.view.width(), self.view.height()) - 120)
        self.vennwidget.resize(size, size)
        self.scene.setSceneRect(self.scene.itemsBoundingRect())

    @property
    def data(self) -> Mapping[Any, _InputData]:
        if self.__data is None:
            self.__data = {
                item.key: item for item in self._data_inputs[:5]
                if item.table is not None
            }
        return self.__data

    @Inputs.data
    @check_sql_input_sequence
    def setData(self, index: int, data: Optional[Table]):
        item = self._data_inputs[index]
        item = item._replace(
            name=data.name if data is not None else "",
            table=data
        )
        self._data_inputs[index] = item
        self.__data = None  # invalidate self.data
        self._setInterAttributes()

    @Inputs.data.insert
    @check_sql_input_sequence
    def insertData(self, index: int, data: Optional[Table]):
        key = next(self.__id_gen)
        item = _InputData(
            key, name=data.name if data is not None else "", table=data
        )
        self._data_inputs.insert(index, item)
        self.__data = None  # invalidate self.data
        if len(self._data_inputs) > 5:
            self.Error.too_many_inputs()
        self._setInterAttributes()

    @Inputs.data.remove
    def removeData(self, index: int):
        self.__data = None  # invalidate self.data
        self._data_inputs.pop(index)
        if len(self._data_inputs) <= 5:
            self.Error.too_many_inputs.clear()
        # Clear possible warnings.
        self.Warning.clear()
        self._setInterAttributes()

    def data_equality(self):
        """ Checks if all input datasets have same ids. """
        if not self.data.values():
            return True
        sets = []
        for val in self.data.values():
            sets.append(set(val.table.ids))
        inter = reduce(set.intersection, sets)
        return len(inter) == max(map(len, sets))

    def settings_compatible(self):
        self.Error.instances_mismatch.clear()
        if not self.rowwise:
            if not self.data_equality():
                self.vennwidget.clear()
                self.Error.instances_mismatch()
                self.itemsets = {}
                return False
        return True

    def handleNewSignals(self):
        self.vennwidget.clear()
        if not self.settings_compatible():
            self.invalidateOutput()
            return

        self._createItemsets()
        self._createDiagram()
        # If autocommit is enabled, _createDiagram already outputs data
        # If not, call commit from here
        if not self.autocommit:
            self.commit.now()

        super().handleNewSignals()

    def _intersection_string_attrs(self):
        sets = [set(string_attributes(data_.table.domain))
                for data_ in self.data.values()]
        if sets:
            return list(reduce(set.intersection, sets))
        return []

    def _all_domains_same(self):
        domains = [data_.table.domain for data_ in self.data.values()]
        # Domain.__hash__ is hacky, let's not use a set here, just for the case
        return not domains or all(domain == domains[0] for domain in domains)

    def _uses_feature(self):
        return isinstance(self.selected_feature, StringVariable)

    def _setInterAttributes(self):
        model = self.controls.selected_feature.model()
        same_domains = self._all_domains_same()
        variables = self._intersection_string_attrs()
        model.set_variables(variables, same_domains)
        if self.selected_feature == EQUALITY_STR and not same_domains \
                or self._uses_feature() and \
                self.selected_feature.name not in (var.name for var in variables):
            self.selected_feature = IDENTITY_STR

    @staticmethod
    def _hashes(table):
        # In the interest of space, we compare hashes. If this is not OK,
        # concatenate bytes instead of xoring hashes. Renaming a method
        # brings bonus points.
        return [hash(inst.x.data.tobytes())
                ^ hash(inst.y.data.tobytes())
                ^ hash(inst.metas.data.tobytes()) for inst in table]

    def _itemsForInput(self, key):
        """
        Calculates input for venn diagram, according to user's settings.
        """
        table = self.data[key].table
        if self.selected_feature == IDENTITY_STR:
            return list(table.ids)
        if self.selected_feature == EQUALITY_STR:
            return self._hashes(table)
        attr = self.selected_feature
        return [str(inst[attr]) for inst in table
                if not np.isnan(inst[attr])]

    def _createItemsets(self):
        """
        Create itemsets over rows or columns (domains) of input tables.
        """
        olditemsets = dict(self.itemsets)
        self.itemsets.clear()

        for key, input_ in self.data.items():
            if self.rowwise:
                items = self._itemsForInput(key)
            else:
                items = [el.name for el in input_.table.domain.attributes]
            name = input_.name
            if key in olditemsets and olditemsets[key].name == name:
                # Reuse the title (which might have been changed by the user)
                title = olditemsets[key].title
            else:
                title = name

            itemset = _ItemSet(key=key, name=name, title=title, items=items)
            self.itemsets[key] = itemset

    def _createDiagram(self):
        self._updating = True

        oldselection = list(self.selection)

        n = len(self.itemsets)
        self.disjoint, self.area_keys = \
            self.get_disjoint(set(s.items) for s in self.itemsets.values())

        vennitems = []
        colors = colorpalettes.LimitedDiscretePalette(n, force_glasbey=True)

        for i, item in enumerate(self.itemsets.values()):
            cnt = len(set(item.items))
            cnt_all = len(item.items)
            if cnt != cnt_all:
                fmt = '{} <i>(all: {})</i>'
            else:
                fmt = '{}'
            counts = fmt.format(cnt, cnt_all)
            gr = VennSetItem(text=item.title, informativeText=counts)
            color = colors[i]
            color.setAlpha(100)
            gr.setBrush(QBrush(color))
            gr.setPen(QPen(Qt.NoPen))
            vennitems.append(gr)

        self.vennwidget.setItems(vennitems)

        for i, area in enumerate(self.vennwidget.vennareas()):
            area_items = list(map(str, list(self.disjoint[i])))
            if i:
                area.setText("{0}".format(len(area_items)))

            label = disjoint_set_label(i, n, simplify=False)
            tooltip = "<h4>|{}| = {}</h4>".format(label, len(area_items))
            if self._uses_feature() or not self.rowwise:
                # Nothing readable to show when matching by identity or equality
                tooltip += "<span>" + ", ".join(map(escape, area_items[:32]))
                if len(area_items) > 32:
                    tooltip += f"</br>({len(area_items) - 32} items not shown)"
                tooltip += "</span>"

            area.setToolTip(tooltip)

            area.setPen(QPen(QColor(10, 10, 10, 200), 1.5))
            area.setFlag(QGraphicsPathItem.ItemIsSelectable, True)
            area.setSelected(i in oldselection)

        self._updating = False
        self._on_selectionChanged()

    def _on_selectionChanged(self):
        if self._updating:
            return

        areas = self.vennwidget.vennareas()
        self.selection = [i for i, area in enumerate(areas) if area.isSelected()]
        self.invalidateOutput()

    def _update_duplicates_cb(self):
        self.output_duplicates_cb.setEnabled(
            self.rowwise and self._uses_feature())

    def _on_matching_changed(self):
        self._update_duplicates_cb()
        if not self.settings_compatible():
            self.invalidateOutput()
            return
        self._createItemsets()
        self._createDiagram()

    def _on_inputAttrActivated(self):
        self.rowwise = True
        self._on_matching_changed()

    def _on_itemTextEdited(self, index, text):
        text = str(text)
        key = list(self.itemsets)[index]
        self.itemsets[key] = self.itemsets[key]._replace(title=text)

    def invalidateOutput(self):
        self.commit.deferred()

    def merge_data(self, domain, values, ids=None):
        X, metas, class_vars = None, None, None
        renamed = []
        names = [var.name for val in domain.values() for var in val]
        unique_names = iter(get_unique_names_duplicates(names))

        for val in domain.values():
            for n, idx, var in zip(names, count(), val):
                u = next(unique_names)
                if n != u:
                    val[idx] = var.copy(name=u)
                    renamed.append(n)
        if renamed:
            self.Warning.renamed_vars(', '.join(renamed))
        if 'attributes' in values:
            X = np.hstack(values['attributes'])
        if 'metas' in values:
            metas = np.hstack(values['metas'])
            n = len(metas)
        if 'class_vars' in values:
            class_vars = np.hstack(values['class_vars'])
            n = len(class_vars)
        if X is None:
            X = np.empty((n, 0))
        table = Table.from_numpy(Domain(**domain), X, class_vars, metas)
        if ids is not None:
            table.ids = ids
        return table

    def extract_columnwise(self, var_dict, columns=None):
        domain = {type_ : [] for type_ in self.atr_types}
        values = defaultdict(list)
        renamed = []
        for atr_type, vars_dict in var_dict.items():
            for var_name, var_data in vars_dict.items():
                is_selected = bool(columns) and var_name.name in columns
                if var_data[0]:
                    #columns are different, copy all, rename them
                    for var, table_key in var_data[1]:
                        idx = list(self.data).index(table_key) + 1
                        new_atr = var.copy(name=f'{var_name.name} ({idx})')
                        if columns and atr_type == 'attributes':
                            new_atr.attributes['Selected'] = is_selected
                        domain[atr_type].append(new_atr)
                        renamed.append(var_name.name)
                        values[atr_type].append(getattr(self.data[table_key].table[:, var_name],
                                                        self.atr_vals[atr_type])
                                                .reshape(-1, 1))
                else:
                    new_atr = var_data[1][0][0].copy()
                    if columns and atr_type == 'attributes':
                        new_atr.attributes['Selected'] = is_selected
                    domain[atr_type].append(new_atr)
                    values[atr_type].append(getattr(self.data[var_data[1][0][1]].table[:, var_name],
                                                    self.atr_vals[atr_type])
                                            .reshape(-1, 1))
        if renamed:
            self.Warning.renamed_vars(', '.join(renamed))
        return self.merge_data(domain, values)

    def curry_merge(self, table_key, atr_type, ids=None, selection=False):
        if self.rowwise:
            check_equality = self.arrays_equal_rows
        else:
            check_equality = self.arrays_equal_cols

        def inner(new_atrs, atr):
            """
            Atrs - list of variables we wish to merge
            new_atrs - dictionary where key is old var, val
                is [is_different:bool, table_keys:list]), is_different is set to True,
                if we are outputing duplicates, but the value is arbitrary
            """
            if atr in new_atrs:
                if not selection and self.output_duplicates:
                    #if output_duplicates, we just check if compute value is the same
                    new_atrs[atr][0] = True
                elif not new_atrs[atr][0]:
                    for var, key in new_atrs[atr][1]:
                        if not check_equality(table_key,
                                              key,
                                              atr.name,
                                              self.atr_vals[atr_type],
                                              type(var), ids):
                            new_atrs[atr][0] = True
                            break
                new_atrs[atr][1].append((atr, table_key))
            else:
                new_atrs[atr] = [False, [(atr, table_key)]]
            return new_atrs
        return inner

    def arrays_equal_rows(self, key1, key2, name, data_type, type_, ids):
        #gets masks, compares same as cols
        t1 = self.data[key1].table
        t2 = self.data[key2].table
        inter_val = set(ids[key1]) & set(ids[key2])
        t1_inter = [ids[key1][val] for val in inter_val]
        t2_inter = [ids[key2][val] for val in inter_val]
        return arrays_equal(
            getattr(t1[t1_inter, name],
                    data_type).reshape(-1, 1),
            getattr(t2[t2_inter, name],
                    data_type).reshape(-1, 1),
            type_)

    def arrays_equal_cols(self, key1, key2, name, data_type, type_, _ids=None):
        return arrays_equal(
            getattr(self.data[key1].table[:, name],
                    data_type),
            getattr(self.data[key2].table[:, name],
                    data_type),
            type_)

    def create_from_columns(self, columns, relevant_keys, get_selected):
        """
        Columns are duplicated only if values differ (even
        if only in order of values), origin table name and input slot is added to column name.
        """
        var_dict = {}
        for atr_type in self.atr_types:
            container = {}
            for table_key in relevant_keys:
                table = self.data[table_key].table
                if atr_type == 'attributes':
                    if get_selected:
                        atrs = list(compress(table.domain.attributes,
                                             [c.name in columns for c in table.domain.attributes]))
                    else:
                        atrs = getattr(table.domain, atr_type)
                else:
                    atrs = getattr(table.domain, atr_type)
                merge_vars = self.curry_merge(table_key, atr_type)
                container = reduce(merge_vars, atrs, container)
            var_dict[atr_type] = container

        if get_selected:
            annotated = self.extract_columnwise(var_dict, None)
        else:
            annotated = self.extract_columnwise(var_dict, columns)

        return annotated

    def extract_rowwise(self, var_dict, ids=None, selection=False):
        """
        keys : ['attributes', 'metas', 'class_vars']
        vals: new_atrs - dictionary where key is old name, val
            is [is_different:bool, table_keys:list])
        ids: dict with ids for each table
        """
        all_ids = sorted(reduce(set.union, [set(val) for val in ids.values()], set()))

        permutations = {}
        for table_key, dict_ in ids.items():
            permutations[table_key] = get_perm(list(dict_), all_ids)

        domain = {type_ : [] for type_ in self.atr_types}
        values = defaultdict(list)
        renamed = []
        for atr_type, vars_dict in var_dict.items():
            for var_name, var_data in vars_dict.items():
                different = var_data[0]
                if different:
                    # Columns are different, copy and rename them.
                    # Renaming is done here to mark appropriately the source table.
                    # Additional strange clashes are checked later in merge_data
                    for var, table_key in var_data[1]:
                        temp = self.data[table_key].table
                        idx = list(self.data).index(table_key) + 1
                        domain[atr_type].append(var.copy(name='{} ({})'.format(var_name, idx)))
                        renamed.append(var_name.name)
                        v = getattr(temp[list(ids[table_key].values()), var_name],
                                    self.atr_vals[atr_type])
                        perm = permutations[table_key]
                        if len(v) < len(all_ids):
                            values[atr_type].append(pad_columns(v, perm, len(all_ids)))
                        else:
                            values[atr_type].append(v[perm].reshape(-1, 1))
                else:
                    value = np.full((len(all_ids), 1), np.nan)
                    domain[atr_type].append(var_data[1][0][0].copy())
                    for _, table_key in var_data[1]:
                        #different tables have different part of the same attribute vector
                        perm = permutations[table_key]
                        v = getattr(self.data[table_key].table[list(ids[table_key].values()),
                                                               var_name],
                                    self.atr_vals[atr_type]).reshape(-1, 1)
                        value = value.astype(v.dtype, copy=False)
                        value[perm] = v
                    values[atr_type].append(value)

        if renamed:
            self.Warning.renamed_vars(', '.join(renamed))
        ids = None if self._uses_feature() else np.array(all_ids)
        table = self.merge_data(domain, values, ids)
        if selection:
            mask = [idx in self.selected_items for idx in all_ids]
            return create_annotated_table(table, mask)
        return table

    def get_indices(self, table, selection):
        """Returns mappings of ids (be it row id or string) to indices in tables"""
        if self.selected_feature == IDENTITY_STR:
            items = table.ids
            ids = range(len(table))
        elif self.selected_feature == EQUALITY_STR:
            items, ids = np.unique(self._hashes(table), return_index=True)
        else:
            items = getattr(table[:, self.selected_feature], 'metas')
            if self.output_duplicates and selection:
                items, inverse = np.unique(items, return_inverse=True)
                ids = [np.nonzero(inverse == idx)[0] for idx in range(len(items))]
            else:
                items, ids = np.unique(items, return_index=True)

        if selection:
            return {item: idx for item, idx in zip(items, ids)
                    if item in self.selected_items}

        return dict(zip(items, ids))

    def get_indices_to_match_by(self, relevant_keys, selection=False):
        dict_ = {}
        for key in relevant_keys:
            table = self.data[key].table
            dict_[key] = self.get_indices(table, selection)
        return dict_

    def create_from_rows(self, relevant_ids, selection=False):
        var_dict = {}
        for atr_type in self.atr_types:
            container = {}
            for table_key in relevant_ids:
                merge_vars = self.curry_merge(table_key, atr_type, relevant_ids, selection)
                atrs = getattr(self.data[table_key].table.domain, atr_type)
                container = reduce(merge_vars, atrs, container)
            var_dict[atr_type] = container
        if self.output_duplicates and not selection:
            return self.extract_rowwise_duplicates(var_dict, relevant_ids)
        return self.extract_rowwise(var_dict, relevant_ids, selection)

    def expand_table(self, table, atrs, metas, cv):
        exp = []
        n = 1 if isinstance(table, RowInstance) else len(table)
        if isinstance(table, RowInstance):
            ids = table.id.reshape(-1, 1)
            atr_vals = self.row_vals
        else:
            ids = table.ids.reshape(-1, 1)
            atr_vals = self.atr_vals
        for all_el, atr_type in zip([atrs, metas, cv], self.atr_types):
            cur_el = getattr(table.domain, atr_type)
            array = np.full((n, len(all_el)), np.nan)
            if cur_el:
                perm = get_perm(cur_el, all_el)
                b = getattr(table, atr_vals[atr_type]).reshape(len(array), len(perm))
                array = array.astype(b.dtype, copy=False)
                array[:, perm] = b
            exp.append(array)
        return (*exp, ids)

    def extract_rowwise_duplicates(self, var_dict, ids):
        all_ids = sorted(reduce(set.union, [set(val) for val in ids.values()], set()))
        sort_key = attrgetter("name")
        all_atrs = sorted(var_dict['attributes'], key=sort_key)
        all_metas = sorted(var_dict['metas'], key=sort_key)
        all_cv = sorted(var_dict['class_vars'], key=sort_key)

        all_x, all_y, all_m = [], [], []
        new_table_ids = []
        for idx in all_ids:
            #iterate trough tables with same idx
            for table_key, t_indices in ids.items():
                if idx not in t_indices:
                    continue
                map_ = t_indices[idx]
                extracted = self.data[table_key].table[map_]
                # pylint: disable=unbalanced-tuple-unpacking
                x, m, y, t_ids = self.expand_table(extracted, all_atrs, all_metas, all_cv)
                all_x.append(x)
                all_y.append(y)
                all_m.append(m)
                new_table_ids.append(t_ids)
        domain = {'attributes': all_atrs, 'metas': all_metas, 'class_vars': all_cv}
        values = {'attributes': [np.vstack(all_x)],
                  'metas': [np.vstack(all_m)],
                  'class_vars': [np.vstack(all_y)]}
        return self.merge_data(domain, values, np.vstack(new_table_ids))

    @gui.deferred
    def commit(self):
        if not self.vennwidget.vennareas() or not self.data:
            self.Outputs.selected_data.send(None)
            self.Outputs.annotated_data.send(None)
            return

        self.selected_items = reduce(
            set.union, [self.disjoint[index] for index in self.selection],
            set()
        )
        selected_keys = reduce(
            set.union, [set(self.area_keys[area]) for area in self.selection],
            set())
        selected = None

        if self.rowwise:
            if self.selected_items:
                selected_ids = self.get_indices_to_match_by(
                    selected_keys, bool(self.selection))
                selected = self.create_from_rows(selected_ids, False)
            annotated_ids = self.get_indices_to_match_by(self.data)
            annotated = self.create_from_rows(annotated_ids, True)
        else:
            annotated = self.create_from_columns(self.selected_items, self.data, False)
            if self.selected_items:
                selected = self.create_from_columns(self.selected_items, selected_keys, True)

        self.Outputs.selected_data.send(selected)
        self.Outputs.annotated_data.send(annotated)

    def send_report(self):
        self.report_plot()

    def get_disjoint(self, sets):
        """
        Return all disjoint subsets.
        """
        sets = list(sets)
        n = len(sets)
        disjoint_sets = [None] * (2 ** n)
        included_tables = [None] * (2 ** n)
        for i in range(2 ** n):
            key = setkey(i, n)
            included = [s for s, inc in zip(sets, key) if inc]
            if included:
                excluded = [s for s, inc in zip(sets, key) if not inc]
                s = reduce(set.intersection, included)
                s = reduce(set.difference, excluded, s)
            else:
                s = set()
            disjoint_sets[i] = s
            included_tables[i] = [k for k, inc in zip(self.data, key) if inc]

        return disjoint_sets, included_tables

    @classmethod
    def migrate_settings(cls, settings, version):
        if version < 3:
            if settings.pop("selected_feature", None) is None:
                settings["selected_feature"] = IDENTITY_STR


def string_attributes(domain):
    """
    Return all string attributes from the domain.
    """
    return [attr for attr in domain.variables + domain.metas if attr.is_string]


def disjoint_set_label(i, n, simplify=False):
    """
    Return a html formated label for a disjoint set indexed by `i`.
    """
    intersection = unicodedata.lookup("INTERSECTION")
    # comp = unicodedata.lookup("COMPLEMENT")  #
    # This depends on the font but the unicode complement in
    # general does not look nice in a super script so we use
    # plain c instead.
    comp = "c"

    def label_for_index(i):
        return chr(ord("A") + i)

    if simplify:
        return "".join(label_for_index(i) for i, b in enumerate(setkey(i, n))
                       if b)
    else:
        return intersection.join(label_for_index(i) +
                                 ("" if b else "<sup>" + comp + "</sup>")
                                 for i, b in enumerate(setkey(i, n)))


class VennSetItem(QGraphicsPathItem):
    def __init__(self, parent=None, text="", informativeText=""):
        super(VennSetItem, self).__init__(parent)
        # Plain text title (editable by the VennDiagram)
        self.text = text
        # Extra informative text (possibly rich text)
        self.informativeText = informativeText

# TODO: Use palette's selected/highligted text / background colors to
# indicate selection


class VennIntersectionArea(QGraphicsPathItem):
    def __init__(self, parent=None, text=""):
        super().__init__(parent)
        self.setAcceptHoverEvents(True)
        self.setPen(QPen(Qt.NoPen))

        self.text = QGraphicsTextItem(self)
        layout = self.text.document().documentLayout()
        layout.documentSizeChanged.connect(self._onLayoutChanged)

        self._text = text
        self._anchor = QPointF()

    def setText(self, text):
        if self._text != text:
            self._text = text
            self.text.setPlainText(text)

    def setTextAnchor(self, pos):
        if self._anchor != pos:
            self._anchor = pos
            self._updateTextAnchor()

    def hoverEnterEvent(self, event):
        self.setZValue(self.zValue() + 1)
        return QGraphicsPathItem.hoverEnterEvent(self, event)

    def hoverLeaveEvent(self, event):
        self.setZValue(self.zValue() - 1)
        return QGraphicsPathItem.hoverLeaveEvent(self, event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.AltModifier:
                self.setSelected(False)
            elif event.modifiers() & Qt.ControlModifier:
                self.setSelected(not self.isSelected())
            elif event.modifiers() & Qt.ShiftModifier:
                self.setSelected(True)
            else:
                for area in self.parentWidget().vennareas():
                    area.setSelected(False)
                self.setSelected(True)

    def mouseReleaseEvent(self, event):
        pass

    def paint(self, painter, option, _widget=None):
        painter.save()
        path = self.path()
        brush = QBrush(self.brush())
        pen = QPen(self.pen())

        if option.state & QStyle.State_Selected:
            pen.setColor(Qt.red)
            brush.setStyle(Qt.DiagCrossPattern)
            brush.setColor(QColor(40, 40, 40, 100))

        elif option.state & QStyle.State_MouseOver:
            pen.setColor(Qt.blue)

        if option.state & QStyle.State_MouseOver:
            brush.setColor(QColor(100, 100, 100, 100))
            if brush.style() == Qt.NoBrush:
                # Make sure the highlight is actually visible.
                brush.setStyle(Qt.SolidPattern)

        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawPath(path)
        painter.restore()

    def itemChange(self, change, value):
        if change == QGraphicsPathItem.ItemSelectedHasChanged:
            self.setZValue(self.zValue() + (1 if value else -1))
        return QGraphicsPathItem.itemChange(self, change, value)

    def _updateTextAnchor(self):
        rect = self.text.boundingRect()
        pos = anchor_rect(rect, self._anchor)
        self.text.setPos(pos)

    def _onLayoutChanged(self):
        self._updateTextAnchor()


class GraphicsTextEdit(QGraphicsTextItem):
    #: Edit triggers
    NoEditTriggers, DoubleClicked = 0, 1

    editingFinished = Signal()
    editingStarted = Signal()

    documentSizeChanged = Signal()

    def __init__(self, *args, **kwargs):
        super(GraphicsTextEdit, self).__init__(*args, **kwargs)
        self.setCursor(Qt.IBeamCursor)
        self.setTabChangesFocus(True)
        self._edittrigger = GraphicsTextEdit.DoubleClicked
        self._editing = False
        self.document().documentLayout().documentSizeChanged.connect(
            self.documentSizeChanged
        )

    def mouseDoubleClickEvent(self, event):
        super(GraphicsTextEdit, self).mouseDoubleClickEvent(event)
        if self._edittrigger == GraphicsTextEdit.DoubleClicked:
            self._start()

    def focusOutEvent(self, event):
        super(GraphicsTextEdit, self).focusOutEvent(event)

        if self._editing:
            self._end()

    def _start(self):
        self._editing = True
        self.setTextInteractionFlags(Qt.TextEditorInteraction)
        self.setFocus(Qt.MouseFocusReason)
        self.editingStarted.emit()

    def _end(self):
        self._editing = False
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        self.editingFinished.emit()


class VennDiagram(QGraphicsWidget):
    # rect and petal are for future work
    Circle, Ellipse, Rect, Petal = 1, 2, 3, 4

    TitleFormat = "<center><h4>{0}</h4>{1}</center>"

    selectionChanged = Signal()
    itemTextEdited = Signal(int, str)

    def __init__(self, parent=None):
        super(VennDiagram, self).__init__(parent)
        self.shapeType = VennDiagram.Circle
        self._items = []
        self._vennareas = []
        self._textitems = []
        self._subsettextitems = []
        self._textanchors = []

    def item(self, index):
        return self._items[index]

    def items(self):
        return list(self._items)

    def count(self):
        return len(self._items)

    def setItems(self, items):
        if self._items:
            self.clear()

        self._items = list(items)

        for item in self._items:
            item.setParentItem(self)
            item.setVisible(True)

        fmt = self.TitleFormat.format

        font = self.font()
        font.setPixelSize(14)

        for item in items:
            text = GraphicsTextEdit(self)
            text.setFont(font)
            text.setDefaultTextColor(QColor("#333"))
            text.setHtml(fmt(escape(item.text), item.informativeText))
            text.adjustSize()
            text.editingStarted.connect(self._on_editingStarted)
            text.editingFinished.connect(self._on_editingFinished)
            text.documentSizeChanged.connect(
                self._on_itemTextSizeChanged
            )

            self._textitems.append(text)

        self._vennareas = [
            VennIntersectionArea(parent=self)
            for i in range(2 ** len(items))
        ]
        self._subsettextitems = [
            QGraphicsTextItem(parent=self)
            for i in range(2 ** len(items))
        ]

        self._updateLayout()

    def clear(self):
        scene = self.scene()
        items = self.vennareas() + list(self.items()) + self._textitems

        for item in self._textitems:
            item.editingStarted.disconnect(self._on_editingStarted)
            item.editingFinished.disconnect(self._on_editingFinished)
            item.documentSizeChanged.disconnect(
                self._on_itemTextSizeChanged
            )

        self._items = []
        self._vennareas = []
        self._textitems = []
        self._subsettextitems = []
        self._textanchors = []

        for item in items:
            item.setVisible(False)
            item.setParentItem(None)
            if scene is not None:
                scene.removeItem(item)

    def vennareas(self):
        return list(self._vennareas)

    def setFont(self, font):
        if font != self.font():
            self.prepareGeometryChange()
            super().setFont(font)

            for item in self.items():
                item.setFont(font)

    def _updateLayout(self):
        rect = self.geometry()
        n = len(self._items)
        if not n:
            return

        regions = venn_diagram(n)

        # The y axis in Qt points downward
        transform = QTransform().scale(1, -1)
        regions = list(map(transform.map, regions))

        union_brect = reduce(QRectF.united,
                             (path.boundingRect() for path in regions))

        scalex = rect.width() / union_brect.width()
        scaley = rect.height() / union_brect.height()
        scale = min(scalex, scaley)

        transform = QTransform().scale(scale, scale)

        regions = [transform.map(path) for path in regions]

        center = rect.width() / 2, rect.height() / 2
        for item, path in zip(self.items(), regions):
            item.setPath(path)
            item.setPos(*center)

        intersections = venn_intersections(regions)
        assert len(intersections) == 2 ** n
        assert len(self.vennareas()) == 2 ** n

        anchors = [(0, 0)] + subset_anchors(self._items)

        anchor_transform = QTransform().scale(rect.width(), -rect.height())
        for i, area in enumerate(self.vennareas()):
            area.setPath(intersections[setkey(i, n)])
            area.setPos(*center)
            x, y = anchors[i]
            anchor = anchor_transform.map(QPointF(x, y))
            area.setTextAnchor(anchor)
            area.setZValue(30)

        self._updateTextAnchors()

    def _updateTextAnchors(self):
        n = len(self._items)

        items = self._items
        dist = 15

        shape = reduce(QPainterPath.united, [item.path() for item in items])
        brect = shape.boundingRect()
        bradius = max(brect.width() / 2, brect.height() / 2)

        center = self.boundingRect().center()

        anchors = _category_anchors(items)
        self._textanchors = []
        for angle, anchor_h, anchor_v in anchors:
            line = QLineF.fromPolar(bradius, angle)
            ext = QLineF.fromPolar(dist, angle)
            line = QLineF(line.p1(), line.p2() + ext.p2())
            line = line.translated(center)

            anchor_pos = line.p2()
            self._textanchors.append((anchor_pos, anchor_h, anchor_v))

        for i in range(n):
            self._updateTextItemPos(i)

    def _updateTextItemPos(self, i):
        item = self._textitems[i]
        anchor_pos, anchor_h, anchor_v = self._textanchors[i]
        rect = item.boundingRect()
        pos = anchor_rect(rect, anchor_pos, anchor_h, anchor_v)
        item.setPos(pos)

    def setGeometry(self, geometry):
        super(VennDiagram, self).setGeometry(geometry)
        self._updateLayout()

    def _on_editingStarted(self):
        item = self.sender()
        index = self._textitems.index(item)
        text = self._items[index].text
        item.setTextWidth(-1)
        item.setHtml(self.TitleFormat.format(escape(text), "<br/>"))

    def _on_editingFinished(self):
        item = self.sender()
        index = self._textitems.index(item)
        text = item.toPlainText()
        if text != self._items[index].text:
            self._items[index].text = text

            self.itemTextEdited.emit(index, text)

        item.setHtml(
            self.TitleFormat.format(
                escape(text), self._items[index].informativeText))
        item.adjustSize()

    def _on_itemTextSizeChanged(self):
        item = self.sender()
        index = self._textitems.index(item)
        self._updateTextItemPos(index)


def anchor_rect(rect, anchor_pos,
                anchor_h=Qt.AnchorHorizontalCenter,
                anchor_v=Qt.AnchorVerticalCenter):

    if anchor_h == Qt.AnchorLeft:
        x = anchor_pos.x()
    elif anchor_h == Qt.AnchorHorizontalCenter:
        x = anchor_pos.x() - rect.width() / 2
    elif anchor_h == Qt.AnchorRight:
        x = anchor_pos.x() - rect.width()
    else:
        raise ValueError(anchor_h)

    if anchor_v == Qt.AnchorTop:
        y = anchor_pos.y()
    elif anchor_v == Qt.AnchorVerticalCenter:
        y = anchor_pos.y() - rect.height() / 2
    elif anchor_v == Qt.AnchorBottom:
        y = anchor_pos.y() - rect.height()
    else:
        raise ValueError(anchor_v)

    return QPointF(x, y)


def radians(angle):
    return 2 * math.pi * angle / 360


def unit_point(x, r=1.0):
    x = radians(x)
    return (r * math.cos(x), r * math.sin(x))


def _category_anchors(shapes):
    n = len(shapes)
    return _CATEGORY_ANCHORS[n - 1]


# (angle, horizontal anchor, vertical anchor)
_CATEGORY_ANCHORS = (
    # n == 1
    ((90, Qt.AnchorHorizontalCenter, Qt.AnchorBottom),),
    # n == 2
    ((180, Qt.AnchorRight, Qt.AnchorVerticalCenter),
     (0, Qt.AnchorLeft, Qt.AnchorVerticalCenter)),
    # n == 3
    ((150, Qt.AnchorRight, Qt.AnchorBottom),
     (30, Qt.AnchorLeft, Qt.AnchorBottom),
     (270, Qt.AnchorHorizontalCenter, Qt.AnchorTop)),
    # n == 4
    ((270 + 45, Qt.AnchorLeft, Qt.AnchorTop),
     (270 - 45, Qt.AnchorRight, Qt.AnchorTop),
     (90 - 15, Qt.AnchorLeft, Qt.AnchorBottom),
     (90 + 15, Qt.AnchorRight, Qt.AnchorBottom)),
    # n == 5
    ((90 - 5, Qt.AnchorHorizontalCenter, Qt.AnchorBottom),
     (18 - 5, Qt.AnchorLeft, Qt.AnchorVerticalCenter),
     (306 - 5, Qt.AnchorLeft, Qt.AnchorTop),
     (234 - 5, Qt.AnchorRight, Qt.AnchorTop),
     (162 - 5, Qt.AnchorRight, Qt.AnchorVerticalCenter),)
)


def subset_anchors(shapes):
    n = len(shapes)
    if n == 1:
        return [(0, 0)]
    elif n == 2:
        return [unit_point(180, r=1/3),
                unit_point(0, r=1/3),
                (0, 0)]
    elif n == 3:
        return [unit_point(150, r=0.35),  # A
                unit_point(30, r=0.35),   # B
                unit_point(90, r=0.27),   # AB
                unit_point(270, r=0.35),  # C
                unit_point(210, r=0.27),  # AC
                unit_point(330, r=0.27),  # BC
                unit_point(0, r=0),]      # ABC
    elif n == 4:
        anchors = [
            (0.400, 0.110),    # A
            (-0.400, 0.110),   # B
            (0.000, -0.285),   # AB
            (0.180, 0.330),    # C
            (0.265, 0.205),    # AC
            (-0.240, -0.110),  # BC
            (-0.100, -0.190),  # ABC
            (-0.180, 0.330),   # D
            (0.240, -0.110),   # AD
            (-0.265, 0.205),   # BD
            (0.100, -0.190),   # ABD
            (0.000, 0.250),    # CD
            (0.153, 0.090),    # ACD
            (-0.153, 0.090),   # BCD
            (0.000, -0.060),   # ABCD
        ]
        return anchors

    elif n == 5:
        anchors = [None] * 32
        # Base anchors
        A = (0.033, 0.385)
        AD = (0.095, 0.250)
        AE = (-0.100, 0.265)
        ACE = (-0.130, 0.220)
        ADE = (0.010, 0.225)
        ACDE = (-0.095, 0.175)
        ABCDE = (0.0, 0.0)

        anchors[-1] = ABCDE

        bases = [(0b00001, A),
                 (0b01001, AD),
                 (0b10001, AE),
                 (0b10101, ACE),
                 (0b11001, ADE),
                 (0b11101, ACDE)]

        for i in range(5):
            for index, anchor in bases:
                index = bit_rot_left(index, i, bits=5)
                assert anchors[index] is None
                anchors[index] = rotate_point(anchor, - 72 * i)

        assert all(anchors[1:])
        return anchors[1:]
    return None


def bit_rot_left(x, y, bits=32):
    mask = 2 ** bits - 1
    x_masked = x & mask
    return (x << y) & mask | (x_masked >> bits - y)


def rotate_point(p, angle):
    r = radians(angle)
    R = np.array([[math.cos(r), -math.sin(r)],
                  [math.sin(r), math.cos(r)]])
    x, y = np.dot(R, p)
    return (float(x), float(y))


def line_extended(line, distance):
    """
    Return an QLineF extended by `distance` units in the positive direction.
    """
    angle = line.angle() / 360 * 2 * math.pi
    dx, dy = unit_point(angle, r=distance)
    return QLineF(line.p1(), line.p2() + QPointF(dx, dy))


def circle_path(center, r=1.0):
    return ellipse_path(center, r, r, rotation=0)


def ellipse_path(center, a, b, rotation=0):
    if not isinstance(center, QPointF):
        center = QPointF(*center)

    brect = QRectF(-a, -b, 2 * a, 2 * b)

    path = QPainterPath()
    path.addEllipse(brect)

    if rotation != 0:
        transform = QTransform().rotate(rotation)
        path = transform.map(path)

    path.translate(center)
    return path


# TODO: Should include anchors for text layout (both inside and outside).
# for each item {path: QPainterPath,
#                text_anchors: [{center}] * (2 ** n)
#                mayor_axis: QLineF,
#                boundingRect QPolygonF (with 4 vertices)}
#
# Should be a new class with overloads for ellipse/circle, rect, and petal
# shapes, should store all constructor parameters, rotation, center,
# mayor/minor axis.


def venn_diagram(n):
    if n < 1 or n > 5:
        raise ValueError()

    paths = []

    if n == 1:
        paths = [circle_path(center=(0, 0), r=0.5)]
    elif n == 2:
        angles = [180, 0]
        paths = [circle_path(center=unit_point(x, r=1/6), r=1/3)
                 for x in angles]
    elif n == 3:
        angles = [150 - 120 * i for i in range(3)]
        paths = [circle_path(center=unit_point(x, r=1/6), r=1/3)
                 for x in angles]
    elif n == 4:
        # Constants shamelessly stolen from VennDiagram R package
        paths = [
            ellipse_path((0.65 - 0.5, 0.47 - 0.5), 0.35, 0.20, 45),
            ellipse_path((0.35 - 0.5, 0.47 - 0.5), 0.35, 0.20, 135),
            ellipse_path((0.5 - 0.5, 0.57 - 0.5), 0.35, 0.20, 45),
            ellipse_path((0.5 - 0.5, 0.57 - 0.5), 0.35, 0.20, 134),
        ]
    elif n == 5:
        # Constants shamelessly stolen from VennDiagram R package
        d = 0.13
        a, b = 0.24, 0.48
        a, b = b, a
        a, b = 0.48, 0.24
        paths = [ellipse_path(unit_point((1 - i) * 72, r=d),
                              a, b, rotation=90 - (i * 72))
                 for i in range(5)]

    return paths


def setkey(intval, n):
    return tuple(bool(intval & (2 ** i)) for i in range(n))


def keyrange(n):
    if n < 0:
        raise ValueError()

    for i in range(2 ** n):
        yield setkey(i, n)


def venn_intersections(paths):
    n = len(paths)
    return {key: venn_intersection(paths, key) for key in keyrange(n)}


def venn_intersection(paths, key):
    if not any(key):
        return QPainterPath()

    # first take the intersection of all included paths
    path = reduce(QPainterPath.intersected,
                  (path for path, included in zip(paths, key) if included))

    # subtract all the excluded sets (i.e. take the intersection
    # with the excluded set complements)
    path = reduce(QPainterPath.subtracted,
                  (path for path, included in zip(paths, key) if not included),
                  path)

    return path


def append_column(data, where, variable, column):
    X, Y, M = data.X, data.Y, data.metas
    domain = data.domain
    attr = domain.attributes
    class_vars = domain.class_vars
    metas = domain.metas
    if where == "X":
        attr = attr + (variable,)
        X = np.hstack((X, column))
    elif where == "Y":
        class_vars = class_vars + (variable,)
        Y = np.hstack((Y, column))
    elif where == "M":
        metas = metas + (variable,)
        M = np.hstack((M, column))
    else:
        raise ValueError
    domain = Domain(attr, class_vars, metas)
    new_data = data.transform(domain)
    new_data[:, variable] = column
    return new_data

def arrays_equal(a, b, type_):
    """
    checks if arrays have nans in same places and if not-nan elements
    are equal
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if type_ is not StringVariable:
        nana = np.isnan(a)
        nanb = np.isnan(b)
        return np.all(nana == nanb) and np.all(a[~nana] == b[~nanb])
    else:
        return np.all(a == b)

def pad_columns(values, mask, l):
    #inflates columns with nans
    a = np.full((l, 1), np.nan, dtype=values.dtype)
    a[mask] = values.reshape(-1, 1)
    return a

def get_perm(ids, all_ids):
    return [all_ids.index(el) for el in ids if el in all_ids]


def main():  # pragma: no cover
    # pylint: disable=import-outside-toplevel
    from Orange.evaluation import ShuffleSplit

    data = Table("brown-selected")

    if not "test_rows":  # change to `if not "test_rows" to test columns
        data = append_column(data, "M", StringVariable("Test"),
                             (np.arange(len(data)).reshape(-1, 1) % 30).astype(str))
        res = ShuffleSplit(n_resamples=5, test_size=0.7, stratified=False)
        indices = iter(res.get_indices(data))
        datasets = []
        for i in range(5):
            sample, _ = next(indices)
            data1 = data[sample]
            data1.name = chr(ord("A") + i)
            datasets.append((i, data1))
    else:
        domain = data.domain
        data1 = data.transform(Domain(domain.attributes[:15], domain.class_var))
        data2 = data.transform(Domain(domain.attributes[10:], domain.class_var))
        datasets = [(0, data1), (1, data2)]

    WidgetPreview(OWVennDiagram).run(insertData=datasets)


if __name__ == "__main__":  # pragma: no cover
    main()
