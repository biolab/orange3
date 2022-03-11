"""
Edit Domain
-----------

A widget for manual editing of a domain's attributes.

"""
import warnings
from xml.sax.saxutils import escape
from itertools import zip_longest, repeat, chain
from contextlib import contextmanager
from collections import namedtuple, Counter
from functools import singledispatch, partial
from typing import (
    Tuple, List, Any, Optional, Union, Dict, Sequence, Iterable, NamedTuple,
    FrozenSet, Type, Callable, TypeVar, Mapping, Hashable, cast
)

import numpy as np
import pandas as pd
from AnyQt.QtWidgets import (
    QWidget, QListView, QTreeView, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QAction, QActionGroup, QGroupBox,
    QStyledItemDelegate, QStyleOptionViewItem, QStyle, QSizePolicy,
    QDialogButtonBox, QPushButton, QCheckBox, QComboBox, QStackedLayout,
    QDialog, QRadioButton, QGridLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QAbstractItemView, QMenu, QToolTip
)
from AnyQt.QtGui import (
    QStandardItemModel, QStandardItem, QKeySequence, QIcon, QBrush, QPalette,
    QHelpEvent
)
from AnyQt.QtCore import (
    Qt, QSize, QModelIndex, QAbstractItemModel, QPersistentModelIndex, QRect,
    QPoint,
)
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

import Orange.data

from Orange.preprocess.transformation import Transformation, Identity, Lookup
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.buttons import FixedSizeButton
from Orange.widgets.utils.itemmodels import signal_blocking
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output

ndarray = np.ndarray  # pylint: disable=invalid-name
MArray = np.ma.MaskedArray
DType = Union[np.dtype, type]

A = TypeVar("A")  # pylint: disable=invalid-name
B = TypeVar("B")  # pylint: disable=invalid-name
V = TypeVar("V", bound=Orange.data.Variable)  # pylint: disable=invalid-name
H = TypeVar("H", bound=Hashable)  # pylint: disable=invalid-name


def unique(sequence: Iterable[H]) -> Iterable[H]:
    """
    Return unique elements in `sequence`, preserving their (first seen) order.
    """
    # depending on Python >= 3.6 'ordered' dict implementation detail.
    return iter(dict.fromkeys(sequence))


class _DataType:
    def __eq__(self, other):
        """Equal if `other` has the same type and all elements compare equal."""
        if type(self) is not type(other):
            return False
        return super().__eq__(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((type(self), super().__hash__()))

    def name_type(self):
        """
        Returns a tuple with name and type of the variable.
        It is used since it is forbidden to use names of variables in settings.
        """
        type_number = {
            "Categorical": 0,
            "Real": 2,
            "Time": 3,
            "String": 4
        }
        return self.name, type_number[type(self).__name__]


#: An ordered sequence of key, value pairs (variable annotations)
AnnotationsType = Tuple[Tuple[str, str], ...]


# Define abstract representation of the variable types edited

class Categorical(
    _DataType, NamedTuple("Categorical", [
        ("name", str),
        ("categories", Tuple[str, ...]),
        ("annotations", AnnotationsType),
        ("linked", bool)
    ])): pass


class Real(
    _DataType, NamedTuple("Real", [
        ("name", str),
        # a precision (int, and a format specifier('f', 'g', or '')
        ("format", Tuple[int, str]),
        ("annotations", AnnotationsType),
        ("linked", bool)
    ])): pass


class String(
    _DataType, NamedTuple("String", [
        ("name", str),
        ("annotations", AnnotationsType),
        ("linked", bool)
    ])): pass


class Time(
    _DataType, NamedTuple("Time", [
        ("name", str),
        ("annotations", AnnotationsType),
        ("linked", bool)
    ])): pass


Variable = Union[Categorical, Real, Time, String]
VariableTypes = (Categorical, Real, Time, String)


# Define variable transformations.

class Rename(_DataType, namedtuple("Rename", ["name"])):
    """
    Rename a variable.

    Parameters
    ----------
    name : str
        The new name
    """
    def __call__(self, var):
        # type: (Variable) -> Variable
        return var._replace(name=self.name)


#: Mapping of categories.
#: A list of pairs with the first element the original value and the second
#: element the new value. If the first element is None then a category level
#: is added. If the second element is None than the corresponding first level
#: is dropped. If there are duplicated elements on the right the corresponding
#: categories on the left are merged.
#: The mapped order is defined by the translated elements after None removal
#: and merges (the first occurrence of a multiplied elements defines its
#: position):
CategoriesMappingType = List[Tuple[Optional[str], Optional[str]]]


class CategoriesMapping(_DataType, namedtuple("CategoriesMapping", ["mapping"])):
    """
    Change categories of a categorical variable.

    Parameters
    ----------
    mapping : CategoriesMappingType
    """
    def __call__(self, var):
        # type: (Categorical) -> Categorical
        cat = tuple(unique(cj for _, cj in self.mapping if cj is not None))
        return var._replace(categories=cat)


class Annotate(_DataType, namedtuple("Annotate", ["annotations"])):
    """
    Replace variable annotations.
    """
    def __call__(self, var):
        return var._replace(annotations=self.annotations)


class Unlink(_DataType, namedtuple("Unlink", [])):
    """Unlink variable from its source, that is, remove compute_value"""


class StrpTime(_DataType, namedtuple("StrpTime", ["label", "formats", "have_date", "have_time"])):
    """Use format on variable interpreted as time"""


Transform = Union[Rename, CategoriesMapping, Annotate, Unlink, StrpTime]
TransformTypes = (Rename, CategoriesMapping, Annotate, Unlink, StrpTime)

CategoricalTransformTypes = (CategoriesMapping, Unlink)


# Reinterpret vector transformations.
class CategoricalVector(
    _DataType, NamedTuple("CategoricalVector", [
        ("vtype", Categorical),
        ("data", Callable[[], MArray]),
    ])): ...


class RealVector(
    _DataType, NamedTuple("RealVector", [
        ("vtype", Real),
        ("data", Callable[[], MArray]),
    ])): ...


class StringVector(
    _DataType, NamedTuple("StringVector", [
        ("vtype", String),
        ("data", Callable[[], MArray]),
    ])): ...


class TimeVector(
    _DataType, NamedTuple("TimeVector", [
        ("vtype", Time),
        ("data", Callable[[], MArray]),
    ])): ...


DataVector = Union[CategoricalVector, RealVector, StringVector, TimeVector]
DataVectorTypes = (CategoricalVector, RealVector, StringVector, TimeVector)


class AsString(_DataType, NamedTuple("AsString", [])):
    """Reinterpret a data vector as a string."""
    def __call__(self, vector: DataVector) -> StringVector:
        var, _ = vector
        if isinstance(var, String):
            return vector
        return StringVector(
            String(var.name, var.annotations, False),
            lambda: as_string(vector.data()),
        )


class AsContinuous(_DataType, NamedTuple("AsContinuous", [])):
    """
    Reinterpret as a continuous variable (values that do not parse as
    float are NaN).
    """
    def __call__(self, vector: DataVector) -> RealVector:
        var, _ = vector
        if isinstance(var, Real):
            return vector
        elif isinstance(var, Categorical):
            def data() -> MArray:
                d = vector.data()
                a = categorical_to_string_vector(d, var.values)
                return MArray(as_float_or_nan(a, where=a.mask), mask=a.mask)
            return RealVector(
                Real(var.name, (6, 'g'), var.annotations, var.linked), data
            )
        elif isinstance(var, Time):
            return RealVector(
                Real(var.name, (6, 'g'), var.annotations, var.linked),
                lambda: vector.data().astype(float)
            )
        elif isinstance(var, String):
            def data():
                s = vector.data()
                return MArray(as_float_or_nan(s, where=s.mask), mask=s.mask)
            return RealVector(
                Real(var.name, (6, "g"), var.annotations, var.linked), data
            )
        raise AssertionError


class AsCategorical(_DataType, namedtuple("AsCategorical", [])):
    """Reinterpret as a categorical variable"""
    def __call__(self, vector: DataVector) -> CategoricalVector:
        # this is the main complication in type transformation since we need
        # the data and not just the variable description
        var, _ = vector
        if isinstance(var, Categorical):
            return vector
        if isinstance(var, (Real, Time, String)):
            data, values = categorical_from_vector(vector.data())
            return CategoricalVector(
                Categorical(var.name, values, var.annotations, var.linked),
                lambda: data
            )
        raise AssertionError


class AsTime(_DataType, namedtuple("AsTime", [])):
    """Reinterpret as a datetime vector"""
    def __call__(self, vector: DataVector) -> TimeVector:
        var, _ = vector
        if isinstance(var, Time):
            return vector
        elif isinstance(var, Real):
            return TimeVector(
                Time(var.name, var.annotations, var.linked),
                lambda: vector.data().astype("M8[us]")
            )
        elif isinstance(var, Categorical):
            def data():
                d = vector.data()
                s = categorical_to_string_vector(d, var.values)
                dt = pd.to_datetime(s, errors="coerce").values.astype("M8[us]")
                return MArray(dt, mask=d.mask)
            return TimeVector(
                Time(var.name, var.annotations, var.linked), data
            )
        elif isinstance(var, String):
            def data():
                s = vector.data()
                dt = pd.to_datetime(s, errors="coerce").values.astype("M8[us]")
                return MArray(dt, mask=s.mask)
            return TimeVector(
                Time(var.name, var.annotations, var.linked), data
            )
        raise AssertionError


ReinterpretTransform = Union[AsCategorical, AsContinuous, AsTime, AsString]
ReinterpretTransformTypes = (AsCategorical, AsContinuous, AsTime, AsString)


def deconstruct(obj):
    # type: (tuple) -> Tuple[str, Tuple[Any, ...]]
    """
    Deconstruct a tuple subclass to its class name and its contents.

    Parameters
    ----------
    obj : A tuple

    Returns
    -------
    value: Tuple[str, Tuple[Any, ...]]
    """
    cname = type(obj).__name__
    args = tuple(obj)
    return cname, args


def reconstruct(tname, args):
    # type: (str, Tuple[Any, ...]) -> Tuple[Any, ...]
    """
    Reconstruct a tuple subclass (inverse of deconstruct).

    Parameters
    ----------
    tname : str
        Type name
    args : Tuple[Any, ...]

    Returns
    -------
    rval: Tuple[Any, ...]
    """
    try:
        constructor = globals()[tname]
    except KeyError:
        raise NameError(tname)
    return constructor(*args)


def formatter_for_dtype(dtype: np.dtype) -> Callable[[Any], str]:
    if dtype.metadata is None:
        return str
    else:
        return dtype.metadata.get("__formatter", str)  # metadata abuse


def masked_unique(data: MArray) -> Tuple[MArray, ndarray]:
    if not np.any(data.mask):
        return np.ma.unique(data, return_inverse=True)
    elif data.dtype.kind == "O":
        # np.ma.unique does not work for object arrays
        # (no ma.minimum_fill_value for object arrays)
        # maybe sorted(set(data.data[...]))
        unq = np.unique(data.data[~data.mask])
        mapper = make_dict_mapper(
            DictMissingConst(len(unq), ((v, i) for i, v in enumerate(unq)))
        )
        index = mapper(data.data)
        unq = np.array(unq.tolist() + [data.fill_value], dtype=data.dtype)
        unq_mask = [False] * unq.size
        unq_mask[-1] = True
        unq = MArray(unq, mask=unq_mask)
        return unq, index
    else:
        unq, index = np.ma.unique(data, return_inverse=True)
        assert not np.any(unq.mask[:-1]), \
            "masked value if present must be in last position"
        return unq, index


def categorical_from_vector(data: MArray) -> Tuple[MArray, Tuple[str, ...]]:
    formatter = formatter_for_dtype(data.dtype)
    unq, index = categorize_unique(data)
    if formatter is not str:
        # str(np.array([0], "M8[s]")[0]) is different then
        # str(np.array([0], "M8[s]").astype(object)[0]) which is what
        # as_string is doing
        names = tuple(map(formatter, unq.astype(object)))
    else:
        names = tuple(as_string(unq))
    data = MArray(
        index, mask=data.mask,
        dtype=np.dtype(int, metadata={
            "__formater": lambda i: names[i] if 0 <= i < unq.size else "?"
        })
    )
    return data, names


def categorize_unique(data: MArray) -> Tuple[ndarray, MArray]:
    unq, index = masked_unique(data)
    if np.any(unq.mask):
        unq = unq[:-1]
        assert not np.any(unq.mask), "masked value if present must be last"
    unq = unq.data
    index[data.mask] = -1
    index = MArray(index, mask=data.mask)
    return unq, index


def categorical_to_string_vector(data: MArray, values: Tuple[str, ...]) -> MArray:
    lookup = np.asarray(values, object)
    out = np.full(data.shape, "", dtype=object)
    mask_ = ~data.mask
    out[mask_] = lookup[data.data[mask_]]
    return MArray(out, mask=data.mask, fill_value="")


# Item models


class DictItemsModel(QStandardItemModel):
    """A Qt Item Model class displaying the contents of a python
    dictionary.

    """
    # Implement a proper model with in-place editing.
    # (Maybe it should be a TableModel with 2 columns)
    def __init__(self, parent=None, a_dict=None):
        super().__init__(parent)
        self._dict = {}
        self.setHorizontalHeaderLabels(["Key", "Value"])
        if a_dict is not None:
            self.set_dict(a_dict)

    def set_dict(self, a_dict):
        # type: (Dict[str, str]) -> None
        self._dict = a_dict
        self.setRowCount(0)
        for key, value in sorted(a_dict.items()):
            key_item = QStandardItem(key)
            value_item = QStandardItem(value)
            key_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            value_item.setFlags(value_item.flags() | Qt.ItemIsEditable)
            self.appendRow([key_item, value_item])

    def get_dict(self):
        # type: () -> Dict[str, str]
        rval = {}
        for row in range(self.rowCount()):
            key_item = self.item(row, 0)
            value_item = self.item(row, 1)
            rval[key_item.text()] = value_item.text()
        return rval


class VariableEditor(QWidget):
    """
    An editor widget for a variable.

    Can edit the variable name, and its attributes dictionary.
    """
    variable_changed = Signal()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.var = None  # type: Optional[Variable]

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.form = form = QFormLayout(
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
            objectName="editor-form-layout"
        )
        layout.addLayout(self.form)

        self.name_edit = QLineEdit(objectName="name-editor")
        self.name_edit.editingFinished.connect(
            lambda: self.name_edit.isModified() and self.on_name_changed()
        )
        form.addRow("Name:", self.name_edit)

        self.unlink_var_cb = QCheckBox(
            "Unlink variable from its source variable", self,
            toolTip="Make Orange forget that the variable is derived from "
                    "another.\n"
                    "Use this for instance when you want to consider variables "
                    "with the same name but from different sources as the same "
                    "variable."
        )
        self.unlink_var_cb.toggled.connect(self._set_unlink)
        form.addRow("", self.unlink_var_cb)

        vlayout = QVBoxLayout(margin=0, spacing=1)
        self.labels_edit = view = QTreeView(
            objectName="annotation-pairs-edit",
            rootIsDecorated=False,
            editTriggers=QTreeView.DoubleClicked | QTreeView.EditKeyPressed,
        )
        self.labels_model = model = DictItemsModel()
        view.setModel(model)

        view.selectionModel().selectionChanged.connect(
            self.on_label_selection_changed)

        agrp = QActionGroup(view, objectName="annotate-action-group")
        action_add = QAction(
            "+", self, objectName="action-add-label",
            toolTip="Add a new label.",
            shortcut=QKeySequence(QKeySequence.New),
            shortcutContext=Qt.WidgetShortcut
        )
        action_delete = QAction(
            "\N{MINUS SIGN}", self, objectName="action-delete-label",
            toolTip="Remove selected label.",
            shortcut=QKeySequence(QKeySequence.Delete),
            shortcutContext=Qt.WidgetShortcut
        )
        agrp.addAction(action_add)
        agrp.addAction(action_delete)
        view.addActions([action_add, action_delete])

        def add_label():
            row = [QStandardItem(), QStandardItem()]
            model.appendRow(row)
            idx = model.index(model.rowCount() - 1, 0)
            view.setCurrentIndex(idx)
            view.edit(idx)

        def remove_label():
            rows = view.selectionModel().selectedRows(0)
            if rows:
                assert len(rows) == 1
                idx = rows[0].row()
                model.removeRow(idx)

        action_add.triggered.connect(add_label)
        action_delete.triggered.connect(remove_label)
        agrp.setEnabled(False)

        self.add_label_action = action_add
        self.remove_label_action = action_delete

        # Necessary signals to know when the labels change
        model.dataChanged.connect(self.on_labels_changed)
        model.rowsInserted.connect(self.on_labels_changed)
        model.rowsRemoved.connect(self.on_labels_changed)

        vlayout.addWidget(self.labels_edit)
        hlayout = QHBoxLayout()
        hlayout.setContentsMargins(0, 0, 0, 0)
        button = FixedSizeButton(
            self, defaultAction=self.add_label_action,
            accessibleName="Add",
        )
        hlayout.addWidget(button)

        button = FixedSizeButton(
            self, defaultAction=self.remove_label_action,
            accessibleName="Remove",
        )

        hlayout.addWidget(button)
        hlayout.addStretch(10)
        vlayout.addLayout(hlayout)
        form.addRow("Labels:", vlayout)

    def set_data(self, var, transform=()):
        # type: (Optional[Variable], Sequence[Transform]) -> None
        """
        Set the variable to edit.
        """
        self.clear()
        self.var = var
        if var is not None:
            name = var.name
            annotations = var.annotations
            unlink = False
            for tr in transform:
                if isinstance(tr, Rename):
                    name = tr.name
                elif isinstance(tr, Annotate):
                    annotations = tr.annotations
                elif isinstance(tr, Unlink):
                    unlink = True
            self.name_edit.setText(name)
            self.labels_model.set_dict(dict(annotations))
            self.add_label_action.actionGroup().setEnabled(True)
            self.unlink_var_cb.setChecked(unlink)
        else:
            self.add_label_action.actionGroup().setEnabled(False)

        self.unlink_var_cb.setDisabled(var is None or not var.linked)

    def get_data(self):
        """Retrieve the modified variable.
        """
        if self.var is None:
            return None, []
        name = self.name_edit.text().strip()
        labels = tuple(sorted(self.labels_model.get_dict().items()))
        tr = []
        if self.var.name != name:
            tr.append(Rename(name))
        if self.var.annotations != labels:
            tr.append(Annotate(labels))
        if self.var.linked and self.unlink_var_cb.isChecked():
            tr.append(Unlink())
        return self.var, tr

    def clear(self):
        """Clear the editor state.
        """
        self.var = None
        self.name_edit.setText("")
        self.labels_model.setRowCount(0)
        self.unlink_var_cb.setChecked(False)

    @Slot()
    def on_name_changed(self):
        self.variable_changed.emit()

    @Slot()
    def on_labels_changed(self):
        self.variable_changed.emit()

    @Slot()
    def on_label_selection_changed(self):
        selected = self.labels_edit.selectionModel().selectedRows()
        self.remove_label_action.setEnabled(bool(len(selected)))

    def _set_unlink(self, unlink):
        self.unlink_var_cb.setChecked(unlink)
        self.variable_changed.emit()


class GroupItemsDialog(QDialog):
    """
    A dialog for group less frequent values.
    """
    DEFAULT_LABEL = "other"

    def __init__(
            self, variable: Categorical,
            data: Union[np.ndarray, List, MArray],
            selected_attributes: List[str], dialog_settings: Dict[str, Any],
            parent: QWidget = None, flags: Qt.WindowFlags = Qt.Dialog, **kwargs
    ) -> None:
        super().__init__(parent, flags, **kwargs)
        self.variable = variable
        self.data = data
        self.selected_attributes = selected_attributes

        # grouping strategy
        self.selected_radio = radio1 = QRadioButton("Group selected values")
        self.frequent_abs_radio = radio2 = QRadioButton(
            "Group values with less than"
        )
        self.frequent_rel_radio = radio3 = QRadioButton(
            "Group values with less than"
        )
        self.n_values_radio = radio4 = QRadioButton(
            "Group all except"
        )

        # if selected attributes available check the first radio button,
        # otherwise disable it
        if selected_attributes:
            radio1.setChecked(True)
        else:
            radio1.setEnabled(False)
            # they are remembered by number since radio button instance is
            # new object for each dialog
            checked = dialog_settings.get("selected_radio", 0)
            [radio2, radio3, radio4][checked].setChecked(True)

        label2 = QLabel("occurrences")
        label3 = QLabel("occurrences")
        label4 = QLabel("most frequent values")

        self.frequent_abs_spin = spin2 = QSpinBox()
        max_val = len(data)
        spin2.setMinimum(1)
        spin2.setMaximum(max_val)
        spin2.setValue(dialog_settings.get("frequent_abs_spin", 10))
        spin2.setMinimumWidth(
            self.fontMetrics().horizontalAdvance("X") * (len(str(max_val)) + 1) + 20
        )
        spin2.valueChanged.connect(self._frequent_abs_spin_changed)

        self.frequent_rel_spin = spin3 = QDoubleSpinBox()
        spin3.setMinimum(0)
        spin3.setDecimals(1)
        spin3.setSingleStep(0.1)
        spin3.setMaximum(100)
        spin3.setValue(dialog_settings.get("frequent_rel_spin", 10))
        spin3.setMinimumWidth(self.fontMetrics().horizontalAdvance("X") * (2 + 1) + 20)
        spin3.setSuffix(" %")
        spin3.valueChanged.connect(self._frequent_rel_spin_changed)

        self.n_values_spin = spin4 = QSpinBox()
        spin4.setMinimum(0)
        spin4.setMaximum(len(variable.categories))
        spin4.setValue(
            dialog_settings.get(
                "n_values_spin", min(10, len(variable.categories))
            )
        )
        spin4.setMinimumWidth(
            self.fontMetrics().horizontalAdvance("X") * (len(str(max_val)) + 1) + 20
        )
        spin4.valueChanged.connect(self._n_values_spin_spin_changed)

        grid_layout = QGridLayout()
        # first row
        grid_layout.addWidget(radio1, 0, 0, 1, 2)
        # second row
        grid_layout.addWidget(radio2, 1, 0, 1, 2)
        grid_layout.addWidget(spin2, 1, 2)
        grid_layout.addWidget(label2, 1, 3)
        # third row
        grid_layout.addWidget(radio3, 2, 0, 1, 2)
        grid_layout.addWidget(spin3, 2, 2)
        grid_layout.addWidget(label3, 2, 3)
        # fourth row
        grid_layout.addWidget(radio4, 3, 0)
        grid_layout.addWidget(spin4, 3, 1)
        grid_layout.addWidget(label4, 3, 2, 1, 2)

        group_box = QGroupBox()
        group_box.setLayout(grid_layout)

        # grouped variable name
        new_name_label = QLabel("New value name: ")
        self.new_name_line_edit = n_line_edit = QLineEdit(
            dialog_settings.get("name_line_edit", self.DEFAULT_LABEL)
        )
        # it is shown gray when user removes the text and let user know that
        # word others is default one
        n_line_edit.setPlaceholderText(self.DEFAULT_LABEL)
        name_hlayout = QHBoxLayout()
        name_hlayout.addWidget(new_name_label)
        name_hlayout.addWidget(n_line_edit)

        # confirm_button = QPushButton("Apply")
        # cancel_button = QPushButton("Cancel")
        buttons = QDialogButtonBox(
            orientation=Qt.Horizontal,
            standardButtons=(QDialogButtonBox.Ok | QDialogButtonBox.Cancel),
            objectName="dialog-button-box",
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # join components
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(group_box)
        self.layout().addLayout(name_hlayout)
        self.layout().addWidget(buttons)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _frequent_abs_spin_changed(self) -> None:
        self.frequent_abs_radio.setChecked(True)

    def _n_values_spin_spin_changed(self) -> None:
        self.n_values_radio.setChecked(True)

    def _frequent_rel_spin_changed(self) -> None:
        self.frequent_rel_radio.setChecked(True)

    def get_merge_attributes(self) -> List[str]:
        """
        Returns attributes that will be merged

        Returns
        -------
        List of attributes' to be merged names
        """
        if self.selected_radio.isChecked():
            return self.selected_attributes

        if isinstance(self.data, MArray):
            non_nan = self.data[~self.data.mask]
        elif isinstance(self.data, np.ndarray):
            non_nan = self.data[~np.isnan(self.data)]
        else:  # list
            non_nan = [x for x in self.data if x is not None]

        counts = Counter(non_nan)
        if self.n_values_radio.isChecked():
            keep_values = self.n_values_spin.value()
            values = counts.most_common()[keep_values:]
            indices = [i for i, _ in values]
        elif self.frequent_abs_radio.isChecked():
            indices = [v for v, c in counts.most_common()
                       if c < self.frequent_abs_spin.value()]
        else:  # self.frequent_rel_radio.isChecked():
            n_all = sum(counts.values())
            indices = [v for v, c in counts.most_common()
                       if c / n_all * 100 < self.frequent_rel_spin.value()]

        indices = np.array(indices, dtype=int)  # indices must be ints
        return np.array(self.variable.categories)[indices].tolist()

    def get_merged_value_name(self) -> str:
        """
        Returns
        -------
        New label of merged values
        """
        return self.new_name_line_edit.text() or self.DEFAULT_LABEL

    def get_dialog_settings(self) -> Dict[str, Any]:
        """
        Returns
        -------
        Return the dictionary with vlues set by user in each of the line edits
        and selected radio button.
        """
        settings_dict = {
            "frequent_abs_spin": self.frequent_abs_spin.value(),
            "frequent_rel_spin": self.frequent_rel_spin.value(),
            "n_values_spin": self.n_values_spin.value(),
            "name_line_edit": self.new_name_line_edit.text()
        }
        checked = [
            i for i, s in enumerate(
                [self.frequent_abs_radio,
                 self.frequent_rel_radio,
                 self.n_values_radio]
            ) if s.isChecked()]
        # when checked empty radio button for selected values is selected
        # it is not stored in setting since its selection depends on users
        # selection of values in list
        if checked:
            settings_dict["selected_radio"] = checked[0]
        return settings_dict


@contextmanager
def disconnected(signal, slot, connection_type=Qt.AutoConnection):
    signal.disconnect(slot)
    try:
        yield
    finally:
        signal.connect(slot, connection_type)


#: In 'reordable' models holds the original position of the item
#: (if applicable).
SourcePosRole = Qt.UserRole
#: The original name
SourceNameRole = Qt.UserRole + 2

#: The added/dropped state (type is ItemEditState)
EditStateRole = Qt.UserRole + 1


class ItemEditState:
    NoState = 0
    Dropped = 1
    Added = 2


#: Role used to retrieve the count of 'key' values in the model.
MultiplicityRole = Qt.UserRole + 0x67


class CountedListModel(itemmodels.PyListModel):
    """
    A list model counting how many times unique `key` values appear in
    the list.

    The counts are cached and invalidated on any change to the model involving
    the changes to `keyRoles`.
    """
    #: cached counts
    __counts_cache = None  # type: Optional[Counter]

    def data(self, index, role=Qt.DisplayRole):
        # type: (QModelIndex, int) -> Any
        if role == MultiplicityRole:
            key = self.key(index)
            counts = self.__counts()
            return counts.get(key, 1)
        return super().data(index, role)

    def setData(self, index, value, role=Qt.EditRole):
        # type: (QModelIndex, Any, int)-> bool
        rval = super().setData(index, value, role)
        if role in self.keyRoles():
            self.invalidateCounts()
        return rval

    def setItemData(self, index, data):
        # type: (QModelIndex, Dict[int, Any]) -> bool
        rval = super().setItemData(index, data)
        if self.keyRoles().intersection(set(data.keys())):
            self.invalidateCounts()
        return rval

    def endInsertRows(self):
        super().endInsertRows()
        self.invalidateCounts()

    def endRemoveRows(self):
        super().endRemoveRows()
        self.invalidateCounts()

    def endResetModel(self) -> None:
        super().endResetModel()
        self.invalidateCounts()

    def invalidateCounts(self) -> None:
        """
        Invalidate the cached counts.
        """
        self.__counts_cache = None
        # emit the change for the whole model
        self.dataChanged.emit(
            self.index(0), self.index(self.rowCount() - 1), [MultiplicityRole]
        )

    def __counts(self):
        # type: () -> Counter
        if self.__counts_cache is not None:
            return self.__counts_cache
        counts = Counter()
        for index in map(self.index, range(self.rowCount())):
            key = self.key(index)
            try:
                counts[key] += 1
            except TypeError:  # pragma: no cover
                warnings.warn(f"key value '{key}' is not hashable")
        self.__counts_cache = counts
        return self.__counts_cache

    def key(self, index):
        # type: (QModelIndex) -> Any
        """
        Return the 'key' value that is to be counted.

        The default implementation returns Qt.EditRole value for the index

        Parameters
        ----------
        index : QModelIndex
            The model index.

        Returns
        -------
        key : Any
        """
        return self.data(index, Qt.EditRole)

    def keyRoles(self):
        # type: () -> FrozenSet[int]
        """
        Return a set of item roles on which `key` depends.

        The counts are invalidated and recomputed whenever any of the roles in
        this set changes.

        By default the only role returned is Qt.EditRole
        """
        return frozenset({Qt.EditRole})


class CountedStateModel(CountedListModel):
    """
    Count by EditRole (name) and EditStateRole (ItemEditState)
    """
    # The purpose is to count the items with target name only for
    # ItemEditState.NoRole, i.e. excluding added/dropped values.
    #
    def key(self, index):  # type: (QModelIndex) -> Tuple[Any, Any]
        # reimplemented
        return self.data(index, Qt.EditRole), self.data(index, EditStateRole)

    def keyRoles(self):  # type: () -> FrozenSet[int]
        # reimplemented
        return frozenset({Qt.EditRole, EditStateRole})


def mapRectTo(widget: QWidget, parent: QWidget, rect: QRect) -> QRect:  # pylint: disable=redefined-outer-name
    return QRect(widget.mapTo(parent, rect.topLeft()), rect.size())


def mapRectToGlobal(widget: QWidget, rect: QRect) -> QRect:  # pylint: disable=redefined-outer-name
    return QRect(widget.mapToGlobal(rect.topLeft()), rect.size())


class CategoriesEditDelegate(QStyledItemDelegate):
    """
    Display delegate for editing categories.

    Displayed items are styled for add, remove, merge and rename operations.
    """
    def initStyleOption(self, option, index):
        # type: (QStyleOptionViewItem, QModelIndex)-> None
        super().initStyleOption(option, index)
        text = str(index.data(Qt.EditRole))
        sourcename = str(index.data(SourceNameRole))
        editstate = index.data(EditStateRole)
        counts = index.data(MultiplicityRole)
        if not isinstance(counts, int):
            counts = 1
        suffix = None
        if editstate == ItemEditState.Dropped:
            option.state &= ~QStyle.State_Enabled
            option.font.setStrikeOut(True)
            text = sourcename
            suffix = "(dropped)"
        elif editstate == ItemEditState.Added:
            suffix = "(added)"
        else:
            text = f"{sourcename} \N{RIGHTWARDS ARROW} {text}"
            if counts > 1:
                suffix = "(merged)"
        if suffix is not None:
            text = text + " " + suffix
        option.text = text

    class CatEditComboBox(QComboBox):
        prows: List[QPersistentModelIndex]

    def createEditor(
            self, parent: QWidget, option: 'QStyleOptionViewItem',
            index: QModelIndex
    ) -> QWidget:
        view = option.widget
        assert isinstance(view, QAbstractItemView)
        selmodel = view.selectionModel()
        rows = selmodel.selectedRows(0)
        if len(rows) < 2:
            return super().createEditor(parent, option, index)
        # edit multiple selection
        cb = CategoriesEditDelegate.CatEditComboBox(
            editable=True, insertPolicy=QComboBox.InsertAtBottom)
        cb.setParent(view, Qt.Popup)
        cb.addItems(
            list(unique(str(row.data(Qt.EditRole)) for row in rows)))
        prows = [QPersistentModelIndex(row) for row in rows]
        cb.prows = prows
        return cb

    def updateEditorGeometry(
            self, editor: QWidget, option: 'QStyleOptionViewItem',
            index: QModelIndex
    ) -> None:
        if isinstance(editor, CategoriesEditDelegate.CatEditComboBox):
            view = cast(QAbstractItemView, option.widget)
            view.scrollTo(index)
            vport = view.viewport()
            vrect = view.visualRect(index)
            vrect = mapRectTo(vport, view, vrect)
            vrect = vrect.intersected(vport.geometry())
            vrect = mapRectToGlobal(vport, vrect)
            size = editor.sizeHint().expandedTo(vrect.size())
            editor.resize(size)
            editor.move(vrect.topLeft())
        else:
            super().updateEditorGeometry(editor, option, index)

    def setModelData(
            self, editor: QWidget, model: QAbstractItemModel, index: QModelIndex
    ) -> None:
        if isinstance(editor, CategoriesEditDelegate.CatEditComboBox):
            text = editor.currentText()
            with signal_blocking(model):
                for prow in editor.prows:
                    if prow.isValid():
                        model.setData(QModelIndex(prow), text, Qt.EditRole)
            # this could be better
            model.dataChanged.emit(
                model.index(0, 0), model.index(model.rowCount() - 1, 0),
                (Qt.EditRole,)
            )
        else:
            super().setModelData(editor, model, index)


class DiscreteVariableEditor(VariableEditor):
    """An editor widget for editing a discrete variable.

    Extends the :class:`VariableEditor` to enable editing of
    variables values.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.merge_dialog_settings = {}
        self._values = None

        form = self.layout().itemAt(0)
        assert isinstance(form, QFormLayout)
        #: A list model of discrete variable's values.
        self.values_model = CountedStateModel(
            flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        )

        vlayout = QVBoxLayout(spacing=1, margin=0)
        self.values_edit = QListView(
            editTriggers=QListView.DoubleClicked | QListView.EditKeyPressed,
            selectionMode=QListView.ExtendedSelection,
            uniformItemSizes=True,
        )
        self.values_edit.setItemDelegate(CategoriesEditDelegate(self))
        self.values_edit.setModel(self.values_model)
        self.values_model.dataChanged.connect(self.on_values_changed)

        self.values_edit.selectionModel().selectionChanged.connect(
            self.on_value_selection_changed)
        self.values_model.layoutChanged.connect(self.on_value_selection_changed)
        self.values_model.rowsMoved.connect(self.on_value_selection_changed)

        vlayout.addWidget(self.values_edit)
        hlayout = QHBoxLayout(spacing=1, margin=0)

        self.categories_action_group = group = QActionGroup(
            self, objectName="action-group-categories", enabled=False
        )
        self.move_value_up = QAction(
            "Move up", group,
            iconText="\N{UPWARDS ARROW}",
            toolTip="Move the selected item up.",
            shortcut=QKeySequence(Qt.ControlModifier | Qt.AltModifier |
                                  Qt.Key_BracketLeft),
            shortcutContext=Qt.WidgetShortcut,
        )
        self.move_value_up.triggered.connect(self.move_up)

        self.move_value_down = QAction(
            "Move down", group,
            iconText="\N{DOWNWARDS ARROW}",
            toolTip="Move the selected item down.",
            shortcut=QKeySequence(Qt.ControlModifier | Qt.AltModifier |
                                  Qt.Key_BracketRight),
            shortcutContext=Qt.WidgetShortcut,
        )
        self.move_value_down.triggered.connect(self.move_down)

        self.add_new_item = QAction(
            "Add", group,
            iconText="+",
            objectName="action-add-item",
            toolTip="Append a new item.",
            shortcut=QKeySequence(QKeySequence.New),
            shortcutContext=Qt.WidgetShortcut,
        )
        self.remove_item = QAction(
            "Remove item", group,
            iconText="\N{MINUS SIGN}",
            objectName="action-remove-item",
            toolTip="Delete the selected item.",
            shortcut=QKeySequence(QKeySequence.Delete),
            shortcutContext=Qt.WidgetShortcut,
        )
        self.rename_selected_items = QAction(
            "Rename selected items", group,
            iconText="=",
            objectName="action-rename-selected-items",
            toolTip="Rename selected items.",
            shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_Equal),
            shortcutContext=Qt.WidgetShortcut,
        )
        self.merge_items = QAction(
            "Merge", group,
            iconText="M",
            objectName="action-activate-merge-dialog",
            toolTip="Merge infrequent items.",
            shortcut=QKeySequence(Qt.ControlModifier | Qt.MetaModifier | Qt.Key_Equal),
            shortcutContext=Qt.WidgetShortcut
        )

        self.add_new_item.triggered.connect(self._add_category)
        self.remove_item.triggered.connect(self._remove_category)
        self.rename_selected_items.triggered.connect(self._rename_selected_categories)
        self.merge_items.triggered.connect(self._merge_categories)

        button1 = FixedSizeButton(
            self, defaultAction=self.move_value_up,
            accessibleName="Move up"
        )
        button2 = FixedSizeButton(
            self, defaultAction=self.move_value_down,
            accessibleName="Move down"
        )
        button3 = FixedSizeButton(
            self, defaultAction=self.add_new_item,
            accessibleName="Add"
        )
        button4 = FixedSizeButton(
            self, defaultAction=self.remove_item,
            accessibleName="Remove"
        )
        button5 = FixedSizeButton(
            self, defaultAction=self.rename_selected_items,
            accessibleName="Merge selected items"
        )
        button6 = FixedSizeButton(
            self, defaultAction=self.merge_items,
            accessibleName="Merge infrequent",
        )

        self.values_edit.addActions([
            self.move_value_up, self.move_value_down,
            self.add_new_item, self.remove_item, self.rename_selected_items
        ])
        self.values_edit.setContextMenuPolicy(Qt.CustomContextMenu)

        def context_menu(pos: QPoint):
            viewport = self.values_edit.viewport()
            menu = QMenu(self.values_edit)
            menu.setAttribute(Qt.WA_DeleteOnClose)
            menu.addActions([self.rename_selected_items, self.remove_item])
            menu.popup(viewport.mapToGlobal(pos))
        self.values_edit.customContextMenuRequested.connect(context_menu)

        hlayout.addWidget(button1)
        hlayout.addWidget(button2)
        hlayout.addSpacing(3)
        hlayout.addWidget(button3)
        hlayout.addWidget(button4)
        hlayout.addSpacing(3)
        hlayout.addWidget(button5)
        hlayout.addWidget(button6)

        hlayout.addStretch(10)
        vlayout.addLayout(hlayout)

        form.insertRow(2, "Values:", vlayout)

        QWidget.setTabOrder(self.name_edit, self.values_edit)
        QWidget.setTabOrder(self.values_edit, button1)
        QWidget.setTabOrder(button1, button2)
        QWidget.setTabOrder(button2, button3)
        QWidget.setTabOrder(button3, button4)
        QWidget.setTabOrder(button4, button5)
        QWidget.setTabOrder(button5, button6)

    def set_data(self, var, transform=()):
        raise NotImplementedError

    def set_data_categorical(self, var, values, transform=()):
        # type: (Optional[Categorical], Optional[Sequence[float]], Sequence[Transform]) -> None
        """
        Set the variable to edit.

        `values` is needed for categorical features to perform grouping.
        """
        # pylint: disable=too-many-branches
        super().set_data(var, transform=transform)
        self._values = values
        tr = None  # type: Optional[CategoriesMapping]
        for tr_ in transform:
            if isinstance(tr_, CategoriesMapping):
                tr = tr_

        items = []
        if tr is not None:
            ci_index = {c: i for i, c in enumerate(var.categories)}
            for ci, cj in tr.mapping:
                if ci is None and cj is not None:
                    # level added
                    item = {
                        Qt.EditRole: cj,
                        EditStateRole: ItemEditState.Added,
                        SourcePosRole: None
                    }
                elif ci is not None and cj is None:
                    # ci level dropped
                    item = {
                        Qt.EditRole: ci,
                        EditStateRole: ItemEditState.Dropped,
                        SourcePosRole: ci_index[ci],
                        SourceNameRole: ci
                    }
                elif ci is not None and cj is not None:
                    # rename or reorder
                    item = {
                        Qt.EditRole: cj,
                        EditStateRole: ItemEditState.NoState,
                        SourcePosRole: ci_index[ci],
                        SourceNameRole: ci
                    }
                else:
                    assert False, "invalid mapping: {!r}".format(tr.mapping)
                items.append(item)
        elif var is not None:
            items = [
                {Qt.EditRole: c,
                 EditStateRole: ItemEditState.NoState,
                 SourcePosRole: i,
                 SourceNameRole: c}
                for i, c in enumerate(var.categories)
            ]
        else:
            items = []

        with disconnected(self.values_model.dataChanged,
                          self.on_values_changed):
            self.values_model.clear()
            self.values_model.insertRows(0, len(items))
            for i, item in enumerate(items):
                self.values_model.setItemData(
                    self.values_model.index(i, 0),
                    item
                )

        self.add_new_item.actionGroup().setEnabled(var is not None)

    def __categories_mapping(self):
        # type: () -> CategoriesMappingType
        """
        Encode and return the current state as a CategoriesMappingType
        """
        model = self.values_model
        source = self.var.categories

        res = []  # type: CategoriesMappingType
        for i in range(model.rowCount()):
            midx = model.index(i, 0)
            category = midx.data(Qt.EditRole)
            source_pos = midx.data(SourcePosRole)  # type: Optional[int]
            if source_pos is not None:
                source_name = source[source_pos]
            else:
                source_name = None
            state = midx.data(EditStateRole)
            if state == ItemEditState.Dropped:
                res.append((source_name, None))
            elif state == ItemEditState.Added:
                res.append((None, category))
            else:
                res.append((source_name, category))
        return res

    def get_data(self):
        """Retrieve the modified variable
        """
        var, tr = super().get_data()
        if var is None:
            return var, tr
        mapping = self.__categories_mapping()
        assert len(mapping) >= len(var.categories), f'{mapping}, {var}'
        if any(_1 != _2 or _2 != _3
               for (_1, _2), _3 in zip_longest(mapping, var.categories)):
            tr.append(CategoriesMapping(mapping))
        return var, tr

    def clear(self):
        """Clear the model state.
        """
        super().clear()
        self.values_model.clear()

    def move_rows(self, rows, offset):
        if not rows:
            return
        assert len(rows) == 1
        i = rows[0].row()
        if offset > 0:
            offset += 1
        self.values_model.moveRows(QModelIndex(), i, 1, QModelIndex(), i + offset)
        self.variable_changed.emit()

    def move_up(self):
        rows = self.values_edit.selectionModel().selectedRows()
        self.move_rows(rows, -1)

    def move_down(self):
        rows = self.values_edit.selectionModel().selectedRows()
        self.move_rows(rows, 1)

    @Slot()
    def on_values_changed(self):
        self.variable_changed.emit()

    @Slot()
    def on_value_selection_changed(self):
        rows = self.values_edit.selectionModel().selectedRows()
        if len(rows) == 1:
            i = rows[0].row()
            self.move_value_up.setEnabled(i != 0)
            self.move_value_down.setEnabled(i != self.values_model.rowCount() - 1)
        else:
            self.move_value_up.setEnabled(False)
            self.move_value_down.setEnabled(False)

    def _remove_category(self):
        """
        Remove the current selected category.

        If the item is an existing category present in the source variable it
        is marked as removed in the view. But if it was added in the set
        transformation it is removed entirely from the model and view.
        """
        view = self.values_edit
        rows = view.selectionModel().selectedRows(0)
        if not rows:
            return  # pragma: no cover
        for index in rows:
            model = index.model()
            state = index.data(EditStateRole)
            pos = index.data(SourcePosRole)
            if pos is not None and pos >= 0:
                # existing level -> only mark/toggle its dropped state,
                model.setData(
                    index,
                    ItemEditState.Dropped if state != ItemEditState.Dropped
                    else ItemEditState.NoState,
                    EditStateRole)
            elif state == ItemEditState.Added:
                # new level -> remove it
                model.removeRow(index.row())
            else:
                assert False, "invalid state '{}' for {}" \
                    .format(state, index.row())

    def _add_category(self):
        """
        Add a new category
        """
        view = self.values_edit
        model = view.model()

        with disconnected(model.dataChanged, self.on_values_changed,
                          Qt.UniqueConnection):
            row = model.rowCount()
            if not model.insertRow(model.rowCount()):
                return
            index = model.index(row, 0)
            model.setItemData(
                index, {
                    Qt.EditRole: "",
                    SourcePosRole: None,
                    EditStateRole: ItemEditState.Added
                }
            )
            view.setCurrentIndex(index)
            view.edit(index)
        self.on_values_changed()

    def _merge_categories(self) -> None:
        """
        Merge less common categories into one with the dialog for merge
        selection.
        """
        view = self.values_edit
        model = view.model()  # type: QAbstractItemModel

        selected_attributes = [ind.data(SourceNameRole)
                               for ind in view.selectedIndexes()]
        dlg = GroupItemsDialog(
            self.var, self._values, selected_attributes,
            self.merge_dialog_settings.get(self.var, {}), self,
            windowTitle="Import Options",
            sizeGripEnabled=True,
        )
        dlg.setWindowModality(Qt.WindowModal)
        status = dlg.exec()
        dlg.deleteLater()
        self.merge_dialog_settings[self.var] = dlg.get_dialog_settings()

        rows = (model.index(i, 0) for i in range(model.rowCount()))

        def complete_merge(text, merge_attributes):
            # write the new text for edit role in all rows
            with disconnected(model.dataChanged, self.on_values_changed):
                for row in rows:
                    if row.data(SourceNameRole) in merge_attributes:
                        model.setData(row, text, Qt.EditRole)
            self.variable_changed.emit()

        if status == QDialog.Accepted:
            complete_merge(
                dlg.get_merged_value_name(), dlg.get_merge_attributes()
            )

    def _rename_selected_categories(self):
        """
        Rename selected categories and merging them.

        Popup an editable combo box for selection/edit of a new value.
        """
        view = self.values_edit
        selmodel = view.selectionModel()
        index = view.currentIndex()
        if not selmodel.isSelected(index):
            indices = selmodel.selectedRows(0)
            if indices:
                index = indices[0]
        # delegate to the CategoriesEditDelegate
        view.edit(index)


class ContinuousVariableEditor(VariableEditor):
    # TODO: enable editing of display format...
    pass


class TimeVariableEditor(VariableEditor):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        form = self.layout().itemAt(0)

        self.format_cb = QComboBox()
        for item, data in [("Detect automatically", (None, 1, 1))] + list(
            Orange.data.TimeVariable.ADDITIONAL_FORMATS.items()
        ):
            self.format_cb.addItem(item, StrpTime(item, *data))
        self.format_cb.currentIndexChanged.connect(self.variable_changed)
        form.insertRow(2, "Format:", self.format_cb)

    def set_data(self, var, transform=()):
        super().set_data(var, transform)
        if self.parent() is not None and isinstance(self.parent().var, Time):
            # when transforming from time to time disable format selection combo
            self.format_cb.setEnabled(False)
        else:
            # select the format from StrpTime transform
            for tr in transform:
                if isinstance(tr, StrpTime):
                    index = self.format_cb.findText(tr.label)
                    self.format_cb.setCurrentIndex(index)
            self.format_cb.setEnabled(True)

    def get_data(self):
        var, tr = super().get_data()
        if var is not None and (self.parent() is None or not isinstance(self.parent().var, Time)):
            # do not add StrpTime when transforming from time to time
            tr.insert(0, self.format_cb.currentData())
        return var, tr


def variable_icon(var):
    # type: (Union[Variable, Type[Variable], ReinterpretTransform]) -> QIcon
    if not isinstance(var, type):
        var = type(var)

    if issubclass(var, (Categorical, AsCategorical)):
        return gui.attributeIconDict[1]
    elif issubclass(var, (Real, AsContinuous)):
        return gui.attributeIconDict[2]
    elif issubclass(var, (String, AsString)):
        return gui.attributeIconDict[3]
    elif issubclass(var, (Time, AsTime)):
        return gui.attributeIconDict[4]
    else:
        return gui.attributeIconDict[-1]


#: ItemDataRole storing the data vector transform
#: (`List[Union[ReinterpretTransform, Transform]]`)
TransformRole = Qt.UserRole + 42


class VariableEditDelegate(QStyledItemDelegate):
    ReinterpretNames = {
        AsCategorical: "categorical", AsContinuous: "numeric",
        AsString: "string", AsTime: "time"
    }

    def initStyleOption(self, option, index):
        # type: (QStyleOptionViewItem, QModelIndex) -> None
        super().initStyleOption(option, index)
        item = index.data(Qt.EditRole)
        var = tr = None
        if isinstance(item, DataVectorTypes):
            var = item.vtype
            option.icon = variable_icon(var)
        if isinstance(item, VariableTypes):
            var = item
            option.icon = variable_icon(item)
        elif isinstance(item, Orange.data.Variable):
            var = item
            option.icon = gui.attributeIconDict[var]

        transform = index.data(TransformRole)
        if not isinstance(transform, list):
            transform = []

        if transform and isinstance(transform[0], ReinterpretTransformTypes):
            option.icon = variable_icon(transform[0])

        if not option.icon.isNull():
            option.features |= QStyleOptionViewItem.HasDecoration

        if var is not None:
            text = var.name
            for tr in transform:
                if isinstance(tr, Rename):
                    text = ("{} \N{RIGHTWARDS ARROW} {}"
                            .format(var.name, tr.name))
            for tr in transform:
                if isinstance(tr, ReinterpretTransformTypes):
                    text += f" (reinterpreted as " \
                            f"{self.ReinterpretNames[type(tr)]})"
            option.text = text
        if transform:
            # mark as changed (maybe also change color, add text, ...)
            option.font.setItalic(True)

        multiplicity = index.data(MultiplicityRole)
        if isinstance(multiplicity, int) and multiplicity > 1:
            option.palette.setBrush(QPalette.Text, QBrush(Qt.red))
            option.palette.setBrush(QPalette.HighlightedText, QBrush(Qt.red))

    def helpEvent(self, event: QHelpEvent, view: QAbstractItemView,
                  option: QStyleOptionViewItem, index: QModelIndex) -> bool:
        multiplicity = index.data(MultiplicityRole)
        name = VariableListModel.effective_name(index)
        if isinstance(multiplicity, int) and multiplicity > 1 \
                and name is not None:
            QToolTip.showText(
                event.globalPos(), f"Name `{name}` is duplicated",
                view.viewport()
            )
            return True
        else:  # pragma: no cover
            return super().helpEvent(event, view, option, index)


# Item model for edited variables (Variable). Define a display role to be the
# source variable name. This is used only in keyboard search. The display is
# otherwise completely handled by a delegate.
class VariableListModel(CountedListModel):
    def data(self, index, role=Qt.DisplayRole):
        # type: (QModelIndex, Qt.ItemDataRole) -> Any
        row = index.row()
        if not index.isValid() or not 0 <= row < self.rowCount():
            return None
        if role == Qt.DisplayRole:
            item = self[row]
            if isinstance(item, VariableTypes):
                return item.name
            if isinstance(item, DataVectorTypes):
                return item.vtype.name
        return super().data(index, role)

    def key(self, index):
        return VariableListModel.effective_name(index)

    def keyRoles(self):  # type: () -> FrozenSet[int]
        return frozenset((Qt.DisplayRole, Qt.EditRole, TransformRole))

    @staticmethod
    def effective_name(index) -> Optional[str]:
        item = index.data(Qt.EditRole)
        if isinstance(item, DataVectorTypes):
            var = item.vtype
        elif isinstance(item, VariableTypes):
            var = item
        else:
            return None
        tr = index.data(TransformRole)
        return effective_name(var, tr or [])


def effective_name(var: Variable, tr: Sequence[Transform]) -> str:
    name = var.name
    for t in tr:
        if isinstance(t, Rename):
            name = t.name
    return name


class ReinterpretVariableEditor(VariableEditor):
    """
    A 'compound' variable editor capable of variable type reinterpretations.
    """
    _editors = {
        Categorical: 0,
        Real: 1,
        String: 2,
        Time: 3,
        type(None): -1,
    }

    def __init__(self, parent=None, **kwargs):
        # Explicitly skip VariableEditor's __init__, this is ugly but we have
        # a completely different layout/logic as a compound editor (should
        # really not subclass VariableEditor).
        super(VariableEditor, self).__init__(parent, **kwargs)  # pylint: disable=bad-super-call
        self.var = None  # type: Optional[Variable]
        self.__transform = None  # type: Optional[ReinterpretTransform]
        self.__data = None  # type: Optional[DataVector]
        #: Stored transform state indexed by variable. Used to preserve state
        #: between type switches.
        self.__history = {}  # type: Dict[Variable, List[Transform]]

        self.setLayout(QStackedLayout())

        def decorate(editor: VariableEditor) -> VariableEditor:
            """insert an type combo box into a `editor`'s layout."""
            form = editor.layout().itemAt(0)
            assert isinstance(form, QFormLayout)
            typecb = QComboBox(objectName="type-combo")
            typecb.addItem(variable_icon(Categorical), "Categorical", Categorical)
            typecb.addItem(variable_icon(Real), "Numeric", Real)
            typecb.addItem(variable_icon(String), "Text", String)
            typecb.addItem(variable_icon(Time), "Time", Time)
            typecb.activated[int].connect(self.__reinterpret_activated)
            form.insertRow(1, "Type:", typecb)
            # Insert the typecb after name edit in the focus chain
            name_edit = editor.findChild(QLineEdit, )
            if name_edit is not None:
                QWidget.setTabOrder(name_edit, typecb)
            return editor
        # This is ugly. Create an editor for each type and insert a type
        # selection combo box into its layout. Switch between widgets
        # on type change.
        self.disc_edit = dedit = decorate(DiscreteVariableEditor())
        cedit = decorate(ContinuousVariableEditor())
        tedit = decorate(TimeVariableEditor())
        sedit = decorate(VariableEditor())

        for ed in [dedit, cedit, tedit, sedit]:
            ed.variable_changed.connect(self.variable_changed)

        self.layout().addWidget(dedit)
        self.layout().addWidget(cedit)
        self.layout().addWidget(sedit)
        self.layout().addWidget(tedit)

    def set_data(self, data, transform=()):  # pylint: disable=arguments-differ
        # type: (Optional[DataVector], Sequence[Transform]) -> None
        """
        Set the editor data.

        Note
        ----
        This must be a `DataVector` as the vector's values are needed for type
        reinterpretation/casts.

        If the `transform` sequence contains ReinterpretTransform then it
        must be in the first position.
        """
        type_transform = None  # type: Optional[ReinterpretTransform]
        if transform:
            _tr = transform[0]
            if isinstance(_tr, ReinterpretTransformTypes):
                type_transform = _tr
                transform = transform[1:]
            assert not any(isinstance(t, ReinterpretTransformTypes)
                           for t in transform)
        self.__transform = type_transform
        self.__data = data
        self.var = data.vtype if data is not None else None

        if type_transform is not None and data is not None:
            data = type_transform(data)
        if data is not None:
            var = data.vtype
        else:
            var = None
        index = self._editors.get(type(var), -1)
        self.layout().setCurrentIndex(index)
        if index != -1:
            w = self.layout().currentWidget()
            assert isinstance(w, VariableEditor)
            if isinstance(var, Categorical):
                w.set_data_categorical(var, data.data(), transform=transform)
            else:
                w.set_data(var, transform=transform)
            self.__history[var] = tuple(transform)
            cb = w.findChild(QComboBox, "type-combo")
            cb.setCurrentIndex(index)

    def get_data(self):
        # type: () -> Tuple[Variable, Sequence[Transform]]
        editor = self.layout().currentWidget()  # type: VariableEditor
        var, tr = editor.get_data()
        if type(var) != type(self.var):  # pylint: disable=unidiomatic-typecheck
            assert self.__transform is not None
            var = self.var
            tr = [self.__transform, *tr]
        return var, tr

    def __reinterpret_activated(self, index):
        layout = self.layout()
        assert isinstance(layout, QStackedLayout)
        if index == layout.currentIndex():
            return
        current = layout.currentWidget()
        assert isinstance(current, VariableEditor)
        Specific = {
            Categorical: CategoricalTransformTypes
        }
        _var, _tr = current.get_data()
        if _var is not None:
            self.__history[_var] = _tr

        var = self.var
        transform = self.__transform
        # take/preserve the general transforms that apply to all types
        specific = Specific.get(type(var), ())
        _tr = [t for t in _tr if not isinstance(t, specific)]

        layout.setCurrentIndex(index)
        w = layout.currentWidget()
        cb = w.findChild(QComboBox, "type-combo")
        cb.setCurrentIndex(index)
        cb.setFocus()
        target = cb.itemData(index, Qt.UserRole)
        assert issubclass(target, VariableTypes)
        if not isinstance(var, target):
            if target == Real:
                transform = AsContinuous()
            elif target == Categorical:
                transform = AsCategorical()
            elif target == Time:
                transform = AsTime()
            elif target == String:
                transform = AsString()
        else:
            transform = None
            var = self.var

        self.__transform = transform
        data = None
        if transform is not None and self.__data is not None:
            data = transform(self.__data)
            var = data.vtype

        if var in self.__history:
            tr = self.__history[var]
        else:
            tr = []
        # type specific transform
        specific = Specific.get(type(var), ())
        # merge tr and _tr
        tr = _tr + [t for t in tr if isinstance(t, specific)]
        with disconnected(
                w.variable_changed, self.variable_changed,
                Qt.UniqueConnection
        ):
            if isinstance(w, DiscreteVariableEditor):
                data = data or self.__data
                w.set_data_categorical(var, data.data(), transform=tr)
            else:
                w.set_data(var, transform=tr)
        self.variable_changed.emit()

    def set_merge_context(self, merge_context):
        self.disc_edit.merge_dialog_settings = merge_context

    def get_merge_context(self):
        return self.disc_edit.merge_dialog_settings


class OWEditDomain(widget.OWWidget):
    name = "Edit Domain"
    description = "Rename variables, edit categories and variable annotations."
    icon = "icons/EditDomain.svg"
    priority = 3125
    keywords = ["rename", "drop", "reorder", "order"]

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        data = Output("Data", Orange.data.Table)

    class Error(widget.OWWidget.Error):
        duplicate_var_name = widget.Msg("A variable name is duplicated.")

    settingsHandler = settings.DomainContextHandler()
    settings_version = 2

    _domain_change_store = settings.ContextSetting({})
    _selected_item = settings.ContextSetting(None)  # type: Optional[Tuple[str, int]]
    _merge_dialog_settings = settings.ContextSetting({})
    output_table_name = settings.ContextSetting("")

    want_main_area = False

    def __init__(self):
        super().__init__()
        self.data = None  # type: Optional[Orange.data.Table]
        #: The current selected variable index
        self.selected_index = -1
        self._invalidated = False
        self.typeindex = 0

        main = gui.hBox(self.controlArea, spacing=6)
        box = gui.vBox(main, "Variables")

        self.variables_model = VariableListModel(parent=self)
        self.variables_view = self.domain_view = QListView(
            selectionMode=QListView.SingleSelection,
            uniformItemSizes=True,
        )
        self.variables_view.setItemDelegate(VariableEditDelegate(self))
        self.variables_view.setModel(self.variables_model)
        self.variables_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        box.layout().addWidget(self.variables_view)

        box = gui.vBox(main, "Edit")
        self._editor = ReinterpretVariableEditor()
        box.layout().addWidget(self._editor)

        self.le_output_name = gui.lineEdit(
            self.buttonsArea, self, "output_table_name", "Output table name: ",
            orientation=Qt.Horizontal)

        gui.rubber(self.buttonsArea)

        bbox = gui.hBox(self.buttonsArea)
        breset_all = gui.button(
            bbox, self, "Reset All",
            objectName="button-reset-all",
            toolTip="Reset all variables to their input state.",
            autoDefault=False,
            callback=self.reset_all
        )
        breset = gui.button(
            bbox, self, "Reset Selected",
            objectName="button-reset",
            toolTip="Rest selected variable to its input state.",
            autoDefault=False,
            callback=self.reset_selected
        )
        bapply = gui.button(
            bbox, self, "Apply",
            objectName="button-apply",
            toolTip="Apply changes and commit data on output.",
            default=True,
            autoDefault=False,
            callback=self.commit
        )

        self.variables_view.setFocus(Qt.NoFocusReason)  # initial focus

    @Inputs.data
    def set_data(self, data):
        """Set input dataset."""
        self.closeContext()
        self.clear()
        self.data = data

        if self.data is not None:
            self.setup_model(data)
            self.le_output_name.setPlaceholderText(data.name)
            self.openContext(self.data)
            self._editor.set_merge_context(self._merge_dialog_settings)
            self._restore()
        else:
            self.le_output_name.setPlaceholderText("")

        self.commit()

    def clear(self):
        """Clear the widget state."""
        self.data = None
        self.variables_model.clear()
        self.clear_editor()
        assert self.selected_index == -1
        self.selected_index = -1

        self._selected_item = None
        self._domain_change_store = {}
        self._merge_dialog_settings = {}

    def reset_selected(self):
        """Reset the currently selected variable to its original state."""
        ind = self.selected_var_index()
        if ind >= 0:
            model = self.variables_model
            midx = model.index(ind)
            var = midx.data(Qt.EditRole)
            tr = midx.data(TransformRole)
            if not tr:
                return  # nothing to reset
            editor = self._editor
            with disconnected(editor.variable_changed,
                              self._on_variable_changed):
                model.setData(midx, [], TransformRole)
                editor.set_data(var, transform=[])
            self._invalidate()

    def reset_all(self):
        """Reset all variables to their original state."""
        self._domain_change_store = {}
        if self.data is not None:
            model = self.variables_model
            for i in range(model.rowCount()):
                midx = model.index(i)
                model.setData(midx, [], TransformRole)
            index = self.selected_var_index()
            if index >= 0:
                self.open_editor(index)
            self._invalidate()

    def selected_var_index(self):
        """Return the current selected variable index."""
        rows = self.variables_view.selectedIndexes()
        assert len(rows) <= 1
        return rows[0].row() if rows else -1

    def setup_model(self, data: Orange.data.Table):
        model = self.variables_model
        vars_ = []
        columns = []
        for i, _, var, coldata in enumerate_columns(data):
            var = abstract(var)
            vars_.append(var)
            if isinstance(var, Categorical):
                data = CategoricalVector(var, coldata)
            elif isinstance(var, Real):
                data = RealVector(var, coldata)
            elif isinstance(var, Time):
                data = TimeVector(var, coldata)
            elif isinstance(var, String):
                data = StringVector(var, coldata)
            columns.append(data)

        model[:] = vars_
        for i, d in enumerate(columns):
            model.setData(model.index(i), d, Qt.EditRole)

    def _restore(self, ):
        """
        Restore the edit transform from saved state.
        """
        model = self.variables_model
        for i in range(model.rowCount()):
            midx = model.index(i, 0)
            coldesc = model.data(midx, Qt.EditRole)  # type: DataVector
            tr = self._restore_transform(coldesc.vtype)
            if tr:
                model.setData(midx, tr, TransformRole)

        # Restore the current variable selection
        i = -1
        if self._selected_item is not None:
            for i, vec in enumerate(model):
                if vec.vtype.name_type() == self._selected_item:
                    break
        if i == -1 and model.rowCount():
            i = 0

        if i != -1:
            itemmodels.select_row(self.variables_view, i)

    def _on_selection_changed(self):
        self.selected_index = self.selected_var_index()
        if self.selected_index != -1:
            self._selected_item = self.variables_model[self.selected_index].vtype.name_type()
        else:
            self._selected_item = None
        self.open_editor(self.selected_index)

    def open_editor(self, index):
        # type: (int) -> None
        self.clear_editor()
        model = self.variables_model
        if not 0 <= index < model.rowCount():
            return
        idx = model.index(index, 0)
        vector = model.data(idx, Qt.EditRole)
        tr = model.data(idx, TransformRole)
        if tr is None:
            tr = []
        editor = self._editor
        editor.set_data(vector, transform=tr)
        editor.variable_changed.connect(
            self._on_variable_changed, Qt.UniqueConnection
        )

    def clear_editor(self):
        current = self._editor
        try:
            current.variable_changed.disconnect(self._on_variable_changed)
        except TypeError:
            pass
        current.set_data(None)
        current.layout().currentWidget().clear()

    @Slot()
    def _on_variable_changed(self):
        """User edited the current variable in editor."""
        assert 0 <= self.selected_index <= len(self.variables_model)
        editor = self._editor
        var, transform = editor.get_data()
        model = self.variables_model
        midx = model.index(self.selected_index, 0)
        model.setData(midx, transform, TransformRole)
        self._store_transform(var, transform)
        self._invalidate()

    def _store_transform(self, var, transform):
        # type: (Variable, List[Transform]) -> None
        self._domain_change_store[deconstruct(var)] = [deconstruct(t) for t in transform]

    def _restore_transform(self, var):
        # type: (Variable) -> List[Transform]
        tr_ = self._domain_change_store.get(deconstruct(var), [])
        tr = []

        for t in tr_:
            try:
                tr.append(reconstruct(*t))
            except (NameError, TypeError) as err:
                warnings.warn(
                    "Failed to restore transform: {}, {!r}".format(t, err),
                    UserWarning, stacklevel=2
                )
        return tr

    def _invalidate(self):
        self._set_modified(True)

    def _set_modified(self, state):
        self._invalidated = state
        b = self.findChild(QPushButton, "button-apply")
        if isinstance(b, QPushButton):
            f = b.font()
            f.setItalic(state)
            b.setFont(f)

    def commit(self):
        """
        Apply the changes to the input data and send the changed data to output.
        """
        self._set_modified(False)
        self.Error.duplicate_var_name.clear()

        data = self.data
        if data is None:
            self.Outputs.data.send(None)
            return
        model = self.variables_model

        def state(i):
            # type: (int) -> Tuple[DataVector, List[Transform]]
            midx = self.variables_model.index(i, 0)
            return (model.data(midx, Qt.EditRole),
                    model.data(midx, TransformRole))

        state = [state(i) for i in range(model.rowCount())]
        input_vars = data.domain.variables + data.domain.metas
        if self.output_table_name in ("", data.name) \
                and not any(requires_transform(var, trs)
                            for var, (_, trs) in zip(input_vars, state)):
            self.Outputs.data.send(data)
            return

        assert all(v_.vtype.name == v.name
                   for v, (v_, _) in zip(input_vars, state))
        output_vars = []
        unlinked_vars = []
        unlink_domain = False
        for (_, tr), v in zip(state, input_vars):
            if tr:
                var = apply_transform(v, data, tr)
                if requires_unlink(v, tr):
                    unlinked_var = var.copy(compute_value=None)
                    unlink_domain = True
                else:
                    unlinked_var = var
            else:
                unlinked_var = var = v
            output_vars.append(var)
            unlinked_vars.append(unlinked_var)

        if len(output_vars) != len({v.name for v in output_vars}):
            self.Error.duplicate_var_name()
            self.Outputs.data.send(None)
            return

        domain = data.domain
        nx = len(domain.attributes)
        ny = len(domain.class_vars)

        def construct_domain(vars_list):
            # Move non primitive Xs, Ys to metas (if they were changed)
            Xs = [v for v in vars_list[:nx] if v.is_primitive()]
            Ys = [v for v in vars_list[nx: nx + ny] if v.is_primitive()]
            Ms = vars_list[nx + ny:] + \
                 [v for v in vars_list[:nx + ny] if not  v.is_primitive()]
            return Orange.data.Domain(Xs, Ys, Ms)

        domain = construct_domain(output_vars)
        new_data = data.transform(domain)
        if unlink_domain:
            unlinked_domain = construct_domain(unlinked_vars)
            new_data = new_data.from_numpy(
                unlinked_domain,
                new_data.X, new_data.Y, new_data.metas, new_data.W,
                new_data.attributes, new_data.ids
            )
        if self.output_table_name:
            new_data.name = self.output_table_name
        self.Outputs.data.send(new_data)

    def sizeHint(self):
        sh = super().sizeHint()
        return sh.expandedTo(QSize(660, 550))

    def storeSpecificSettings(self):
        """
        Update setting before context closes - also when widget closes.
        """
        self._merge_dialog_settings = self._editor.get_merge_context()

    def send_report(self):

        if self.data is not None:
            model = self.variables_model
            state = ((model.data(midx, Qt.EditRole),
                      model.data(midx, TransformRole))
                     for i in range(model.rowCount())
                     for midx in [model.index(i)])
            parts = []
            for vector, trs in state:
                if trs:
                    parts.append(report_transform(vector.vtype, trs))
            if parts:
                html = ("<ul>" +
                        "".join(map("<li>{}</li>".format, parts)) +
                        "</ul>")
            else:
                html = "No changes"
            self.report_raw("", html)
        else:
            self.report_data(None)

    @classmethod
    def migrate_context(cls, context, version):
        # pylint: disable=bad-continuation
        if version is None or version <= 1:
            hints_ = context.values.get("domain_change_hints", ({}, -2))[0]
            store = []
            ns = "Orange.data.variable"
            mapping = {
                "DiscreteVariable":
                    lambda name, args, attrs:
                        ("Categorical", (name, tuple(args[0][1]), ())),
                "TimeVariable":
                    lambda name, _, attrs:
                        ("Time", (name, ())),
                "ContinuousVariable":
                    lambda name, _, attrs:
                        ("Real", (name, (3, "f"), ())),
                "StringVariable":
                    lambda name, _, attrs:
                        ("String", (name, ())),
            }
            for (module, class_name, *rest), target in hints_.items():
                if module != ns:
                    continue
                f = mapping.get(class_name)
                if f is None:
                    continue
                trs = []
                key_mapped = f(*rest)
                item_mapped = f(*target[2:])
                src = reconstruct(*key_mapped)   # type: Variable
                dst = reconstruct(*item_mapped)  # type: Variable
                if src.name != dst.name:
                    trs.append(Rename(dst.name))
                if src.annotations != dst.annotations:
                    trs.append(Annotate(dst.annotations))
                if isinstance(src, Categorical):
                    if src.categories != dst.categories:
                        assert len(src.categories) == len(dst.categories)
                        trs.append(CategoriesMapping(
                            list(zip(src.categories, dst.categories))))
                store.append((deconstruct(src), [deconstruct(tr) for tr in trs]))
            context.values["_domain_change_store"] = (dict(store), -2)


def enumerate_columns(
        table: Orange.data.Table
) -> Iterable[Tuple[int, str, Orange.data.Variable, Callable[[], ndarray]]]:
    domain = table.domain
    for i, (var, role) in enumerate(
            chain(zip(domain.attributes, repeat("x")),
                  zip(domain.class_vars, repeat("y")),
                  zip(domain.metas, repeat("m"))),
    ):
        if i >= len(domain.variables):
            i = len(domain.variables) - i - 1
        data = partial(table_column_data, table, i)
        yield i, role, var, data


def table_column_data(
        table: Orange.data.Table,
        var: Union[Orange.data.Variable, int],
        dtype=None
) -> MArray:
    col, copy = table.get_column_view(var)
    var = table.domain[var]  # type: Orange.data.Variable
    if var.is_primitive() and not np.issubdtype(col.dtype, np.inexact):
        col = col.astype(float)
        copy = True

    if dtype is None:
        if isinstance(var, Orange.data.TimeVariable):
            dtype = np.dtype("M8[us]")
            col = col * 1e6
        elif isinstance(var, Orange.data.ContinuousVariable):
            dtype = np.dtype(float)
        elif isinstance(var, Orange.data.DiscreteVariable):
            _values = tuple(var.values)
            _n_values = len(_values)
            dtype = np.dtype(int, metadata={
                "__formatter": lambda i: _values[i] if 0 <= i < _n_values else "?"
            })
        elif isinstance(var, Orange.data.StringVariable):
            dtype = np.dtype(object)
        else:
            assert False
    mask = orange_isna(var, col)

    if dtype != col.dtype:
        col = col.astype(dtype)
        copy = True

    if not copy:
        col = col.copy()
    return MArray(col, mask=mask)


def report_transform(var, trs):
    # type: (Variable, List[Transform]) -> str
    """
    Return a html fragment summarizing the changes applied by `trs` list.

    Parameters
    ----------
    var : Variable
        A variable descriptor no which trs operates
    trs : List[Transform]
        A non empty list of `Transform` instances.

    Returns
    -------
    report : str
    """
    # pylint: disable=too-many-branches
    ReinterpretTypeCode = {
        AsCategorical:  "C", AsContinuous: "N", AsString: "S", AsTime: "T",
    }

    def type_char(value: ReinterpretTransform) -> str:
        return ReinterpretTypeCode.get(type(value), "?")

    def strike(text):
        return "<s>{}</s>".format(escape(text))

    def i(text):
        return "<i>{}</i>".format(escape(text))

    def text(text):
        return "<span>{}</span>".format(escape(text))
    assert trs
    rename = annotate = catmap = unlink = None
    reinterpret = None

    for tr in trs:
        if isinstance(tr, Rename):
            rename = tr
        elif isinstance(tr, Annotate):
            annotate = tr
        elif isinstance(tr, CategoriesMapping):
            catmap = tr
        elif isinstance(tr, Unlink):
            unlink = tr
        elif isinstance(tr, ReinterpretTransformTypes):
            reinterpret = tr

    if reinterpret is not None:
        header = "{} → ({}) {}".format(
            var.name, type_char(reinterpret),
            rename.name if rename is not None else var.name
        )
    elif rename is not None:
        header = "{} → {}".format(var.name, rename.name)
    else:
        header = var.name
    if unlink is not None:
        header += "(unlinked from source)"

    values_section = None
    if catmap is not None:
        values_section = ("Values", [])
        lines = values_section[1]
        for ci, cj in catmap.mapping:
            if ci is None:
                item = cj + ("&nbsp;" * 3) + "(added)"
            elif cj is None:
                item = strike(ci)
            else:
                item = ci + " → " + cj
            lines.append(item)

    annotate_section = None
    if annotate is not None:
        annotate_section = ("Labels", [])
        lines = annotate_section[1]
        old = dict(var.annotations)
        new = dict(annotate.annotations)
        for name in sorted(set(old) - set(new)):
            lines.append(
                "<s>" + i(name) + " : " + text(old[name]) + "</s>"
            )
        for name in sorted(set(new) - set(old)):
            lines.append(
                i(name) + " : " + text(new[name]) + "&nbsp;" * 3 + i("(new)")
            )

        for name in sorted(set(new) & set(old)):
            if new[name] != old[name]:
                lines.append(
                    i(name) + " : " + text(old[name]) + " → " + text(new[name])
                )

    html = ["<div style='font-weight: bold;'>{}</div>".format(header)]
    for title, contents in filter(None, [values_section, annotate_section]):
        section_header = "<div>{}:</div>".format(title)
        section_contents = "<br/>\n".join(contents)
        html.append(section_header)
        html.append(
            "<div style='padding-left: 1em;'>" +
            section_contents +
            "</div>"
        )
    return "\n".join(html)


def abstract(var):
    # type: (Orange.data.Variable) -> Variable
    """
    Return `Varaible` descriptor for an `Orange.data.Variable` instance.

    Parameters
    ----------
    var : Orange.data.Variable

    Returns
    -------
    var : Variable
    """
    annotations = tuple(sorted(
        (key, str(value))
        for key, value in var.attributes.items()
    ))
    linked = var.compute_value is not None
    if isinstance(var, Orange.data.DiscreteVariable):
        return Categorical(var.name, tuple(var.values), annotations, linked)
    elif isinstance(var, Orange.data.TimeVariable):
        return Time(var.name, annotations, linked)
    elif isinstance(var, Orange.data.ContinuousVariable):
        return Real(var.name, (var.number_of_decimals, 'f'), annotations, linked)
    elif isinstance(var, Orange.data.StringVariable):
        return String(var.name, annotations, linked)
    else:
        raise TypeError


def _parse_attributes(mapping):
    # type: (Iterable[Tuple[str, str]]) -> Dict[str, Any]
    # Use the same functionality that parses attributes
    # when reading text files
    return Orange.data.Flags([
        "{}={}".format(*item) for item in mapping
    ]).attributes


def apply_transform(var, table, trs):
    # type: (Orange.data.Variable, Orange.data.Table, List[Transform]) -> Orange.data.Variable
    """
    Apply a list of `Transform` instances on an `Orange.data.Variable`.
    """
    if trs and isinstance(trs[0], ReinterpretTransformTypes):
        reinterpret, trs = trs[0], trs[1:]
        coldata = table_column_data(table, var)
        var = apply_reinterpret(var, reinterpret, coldata)
    if trs:
        return apply_transform_var(var, trs)
    else:
        return var


def requires_unlink(var: Orange.data.Variable, trs: List[Transform]) -> bool:
    # Variable is only unlinked if it has compute_value  or if it has other
    # transformations (that might had added compute_value)
    return trs is not None \
           and any(isinstance(tr, Unlink) for tr in trs) \
           and (var.compute_value is not None or len(trs) > 1)


def requires_transform(var: Orange.data.Variable, trs: List[Transform]) -> bool:
    # Unlink is treated separately: Unlink is required only if the variable
    # has compute_value. Hence tranform is required if it has any
    # transformations other than Unlink, or if unlink is indeed required.
    return trs is not None and (
        not all(isinstance(tr, Unlink) for tr in trs)
        or requires_unlink(var, trs)
    )


@singledispatch
def apply_transform_var(var, trs):
    # type: (Orange.data.Variable, List[Transform]) -> Orange.data.Variable
    raise NotImplementedError


@apply_transform_var.register(Orange.data.DiscreteVariable)
def apply_transform_discete(var, trs):
    # type: (Orange.data.DiscreteVariable, List[Transform]) -> Orange.data.Variable
    # pylint: disable=too-many-branches
    name, annotations = var.name, var.attributes
    mapping = None
    for tr in trs:
        if isinstance(tr, Rename):
            name = tr.name
        elif isinstance(tr, CategoriesMapping):
            mapping = tr.mapping
        elif isinstance(tr, Annotate):
            annotations = _parse_attributes(tr.annotations)

    source_values = var.values
    if mapping is not None:
        dest_values = list(unique(cj for ci, cj in mapping if cj is not None))
    else:
        dest_values = var.values

    def positions(values):
        rval = {c: i for i, c in enumerate(values)}
        assert len(rval) == len(values)
        return rval
    source_codes = positions(source_values)
    dest_codes = positions(dest_values)
    if mapping is not None:
        # construct a lookup table
        lookup = np.full(len(source_values), np.nan, dtype=float)
        for ci, cj in mapping:
            if ci is not None and cj is not None:
                i, j = source_codes[ci], dest_codes[cj]
                lookup[i] = j
        lookup = Lookup(var, lookup)
    else:
        lookup = Identity(var)
    variable = Orange.data.DiscreteVariable(
        name, values=dest_values, compute_value=lookup
    )
    variable.attributes.update(annotations)
    return variable


@apply_transform_var.register(Orange.data.ContinuousVariable)
def apply_transform_continuous(var, trs):
    # type: (Orange.data.ContinuousVariable, List[Transform]) -> Orange.data.Variable
    name, annotations = var.name, var.attributes
    for tr in trs:
        if isinstance(tr, Rename):
            name = tr.name
        elif isinstance(tr, Annotate):
            annotations = _parse_attributes(tr.annotations)
    variable = Orange.data.ContinuousVariable(
        name=name, compute_value=Identity(var)
    )
    variable.attributes.update(annotations)
    return variable


@apply_transform_var.register(Orange.data.TimeVariable)
def apply_transform_time(var, trs):
    # type: (Orange.data.TimeVariable, List[Transform]) -> Orange.data.Variable
    name, annotations = var.name, var.attributes
    for tr in trs:
        if isinstance(tr, Rename):
            name = tr.name
        elif isinstance(tr, Annotate):
            annotations = _parse_attributes(tr.annotations)
    variable = Orange.data.TimeVariable(
        name=name, have_date=var.have_date, have_time=var.have_time,
        compute_value=Identity(var)
    )
    variable.attributes.update(annotations)
    return variable


@apply_transform_var.register(Orange.data.StringVariable)
def apply_transform_string(var, trs):
    # type: (Orange.data.StringVariable, List[Transform]) -> Orange.data.Variable
    name, annotations = var.name, var.attributes
    out_type = Orange.data.StringVariable
    compute_value = Identity
    for tr in trs:
        if isinstance(tr, Rename):
            name = tr.name
        elif isinstance(tr, Annotate):
            annotations = _parse_attributes(tr.annotations)
        elif isinstance(tr, StrpTime):
            out_type = partial(
                Orange.data.TimeVariable, have_date=tr.have_date, have_time=tr.have_time
            )
            compute_value = partial(ReparseTimeTransform, tr=tr)
    variable = out_type(name=name, compute_value=compute_value(var))
    variable.attributes.update(annotations)
    return variable


def ftry(
        func: Callable[..., A],
        error: Union[Type[BaseException], Tuple[Type[BaseException]]],
        default: B
) -> Callable[..., Union[A, B]]:
    """
    Wrap a `func` such that if `errors` occur `default` is returned instead."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except error:
            return default
    return wrapper


class DictMissingConst(dict):
    """
    `dict` with a constant for `__missing__()` value.
    """
    __slots__ = ("__missing",)

    def __init__(self, missing, *args, **kwargs):
        self.__missing = missing
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        return self.__missing

    def __eq__(self, other):
        return super().__eq__(other) and isinstance(other, DictMissingConst) \
            and (self.__missing == other[()]  # get `__missing`
                 or np.isnan(self.__missing) and np.isnan(other[()]))

    def __hash__(self):
        return hash((self.__missing, frozenset(self.items())))


def make_dict_mapper(
        mapping: Mapping, dtype: Optional[DType] = None
) -> Callable:
    """
    Wrap a `mapping` into a callable ufunc-like function with
    `out`, `dtype`, `where`, ... parameters. If `dtype` is passed to
    `make_dict_mapper` it is used as a the default return dtype,
    otherwise the default dtype is `object`.
    """
    _vmapper = np.frompyfunc(mapping.__getitem__, 1, 1)

    def mapper(arr, out=None, dtype=dtype, **kwargs):
        arr = np.asanyarray(arr)
        if out is None and dtype is not None and arr.shape != ():
            out = np.empty_like(arr, dtype)
        return _vmapper(arr, out, dtype=dtype, casting="unsafe", **kwargs)
    return mapper


as_string = np.frompyfunc(str, 1, 1)
parse_float = ftry(float, ValueError, float("nan"))

_parse_float = np.frompyfunc(parse_float, 1, 1)


def as_float_or_nan(
        arr: ndarray, out: Optional[ndarray] = None, where: Optional[ndarray] = True,
        dtype=None, **kwargs
) -> ndarray:
    """
    Convert elements of the input array using builtin `float`, fill elements
    where conversion failed with NaN.
    """
    if out is None:
        out = np.full(arr.shape, np.nan, float if dtype is None else dtype)
    if np.issubdtype(arr.dtype, np.inexact) or \
            np.issubdtype(arr.dtype, np.integer):
        np.copyto(out, arr, casting="unsafe", where=where)
        return out
    return _parse_float(arr, out, where=where, casting="unsafe", **kwargs)


def copy_attributes(dst: V, src: Orange.data.Variable) -> V:
    # copy `attributes` and `sparse` members from src to dst
    dst.attributes = dict(src.attributes)
    dst.sparse = src.sparse
    return dst


# Building (and applying) concrete type transformations on Table columns

@singledispatch
def apply_reinterpret(var, tr, data):
    # type: (Orange.data.Variable, ReinterpretTransform, MArray) -> Orange.data.Variable
    """
    Apply a re-interpret transform to an `Orange.data.Table`'s column
    """
    raise NotImplementedError


@apply_reinterpret.register(Orange.data.DiscreteVariable)
def apply_reinterpret_d(var, tr, data):
    # type: (Orange.data.DiscreteVariable, ReinterpretTransform, ndarray) -> Orange.data.Variable
    if isinstance(tr, AsCategorical):
        return var
    elif isinstance(tr, (AsString, AsTime)):
        # TimeVar will be interpreted by StrpTime later
        f = Lookup(var, np.array(var.values, dtype=object), unknown="")
        rvar = Orange.data.StringVariable(name=var.name, compute_value=f)
    elif isinstance(tr, AsContinuous):
        f = Lookup(var, np.array(list(map(parse_float, var.values))),
                   unknown=np.nan)
        rvar = Orange.data.ContinuousVariable(
            name=var.name, compute_value=f, sparse=var.sparse
        )
    else:
        assert False
    return copy_attributes(rvar, var)


@apply_reinterpret.register(Orange.data.ContinuousVariable)
def apply_reinterpret_c(var, tr, data: MArray):
    if isinstance(tr, AsCategorical):
        # This is ill defined and should not result in a 'compute_value'
        # (post-hoc expunge from the domain once translated?)
        values, index = categorize_unique(data)
        coldata = index.astype(float)
        coldata[index.mask] = np.nan
        tr = LookupMappingTransform(
            var, DictMissingConst(
                np.nan, {v: i for i, v in enumerate(values)}
            )
        )
        values = tuple(as_string(values))
        rvar = Orange.data.DiscreteVariable(
            name=var.name, values=values, compute_value=tr
        )
    elif isinstance(tr, AsContinuous):
        return var
    elif isinstance(tr, AsString):
        # TimeVar will be interpreted by StrpTime later
        tstr = ToStringTransform(var)
        rvar = Orange.data.StringVariable(name=var.name, compute_value=tstr)
    elif isinstance(tr, AsTime):
        rvar = Orange.data.TimeVariable(name=var.name, compute_value=Identity(var))
    else:
        assert False
    return copy_attributes(rvar, var)


@apply_reinterpret.register(Orange.data.StringVariable)
def apply_reinterpret_s(var: Orange.data.StringVariable, tr, data: MArray):
    if isinstance(tr, AsCategorical):
        # This is ill defined and should not result in a 'compute_value'
        # (post-hoc expunge from the domain once translated?)
        _, values = categorical_from_vector(data)
        mapping = DictMissingConst(
            np.nan, {v: float(i) for i, v in enumerate(values)}
        )
        tr = LookupMappingTransform(var, mapping)
        rvar = Orange.data.DiscreteVariable(
            name=var.name, values=values, compute_value=tr
        )
    elif isinstance(tr, AsContinuous):
        rvar = Orange.data.ContinuousVariable(
            var.name, compute_value=ToContinuousTransform(var)
        )
    elif isinstance(tr, (AsString, AsTime)):
        # TimeVar will be interpreted by StrpTime later
        return var
    else:
        assert False
    return copy_attributes(rvar, var)


@apply_reinterpret.register(Orange.data.TimeVariable)
def apply_reinterpret_t(var: Orange.data.TimeVariable, tr, data):
    if isinstance(tr, AsCategorical):
        values, _ = categorize_unique(data)
        or_values = values.astype(float) / 1e6
        mapping = DictMissingConst(
            np.nan, {v: i for i, v in enumerate(or_values)}
        )
        tr = LookupMappingTransform(var, mapping)
        values = tuple(as_string(values))
        rvar = Orange.data.DiscreteVariable(
            name=var.name, values=values, compute_value=tr
        )
    elif isinstance(tr, AsContinuous):
        rvar = Orange.data.TimeVariable(
            name=var.name, compute_value=Identity(var)
        )
    elif isinstance(tr, AsString):
        rvar = Orange.data.StringVariable(
            name=var.name, compute_value=ToStringTransform(var)
        )
    elif isinstance(tr, AsTime):
        return var
    else:
        assert False
    return copy_attributes(rvar, var)


def orange_isna(variable: Orange.data.Variable, data: ndarray) -> ndarray:
    """
    Return a bool mask masking N/A elements in `data` for the `variable`.
    """
    if variable.is_primitive():
        return np.isnan(data)
    else:
        return data == variable.Unknown


class ToStringTransform(Transformation):
    """
    Transform a variable to string.
    """
    def transform(self, c):
        if self.variable.is_string:
            return c
        elif self.variable.is_discrete or self.variable.is_time:
            r = column_str_repr(self.variable, c)
        elif self.variable.is_continuous:
            r = as_string(c)
        mask = orange_isna(self.variable, c)
        return np.where(mask, "", r)


class ToContinuousTransform(Transformation):
    def transform(self, c):
        if self.variable.is_time:
            return c
        elif self.variable.is_continuous:
            return c
        elif self.variable.is_discrete:
            lookup = Lookup(
                self.variable, as_float_or_nan(self.variable.values),
                unknown=np.nan
            )
            return lookup.transform(c)
        elif self.variable.is_string:
            return as_float_or_nan(c)
        else:
            raise TypeError


def datetime_to_epoch(dti: pd.DatetimeIndex, only_time) -> np.ndarray:
    """Convert datetime to epoch"""
    delta = dti - (dti.normalize() if only_time else pd.Timestamp("1970-01-01"))
    return (delta / pd.Timedelta("1s")).values


class ReparseTimeTransform(Transformation):
    """
    Re-parse the column's string repr as datetime.
    """
    def __init__(self, variable, tr):
        super().__init__(variable)
        self.tr = tr

    def transform(self, c):
        # if self.formats is none guess format option is selected
        formats = self.tr.formats if self.tr.formats is not None else [None]
        for f in formats:
            d = pd.to_datetime(c, errors="coerce", format=f)
            if pd.notnull(d).any():
                return datetime_to_epoch(d, only_time=not self.tr.have_date)
        return np.nan


class LookupMappingTransform(Transformation):
    """
    Map values via a dictionary lookup.
    """
    def __init__(
            self,
            variable: Orange.data.Variable,
            mapping: Mapping,
            dtype: Optional[np.dtype] = None
    ) -> None:
        super().__init__(variable)
        self.mapping = mapping
        self.dtype = dtype
        self._mapper = make_dict_mapper(mapping, dtype)

    def transform(self, c):
        return self._mapper(c)

    def __reduce_ex__(self, protocol):
        return type(self), (self.variable, self.mapping, self.dtype)

    def __eq__(self, other):
        return self.variable == other.variable \
               and self.mapping == other.mapping \
               and self.dtype == other.dtype

    def __hash__(self):
        return hash((type(self), self.variable, self.mapping, self.dtype))


@singledispatch
def column_str_repr(var: Orange.data.Variable, coldata: ndarray) -> ndarray:
    """Return a array of str representations of coldata for the `variable."""
    _f = np.frompyfunc(var.repr_val, 1, 1)
    return _f(coldata)


@column_str_repr.register(Orange.data.DiscreteVariable)
def column_str_repr_discrete(
        var: Orange.data.DiscreteVariable, coldata: ndarray
) -> ndarray:
    values = np.array(var.values, dtype=object)
    lookup = Lookup(var, values, "?")
    return lookup.transform(coldata)


@column_str_repr.register(Orange.data.StringVariable)
def column_str_repr_string(
        var: Orange.data.StringVariable, coldata: ndarray
) -> ndarray:
    return np.where(coldata == var.Unknown, "?", coldata)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWEditDomain).run(Orange.data.Table("iris"))
