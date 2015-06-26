import sys
import bisect
import copy
import contextlib

import pkg_resources

import numpy

from PyQt4.QtGui import (
    QWidget, QButtonGroup, QGroupBox, QRadioButton, QSlider,
    QDoubleSpinBox, QComboBox, QSpinBox, QListView,
    QVBoxLayout, QHBoxLayout, QFormLayout, QSpacerItem, QSizePolicy,
    QCursor, QIcon,  QStandardItemModel, QStandardItem, QStyle,
    QStylePainter, QStyleOptionFrame, QPixmap,
    QApplication, QDrag
)

from PyQt4 import QtGui
from PyQt4.QtCore import (
    Qt, QObject, QEvent, QSize, QModelIndex, QMimeData, QTimer
)
from PyQt4.QtCore import pyqtSignal as Signal, pyqtSlot as Slot


import Orange.data

from Orange import preprocess
from Orange.statistics import distribution
from Orange.preprocess import Continuize

from Orange.widgets import widget, gui, settings
from .owimpute import RandomTransform


@contextlib.contextmanager
def blocked(qobj):
    state = qobj.signalsBlocked()
    qobj.blockSignals(True)
    try:
        yield qobj
    finally:
        qobj.blockSignals(state)


class BaseEditor(QWidget):
    """
    Base widget for editing preprocessor's parameters.
    """
    #: Emitted when parameters have changed.
    changed = Signal()
    #: Emitted when parameters were edited/changed  as a result of
    #: user interaction.
    edited = Signal()

    def setParameters(self, parameters):
        """
        Set parameters.

        Parameters
        ----------
        parameters : dict
            Parameters as a dictionary. It is up to subclasses to
            properly parse the contents.

        """
        raise NotImplementedError

    def parameters(self):
        """Return the parameters as a dictionary.
        """
        raise NotImplementedError

    @staticmethod
    def createinstance(params):
        """
        Create the Preprocessor instance given the stored parameters dict.

        Parameters
        ----------
        params : dict
            Parameters as returned by `parameters`.
        """
        raise NotImplementedError


class _NoneDisc(preprocess.discretize.Discretization):
    """Discretize all variables into None.

    Used in combination with preprocess.Discretize to remove
    all discrete features from the domain.

    """
    def __call__(self, data, variable):
        return None


class DiscretizeEditor(BaseEditor):
    """
    Editor for preprocess.Discretize.
    """
    #: Discretize methods
    NoDisc, EqualWidth, EqualFreq, Drop, EntropyMDL = 0, 1, 2, 3, 4
    Discretizers = {
        NoDisc: (None, {}),
        EqualWidth: (preprocess.discretize.EqualWidth, {"n": 4}),
        EqualFreq:  (preprocess.discretize.EqualFreq, {"n": 4}),
        Drop: (_NoneDisc, {}),
        EntropyMDL: (preprocess.discretize.EntropyMDL, {"force": False})
    }
    Names = {
        NoDisc: "None",
        EqualWidth: "Equal width discretization",
        EqualFreq: "Equal frequency discretization",
        Drop: "Remove continuous attributes",
        EntropyMDL: "Entropy-MDL discretization"
    }

    def __init__(self, parent=None, **kwargs):
        BaseEditor.__init__(self, parent, **kwargs)
        self.__method = DiscretizeEditor.EqualFreq
        self.__nintervals = 4

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.__group = group = QButtonGroup(self, exclusive=True)

        for method in [self.EntropyMDL, self.EqualFreq, self.EqualWidth,
                       self.Drop]:
            rb = QRadioButton(
                self, text=self.Names[method],
                checked=self.__method == method
            )
            layout.addWidget(rb)
            group.addButton(rb, method)

        group.buttonClicked.connect(self.__on_buttonClicked)

        self.__slbox = slbox = QGroupBox(
            title="Number of intervals (for equal width/frequency",
            flat=True
        )
        slbox.setLayout(QVBoxLayout())
        self.__slider = slider = QSlider(
            orientation=Qt.Horizontal,
            minimum=2, maximum=10, value=self.__nintervals,
            enabled=self.__method in [self.EqualFreq, self.EqualWidth],
        )
        slider.valueChanged.connect(self.__on_valueChanged)
        slbox.layout().addWidget(slider)

        container = QHBoxLayout()
        container.setContentsMargins(13, 0, 0, 0)
        container.addWidget(slbox)
        self.layout().insertLayout(3, container)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

    def setMethod(self, method):
        if self.__method != method:
            self.__method = method
            b = self.__group.button(method)
            b.setChecked(True)
            self.__slider.setEnabled(
                method in [self.EqualFreq, self.EqualWidth]
            )
            self.changed.emit()

    def method(self):
        return self.__method

    def intervals(self):
        return self.__nintervals

    def setIntervals(self, n):
        n = numpy.clip(n, self.__slider.minimum(), self.__slider.maximum())
        n = int(n)
        if self.__nintervals != n:
            self.__nintervals = n
            # blocking signals in order to differentiate between
            # changed by user (notified through __on_valueChanged) or
            # changed programmatically (this)
            with blocked(self.__slider):
                self.__slider.setValue(n)
            self.changed.emit()

    def setParameters(self, params):
        method = params.get("method", self.EqualFreq)
        nintervals = params.get("n", 5)
        self.setMethod(method)
        if method in [self.EqualFreq, self.EqualWidth]:
            self.setIntervals(nintervals)

    def parameters(self):
        if self.__method in [self.EqualFreq, self.EqualWidth]:
            return {"method": self.__method, "n": self.__nintervals}
        else:
            return {"method": self.__method}

    def __on_buttonClicked(self):
        # on user 'method' button click
        method = self.__group.checkedId()
        if method != self.__method:
            self.setMethod(self.__group.checkedId())
            self.edited.emit()

    def __on_valueChanged(self):
        # on user n intervals slider change.
        self.__nintervals = self.__slider.value()
        self.changed.emit()
        self.edited.emit()

    @staticmethod
    def createinstance(params):
        params = dict(params)
        method = params.pop("method", DiscretizeEditor.EqualFreq)
        method, defaults = DiscretizeEditor.Discretizers[method]

        if method is None:
            return None

        resolved = dict(defaults)
        # update only keys in defaults?
        resolved.update(params)
        return preprocess.Discretize(method(**params))


class ContinuizeEditor(BaseEditor):
    Continuizers = [
        ("Most frequent is base", Continuize.FrequentAsBase),
        ("One attribute per value", Continuize.Indicators),
        ("Remove multinomial attributes", Continuize.RemoveMultinomial),
        ("Remove all discrete attributes", Continuize.Remove),
        ("Treat as ordinal", Continuize.AsOrdinal),
        ("Divide by number of values", Continuize.AsNormalizedOrdinal)]

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setLayout(QVBoxLayout())

        self.__treatment = Continuize.Indicators
        self.__group = group = QButtonGroup(exclusive=True)
        group.buttonClicked.connect(self.__on_buttonClicked)

        for text, treatment in ContinuizeEditor.Continuizers:
            rb = QRadioButton(
                text=text,
                checked=self.__treatment == treatment)
            group.addButton(rb, int(treatment))
            self.layout().addWidget(rb)

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

    def setTreatment(self, treatment):
        b = self.__group.button(treatment)
        if b is not None:
            b.setChecked(True)
            self.__treatment = treatment
            self.changed.emit()

    def treatment(self):
        return self.__treatment

    def setParameters(self, params):
        treatment = params.get("multinomial_treatment", Continuize.Indicators)
        self.setTreatment(treatment)

    def parameters(self):
        return {"multinomial_treatment": self.__treatment}

    def __on_buttonClicked(self):
        self.__treatment = self.__group.checkedId()
        self.changed.emit()
        self.edited.emit()

    @staticmethod
    def createinstance(params):
        params = dict(params)
        treatment = params.pop("multinomial_treatment", Continuize.Indicators)
        return Continuize(multinomial_treatment=treatment)


class _ImputeRandom:

    class ReplaceUnknownsSampleRandom(RandomTransform):
        def transform(self, column):
            mask = numpy.isnan(column)
            c = column[mask]
            if not c.size:
                return column
            else:
                c = super().transform(c)
                column = numpy.array(column)
                column[mask] = c
                return column

    def __call__(self, data, variable):
        dist = distribution.get_distribution(data, variable)
        return variable.copy(compute_value=self.ReplaceUnknownsSampleRandom(variable, dist))


class _RemoveNaNRows(preprocess.preprocess.Preprocess):
    def __call__(self, data):
        mask = numpy.isnan(data.X)
        mask = numpy.any(mask, axis=1)
        return data[~mask]


class ImputeEditor(BaseEditor):
    (NoImputation, Constant, Average,
     Model, Random, DropRows, DropColumns) = 0, 1, 2, 3, 4, 5, 6

    Imputers = {
        NoImputation: (None, {}),
#         Constant: (None, {"value": 0})
        Average: (preprocess.impute.Average(), {}),
#         Model: (preprocess.impute.Model, {}),
        Random: (_ImputeRandom(), {}),
        DropRows: (None, {})
    }
    Names = {
        NoImputation: "Don't impute.",
        Constant: "Replace with constant",
        Average: "Average/Most frequent",
        Model: "Model based imputer",
        Random: "Replace with random value",
        DropRows: "Remove rows with missing values.",
    }

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setLayout(QVBoxLayout())

        self.__method = ImputeEditor.Average
        self.__group = group = QButtonGroup(self, exclusive=True)
        group.buttonClicked.connect(self.__on_buttonClicked)

        for methodid in [self.Average, self.Random, self.DropRows]:
            text = self.Names[methodid]
            rb = QRadioButton(text=text, checked=self.__method == methodid)
            group.addButton(rb, methodid)
            self.layout().addWidget(rb)

    def setMethod(self, method):
        b = self.__group.button(method)
        if b is not None:
            b.setChecked(True)
            self.__method = method
            self.changed.emit()

    def setParameters(self, params):
        method = params.get("method", ImputeEditor.Average)
        self.setMethod(method)

    def parameters(self):
        return {"method": self.__method}

    def __on_buttonClicked(self):
        self.__method = self.__group.checkedId()
        self.changed.emit()
        self.edited.emit()

    @staticmethod
    def createinstance(params):
        params = dict(params)
        method = params.pop("method", ImputeEditor.Average)
        if method == ImputeEditor.NoImputation:
            return None
        elif method == ImputeEditor.Average:
            return preprocess.SklImpute()
        elif method == ImputeEditor.Model:
            return preprocess.Impute(method=preprocess.impute.Model())
        elif method == ImputeEditor.DropRows:
            return _RemoveNaNRows()
        elif method == ImputeEditor.DropColumns:
            return preprocess.RemoveNaNColumns()
        else:
            method, defaults = ImputeEditor.Imputers[method]
            defaults = dict(defaults)
            defaults.update(params)
            return preprocess.Impute(method=method)


class UnivariateFeatureSelect(QWidget):
    changed = Signal()
    edited = Signal()

    #: Strategy
    Fixed, Percentile, FDR, FPR, FWE = 1, 2, 3, 4, 5

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setLayout(QVBoxLayout())

        self.__scoreidx = 0
        self.__strategy = UnivariateFeatureSelect.Fixed
        self.__k = 10
        self.__p = 75.0

        box = QGroupBox(title="Score", flat=True)
        box.setLayout(QVBoxLayout())
        self.__cb = cb = QComboBox(self, )
        self.__cb.currentIndexChanged.connect(self.setScoreIndex)
        self.__cb.activated.connect(self.edited)
        box.layout().addWidget(cb)

        self.layout().addWidget(box)

        box = QGroupBox(title="Strategy", flat=True)
        self.__group = group = QButtonGroup(self, exclusive=True)
        self.__spins = {}

        form = QFormLayout()
        fixedrb = QRadioButton("Fixed", checked=True)
        group.addButton(fixedrb, UnivariateFeatureSelect.Fixed)
        kspin = QSpinBox(
            minimum=1, value=self.__k,
            enabled=self.__strategy == UnivariateFeatureSelect.Fixed
        )
        kspin.valueChanged[int].connect(self.setK)
        kspin.editingFinished.connect(self.edited)
        self.__spins[UnivariateFeatureSelect.Fixed] = kspin
        form.addRow(fixedrb, kspin)

        percrb = QRadioButton("Percentile")
        group.addButton(percrb, UnivariateFeatureSelect.Percentile)
        pspin = QDoubleSpinBox(
            minimum=0.0, maximum=100.0, singleStep=0.5,
            value=self.__p, suffix="%",
            enabled=self.__strategy == UnivariateFeatureSelect.Percentile
        )

        pspin.valueChanged[float].connect(self.setP)
        pspin.editingFinished.connect(self.edited)
        self.__spins[UnivariateFeatureSelect.Percentile] = pspin
        # Percentile controls disabled for now.
        pspin.setEnabled(False)
        percrb.setEnabled(False)
        form.addRow(percrb, pspin)

#         form.addRow(QRadioButton("FDR"), QDoubleSpinBox())
#         form.addRow(QRadioButton("FPR"), QDoubleSpinBox())
#         form.addRow(QRadioButton("FWE"), QDoubleSpinBox())

        self.__group.buttonClicked.connect(self.__on_buttonClicked)
        box.setLayout(form)
        self.layout().addWidget(box)

    def setScoreIndex(self, scoreindex):
        if self.__scoreidx != scoreindex:
            self.__scoreidx = scoreindex
            self.__cb.setCurrentIndex(scoreindex)
            self.changed.emit()

    def scoreIndex(self):
        return self.__scoreidx

    def setStrategy(self, strategy):
        if self.__strategy != strategy:
            self.__strategy = strategy
            b = self.__group.button(strategy)
            b.setChecked(True)
            for st, rb in self.__spins.items():
                rb.setEnabled(st == strategy)
            self.changed.emit()

    def setK(self, k):
        if self.__k != k:
            self.__k = k
            spin = self.__spins[UnivariateFeatureSelect.Fixed]
            spin.setValue(k)
            if self.__strategy == UnivariateFeatureSelect.Fixed:
                self.changed.emit()

    def setP(self, p):
        if self.__p != p:
            self.__p = p
            spin = self.__spins[UnivariateFeatureSelect.Percentile]
            spin.setValue(p)
            if self.__strategy == UnivariateFeatureSelect.Percentile:
                self.changed.emit()

    def setItems(self, itemlist):
        for item in itemlist:
            self.__cb.addItem(item["text"])

    def __on_buttonClicked(self):
        strategy = self.__group.checkedId()
        self.setStrategy(strategy)
        self.edited.emit()

    def setParameters(self, params):
        score = params.get("score", 0)
        strategy = params.get("strategy", UnivariateFeatureSelect.Fixed)
        self.setScoreIndex(score)
        self.setStrategy(strategy)
        if strategy == UnivariateFeatureSelect.Fixed:
            self.setK(params.get("k", 10))
        else:
            self.setP(params.get("p", 75))

    def parameters(self):
        score = self.__scoreidx
        strategy = self.__strategy
        p = self.__p
        k = self.__k

        return {"score": score, "strategy": strategy, "p": p, "k": k}


class FeatureSelectEditor(BaseEditor):

    MEASURES = [
        ("Information Gain", preprocess.score.InfoGain),
        ("Gain ratio", preprocess.score.GainRatio),
        ("Gini index", preprocess.score.Gini),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.__score = 0
        self.__selecionidx = 0

        self.__uni_fs = UnivariateFeatureSelect()
        self.__uni_fs.setItems(
            [{"text": "Information gain", "tooltip": ""},
             {"text": "Gain ratio"},
             {"text": "Gini index"}
            ]
        )
        self.layout().addWidget(self.__uni_fs)
        self.__uni_fs.changed.connect(self.changed)
        self.__uni_fs.edited.connect(self.edited)

    def setParameters(self, params):
        self.__uni_fs.setParameters(params)

    def parameters(self):
        return self.__uni_fs.parameters()

    @staticmethod
    def createinstance(params):
        params = dict(params)
        score = params.pop("score", 0)
        score = FeatureSelectEditor.MEASURES[score][1]
        strategy = params.get("strategy", UnivariateFeatureSelect.Fixed)
        k = params.get("k", 10)
        if strategy == UnivariateFeatureSelect.Fixed:
            return preprocess.fss.SelectBestFeatures(score, k=k)
        else:
            # TODO: implement top percentile selection
            raise NotImplementedError

# TODO: Model based FS (random forest variable importance, ...), RFE
# Unsupervised (min variance, constant, ...)??


class _Scaling(preprocess.preprocess.Preprocess):
    """
    Scale data preprocessor.
    """
    @staticmethod
    def mean(dist):
        values, counts = numpy.array(dist)
        return numpy.average(values, weights=counts)

    @staticmethod
    def median(dist):
        values, counts = numpy.array(dist)
        cumdist = numpy.cumsum(counts)
        if cumdist[-1] > 0:
            cumdist /= cumdist[-1]

        return numpy.interp(0.5, cumdist, values)

    @staticmethod
    def span(dist):
        values = numpy.array(dist[0])
        minval = numpy.min(values)
        maxval = numpy.max(values)
        return maxval - minval

    @staticmethod
    def std(dist):
        values, counts = numpy.array(dist)
        mean = numpy.average(values, weights=counts)
        diff = values - mean
        return numpy.sqrt(numpy.average(diff ** 2, weights=counts))

    def __init__(self, center=mean, scale=std):
        self.center = center
        self.scale = scale

    def __call__(self, data):
        if self.center is None and self.scale is None:
            return data

        def transform(var):
            dist = distribution.get_distribution(data, var)
            if self.center:
                c = self.center(dist)
                dist[0, :] -= c
            else:
                c = 0

            if self.scale:
                s = self.scale(dist)
                if s < 1e-15:
                    s = 1
            else:
                s = 1
            factor = 1 / s
            return var.copy(compute_value=preprocess.transformation.Normalizer(var, c, factor))

        newvars = []
        for var in data.domain.attributes:
            if var.is_continuous:
                newvars.append(transform(var))
            else:
                newvars.append(var)
        domain = Orange.data.Domain(newvars, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


class Scale(BaseEditor):
    NoCentering, CenterMean, CenterMedian = 0, 1, 2
    NoScaling, ScaleBySD, ScaleBySpan = 0, 1, 2

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setLayout(QVBoxLayout())

        form = QFormLayout()
        self.__centercb = QComboBox()
        self.__centercb.addItems(["No centering", "Center by mean",
                                  "Center by median"])

        self.__scalecb = QComboBox()
        self.__scalecb.addItems(["No scaling", "Scale by std",
                                 "Scale by span"])

        form.addRow("Center", self.__centercb)
        form.addRow("Scale", self.__scalecb)
        self.layout().addLayout(form)
        self.__centercb.currentIndexChanged.connect(self.changed)
        self.__scalecb.currentIndexChanged.connect(self.changed)
        self.__centercb.activated.connect(self.edited)
        self.__scalecb.activated.connect(self.edited)

    def setParameters(self, params):
        center = params.get("center", Scale.CenterMean)
        scale = params.get("scale", Scale.ScaleBySD)
        self.__centercb.setCurrentIndex(center)
        self.__scalecb.setCurrentIndex(scale)

    def parameters(self):
        return {"center": self.__centercb.currentIndex(),
                "scale": self.__scalecb.currentIndex()}

    @staticmethod
    def createinstance(params):
        center = params.get("center", Scale.CenterMean)
        scale = params.get("scale", Scale.ScaleBySD)

        if center == Scale.NoCentering:
            center = None
        elif center == Scale.CenterMean:
            center = _Scaling.mean
        elif center == Scale.CenterMedian:
            center = _Scaling.median
        else:
            assert False

        if scale == Scale.NoScaling:
            scale = None
        elif scale == Scale.ScaleBySD:
            scale = _Scaling.std
        elif scale == Scale.ScaleBySpan:
            scale = _Scaling.span
        else:
            assert False

        return _Scaling(center=center, scale=scale)


# This is intended for future improvements.
# I.e. it should be possible to add/register preprocessor actions
# through entry points (for use by add-ons). Maybe it should be a
# general framework (this is not the only place where such
# functionality is desired (for instance in Orange v2.* Rank widget
# already defines its own entry point).
class Description(object):
    """
    A description of an action/function.
    """
    def __init__(self, title, icon=None, summary=None, input=None, output=None,
                 requires=None, note=None, related=None, keywords=None,
                 helptopic=None):
        self.title = title
        self.icon = icon
        self.summary = summary
        self.input = input
        self.output = output
        self.requires = requires
        self.note = note
        self.related = related
        self.keywords = keywords
        self.helptopic = helptopic


class PreprocessAction(object):
    def __init__(self, name, qualname, category, description, viewclass):
        self.name = name
        self.qualname = qualname
        self.category = category
        self.description = description
        self.viewclass = viewclass


def icon_path(basename):
    return pkg_resources.resource_filename(__name__, "icons/" + basename)


PREPROCESSORS = [
    PreprocessAction(
        "Discretize", "orange.preprocess.discretize", "Discretization",
        Description("Discretize Continuous Variables",
                    icon_path("Discretize.svg")),
        DiscretizeEditor
    ),
    PreprocessAction(
        "Continuize", "orange.preprocess.continuize", "Continuization",
        Description("Continuize Discrete Variables",
                    icon_path("Continuize.svg")),
        ContinuizeEditor
    ),
    PreprocessAction(
        "Impute", "orange.preprocess.impute", "Impute",
        Description("Impute Missing Values",
                    icon_path("Impute.svg")),
        ImputeEditor
    ),
    PreprocessAction(
        "Feature Selection", "orange.preprocess.fss", "Feature Selection",
        Description("Select Relevant Features",
                    icon_path("SelectColumns.svg")),
        FeatureSelectEditor
    ),
    PreprocessAction(
        "Normalize", "orange.preprocess.scale", "Scaling",
        Description("Center and Scale Features",
                    icon_path("Continuize.svg")),
        Scale
    )
]

# TODO: Extend with entry points here
# PREPROCESSORS += iter_entry_points("Orange.widgets.data.owpreprocess")

# ####
# The actual owwidget (with helper classes)
# ####

# Note:
# The preprocessors are drag/dropped onto a sequence widget, where
# they can be reordered/removed/edited.
#
# Model <-> Adapter/Controller <-> View
#
# * `Model`: the current constructed preprocessor model.
# * the source (of drag/drop) is an item model displayed in a list
#   view (source list).
# * the drag/drop is controlled by the controller/adapter,

#: Qt.ItemRole holding the PreprocessAction instance
DescriptionRole = Qt.UserRole
#: Qt.ItemRole storing the preprocess parameters
ParametersRole = Qt.UserRole + 1


class Controller(QObject):
    """
    Controller for displaying/editing QAbstractItemModel using SequenceFlow.

    It creates/deletes updates the widgets in the view when the model
    changes, as well as interprets drop events (with appropriate mime data)
    onto the view, modifying the model appropriately.

    Parameters
    ----------
    view : SeqeunceFlow
        The view to control (required).
    model : QAbstarctItemModel
        A list model
    parent : QObject
        The controller's parent.
    """
    MimeType = "application/x-qwidget-ref"

    def __init__(self, view, model=None, parent=None):
        super().__init__(parent)
        self._model = None

        self.view = view
        view.installEventFilter(self)
        view.widgetCloseRequested.connect(self._closeRequested)
        view.widgetMoved.connect(self._widgetMoved)

        # gruesome
        self._setDropIndicatorAt = view._SequenceFlow__setDropIndicatorAt
        self._insertIndexAt = view._SequenceFlow__insertIndexAt

        if model is not None:
            self.setModel(model)

    def __connect(self, model):
        model.dataChanged.connect(self._dataChanged)
        model.rowsInserted.connect(self._rowsInserted)
        model.rowsRemoved.connect(self._rowsRemoved)
        model.rowsMoved.connect(self._rowsMoved)

    def __disconnect(self, model):
        model.dataChanged.disconnect(self._dataChanged)
        model.rowsInserted.disconnect(self._rowsInserted)
        model.rowsRemoved.disconnect(self._rowsRemoved)
        model.rowsMoved.disconnect(self._rowsMoved)

    def setModel(self, model):
        """Set the model for the view.

        :type model: QAbstarctItemModel.
        """
        if self._model is model:
            return

        if self._model is not None:
            self.__disconnect(self._model)

        self._clear()
        self._model = model

        if self._model is not None:
            self._initialize(model)
            self.__connect(model)

    def model(self):
        """Return the model.
        """
        return self._model

    def _initialize(self, model):
        for i in range(model.rowCount()):
            index = model.index(i, 0)
            self._insertWidgetFor(i, index)

    def _clear(self):
        self.view.clear()

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(self.MimeType) and \
                self.model() is not None:
            event.setDropAction(Qt.CopyAction)
            event.accept()
            return True
        else:
            return False

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(self.MimeType) and \
                self.model() is not None:
            event.accept()
            self._setDropIndicatorAt(event.pos())
            return True
        else:
            return False

    def dragLeaveEvent(self, event):
        return False
        # TODO: Remember if we have seen enter with the proper data
        # (leave event does not have mimeData)
#         if event.mimeData().hasFormat(self.MimeType) and \
#                 event.proposedAction() == Qt.CopyAction:
#             event.accept()
#             self._setDropIndicatorAt(None)
#             return True
#         else:
#             return False

    def dropEvent(self, event):
        if event.mimeData().hasFormat(self.MimeType) and \
                self.model() is not None:
            # Create and insert appropriate widget.
            self._setDropIndicatorAt(None)
            row = self._insertIndexAt(event.pos())
            model = self.model()

            diddrop = model.dropMimeData(
                event.mimeData(), Qt.CopyAction, row, 0, QModelIndex())

            if diddrop:
                event.accept()
            return True
        else:
            return False

    def eventFilter(self, view, event):
        if view is not self.view:
            return False

        if event.type() == QEvent.DragEnter:
            return self.dragEnterEvent(event)
        elif event.type() == QEvent.DragMove:
            return self.dragMoveEvent(event)
        elif event.type() == QEvent.DragLeave:
            return self.dragLeaveEvent(event)
        elif event.type() == QEvent.Drop:
            return self.dropEvent(event)
        else:
            return super().eventFilter(view, event)

    def _dataChanged(self, topleft, bottomright):
        model = self.model()
        widgets = self.view.widgets()

        top, left = topleft.row(), topleft.column()
        bottom, right = bottomright.row(), bottomright.column()
        assert left == 0 and right == 0

        for row in range(top, bottom + 1):
            self.setWidgetData(widgets[row], model.index(row, 0))

    def _rowsInserted(self, parent, start, end):
        model = self.model()
        for row in range(start, end + 1):
            index = model.index(row, 0, parent)
            self._insertWidgetFor(row, index)

    def _rowsRemoved(self, parent, start, end):
        for row in reversed(range(start, end + 1)):
            self._removeWidgetFor(row, None)

    def _rowsMoved(self, srcparetn, srcstart, srcend,
                   dstparent, dststart, dstend):
        raise NotImplementedError

    def _closeRequested(self, row):
        model = self.model()
        assert 0 <= row < model.rowCount()
        model.removeRows(row, 1, QModelIndex())

    def _widgetMoved(self, from_, to):
        # The widget in the view were already swaped, so
        # we must disconnect from the model when moving the rows.
        # It would be better if this class would also filter and
        # handle internal widget moves.
        model = self.model()
        self.__disconnect(model)
        try:
            model.moveRow
        except AttributeError:
            data = model.itemData(model.index(from_, 0))
            model.removeRow(from_, QModelIndex())
            model.insertRow(to, QModelIndex())

            model.setItemData(model.index(to, 0), data)
            assert model.rowCount() == len(self.view.widgets())
        else:
            model.moveRow(QModelIndex(), from_, QModelIndex(), to)
        finally:
            self.__connect(model)

    def _insertWidgetFor(self, row, index):
        widget = self.createWidgetFor(index)
        self.view.insertWidget(row, widget, title=index.data(Qt.DisplayRole))
        self.view.setIcon(row, index.data(Qt.DecorationRole))
        self.setWidgetData(widget, index)
        widget.edited.connect(self.__edited)

    def _removeWidgetFor(self, row, index):
        widget = self.view.widgets()[row]
        self.view.removeWidget(widget)
        widget.edited.disconnect(self.__edited)
        widget.deleteLater()

    def createWidgetFor(self, index):
        """
        Create a QWidget instance for the index (:class:`QModelIndex`)
        """
        definition = index.data(DescriptionRole)
        widget = definition.viewclass()
        return widget

    def setWidgetData(self, widget, index):
        """
        Set/update the widget state from the model at index.
        """
        params = index.data(ParametersRole)
        if not isinstance(params, dict):
            params = {}
        widget.setParameters(params)

    def setModelData(self, widget, index):
        """
        Get the data from the widget state and set/update the model at index.
        """
        params = widget.parameters()
        assert isinstance(params, dict)
        self._model.setData(index, params, ParametersRole)

    @Slot()
    def __edited(self,):
        widget = self.sender()
        row = self.view.indexOf(widget)
        index = self.model().index(row, 0)
        self.setModelData(widget, index)


class SequenceFlow(QWidget):
    """
    A re-orderable list of widgets.
    """
    #: Emitted when the user clicks the Close button in the header
    widgetCloseRequested = Signal(int)
    #: Emitted when the user moves/drags a widget to a new location.
    widgetMoved = Signal(int, int)

    class Frame(QtGui.QDockWidget):
        """
        Widget frame with a handle.
        """
        closeRequested = Signal()

        def __init__(self, parent=None, widget=None, title=None, **kwargs):

            super().__init__(parent, **kwargs)
            self.setFeatures(QtGui.QDockWidget.DockWidgetClosable)
            self.setAllowedAreas(Qt.NoDockWidgetArea)

            self.__title = ""
            self.__icon = ""
            self.__focusframe = None

            self.__deleteaction = QtGui.QAction(
                "Remove", self, shortcut=QtGui.QKeySequence.Delete,
                enabled=False, triggered=self.closeRequested
            )
            self.addAction(self.__deleteaction)

            if widget is not None:
                self.setWidget(widget)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

            if title:
                self.setTitle(title)

            self.setFocusPolicy(Qt.ClickFocus | Qt.TabFocus)

        def setTitle(self, title):
            if self.__title != title:
                self.__title = title
                self.setWindowTitle(title)
                self.update()

        def setIcon(self, icon):
            icon = QIcon(icon)
            if self.__icon != icon:
                self.__icon = icon
                self.setWindowIcon(icon)
                self.update()

        def paintEvent(self, event):
            painter = QStylePainter(self)
            opt = QStyleOptionFrame()
            opt.init(self)
            painter.drawPrimitive(QStyle.PE_FrameDockWidget, opt)
            painter.end()

            super().paintEvent(event)

        def focusInEvent(self, event):
            event.accept()
            self.__focusframe = QtGui.QFocusFrame(self)
            self.__focusframe.setWidget(self)
            self.__deleteaction.setEnabled(True)

        def focusOutEvent(self, event):
            event.accept()
            self.__focusframe.deleteLater()
            self.__focusframe = None
            self.__deleteaction.setEnabled(False)

        def closeEvent(self, event):
            super().closeEvent(event)
            event.ignore()
            self.closeRequested.emit()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__dropindicator = QSpacerItem(
            16, 16, QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.__dragstart = (None, None, None)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.__flowlayout = QVBoxLayout()
        layout.addLayout(self.__flowlayout)
        layout.addSpacerItem(
            QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Expanding))

        self.setLayout(layout)
        self.setAcceptDrops(True)

    def sizeHint(self):
        """Reimplemented."""
        if self.widgets():
            return super().sizeHint()
        else:
            return QSize(150, 100)

    def addWidget(self, widget, title):
        """Add `widget` with `title` to list of widgets (in the last position).

        Parameters
        ----------
        widget : QWidget
            Widget instance.
        title : str
            Widget title.
        """
        index = len(self.widgets())
        self.insertWidget(index, widget, title)

    def insertWidget(self, index, widget, title):
        """Insert `widget` with `title` at `index`.

        Parameters
        ----------
        index : int
            Position at which the widget should be inserted.
        widget : QWidget
            Widget instance.
        title : str
            Widget title.
        """
        # TODO: Check if widget is already inserted.
        frame = SequenceFlow.Frame(widget=widget, title=title)
        frame.closeRequested.connect(self.__closeRequested)

        layout = self.__flowlayout

        frames = [item.widget() for item in self.layout_iter(layout)
                  if item.widget()]

        if 0 < index < len(frames):
            # find the layout index of a widget occupying the current
            # index'th slot.
            insert_index = layout.indexOf(frames[index])
        elif index == 0:
            insert_index = 0
        elif index < 0 or index >= len(frames):
            insert_index = layout.count()
        else:
            assert False

        layout.insertWidget(insert_index, frame)

        frame.installEventFilter(self)

    def removeWidget(self, widget):
        """Remove widget from the list.

        Parameters
        ----------
        widget : QWidget
            Widget instance to remove.
        """
        layout = self.__flowlayout
        frame = self.__widgetFrame(widget)
        if frame is not None:
            frame.setWidget(None)
            widget.setVisible(False)
            widget.setParent(None)
            layout.takeAt(layout.indexOf(frame))
            frame.hide()
            frame.deleteLater()

    def clear(self):
        """Clear the list (remove all widgets).
        """
        for w in reversed(self.widgets()):
            self.removeWidget(w)

    def widgets(self):
        """Return a list of all `widgets`.
        """
        layout = self.__flowlayout
        items = (layout.itemAt(i) for i in range(layout.count()))
        return [item.widget().widget()
                for item in items if item.widget() is not None]

    def indexOf(self, widget):
        """Return the index (logical position) of `widget`
        """
        widgets = self.widgets()
        return widgets.index(widget)

    def setTitle(self, index, title):
        """Set title for `widget` at `index`.
        """
        widget = self.widgets()[index]
        frame = self.__widgetFrame(widget)
        frame.setTitle(title)

    def setIcon(self, index, icon):
        widget = self.widgets()[index]
        frame = self.__widgetFrame(widget)
        frame.setIcon(icon)

    def dropEvent(self, event):
        """Reimplemented."""
        layout = self.__flowlayout
        index = self.__insertIndexAt(self.mapFromGlobal(QCursor.pos()))

        if event.mimeData().hasFormat("application/x-internal-move") and \
                event.source() is self:
            # Complete the internal move
            frame, oldindex, _ = self.__dragstart
            # Remove the drop indicator spacer item before re-inserting
            # the frame
            self.__setDropIndicatorAt(None)
            if oldindex != index:
                layout.insertWidget(index, frame)
                if index > oldindex:
                    self.widgetMoved.emit(oldindex, index - 1)
                else:
                    self.widgetMoved.emit(oldindex, index)

                event.accept()

            self.__dragstart = None, None, None

    def dragEnterEvent(self, event):
        """Reimplemented."""
        if event.mimeData().hasFormat("application/x-internal-move") and \
                event.source() is self:
            assert self.__dragstart[0] is not None
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        """Reimplemented."""
        pos = self.mapFromGlobal(QCursor.pos())
        self.__setDropIndicatorAt(pos)

    def dragLeaveEvent(self, event):
        """Reimplemented."""
        self.__setDropIndicatorAt(None)

    def eventFilter(self, obj, event):
        """Reimplemented."""
        if isinstance(obj, SequenceFlow.Frame) and obj.parent() is self:
            etype = event.type()
            if etype == QEvent.MouseButtonPress and \
                    event.button() == Qt.LeftButton:
                # Is the mouse press on the dock title bar
                # (assume everything above obj.widget is a title bar)
                # TODO: Get the proper title bar geometry.
                if event.pos().y() < obj.widget().y():
                    index = self.indexOf(obj.widget())
                    self.__dragstart = (obj, index, event.pos())
            elif etype == QEvent.MouseMove and \
                    event.buttons() & Qt.LeftButton and \
                    obj is self.__dragstart[0]:
                _, _, down = self.__dragstart
                if (down - event.pos()).manhattanLength() >= \
                        QApplication.startDragDistance():
                    self.__startInternalDrag(obj, event.pos())
                    self.__dragstart = None, None, None
                    return True
            elif etype == QEvent.MouseButtonRelease and \
                    event.button() == Qt.LeftButton and \
                    self.__dragstart[0] is obj:
                self.__dragstart = None, None, None

        return super().eventFilter(obj, event)

    def __setDropIndicatorAt(self, pos):
        # find the index where drop at pos would insert.
        index = -1
        layout = self.__flowlayout
        if pos is not None:
            index = self.__insertIndexAt(pos)
        spacer = self.__dropindicator
        currentindex = self.layout_index_of(layout, spacer)

        if currentindex != -1:
            item = layout.takeAt(currentindex)
            assert item is spacer
            if currentindex < index:
                index -= 1

        if index != -1:
            layout.insertItem(index, spacer)

    def __insertIndexAt(self, pos):
        y = pos.y()
        midpoints = [item.widget().geometry().center().y()
                     for item in self.layout_iter(self.__flowlayout)
                     if item.widget() is not None]
        index = bisect.bisect_left(midpoints, y)
        return index

    def __startInternalDrag(self, frame, hotSpot=None):
        drag = QDrag(self)
        pixmap = QPixmap(frame.size())
        frame.render(pixmap)

        transparent = QPixmap(pixmap.size())
        transparent.fill(Qt.transparent)
        painter = QtGui.QPainter(transparent)
        painter.setOpacity(0.35)
        painter.drawPixmap(0, 0, pixmap.width(), pixmap.height(), pixmap)
        painter.end()

        drag.setPixmap(transparent)
        if hotSpot is not None:
            drag.setHotSpot(hotSpot)
        mime = QMimeData()
        mime.setData("application/x-internal-move", "")
        drag.setMimeData(mime)
        return drag.exec_(Qt.MoveAction)

    def __widgetFrame(self, widget):
        layout = self.__flowlayout
        for item in self.layout_iter(layout):
            if item.widget() is not None and \
                    isinstance(item.widget(), SequenceFlow.Frame) and \
                    item.widget().widget() is widget:
                return item.widget()
        else:
            return None

    def __closeRequested(self):
        frame = self.sender()
        index = self.indexOf(frame.widget())
        self.widgetCloseRequested.emit(index)

    @staticmethod
    def layout_iter(layout):
        return (layout.itemAt(i) for i in range(layout.count()))

    @staticmethod
    def layout_index_of(layout, item):
        for i, item1 in enumerate(SequenceFlow.layout_iter(layout)):
            if item == item1:
                return i
        return -1


class OWPreprocess(widget.OWWidget):
    name = "Preprocess"
    description = "Construct a data preprocessing pipeline."
    icon = "icons/Preprocess.svg"
    priority = 2105

    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Preprocessor", preprocess.preprocess.Preprocess),
               ("Preprocessed Data", Orange.data.Table)]

    storedsettings = settings.Setting({})
    autocommit = settings.Setting(False)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None
        self._invalidated = False

        # List of available preprocessors (DescriptionRole : Description)
        self.preprocessors = QStandardItemModel()

        def mimeData(indexlist):
            assert len(indexlist) == 1
            index = indexlist[0]
            qname = index.data(DescriptionRole).qualname
            m = QMimeData()
            m.setData("application/x-qwidget-ref", qname)
            return m
        # TODO: Fix this (subclass even if just to pass a function
        # for mimeData delegate)
        self.preprocessors.mimeData = mimeData

        box = gui.widgetBox(self.controlArea, "Preprocessors")

        self.preprocessorsView = view = QListView(
            selectionMode=QListView.SingleSelection,
            dragEnabled=True,
            dragDropMode=QListView.DragOnly
        )
        view.setModel(self.preprocessors)
        view.activated.connect(self.__activated)

        box.layout().addWidget(view)

        ####
        self._qname2ppdef = {ppdef.qualname: ppdef for ppdef in PREPROCESSORS}

        # List of 'selected' preprocessors and their parameters.
        self.preprocessormodel = None

        self.flow_view = SequenceFlow()
        self.controler = Controller(self.flow_view, parent=self)

        self.scroll_area = QtGui.QScrollArea(
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn
        )
        self.scroll_area.viewport().setAcceptDrops(True)
        self.scroll_area.setWidget(self.flow_view)
        self.scroll_area.setWidgetResizable(True)
        self.mainArea.layout().addWidget(self.scroll_area)
        self.flow_view.installEventFilter(self)

        box = gui.widgetBox(self.controlArea, "Output")
        gui.auto_commit(box, self, "autocommit", "Commit", box=False)

        self._initialize()

    def _initialize(self):
        for pp_def in PREPROCESSORS:
            description = pp_def.description
            if description.icon:
                icon = QIcon(description.icon)
            else:
                icon = QIcon()
            item = QStandardItem(icon, description.title)
            item.setToolTip(description.summary or "")
            item.setData(pp_def, DescriptionRole)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable |
                          Qt.ItemIsDragEnabled)
            self.preprocessors.appendRow([item])

        try:
            model = self.load(self.storedsettings)
        except Exception:
            model = self.load({})

        self.set_model(model)

        if not model.rowCount():
            # enforce default width constraint if no preprocessors
            # are instantiated (if the model is not empty the constraints
            # will be triggered by LayoutRequest event on the `flow_view`)
            self.__update_size_constraint()

    def load(self, saved):
        """Load a preprocessor list from a dict."""
        name = saved.get("name", "")
        preprocessors = saved.get("preprocessors", [])
        model = QStandardItemModel()

        def dropMimeData(data, action, row, column, parent):
            if data.hasFormat("application/x-qwidget-ref") and \
                    action == Qt.CopyAction:
                qname = bytes(data.data("application/x-qwidget-ref")).decode()

                ppdef = self._qname2ppdef[qname]
                item = QStandardItem(ppdef.description.title)
                item.setData({}, ParametersRole)
                item.setData(ppdef.description.title, Qt.DisplayRole)
                item.setData(ppdef, DescriptionRole)
                self.preprocessormodel.insertRow(row, [item])
                return True
            else:
                return False

        model.dropMimeData = dropMimeData

        for qualname, params in preprocessors:
            pp_def = self._qname2ppdef[qualname]
            description = pp_def.description
            item = QStandardItem(description.title)
            if description.icon:
                icon = QIcon(description.icon)
            else:
                icon = QIcon()
            item.setIcon(icon)
            item.setToolTip(description.summary)
            item.setData(pp_def, DescriptionRole)
            item.setData(params, ParametersRole)

            model.appendRow(item)
        return model

    def save(self, model):
        """Save the preprocessor list to a dict."""
        d = {"name": ""}
        preprocessors = []
        for i in range(model.rowCount()):
            item = model.item(i)
            pp_def = item.data(DescriptionRole)
            params = item.data(ParametersRole)
            preprocessors.append((pp_def.qualname, params))

        d["preprocessors"] = preprocessors
        return d

    def set_model(self, ppmodel):
        if self.preprocessormodel:
            self.preprocessormodel.dataChanged.disconnect(self.commit)
            self.preprocessormodel.rowsInserted.disconnect(self.commit)
            self.preprocessormodel.rowsRemoved.disconnect(self.commit)
            self.preprocessormodel.deleteLater()

        self.preprocessormodel = ppmodel
        self.controler.setModel(ppmodel)
        if ppmodel is not None:
            self.preprocessormodel.dataChanged.connect(self.commit)
            self.preprocessormodel.rowsInserted.connect(self.commit)
            self.preprocessormodel.rowsRemoved.connect(self.commit)

    def set_data(self, data=None):
        """Set the input data set."""
        self.data = data

    def handleNewSignals(self):
        self.apply()

    def __activated(self, index):
        item = self.preprocessors.itemFromIndex(index)
        action = item.data(DescriptionRole)
        item = QStandardItem()
        item.setData({}, ParametersRole)
        item.setData(action.description.title, Qt.DisplayRole)
        item.setData(action, DescriptionRole)
        self.preprocessormodel.appendRow([item])

    def buildpreproc(self):
        plist = []
        for i in range(self.preprocessormodel.rowCount()):
            item = self.preprocessormodel.item(i)
            desc = item.data(DescriptionRole)
            params = item.data(ParametersRole)

            if not isinstance(params, dict):
                params = {}

            create = desc.viewclass.createinstance
            plist.append(create(params))

        if len(plist) == 1:
            return plist[0]
        else:
            return preprocess.preprocess.PreprocessorList(plist)

    def apply(self):
        preprocessor = self.buildpreproc()
        if self.data is not None:
            data = preprocessor(self.data)
        else:
            data = None

        self.send("Preprocessor", preprocessor)
        self.send("Preprocessed Data", data)

    def commit(self):
        # Sync the model into storedsettings on every change commit.
        self.storeSpecificSettings()
        if not self._invalidated:
            self._invalidated = True
            QApplication.postEvent(self, QEvent(QEvent.User))

    def customEvent(self, event):
        if event.type() == QEvent.User and self._invalidated:
            self._invalidated = False
            self.apply()

    def eventFilter(self, receiver, event):
        if receiver is self.flow_view and event.type() == QEvent.LayoutRequest:
            QTimer.singleShot(0, self.__update_size_constraint)

        return super().eventFilter(receiver, event)

    def storeSpecificSettings(self):
        """Reimplemented."""
        self.storedsettings = self.save(self.preprocessormodel)
        super().storeSpecificSettings()

    def saveSettings(self):
        """Reimplemented."""
        self.storedsettings = self.save(self.preprocessormodel)
        super().saveSettings()

    def onDeleteWidget(self):
        self.data = None
        self.set_model(None)
        super().onDeleteWidget()

    @Slot()
    def __update_size_constraint(self):
        # Update minimum width constraint on the scroll area containing
        # the 'instantiated' preprocessor list (to avoid the horizontal
        # scroll bar).
        sh = self.flow_view.minimumSizeHint()
        scroll_width = self.scroll_area.verticalScrollBar().width()
        self.scroll_area.setMinimumWidth(
            min(max(sh.width() + scroll_width + 2, self.controlArea.width()),
                520))


def test_main(argv=sys.argv):
    argv = list(argv)
    app = QtGui.QApplication(argv)

    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "brown-selected"

    w = OWPreprocess()
    w.set_data(Orange.data.Table(filename))
    w.show()
    w.raise_()
    r = app.exec_()
    w.saveSettings()
    w.onDeleteWidget()
    return r

if __name__ == "__main__":
    sys.exit(test_main())
