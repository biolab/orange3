import sys
import bisect
import contextlib
import warnings
from collections import OrderedDict
import pkg_resources

import numpy
from PyQt4.QtGui import (
    QWidget, QButtonGroup, QGroupBox, QRadioButton, QSlider,
    QDoubleSpinBox, QComboBox, QSpinBox, QListView,
    QVBoxLayout, QHBoxLayout, QFormLayout, QSpacerItem, QSizePolicy,
    QCursor, QIcon,  QStandardItemModel, QStandardItem, QStyle,
    QStylePainter, QStyleOptionFrame, QPixmap,
    QApplication, QDrag, QLabel
)
from PyQt4 import QtGui
from PyQt4.QtCore import (
    Qt, QObject, QEvent, QSize, QModelIndex, QMimeData, QTimer
)
from PyQt4.QtCore import pyqtSignal as Signal, pyqtSlot as Slot


import Orange.data
from Orange import preprocess
from Orange.preprocess import Continuize, ProjectPCA, \
    ProjectCUR, Scaling, Randomize as _Randomize
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.overlay import OverlayWidget
from Orange.widgets.utils.sql import check_sql_input


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

    def __repr__(self):
        return "Orange.widgets.data.owpreprocess._NoneDisc()"


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
            title="Number of intervals (for equal width/frequency)",
            flat=True
        )
        slbox.setLayout(QHBoxLayout())
        self.__slider = slider = QSlider(
            orientation=Qt.Horizontal,
            minimum=2, maximum=10, value=self.__nintervals,
            enabled=self.__method in [self.EqualFreq, self.EqualWidth],
            pageStep=1, tickPosition=QSlider.TicksBelow
        )
        slider.valueChanged.connect(self.__on_valueChanged)
        slbox.layout().addWidget(slider)
        self.__slabel = slabel = QLabel()
        slbox.layout().addWidget(slabel)

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
                self.__slabel.setText(str(self.__slider.value()))
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
        self.__slabel.setText(str(self.__slider.value()))

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
        return preprocess.Discretize(method(**params), remove_const=False)

    def __repr__(self):
        n_int = ", Number of intervals: {}".format(self.__nintervals) \
            if self.__method in [self.EqualFreq, self.EqualWidth] else ""
        return "{}{}".format(self.Names[self.__method], n_int)


class ContinuizeEditor(BaseEditor):
    Continuizers = OrderedDict({
        Continuize.FrequentAsBase: "Most frequent is base",
        Continuize.Indicators: "One attribute per value",
        Continuize.RemoveMultinomial: "Remove multinomial attributes",
        Continuize.Remove: "Remove all discrete attributes",
        Continuize.AsOrdinal: "Treat as ordinal",
        Continuize.AsNormalizedOrdinal: "Divide by number of values"})

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setLayout(QVBoxLayout())

        self.__treatment = Continuize.Indicators
        self.__group = group = QButtonGroup(exclusive=True)
        group.buttonClicked.connect(self.__on_buttonClicked)

        for treatment, text in ContinuizeEditor.Continuizers.items():
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

    def __repr__(self):
        return self.Continuizers[self.__treatment]


class _RemoveNaNRows(preprocess.preprocess.Preprocess):
    def __call__(self, data):
        mask = numpy.isnan(data.X)
        mask = numpy.any(mask, axis=1)
        return data[~mask]

    def __repr__(self):
        return "Orange.widgets.data.owpreprocess._RemoveNaNRows()"


class ImputeEditor(BaseEditor):
    (NoImputation, Constant, Average,
     Model, Random, DropRows, DropColumns) = 0, 1, 2, 3, 4, 5, 6

    Imputers = {
        NoImputation: (None, {}),
#         Constant: (None, {"value": 0})
        Average: (preprocess.impute.Average(), {}),
#         Model: (preprocess.impute.Model, {}),
        Random: (preprocess.impute.Random(), {}),
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
            return preprocess.Impute()
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

    def __repr__(self):
        return self.Names[self.__method]


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
        fixedrb = QRadioButton("Fixed:", checked=True)
        group.addButton(fixedrb, UnivariateFeatureSelect.Fixed)
        kspin = QSpinBox(
            minimum=1, value=self.__k,
            enabled=self.__strategy == UnivariateFeatureSelect.Fixed
        )
        kspin.valueChanged[int].connect(self.setK)
        kspin.editingFinished.connect(self.edited)
        self.__spins[UnivariateFeatureSelect.Fixed] = kspin
        form.addRow(fixedrb, kspin)

        percrb = QRadioButton("Percentile:")
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
        ("ReliefF", preprocess.score.ReliefF),
        ("Fast Correlation Based Filter", preprocess.score.FCBF)
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.__score = 0
        self.__selecionidx = 0

        self.__uni_fs = UnivariateFeatureSelect()
        self.__uni_fs.setItems(
            [{"text": "Information Gain", "tooltip": ""},
             {"text": "Gain Ratio"},
             {"text": "Gini Index"},
             {"text": "ReliefF"},
             {"text": "Fast Correlation Based Filter"}
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

    def __repr__(self):
        params = self.__uni_fs.parameters()
        return "Score: {}, Strategy (Fixed): {}".format(
            self.MEASURES[params["score"]][0], params["k"])

# TODO: Model based FS (random forest variable importance, ...), RFE
# Unsupervised (min variance, constant, ...)??

class RandomFeatureSelectEditor(BaseEditor):
    #: Strategy
    Fixed, Percentage = 1, 2

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setLayout(QVBoxLayout())

        self.__strategy = RandomFeatureSelectEditor.Fixed
        self.__k = 10
        self.__p = 75.0

        box = QGroupBox(title="Strategy", flat=True)
        self.__group = group = QButtonGroup(self, exclusive=True)
        self.__spins = {}

        form = QFormLayout()
        fixedrb = QRadioButton("Fixed", checked=True)
        group.addButton(fixedrb, RandomFeatureSelectEditor.Fixed)
        kspin = QSpinBox(
            minimum=1, value=self.__k,
            enabled=self.__strategy == RandomFeatureSelectEditor.Fixed
        )
        kspin.valueChanged[int].connect(self.setK)
        kspin.editingFinished.connect(self.edited)
        self.__spins[RandomFeatureSelectEditor.Fixed] = kspin
        form.addRow(fixedrb, kspin)

        percrb = QRadioButton("Percentage")
        group.addButton(percrb, RandomFeatureSelectEditor.Percentage)
        pspin = QDoubleSpinBox(
            minimum=0.0, maximum=100.0, singleStep=0.5,
            value=self.__p, suffix="%",
            enabled=self.__strategy == RandomFeatureSelectEditor.Percentage
        )
        pspin.valueChanged[float].connect(self.setP)
        pspin.editingFinished.connect(self.edited)
        self.__spins[RandomFeatureSelectEditor.Percentage] = pspin
        form.addRow(percrb, pspin)

        self.__group.buttonClicked.connect(self.__on_buttonClicked)
        box.setLayout(form)
        self.layout().addWidget(box)

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
            spin = self.__spins[RandomFeatureSelectEditor.Fixed]
            spin.setValue(k)
            if self.__strategy == RandomFeatureSelectEditor.Fixed:
                self.changed.emit()

    def setP(self, p):
        if self.__p != p:
            self.__p = p
            spin = self.__spins[RandomFeatureSelectEditor.Percentage]
            spin.setValue(p)
            if self.__strategy == RandomFeatureSelectEditor.Percentage:
                self.changed.emit()

    def __on_buttonClicked(self):
        strategy = self.__group.checkedId()
        self.setStrategy(strategy)
        self.edited.emit()

    def setParameters(self, params):
        strategy = params.get("strategy", RandomFeatureSelectEditor.Fixed)
        self.setStrategy(strategy)
        if strategy == RandomFeatureSelectEditor.Fixed:
            self.setK(params.get("k", 10))
        else:
            self.setP(params.get("p", 75.0))

    def parameters(self):
        strategy = self.__strategy
        p = self.__p
        k = self.__k

        return {"strategy": strategy, "p": p, "k": k}

    @staticmethod
    def createinstance(params):
        params = dict(params)
        strategy = params.get("strategy", RandomFeatureSelectEditor.Fixed)
        k = params.get("k", 10)
        p = params.get("p", 75.0)
        if strategy == RandomFeatureSelectEditor.Fixed:
            return preprocess.fss.SelectRandomFeatures(k=k)
        elif strategy == RandomFeatureSelectEditor.Percentage:
            return preprocess.fss.SelectRandomFeatures(k=p/100)
        else:
            # further implementations
            raise NotImplementedError


class Scale(BaseEditor):
    NoCentering, CenterMean, CenterMedian = 0, 1, 2
    NoScaling, ScaleBySD, ScaleBySpan = 0, 1, 2

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setLayout(QVBoxLayout())

        form = QFormLayout()
        self.__centercb = QComboBox()
        self.__centercb.addItems(["No Centering", "Center by Mean",
                                  "Center by Median"])

        self.__scalecb = QComboBox()
        self.__scalecb.addItems(["No scaling", "Scale by SD",
                                 "Scale by span"])

        form.addRow("Center:", self.__centercb)
        form.addRow("Scale:", self.__scalecb)
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
            center = Scaling.mean
        elif center == Scale.CenterMedian:
            center = Scaling.median
        else:
            assert False

        if scale == Scale.NoScaling:
            scale = None
        elif scale == Scale.ScaleBySD:
            scale = Scaling.std
        elif scale == Scale.ScaleBySpan:
            scale = Scaling.span
        else:
            assert False

        return Scaling(center=center, scale=scale)

    def __repr__(self):
        return "{}, {}".format(self.__centercb.currentText(),
                               self.__scalecb.currentText())


class Randomize(BaseEditor):
    RandomizeClasses, RandomizeAttributes, RandomizeMetas = _Randomize.RandTypes

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setLayout(QVBoxLayout())

        form = QFormLayout()
        self.__rand_type_cb = QComboBox()
        self.__rand_type_cb.addItems(["Classes",
                                      "Features",
                                      "Meta data"])

        form.addRow("Randomize:", self.__rand_type_cb)
        self.layout().addLayout(form)
        self.__rand_type_cb.currentIndexChanged.connect(self.changed)
        self.__rand_type_cb.activated.connect(self.edited)

    def setParameters(self, params):
        rand_type = params.get("rand_type", Randomize.RandomizeClasses)
        self.__rand_type_cb.setCurrentIndex(rand_type)

    def parameters(self):
        return {"rand_type": self.__rand_type_cb.currentIndex()}

    @staticmethod
    def createinstance(params):
        rand_type = params.get("rand_type", Randomize.RandomizeClasses)
        return _Randomize(rand_type=rand_type)

    def __repr__(self):
        return self.__rand_type_cb.currentText()


class PCA(BaseEditor):

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setLayout(QVBoxLayout())

        self.n_components = 10

        form = QFormLayout()
        self.cspin = QSpinBox(minimum=1, value=self.n_components)
        self.cspin.valueChanged[int].connect(self.setC)
        self.cspin.editingFinished.connect(self.edited)

        form.addRow("Components:", self.cspin)
        self.layout().addLayout(form)

    def setParameters(self, params):
        self.n_components = params.get("n_components", 10)

    def parameters(self):
        return {"n_components": self.n_components}

    def setC(self, n_components):
        if self.n_components != n_components:
            self.n_components = n_components
            self.cspin.setValue(n_components)
            self.changed.emit()

    @staticmethod
    def createinstance(params):
        n_components = params.get("n_components", 10)
        return ProjectPCA(n_components=n_components)

    def __repr__(self):
        return "Components: {}".format(self.cspin.value())


class CUR(BaseEditor):

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setLayout(QVBoxLayout())

        self.rank = 10
        self.max_error = 1

        form = QFormLayout()
        self.rspin = QSpinBox(minimum=2, value=self.rank)
        self.rspin.valueChanged[int].connect(self.setR)
        self.rspin.editingFinished.connect(self.edited)
        self.espin = QDoubleSpinBox(
            minimum=0.1, maximum=100.0, singleStep=0.1,
            value=self.max_error)
        self.espin.valueChanged[float].connect(self.setE)
        self.espin.editingFinished.connect(self.edited)

        form.addRow("Rank:", self.rspin)
        form.addRow("Relative error:", self.espin)
        self.layout().addLayout(form)

    def setParameters(self, params):
        self.setR(params.get("rank", 10))
        self.setE(params.get("max_error", 1))

    def parameters(self):
        return {"rank": self.rank, "max_error": self.max_error}

    def setR(self, rank):
        if self.rank != rank:
            self.rank = rank
            self.rspin.setValue(rank)
            self.changed.emit()

    def setE(self, max_error):
        if self.max_error != max_error:
            self.max_error = max_error
            self.espin.setValue(max_error)
            self.changed.emit()

    @staticmethod
    def createinstance(params):
        rank = params.get("rank", 10)
        max_error = params.get("max_error", 1)
        return ProjectCUR(rank=rank, max_error=max_error)

    def __repr__(self):
        return "Rank: {}, Relative error: {}".format(self.rspin.value(),
                                                     self.espin.value())


# This is intended for future improvements.
# I.e. it should be possible to add/register preprocessor actions
# through entry points (for use by add-ons). Maybe it should be a
# general framework (this is not the only place where such
# functionality is desired (for instance in Orange v2.* Rank widget
# already defines its own entry point).
class Description:
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


class PreprocessAction:
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
        "Random Feature Selection", "orange.preprocess.randomfss",
        "Random Feature Selection",
        Description("Select Random Features",
                    icon_path("SelectColumnsRandom.svg")),
        RandomFeatureSelectEditor
    ),
    PreprocessAction(
        "Normalize", "orange.preprocess.scale", "Scaling",
        Description("Normalize Features",
                    icon_path("Normalize.svg")),
        Scale
    ),
    PreprocessAction(
        "Randomize", "orange.preprocess.randomize", "Randomization",
        Description("Randomize",
                    icon_path("Random.svg")),
        Randomize
    ),
    PreprocessAction(
        "PCA", "orange.preprocess.pca", "PCA",
        Description("Principal Component Analysis",
                    icon_path("PCA.svg")),
        PCA
    ),
    PreprocessAction(
        "CUR", "orange.preprocess.cur", "CUR",
        Description("CUR Matrix Decomposition",
                    icon_path("SelectColumns.svg")),
        CUR
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


def list_model_move_row_helper(model, parent, src, dst):
    assert src != dst and src != dst - 1
    data = model.itemData(model.index(src, 0, parent))
    removed = model.removeRow(src, parent)
    if not removed:
        return False

    realdst = dst - 1 if dst > src else dst
    inserted = model.insertRow(realdst, parent)
    if not inserted:
        return False

    dataset = model.setItemData(model.index(realdst, 0, parent), data)

    return removed and inserted and dataset


def list_model_move_rows_helper(model, parent, src, count, dst):
    assert not (src <= dst < src + count + 1)
    rowdata = [model.itemData(model.index(src + i, 0, parent))
               for i in range(count)]
    removed = model.removeRows(src, count, parent)
    if not removed:
        return False

    realdst = dst - count if dst > src else dst
    inserted = model.insertRows(realdst, count, parent)
    if not inserted:
        return False

    setdata = True
    for i, data in enumerate(rowdata):
        didset = model.setItemData(model.index(realdst + i, 0, parent), data)
        setdata = setdata and didset
    return setdata


class StandardItemModel(QtGui.QStandardItemModel):
    """
    A QStandardItemModel improving support for internal row moves.

    The QStandardItemModel is missing support for explicitly moving
    rows internally. Therefore to move a row it is first removed
    reinserted as an empty row and it's data repopulated.
    This triggers rowsRemoved/rowsInserted and dataChanged signals.
    If an observer is monitoring the model state it would see all the model
    changes. By using moveRow[s] only one `rowsMoved` signal is emitted
    coalescing all the updates.

    .. note:: The semantics follow Qt5's QAbstractItemModel.moveRow[s]

    """

    def moveRow(self, sourceParent, sourceRow, destParent, destRow):
        """
        Move sourceRow from sourceParent to destinationRow under destParent.

        Returns True if the row was successfully moved; otherwise
        returns false.

        .. note:: Only moves within the same parent are currently supported

        """
        if not sourceParent == destParent:
            return False

        if not self.beginMoveRows(sourceParent, sourceRow, sourceRow,
                                  destParent, destRow):
            return False

        # block so rowsRemoved/Inserted and dataChanged signals
        # are not emitted during the move. Instead the rowsMoved
        # signal will be emitted from self.endMoveRows().
        # I am mostly sure this is safe (a possible problem would be if the
        # base class itself would connect to the rowsInserted, ... to monitor
        # ensure internal invariants)
        with blocked(self):
            didmove = list_model_move_row_helper(
                self, sourceParent, sourceRow, destRow)
        self.endMoveRows()

        if not didmove:
            warnings.warn(
                "`moveRow` did not succeed! Data model might be "
                "in an inconsistent state.",
                RuntimeWarning)
        return didmove

    def moveRows(self, sourceParent, sourceRow, count,
                 destParent, destRow):
        """
        Move count rows starting with the given sourceRow under parent
        sourceParent to row destRow under parent destParent.

        Return true if the rows were successfully moved; otherwise
        returns false.

        .. note:: Only moves within the same parent are currently supported

        """
        if not self.beginMoveRows(sourceParent, sourceRow, sourceRow + count,
                                  destParent, destRow):
            return False

        # block so rowsRemoved/Inserted and dataChanged signals
        # are not emitted during the move. Instead the rowsMoved
        # signal will be emitted from self.endMoveRows().
        with blocked(self):
            didmove = list_model_move_rows_helper(
                self, sourceParent, sourceRow, count, destRow)
        self.endMoveRows()

        if not didmove:
            warnings.warn(
                "`moveRows` did not succeed! Data model might be "
                "in an inconsistent state.",
                RuntimeWarning)
        return didmove

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
        # The widget in the view were already swapped, so
        # we must disconnect from the model when moving the rows.
        # It would be better if this class would also filter and
        # handle internal widget moves.
        model = self.model()
        self.__disconnect(model)
        try:
            model.moveRow
        except AttributeError:
            data = model.itemData(model.index(from_, 0))
            removed = model.removeRow(from_, QModelIndex())
            inserted = model.insertRow(to, QModelIndex())
            model.setItemData(model.index(to, 0), data)
            assert removed and inserted
            assert model.rowCount() == len(self.view.widgets())
        else:
            if to > from_:
                to = to + 1
            didmove = model.moveRow(QModelIndex(), from_, QModelIndex(), to)
            assert didmove
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
            opt.initFrom(self)
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
            if self.__focusframe is not None:
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
            return QSize(250, 350)

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

            if index > oldindex:
                index = index - 1

            if index != oldindex:
                item = layout.takeAt(oldindex)
                assert item.widget() is frame
                layout.insertWidget(index, frame)
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
        mime.setData("application/x-internal-move", b"")
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

    def __init__(self):
        super().__init__()

        self.data = None
        self._invalidated = False

        # List of available preprocessors (DescriptionRole : Description)
        self.preprocessors = QStandardItemModel()

        def mimeData(indexlist):
            assert len(indexlist) == 1
            index = indexlist[0]
            qname = index.data(DescriptionRole).qualname
            m = QMimeData()
            m.setData("application/x-qwidget-ref", qname.encode("utf-8"))
            return m
        # TODO: Fix this (subclass even if just to pass a function
        # for mimeData delegate)
        self.preprocessors.mimeData = mimeData

        box = gui.vBox(self.controlArea, "Preprocessors")

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

        self.overlay = OverlayWidget(self)
        self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.overlay.setWidget(self.flow_view)
        self.overlay.setLayout(QVBoxLayout())
        self.overlay.layout().addWidget(
            QtGui.QLabel("Drag items from the list on the left",
                         wordWrap=True))

        self.scroll_area = QtGui.QScrollArea(
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn
        )
        self.scroll_area.viewport().setAcceptDrops(True)
        self.scroll_area.setWidget(self.flow_view)
        self.scroll_area.setWidgetResizable(True)
        self.mainArea.layout().addWidget(self.scroll_area)
        self.flow_view.installEventFilter(self)

        box = gui.vBox(self.controlArea, "Output")
        gui.auto_commit(box, self, "autocommit", "Send", box=False)

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

        self.apply()

    def load(self, saved):
        """Load a preprocessor list from a dict."""
        name = saved.get("name", "")
        preprocessors = saved.get("preprocessors", [])
        model = StandardItemModel()

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

    def init_code_gen(self):
        gen = self.code_gen()
        gen.set_widget(self)
        gen.add_import(preprocess.preprocess.PreprocessorList)
        gen.add_preamble("from Orange.preprocess import *")
        gen.add_preamble("from Orange.preprocess.score import *")
        gen.add_init("preprocessor", repr(self.buildpreproc()), iscode=True)
        gen.add_output("preprocessor", "preprocessor", iscode=True)
        gen.add_output("preprocessed_data", "preprocessor(input_data)", iscode=True)

        return gen

    def set_model(self, ppmodel):
        if self.preprocessormodel:
            self.preprocessormodel.dataChanged.disconnect(self.__on_modelchanged)
            self.preprocessormodel.rowsInserted.disconnect(self.__on_modelchanged)
            self.preprocessormodel.rowsRemoved.disconnect(self.__on_modelchanged)
            self.preprocessormodel.rowsMoved.disconnect(self.__on_modelchanged)
            self.preprocessormodel.deleteLater()

        self.preprocessormodel = ppmodel
        self.controler.setModel(ppmodel)
        if ppmodel is not None:
            self.preprocessormodel.dataChanged.connect(self.__on_modelchanged)
            self.preprocessormodel.rowsInserted.connect(self.__on_modelchanged)
            self.preprocessormodel.rowsRemoved.connect(self.__on_modelchanged)
            self.preprocessormodel.rowsMoved.connect(self.__on_modelchanged)

        self.__update_overlay()

    def __update_overlay(self):
        if self.preprocessormodel is None or \
                self.preprocessormodel.rowCount() == 0:
            self.overlay.setWidget(self.flow_view)
            self.overlay.show()
        else:
            self.overlay.setWidget(None)
            self.overlay.hide()

    def __on_modelchanged(self):
        self.__update_overlay()
        self.commit()

    @check_sql_input
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
        # Sync the model into storedsettings on every apply.
        self.storeSpecificSettings()
        preprocessor = self.buildpreproc()

        if self.data is not None:
            self.error(0)
            try:
                data = preprocessor(self.data)
            except ValueError as e:
                self.error(0, str(e))
                return
        else:
            data = None

        self.send("Preprocessor", preprocessor)
        self.send("Preprocessed Data", data)

    def commit(self):
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

    def sizeHint(self):
        sh = super().sizeHint()
        return sh.expandedTo(QSize(sh.width(), 500))

    def send_report(self):
        pp = [(self.controler.model().index(i, 0).data(Qt.DisplayRole), w)
              for i, w in enumerate(self.controler.view.widgets())]
        if len(pp):
            self.report_items("Settings", pp)


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
    w.set_data(None)
    w.saveSettings()
    w.onDeleteWidget()
    return r

if __name__ == "__main__":
    sys.exit(test_main())

