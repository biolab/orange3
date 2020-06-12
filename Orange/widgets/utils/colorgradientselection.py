from typing import Any, Tuple

from AnyQt.QtCore import Qt, QSize, QAbstractItemModel, Property
from AnyQt.QtWidgets import (
    QWidget, QSlider, QFormLayout, QComboBox, QStyle,
    QHBoxLayout, QLineEdit, QLabel)
from AnyQt.QtCore import Signal
from AnyQt.QtGui import QFontMetrics, QDoubleValidator

from Orange.widgets.utils import itemmodels, colorpalettes


class ColorGradientSelection(QWidget):
    activated = Signal(int)

    currentIndexChanged = Signal(int)
    thresholdsChanged = Signal(float, float)
    centerChanged = Signal(float)

    def __init__(self, *args, thresholds=(0.0, 1.0), center=None, **kwargs):
        super().__init__(*args, **kwargs)

        low = round(clip(thresholds[0], 0., 1.), 2)
        high = round(clip(thresholds[1], 0., 1.), 2)
        high = max(low, high)
        self.__threshold_low, self.__threshold_high = low, high
        self.__center = center
        form = QFormLayout(
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow
        )
        form.setContentsMargins(0, 0, 0, 0)
        self.gradient_cb = QComboBox(
            None, objectName="gradient-combo-box",
        )
        self.gradient_cb.setAttribute(Qt.WA_LayoutUsesWidgetRect)
        icsize = self.style().pixelMetric(
            QStyle.PM_SmallIconSize, None, self.gradient_cb
        )
        self.gradient_cb.setIconSize(QSize(64, icsize))
        model = itemmodels.ContinuousPalettesModel()
        model.setParent(self)

        self.gradient_cb.setModel(model)
        self.gradient_cb.activated[int].connect(self.activated)
        self.gradient_cb.currentIndexChanged.connect(self.currentIndexChanged)

        if center is not None:
            def __on_center_changed():
                self.__center = float(self.center_edit.text() or "0")
                self.centerChanged.emit(self.__center)

            self.center_box = QWidget()
            center_layout = QHBoxLayout()
            self.center_box.setLayout(center_layout)
            width = QFontMetrics(self.font()).boundingRect("9999999").width()
            self.center_edit = QLineEdit(
                text=f"{self.__center}",
                maximumWidth=width, placeholderText="0", alignment=Qt.AlignRight)
            self.center_edit.setValidator(QDoubleValidator())
            self.center_edit.editingFinished.connect(__on_center_changed)
            center_layout.setContentsMargins(0, 0, 0, 0)
            center_layout.addStretch(1)
            center_layout.addWidget(QLabel("Centered at"))
            center_layout.addWidget(self.center_edit)
            self.gradient_cb.currentIndexChanged.connect(
                self.__update_center_visibility)
        else:
            self.center_box = None

        slider_low = QSlider(
            objectName="threshold-low-slider", minimum=0, maximum=100,
            value=int(low * 100), orientation=Qt.Horizontal,
            tickPosition=QSlider.TicksBelow, pageStep=10,
            toolTip=self.tr("Low gradient threshold"),
            whatsThis=self.tr("Applying a low threshold will squeeze the "
                              "gradient from the lower end")
        )
        slider_high = QSlider(
            objectName="threshold-low-slider", minimum=0, maximum=100,
            value=int(high * 100), orientation=Qt.Horizontal,
            tickPosition=QSlider.TicksAbove, pageStep=10,
            toolTip=self.tr("High gradient threshold"),
            whatsThis=self.tr("Applying a high threshold will squeeze the "
                              "gradient from the higher end")
        )
        form.setWidget(0, QFormLayout.SpanningRole, self.gradient_cb)
        if self.center_box:
            form.setWidget(1, QFormLayout.SpanningRole, self.center_box)
        form.addRow(self.tr("Low:"), slider_low)
        form.addRow(self.tr("High:"), slider_high)
        self.slider_low = slider_low
        self.slider_high = slider_high
        self.slider_low.valueChanged.connect(self.__on_slider_low_moved)
        self.slider_high.valueChanged.connect(self.__on_slider_high_moved)
        self.setLayout(form)

    def setModel(self, model: QAbstractItemModel) -> None:
        self.gradient_cb.setModel(model)

    def model(self) -> QAbstractItemModel:
        return self.gradient_cb.model()

    def findData(self, data: Any, role: Qt.ItemDataRole) -> int:
        return self.gradient_cb.findData(data, role)

    def setCurrentIndex(self, index: int) -> None:
        self.gradient_cb.setCurrentIndex(index)
        self.__update_center_visibility()

    def currentIndex(self) -> int:
        return self.gradient_cb.currentIndex()

    currentIndex_ = Property(
        int, currentIndex, setCurrentIndex, notify=currentIndexChanged)

    def currentData(self, role=Qt.UserRole) -> Any:
        return self.gradient_cb.currentData(role)

    def thresholds(self) -> Tuple[float, float]:
        return self.__threshold_low, self.__threshold_high

    thresholds_ = Property(object, thresholds, notify=thresholdsChanged)

    def thresholdLow(self) -> float:
        return self.__threshold_low

    def setThresholdLow(self, low: float) -> None:
        self.setThresholds(low, max(self.__threshold_high, low))

    thresholdLow_ = Property(
        float, thresholdLow, setThresholdLow, notify=thresholdsChanged)

    def thresholdHigh(self) -> float:
        return self.__threshold_high

    def setThresholdHigh(self, high: float) -> None:
        self.setThresholds(min(self.__threshold_low, high), high)

    def center(self) -> float:
        return self.__center

    def setCenter(self, center: float) -> None:
        self.__center = center
        self.center_edit.setText(f"{center}")
        self.centerChanged.emit(center)

    thresholdHigh_ = Property(
        float, thresholdLow, setThresholdLow, notify=thresholdsChanged)

    def __on_slider_low_moved(self, value: int) -> None:
        high = self.slider_high
        old = self.__threshold_low, self.__threshold_high
        self.__threshold_low = value / 100.
        if value >= high.value():
            self.__threshold_high = value / 100.
            high.setSliderPosition(value)
        new = self.__threshold_low, self.__threshold_high
        if new != old:
            self.thresholdsChanged.emit(*new)

    def __on_slider_high_moved(self, value: int) -> None:
        low = self.slider_low
        old = self.__threshold_low, self.__threshold_high
        self.__threshold_high = value / 100.
        if low.value() >= value:
            self.__threshold_low = value / 100
            low.setSliderPosition(value)
        new = self.__threshold_low, self.__threshold_high
        if new != old:
            self.thresholdsChanged.emit(*new)

    def setThresholds(self, low: float, high: float) -> None:
        low = round(clip(low, 0., 1.), 2)
        high = round(clip(high, 0., 1.), 2)
        if low > high:
            high = low
        if self.__threshold_low != low or self.__threshold_high != high:
            self.__threshold_high = high
            self.__threshold_low = low
            self.slider_low.setSliderPosition(low * 100)
            self.slider_high.setSliderPosition(high * 100)
            self.thresholdsChanged.emit(high, low)

    def __update_center_visibility(self):
        if self.center_box is None:
            return

        palette = self.currentData()
        self.center_box.setVisible(
            isinstance(palette, colorpalettes.Palette)
            and palette.flags & palette.Flags.Diverging != 0)


def clip(a, amin, amax):
    return min(max(a, amin), amax)
