import math
from decimal import Decimal

import numpy as np

from AnyQt.QtCore import QLocale, QSize
from AnyQt.QtWidgets import QDoubleSpinBox, QStyle, QStyleOptionSpinBox

DBL_MIN = float(np.finfo(float).min)
DBL_MAX = float(np.finfo(float).max)
DBL_MAX_10_EXP = math.floor(math.log10(DBL_MAX))
DBL_DIG = math.floor(math.log10(2 ** np.finfo(float).nmant))


class DoubleSpinBox(QDoubleSpinBox):
    """
    A QDoubleSpinSubclass with non-fixed decimal precision/rounding.
    """
    def __init__(self, parent=None, decimals=-1, minimumStep=1e-5,
                 minimumContentsLenght=-1, **kwargs):
        self.__decimals = decimals
        self.__minimumStep = minimumStep
        self.__minimumContentsLength = minimumContentsLenght
        stepType = kwargs.pop("stepType", DoubleSpinBox.DefaultStepType)
        super().__init__(parent, **kwargs)
        if decimals < 0:
            super().setDecimals(DBL_MAX_10_EXP + DBL_DIG)
        else:
            super().setDecimals(decimals)
        self.setStepType(stepType)

    def setDecimals(self, prec: int) -> None:
        """
        Set the number of decimals in display/edit

        If negative value then no rounding takes place and the value is
        displayed using `QLocale.FloatingPointShortest` precision.
        """
        self.__decimals = prec
        if prec < 0:
            # disable rounding in base implementation.
            super().setDecimals(DBL_MAX_10_EXP + DBL_DIG)
        else:
            super().setDecimals(prec)

    def decimals(self):
        return self.__decimals

    def setMinimumStep(self, step):
        """
        Minimum step size when `stepType() == AdaptiveDecimalStepType`
        and `decimals() < 0`.
        """
        self.__minimumStep = step

    def minimumStep(self):
        return self.__minimumStep

    def textFromValue(self, v: float) -> str:
        """Reimplemented."""
        if self.__decimals < 0:
            locale = self.locale()
            return locale.toString(v, 'f', QLocale.FloatingPointShortest)
        else:
            return super().textFromValue(v)

    def stepBy(self, steps: int) -> None:
        """
        Reimplemented.
        """
        # Compute up/down step using decimal type without rounding
        value = self.value()
        value_dec = Decimal(str(value))
        if self.stepType() == DoubleSpinBox.AdaptiveDecimalStepType:
            step_dec = self.__adaptiveDecimalStep(steps)
        else:
            step_dec = Decimal(str(self.singleStep()))
        # print(str(step_dec.fma(steps, value_dec)))
        value_dec = value_dec + step_dec * steps
        # print(str(value), "+", str(step_dec), "*", steps, "=", str(vinc))
        self.setValue(float(value_dec))

    def __adaptiveDecimalStep(self, steps: int) -> Decimal:
        # adapted from QDoubleSpinBoxPrivate::calculateAdaptiveDecimalStep
        decValue: Decimal = Decimal(str(self.value()))
        decimals = self.__decimals
        if decimals < 0:
            minStep = Decimal(str(self.__minimumStep))
        else:
            minStep = Decimal(10) ** -decimals

        absValue = abs(decValue)
        if absValue < minStep:
            return minStep
        valueNegative = decValue < 0
        stepsNegative = steps < 0
        if valueNegative != stepsNegative:
            absValue /= Decimal("1.01")
        step = Decimal(10) ** (math.floor(absValue.log10()) - 1)
        return max(minStep, step)

    if not hasattr(QDoubleSpinBox, "stepType"):  # pragma: no cover
        DefaultStepType = 0
        AdaptiveDecimalStepType = 1
        __stepType = AdaptiveDecimalStepType

        def setStepType(self, stepType):
            self.__stepType = stepType

        def stepType(self):
            return self.__stepType

    def setMinimumContentsLength(self, characters: int):
        self.__minimumContentsLength = characters
        self.updateGeometry()

    def minimumContentsLength(self):
        return self.__minimumContentsLength

    def sizeHint(self) -> QSize:
        if self.minimumContentsLength() < 0:
            return super().sizeHint()
        self.ensurePolished()
        fm = self.fontMetrics()
        template = "X" * self.minimumContentsLength()
        template += "."
        if self.prefix():
            template = self.prefix() + " " + template
        if self.suffix():
            template = template + self.suffix()
        if self.minimum() < 0.0:
            template = "-" + template
        if self.specialValueText():
            templates = [template, self.specialValueText()]
        else:
            templates = [template]
        height = self.lineEdit().sizeHint().height()
        width = max(map(fm.horizontalAdvance, templates))
        width += 2  # cursor blinking space
        hint = QSize(width, height)
        opt = QStyleOptionSpinBox()
        self.initStyleOption(opt)
        sh = self.style().sizeFromContents(QStyle.CT_SpinBox, opt, hint, self)
        return sh

    def minimumSizeHint(self) -> QSize:
        if self.minimumContentsLength() < 0:
            return super().minimumSizeHint()
        else:
            return self.sizeHint()
