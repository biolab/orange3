import numpy as np

from AnyQt import QtGui
from AnyQt.QtWidgets import (
    QTableWidget,
    QTableWidgetItem,
    QSlider,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QStyle,
    QToolTip,
    QStyleOptionSlider,
)
from AnyQt.QtCore import Qt, QRect
from AnyQt.QtGui import QPainter, QFontMetrics

from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting
from Orange.widgets.widget import Input, Output, OWWidget, AttributeList, Msg
from Orange.data import Table
from Orange.classification import Model

from Orange.classification.scoringsheet import ScoringSheetModel
from Orange.classification.utils.fasterrisk.utils import (
    get_support_indices,
    get_all_product_booleans,
)


class ScoringSheetTable(QTableWidget):
    def __init__(self, main_widget, parent=None):
        """
        Initialize the ScoringSheetTable.

        It sets the column headers and connects the itemChanged
        signal to the handle_item_changed method.
        """
        super().__init__(parent)
        self.main_widget = main_widget
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Attribute Name", "Points", "Selected"])
        self.itemChanged.connect(self.handle_item_changed)

    def populate_table(self, attributes, coefficients):
        """
        Populates the table with the given attributes and coefficients.

        It creates a row for each attribute and populates the first two columns with
        the attribute name and coefficient respectively. The third column contains a
        checkbox that allows the user to select the attribute.
        """
        self.setRowCount(len(attributes))
        for i, (attr, coef) in enumerate(zip(attributes, coefficients)):
            # First column
            self.setItem(i, 0, QTableWidgetItem(attr))

            # Second column (align text to the right)
            coef_item = QTableWidgetItem(str(coef))
            coef_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.setItem(i, 1, coef_item)

            # Third column (checkbox)
            checkbox = QTableWidgetItem()
            checkbox.setCheckState(Qt.Unchecked)
            self.setItem(i, 2, checkbox)

            for col in range(self.columnCount()):
                item = self.item(i, col)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable & ~Qt.ItemIsSelectable)

        # Resize columns to fit the contents
        self.resize_columns_to_contents()

    def resize_columns_to_contents(self):
        """
        Resize each column to fit the content.
        """
        for column in range(self.columnCount()):
            self.resizeColumnToContents(column)

    def handle_item_changed(self, item):
        """
        Handles the change in the state of the checkbox.

        It updates the slider value depending on the collected points.
        """
        if item.column() == 2:
            self.main_widget._update_slider_value()


class RiskSlider(QWidget):
    def __init__(self, points, probabilities, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)

        # Set the margins for the layout
        self.leftMargin = 20
        self.topMargin = 20
        self.rightMargin = 20
        self.bottomMargin = 20
        self.layout.setContentsMargins(
            self.leftMargin, self.topMargin, self.rightMargin, self.bottomMargin
        )

        # Setup the labels
        self.setup_labels()

        # Create the slider
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setEnabled(False)
        self.layout.addWidget(self.slider)

        self.points = points
        self.probabilities = probabilities
        self.setup_slider()

        # Set the margin for drawing text
        self.textMargin = 1

        # This is needed to show the tooltip when the mouse is over the slider thumb
        self.slider.installEventFilter(self)
        self.setMouseTracking(True)
        self.target_class = None

        self.label_frequency = 1

    def setup_labels(self):
        """
        Set up the labels for the slider.

        It creates a vertical layout for the labels and adds it to the main layout.
        It is only called once when the widget is initialized.
        """
        # Create the labels for the slider
        self.label_layout = QVBoxLayout()
        # Add the label for the points "Points:"
        self.points_label = QLabel("<b>Total:</b>")
        self.label_layout.addWidget(self.points_label)
        # Add stretch to the label layout
        self.label_layout.addSpacing(23)
        # Add the label for the probability "Probability:"
        self.probability_label = QLabel("<b>Probabilities (%):</b>")
        self.label_layout.addWidget(self.probability_label)
        self.layout.addLayout(self.label_layout)
        # Add a spacer
        self.layout.addSpacing(28)

    def setup_slider(self):
        """
        Set up the slider with the given points and probabilities.

        It sets the minimum and maximum values (of the indexes for the ticks) of the slider.
        It is called when the points and probabilities are updated.
        """
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.points) - 1 if self.points else 0)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(1)  # Set tick interval

    def move_to_value(self, value):
        """
        Move the slider to the closest tick mark to the given value.
        """
        if not self.points:
            return
        closest_point_index = min(
            range(len(self.points)), key=lambda i: abs(self.points[i] - value)
        )
        self.slider.setValue(closest_point_index)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_label_frequency()
        self.update()

    def update_label_frequency(self):
        """
        Update the label frequency based on the width of the slider and the number of points.

        Label frequency determines how many labels are shown on the slider.
        """
        total_width = self.slider.width()
        label_width = QFontMetrics(self.font()).boundingRect("100.0%").width()
        max_labels = total_width // label_width

        frequencies = [1, 2, 5, 10, 20, 50, 100]
        for frequency in frequencies:
            if max_labels >= len(self.points) / frequency:
                self.label_frequency = frequency
                break

    def paintEvent(self, event):
        """
        Paint the point and probabilitie labels above and below the tick marks respectively.
        """
        super().paintEvent(event)

        if not self.points:
            return

        painter = QPainter(self)
        fm = QFontMetrics(painter.font())

        for i, point in enumerate(self.points):
            if i % self.label_frequency == 0:
                # Calculate the x position of the tick mark
                x_pos = (
                    QStyle.sliderPositionFromValue(
                        self.slider.minimum(),
                        self.slider.maximum(),
                        i,
                        self.slider.width(),
                    )
                    + self.slider.x()
                )

                # Draw the point label above the tick mark
                point_str = str(point)
                point_rect = fm.boundingRect(point_str)
                point_x = int(x_pos - point_rect.width() / 2)
                point_y = int(self.slider.y() - self.textMargin - point_rect.height())
                painter.drawText(
                    QRect(point_x, point_y, point_rect.width(), point_rect.height()),
                    Qt.AlignCenter,
                    point_str,
                )

                # Draw the probability label below the tick mark
                prob_str = str(round(self.probabilities[i], 1)) + "%"
                prob_rect = fm.boundingRect(prob_str)
                prob_x = int(x_pos - prob_rect.width() / 2)
                prob_y = int(self.slider.y() + self.slider.height() + self.textMargin)
                painter.drawText(
                    QRect(prob_x, prob_y, prob_rect.width(), prob_rect.height()),
                    Qt.AlignCenter,
                    prob_str,
                )

        painter.end()

    def eventFilter(self, watched, event):
        """
        Event filter to intercept hover events on the slider.

        This is needed to show the tooltip when the mouse is over the slider thumb.
        """
        if watched == self.slider and isinstance(event, QtGui.QHoverEvent):
            # Handle the hover event when it's over the slider
            self.handle_hover_event(event.pos())
            return True
        else:
            # Call the base class method to continue default event processing
            return super().eventFilter(watched, event)

    def handle_hover_event(self, pos):
        """
        Handle hover events for the slider.

        Display the tooltip when the mouse is over the slider thumb.
        """
        thumbRect = self.get_thumb_rect()
        if thumbRect.contains(pos) and self.points:
            value = self.slider.value()
            points = self.points[value]
            probability = self.probabilities[value]
            tooltip = str(
                f"<b>{self.target_class}</b>\n "
                f"<hr style='margin: 0px; padding: 0px; border: 0px; height: 1px; background-color: #000000'>"
                f"<b>Points:</b> {int(points)}<br>"
                f"<b>Probability:</b> {probability:.1f}%"
            )
            QToolTip.showText(self.slider.mapToGlobal(pos), tooltip)
        else:
            QToolTip.hideText()

    def get_thumb_rect(self):
        """
        Get the rectangle of the slider thumb.
        """
        opt = QStyleOptionSlider()
        self.slider.initStyleOption(opt)

        style = self.slider.style()

        # Get the area of the slider that contains the handle
        handle_rect = style.subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self.slider
        )

        # Calculate the position and size of the thumb
        thumb_x = handle_rect.x()
        thumb_y = handle_rect.y()
        thumb_width = handle_rect.width()
        thumb_height = handle_rect.height()

        return QRect(thumb_x, thumb_y, thumb_width, thumb_height)


class OWScoringSheetViewer(OWWidget):
    """
    Allows visualization of the scoring sheet model.
    """

    name = "Scoring Sheet Viewer"
    description = "Visualize the scoring sheet model."
    want_control_area = False
    icon = "icons/ScoringSheetViewer.svg"
    replaces = [
        "orangecontrib.prototypes.widgets.owscoringsheetviewer.OWScoringSheetViewer"
    ]
    priority = 2010
    keywords = "scoring sheet viewer"

    class Inputs:
        classifier = Input("Classifier", Model)
        data = Input("Data", Table)

    class Outputs:
        features = Output("Features", AttributeList)

    target_class_index = ContextSetting(0)

    class Error(OWWidget.Error):
        invalid_classifier = Msg(
            "Scoring Sheet Viewer only accepts a Scoring Sheet model."
        )

    class Information(OWWidget.Information):
        multiple_instances = Msg(
            "The input data contains multiple instances. Only the first instance will be used."
        )

    def __init__(self):
        super().__init__()
        self.data = None
        self.instance = None
        self.instance_points = []
        self.classifier = None
        self.coefficients = None
        self.attributes = None
        self.all_scores = None
        self.all_risks = None
        self.domain = None
        self.old_target_class_index = self.target_class_index

        self._setup_gui()
        self.resize(700, 400)

    # GUI Methods ----------------------------------------------------------------------------------

    def _setup_gui(self):
        # Create a new widget box for the combo box in the main area
        combo_box_layout = gui.widgetBox(self.mainArea, orientation="horizontal")
        self.class_combo = gui.comboBox(
            combo_box_layout,
            self,
            "target_class_index",
            callback=self._class_combo_changed,
        )
        self.class_combo.setFixedWidth(100)
        combo_box_layout.layout().addWidget(QLabel("Target class:"))
        combo_box_layout.layout().addWidget(self.class_combo)
        combo_box_layout.layout().addStretch()

        self.coefficient_table = ScoringSheetTable(main_widget=self, parent=self)
        gui.widgetBox(self.mainArea).layout().addWidget(self.coefficient_table)

        self.risk_slider = RiskSlider([], [], self)
        gui.widgetBox(self.mainArea).layout().addWidget(self.risk_slider)

    def _reset_ui_to_original_state(self):
        """
        Reset all UI components to their original state.
        """
        # Reset the coefficient table
        self.coefficient_table.clearContents()
        self.coefficient_table.setRowCount(0)

        # Reset the risk slider
        self.risk_slider.slider.setValue(0)
        self.risk_slider.points = []
        self.risk_slider.probabilities = []
        self.risk_slider.setup_slider()
        self.risk_slider.update()

        # Reset class combo box
        self.class_combo.clear()

    def _populate_interface(self):
        """Populate the scoring sheet based on extracted data."""
        if self.attributes and self.coefficients:
            self.coefficient_table.populate_table(self.attributes, self.coefficients)

            # Update points and probabilities in the custom slider
            class_var_name = self.domain.class_vars[0].name
            class_var_value = self.domain.class_vars[0].values[self.target_class_index]

            self.risk_slider.points = self.all_scores
            self.risk_slider.probabilities = self.all_risks
            self.risk_slider.target_class = f"{class_var_name} = {class_var_value}"
            self.risk_slider.setup_slider()
            self.risk_slider.update()

    def _update_slider_value(self):
        """
        Updates the slider value to reflect the total points collected.

        This method is called when user changes the state of the checkbox in the coefficient table.
        """
        if not self.coefficient_table:
            return
        total_coefficient = sum(
            float(self.coefficient_table.item(row, 1).text())
            for row in range(self.coefficient_table.rowCount())
            if self.coefficient_table.item(row, 2)
            and self.coefficient_table.item(row, 2).checkState() == Qt.Checked
        )
        self.risk_slider.move_to_value(total_coefficient)

    def _update_controls(self):
        """
        It updates the interface components based on the extracted data.

        This method is called when the user inputs data, changes the classifier or the target class.
        """
        self._populate_interface()
        self._update_slider_value()
        self._setup_class_combo()
        self._set_instance_points()

    # Class Combo Methods --------------------------------------------------------------------------

    def _setup_class_combo(self):
        """
        This method is used to populate the class combo box with the target classes.
        """
        self.class_combo.clear()
        if self.domain is not None:
            values = self.domain.class_vars[0].values
            if values:
                self.class_combo.addItems(values)
                self.class_combo.setCurrentIndex(self.target_class_index)

    def _class_combo_changed(self):
        """
        This method is called when the user changes the target class.
        It updates the interface components based on the selected class.
        """
        self.target_class_index = self.class_combo.currentIndex()
        if self.target_class_index == self.old_target_class_index:
            return
        self.old_target_class_index = self.target_class_index

        self._adjust_for_target_class()
        self._update_controls()

    def _adjust_for_target_class(self):
        """
        Adjusts the coefficients, scores, and risks for the negative/positive class.

        This allows user to select the target class and see the
        corresponding coefficients, scores, and risks.
        """
        # Negate the coefficients
        self.coefficients = [-coef for coef in self.coefficients]
        # Negate the scores
        self.all_scores = [-score if score != 0 else score for score in self.all_scores]
        self.all_scores.sort()
        # Adjust the risks
        self.all_risks = [100 - risk for risk in self.all_risks]
        self.all_risks.sort()

    # Classifier Input Methods ---------------------------------------------------------------------

    def _extract_data_from_model(self, classifier):
        """
        Extracts the attributes, non-zero coefficients, all possible
        scores, and corresponding probabilities from the model.
        """
        model = classifier.model

        # 1. Extracting attributes and non-zero coefficients
        nonzero_indices = get_support_indices(model.coefficients)
        attributes = [model.featureNames[i] for i in nonzero_indices]
        coefficients = [int(model.coefficients[i]) for i in nonzero_indices]

        # 2. Extracting possible points and corresponding probabilities
        len_nonzero_indices = len(nonzero_indices)
        # If we have less than 10 attributes, we can calculate all possible combinations of scores.
        if len_nonzero_indices <= 10:
            all_product_booleans = get_all_product_booleans(len_nonzero_indices)
            all_scores = all_product_booleans.dot(model.coefficients[nonzero_indices])
            all_scores = np.unique(all_scores)
        # If there are more than 10 non-zero coefficients, calculating all possible combinations
        # of scores might be computationally intensive. Instead, the method calculates all possible
        # scores from the training dataset (X_train) and then picks some quantile points
        # (in this case, a maximum of 20) to represent the possible scores.
        else:
            all_scores = model.X_train.dot(model.coefficients)
            all_scores = np.unique(all_scores)
            quantile_len = min(20, len(all_scores))
            quantile_points = np.asarray(range(1, 1 + quantile_len)) / quantile_len
            all_scores = np.quantile(
                all_scores, quantile_points, method="closest_observation"
            )

        all_scaled_scores = (model.intercept + all_scores) / model.multiplier
        all_risks = 1 / (1 + np.exp(-all_scaled_scores))

        self.attributes = attributes
        self.coefficients = coefficients
        self.all_scores = all_scores.tolist()
        self.all_risks = (all_risks * 100).tolist()
        self.domain = classifier.domain

        # For some reason when leading the model the scores and probabilities are
        # set for the wrong target class. This is a workaround to fix that.
        self._adjust_for_target_class()

    def _is_valid_classifier(self, classifier):
        """Check if the classifier is a valid ScoringSheetModel."""
        if not isinstance(classifier, ScoringSheetModel):
            self.Error.invalid_classifier()
            return False
        return True

    def _clear_classifier_data(self):
        """Clear classifier data and associated interface components."""
        self.coefficients = None
        self.attributes = None
        self.all_scores = None
        self.all_risks = None
        self.classifier = None
        self.Outputs.features.send(None)

    # Data Input Methods ---------------------------------------------------------------------------

    def _clear_table_data(self):
        """Clear data and associated interface components."""
        self.data = None
        self.instance = None
        self.instance_points = []
        self._set_table_checkboxes()

    def _set_instance_points(self):
        """
        Initializes the instance and its points and sets the checkboxes in the coefficient table.
        """
        if self.data and self.domain is not None:
            self._init_instance_points()

        self._set_table_checkboxes()

    def _set_table_checkboxes(self):
        """
        Sets the checkboxes in the coefficient table based on the instance points.
        Or clears the checkboxes if the instance points are not initialized.
        """
        for row in range(self.coefficient_table.rowCount()):
            if self.instance_points and self.instance_points[row] != 0:
                self.coefficient_table.item(row, 2).setCheckState(Qt.Checked)
            else:
                self.coefficient_table.item(row, 2).setCheckState(Qt.Unchecked)

    def _init_instance_points(self):
        """
        Initialize the instance which is used to show the points collected for each attribute.
        Get the values of the features for the instance and store them in a list.
        """
        instances = self.data.transform(self.domain)
        self.instance = instances[0]
        self.instance_points = [
            self.instance.list[i]
            for i in get_support_indices(self.classifier.model.coefficients)
        ]

    # Input Methods --------------------------------------------------------------------------------

    @Inputs.classifier
    def set_classifier(self, classifier):
        self.Error.invalid_classifier.clear()
        if not classifier or not self._is_valid_classifier(classifier):
            self._clear_classifier_data()
            self._reset_ui_to_original_state()
            return

        self.classifier = classifier
        self._extract_data_from_model(classifier)
        self._update_controls()
        # Output the features
        self.Outputs.features.send(
            AttributeList(
                [feature for feature in self.domain if feature.name in self.attributes]
            )
        )

    @Inputs.data
    def set_data(self, data):
        self.Information.multiple_instances.clear()
        if not data or len(data) < 1:
            self._clear_table_data()
            return

        self.data = data
        if len(data) > 1:
            self.Information.multiple_instances()
        self._update_controls()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from Orange.classification.scoringsheet import ScoringSheetLearner

    mock_data = Table("heart_disease")
    mock_learner = ScoringSheetLearner(15, 5, 5, None)
    mock_model = mock_learner(mock_data)
    WidgetPreview(OWScoringSheetViewer).run(
        set_classifier=mock_model, set_data=mock_data
    )
