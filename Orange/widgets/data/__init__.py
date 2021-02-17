
NAME = "Data"

ID = "orange.widgets.data"

DESCRIPTION = """Widgets for data manipulation."""

LONG_DESRIPTION = """
This category contains widgets for data manipulation. This includes
loading, importing, saving, preprocessing, selection, etc.

"""

ICON = "icons/Category-Data.svg"

from AnyQt.QtGui import QPalette

BACKGROUND = {
    'light': {
        QPalette.Light: '#febc57',
        QPalette.Midlight: '#fe9457',
        QPalette.Button: '#fe9057',
    },
    'plain': "#FFD39F",
}

PRIORITY = 1
