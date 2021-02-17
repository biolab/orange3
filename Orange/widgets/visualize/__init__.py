"""
=========
Visualize
=========

Widgets for data visualization.

"""

# Category description for the widget registry

NAME = "Visualize"

ID = "orange.widgets.visualize"

DESCRIPTION = "Widgets for data visualization."

from AnyQt.QtGui import QPalette

BACKGROUND = {
    'light': {
        QPalette.Light: '#f2b6b8',
        QPalette.Midlight: '#ff8a99',
        QPalette.Button: '#ff7a8a',
    },
    'plain': "#FFB7B1",
}

ICON = "icons/Category-Visualize.svg"

PRIORITY = 2
