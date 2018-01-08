from AnyQt.QtGui import QPalette
from AnyQt.QtCore import Qt
import pyqtgraph as pg

__all__ = ["create_palette", "OWPalette"]


pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOptions(antialias=True)


def create_palette(colors):
    p = QPalette()
    for role, color in colors.items():
        p.setColor(role, color)
    return p


class OWPalette:
    """
        These constants are defined here so that they can be changed without
        extensive changes to the visualizations
    """
    Canvas = QPalette.Base
    Grid = QPalette.Button
    Text = QPalette.Text
    Data = QPalette.Text
    Axis = QPalette.Text

    System = QPalette()
    Light = create_palette({Canvas: Qt.white,
                            Grid: Qt.lightGray,
                            Text: Qt.black})
    Dark = create_palette({Canvas: Qt.black,
                           Grid: Qt.darkGray,
                           Text: Qt.white})
