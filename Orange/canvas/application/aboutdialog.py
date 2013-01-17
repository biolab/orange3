"""
Orange canvas about dialog
"""

import sys
import pkg_resources

from PyQt4.QtGui import QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QPixmap
from PyQt4.QtCore import Qt


import Orange
from Orange import canvas

ABOUT_TEMPLATE = """\
<center>
<h4>Orange</h4>
<p>Version: {version}</p>
<p>(hg revision: {hg_revision})</p>
</center>

"""


class AboutDialog(QDialog):
    def __init__(self, parent=None, **kwargs):
        QDialog.__init__(self, parent, **kwargs)

        if sys.platform == "darwin":
            self.setAttribute(Qt.WA_MacSmallSize, True)

        self.__setupUi()

    def __setupUi(self):
        layout = QVBoxLayout()
        label = QLabel(self)

        filename = pkg_resources.resource_filename(
                        canvas.__name__,
                        "icons/orange-splash-screen.png")

        label.setPixmap(QPixmap(filename))
        layout.addWidget(label, Qt.AlignCenter)

        try:
            from Orange.version import version
            from Orange.version import hg_revision
        except ImportError:
            dist = pkg_resources.get_distribution("Orange")
            version = dist.version
            hg_revision = "Unknown"

        text = ABOUT_TEMPLATE.format(version=version,
                                     hg_revision=hg_revision)
        # TODO: Also list all known add-on versions.
        text_label = QLabel(text)
        layout.addWidget(text_label, Qt.AlignCenter)

        buttons = QDialogButtonBox(QDialogButtonBox.Close,
                                   Qt.Horizontal,
                                   self)
        layout.addWidget(buttons)
        buttons.rejected.connect(self.accept)
        layout.setSizeConstraint(QVBoxLayout.SetFixedSize)
        self.setLayout(layout)
