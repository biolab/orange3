import re
from collections import namedtuple
from datetime import datetime
import logging

from PyQt4 import QtGui, QtCore

from Orange.widgets import widget, gui, settings
from Orange.data.table import Table


log = logging.getLogger()


SHEETS_PATTERN = re.compile(
    r'(?:https?://)?(?:www\.)?'
     'docs\.google\.com/spreadsheets/d/'
     '(?P<workbook_id>[-\w_]+)'
     '(?:/.*?gid=(?P<sheet_id>\d+).*|.*)?',
    re.IGNORECASE
)


def SHEETS_URL(url):
    match = SHEETS_PATTERN.match(url)
    workbook, sheet = match.group('workbook_id'), match.group('sheet_id')
    if not workbook: raise ValueError
    url = 'https://docs.google.com/spreadsheets/d/{}/export?format=tsv'.format(workbook)
    if sheet: url += '&gid=' + sheet
    return url


Sheet = namedtuple('Sheet', ('name', 'url'))


class OWGoogleSheets(widget.OWWidget):
    name = "Google Sheets"
    description = "Read data from a Google Sheets spreadsheet."
    icon = "icons/GoogleSheets.svg"
    priority = 20
    outputs = [("Data", Table)]

    want_main_area = False

    recent = settings.Setting([])
    autocommit = settings.Setting(True)

    def __init__(self):
        super().__init__()
        hb = gui.widgetBox(self.controlArea, 'Google Sheets', orientation='horizontal')
        self.combo = combo = QtGui.QComboBox(hb)
        combo.setEditable(True)
        combo.setMinimumWidth(300)
        hb.layout().addWidget(QtGui.QLabel('URL:', hb))
        hb.layout().addWidget(combo)
        hb.layout().setStretch(1, 2)
        box = gui.widgetBox(self.controlArea, "Info", addSpace=True)
        info = self.data_info = gui.widgetLabel(box, '')
        info.setWordWrap(True)
        self.controlArea.layout().addStretch(1)
        gui.auto_commit(self.controlArea, self, 'autocommit', label='Commit')

        self.set_combo_items()
        self.table = None
        self.set_info()
        self.timer = QtCore.QTimer(self)
        combo.editTextChanged.connect(self.on_combo_textchanged)
        combo.currentIndexChanged.connect(self.on_combo_activated)
        combo.currentIndexChanged.emit(0)

    def set_combo_items(self):
        self.combo.clear()
        for sheet in self.recent:
            self.combo.addItem(sheet.name, sheet.url)

    def commit(self):
        self.send('Data', self.table)

    def on_combo_textchanged(self, text):
        self.timer.stop()
        try: url = SHEETS_URL(text)
        except (ValueError, AttributeError):
            self.error('Unrecognized URL; should be "docs.google.com/spreadsheets/d/<SHEET-ID>"')
            return
        self.error()
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(lambda: self.on_combo_activated(url=url))
        self.timer.start(500)

    def on_combo_activated(self, index=float('inf'), url=''):
        self.error()
        # Index from combobox selection
        if 0 <= index < len(self.recent):
            sheet = self.recent.pop(index)
            self.table = self.retrieve(sheet.url)
            self.recent.insert(0, sheet)
        # URL from textchanged event
        elif url:
            table = self.table = self.retrieve(url)
            if not table: return
            sheet = Sheet(table.name, url)
            self.recent = [s for s in self.recent if s.url != url]
            self.recent.insert(0, sheet)
        else: return
        self.set_info()
        self.commit()

        self.combo.editTextChanged.disconnect(self.on_combo_textchanged)
        self.combo.currentIndexChanged.disconnect(self.on_combo_activated)
        self.set_combo_items()
        self.combo.editTextChanged.connect(self.on_combo_textchanged)
        self.combo.currentIndexChanged.connect(self.on_combo_activated)

    def set_info(self):
        data = self.table
        if not data:
            self.data_info.setText('No spreadsheet loaded.')
            return
        text = "{} instance(s), {} feature(s), {} meta attribute(s)".format(
            len(data), len(data.domain.attributes), len(data.domain.metas))
        try: text += '\nFirst entry: {}\nLast entry: {}'.format(data[0, 'Timestamp'],
                                                                data[-1, 'Timestamp'])
        except Exception: pass  # no Timestamp header
        self.data_info.setText(text)

    def retrieve(self, url):
        if not url: return
        progress = gui.ProgressBar(self, 10)
        for i in range(3): progress.advance()
        try: table = Table.from_url(url)
        except Exception as e:
            import traceback
            log.error(traceback.format_exc())
            log.error("Couldn't load spreadsheet %s: %s", url, e)
            self.error("Couldn't load spreadsheet. Ensure correct read permissions; rectangular, top-left aligned sheet data ...")
            return
        else:
            for i in range(7): progress.advance()
        finally:
            progress.finish()
        return table


if __name__ == "__main__":
    a = QtGui.QApplication([])
    ow = OWGoogleSheets()
    ow.show()
    a.exec_()
    ow.saveSettings()
