import logging
import html
import sys
from collections import namedtuple
from os.path import join, dirname

from AnyQt.QtCore import QObject, QSize
from AnyQt.QtCore import QPoint, Qt, Signal
from AnyQt.QtGui import (QFontMetrics, QIcon, QTextDocument,
                         QAbstractTextDocumentLayout)
from AnyQt.QtWidgets import (QApplication, QListWidget, QListWidgetItem,
                             QToolTip, QStyledItemDelegate,
                             QStyleOptionViewItem, QStyle)

from qtconsole.base_frontend_mixin import BaseFrontendMixin

log = logging.getLogger(__name__)

DEFAULT_COMPLETION_ITEM_WIDTH = 250

JEDI_TYPES = frozenset({'module', 'class', 'instance', 'function', 'param',
                        'path', 'keyword', 'property', 'statement', None})


class HTMLDelegate(QStyledItemDelegate):
    """With this delegate, a QListWidgetItem or a QTableItem can render HTML.

    Taken from https://stackoverflow.com/a/5443112/2399799
    """

    def __init__(self, parent, margin=0):
        super().__init__(parent)
        self._margin = margin

    def _prepare_text_document(self, option, index):
        # This logic must be shared between paint and sizeHint for consitency
        options = QStyleOptionViewItem(option)
        self.initStyleOption(options, index)

        doc = QTextDocument()
        doc.setDocumentMargin(self._margin)
        doc.setHtml(options.text)
        icon_height = doc.size().height() - 2
        options.decorationSize = QSize(icon_height, icon_height)
        return options, doc

    def paint(self, painter, option, index):
        options, doc = self._prepare_text_document(option, index)

        style = (QApplication.style() if options.widget is None
                 else options.widget.style())
        options.text = ""

        # Note: We need to pass the options widget as an argument of
        # drawControl to make sure the delegate is painted with a style
        # consistent with the widget in which it is used.
        # See spyder-ide/spyder#10677.
        style.drawControl(QStyle.CE_ItemViewItem, options, painter,
                          options.widget)

        ctx = QAbstractTextDocumentLayout.PaintContext()

        textRect = style.subElementRect(QStyle.SE_ItemViewItemText,
                                        options, None)
        painter.save()

        painter.translate(textRect.topLeft() + QPoint(0, -3))
        doc.documentLayout().draw(painter, ctx)
        painter.restore()

    def sizeHint(self, option, index):
        _, doc = self._prepare_text_document(option, index)
        return QSize(round(doc.idealWidth()), round(doc.size().height() - 2))


class CompletionWidget(QListWidget):
    """
    Modelled after spyder-ide's ComlpetionWidget.
    Copyright © Spyder Project Contributors
    Licensed under the terms of the MIT License
    (see spyder/__init__.py in spyder-ide/spyder for details)
    """
    ICON_MAP = {}

    sig_show_completions = Signal(object)

    # Signal with the info about the current completion item documentation
    # str: completion name
    # str: completion signature/documentation,
    # QPoint: QPoint where the hint should be shown
    sig_completion_hint = Signal(str, str, QPoint)

    def __init__(self, parent, ancestor):
        super().__init__(ancestor)
        self.textedit = parent
        self._language = None
        self.setWindowFlags(Qt.SubWindow | Qt.FramelessWindowHint)
        self.hide()
        self.itemActivated.connect(self.item_selected)
        # self.currentRowChanged.connect(self.row_changed)
        self.is_internal_console = False
        self.completion_list = None
        self.completion_position = None
        self.automatic = False
        self.display_index = []

        # Setup item rendering
        self.setItemDelegate(HTMLDelegate(self, margin=3))
        self.setMinimumWidth(DEFAULT_COMPLETION_ITEM_WIDTH)

        # Initial item height and width
        fm = QFontMetrics(self.textedit.font())
        self.item_height = fm.height()
        self.item_width = self.width()

        self.setStyleSheet('QListWidget::item:selected {'
                           'background-color: lightgray;'
                           '}')

    def setup_appearance(self, size, font):
        """Setup size and font of the completion widget."""
        self.resize(*size)
        self.setFont(font)
        fm = QFontMetrics(font)
        self.item_height = fm.height()

    def is_empty(self):
        """Check if widget is empty."""
        if self.count() == 0:
            return True
        return False

    def show_list(self, completion_list, position, automatic):
        """Show list corresponding to position."""
        if not completion_list:
            self.hide()
            return

        self.automatic = automatic

        if position is None:
            # Somehow the position was not saved.
            # Hope that the current position is still valid
            self.completion_position = self.textedit.textCursor().position()

        elif self.textedit.textCursor().position() < position:
            # hide the text as we moved away from the position
            self.hide()
            return

        else:
            self.completion_position = position

        self.completion_list = completion_list

        # Check everything is in order
        self.update_current()

        # If update_current called close, stop loading
        if not self.completion_list:
            return

        # If only one, must be chosen if not automatic
        single_match = self.count() == 1
        if single_match and not self.automatic:
            self.item_selected(self.item(0))
            # signal used for testing
            self.sig_show_completions.emit(completion_list)
            return

        self.show()
        self.setFocus()
        self.raise_()

        self.textedit.position_widget_at_cursor(self)

        if not self.is_internal_console:
            tooltip_point = self.rect().topRight()
            tooltip_point = self.mapToGlobal(tooltip_point)

            if self.completion_list is not None:
                for completion in self.completion_list:
                    completion['point'] = tooltip_point

        # Show hint for first completion element
        self.setCurrentRow(0)

        # signal used for testing
        self.sig_show_completions.emit(completion_list)

    def set_language(self, language):
        """Set the completion language."""
        self._language = language.lower()

    def update_list(self, current_word):
        """
        Update the displayed list by filtering self.completion_list based on
        the current_word under the cursor (see check_can_complete).

        If we're not updating the list with new completions, we filter out
        textEdit completions, since it's difficult to apply them correctly
        after the user makes edits.

        If no items are left on the list the autocompletion should stop
        """
        self.clear()

        self.display_index = []
        height = self.item_height
        width = self.item_width

        if current_word:
            for c in self.completion_list:
                c['end'] = c['start'] + len(current_word)

        for i, completion in enumerate(self.completion_list):
            text = completion['text']
            if not self.check_can_complete(text, current_word):
                continue
            item = QListWidgetItem()
            self.set_item_display(
                item, completion, height=height, width=width)
            item.setData(Qt.UserRole, completion)

            self.addItem(item)
            self.display_index.append(i)

        if self.count() == 0:
            self.hide()

    def _get_cached_icon(self, name):
        if name not in JEDI_TYPES:
            log.error('%s is not a valid jedi type', name)
            return None
        if name not in self.ICON_MAP:
            if name is None:
                self.ICON_MAP[name] = QIcon()
            else:
                icon_path = join(dirname(__file__), '..', '..', 'icons',
                                 'pythonscript', name + '.svg')
                self.ICON_MAP[name] = QIcon(icon_path)
        return self.ICON_MAP[name]

    def set_item_display(self, item_widget, item_info, height, width):
        """Set item text & icons using the info available."""
        item_label = item_info['text']
        item_type = item_info['type']

        item_text = self.get_html_item_representation(
            item_label, item_type,
            height=height, width=width)

        item_widget.setText(item_text)
        item_widget.setIcon(self._get_cached_icon(item_type))

    @staticmethod
    def get_html_item_representation(item_completion, item_type=None,
                                     height=14,
                                     width=250):
        """Get HTML representation of and item."""
        height = str(height)
        width = str(width)

        # Unfortunately, both old- and new-style Python string formatting
        # have poor performance due to being implemented as functions that
        # parse the format string.
        # f-strings in new versions of Python are fast due to Python
        # compiling them into efficient string operations, but to be
        # compatible with old versions of Python, we manually join strings.
        parts = [
            '<table width="', width, '" height="', height, '">', '<tr>',

            '<td valign="middle" style="color: black;" style="margin-left:22px;">',
            html.escape(item_completion).replace(' ', '&nbsp;'),
            '</td>',
        ]
        if item_type is not None:
            parts.extend(['<td valign="middle" align="right" float="right" '
                          'style="color: black;">',
                          item_type,
                          '</td>'
                          ])
        parts.extend([
            '</tr>', '</table>',
        ])

        return ''.join(parts)

    def hide(self):
        """Override Qt method."""
        self.completion_position = None
        self.completion_list = None
        self.clear()
        self.textedit.setFocus()
        tooltip = getattr(self.textedit, 'tooltip_widget', None)
        if tooltip:
            tooltip.hide()

        QListWidget.hide(self)
        QToolTip.hideText()

    def keyPressEvent(self, event):
        """Override Qt method to process keypress."""
        # pylint: disable=too-many-branches
        text, key = event.text(), event.key()
        alt = event.modifiers() & Qt.AltModifier
        shift = event.modifiers() & Qt.ShiftModifier
        ctrl = event.modifiers() & Qt.ControlModifier
        altgr = event.modifiers() and (key == Qt.Key_AltGr)
        # Needed to properly handle Neo2 and other keyboard layouts
        # See spyder-ide/spyder#11293
        neo2_level4 = (key == 0)  # AltGr (ISO_Level5_Shift) in Neo2 on Linux
        modifier = shift or ctrl or alt or altgr or neo2_level4
        if key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Tab):
            # Check that what was selected can be selected,
            # otherwise timing issues
            item = self.currentItem()
            if item is None:
                item = self.item(0)

            if self.is_up_to_date(item=item):
                self.item_selected(item=item)
            else:
                self.hide()
                self.textedit.keyPressEvent(event)
        elif key == Qt.Key_Escape:
            self.hide()
        elif key in (Qt.Key_Left, Qt.Key_Right) or text in ('.', ':'):
            self.hide()
            self.textedit.keyPressEvent(event)
        elif key in (Qt.Key_Up, Qt.Key_Down, Qt.Key_PageUp, Qt.Key_PageDown,
                     Qt.Key_Home, Qt.Key_End) and not modifier:
            if key == Qt.Key_Up and self.currentRow() == 0:
                self.setCurrentRow(self.count() - 1)
            elif key == Qt.Key_Down and self.currentRow() == self.count() - 1:
                self.setCurrentRow(0)
            else:
                QListWidget.keyPressEvent(self, event)
        elif len(text) > 0 or key == Qt.Key_Backspace:
            self.textedit.keyPressEvent(event)
            self.update_current()
        elif modifier:
            self.textedit.keyPressEvent(event)
        else:
            self.hide()
            QListWidget.keyPressEvent(self, event)

    def is_up_to_date(self, item=None):
        """
        Check if the selection is up to date.
        """
        if self.is_empty():
            return False
        if not self.is_position_correct():
            return False
        if item is None:
            item = self.currentItem()
        current_word = self.textedit.get_current_word(completion=True)
        completion = item.data(Qt.UserRole)
        filter_text = completion['text']
        return self.check_can_complete(filter_text, current_word)

    @staticmethod
    def check_can_complete(filter_text, current_word):
        """Check if current_word matches filter_text."""
        if not filter_text:
            return True

        if not current_word:
            return True

        return str(filter_text).lower().startswith(
            str(current_word).lower())

    def is_position_correct(self):
        """Check if the position is correct."""

        if self.completion_position is None:
            return False

        cursor_position = self.textedit.textCursor().position()

        # Can only go forward from the data we have
        if cursor_position < self.completion_position:
            return False

        completion_text = self.textedit.get_current_word_and_position(
            completion=True)

        # If no text found, we must be at self.completion_position
        if completion_text is None:
            return self.completion_position == cursor_position

        completion_text, text_position = completion_text
        completion_text = str(completion_text)

        # The position of text must compatible with completion_position
        if not text_position <= self.completion_position <= (
                text_position + len(completion_text)):
            return False

        return True

    def update_current(self):
        """
        Update the displayed list.
        """
        if not self.is_position_correct():
            self.hide()
            return

        current_word = self.textedit.get_current_word(completion=True)
        self.update_list(current_word)
        # self.setFocus()
        # self.raise_()
        self.setCurrentRow(0)

    def focusOutEvent(self, event):
        """Override Qt method."""
        event.ignore()
        # Don't hide it on Mac when main window loses focus because
        # keyboard input is lost.
        # Fixes spyder-ide/spyder#1318.
        if sys.platform == "darwin":
            if event.reason() != Qt.ActiveWindowFocusReason:
                self.hide()
        else:
            # Avoid an error when running tests that show
            # the completion widget
            try:
                self.hide()
            except RuntimeError:
                pass

    def item_selected(self, item=None):
        """Perform the item selected action."""
        if item is None:
            item = self.currentItem()

        if item is not None and self.completion_position is not None:
            self.textedit.insert_completion(item.data(Qt.UserRole),
                                            self.completion_position)
        self.hide()

    def trigger_completion_hint(self, row=None):
        if not self.completion_list:
            return
        if row is None:
            row = self.currentRow()
        if row < 0 or len(self.completion_list) <= row:
            return

        item = self.completion_list[row]
        if 'point' not in item:
            return

        if 'textEdit' in item:
            insert_text = item['textEdit']['newText']
        else:
            insert_text = item['insertText']

            # Split by starting $ or language specific chars
            chars = ['$']
            if self._language == 'python':
                chars.append('(')

            for ch in chars:
                insert_text = insert_text.split(ch)[0]

        self.sig_completion_hint.emit(
            insert_text,
            item['documentation'],
            item['point'])

    # @Slot(int)
    # def row_changed(self, row):
    #     """Set completion hint info and show it."""
    #     self.trigger_completion_hint(row)


class Completer(BaseFrontendMixin, QObject):
    """
    Uses qtconsole's kernel to generate jedi completions, showing a list.
    """

    def __init__(self, qpart):
        QObject.__init__(self, qpart)
        self._request_info = {}
        self.ready = False
        self._qpart = qpart
        self._widget = CompletionWidget(self._qpart, self._qpart.parent())
        self._opened_automatically = True

        self._complete()

    def terminate(self):
        """Object deleted. Cancel timer
        """

    def isVisible(self):
        return self._widget.isVisible()

    def setup_appearance(self, size, font):
        self._widget.setup_appearance(size, font)

    def invokeCompletion(self):
        """Invoke completion manually"""
        self._opened_automatically = False
        self._complete()

    def invokeCompletionIfAvailable(self):
        if not self._opened_automatically:
            return
        self._complete()

    def _show_completions(self, matches, pos):
        self._widget.show_list(matches, pos, self._opened_automatically)

    def _close_completions(self):
        self._widget.hide()

    def _complete(self):
        """ Performs completion at the current cursor location.
        """
        if not self.ready:
            return
        code = self._qpart.text
        cursor_pos = self._qpart.textCursor().position()
        self._send_completion_request(code, cursor_pos)

    def _send_completion_request(self, code, cursor_pos):
        # Send the completion request to the kernel
        msg_id = self.kernel_client.complete(code=code, cursor_pos=cursor_pos)
        info = self._CompletionRequest(msg_id, code, cursor_pos)
        self._request_info['complete'] = info

    # ---------------------------------------------------------------------------
    # 'BaseFrontendMixin' abstract interface
    # ---------------------------------------------------------------------------

    _CompletionRequest = namedtuple('_CompletionRequest',
                                    ['id', 'code', 'pos'])

    def _handle_complete_reply(self, rep):
        """Support Jupyter's improved completion machinery.
        """
        info = self._request_info.get('complete')
        if (info and info.id == rep['parent_header']['msg_id']):
            content = rep['content']

            if 'metadata' not in content or \
                    '_jupyter_types_experimental' not in content['metadata']:
                log.error('Jupyter API has changed, completions are unavailable.')
                return
            matches = content['metadata']['_jupyter_types_experimental']
            start = content['cursor_start']

            start = max(start, 0)

            for m in matches:
                if m['type'] == '<unknown>':
                    m['type'] = None

            self._show_completions(matches, start)
        self._opened_automatically = True

    def _handle_kernel_info_reply(self, _):
        """ Called when the KernelManager channels have started listening or
            when the frontend is assigned an already listening KernelManager.
        """
        if not self.ready:
            self.ready = True

    def _handle_kernel_restarted(self):
        self.ready = True

    def _handle_kernel_died(self, _):
        self.ready = False
