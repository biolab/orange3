# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from unittest.mock import patch

from Orange.data import Table
from Orange.data.io import TabReader, PickleReader
from Orange.tests import named_file
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.filedialogs import format_filter, fix_extension, open_filename_dialog_save
from Orange.widgets.data.owsave import OWSave


class TestOWSave(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWSave)  # type: OWSave

    def test_ordinary_save(self):
        self.send_signal(self.widget.Inputs.data, Table("iris"))

        for ext, writer in [('.tab', TabReader), ('.pickle', PickleReader)]:
            with named_file("", suffix=ext) as filename:
                def choose_file(a, b, c, d, e, fn=filename, w=writer):
                    return fn, format_filter(w)
                with patch("AnyQt.QtWidgets.QFileDialog.getSaveFileName", choose_file):
                    self.widget.save_file_as()
                self.assertEqual(len(Table(filename)), 150)

    def test_filename_with_fix_extension(self):

        def mock_fix_choice(ret):
            f = lambda *x: ret
            f.__dict__.update(fix_extension.__dict__)
            return f

        change_filter = iter([PickleReader, TabReader])

        for file_choice, fix in [
                [lambda *x: ("o.pickle", format_filter(TabReader)),
                 mock_fix_choice(fix_extension.CHANGE_EXT)],
                [lambda *x: ("o.tab", format_filter(PickleReader)),
                 mock_fix_choice(fix_extension.CHANGE_FORMAT)],
                [lambda *x: ("o.tab", format_filter(next(change_filter))),
                 mock_fix_choice(fix_extension.CANCEL)]
        ]:

            with patch("AnyQt.QtWidgets.QFileDialog.getSaveFileName", file_choice),\
                 patch("Orange.widgets.utils.filedialogs.fix_extension", fix):
                saved_filename, format, filter = \
                    open_filename_dialog_save(".", None, OWSave.writers)
                self.assertEqual(saved_filename, "o.tab")
                self.assertEqual(format, TabReader)
                self.assertEqual(filter, format_filter(TabReader))
