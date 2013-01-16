from ...scheme import Scheme
from ..schemeinfo import SchemeInfoDialog
from ...gui import test


class TestSchemeInfo(test.QAppTestCase):
    def test_scheme_info(self):
        scheme = Scheme(title="A Scheme", description="A String\n")
        dialog = SchemeInfoDialog()
        dialog.setScheme(scheme)

        status = dialog.exec_()

        if status == dialog.Accepted:
            self.assertEqual(scheme.title.strip(),
                             str(dialog.editor.name_edit.text()).strip())
            self.assertEqual(scheme.description,
                             str(dialog.editor.desc_edit \
                                           .toPlainText()).strip())
