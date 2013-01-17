"""
Unittests for PrewiewDialog widget.

"""

from ...gui import test

from ..previewdialog import PreviewDialog
from .test_previewbrowser import construct_test_preview_model


class TestPreviewDialog(test.QAppTestCase):
    def test_preview_dialog(self):
        w = PreviewDialog()
        model = construct_test_preview_model()
        w.setModel(model)
        w.show()

        current = [None]
        w.currentIndexChanged.connect(current.append)
        status = w.exec_()

        if status and len(current) > 1:
            self.assertIs(current[-1], w.currentIndex())

        w.setItems(["A", "B"])
        w.show()
        status = w.exec_()
        if status:
            self.assertTrue(w.currentIndex() != -1)
