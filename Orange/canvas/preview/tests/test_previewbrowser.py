"""
Unittests for PrewiewBrowser widget.

"""

from ...gui import test

from ..previewbrowser import PreviewBrowser
from ..previewmodel import PreviewItem, PreviewModel

import pkg_resources

svg1 = pkg_resources.resource_string("Orange.OrangeCanvas",
                                     "icons/default-category.svg")

svg2 = pkg_resources.resource_string("Orange.OrangeCanvas",
                                     "icons/default-widget.svg")


def construct_test_preview_model():
    items = [("Name1", "A preview item 1", svg1, u"~/bla", ),
             ("Name2", "A preview item 2" + "long text" * 5, svg2, "~/item")
             ]

    items = [PreviewItem(*arg[:-1], path=arg[-1]) for arg in items]
    model = PreviewModel(items=items)
    return model


class TestPreviewBrowser(test.QAppTestCase):
    def test_preview_browser(self):
        w = PreviewBrowser()
        model = construct_test_preview_model()
        w.setModel(model)
        w.show()

        def p(index):
            print index

        w.currentIndexChanged.connect(p)
        self.app.exec_()
