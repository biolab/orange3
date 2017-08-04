from AnyQt.QtCore import Qt, QSize

from Orange.widgets.tests.base import GuiTest
from Orange.widgets.utils.messagewidget import (
    MessagesWidget, Message, IconWidget
)


class TestMessageWidget(GuiTest):
    def test_widget(self):
        w = MessagesWidget()
        w.setMessage(0, Message())
        self.assertTrue(w.summarize().isEmpty())
        self.assertSequenceEqual(w.messages(), [Message()])
        w.setMessage(0, Message(Message.Warning, text="a"))
        self.assertFalse(w.summarize().isEmpty())
        self.assertEqual(w.summarize().severity, Message.Warning)
        self.assertEqual(w.summarize().text, "a")
        w.setMessage(1, Message(Message.Error, text="#error#"))
        self.assertEqual(w.summarize().severity, Message.Error)
        self.assertTrue(w.summarize().text.startswith("#error#"))
        self.assertSequenceEqual(
            w.messages(),
            [Message(Message.Warning, text="a"),
             Message(Message.Error, text="#error#")])
        w.setMessage(2, Message(Message.Information, text="<em>Hello</em>",
                                textFormat=Qt.RichText))
        self.assertSequenceEqual(
            w.messages(),
            [Message(Message.Warning, text="a"),
             Message(Message.Error, text="#error#"),
             Message(Message.Information, text="<em>Hello</em>",
                     textFormat=Qt.RichText)])
        w.grab()
        w.removeMessage(2)
        w.clear()
        w.setOpenExternalLinks(True)
        assert w.openExternalLinks()
        self.assertEqual(len(w.messages()), 0)
        self.assertTrue(w.summarize().isEmpty())


class TestIconWidget(GuiTest):
    def test_widget(self):
        w = IconWidget()
        s = w.style()
        icon = s.standardIcon(s.SP_BrowserStop)
        w.setIcon(icon)
        self.assertEqual(w.icon().cacheKey(), icon.cacheKey())
        w.setIconSize(QSize(42, 42))
        self.assertEqual(w.iconSize(), QSize(42, 42))
        self.assertGreaterEqual(w.sizeHint().width(), 42)
        self.assertGreaterEqual(w.sizeHint().height(), 42)
        w.setIconSize(QSize())
