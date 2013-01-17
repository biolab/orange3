from ..framelesswindow import FramelessWindow

from ..test import QAppTestCase


class TestFramelessWindow(QAppTestCase):
    def test_framelesswindow(self):
        window = FramelessWindow()
        window.show()

        def cycle():
            window.setRadius((window.radius() + 3) % 30)
            self.singleShot(250, cycle)

        cycle()
        self.app.exec_()
