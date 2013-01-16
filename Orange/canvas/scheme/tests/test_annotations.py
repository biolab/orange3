"""
Tests for scheme annotations.

"""

from ...gui import test


from .. import SchemeArrowAnnotation, SchemeTextAnnotation


class TestAnnotations(test.QCoreApplication):
    def test_arrow(self):
        arrow = SchemeArrowAnnotation((0, 0), (10, 10))
        self.assertTrue(arrow.start_pos == (0, 0))
        self.assertTrue(arrow.end_pos == (10, 10))

        def count():
            count.i += 1
        count.i = 0

        arrow.geometry_changed.connect(count)
        arrow.set_line((10, 10), (0, 0))
        self.assertTrue(arrow.start_pos == (10, 10))
        self.assertTrue(arrow.end_pos == (0, 0))
        self.assertTrue(count.i == 1)

    def test_text(self):
        text = SchemeTextAnnotation((0, 0, 10, 100), "--")
        self.assertEqual(text.rect, (0, 0, 10, 100))
        self.assertEqual(text.text, "--")

        def count():
            count.i += 1
        count.i = 0

        text.geometry_changed.connect(count)
        text.set_rect((9, 9, 30, 30))
        self.assertEqual(text.rect, (9, 9, 30, 30))
        self.assertEqual(count.i == 1)

        text.rect = (4, 4, 4, 4)
        self.assertEqual(count.i == 2)

        count.i = 0
        text.text_changed.connect(count)

        text.set_text("...")
        self.assertEqual(text.text, "...")
        self.assertTrue(count.i == 1)

        text.text = '=='
        self.assertEqual(text.text, "--")
        self.assertTrue(count.i == 2)
