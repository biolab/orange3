import unittest
import math

from Orange.widgets.visualize.owpythagorastree import \
    PythagorasTree, Point, Square


class TestPythagorasTree(unittest.TestCase):
    def setUp(self):
        self.builder = PythagorasTree()

    def test_get_point_on_square_edge_with_no_angle(self):
        point = self.builder._get_point_on_square_edge(
            center=Point(0, 0), length=2, angle=0
        )
        expected_point = Point(1, 0)
        self.assertAlmostEqual(point.x, expected_point.x, places=1)
        self.assertAlmostEqual(point.y, expected_point.y, places=1)

    def test_get_point_on_square_edge_with_non_zero_angle(self):
        point = self.builder._get_point_on_square_edge(
            center=Point(2.7, 2.77), length=1.65, angle=math.radians(20.97)
        )
        expected_point = Point(3.48, 3.07)
        self.assertAlmostEqual(point.x, expected_point.x, places=1)
        self.assertAlmostEqual(point.y, expected_point.y, places=1)

    def test_compute_center_with_simple_square_angle(self):
        initial_square = Square(Point(0, 0), length=2, angle=math.pi / 2)
        point = self.builder._compute_center(
            initial_square, length=1.13, alpha=math.radians(68.57))
        expected_point = Point(1.15, 1.78)
        self.assertAlmostEqual(point.x, expected_point.x, places=1)
        self.assertAlmostEqual(point.y, expected_point.y, places=1)

    def test_compute_center_with_complex_square_angle(self):
        initial_square = Square(
            Point(1.5, 1.5), length=2.24, angle=math.radians(63.43)
        )
        point = self.builder._compute_center(
            initial_square, length=1.65, alpha=math.radians(95.06))
        expected_point = Point(3.48, 3.07)
        self.assertAlmostEqual(point.x, expected_point.x, places=1)
        self.assertAlmostEqual(point.y, expected_point.y, places=1)

    def test_compute_center_with_complex_square_angle_with_base_angle(self):
        initial_square = Square(
            Point(1.5, 1.5), length=2.24, angle=math.radians(63.43)
        )
        point = self.builder._compute_center(
            initial_square, length=1.51, alpha=math.radians(180 - 95.06),
            base_angle=math.radians(95.06))
        expected_point = Point(1.43, 3.98)
        self.assertAlmostEqual(point.x, expected_point.x, places=1)
        self.assertAlmostEqual(point.y, expected_point.y, places=1)
