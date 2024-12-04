import math
from abc import ABC, abstractmethod
import unittest


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def p(x, y):
        return Point(x, y)

    @staticmethod
    def p0():
        return Point(0, 0)

    @staticmethod
    def subX(a, b):
        return a.x - b.x

    @staticmethod
    def subY(a, b):
        return a.y - b.y

    @staticmethod
    def length(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    @staticmethod
    def l(p1, p2):
        return Line(p1, p2)

    @staticmethod
    def OX():
        return Line(Point.p(0, 0), Point.p(1, 0))

    @staticmethod
    def OY():
        return Line(Point.p(0, 0), Point.p(0, 1))


def compare_point_to_line(point, line):
    a = line.p2.y - line.p1.y
    b = line.p1.x - line.p2.x
    c = -(a * line.p1.x + b * line.p1.y)
    px, py = point.x, point.y
    result = a * px + b * py + c
    return result


class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    @staticmethod
    def c(center, radius):
        return Circle(center, radius)

    @staticmethod
    def unitCircle():
        return Circle(Point.p0(), 1)


class Area(ABC):
    @abstractmethod
    def test(self, point):
        pass

    @abstractmethod
    def includeBorders(self, flag):
        return NotImplemented


class Rectangle(Area):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def test(self, point):
        return (self.p1.x < point.x < self.p2.x and self.p1.y < point.y < self.p2.y)

    def includeBorders(self, flag):
        if flag:
            return self.p1.x <= point.x <= self.p2.x and self.p1.y <= point.y <= self.p2.y
        else:
            return self.test(point)


# Test class using unittest framework
class TestGeometry(unittest.TestCase):
    def test_point_operations(self):
        origin = Point.p0()
        self.assertEqual(origin.x, 0)
        self.assertEqual(origin.y, 0)

        a = Point.p(1, 2)
        b = Point.p(3, 4)
        length = Point.length(a, b)
        expected_length = math.sqrt((1 - 3) ** 2 + (2 - 4) ** 2)
        self.assertEqual(length, expected_length)

    def test_line_orientation(self):
        line = Line.l(Point.p(0, 0), Point.p(5, 5))
        self.assertGreater(compare_point_to_line(Point.p(5, 0), line), 0)
        self.assertEqual(compare_point_to_line(Point.p(5, 5), line), 0)
        self.assertLess(compare_point_to_line(Point.p(0, 5), line), 0)

    def test_circle_area(self):
        center = Point.p(0, 0)
        radius = 5
        circle = Circle.c(center, radius)
        self.assertEqual(circle.radius, radius)

        unit_circle = Circle.unitCircle()
        self.assertEqual(unit_circle.radius, 1)
        self.assertEqual(unit_circle.center.x, 0)
        self.assertEqual(unit_circle.center.y, 0)

    def test_rectangle_inclusions(self):
        rect = Rectangle(Point.p(0, 0), Point.p(10, 10))
        self.assertTrue(rect.test(Point.p(5, 5)))
        self.assertFalse(rect.test(Point.p(10, 10)))
        self.assertTrue(rect.includeBorders(True), Point.p(10, 10))
        self.assertFalse(rect.test(Point.p(15, 15)))


if __name__ == '__main__':
    unittest.main()
