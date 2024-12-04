import math
from abc import ABC, abstractmethod
import unittest

class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    @staticmethod
    def create(x: float, y: float) -> 'Point':
        return Point(x, y)

    @staticmethod
    def origin() -> 'Point':
        return Point(0, 0)

    @staticmethod
    def subtract_x(a: 'Point', b: 'Point') -> float:
        return a.x - b.x

    @staticmethod
    def subtract_y(a: 'Point', b: 'Point') -> float:
        return a.y - b.y

    @staticmethod
    def distance(a: 'Point', b: 'Point') -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

class Line:
    def __init__(self, p1: Point, p2: Point) -> None:
        self.p1 = p1
        self.p2 = p2

    @staticmethod
    def create(p1: Point, p2: Point) -> 'Line':
        return Line(p1, p2)

    @staticmethod
    def x_axis() -> 'Line':
        return Line(Point.create(0, 0), Point.create(1, 0))

    @staticmethod
    def y_axis() -> 'Line':
        return Line(Point.create(0, 0), Point.create(0, 1))

def compare_point_to_line(point: Point, line: Line) -> float:
    a = line.p2.y - line.p1.y
    b = line.p1.x - line.p2.x
    c = -(a * line.p1.x + b * line.p1.y)
    return a * point.x + b * point.y + c

class Circle:
    def __init__(self, center: Point, radius: float) -> None:
        self.center = center
        self.radius = radius

    @staticmethod
    def create(center: Point, radius: float) -> 'Circle':
        return Circle(center, radius)

    @staticmethod
    def unit_circle() -> 'Circle':
        return Circle(Point.origin(), 1)

class Area(ABC):
    @abstractmethod
    def test(self, point: Point) -> bool:
        pass

    @abstractmethod
    def include_borders(self, flag: bool) -> bool:
        pass

class Rectangle(Area):
    def __init__(self, p1: Point, p2: Point) -> None:
        self.p1 = p1
        self.p2 = p2

    def test(self, point: Point) -> bool:
        return self.p1.x < point.x < self.p2.x and self.p1.y < point.y < self.p2.y

    def include_borders(self, flag: bool) -> bool:
        if flag:
            return self.p1.x <= point.x <= self.p2.x and self.p1.y <= point.y <= self.p2.y
        else:
            return self.test(point)

class TestGeometry(unittest.TestCase):
    def test_point_operations(self) -> None:
        origin = Point.origin()
        self.assertEqual(origin.x, 0)
        self.assertEqual(origin.y, 0)

        a = Point.create(1, 2)
        b = Point.create(3, 4)
        length = Point.distance(a, b)
        expected_length = math.sqrt((1 - 3) ** 2 + (2 - 4) ** 2)
        self.assertEqual(length, expected_length)

    def test_line_orientation(self) -> None:
        line = Line.create(Point.create(0, 0), Point.create(5, 5))
        self.assertGreater(compare_point_to_line(Point.create(5, 0), line), 0)
        self.assertEqual(compare_point_to_line(Point.create(5, 5), line), 0)
        self.assertLess(compare_point_to_line(Point.create(0, 5), line), 0)

    def test_circle_area(self) -> None:
        center = Point.create(0, 0)
        radius = 5
        circle = Circle.create(center, radius)
        self.assertEqual(circle.radius, radius)

        unit_circle = Circle.unit_circle()
        self.assertEqual(unit_circle.radius, 1)
        self.assertEqual(unit_circle.center.x, 0)
        self.assertEqual(unit_circle.center.y, 0)

    def test_rectangle_inclusions(self) -> None:
        rect = Rectangle(Point.create(0, 0), Point.create(10, 10))
        self.assertTrue(rect.test(Point.create(5, 5)))
        self.assertFalse(rect.test(Point.create(10, 10)))
        self.assertTrue(rect.include_borders(True), Point.create(10, 10))
        self.assertFalse(rect.test(Point.create(15, 15)))

if __name__ == '__main__':
    unittest.main()
