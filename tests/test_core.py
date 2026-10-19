import math
import unittest

from li_chao_tree_utils import LiChaoTree, Line


class TestEmpty(unittest.TestCase):
    def test_query_none_when_empty(self):
        t = LiChaoTree(0, 10)
        self.assertIsNone(t.query_max(5))
        self.assertEqual(t.query_max_value(5), float("-inf"))


class TestSingleLine(unittest.TestCase):
    def test_single_line_returned(self):
        t = LiChaoTree(0, 10)
        line = t.add_line(2.0, 1.0)  # y = 2x + 1
        self.assertEqual(line, Line(2.0, 1.0))
        got = t.query_max(3)
        self.assertIsNotNone(got)
        self.assertEqual(got.m, 2.0)
        self.assertEqual(got.b, 1.0)
        self.assertAlmostEqual(t.query_max_value(3), 7.0)

    def test_single_line_at_endpoints(self):
        t = LiChaoTree(-5, 5)
        t.add_line(-1.0, 4.0)
        self.assertAlmostEqual(t.query_max_value(-5), 9.0)
        self.assertAlmostEqual(t.query_max_value(5), -1.0)


class TestTwoLines(unittest.TestCase):
    def test_two_lines_crossing_inside(self):
        # y = x and y = -x + 4 cross at x=2, y=2
        t = LiChaoTree(0, 4)
        t.add_line(1.0, 0.0)
        t.add_line(-1.0, 4.0)
        self.assertAlmostEqual(t.query_max_value(0), 4.0)
        self.assertAlmostEqual(t.query_max_value(4), 4.0)
        self.assertAlmostEqual(t.query_max_value(2), 2.0)
        # left of crossing, second line wins
        self.assertAlmostEqual(t.query_max_value(1), 3.0)
        # right of crossing, first line wins
        self.assertAlmostEqual(t.query_max_value(3), 3.0)

    def test_two_parallel_lines(self):
        t = LiChaoTree(0, 10)
        t.add_line(1.0, 0.0)  # lower
        t.add_line(1.0, 5.0)  # upper, dominates everywhere
        for x in range(0, 11):
            self.assertAlmostEqual(t.query_max_value(x), x + 5.0)

    def test_dominating_line_added_first(self):
        t = LiChaoTree(0, 10)
        t.add_line(1.0, 5.0)  # upper
        t.add_line(1.0, 0.0)  # lower
        for x in range(0, 11):
            self.assertAlmostEqual(t.query_max_value(x), x + 5.0)


class TestFlatLines(unittest.TestCase):
    def test_flat_lines(self):
        t = LiChaoTree(0, 5)
        t.add_line(0.0, 1.0)
        t.add_line(0.0, 3.0)
        t.add_line(0.0, 2.0)
        self.assertAlmostEqual(t.query_max_value(2), 3.0)
        self.assertAlmostEqual(t.query_max_value(0), 3.0)
        self.assertAlmostEqual(t.query_max_value(5), 3.0)


class TestNegativeRange(unittest.TestCase):
    def test_negative_coordinates(self):
        t = LiChaoTree(-10, 10)
        t.add_line(1.0, 0.0)
        t.add_line(-1.0, 0.0)
        self.assertAlmostEqual(t.query_max_value(-10), 10.0)
        self.assertAlmostEqual(t.query_max_value(10), 10.0)
        self.assertAlmostEqual(t.query_max_value(0), 0.0)


class TestValidation(unittest.TestCase):
    def test_bad_range_raises(self):
        with self.assertRaises(ValueError):
            LiChaoTree(5, 0)

    def test_query_out_of_range_raises(self):
        t = LiChaoTree(0, 10)
        with self.assertRaises(ValueError):
            t.query_max(-1)
        with self.assertRaises(ValueError):
            t.query_max(11)
        with self.assertRaises(ValueError):
            t.query_max_value(11)

    def test_query_non_integer_raises(self):
        t = LiChaoTree(0, 10)
        with self.assertRaises(ValueError):
            t.query_max(3.5)
        with self.assertRaises(ValueError):
            t.query_max_value(3.5)


class TestSinglePointRange(unittest.TestCase):
    def test_single_point(self):
        t = LiChaoTree(3, 3)
        t.add_line(1.0, 0.0)
        t.add_line(0.0, 4.0)  # wins at x=3 (value 4 > 3)
        self.assertAlmostEqual(t.query_max_value(3), 4.0)
        got = t.query_max(3)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got.eval(3), 4.0)


class TestManyLines(unittest.TestCase):
    def test_many_lines_brute_force(self):
        t = LiChaoTree(-20, 20)
        lines = []
        # a deterministic family of lines
        for i in range(15):
            m = (i - 7) * 0.5
            b = (7 - i) * 1.0
            t.add_line(m, b)
            lines.append((m, b))
        for x in range(-20, 21):
            expected = max(m * x + b for (m, b) in lines)
            self.assertAlmostEqual(
                t.query_max_value(x), expected,
                msg=f"mismatch at x={x}"
            )

    def test_repeated_lines(self):
        t = LiChaoTree(0, 10)
        for _ in range(5):
            t.add_line(2.0, 3.0)
        self.assertAlmostEqual(t.query_max_value(4), 11.0)


class TestLineEquality(unittest.TestCase):
    def test_line_equality_and_hash(self):
        a = Line(1.0, 2.0)
        b = Line(1.0, 2.0)
        c = Line(1.0, 3.0)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, "not a line")

    def test_line_repr(self):
        r = repr(Line(1.5, 2.5))
        self.assertIn("1.5", r)
        self.assertIn("2.5", r)


class TestTieHandling(unittest.TestCase):
    def test_exact_tie_at_midpoint(self):
        # Two lines that tie exactly at the midpoint of the range.
        # y = 2 and y = x - 1 tie at x = 3 with value 2.
        t = LiChaoTree(0, 6)
        t.add_line(0.0, 2.0)
        t.add_line(1.0, -1.0)
        # At x=3 both equal 2; tree must return one of them with value 2.
        self.assertAlmostEqual(t.query_max_value(3), 2.0)
        # At x=0, flat line wins (2 vs -1)
        self.assertAlmostEqual(t.query_max_value(0), 2.0)
        # At x=6, slope line wins (5 vs 2)
        self.assertAlmostEqual(t.query_max_value(6), 5.0)


if __name__ == "__main__":
    unittest.main()
