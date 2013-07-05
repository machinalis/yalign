# -*- coding: utf-8 -*-

import unittest
from yalign.sequencealigner import SequenceAligner


class BaseTestAlignSequences(object):
    def setUp(self):
        self.align = SequenceAligner(self.weight, self.gap_penalty)

    def test_EmptySequences(self):
        align = self.align([], [])
        self.assertEqual(align, [])

    def test_FillsWithGaps(self):
        align = self.align(self.xs, [])
        expected = [(i, None, self.gap_penalty) for i, x in enumerate(self.xs)]
        self.assertEqual(align, expected)

    def test_mingaps(self):
        weight = lambda a, b: 0
        gap = 1
        align = self.align(self.xs, self.ys, weight, gap)
        self.assertEqual(len(align), max(len(self.xs), len(self.ys)))

    def test_maxgaps(self):
        weight = lambda a, b: 1
        gap = 0
        align = self.align(self.xs, self.ys, weight, gap)
        self.assertEqual(len(align), len(self.xs) + len(self.ys))

    def test_no_negative_gap_penalty(self):
        gap = self.gap_penalty
        if gap > 0.0:
            gap = -gap
        gap = gap - 1
        self.assertRaises(ValueError, self.align, self.xs, self.ys,
                                                               self.weight, gap)

    def test_no_negative_weights(self):
        def proxy(a, b):
            w = self.weight(a, b)
            if w > 0.0:
                w = -w
            w = w - 1
        self.assertRaises(ValueError, self.align, self.xs, self.ys,
                                                        proxy, self.gap_penalty)

    def test_weight_not_called_twice(self):
        seen = set()

        def proxy(i, j):
            self.assertNotIn((i, j), seen)
            seen.add((i, j))
            a, b = self.xs[i], self.ys[j]
            return self.weight(a, b)
        self.align(range(len(self.xs)), range(len(self.ys)),
                   proxy, self.gap_penalty)

    def test_weight_not_for_all(self):
        seen = set()

        def proxy(i, j):
            seen.add((i, j))
            a, b = self.xs[i], self.ys[j]
            return self.weight(a, b)
        self.align(range(len(self.xs)), range(len(self.ys)),
                   proxy, self.gap_penalty)
        self.assertLess(len(seen), len(self.xs) * len(self.ys))


class BaseTestAlignSequences_ED(BaseTestAlignSequences):
    gap_penalty = 1

    def weight(self, a, b):
        if a == b:
            return 0
        else:
            return 1

    def test_known_examples(self):
        align = self.align(self.xs, self.ys)
        score = sum(cost for _, _, cost in align)
        self.assertEqual(score, self.expected_cost)


class TestAlignSequences_EditDistance1(BaseTestAlignSequences_ED, unittest.TestCase):
    # Example taken from http://en.wikipedia.org/wiki/Levenshtein_distance
    xs = "kitten"
    ys = "sitting"
    expected_cost = 3


class TestAlignSequences_EditDistance2(BaseTestAlignSequences_ED, unittest.TestCase):
    # Example taken from http://en.wikipedia.org/wiki/Levenshtein_distance
    xs = "Saturday"
    ys = "Sunday"
    expected_cost = 3


class TestAlignSequences_WeirdAlignment(BaseTestAlignSequences, unittest.TestCase):
    xs = u"house µa µb µc µd µe".split()
    ys = u"casa  µ1 µ2 µ3 µ4".split()
    gap_penalty = 0.4999
    C = 1 - 0.19813635000000002

    def weight(self, a, b):
        if a == u"house" and b == u"casa":
            return self.C
        else:
            return 1.0

    def test_alignments_ok(self):
        align = self.align(self.xs, self.ys)
        x0 = list(align[0][0:2])
        self.assertEqual([0, 0], x0)
        for a, b, c in align[1:]:
            self.assertTrue(a is None or b is None)  # Rest is unaligned

    def test_weight_not_for_all(self):
        pass  # 0.4999 ensures that corners are explored


class TestAlignSequences_Sintetic1(BaseTestAlignSequences, unittest.TestCase):
    xs = "AabbbaabaAbbabaAA"
    ys = "BBB"
    gap_penalty = 1

    def weight(self, a, b):
        if a == "A" and b == "B":
            return 0
        return 1

    def test_all_As_and_Bs_are_aligned(self):
        align = self.align(self.xs, self.ys)
        for i, j, cost in align:
            if i is None or j is None:
                continue
            self.assertEqual(self.xs[i], "A")
            self.assertEqual(self.ys[j], "B")


class TestAlignSequences_Sintetic2(BaseTestAlignSequences, unittest.TestCase):
    xs = "Aaaaaa"
    ys = "aaaaaA"
    gap_penalty = 2000

    def weight(self, a, b):
        if a == "A" and b == "A":
            return 3999
        return 4000

    def test_As_are_aligned(self):
        align = self.align(self.xs, self.ys)
        for i, j, cost in align:
            if i is None or j is None:
                continue
            self.assertEqual(self.xs[i], "A")
            self.assertEqual(self.ys[j], "A")

    def test_weight_not_for_all(self):
        pass


if __name__ == "__main__":
    unittest.main()
