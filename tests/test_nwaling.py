# -*- coding: utf-8 -*-

import unittest
from yalign.nwalign import AlignSequences


class BaseTestAlignSequences(object):
    def test_EmptySequences(self):
        align = AlignSequences([], [], self.weight, self.gap_penalty)
        self.assertEqual(align, [])

    def test_FillsWithGaps(self):
        align = AlignSequences(self.xs, [], self.weight, self.gap_penalty)
        expected = [(i, None, self.gap_penalty) for i, x in enumerate(self.xs)]
        self.assertEqual(align, expected)

    def test_mingaps(self):
        weight = lambda a, b: 0
        gap = -1
        align = AlignSequences(self.xs, self.ys, weight, gap)
        self.assertEqual(len(align), max(len(self.xs), len(self.ys)))

    def test_maxgaps(self):
        weight = lambda a, b: 0
        gap = 1
        align = AlignSequences(self.xs, self.ys, weight, gap)
        self.assertEqual(len(align), len(self.xs) + len(self.ys))


class TestAlignSequences_EditDistance(BaseTestAlignSequences, unittest.TestCase):
    xs = "abcbbabaaa"
    ys = "abbccc"
    gap_penalty = 1

    def weight(self, a, b):
        if a == b:
            return 0
        else:
            return 1

    def test_known_examples_1(self):
        """
        Example taken from http://en.wikipedia.org/wiki/Levenshtein_distance
        """
        align = AlignSequences("kitten", "sitting",
                               self.weight, self.gap_penalty, minimize=True)
        score = sum(cost for _, _, cost in align)
        self.assertEqual(score, 3)


    def test_known_examples_2(self):
        """
        Example taken from http://en.wikipedia.org/wiki/Levenshtein_distance
        """
        align = AlignSequences("Saturday", "Sunday",
                               self.weight, self.gap_penalty, minimize=True)
        score = sum(cost for _, _, cost in align)
        self.assertEqual(score, 3)


class TestAlignSequences_Sintetic(BaseTestAlignSequences, unittest.TestCase):
    xs = "AabbbaabaAbbabaAA"
    ys = "BBB"
    gap_penalty = 0

    def weight(self, a, b):
        if a == "A" and b == "B":
            return 1
        return -1

    def test_all_As_and_Bs__are_aligned(self):
        align = AlignSequences(self.xs, self.ys, self.weight, self.gap_penalty)
        for i, j, cost in align:
            if i is None or j is None:
                continue
            self.assertEqual(self.xs[i], "A")
            self.assertEqual(self.ys[j], "B")



if __name__ == "__main__":
    unittest.main()
