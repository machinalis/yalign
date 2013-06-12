# -*- coding: utf-8 -*-

import unittest
from nwalign import AlignSequences


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


if __name__ == "__main__":
    unittest.main()


"""
COMPLETE!!!

def _edit_cost(a, b):
    if a == b:
        return 0
    return 10


def edit_distance(xs, ys):
    return AlignSequences(xs, ys, _edit_cost)


_dna_distance = {"A": {"A": 10, "G": -1, "C": -3, "T": -4},
    "G": {"A": -1, "G": 7, "C": -5, "T": -3},
    "C": {"A": -3, "G": -5, "C": 9, "T": 0},
    "T": {"A": -4, "G": -3, "C": 0, "T": 8},
}


_test_distance = {"A": {"A": 10, "G": -1, "C": -3, "T": -4},
    "G": {"A": -1, "G": -1, "C": -5, "T": -3},
    "C": {"A": -3, "G": -5, "C": -1, "T": -2},
    "T": {"A": -4, "G": -3, "C": -2, "T": -1},
}


def dna_align(xs, ys):
    return AlignSequences(xs, ys, lambda a, b: -_dna_distance[a][b], 5)


def pretty_align(xs, ys, align):
    sx = []
    sy = []
    for i, j, cost in align:
        if i is None:
            sx.append("-")
        elif i != len(xs):
            sx.append(xs[i])
        if j is None:
            sy.append("-")
        elif j != len(ys):
            sy.append(ys[j])
    return "".join(sx), "".join(sy)


if __name__ == "__main__":
    xs = "AGACTAGTTAC"
    ys = "CGAGACGT"
    a = dna_align(xs, ys)
    print a
    print "{}\n{}".format(*pretty_align(xs, ys, a))
    print -sum(cost for _, _, cost in a)
"""
