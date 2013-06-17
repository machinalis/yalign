# -*- coding: utf-8 -*-

import unittest
from yalign.evaluation import F_score, precision, recall


class TestFScore(unittest.TestCase):

    def test_recall(self):
        self.assertEquals(0, recall([], []))
        self.assertEquals(0, recall([], [1]))
        self.assertAlmostEquals(1, recall([1], [1]))
        self.assertAlmostEquals(.5, recall([1], [1, 2]))
        self.assertAlmostEqual(1, recall([1, 2], [1, 2]))
        self.assertAlmostEquals(1, recall([1, 2], [1]))

    def test_precision(self):
        self.assertEquals(0, precision([], []))
        self.assertEquals(0, precision([], [1]))
        self.assertAlmostEquals(1, precision([1], [1]))
        self.assertAlmostEquals(1, precision([1], [1, 2]))
        self.assertAlmostEqual(1, precision([1, 2], [1, 2]))
        self.assertAlmostEquals(.5, precision([1, 2], [1]))

    def test_F_score(self):
        delta = 0.0001
        self.assertEquals(0, F_score([], [])[0])
        self.assertEquals(0, F_score([], [1])[0])
        self.assertAlmostEquals(0.9181, F_score([1], [1])[0], delta=delta)
        self.assertAlmostEquals(0.8416, F_score([1], [1, 2])[0], delta=delta)
        self.assertAlmostEqual(0.9181, F_score([1, 2], [1, 2])[0], delta=delta)
        self.assertAlmostEquals(0.4809, F_score([1, 2], [1])[0], delta=delta)

if __name__ == "__main__":
    unittest.main()
