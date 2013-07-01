# -*- coding: utf-8 -*-

import os
import json
import tempfile
import unittest
import subprocess
from itertools import izip

from yalign import yalignmodel
from yalign.evaluation import *
from helpers import default_sentence_pair_score
from yalign.input_conversion import parallel_corpus_to_documents


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
        self.assertAlmostEquals(1, F_score([1], [1])[0], delta=delta)
        self.assertAlmostEquals(0.9901, F_score([1], [1, 2])[0], delta=delta)
        self.assertAlmostEqual(1, F_score([1, 2], [1, 2])[0], delta=delta)
        self.assertAlmostEquals(0.5024, F_score([1, 2], [1])[0], delta=delta)

    def test_beta_value(self):
        # Should get a perfect score:
        self.assertEquals(1, F_score([1], [1], beta=1)[0])
        self.assertEquals(1, F_score([1], [1], beta=.2)[0])
        a = F_score([1, 2], [1], beta=.2)[0]
        b = F_score([1, 2], [1], beta=.25)[0]
        #lower beta give more emphasis to precision
        self.assertTrue(a < b)


class TestEvaluate(unittest.TestCase):

    def setUp(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.parallel_corpus = os.path.join(base_path, "data", "canterville.txt")
        metadata_filename = os.path.join(base_path, "data", "metadata.json")
        metadata = json.load(open(metadata_filename))
        self.gap_penalty = metadata['gap_penalty']
        self.threshold = metadata['threshold']
        self.classifier, _ = default_sentence_pair_score()

    def test_evaluate(self):
        stats = evaluate(self.parallel_corpus,
                         self.classifier,
                         self.gap_penalty,
                         self.threshold, 20)
        for x, y in izip(stats['max'], stats['mean']):
            self.assertTrue(x > y > 0)
        for x in stats['std']:
            self.assertTrue(x > 0)

if __name__ == "__main__":
    unittest.main()
