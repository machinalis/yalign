# -*- coding: utf-8 -*-

import unittest
import os

from yalign.api import AlignDocuments
from yalign.datatypes import Sentence
from helpers import default_sentence_pair_score


class TestAlignDocuments(unittest.TestCase):

    def setUp(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.parallel_corpus = os.path.join(base_path, "data", "canterville.txt")
        self.classifier, _ = default_sentence_pair_score()
        self.align_documents = AlignDocuments(self.classifier, 0.2, 1)

    def test_boundries(self):
        self.assertEquals([], self.align_documents([],[]))
        self.assertEquals(1, len(self.align_documents(['A'],[])))
        self.assertEquals(1, len(self.align_documents([],['B'])))

    def test_filter_by_threshold(self):
        fn = self.align_documents._filter_by_threshold
        a, b, c = (0,0,.005),(0,0,.25),(0,0,.5)
        self.assertEquals([a, b, c], fn([a,b,c], 1))
        self.assertEquals([a, b, c], fn([a,b,c], .5))
        self.assertEquals([a, b], fn([a,b,c], .4))
        self.assertEquals([a, ], fn([a,b,c], .2))
        self.assertEquals([], fn([a,b,c], 0))

    def test_threshold(self):
        fn = AlignDocuments(self.classifier, 0.2)._threshold
        self.assertRaises(ValueError, fn, None)
        fn = AlignDocuments(self.classifier, threshold=1)._threshold
        self.assertEquals(1, fn(None))
        self.assertEquals(.5, fn(.5))

    def test_gap_penalty(self):
        fn = AlignDocuments(self.classifier,threshold=1)._gap_penalty
        self.assertRaises(ValueError, fn, None)
        fn = AlignDocuments(self.classifier, gap_penalty=1)._gap_penalty
        self.assertEquals(1, fn(None))
        self.assertEquals(.5, fn(.5))

    def test_weight(self):
        x = self.align_documents.weight(Sentence(u'Hello', position=0.2), Sentence(u'Hola', position=0.2))
        self.assertTrue(0 < x < 1)
        y = self.align_documents.weight(Sentence(u'Hello', position=0), Sentence(u'Hola', position=0.5))
        self.assertTrue(0 < y < x)


if __name__ == "__main__":
    unittest.main()
