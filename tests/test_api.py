# -*- coding: utf-8 -*-

import unittest
import os

from yalign.api import *
from yalign.weightfunctions import TUScore
from helpers import default_tuscore


class TestAlignDocuments(unittest.TestCase):

    def setUp(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.parallel_corpus = os.path.join(base_path, "data", "canterville.txt")
        classifier_filepath = default_tuscore()
        self.tu_scorer = TUScore(classifier_filepath)
        self.align_documents = AlignDocuments(self.tu_scorer, 0.2, 1)
        
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
        fn = AlignDocuments(self.tu_scorer, 0.2)._threshold
        self.assertRaises(ValueError, fn, None)         
        fn = AlignDocuments(self.tu_scorer, threshold=1)._threshold
        self.assertEquals(1, fn(None))
        self.assertEquals(.5, fn(.5))
    
    def test_gap_penalty(self):
        fn = AlignDocuments(self.tu_scorer,threshold=1)._gap_penalty
        self.assertRaises(ValueError, fn, None)         
        fn = AlignDocuments(self.tu_scorer, gap_penalty=1)._gap_penalty
        self.assertEquals(1, fn(None))
        self.assertEquals(.5, fn(.5))
    
    def test_items(self):
        xs = self.align_documents._items(u'A B C D'.split())
        self.assertEquals([pos for _, pos in xs], [0, .25, .5, .75])    
        self.assertEquals([val for val, _ in xs], [u'A', u'B', u'C', u'D'])    
    
    def test_weight(self):
        x = self.align_documents.weight((u'Hello', 0.2), (u'Hola', 0.2))    
        self.assertTrue(0 < x < 1)
        y = self.align_documents.weight((u'Hello', 0), (u'Hola', 0.5))    
        self.assertTrue(0 < y < x)


if __name__ == "__main__":
    unittest.main()
