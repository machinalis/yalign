import unittest
import os
import tempfile

from StringIO import StringIO
from yalign.tu import TU
from yalign import tuscore
from yalign.weightfunctions import TUScore, WordScore
from yalign.optimize import optimize, _items, _weight, _optimize_threshold 
from helpers import default_tuscore

class TestOptimize(unittest.TestCase):   

    def setUp(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.parallel_corpus = os.path.join(base_path, "data", "canterville.txt")
        classifier_filepath = default_tuscore()
        self.tu_scorer = TUScore(classifier_filepath)
   
    def test_optimize(self):
        score, gap_penalty, threshold = optimize(open(self.parallel_corpus), self.tu_scorer) 
        self.assertTrue(score > 0 and gap_penalty > 0 and threshold > 0)        

    def test_items(self):
        xs = _items(u'A B C D'.split())
        self.assertEquals([pos for _, pos in xs], [0, .25, .5, .75])    
        self.assertEquals([val for val, _ in xs], [u'A', u'B', u'C', u'D'])    
    
    def test_weight(self):
        x = _weight(self.tu_scorer, (u'Hello', 0.2), (u'Hola', 0.2))    
        self.assertTrue(0 < x < 1)
        y = _weight(self.tu_scorer, (u'Hello', 0), (u'Hola', 0.5))    
        self.assertTrue(0 < y < x)

    def test_optimize_threshold(self):
        self.assertEquals((0, 1), _optimize_threshold(0.5, [], []))        
        xs, ys = [(0,0)], [(0,0,0.2)]
        self.assertEquals( 0.2, _optimize_threshold(0.5, xs, ys)[1])        
        xs, ys = [(0,0),(1,1)], [(0,0,0.2),(1,1,0.3)]
        self.assertEquals( 0.3, _optimize_threshold(0.5, xs, ys)[1])        
        xs, ys = [(0,0),(1,1),(2,2),(3,3)], [(0,0,0.2),(1,1,0.3),(2,3,0.4),(3,2,0.5)]
        # The better threshold should exclude the misalignments
        self.assertEquals(0.3, _optimize_threshold(0.5, xs, ys)[1])        
        

if __name__ == "__main__":
    unittest.main()
