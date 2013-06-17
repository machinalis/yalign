# -*- coding: utf-8 -*-

import unittest
from StringIO import StringIO

from yalign.train import read_lines, alignments

class TestReadLines(unittest.TestCase):

    def check(self, lines, values):
        for idx, value in enumerate(values):
            self.assertEquals((idx, value), lines[idx])

    def test_read_empty_corpus(self):
        corpus = StringIO("")
        src, tgt = read_lines(corpus, 10)
        self.assertEquals(0, len(src))
        self.assertEquals(0, len(tgt))

    def test_read_one_line(self):
        corpus = StringIO("Hello\nHola")
        src, tgt = read_lines(corpus, 10)
        self.assertEquals(1, len(src))
        self.assertEquals(1, len(tgt))
        self.check(src, [u'Hello'])
        self.check(tgt, [u'Hola'])

    def test_read_more_than_one_line(self):
        corpus = StringIO("Hello\nHola\nBye\nAdios")
        src, tgt = read_lines(corpus, 10)
        self.assertEquals(2, len(src))
        self.assertEquals(2, len(tgt))
        self.check(src, [u'Hello', u'Bye'])
        self.check(tgt, [u'Hola', u'Adios'])

    def test_read_continuously(self):
        corpus = StringIO("Hello\nHola\nBye\nAdios")
        src, tgt = read_lines(corpus, 1)
        self.assertEquals(1, len(src))
        self.assertEquals(1, len(tgt))
        self.check(src, [u'Hello'])
        self.check(tgt, [u'Hola'])
        src, tgt = read_lines(corpus, 1)
        self.assertEquals(1, len(src))
        self.assertEquals(1, len(tgt))
        self.check(src, [u'Bye'])
        self.check(tgt, [u'Adios'])


class TestAlignments(unittest.TestCase):

    def test_empty_alignments(self):
        src, tgt = [], []
        self.assertEquals([], list(alignments(src, tgt)))

    def test_single_values(self):
        src = [(5, u'hello')]
        tgt = []
        self.assertEquals([], list(alignments(src, tgt)))
        tgt = [(1, u'hola')]
        self.assertEquals([], list(alignments(src, tgt)))
        tgt = [(5, u'hola')]
        self.assertEquals([(0, 0)], list(alignments(src, tgt)))

if __name__ == "__main__":
    unittest.main()
