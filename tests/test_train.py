# -*- coding: utf-8 -*-

import unittest
from StringIO import StringIO

from yalign.train import read_lines, alignments, html_to_corpus, text_to_corpus

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


class TestHTMLToCorpus(unittest.TestCase):

    def test_extract(self):
        html = "<html><head></head><body><p>Hello Peter</p></body></html>"
        self.assertEquals([u'Hello Peter'], html_to_corpus(html))
        html = "<html><head></head><body><p>Hello Peter. Go for gold.</p></body></html>"
        self.assertEquals([u'Hello Peter.', u'Go for gold.'], html_to_corpus(html))

    def test_newlines(self):
        html = "<html><head></head>\n\n<body><p>\nHello Peter.\n\n\n Go for gold.\n</p>\n</body></html>"
        self.assertEquals([u'Hello Peter.', u'Go for gold.'], html_to_corpus(html))

    def test_remove_whitespacing(self):
        html = "<html><head></head><body><p>Wow\n\tWhat now?\t\t</p></body></html>"
        self.assertEquals([u'Wow What now?'], html_to_corpus(html))

    def test_sentence_splitting(self):
        html = "<html><head></head><body><p>Wow!! I did not know! Are you sure?</p></body></html>"
        self.assertEquals([u'Wow!!', u'I did not know!', u'Are you sure?'], html_to_corpus(html))


class TestTextToCorpus(unittest.TestCase):

    def test_sentence_splitting(self):
        lines = ['So there we were.', 'In the middle of nowhere!', 'Where to from here?', 'I cried..']
        paragraph = ' '.join(lines)
        self.assertEquals(lines, text_to_corpus(paragraph))

    def test_remove_whitespacing(self):
        html = "Wow\n\tWhat now?\t\t"
        self.assertEquals([u'Wow What now?'], text_to_corpus(html))

if __name__ == "__main__":
    unittest.main()
