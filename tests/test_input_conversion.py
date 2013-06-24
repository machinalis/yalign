# -*- coding: utf-8 -*-


import unittest
from StringIO import StringIO

from yalign.input_conversion import documents_from_parallel_corpus


def reader(N):
    return StringIO('\n'.join([str(x) for x in xrange(N)]))


class TestDocumentsFromParallelCorpus(unittest.TestCase):

    def test_empty_input(self):
        self.assertEquals([], list(documents_from_parallel_corpus(StringIO())))


    def test_document_sizes_between_min_and_max(self):
        cnt, m, n = 0, 5, 10
        N = 10000
        for A, B in documents_from_parallel_corpus(reader(N * 2), m, n):
            self.assertTrue(m <= len(A) <= n)
            self.assertTrue(m <= len(B) <= n)
            cnt += 1
        self.assertTrue(N / n <= cnt <= N / m)

    def test_no_zero_as_min(self):
        for A, B in documents_from_parallel_corpus(reader(20), 0, 1):
            self.assertTrue(1 <= len(A) <= 1)
            self.assertTrue(1 <= len(B) <= 1)

if __name__ == "__main__":
    unittest.main()
