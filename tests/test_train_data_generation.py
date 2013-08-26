# -*- coding: utf-8 -*-

import unittest

from StringIO import StringIO
from mock import patch
from yalign.datatypes import Sentence
from yalign.train_data_generation import *
from yalign.train_data_generation import _aligned_samples, _misaligned_samples, _reorder, _random_range


def swap_start_and_end(xs):
    xs[0], xs[-1] = xs[-1], xs[0]


def reader(N):
    return StringIO('\n'.join([str(x) for x in xrange(N)]))


def sentences(xs):
    return [Sentence([unicode(x)]) for x in xs]


class TestTrainingAlignmentsFromDocuments(unittest.TestCase):

    def test_empty_inputs(self):
        samples = list(training_alignments_from_documents([], []))
        self.assertEquals(0, len(samples))

    def test_samples_are_alignments_and_misalignments(self):
        samples = list(training_alignments_from_documents(sentences([u'A', u'B', u'C']),
                                        sentences([u'X', u'Y', u'Z'])))
        self.assertEquals(6, len(samples))
        self.assertEquals(3, len([x for x in samples if x.aligned]))
        self.assertEquals(3, len([x for x in samples if not x.aligned]))

    def test_documents_equal_length(self):
        try:
            list(training_alignments_from_documents([], sentences([u'A'])))
            self.assertFalse("Failed to raise exception")
        except ValueError as e:
            self.assertEqual("Documents must be the same size", str(e))


class TestAlignedSamples(unittest.TestCase):
    def test_empty_alignments(self):
        A, B = [], []
        samples = list(_aligned_samples(A, B, []))
        self.assertEquals(0, len(samples))

    def test_sample_values(self):
        A, B = sentences([u'A', u'B']), sentences([u'Y', u'Z'])
        samples = list(_aligned_samples(A, B, [(0, 1), (1, 0)]))
        # Note alignments swapped so A -> Z and B-> Y
        s0 = SentencePair(Sentence([u'A']), Sentence([u'Z']), aligned=True)
        s1 = SentencePair(Sentence([u'B']), Sentence([u'Y']), aligned=True)
        self.assertEquals(2, len(samples))
        self.assertEquals([s0, s1], samples)


class TestNonAlignedSamples(unittest.TestCase):

    def test_empty_alignments(self):
        A, B = [], []
        samples = list(_misaligned_samples(A, B, []))
        self.assertEquals(0, len(samples))

    def test_one_alignment(self):
        # no misalignments when we have only one alignment
        A, B = sentences(['A']), sentences(['Z'])
        samples = list(_misaligned_samples(A, B, [(0, 0)]))
        self.assertEquals(0, len(samples))

    def test_sample_values(self):
        A, B = sentences([u'A', u'B']), sentences([u'Y', u'Z'])
        samples = list(_misaligned_samples(A, B, [(0, 0), (1, 1)]))
        s0 = SentencePair(Sentence([u'A']), Sentence([u'Z']))
        s1 = SentencePair(Sentence([u'B']), Sentence([u'Y']))
        self.assertEquals(2, len(samples))
        for sample in samples:
            self.assertTrue(sample in [s0, s1])

    def test_randomly(self):
        for i in xrange(1000):
            n = random.randint(2, 100)
            A, B = sentences(xrange(n)), sentences(xrange(n))
            alignments = zip(range(n), range(n))
            samples = list(_misaligned_samples(A, B, alignments))
            self.assertEquals(n, len(samples))
            pairs = set([(a[0], b[0]) for a, b in samples])
            #No duplicates
            self.assertEquals(len(pairs), len(samples))

class TestReorder(unittest.TestCase):

    def test_reoroder(self):
        self.assertEquals([], _reorder([], []))
        self.assertEquals([0], _reorder([0], [0]))
        self.assertEquals([0, 1], _reorder([0, 1], [0, 1]))
        self.assertEquals([1, 0], _reorder([0, 1], [1, 0]))
        self.assertEquals([1, 2, 0], _reorder([0, 1, 2], [2, 0, 1]))

    def test_indexes_size_correct(self):
        self.assertRaises(ValueError, _reorder, [1, 2], [0])
        self.assertRaises(ValueError, _reorder, [1], [0, 1])


class TestRandomAlign(unittest.TestCase):
    def test_boundries(self):
        self.assertEquals(([], [], []), training_scrambling_from_documents([], []))
        self.assertEquals((sentences(['Y']), [], [(0, None)]), training_scrambling_from_documents(sentences(['Y']), []))
        self.assertEquals(([], sentences(['Z']), [(None, 0)]), training_scrambling_from_documents([], sentences(['Z'])))

    @patch('random.shuffle')
    def test_unshuffled(self, mock_shuffle):
        mock_shuffle.side_effect = lambda x: x
        A, B = sentences(['A', 'B']), sentences(['Y', 'Z'])
        self.assertEquals((A, B, [(0, 0), (1, 1)]), training_scrambling_from_documents(A, B))
        A, B = sentences(['A', 'B']), sentences(['Y'])
        self.assertEquals((A, B, [(0, 0), (1, None)]), training_scrambling_from_documents(A, B))
        A, B = sentences(['A']), sentences(['Y', 'Z'])
        self.assertEquals((A, B, [(None, 1), (0, 0)]), training_scrambling_from_documents(A, B))

    @patch('random.randint')
    @patch('random.shuffle')
    def test_shuffled(self, mock_shuffle, mock_randint):
        mock_randint.return_value = 2
        mock_shuffle.side_effect = swap_start_and_end
        A, B = sentences(['A', 'B']), sentences(['Y', 'Z'])
        self.assertEquals((sentences(['B', 'A']), sentences(['Z', 'Y']), [(0, 0), (1, 1)]),
                          training_scrambling_from_documents(A, B))
        A, B = sentences(['A', 'B']), sentences(['Y'])
        self.assertEquals((sentences(['B', 'A']), B, [(0, None), (1, 0)]), training_scrambling_from_documents(A, B))
        A, B = sentences(['A']), sentences(['Y', 'Z'])
        self.assertEquals((A, sentences(['Z', 'Y']), [(None, 0), (0, 1)]), training_scrambling_from_documents(A, B))


class TestRandomRange(unittest.TestCase):

    def test_boundries(self):
        self.assertEquals([], _random_range(-1))
        self.assertEquals([], _random_range(0))
        self.assertEquals([0], _random_range(1))
        self.assertEquals(5, len(_random_range(5)))

    def test_span(self):
        # any span <= 1 will lead to no shuffling
        self.assertEquals(range(5), _random_range(5, span=1))
        self.assertEquals(range(5), _random_range(5, span=0))
        self.assertEquals(range(5), _random_range(5, span=-1))
        self.assertEquals(3, len(_random_range(3, span=1000)))

    def test_some_shuffling_happens(self):
        self.assertNotEquals(range(100), _random_range(100))

    @patch('random.randint')
    @patch('random.shuffle')
    def test_shuffling(self, mock_shuffle, mock_randint):
        mock_randint.return_value = 2
        mock_shuffle.side_effect = swap_start_and_end
        self.assertEquals([1, 0, 3, 2], _random_range(4))
        self.assertEquals([1, 0, 3, 2, 4], _random_range(5))


if __name__ == "__main__":
    unittest.main()
