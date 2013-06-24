# -*- coding: utf-8 -*-

import unittest

from StringIO import StringIO
from mock import patch
from yalign.train_data_generation import *
from yalign.train_data_generation import _aligned_samples, _misaligned_samples
from yalign.train_data_generation import _reorder, _random_remove


def swap_start_and_end(xs):
    xs[0], xs[-1] = xs[-1], xs[0]


def reader(N):
    return StringIO('\n'.join([str(x) for x in xrange(N)]))

class TestTrainingSamples(unittest.TestCase):

    def test_empty_input(self):
        samples = list(training_samples(StringIO()))
        self.assertEquals(0, len(samples))

    def test_aligned_and_misaligned_samples(self):
        samples = list(training_samples(reader(8),2,2))
        self.assertEquals(8, len(samples))
        self.assertEquals(4, len([x for x in samples if x[0]]))
        self.assertEquals(4, len([x for x in samples if not x[0]]))

class TestDocuments(unittest.TestCase):

    def test_empty_input(self):
        self.assertEquals([], list(documents(StringIO())))

    def test_document_sizes_between_min_and_max(self):
        cnt, m, n  = 0, 5, 10
        N = 10000
        for A,B in documents(reader(N*2), m, n):
            self.assertTrue(m <= len(A) <= n)
            self.assertTrue(m <= len(B) <= n)
            cnt += 1
        self.assertTrue(N / n <= cnt <= N / m)

    def test_no_zero_as_min(self):
        for A,B in documents(reader(20), 0, 1):
            self.assertTrue(1 <= len(A) <= 1)
            self.assertTrue(1 <= len(B) <= 1)


class TestGenerateSamples(unittest.TestCase):

    def test_empty_inputs(self):
        samples = list(generate_samples([], []))
        self.assertEquals(0, len(samples))

    def test_samples_are_alignments_and_misalignments(self):
        samples = list(generate_samples(['A','B','C'], ['X','Y','Z']))
        self.assertEquals(6, len(samples))
        self.assertEquals(3, len([x for x in samples if x[0]]))
        self.assertEquals(3, len([x for x in samples if not x[0]]))

    def test_documents_equal_length(self):
        try:
            list(generate_samples([],['A']))
            self.assertFalse("Failed to raise exception")
        except ValueError as e:
            self.assertEqual("Documents must be the same size", str(e))


class TestAlignedSamples(unittest.TestCase):

    def test_empty_alignments(self):
        A, B = [], []
        samples = list(_aligned_samples(A, B, []))
        self.assertEquals(0, len(samples))

    def test_sample_values(self):
        A, B = ['A', 'B'], ['Y', 'Z']
        samples = list(_aligned_samples(A, B, [(0, 1),(1, 0)]))
        # Note alignments swapped so A -> Z and B-> Y
        s0 = (1, 2, 0, 'A', 2, 1, 'Z')
        s1 = (1, 2, 1, 'B', 2, 0, 'Y')
        self.assertEquals(2, len(samples))
        self.assertEquals([s0, s1], samples)


class TestNonAlignedSamples(unittest.TestCase):

    def test_empty_alignments(self):
        A, B = [], []
        samples = list(_misaligned_samples(A, B, []))
        self.assertEquals(0, len(samples))

    def test_one_alignment(self):
        # no misalignments when we have only one alignment
        A, B = ['A'], ['Z']
        samples = list(_misaligned_samples(A, B, [(0, 0)]))
        self.assertEquals(0, len(samples))

    def test_sample_values(self):
        A, B = ['A','B'], ['Y','Z']
        samples = list(_misaligned_samples(A, B, [(0, 0),(1, 1)]))
        s0 = (0, 2, 0, 'A', 2, 1, 'Z')
        s1 = (0, 2, 1, 'B', 2, 0, 'Y')
        self.assertEquals(2, len(samples))
        self.assertEquals(set([s0,s1]), set(samples))

    def test_randomly(self):
        for i in xrange(1000):
            n = random.randint(2, 100)
            A, B = list(reader(n)), list(reader(n))
            alignments = zip(A,B)
            samples = list(_misaligned_samples(A, B, alignments))
            self.assertEquals(n, len(samples))
            # check no duplicates
            self.assertEquals(len(set(samples)), len(samples))
            # check only misalignments
            misalignments = [(x[2], x[5]) for x in samples]
            common_samples = len(set(alignments).intersection(set(misalignments)))
            self.assertEquals(0, common_samples)


class TestReorder(unittest.TestCase):

    def test_reoroder(self):
        self.assertEquals([], _reorder([],[]))
        self.assertEquals([0], _reorder([0],[0]))
        self.assertEquals([0,1], _reorder([0,1],[0,1]))
        self.assertEquals([1,0], _reorder([0,1],[1,0]))
        self.assertEquals([1,2,0], _reorder([0,1,2],[2,0,1]))

    def test_indexes_size_correct(self):
        self.assertRaises(ValueError, _reorder, [1,2],[0])
        self.assertRaises(ValueError, _reorder, [1],[0,1])


class TestRandomAlign(unittest.TestCase):

    def test_boundries(self):
        self.assertEquals(([],[],[]), random_align([],[]))
        self.assertEquals((['Y'],[],[(0, None)]), random_align(['Y'],[]))
        self.assertEquals(([],['Z'],[(None, 0)]), random_align([],['Z']))

    @patch('random.shuffle')
    def test_unshuffled(self, mock_shuffle):
        mock_shuffle.side_effect=lambda x: x
        A, B = ['A','B'],['Y','Z']
        self.assertEquals((A, B,[(0, 0),(1, 1)]), random_align(A, B))
        A, B = ['A','B'],['Y']
        self.assertEquals((A, B,[(0, 0),(1, None)]), random_align(A, B))
        A, B = ['A'],['Y','Z']
        self.assertEquals((A, B,[(None, 1), (0, 0)]), random_align(A, B))

    @patch('random.randint')
    @patch('random.shuffle')
    def test_shuffled(self, mock_shuffle, mock_randint):
        mock_randint.return_value=2
        mock_shuffle.side_effect=swap_start_and_end
        A, B = ['A','B'],['Y','Z']
        self.assertEquals((['B','A'],['Z', 'Y'] ,[(0, 0),(1, 1)]), random_align(A, B))
        A, B = ['A','B'],['Y']
        self.assertEquals((['B', 'A'], B,[(0, None), (1, 0)]), random_align(A, B))
        A, B = ['A'],['Y','Z']
        self.assertEquals((A, ['Z', 'Y'],[(None, 0), (0, 1)]), random_align(A, B))

    def test_perc_to_remove_range(self):
       self.assertRaises(ValueError, random_align, [], [], 1.1)
       self.assertRaises(ValueError, random_align, [], [], -1)

    @patch('random.uniform')
    def test_perc_to_remove(self, mock_uniform):
        mock_uniform.return_value=.2
        A, B = range(10), range(10)
        A, B, alignments = random_align(A, B, perc_to_remove=0.2)
        self.assertEquals(8,len(A))
        self.assertEquals(8, len(B))

class TestRandomRemove(unittest.TestCase):

    def test_empty_input(self):
        xs = []
        _random_remove(xs)
        self.assertEquals([], xs)

    @patch('random.uniform')
    def test_remove(self, mock_uniform):
        mock_uniform.return_value=.2
        xs = range(10)
        _random_remove(xs)
        self.assertEquals(8, len(xs))


class TestRandomRange(unittest.TestCase):

    def test_boundries(self):
        self.assertEquals([], random_range(-1))
        self.assertEquals([], random_range(0))
        self.assertEquals([0], random_range(1))
        self.assertEquals(5, len(random_range(5)))

    def test_span(self):
        # any span <= 1 will lead to no shuffling
        self.assertEquals(range(5), random_range(5,span=1))
        self.assertEquals(range(5), random_range(5,span=0))
        self.assertEquals(range(5), random_range(5,span=-1))
        self.assertEquals(3, len(random_range(3,span=1000)))

    def test_some_shuffling_happens(self):
        self.assertNotEquals(range(100), random_range(100))

    @patch('random.randint')
    @patch('random.shuffle')
    def test_shuffling(self, mock_shuffle, mock_randint):
        mock_randint.return_value=2
        mock_shuffle.side_effect=swap_start_and_end
        self.assertEquals([1,0,3,2], random_range(4))
        self.assertEquals([1,0,3,2,4], random_range(5))


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
