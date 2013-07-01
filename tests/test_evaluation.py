# -*- coding: utf-8 -*-

import os
import json
import tempfile
import unittest
import subprocess
import random
from itertools import izip

from yalign import yalignmodel
from yalign.evaluation import *
from yalign.wordpairscore import WordPairScore
from yalign.sentencepairscore import SentencePairScore
from yalign.input_conversion import parallel_corpus_to_documents
from yalign.train_data_generation import training_alignments_from_documents

from helpers import default_sentence_pair_score

basepath = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(basepath, "data")


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
        self.parallel_corpus = os.path.join(data_path, "canterville.txt")
        metadata_filename = os.path.join(data_path, "metadata.json")
        metadata = json.load(open(metadata_filename))
        self.gap_penalty = metadata['gap_penalty']
        self.threshold = metadata['threshold']
        self.classifier, _ = default_sentence_pair_score()

    def test_evaluate(self):
        random.seed(123)
        stats = evaluate(self.parallel_corpus,
                         self.classifier,
                         self.gap_penalty,
                         self.threshold, 20)
        for x, y in izip(stats['max'], stats['mean']):
            self.assertTrue(x > y > 0)
        for x in stats['std']:
            self.assertTrue(x > 0)


class TestAlignmentPercentage(unittest.TestCase):
    def setUp(self):
        self.parallel_corpus = os.path.join(data_path, "canterville.txt")
        word_scores = os.path.join(data_path, "test_word_scores.csv")
        training_filepath = os.path.join(data_path, "test_training.csv")
        self.model = yalignmodel.basic_model(word_scores, training_filepath)
        A, B = parallel_corpus_to_documents(self.parallel_corpus)
        self.document_a = A
        self.document_b = B

    def test_empty_percentage(self):
        p = alignment_percentage([], [], self.model)
        self.assertEqual(p, 100.0)

    def test_empty2_percentage(self):
        p = alignment_percentage(self.document_a, [], self.model)
        self.assertEqual(p, 0.0)

    def test_empty3_percentage(self):
        p = alignment_percentage([], self.document_b, self.model)
        self.assertEqual(p, 0.0)

    def test_alignment(self):
        p = alignment_percentage(self.document_a, self.document_b, self.model)
        self.assertEqual(p, 50.0)

    def test_command_tool(self):
        tmpdir = tempfile.mkdtemp()
        _, tmpfile = tempfile.mkstemp()
        self.model.save(tmpdir)

        cmd = "yalign-evaluate-alignment %s %s" % (self.parallel_corpus, tmpdir)
        outputfh = open(tmpfile, "w")
        subprocess.call(cmd, shell=True, stdout=outputfh)
        outputfh = open(tmpfile)
        output = outputfh.read()
        self.assertTrue("50.0%" in output)


class TestWordAlignmentPercentage(unittest.TestCase):
    def setUp(self):
        self.parallel_corpus = os.path.join(data_path, "canterville.txt")
        word_scores = os.path.join(data_path, "test_word_scores.csv")
        training_filepath = os.path.join(data_path, "test_training.csv")
        self.model = yalignmodel.basic_model(word_scores, training_filepath)
        A, B = parallel_corpus_to_documents(self.parallel_corpus)
        self.document_a = A
        self.document_b = B

    def test_empty_percentage(self):
        p = word_alignment_percentage([], [], self.model)
        self.assertEqual(p, 100.0)

    def test_empty2_percentage(self):
        p = word_alignment_percentage(self.document_a, [], self.model)
        self.assertEqual(p, 0.0)

    def test_empty3_percentage(self):
        p = word_alignment_percentage([], self.document_b, self.model)
        self.assertEqual(p, 0.0)

    def test_alignment(self):
        p = word_alignment_percentage(self.document_a, self.document_b, self.model)
        self.assertTrue(0.0 <= p <= 100.0)


if __name__ == "__main__":
    unittest.main()
