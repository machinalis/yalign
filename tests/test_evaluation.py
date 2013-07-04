# -*- coding: utf-8 -*-

import os
import json
import random
import tempfile
import unittest
import subprocess
from itertools import izip

from yalign import yalignmodel
from yalign.evaluation import *
from yalign.yalignmodel import YalignModel
from yalign.wordpairscore import WordPairScore
from yalign.sequencealigner import SequenceAligner
from yalign.sentencepairscore import SentencePairScore
from yalign.input_conversion import parallel_corpus_to_documents

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


class BaseTestPercentage(object):
    cmdline = None

    def setUp(self):
        self.parallel_corpus = os.path.join(data_path, "canterville.txt")
        word_scores = os.path.join(data_path, "test_word_scores_big.csv")
        training_filepath = os.path.join(data_path, "test_training.csv")
        self.model = yalignmodel.basic_model(training_filepath, word_scores)
        A, B = parallel_corpus_to_documents(self.parallel_corpus)
        self.document_a = A
        self.document_b = B

    @staticmethod
    def alignment_function(document_a, document_b, model):
        raise NotImplementedError()

    def test_empty_percentage(self):
        p = self.alignment_function([], [], self.model)
        self.assertEqual(p, 100.0)

    def test_empty2_percentage(self):
        p = self.alignment_function(self.document_a, [], self.model)
        self.assertEqual(p, 0.0)

    def test_empty3_percentage(self):
        p = self.alignment_function([], self.document_b, self.model)
        self.assertEqual(p, 0.0)

    def test_valid_value(self):
        p = self.alignment_function(self.document_a, self.document_b,
                                    self.model)
        self.assertTrue(0.0 <= p <= 100.0)

    def test_command_tool(self):
        if self.cmdline is None:
            return

        tmpdir = tempfile.mkdtemp()
        _, tmpfile = tempfile.mkstemp()
        self.model.save(tmpdir)

        cmd = self.cmdline.format(corpus=self.parallel_corpus, model=tmpdir)
        outputfh = open(tmpfile, "w")
        subprocess.call(cmd, shell=True, stdout=outputfh)
        outputfh = open(tmpfile)
        output = outputfh.read()

        A, B = parallel_corpus_to_documents(self.parallel_corpus)
        model = YalignModel()
        model.load(tmpdir)
        value = self.alignment_function(A, B, model)

        self.assertIn("{}%".format(value), output)


class TestAlignmentPercentage(BaseTestPercentage, unittest.TestCase):
    cmdline = "yalign-evaluate-alignment {corpus} {model}"

    @staticmethod
    def alignment_function(document_a, document_b, model):
        return alignment_percentage(document_a, document_b, model)

    def test_alignment(self):
        p = self.alignment_function(self.document_a, self.document_b,
                                    self.model)
        self.assertEqual(p, 50.0)


class TestWordAlignmentPercentage(BaseTestPercentage, unittest.TestCase):
    cmdline = "yalign-evaluate-word-alignment {corpus} {model}"

    @staticmethod
    def alignment_function(document_a, document_b, model):
        return word_alignment_percentage(document_a, document_b, model)


class TestTranslationPercentage(BaseTestPercentage, unittest.TestCase):
    cmdline = "yalign-evaluate-translations-alignment {corpus} {model}"

    @staticmethod
    def alignment_function(document_a, document_b, model):
        return word_translations_percentage(document_a, document_b, model)


class TestClassifierPrecision(unittest.TestCase):
    def setUp(self):
        word_scores = os.path.join(data_path, "test_word_scores_big.csv")
        self.parallel_corpus = os.path.join(data_path, "parallel-en-es.txt")
        # Documents
        A, B = parallel_corpus_to_documents(self.parallel_corpus)
        self.document_a = A[:30]
        self.document_b = B[:30]
        training = training_alignments_from_documents(self.document_a,
                                                      self.document_b)
        # Word score
        word_pair_score = WordPairScore(word_scores)
        # Sentence Score
        sentence_pair_score = SentencePairScore()
        sentence_pair_score.train(training, word_pair_score)
        # Yalign model
        document_aligner = SequenceAligner(sentence_pair_score, 0.49)
        self.model = YalignModel(document_aligner)

    def test_empty(self):
        value = classifier_precision([], [], self.model)
        self.assertEqual(value, 0.0)

    def test_precision(self):
        value = classifier_precision(self.document_a, self.document_b,
                                     self.model)
        self.assertTrue(0.0 <= value <= 100.0)


if __name__ == "__main__":
    unittest.main()
